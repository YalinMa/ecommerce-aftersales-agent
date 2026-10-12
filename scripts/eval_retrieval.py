"""检索评测：对比 BM25、向量、混合检索在同一组问题上的表现。

用法：
    python scripts/eval_retrieval.py

指标：
    Hit@1     第一条结果就是正确条款的比例
    Recall@3  正确条款出现在前 3 条结果里的比例（多个正确条款时按命中比例计）
    MRR       第一个正确条款排名的倒数的平均值，越接近 1 越好

只调用 embedding 接口（很便宜），不调用大模型。结果写入 runs/retrieval-<时间>.md。
"""

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import config  # noqa: E402
from mall.knowledge import EMBEDDERS, INDEX_PATH, Retriever  # noqa: E402

CASES = json.loads((config.ROOT / "scenarios" / "v2_retrieval.json").read_text(encoding="utf-8"))
K = 3


def evaluate(retriever: Retriever) -> dict:
    hit1 = recall = mrr = 0.0
    misses = []
    for case in CASES:
        ranked = [h.chunk.id for h in retriever.search(case["q"], top_k=10)]
        relevant = set(case["relevant"])
        hit1 += ranked[0] in relevant
        recall += len(relevant & set(ranked[:K])) / len(relevant)
        first = next((i for i, cid in enumerate(ranked) if cid in relevant), None)
        mrr += 1 / (first + 1) if first is not None else 0
        if not relevant & set(ranked[:K]):
            misses.append(f"{case['q']} → 期望 {'/'.join(case['relevant'])}，得到 {', '.join(ranked[:K])}")
    n = len(CASES)
    return {"hit1": hit1 / n, "recall": recall / n, "mrr": mrr / n, "misses": misses}


def main():
    index = json.loads(INDEX_PATH.read_text(encoding="utf-8")) if INDEX_PATH.exists() else {"embeddings": {}}
    configs = [("BM25（关键词）", "bm25", None)]
    for provider in EMBEDDERS:
        if provider in index["embeddings"]:
            model = index["embeddings"][provider]["model"]
            configs.append((f"向量 · {model}", "embedding", provider))
            configs.append((f"混合 · BM25 + {model}", "hybrid", provider))

    rows = []
    for label, method, embedder in configs:
        print(f"评测 {label} ...", flush=True)
        rows.append((label, evaluate(Retriever(method, embedder))))

    lines = [
        "# 检索评测",
        "",
        f"- 时间：{datetime.now():%Y-%m-%d %H:%M}",
        f"- 问题数：{len(CASES)}，知识库条款数：{len(Retriever('bm25').chunks)}",
        "",
        f"| 检索方式 | Hit@1 | Recall@{K} | MRR |",
        "|---|---|---|---|",
    ]
    lines += [f"| {label} | {r['hit1']:.0%} | {r['recall']:.0%} | {r['mrr']:.3f} |" for label, r in rows]
    for label, r in rows:
        lines += ["", f"## {label}：前 {K} 条没找到正确条款的问题（{len(r['misses'])} 个）", ""]
        lines += [f"- {m}" for m in r["misses"]] or ["- 无"]

    report = "\n".join(lines)
    out = config.RUNS_DIR / f"retrieval-{datetime.now():%Y%m%d-%H%M%S}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print("\n" + report + f"\n\n报告：{out}")


if __name__ == "__main__":
    main()
