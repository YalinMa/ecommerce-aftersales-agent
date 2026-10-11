"""构建知识库索引：切分 knowledge/*.md，为每个可用的向量模型计算向量。

用法：
    python scripts/build_index.py

修改了 knowledge/ 里的文档后需要重新运行。没有配置任何向量模型的 key 时，
仍可以只用关键词检索（BM25），不需要索引文件。
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.knowledge import EMBEDDERS, INDEX_PATH, embed, fingerprint, load_chunks  # noqa: E402
from agent.providers import has_key  # noqa: E402


def main():
    chunks = load_chunks()
    print(f"切分完成：{len(chunks)} 个条款，来自 {len({c.doc for c in chunks})} 个文档")
    for c in chunks[:3]:
        print(f"  {c.id} {c.title}（{len(c.content)} 字）")
    print("  ...")

    index = {"fingerprint": fingerprint(chunks), "chunk_ids": [c.id for c in chunks], "embeddings": {}}
    for provider, model in EMBEDDERS.items():
        if not has_key(provider):
            print(f"跳过 {provider}:{model}（未配置 key）")
            continue
        start = time.time()
        vectors = embed(provider, [c.text() for c in chunks])
        index["embeddings"][provider] = {"model": model, "dim": len(vectors[0]), "vectors": vectors}
        print(f"已计算 {provider}:{model} 向量，{len(vectors[0])} 维，耗时 {time.time() - start:.1f}s")

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(index), encoding="utf-8")
    print(f"索引已写入 {INDEX_PATH.relative_to(INDEX_PATH.parent.parent)}")


if __name__ == "__main__":
    main()
