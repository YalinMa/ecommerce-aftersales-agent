"""评分模型校准：用一组已知答案的对话，检验评分模型靠不靠谱。

用法：
    python scripts/judge_calibration.py                                  # 默认评分模型
    python scripts/judge_calibration.py -j deepseek:deepseek-v4-pro -j openai:gpt-5.5   # 对比多个评分模型

校准集 scenarios/v5_judge_calibration.json 来自 V1–V3 的真实测试记录：
    坏例：之前人工发现的问题（编造政策、被伪造通知骗、谎称已转人工……）
    好例：人工确认没问题的对话

指标：
    一致率      评分结论和人工结论相同的比例
    漏判        坏例被判为通过。最危险：说明评分模型会放过真实问题
    误判        好例被判为不通过。会浪费人工复核时间，但不会放过问题
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import config  # noqa: E402
from evals.judge import DEFAULT_JUDGE, DIMENSIONS, Judge  # noqa: E402

CASES = json.loads((config.ROOT / "scenarios" / "v5_judge_calibration.json").read_text(encoding="utf-8"))


def calibrate(spec: str) -> dict:
    judge = Judge(spec)
    print(f"评分模型 {judge.cfg.label}：{len(CASES)} 条 …", flush=True)

    def one(case):
        try:
            return case, judge.grade(case["scenario"], case["transcript"], date=case["date"])
        except Exception as e:
            return case, {"error": f"{type(e).__name__}: {e}"}

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(one, CASES))

    rows, agree, missed, false_alarm = [], 0, [], []
    for case, g in results:
        if "error" in g:
            rows.append((case, None, g))
            continue
        verdict = "pass" if g["passed"] else "fail"
        ok = verdict == case["expected"]
        agree += ok
        if case["expected"] == "fail" and verdict == "pass":
            missed.append(case["id"])
        if case["expected"] == "pass" and verdict == "fail":
            false_alarm.append(case["id"])
        rows.append((case, verdict, g))
    graded = sum(1 for _, v, _ in rows if v)
    return {"judge": judge.cfg.label, "rows": rows, "agree": agree, "graded": graded,
            "missed": missed, "false_alarm": false_alarm}


def render(results: list[dict]) -> str:
    lines = ["# 评分模型校准", "", f"- 时间：{datetime.now():%Y-%m-%d %H:%M}",
             f"- 校准集：{len(CASES)} 条（坏例 {sum(c['expected'] == 'fail' for c in CASES)}，"
             f"好例 {sum(c['expected'] == 'pass' for c in CASES)}）", "",
             "| 评分模型 | 一致率 | 漏判（坏例判成通过） | 误判（好例判成不通过） |", "|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['judge']} | {r['agree']}/{r['graded']} | {len(r['missed'])} {'、'.join(r['missed'])} | "
                     f"{len(r['false_alarm'])} {'、'.join(r['false_alarm'])} |")
    for r in results:
        lines += ["", f"## {r['judge']} 逐条结果", ""]
        for case, verdict, g in r["rows"]:
            if verdict is None:
                lines.append(f"### ⚠️ {case['id']} 评分失败：{g['error']}")
                continue
            mark = "✅" if verdict == case["expected"] else "❌"
            s = g["scores"]
            lines += [
                f"### {mark} {case['id']}（人工：{case['expected']}，评分：{verdict}）",
                f"- 人工说明：{case['note']}",
                f"- 分数：" + "，".join(f"{DIMENSIONS[k]} {s[k]}" for k in DIMENSIONS),
                f"- 评分模型总结：{g['summary']}",
            ]
            lines += [f"- 发现的问题：{i}" for i in g["issues"]]
            lines += [f"- 期望 {'✓' if e['pass'] else '✗'} {e['item']}：{e['reason']}" for e in g["expectations"]]
            lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="评分模型校准")
    parser.add_argument("-j", "--judge", action="append", help="评分模型，格式 厂商:模型，可重复")
    args = parser.parse_args()
    results = [calibrate(spec) for spec in (args.judge or [DEFAULT_JUDGE])]
    report = render(results)
    out = config.RUNS_DIR / f"judge-calibration-{datetime.now():%Y%m%d-%H%M%S}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print("\n" + "\n".join(report.splitlines()[:8 + len(results)]) + f"\n\n完整报告：{out}")


if __name__ == "__main__":
    main()
