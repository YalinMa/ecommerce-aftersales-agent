"""评测：用评分模型给一次回放测试的所有对话打分，生成可在浏览器打开的评测报告。

用法：
    python scripts/run_scenarios.py -p deepseek -p openai -p qwen   # 先跑回放（会生成 runs/<时间>/）
    python scripts/eval.py                                          # 给最近一次回放打分
    python scripts/eval.py -r runs/20261011-215340 -j openai:gpt-5.5   # 指定回放目录和评分模型

输出（写在回放目录里）：
    <厂商>/judge.jsonl    每个场景的评分明细
    eval_report.html      评测报告：总览、各场景得分矩阵、未通过原因、耗时与用量
    eval_summary.md       文字版总览
"""

import argparse
import html
import json
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import config  # noqa: E402
from evals.judge import DEFAULT_JUDGE, DIMENSIONS, Judge, render_transcript  # noqa: E402


def latest_run() -> Path:
    runs = sorted(p for p in config.RUNS_DIR.iterdir() if p.is_dir() and p.name[:8].isdigit())
    if not runs:
        raise SystemExit("还没有回放记录，请先运行 scripts/run_scenarios.py")
    return runs[-1]


def grade_run(run_dir: Path, judge: Judge) -> dict[str, list[dict]]:
    """给目录下每个厂商的每个场景打分，返回 {厂商目录名: [结果...]}。"""
    date = f"{run_dir.name[:4]}-{run_dir.name[4:6]}-{run_dir.name[6:8]}"
    out = {}
    for sub in sorted(p for p in run_dir.iterdir() if (p / "raw.jsonl").exists()):
        records = [json.loads(line) for line in (sub / "raw.jsonl").open(encoding="utf-8")]
        print(f"评分 {sub.name}：{len(records)} 个场景 …", flush=True)

        def one(r):
            try:
                return {**r, "judgment": judge.grade(r["scenario"], r["transcript"], date=date)}
            except Exception as e:
                return {**r, "judgment": {"error": f"{type(e).__name__}: {e}"}}

        with ThreadPoolExecutor(max_workers=6) as pool:
            graded = list(pool.map(one, records))
        with (sub / "judge.jsonl").open("w", encoding="utf-8") as f:
            for g in graded:
                f.write(json.dumps({"scenario_id": g["scenario"]["id"], **g["judgment"]}, ensure_ascii=False) + "\n")
        out[sub.name] = graded
    return out


def _avg(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _p90(xs):
    xs = sorted(x for x in xs if x is not None)
    return xs[min(len(xs) - 1, int(len(xs) * 0.9))] if xs else None


def stats(run_dir: Path, graded: list[dict]) -> dict:
    meta = json.loads((run_dir / graded[0]["_sub"] / "meta.json").read_text(encoding="utf-8")) if graded else {}
    ok = [g for g in graded if "error" not in g["judgment"]]
    turns = [t for g in graded for t in g["transcript"] if not t.get("is_action") or t.get("assistant")]
    exps = [e for g in ok for e in g["judgment"]["expectations"]]
    return {
        "label": meta.get("label", ""), "model": meta.get("model", ""),
        "n": len(graded), "graded": len(ok),
        "judge_pass": sum(g["judgment"]["passed"] for g in ok),
        "auto_pass": sum(all(c["ok"] for c in g.get("checks", [])) for g in graded),
        "exp_pass": sum(e["pass"] for e in exps), "exp_total": len(exps),
        "scores": {k: _avg([g["judgment"]["scores"][k] for g in ok]) for k in DIMENSIONS},
        "latency": _avg([t.get("latency_ms") for t in turns]),
        "latency_p90": _p90([t.get("latency_ms") for t in turns]),
        "first_token": _avg([t.get("first_token_ms") for t in turns]),
        "tokens_per_turn": _avg([sum((t.get("usage") or {}).get(k, 0) for k in ("input_tokens", "output_tokens"))
                                 for t in turns]),
        "tools_per_turn": _avg([len(t.get("tools", [])) for t in turns]),
    }


def _fmt(v, unit="", digits=1, scale=1):
    return "—" if v is None else f"{v / scale:.{digits}f}{unit}"


# ---------- 报告 ----------

CSS = """
:root { --bg:#f6f5f2; --panel:#fff; --line:#e6e3dc; --text:#1f1d1a; --muted:#77726a; --accent:#2f6f4f;
        --accent-soft:#e6f0ea; --err:#b4432f; --err-soft:#fbeae6; --warn:#8a5a00; --warn-soft:#fff4dc; }
@media (prefers-color-scheme: dark) { :root { --bg:#161614; --panel:#1f1f1c; --line:#33322e; --text:#ecebe6;
        --muted:#9d998f; --accent:#6cbf93; --accent-soft:#23352b; --err:#ef8a76; --err-soft:#3a2420;
        --warn:#e8c070; --warn-soft:#3a3020; } }
* { box-sizing:border-box; } body { margin:0; background:var(--bg); color:var(--text);
  font:14px/1.6 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif; }
main { max-width:1180px; margin:0 auto; padding:24px 16px 60px; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:17px; margin:32px 0 10px; }
.sub { color:var(--muted); font-size:13px; }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:12px; margin-top:16px; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px 16px; }
.card h3 { margin:0 0 2px; font-size:15px; } .card .model { color:var(--muted); font-size:12px; }
.big { font-size:30px; font-weight:600; color:var(--accent); margin:6px 0 0; }
.big small { font-size:13px; color:var(--muted); font-weight:normal; }
.bar { display:flex; align-items:center; gap:8px; font-size:12px; margin-top:6px; }
.bar span:first-child { width:56px; color:var(--muted); }
.bar .track { flex:1; height:8px; background:var(--line); border-radius:4px; overflow:hidden; }
.bar .fill { height:100%; background:var(--accent); }
.kv { display:grid; grid-template-columns:auto 1fr; gap:2px 12px; font-size:12px; margin-top:10px; color:var(--muted); }
.kv b { color:var(--text); font-weight:500; }
table { width:100%; border-collapse:collapse; background:var(--panel); border:1px solid var(--line); border-radius:10px; overflow:hidden; }
th, td { padding:8px 10px; border-top:1px solid var(--line); text-align:left; vertical-align:top; font-size:13px; }
th { background:var(--bg); font-weight:600; border-top:none; }
td.cell { text-align:center; white-space:nowrap; }
.pass { color:var(--accent); } .fail { color:var(--err); font-weight:600; }
.tag { display:inline-block; font-size:11px; padding:1px 6px; border-radius:4px; background:var(--warn-soft); color:var(--warn); }
details { background:var(--panel); border:1px solid var(--line); border-radius:10px; margin-bottom:10px; }
summary { cursor:pointer; padding:10px 14px; } details[open] summary { border-bottom:1px solid var(--line); }
.detail { padding:10px 14px; font-size:13px; } .detail ul { margin:4px 0 8px; padding-left:20px; }
pre { white-space:pre-wrap; word-break:break-word; background:var(--bg); padding:10px; border-radius:8px; font-size:12px;
      max-height:420px; overflow:auto; }
.scroll { overflow-x:auto; }
"""


def render_html(run_dir: Path, judge_label: str, results: dict[str, list[dict]], st: dict[str, dict]) -> str:
    e = html.escape
    subs = list(results)
    out = [f"<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>",
           "<meta name='viewport' content='width=device-width, initial-scale=1'>",
           f"<title>评测报告 {e(run_dir.name)}</title><style>{CSS}</style></head><body><main>",
           "<h1>售后 Agent 评测报告</h1>",
           f"<div class='sub'>回放目录 {e(run_dir.name)} · 评分模型 {e(judge_label)} · 生成于 {datetime.now():%Y-%m-%d %H:%M}</div>"]

    # 总览卡片
    out.append("<div class='cards'>")
    for sub in subs:
        s = st[sub]
        bars = "".join(
            f"<div class='bar'><span>{DIMENSIONS[k]}</span><div class='track'><div class='fill' "
            f"style='width:{(s['scores'][k] or 0) / 5 * 100:.0f}%'></div></div><span>{_fmt(s['scores'][k])}</span></div>"
            for k in DIMENSIONS)
        out.append(
            f"<div class='card'><h3>{e(s['label'])}</h3><div class='model'>{e(s['model'])}</div>"
            f"<div class='big'>{s['judge_pass']}/{s['graded']}<small> 场景通过评分</small></div>{bars}"
            f"<div class='kv'><span>期望达成</span><b>{s['exp_pass']}/{s['exp_total']}</b>"
            f"<span>自动检查全过</span><b>{s['auto_pass']}/{s['n']}</b>"
            f"<span>每轮耗时</span><b>{_fmt(s['latency'], ' 秒', scale=1000)}（P90 {_fmt(s['latency_p90'], ' 秒', scale=1000)}）</b>"
            f"<span>首字时间</span><b>{_fmt(s['first_token'], ' 秒', scale=1000)}</b>"
            f"<span>每轮 tokens</span><b>{_fmt(s['tokens_per_turn'], digits=0)}</b>"
            f"<span>每轮工具调用</span><b>{_fmt(s['tools_per_turn'])}</b></div></div>")
    out.append("</div>")

    # 场景矩阵
    scenarios = [g["scenario"] for g in results[subs[0]]]
    by_id = {sub: {g["scenario"]["id"]: g for g in results[sub]} for sub in subs}
    out.append("<h2>各场景得分</h2><div class='sub'>格子里是：评分结论 · 准确 / 合规 / 体验。"
               "<span class='tag'>不一致</span> 表示自动检查和评分模型结论不同，值得人工看一眼。</div><div class='scroll'><table>")
    out.append("<tr><th>场景</th>" + "".join(f"<th>{e(st[s]['label'])}</th>" for s in subs) + "</tr>")
    for sc in scenarios:
        cells = []
        for sub in subs:
            g = by_id[sub].get(sc["id"])
            if not g or "error" in g["judgment"]:
                cells.append("<td class='cell'>—</td>")
                continue
            j = g["judgment"]
            auto_ok = all(c["ok"] for c in g.get("checks", []))
            mismatch = "<br><span class='tag'>不一致</span>" if auto_ok != j["passed"] else ""
            sc_ = "/".join(str(j["scores"][k]) for k in DIMENSIONS)
            cells.append(f"<td class='cell'><span class='{'pass' if j['passed'] else 'fail'}'>"
                         f"{'✓ 通过' if j['passed'] else '✗ 未通过'}</span><br>{sc_}{mismatch}</td>")
        out.append(f"<tr><td><b>{e(sc['id'])}</b> {e(sc['title'])}<br><span class='sub'>{e(sc['category'])}</span></td>"
                   + "".join(cells) + "</tr>")
    out.append("</table></div>")

    # 未通过 / 不一致的场景详情
    out.append("<h2>需要关注的场景</h2><div class='sub'>评分未通过，或自动检查与评分结论不一致。点开查看评分理由和完整对话。</div>")
    any_flag = False
    for sub in subs:
        for g in results[sub]:
            j = g["judgment"]
            if "error" in j:
                continue
            auto_ok = all(c["ok"] for c in g.get("checks", []))
            if j["passed"] and auto_ok:
                continue
            any_flag = True
            failed_checks = [c["desc"] for c in g.get("checks", []) if not c["ok"]]
            exps = "".join(f"<li>{'✓' if x['pass'] else '<span class=fail>✗</span>'} {e(x['item'])}："
                           f"<span class='sub'>{e(x['reason'])}</span></li>" for x in j["expectations"])
            issues = "".join(f"<li>{e(i)}</li>" for i in j["issues"]) or "<li>无</li>"
            out.append(
                f"<details><summary><b>{e(st[sub]['label'])}</b> · {e(g['scenario']['id'])} {e(g['scenario']['title'])} — "
                f"<span class='{'pass' if j['passed'] else 'fail'}'>评分{'通过' if j['passed'] else '未通过'}</span>"
                f"{'，自动检查未过：' + e('；'.join(failed_checks)) if failed_checks else ''}</summary><div class='detail'>"
                f"<b>评分总结</b>：{e(j['summary'])}<br><b>期望逐条</b><ul>{exps}</ul><b>发现的问题</b><ul>{issues}</ul>"
                f"<b>完整对话</b><pre>{e(render_transcript(g['transcript']))}</pre></div></details>")
    if not any_flag:
        out.append("<p class='sub'>全部场景都通过了评分和自动检查。</p>")
    out.append("</main></body></html>")
    return "\n".join(out)


def render_md(run_dir: Path, judge_label: str, st: dict[str, dict]) -> str:
    lines = [f"# 评测总览 {run_dir.name}", "", f"评分模型：{judge_label}", "",
             "| 厂商 | 模型 | 评分通过 | 期望达成 | 准确 | 合规 | 体验 | 自动检查全过 | 每轮耗时 | 首字 | 每轮 tokens |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in st.values():
        lines.append(f"| {s['label']} | {s['model']} | {s['judge_pass']}/{s['graded']} | {s['exp_pass']}/{s['exp_total']} | "
                     + " | ".join(_fmt(s['scores'][k], digits=2) for k in DIMENSIONS)
                     + f" | {s['auto_pass']}/{s['n']} | {_fmt(s['latency'], 's', scale=1000)} | "
                     f"{_fmt(s['first_token'], 's', scale=1000)} | {_fmt(s['tokens_per_turn'], digits=0)} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="用评分模型评测一次回放")
    parser.add_argument("-r", "--run", type=Path, help="回放目录，默认最近一次")
    parser.add_argument("-j", "--judge", default=DEFAULT_JUDGE, help="评分模型，格式 厂商:模型")
    args = parser.parse_args()

    run_dir = args.run or latest_run()
    judge = Judge(args.judge)
    results = grade_run(run_dir, judge)
    for sub, graded in results.items():
        for g in graded:
            g["_sub"] = sub
    st = {sub: stats(run_dir, graded) for sub, graded in results.items()}

    (run_dir / "eval_report.html").write_text(render_html(run_dir, judge.cfg.label, results, st), encoding="utf-8")
    summary = render_md(run_dir, judge.cfg.label, st)
    (run_dir / "eval_summary.md").write_text(summary, encoding="utf-8")
    errors = sum("error" in g["judgment"] for gs in results.values() for g in gs)
    print("\n" + summary + (f"\n\n⚠ {errors} 个场景评分失败" if errors else "")
          + f"\n\n报告：{run_dir / 'eval_report.html'}")


if __name__ == "__main__":
    main()
