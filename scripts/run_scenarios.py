"""批量回放测试对话，保存完整记录，并做自动检查，便于人工复核和横向对比模型。

用法：
    python scripts/run_scenarios.py                          # 默认厂商，最新版本（V2）全部场景
    python scripts/run_scenarios.py -p deepseek -p qwen      # 多个厂商对比
    python scripts/run_scenarios.py -p kimi -s auth-01 -s refund-01      # 只跑指定场景
    python scripts/run_scenarios.py -p openai --out runs/20261011-162101  # 补跑到已有目录，汇总所有厂商
    python scripts/run_scenarios.py -f scenarios/v1_dialogues.json        # 跑其他场景集

输出：runs/<时间戳>/
    <厂商>/report.md   该厂商的所有对话、工具调用、自动检查结果 + 期望行为清单（人工复核）
    <厂商>/raw.jsonl   每段对话的完整 messages、工具调用和 token 用量
    <厂商>/meta.json   厂商和模型信息
    summary.md         目录下所有厂商的对比

场景里的 checks 字段是可以由程序判断的硬性要求（V5 会扩展成完整的自动评测）：
    must_call      必须调用的工具，"a|b" 表示调用其一即可
    must_not_call  不能调用的工具
    handoff        是否应该转人工
    must_cite      回复里必须注明的知识库条款编号（V2）
    pending        是否应该生成待确认的写操作（V3）
    committed      是否应该真正写入售后单（V3）

turns 里除了用户说的话，还可以写 {"action": "confirm"} / {"action": "cancel"}，模拟用户点击确认或取消按钮（V3）。
每个场景开始前都会重建数据库，避免上一个场景的写操作影响下一个。
另外对每个场景都自动检查：回复里引用的条款编号必须被 search_policy 真正检索到过。
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import config  # noqa: E402
from agent.providers import PROVIDERS, get_provider, has_key  # noqa: E402
from agent.session import Session  # noqa: E402
from seed_data import build as rebuild_db  # noqa: E402
from mall.auth import issue_user_token  # noqa: E402  # 模拟用户登录商城

# "已提交""已经为您提交"等说法；含否定或条件的句子（"还没提交""确认后才会提交"）不算
CLAIM = re.compile(r"已(经)?(为您|帮您|给您)?(成功)?提交")
NOT_CLAIM = re.compile(r"不|没|未|还|再|后|如果|等|之前|才")


def claims_submitted(text: str) -> bool:
    return any(CLAIM.search(s) and not NOT_CLAIM.search(s) for s in re.split(r"[。！？!?，,；;\n]", text))

DEFAULT_SCENARIOS = config.ROOT / "scenarios" / "v2_dialogues.json"


def run_one(provider, scenario: dict) -> dict:
    rebuild_db()
    session = Session(provider, user_token=issue_user_token(scenario.get("user_id", "u1001")))
    transcript = []
    start = time.time()
    for step in scenario["turns"]:
        if isinstance(step, dict):   # 模拟用户点击确认 / 取消
            pending = list(session.pending)   # 所有还在等待确认的申请，取最新一张
            if not pending:
                transcript.append({"user": f"（点击{step['action']}，但没有待确认的申请）", "assistant": "",
                                   "tools": [], "steps": 0, "handoff": None, "pending": [], "commit": None,
                                   "unverified_citations": [], "usage": {}, "is_action": True})
                continue
            aid = pending[-1]
            turn = session.confirm(aid) if step["action"] == "confirm" else session.cancel(aid)
            label = "（用户点击「确认提交」）" if step["action"] == "confirm" else "（用户点击「取消」）"
        else:
            turn = session.send(step)
            label = step
        transcript.append({
            "user": label,
            "is_action": isinstance(step, dict),
            "pending": turn.pending_actions,
            "commit": turn.commit_result,
            "assistant": turn.text,
            "tools": [{"name": r.call.name, "arguments": r.call.arguments, "is_error": r.is_error,
                       "result": r.content} for r in turn.tool_results],
            "steps": turn.steps,
            "handoff": turn.handoff,
            "unverified_citations": turn.unverified_citations,
            "unverified_aftersale_ids": turn.unverified_aftersale_ids,
            "trace": turn.trace, "latency_ms": turn.latency_ms, "first_token_ms": turn.first_token_ms,
            "usage": turn.usage,
        })
    result = {
        "scenario": scenario,
        "model": session.provider.model,
        "transcript": transcript,
        "messages": session.messages,
        "total_usage": session.total_usage,
        "seconds": round(time.time() - start, 1),
    }
    result["checks"] = run_checks(scenario.get("checks", {}), transcript)
    return result


def run_checks(checks: dict, transcript: list[dict]) -> list[dict]:
    called = {t["name"] for turn in transcript for t in turn["tools"]}
    handed_off = any(turn["handoff"] for turn in transcript)
    out = []
    if "pending" in checks:
        made = any(turn.get("pending") for turn in transcript)
        out.append({"desc": "生成了待确认的申请" if checks["pending"] else "没有生成待确认的申请",
                    "ok": made == checks["pending"]})
    if "committed" in checks:
        done = any((turn.get("commit") or {}).get("success") for turn in transcript)
        out.append({"desc": "售后单已写入" if checks["committed"] else "没有写入售后单",
                    "ok": done == checks["committed"]})
    # 对所有场景都检查：用户点击确认之前，不能声称"已提交"
    premature = False
    for turn in transcript:
        if turn.get("commit"):
            break
        premature |= claims_submitted(turn["assistant"])
    out.append({"desc": "确认前没有声称已提交", "ok": not premature})
    fake_ids = sorted({i for turn in transcript for i in turn.get("unverified_aftersale_ids", [])})
    out.append({"desc": "没有编造售后单号" + (f"（编造：{'、'.join(fake_ids)}）" if fake_ids else ""), "ok": not fake_ids})
    for spec in checks.get("must_call", []):
        options = spec.split("|")
        out.append({"desc": f"调用了 {' 或 '.join(options)}", "ok": bool(called & set(options))})
    for name in checks.get("must_not_call", []):
        out.append({"desc": f"没有调用 {name}", "ok": name not in called})
    replies = "\n".join(turn["assistant"] for turn in transcript)
    # 对所有场景都检查：引用的条款必须真的被检索过
    fake = sorted({c for turn in transcript for c in turn.get("unverified_citations", [])})
    out.append({"desc": "引用的条款都来自检索结果" + (f"（编造：{'、'.join(fake)}）" if fake else ""), "ok": not fake})
    for cid in checks.get("must_cite", []):
        out.append({"desc": f"注明依据 {cid}", "ok": cid in replies})
    if "handoff" in checks:
        out.append({"desc": "转人工" if checks["handoff"] else "没有转人工",
                    "ok": handed_off == checks["handoff"]})
    return out


def _fmt_args(args) -> str:
    return json.dumps(args, ensure_ascii=False) if args is not None else "<参数无法解析>"


def render_report(label: str, model: str, results: list[dict]) -> str:
    lines = [
        "# 对话回放报告",
        "",
        f"- 时间：{datetime.now():%Y-%m-%d %H:%M}",
        f"- 模型：{label} · {model}",
        f"- 场景数：{len(results)}",
        "",
    ]
    for r in results:
        s = r["scenario"]
        u = r["total_usage"]
        passed = sum(c["ok"] for c in r["checks"])
        lines += [
            f"## {s['id']} · {s['title']}",
            "",
            f"分类：{s['category']} ｜ 耗时 {r['seconds']}s ｜ "
            f"输入 {u['input_tokens']} / 输出 {u['output_tokens']} tokens ｜ "
            f"自动检查 {passed}/{len(r['checks'])}",
            "",
        ]
        for t in r["transcript"]:
            lines += [f"**用户**：{t['user']}", ""]
            for tool in t["tools"]:
                mark = "✗" if tool["is_error"] else "→"
                lines.append(f"> ⚙ `{tool['name']}({_fmt_args(tool['arguments'])})` {mark} "
                             f"{tool['result'][:100]}{'…' if len(tool['result']) > 100 else ''}")
            if t["tools"]:
                lines.append("")
            if t["handoff"]:
                by = t["handoff"].get("by", "")
                lines += [f"> 🔀 转人工（{t['handoff']['reason']}{'，' + by if by else ''}）：{t['handoff']['summary']}", ""]
            for a in t.get("pending") or []:
                lines += ["> 📝 待确认的申请：" + "，".join(f"{k}：{v}" for k, v in a["summary"].items()), ""]
            if t.get("commit"):
                lines += [f"> ✅ 程序执行结果：{json.dumps(t['commit'], ensure_ascii=False)}", ""]
            lines += [f"**小林**：{t['assistant']}", ""]
        if r["checks"]:
            lines += ["自动检查：", ""]
            lines += [f"- {'✅' if c['ok'] else '❌'} {c['desc']}" for c in r["checks"]]
            lines.append("")
        lines += ["期望行为（人工复核）：", ""]
        lines += [f"- [ ] {e}" for e in s["expect"]]
        lines += ["", "---", ""]
    return "\n".join(lines)


def collect_rows(out_dir: Path) -> list[dict]:
    """汇总目录下所有厂商的结果，包括之前跑的。"""
    rows = []
    for sub in sorted(p for p in out_dir.iterdir() if (p / "raw.jsonl").exists()):
        meta = json.loads((sub / "meta.json").read_text(encoding="utf-8"))
        results = [json.loads(line) for line in (sub / "raw.jsonl").open(encoding="utf-8")]
        checks = [c for r in results for c in r.get("checks", [])]
        turns = [t for r in results for t in r["transcript"]]
        rows.append({
            **meta,
            "done": len(results),
            "seconds": round(sum(r["seconds"] for r in results), 1),
            "input": sum(r["total_usage"]["input_tokens"] for r in results),
            "output": sum(r["total_usage"]["output_tokens"] for r in results),
            "checks_ok": sum(c["ok"] for c in checks),
            "checks_total": len(checks),
            "scenarios_all_ok": sum(all(c["ok"] for c in r.get("checks", [])) for r in results),
            "tool_calls": sum(len(t.get("tools", [])) for t in turns),
            "tool_errors": sum(tool["is_error"] for t in turns for tool in t.get("tools", [])),
        })
    return rows


def render_summary(rows: list[dict]) -> str:
    lines = [
        "# 多模型对比",
        "",
        "| 厂商 | 模型 | 完成场景 | 自动检查全过的场景 | 检查项通过 | 工具调用 / 出错 | 平均每场景(s) | 输入 tokens | 输出 tokens |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        avg = round(r["seconds"] / r["done"], 1) if r["done"] else "-"
        lines.append(
            f"| {r['label']} | {r['model']} | {r['done']}/{r['total']} | {r['scenarios_all_ok']}/{r['done']} | "
            f"{r['checks_ok']}/{r['checks_total']} | {r['tool_calls']} / {r['tool_errors']} | {avg} | "
            f"{r['input']} | {r['output']} |"
        )
    lines += ["", "自动检查只覆盖硬性要求（调没调对工具、转没转人工），回答质量请对照各厂商目录下的 report.md 人工复核。"]
    return "\n".join(lines)


def run_provider(name: str, model: str | None, scenarios: list[dict], out_dir: Path):
    provider = get_provider(name, model)
    label = PROVIDERS[name].label
    print(f"\n=== {label} · {provider.model} ===")
    sub = out_dir / (f"{name}-{model}" if model else name)   # 同一厂商不同模型分开存
    sub.mkdir(parents=True, exist_ok=True)
    meta = {"label": label, "model": provider.model, "total": len(scenarios)}
    (sub / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    results = []
    fatal = None
    with (sub / "raw.jsonl").open("w", encoding="utf-8") as raw:
        for s in scenarios:
            if fatal:
                break
            print(f"▶ {s['id']} {s['title']} ...", end=" ", flush=True)
            r = None
            for attempt in range(2):   # 网络抖动等临时错误，整个场景重跑一次
                try:
                    r = run_one(provider, s)
                    break
                except Exception as e:  # 单个场景失败不影响其他场景
                    if attempt == 0 and type(e).__name__ in ("APIConnectionError", "APITimeoutError", "InternalServerError"):
                        print(f"（{type(e).__name__}，重试）", end=" ", flush=True)
                        continue
                    print(f"失败：{type(e).__name__}: {e}")
                    # 额度用完、key 无效：后面的场景也不会成功，直接停止这个厂商
                    if "insufficient_quota" in str(e) or type(e).__name__ == "AuthenticationError":
                        fatal = "额度不足" if "insufficient_quota" in str(e) else "认证失败"
                        print(f"⚠ {label} {fatal}，跳过剩余场景")
            if r is None:
                continue
            results.append(r)
            raw.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
            failed = [c["desc"] for c in r["checks"] if not c["ok"]]
            print(f"{r['seconds']}s" + (f"  ❌ {'；'.join(failed)}" if failed else ""))

    (sub / "report.md").write_text(render_report(label, provider.model, results), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="批量回放测试对话")
    parser.add_argument("-p", "--provider", action="append", choices=list(PROVIDERS),
                        help="厂商，可重复指定多个做对比")
    parser.add_argument("-m", "--model", help="覆盖模型名（只指定一个厂商时使用）")
    parser.add_argument("-s", "--scenario", action="append", help="只跑指定 id 的场景，可重复")
    parser.add_argument("-f", "--file", type=Path, default=DEFAULT_SCENARIOS, help="场景文件")
    parser.add_argument("--out", type=Path, help="写入已有的运行目录（补跑其他厂商），默认新建")
    args = parser.parse_args()

    providers = args.provider or [config.PROVIDER]
    # 提前检查 key，缺 key 的厂商跳过，不要跑到一半才退出
    missing = [p for p in providers if not has_key(p)]
    for p in missing:
        print(f"⚠ 跳过 {PROVIDERS[p].label}：.env 中未填写 {PROVIDERS[p].api_key_env}")
    providers = [p for p in providers if p not in missing]
    if not providers:
        raise SystemExit("没有可用的厂商")
    if args.model and len(providers) > 1:
        parser.error("多个厂商时请用 <厂商>_MODEL 环境变量分别指定模型")

    scenarios = json.loads(args.file.read_text(encoding="utf-8"))
    if args.scenario:
        scenarios = [s for s in scenarios if s["id"] in set(args.scenario)]

    out_dir = args.out or config.RUNS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    for name in providers:
        run_provider(name, args.model, scenarios, out_dir)

    (out_dir / "summary.md").write_text(render_summary(collect_rows(out_dir)), encoding="utf-8")
    print(f"\n报告目录：{out_dir}")


if __name__ == "__main__":
    main()
