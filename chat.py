"""命令行客服窗口。

用法：
    python chat.py                       # 用 .env 里 AGENT_PROVIDER 指定的厂商（默认 anthropic）
    python chat.py -p deepseek           # 指定厂商
    python chat.py -p qwen -m qwen-max   # 指定厂商和模型
    python chat.py -u u1002              # 以其他用户身份登录（u1001 / u1002 / u1003）

对话中可用的命令：
    /orders       查看当前用户的订单（方便测试时找订单号）
    /providers    列出所有厂商及 API key 是否已配置
    /use <厂商>   切换厂商（会开始新会话）
    /debug        打印下一次请求会发给模型的完整参数
    /history      查看当前历史消息条数和累计 token
    /reset        清空历史，开始新会话
    /exit         退出
"""

import argparse
import json

import anthropic
import openai

from agent import config
from mall import db as mall_db  # 仅用于模拟"商城 App"一侧：/orders 查看订单、用户登录
from mall.auth import issue_user_token
from agent.providers import PROVIDERS, get_provider, has_key, resolve_model
from agent.session import Session
from agent.tools import ToolResult

WELCOME = "您好，我是林间优选售后客服小林，请问有什么可以帮您？"

AUTH_ERRORS = (anthropic.AuthenticationError, openai.AuthenticationError)
RATE_ERRORS = (anthropic.RateLimitError, openai.RateLimitError)
STATUS_ERRORS = (anthropic.APIStatusError, openai.APIStatusError)
CONN_ERRORS = (anthropic.APIConnectionError, openai.APIConnectionError)

GRAY, RESET = "\033[90m", "\033[0m"


def print_providers(current: str):
    for name, spec in PROVIDERS.items():
        mark = "→" if name == current else " "
        key = "✓" if has_key(name) else "✗ 未配置 " + spec.api_key_env
        print(f" {mark} {name:<10} {spec.label:<16} {resolve_model(name):<22} {key}")
    print()


def print_orders(user_id: str):
    with mall_db.connect() as conn:
        rows = conn.execute(
            """SELECT o.id, o.status, group_concat(p.name, '、') AS items FROM orders o
               JOIN order_items i ON i.order_id = o.id JOIN products p ON p.id = i.product_id
               WHERE o.user_id = ? GROUP BY o.id ORDER BY o.paid_at DESC LIMIT 12""", (user_id,)
        ).fetchall()
    for r in rows:
        print(f"  {r['id']}  {r['status']:<4}  {r['items']}")
    print()


def print_debug(session: Session):
    params = session.build_request()
    short = session.system_prompt[:120] + f"...（共 {len(session.system_prompt)} 字）"
    if isinstance(params.get("system"), str):
        params["system"] = short
    for m in params.get("messages", []):
        if m.get("role") == "system":
            m["content"] = short
    if "tools" in params:
        params["tools"] = f"<{len(params['tools'])} 个工具：{', '.join(t.name for t in session.tools)}>"
    print(json.dumps(params, ensure_ascii=False, indent=2, default=str))


def print_tool(result: ToolResult):
    args = json.dumps(result.call.arguments, ensure_ascii=False) if result.call.arguments is not None \
        else result.call.raw_arguments
    status = "失败" if result.is_error else "成功"
    preview = result.content if len(result.content) <= 120 else result.content[:120] + "…"
    print(f"\n{GRAY}  ⚙ {result.call.name}({args}) → {status}：{preview}{RESET}")


def start(session: Session):
    p = session.provider
    print(f"[{PROVIDERS[p.name].label} · {p.model} · 登录用户 {session.nickname}（{session.ctx.user_id}）"
          f" · 商城工具 {len(session.tools) - 1} 个（MCP）]")
    print("/orders 查看我的订单 · /debug 查看请求 · /use 切换厂商 · /exit 退出\n")
    print(f"小林：{WELCOME}\n")


def main():
    parser = argparse.ArgumentParser(description="林间优选售后客服")
    parser.add_argument("-p", "--provider", default=config.PROVIDER, choices=list(PROVIDERS))
    parser.add_argument("-m", "--model", help="覆盖默认模型名")
    parser.add_argument("-u", "--user", default="u1001", help="登录用户 id")
    args = parser.parse_args()

    session = Session(get_provider(args.provider, args.model), user_token=issue_user_token(args.user))
    start(session)

    while True:
        try:
            user_text = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue

        if user_text == "/exit":
            break
        if user_text == "/reset":
            session.reset()
            print("\n[已开始新会话]\n")
            start(session)
            continue
        if user_text == "/debug":
            print_debug(session)
            continue
        if user_text == "/orders":
            print_orders(session.ctx.user_id)
            continue
        if user_text == "/history":
            print(f"[历史 {len(session.messages)} 条消息 · 累计用量 {session.total_usage}]\n")
            continue
        if user_text == "/providers":
            print_providers(session.provider.name)
            continue
        if user_text.startswith("/use"):
            name = user_text.removeprefix("/use").strip()
            if name not in PROVIDERS:
                print(f"可选厂商：{', '.join(PROVIDERS)}\n")
                continue
            if not has_key(name):
                print(f"[未配置 {PROVIDERS[name].api_key_env}，请先在 .env 中填写]\n")
                continue
            session = Session(get_provider(name), user_token=issue_user_token(session.ctx.user_id))
            print()
            start(session)
            continue

        print("小林：", end="", flush=True)
        try:
            turn = session.send(user_text, on_text=lambda t: print(t, end="", flush=True), on_tool=print_tool)
        except AUTH_ERRORS:
            print(f"\n[认证失败：请检查 .env 中的 {PROVIDERS[session.provider.name].api_key_env}]\n")
            continue
        except RATE_ERRORS:
            print("\n[请求太频繁或额度不足，稍等再试]\n")
            continue
        except STATUS_ERRORS as e:
            print(f"\n[接口错误 {e.status_code}：{e.message}]\n")
            continue
        except CONN_ERRORS:
            print("\n[网络连接失败，请检查网络或 BASE_URL 后重试]\n")
            continue

        report(turn)
        # 写操作：在命令行里让用户确认（相当于网页上的确认按钮）
        for a in turn.pending_actions:
            print("  ┌ 待确认的售后申请（确认后才会提交）")
            for k, v in a["summary"].items():
                print(f"  │ {k}：{v}")
            answer = input("  └ 确认提交？输入 y 确认，其他任意键取消：").strip().lower()
            print("小林：", end="", flush=True)
            if answer == "y":
                t = session.confirm(a["id"], on_text=lambda x: print(x, end="", flush=True), on_tool=print_tool)
                print(f"\n{GRAY}  ↳ 程序执行结果：{json.dumps(t.commit_result, ensure_ascii=False)}{RESET}")
            else:
                t = session.cancel(a["id"], on_text=lambda x: print(x, end="", flush=True))
            report(t)


def report(turn):
    u = turn.usage
    print(f"\n{GRAY}  ↳ {turn.steps} 步 · 调用工具 {len(turn.tool_results)} 次 · "
          f"输入 {u['input_tokens']} / 输出 {u['output_tokens']} tokens · {turn.model}{RESET}\n")
    if turn.handoff:
        print(f"  ↳ [系统] 已转接人工客服（{turn.handoff['reason']} · {turn.handoff.get('by', '')}）："
              f"{turn.handoff['summary']}\n")


if __name__ == "__main__":
    main()
