"""集中管理配置，后续版本（工具、RAG、MCP）也从这里读取。

各厂商的 API key、模型名、接入点见 agent/providers/__init__.py。
"""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# 默认使用的厂商，可在命令行用 -p 覆盖
PROVIDER = os.getenv("AGENT_PROVIDER", "deepseek")

# 以下仅对 Anthropic 生效。客服对话延迟敏感、推理不重，effort 默认用 low。
EFFORT = os.getenv("AGENT_EFFORT", "low")
MAX_TOKENS = int(os.getenv("AGENT_MAX_TOKENS", "16000"))
USE_FALLBACK = os.getenv("AGENT_USE_FALLBACK", "1") == "1"

SYSTEM_PROMPT_PATH = ROOT / "agent" / "prompts" / "system_prompt.md"
RUNS_DIR = ROOT / "runs"


WEEKDAYS = "一二三四五六日"


def load_system_prompt(nickname: str | None = None) -> str:
    """固定的 prompt 模板 + 本次会话的上下文（登录用户、今天日期）。

    上下文放在最后：模板部分每次请求都一样，便于厂商做 prompt 缓存；
    日期只精确到天，同一天内也不会变。
    """
    from datetime import date

    prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    if not nickname:
        return prompt
    today = date.today()
    return prompt + (
        "\n# 当前会话\n\n"
        f"- 登录用户昵称：{nickname}（已通过登录验证，工具查询自动限定为该用户的数据）\n"
        f"- 今天日期：{today:%Y-%m-%d}，星期{WEEKDAYS[today.weekday()]}\n"
    )
