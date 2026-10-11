"""一次客服会话：维护多轮对话历史，并运行 agent loop。

大模型本身是无状态的，所谓多轮对话，就是每一轮都把完整历史重新发一遍。

V1 起，用户说一句话后不再是"请求一次模型就结束"，而是一个循环：

    用户消息
      ↓
    ┌→ 请求模型（带上工具定义）
    │    ↓
    │  模型要调用工具吗？── 否 ──→ 本轮结束，回复用户
    │    ↓ 是
    │  程序执行工具（查数据库），把结果追加进历史
    └────┘

模型只负责"决定调什么、用什么参数"，真正执行的永远是程序。
"""

import json
import re
from dataclasses import dataclass, field
from typing import Callable

from agent import config, db
from agent.providers import Provider, get_provider
from agent.providers.base import add_usage, empty_usage
from agent.tools import TOOLS, Tool, ToolContext, ToolResult, execute

# 一轮对话里最多调用模型几次，防止模型反复调用工具陷入死循环
MAX_STEPS = 6
# 知识库条款编号，如 R-01、F-02
CLAUSE_ID = re.compile(r"\b([A-Z]-\d{2})\b")


@dataclass
class TurnResult:
    """一轮用户对话的结果。"""
    text: str
    steps: int
    tool_results: list[ToolResult] = field(default_factory=list)
    usage: dict = field(default_factory=empty_usage)
    model: str = ""
    handoff: dict | None = None    # 本轮触发了转人工时有值
    unverified_citations: list[str] = field(default_factory=list)  # 引用了但从未检索到的条款编号


class Session:
    def __init__(self, provider: Provider | None = None, user_id: str = "u1001",
                 tools: list[Tool] | None = None, system_prompt: str | None = None):
        db.ensure_exists()
        self.provider = provider or get_provider()
        self.ctx = ToolContext(user_id=user_id)
        self.tools = TOOLS if tools is None else tools
        self.system_prompt = system_prompt or config.load_system_prompt(user_id)
        self.messages: list[dict] = []
        self.total_usage = empty_usage()
        self.retrieved_ids: set[str] = set()   # 本会话里 search_policy 实际返回过的条款

    def send(self, user_text: str,
             on_text: Callable[[str], None] | None = None,
             on_tool: Callable[[ToolResult], None] | None = None) -> TurnResult:
        checkpoint = len(self.messages)
        self.messages.append({"role": "user", "content": user_text})
        turn = TurnResult(text="", steps=0)
        texts: list[str] = []
        self.ctx.handoff = None

        try:
            empty_retried = False
            for step in range(MAX_STEPS):
                # 最后一步禁止再调工具，强制模型基于已有信息给出回复
                allow_tools = step < MAX_STEPS - 1
                r = self.provider.step(self.system_prompt, self.messages, self.tools,
                                       on_text=on_text, allow_tools=allow_tools)
                add_usage(turn.usage, r.usage)
                turn.steps, turn.model = step + 1, r.model
                # 兜底：整轮既没有文字也没有工具调用（推理模型偶尔把内容全写进思考过程），重试一次
                if not r.text and not r.tool_calls and not texts and not empty_retried:
                    empty_retried = True
                    continue
                self.messages.append(r.assistant_message)
                if r.text:
                    texts.append(r.text)
                if not r.tool_calls:
                    break

                results = []
                for call in r.tool_calls:
                    result = execute(call, self.ctx)
                    results.append(result)
                    if on_tool:
                        on_tool(result)
                turn.tool_results.extend(results)
                self.messages.extend(self.provider.tool_result_messages(results))
        except Exception:
            # 请求失败时撤回本轮所有消息，保证历史结构完整（不会留下没有结果的工具调用）
            del self.messages[checkpoint:]
            raise

        turn.text = "\n".join(texts)
        turn.handoff = self.ctx.handoff
        turn.unverified_citations = self._check_citations(turn)
        add_usage(self.total_usage, turn.usage)
        return turn

    def _check_citations(self, turn: TurnResult) -> list[str]:
        """引用核查：模型可能编造"依据：X-01"这样看起来很正规的出处。

        规则很简单：回复里出现的条款编号，必须在本会话里被 search_policy 真正返回过。
        这一步完全由程序完成，不依赖模型自觉。
        """
        for r in turn.tool_results:
            if r.call.name == "search_policy" and not r.is_error:
                self.retrieved_ids.update(h["id"] for h in json.loads(r.content)["results"])
        return sorted(set(CLAUSE_ID.findall(turn.text)) - self.retrieved_ids)

    def build_request(self, next_user_text: str = "<你的下一句话>") -> dict:
        """下一轮将要发给模型的完整请求（不发送）。"""
        return self.provider.build_request(
            self.system_prompt, self.messages + [{"role": "user", "content": next_user_text}], self.tools
        )

    def reset(self):
        self.messages.clear()
        self.retrieved_ids.clear()
        self.ctx.handoff = None
