"""所有模型厂商共用的数据结构和接口。

不同厂商的消息格式不一样，tool calling 的格式差异更大：

                     Anthropic                         OpenAI 兼容
  工具定义           {name, description, input_schema}  {type: function, function: {name, description, parameters}}
  模型发起调用       content 里的 tool_use 块            message.tool_calls 数组
  工具结果怎么回传   一条 user 消息，content 是         每个调用一条 role=tool 的消息
                     一组 tool_result 块

所以每个 Provider 负责：
  1. 把「system + 历史 + 工具定义」组装成自己厂商的请求
  2. 把一次模型输出解析成统一的 StepResult（文字 + 工具调用）
  3. 把工具执行结果转换成自己厂商的消息格式，追加进历史
"""

from dataclasses import dataclass, field
from typing import Callable, Protocol

from agent.tools.base import Tool, ToolCall, ToolResult


@dataclass
class StepResult:
    """模型的一次输出。一轮用户对话里可能有多步：调用工具 → 拿到结果 → 再输出。"""
    text: str                       # 这一步里给用户看的文字
    assistant_message: dict         # 原样追加进历史的 assistant 消息（厂商原生格式）
    tool_calls: list[ToolCall]      # 模型要求调用的工具，为空表示这一轮结束
    stop_reason: str | None
    model: str
    usage: dict = field(default_factory=dict)  # input_tokens / output_tokens / cache_read_input_tokens


class Provider(Protocol):
    name: str
    model: str

    def build_request(self, system_prompt: str, messages: list, tools: list[Tool],
                      allow_tools: bool = True) -> dict:
        """组装请求参数（不发送），用于 /debug 查看。"""
        ...

    def step(self, system_prompt: str, messages: list, tools: list[Tool],
             on_text: Callable[[str], None] | None = None, allow_tools: bool = True) -> StepResult:
        ...

    def tool_result_messages(self, results: list[ToolResult]) -> list[dict]:
        ...


def empty_usage() -> dict:
    return {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0}


def add_usage(total: dict, usage: dict):
    for k, v in usage.items():
        total[k] = total.get(k, 0) + v
