"""工具执行器，以及客服 Agent 自己的工具。

V4 起，订单、物流、售后、知识库这些业务工具由商城 MCP 服务提供，运行时通过 mall_client 自动发现；
这里只保留属于客服平台自身的工具：转人工。
"""

from agent.tools.base import (Tool, ToolCall, ToolContext, ToolError, ToolResult, to_json,
                              validate_arguments)
from agent.tools.handoff import TRANSFER_TO_HUMAN

LOCAL_TOOLS: list[Tool] = [TRANSFER_TO_HUMAN]


def execute(call: ToolCall, ctx: ToolContext, tools: list[Tool]) -> ToolResult:
    """执行一次工具调用。任何错误都转成 is_error 结果交给模型，而不是让整个对话崩溃。"""
    tool = next((t for t in tools if t.name == call.name), None)
    if tool is None:
        return ToolResult(call, f"不存在名为 {call.name} 的工具", is_error=True)
    if call.arguments is None:
        return ToolResult(call, to_json({"INVALID_JSON": call.raw_arguments}), is_error=True)
    if err := validate_arguments(tool.parameters, call.arguments):
        return ToolResult(call, f"参数错误：{err}", is_error=True)
    try:
        return ToolResult(call, to_json(tool.handler(ctx, **call.arguments)))
    except ToolError as e:
        return ToolResult(call, str(e), is_error=True)
    except Exception as e:  # 工具内部 bug 也不应让对话中断
        return ToolResult(call, f"工具执行失败：{type(e).__name__}", is_error=True)


__all__ = ["LOCAL_TOOLS", "Tool", "ToolCall", "ToolContext", "ToolResult", "execute"]
