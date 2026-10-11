"""工具注册表和执行器。"""

from agent.tools.base import (Tool, ToolCall, ToolContext, ToolError, ToolResult, to_json,
                              validate_arguments)
from agent.tools.handoff import TRANSFER_TO_HUMAN
from agent.tools.orders import GET_AFTERSALE_STATUS, GET_LOGISTICS, GET_ORDER_DETAIL, LIST_MY_ORDERS
from agent.tools.policy import SEARCH_POLICY

TOOLS: list[Tool] = [SEARCH_POLICY, LIST_MY_ORDERS, GET_ORDER_DETAIL, GET_LOGISTICS, GET_AFTERSALE_STATUS,
                     TRANSFER_TO_HUMAN]
_BY_NAME = {t.name: t for t in TOOLS}


def execute(call: ToolCall, ctx: ToolContext) -> ToolResult:
    """执行一次工具调用。任何错误都转成 is_error 结果交给模型，而不是让整个对话崩溃。"""
    tool = _BY_NAME.get(call.name)
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


__all__ = ["TOOLS", "Tool", "ToolCall", "ToolContext", "ToolResult", "execute"]
