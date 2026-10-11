"""转人工工具：取代 V0 的 [转人工] 文本标记。

以前靠模型在回复末尾写一个标记，程序再做字符串匹配，两轮测试里出现过
"说了要转却没写标记""只是询问却写了标记"两类错误。
改成工具后，转接是一个明确的动作：模型调用了就转，没调用就不转。
"""

from agent.tools.base import Tool, ToolContext


def transfer_to_human(ctx: ToolContext, reason: str, summary: str) -> dict:
    ctx.handoff = {"reason": reason, "summary": summary}
    # V6 会在这里把会话推送到真正的人工客服队列
    return {"transferred": True, "note": "已加入人工客服队列。请告诉用户人工客服即将接入，不要再继续处理该问题。"}


TRANSFER_TO_HUMAN = Tool(
    name="transfer_to_human",
    description=(
        "把当前对话转接给人工客服。调用后系统会立即转接。"
        "在以下情况调用：用户明确要求人工；涉及人身安全（如电器冒烟起火、商品致伤）、法律纠纷或投诉曝光；"
        "用户要求政策以外的赔偿或特殊处理；用户情绪激烈且安抚无效；需要操作订单而你没有对应能力。"
        "只是询问用户是否需要人工时不要调用，等用户确认后再调用。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "reason": {"type": "string", "enum": ["用户要求", "安全问题", "政策外诉求", "投诉纠纷", "情绪激烈", "超出能力"],
                       "description": "转接原因"},
            "summary": {"type": "string", "description": "一句话概括已了解的情况（订单号、问题、用户诉求），供人工客服快速接手"},
        },
        "required": ["reason", "summary"],
    },
    handler=transfer_to_human,
)
