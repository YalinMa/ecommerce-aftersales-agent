"""售后政策检索工具（V2）。

V0、V1 把全部政策写在 system prompt 里，每次请求都带上，政策越多 prompt 越长。
V2 把政策放进知识库，模型需要时调用本工具检索，只把相关的几条交给模型。

这种"模型自己决定什么时候检索、检索什么"的方式叫 agentic RAG。
另一种做法是每轮对话前固定检索一次、把结果塞进 prompt，实现简单，但闲聊和纯查订单时会浪费 token。
"""

from agent.knowledge import Retriever, default_retriever
from agent.tools.base import Tool, ToolContext

_bm25_fallback: Retriever | None = None


def search_policy(ctx: ToolContext, query: str, top_k: int = 4) -> dict:
    global _bm25_fallback
    top_k = max(1, min(top_k, 6))
    note = None
    try:
        hits = default_retriever().search(query, top_k)
    except Exception:
        # 向量接口出错时退回关键词检索，保证客服还能用
        _bm25_fallback = _bm25_fallback or Retriever("bm25")
        hits = _bm25_fallback.search(query, top_k)
        note = "向量检索暂不可用，本次为关键词检索结果"
    result = {
        "results": [{"id": h.chunk.id, "title": h.chunk.title, "doc": h.chunk.doc, "content": h.chunk.content}
                    for h in hits],
        "instruction": "只依据与问题相关的条款回答，并在回复末尾注明依据的条款编号和标题；"
                       "如果这些条款都不能回答用户的问题，说明没有查到明确规定，不要自行补充。",
    }
    if note:
        result["note"] = note
    return result


SEARCH_POLICY = Tool(
    name="search_policy",
    description=(
        "检索林间优选的平台规则知识库，返回最相关的几条条款（含编号、标题、全文）。"
        "知识库覆盖：退换货、质量问题与保修、退款与优惠券积分、发货物流、运费险、价格保护、发票、"
        "人工客服服务时间、投诉渠道、支付方式、订单取消。"
        "回答任何关于平台规则、时效、费用、流程、服务安排的问题前都必须调用，不要凭印象或常识回答。"
        "query 用简洁的问题或关键词，如'贴身衣物 无理由退货'。一个问题涉及多个方面时，可以分别检索。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "检索内容，简洁的问题或关键词"},
            "top_k": {"type": "integer", "description": "返回条数，默认 4，最多 6"},
        },
        "required": ["query"],
    },
    handler=search_policy,
)
