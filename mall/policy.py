"""售后政策检索（V2 引入，V4 起由商城 MCP 服务提供）。

V0、V1 把全部政策写在 system prompt 里，每次请求都带上，政策越多 prompt 越长。
V2 把政策放进知识库，模型需要时调用本工具检索，只把相关的几条交给模型。

这种"模型自己决定什么时候检索、检索什么"的方式叫 agentic RAG。
另一种做法是每轮对话前固定检索一次、把结果塞进 prompt，实现简单，但闲聊和纯查订单时会浪费 token。
"""

from mall.knowledge import Retriever, default_retriever

_bm25_fallback: Retriever | None = None


def search_policy(query: str, top_k: int = 4) -> dict:
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
