"""Anthropic（Claude）官方 SDK。"""

from typing import Callable

import anthropic

from agent.providers.base import StepResult
from agent.tools.base import Tool, ToolCall, ToolResult

# 模型内部的块类型：发生中途换模型（fallback）时，换模型之前的这些块不能回传
_INTERNAL_BLOCK_TYPES = {"thinking", "redacted_thinking", "tool_use"}
# 工具参数 JSON 完全无法解析时 SDK 会在流中抛 ValueError，此时没有 tool_use_id 可回复，只能重发请求
_MAX_INVALID_JSON_RETRIES = 2


class AnthropicProvider:
    def __init__(self, name: str, model: str, api_key: str | None,
                 effort: str = "low", max_tokens: int = 16000, use_fallback: bool = True):
        self.name = name
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.use_fallback = use_fallback
        self.client = anthropic.Anthropic(api_key=api_key)

    def build_request(self, system_prompt: str, messages: list, tools: list[Tool],
                      allow_tools: bool = True) -> dict:
        params = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system_prompt,          # system 是独立参数，不在 messages 里
            "messages": messages,
            "output_config": {"effort": self.effort},
        }
        if tools:
            params["tools"] = [{
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
                # 流式请求下让工具参数边生成边返回；代价是服务端不再校验参数，由我们自己校验
                "eager_input_streaming": True,
            } for t in tools]
            if not allow_tools:
                params["tool_choice"] = {"type": "none"}
        if self.use_fallback:
            # 安全分类器偶尔误判拒答时，服务端自动换模型重试
            params["betas"] = ["server-side-fallback-2026-07-01"]
            params["fallbacks"] = "default"
        return params

    def step(self, system_prompt: str, messages: list, tools: list[Tool],
             on_text: Callable[[str], None] | None = None, allow_tools: bool = True) -> StepResult:
        params = self.build_request(system_prompt, messages, tools, allow_tools)
        api = self.client.beta.messages if self.use_fallback else self.client.messages

        for attempt in range(_MAX_INVALID_JSON_RETRIES + 1):
            try:
                with api.stream(**params) as stream:
                    for chunk in stream.text_stream:
                        if on_text:
                            on_text(chunk)
                    final = stream.get_final_message()
                break
            except ValueError:  # 只捕获 JSON 解析错误；认证、限流等 API 错误照常抛出
                if attempt == _MAX_INVALID_JSON_RETRIES:
                    raise

        text = "".join(b.text for b in final.content if b.type == "text")
        tool_calls = [ToolCall(b.id, b.name, b.input if isinstance(b.input, dict) else None, str(b.input))
                      for b in final.content if b.type == "tool_use"]

        if final.stop_reason == "refusal":
            text = text or "抱歉，这个问题我暂时无法处理。"
            tool_calls = []  # 拒答可能把工具调用截断在一半，不能执行
        elif final.stop_reason == "max_tokens" and tool_calls:
            tool_calls = []  # 参数可能被截断，看起来完整也不能执行

        return StepResult(
            text=text,
            # assistant 的 content 是一组 block（thinking + text + tool_use），原样回传
            assistant_message={"role": "assistant", "content": _content_for_history(final.content)},
            tool_calls=tool_calls,
            stop_reason=final.stop_reason,
            model=final.model,
            usage={
                "input_tokens": final.usage.input_tokens,
                "output_tokens": final.usage.output_tokens,
                "cache_read_input_tokens": final.usage.cache_read_input_tokens or 0,
            },
        )

    def tool_result_messages(self, results: list[ToolResult]) -> list[dict]:
        # 同一步的所有工具结果放进同一条 user 消息；拆成多条会让模型以后不再并行调用工具
        return [{"role": "user", "content": [{
            "type": "tool_result",
            "tool_use_id": r.call.id,
            "content": r.content,
            "is_error": r.is_error,
        } for r in results]}]


def _content_for_history(blocks) -> list:
    """正常情况下原样保留（包括 thinking 块，删掉反而可能导致 400）。
    只有发生中途换模型时，才去掉换模型之前的内部块。"""
    dumped = [b.model_dump(exclude_none=True) for b in blocks]
    fallback_idx = max((i for i, b in enumerate(dumped) if b["type"] == "fallback"), default=-1)
    return [
        b for i, b in enumerate(dumped)
        if not (i < fallback_idx and b["type"] in _INTERNAL_BLOCK_TYPES)
    ]
