"""OpenAI 兼容接口（Chat Completions）。

OpenAI、DeepSeek、MiniMax、通义千问、Kimi、智谱 GLM 都提供这种接口，
区别只在 base_url、模型名和少量私有参数，所以共用一个实现。
"""

import json
from typing import Callable

import openai

from agent.providers.base import StepResult, empty_usage
from agent.tools.base import Tool, ToolCall, ToolResult


class OpenAICompatProvider:
    def __init__(self, name: str, model: str, api_key: str | None,
                 base_url: str | None = None, extra_params: dict | None = None):
        self.name = name
        self.model = model
        self.extra_params = extra_params or {}
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def build_request(self, system_prompt: str, messages: list, tools: list[Tool],
                      allow_tools: bool = True) -> dict:
        params = {
            "model": self.model,
            # system 作为第一条消息；历史里只存 user / assistant / tool，每次请求时再拼上
            "messages": [{"role": "system", "content": system_prompt}, *messages],
            "stream": True,
            "stream_options": {"include_usage": True},
            **self.extra_params,
        }
        if tools:
            params["tools"] = [{
                "type": "function",
                "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
            } for t in tools]
            if not allow_tools:
                params["tool_choice"] = "none"
        return params

    def step(self, system_prompt: str, messages: list, tools: list[Tool],
             on_text: Callable[[str], None] | None = None, allow_tools: bool = True) -> StepResult:
        params = self.build_request(system_prompt, messages, tools, allow_tools)
        think = _ThinkTagFilter()
        parts: list[str] = []
        reasoning: list[str] = []
        calls: dict[int, dict] = {}   # 流式返回时，工具调用按 index 分片到达，需要自己拼接
        usage = empty_usage()
        model, finish_reason = self.model, None

        def emit(s: str):
            if s:
                parts.append(s)
                if on_text:
                    on_text(s)

        for chunk in self.client.chat.completions.create(**params):
            model = chunk.model or model
            if chunk.usage:  # 开启 include_usage 后，最后一个 chunk 带用量
                usage = _normalize_usage(chunk.usage)
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            finish_reason = choice.finish_reason or finish_reason
            delta = choice.delta
            # 推理模型的思考过程在 reasoning_content 字段里，不展示给用户
            if r := getattr(delta, "reasoning_content", None):
                reasoning.append(r)
            emit(think.feed(delta.content or ""))
            for tc in delta.tool_calls or []:
                slot = calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    slot["name"] += tc.function.name
                if tc.function and tc.function.arguments:
                    slot["arguments"] += tc.function.arguments
        emit(think.flush())

        text = "".join(parts).strip()
        tool_calls = [ToolCall(c["id"], c["name"], _parse_args(c["arguments"]), c["arguments"])
                      for _, c in sorted(calls.items())]
        if finish_reason == "length":
            tool_calls = []  # 输出被截断，参数可能不完整，不能执行

        assistant_message: dict = {"role": "assistant", "content": text or None}
        if tool_calls:
            assistant_message["tool_calls"] = [{
                "id": c.id, "type": "function",
                "function": {"name": c.name, "arguments": c.raw_arguments or "{}"},
            } for c in tool_calls]
            # 思考模式下调用工具时，DeepSeek 等厂商要求在同一轮里回传思考过程；
            # 没有工具调用的普通回复则不回传
            if reasoning:
                assistant_message["reasoning_content"] = "".join(reasoning)

        return StepResult(text=text, assistant_message=assistant_message, tool_calls=tool_calls,
                          stop_reason=finish_reason, model=model, usage=usage)

    def tool_result_messages(self, results: list[ToolResult]) -> list[dict]:
        # OpenAI 格式：每个工具结果单独一条 role=tool 的消息，用 tool_call_id 对应
        return [{
            "role": "tool",
            "tool_call_id": r.call.id,
            "content": ("[错误] " + r.content) if r.is_error else r.content,
        } for r in results]


def _parse_args(raw: str) -> dict | None:
    try:
        args = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return None
    return args if isinstance(args, dict) else None


def _normalize_usage(u) -> dict:
    cached = 0
    if u.prompt_tokens_details and u.prompt_tokens_details.cached_tokens:
        cached = u.prompt_tokens_details.cached_tokens
    else:  # DeepSeek 用自己的字段名
        cached = (u.model_extra or {}).get("prompt_cache_hit_tokens", 0) or 0
    return {
        "input_tokens": u.prompt_tokens or 0,
        "output_tokens": u.completion_tokens or 0,
        "cache_read_input_tokens": cached,
    }


class _ThinkTagFilter:
    """有些模型把思考过程用 <think>...</think> 写在正文里，流式过滤掉。
    标签可能被切在两个 chunk 之间，所以要缓存可能是标签开头的尾巴。"""

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self):
        self.buf = ""
        self.inside = False

    def feed(self, s: str) -> str:
        self.buf += s
        out = []
        while True:
            tag = self.CLOSE if self.inside else self.OPEN
            idx = self.buf.find(tag)
            if idx >= 0:
                if not self.inside:
                    out.append(self.buf[:idx])
                self.buf = self.buf[idx + len(tag):]
                self.inside = not self.inside
                continue
            # 保留可能是半个标签的尾巴，其余输出（或在思考中时丢弃）
            keep = next((k for k in range(len(tag) - 1, 0, -1) if self.buf.endswith(tag[:k])), 0)
            ready, self.buf = self.buf[:len(self.buf) - keep], self.buf[len(self.buf) - keep:]
            if not self.inside:
                out.append(ready)
            return "".join(out)

    def flush(self) -> str:
        rest, self.buf = ("" if self.inside else self.buf), ""
        return rest
