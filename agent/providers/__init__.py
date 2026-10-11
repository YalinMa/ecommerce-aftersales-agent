"""模型厂商注册表。

新增一个厂商：如果它提供 OpenAI 兼容接口，只需要在 PROVIDERS 里加一项。

模型名更新很快，这里的默认值截至 2026-10。可以用环境变量覆盖：
  - AGENT_MODEL：覆盖当前所选厂商的模型
  - <厂商>_MODEL，如 DEEPSEEK_MODEL：只覆盖该厂商
  - <厂商>_BASE_URL，如 MINIMAX_BASE_URL：切换国内 / 海外接入点
运行 `python scripts/list_models.py -p <厂商>` 可以查看账号下真实可用的模型名。
"""

import os
from dataclasses import dataclass, field

from agent import config
from agent.providers.base import Provider, StepResult


@dataclass
class ProviderSpec:
    kind: str                     # "anthropic" 或 "openai"（OpenAI 兼容接口）
    label: str
    api_key_env: str
    default_model: str
    base_url: str | None = None
    extra_params: dict = field(default_factory=dict)  # 厂商私有参数


PROVIDERS: dict[str, ProviderSpec] = {
    "anthropic": ProviderSpec(
        kind="anthropic", label="Anthropic Claude",
        api_key_env="ANTHROPIC_API_KEY", default_model="claude-opus-5-5",
    ),
    "openai": ProviderSpec(
        kind="openai", label="OpenAI",
        api_key_env="OPENAI_API_KEY", default_model="gpt-5.5",
        # Chat Completions 接口下，gpt-5.5 带工具调用时只支持 reasoning_effort=none（要推理需改用 Responses API）
        extra_params={"reasoning_effort": "none"},
    ),
    "deepseek": ProviderSpec(
        kind="openai", label="DeepSeek",
        api_key_env="DEEPSEEK_API_KEY", default_model="deepseek-flash",
        base_url="https://api.deepseek.com",
    ),
    "minimax": ProviderSpec(
        kind="openai", label="MiniMax",
        api_key_env="MINIMAX_API_KEY", default_model="MiniMax-M3",
        base_url="https://api.minimaxi.com/v1",          # 海外账号用 https://api.minimax.io/v1
        extra_params={"extra_body": {"reasoning_split": True}},  # 思考过程与正文分开返回
    ),
    "qwen": ProviderSpec(
        kind="openai", label="通义千问",
        api_key_env="DASHSCOPE_API_KEY", default_model="qwen3.7-plus",  # qwen-plus 是旧别名，V2 测试中工具使用明显更差
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
    "kimi": ProviderSpec(
        kind="openai", label="Kimi（月之暗面）",
        api_key_env="MOONSHOT_API_KEY", default_model="kimi-k3",
        base_url="https://api.moonshot.cn/v1",           # 海外账号用 https://api.moonshot.ai/v1
    ),
    "glm": ProviderSpec(
        kind="openai", label="智谱 GLM",
        api_key_env="ZHIPU_API_KEY", default_model="glm-5.3",
        base_url="https://open.bigmodel.cn/api/paas/v4",  # 海外账号用 https://api.z.ai/api/paas/v4
    ),
}


def resolve_model(name: str, model: str | None = None) -> str:
    spec = PROVIDERS[name]
    return (model
            or os.getenv(f"{name.upper()}_MODEL")
            or (os.getenv("AGENT_MODEL") if name == config.PROVIDER else None)
            or spec.default_model)


def has_key(name: str) -> bool:
    return bool(os.getenv(PROVIDERS[name].api_key_env))


def get_provider(name: str | None = None, model: str | None = None) -> Provider:
    name = name or config.PROVIDER
    if name not in PROVIDERS:
        raise SystemExit(f"未知的厂商 {name!r}，可选：{', '.join(PROVIDERS)}")
    spec = PROVIDERS[name]
    api_key = os.getenv(spec.api_key_env)
    if not api_key:
        raise SystemExit(f"未找到 {spec.label} 的 API key：请在 .env 中填写 {spec.api_key_env}")

    model = resolve_model(name, model)
    if spec.kind == "anthropic":
        from agent.providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(name, model, api_key, effort=config.EFFORT,
                                 max_tokens=config.MAX_TOKENS, use_fallback=config.USE_FALLBACK)

    from agent.providers.openai_compat import OpenAICompatProvider
    base_url = os.getenv(f"{name.upper()}_BASE_URL") or spec.base_url
    return OpenAICompatProvider(name, model, api_key, base_url=base_url, extra_params=spec.extra_params)


__all__ = ["PROVIDERS", "Provider", "StepResult", "get_provider", "has_key", "resolve_model"]
