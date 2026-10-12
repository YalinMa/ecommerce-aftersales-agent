"""工具的通用定义：与模型厂商无关。

一个工具 = 名称 + 描述 + 参数 JSON Schema + 处理函数。
各厂商的适配器负责把它转换成自己的格式（Anthropic 的 input_schema / OpenAI 的 function）。
"""

import json
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class PendingAction:
    """等待用户确认的写操作。模型只能创建它，执行必须由用户在界面上确认。"""
    id: str
    kind: str          # 操作类型，如 create_aftersale
    params: dict       # 执行所需参数（已经过程序校验）
    summary: dict      # 展示给用户确认的内容


@dataclass
class ToolContext:
    """工具执行时的上下文，由程序注入，模型看不到也改不了。

    关键点：user_id 来自登录态，不是工具参数。这样即使用户在对话里说
    "帮我查一下 u1002 的订单"，模型也没有任何途径查到别人的数据。
    """
    user_id: str
    handoff: dict | None = None   # transfer_to_human 被调用后写入
    pending: dict[str, PendingAction] = field(default_factory=dict)   # 待确认的写操作


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict | None        # 解析失败时为 None
    raw_arguments: str = ""       # 模型输出的原始参数，便于排查


@dataclass
class ToolResult:
    call: ToolCall
    content: str
    is_error: bool = False
    elapsed_ms: int = 0           # 工具执行耗时，由 Session 记录


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict              # JSON Schema
    handler: Callable[..., Any]   # handler(ctx, **arguments) -> 可 JSON 序列化的结果
    extra: dict = field(default_factory=dict)


class ToolError(Exception):
    """业务层面的错误（如订单不存在），会作为 is_error 结果返回给模型。"""


def validate_arguments(schema: dict, args: Any) -> str | None:
    """最小化的参数校验，返回错误信息或 None。

    模型生成的参数是不可信输入：可能缺字段、类型不对、枚举值超出范围。
    """
    if not isinstance(args, dict):
        return "参数必须是 JSON 对象"
    props = schema.get("properties", {})
    for name in schema.get("required", []):
        if name not in args:
            return f"缺少必填参数 {name}"
    for name, value in args.items():
        if name not in props:
            return f"未知参数 {name}"
        spec = props[name]
        expected = {"string": str, "integer": int, "boolean": bool}.get(spec.get("type"))
        if expected and not isinstance(value, expected) or (expected is int and isinstance(value, bool)):
            return f"参数 {name} 应为 {spec['type']} 类型"
        if "enum" in spec and value not in spec["enum"]:
            return f"参数 {name} 只能取 {spec['enum']}"
    return None


def to_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)
