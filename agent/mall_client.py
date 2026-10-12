"""商城 MCP 客户端：客服 Agent 通过它使用商城的能力（V4）。

V1–V3 里客服 Agent 直接 import 商城的代码、直接读数据库。V4 起两边彻底分开：
- 商城是独立的服务（mall/mcp_server.py），自己管数据、规则和身份校验。
- 客服 Agent 只认识 MCP 协议：连接服务 → 自动发现有哪些工具 → 把工具交给大模型 → 转发调用。
  这里没有任何一行代码知道订单表长什么样。

身份与确认：
- 每个会话带着用户的登录令牌（由商城登录服务签发），放在请求头里，商城验签后才知道是谁。
- 用户点击「确认提交」时，这里用共享密钥计算确认签名一起发给商城。大模型拿不到密钥，
  也看不到确认类工具（它们带 host_only 标记，不会交给大模型）。
"""

import asyncio
import atexit
import hashlib
import hmac
import json
import os
import subprocess
import sys
import time
import urllib.request
from functools import lru_cache

from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client

from agent import config
from agent.tools.base import Tool, ToolError

DEFAULT_URL = os.getenv("MALL_MCP_URL", "http://127.0.0.1:8765/mcp")
# 与商城约定的确认签名密钥（双方都从环境变量读取，演示环境用默认值）
CONFIRM_SECRET = os.getenv("MALL_CONFIRM_SECRET", "dev-only-change-me-confirm").encode()


class MallClient:
    def __init__(self, user_token: str, url: str = DEFAULT_URL):
        self.url = url
        self.headers = {"Authorization": f"Bearer {user_token}"}
        ensure_server(url)

    # ---------- MCP 调用 ----------

    async def _with_client(self, fn):
        transport = streamable_http_client(self.url, http_client=create_mcp_http_client(headers=self.headers))
        async with Client(transport) as client:
            return await fn(client)

    def _run(self, fn):
        # 每次调用新建一个连接：实现简单，本机调用开销在几十毫秒，可接受
        return asyncio.run(self._with_client(fn))

    def call(self, name: str, arguments: dict) -> dict:
        """调用商城工具，返回解析后的 JSON；商城返回错误时抛出 ToolError，由执行器转给大模型。"""
        try:
            result = self._run(lambda c: c.call_tool(name, arguments))
        except Exception as e:   # 网络或服务故障：告诉大模型"暂时查不了"，而不是让对话崩溃
            raise ToolError(f"商城服务暂时不可用（{type(e).__name__}），请稍后再试或转人工") from None
        text = "".join(getattr(part, "text", "") for part in result.content)
        if result.is_error:
            raise ToolError(text.removeprefix(f"Error executing tool {name}: "))
        return json.loads(text) if text else {}

    # ---------- 工具发现 ----------

    def model_tools(self) -> list[Tool]:
        """商城提供的、可以交给大模型的工具（自动发现，不在客服 Agent 里写死）。"""
        tools = self._run(lambda c: c.list_tools()).tools
        return [Tool(name=t.name, description=t.description or "", parameters=simplify_schema(t.input_schema),
                     handler=self._handler(t.name))
                for t in tools if not (t.meta or {}).get("host_only")]

    def _handler(self, name: str):
        def handler(ctx, **arguments):
            return self.call(name, arguments)
        return handler

    # ---------- 宿主程序专用 ----------

    def profile(self) -> dict:
        return self.call("get_my_profile", {})

    def confirm(self, user_id: str, action_id: str) -> dict:
        return self.call("confirm_aftersale_request",
                         {"action_id": action_id, "signature": _signature(user_id, action_id, "confirm")})

    def cancel(self, user_id: str, action_id: str) -> dict:
        return self.call("cancel_aftersale_request",
                         {"action_id": action_id, "signature": _signature(user_id, action_id, "cancel")})

    def clause_titles(self) -> dict[str, str]:
        return _clause_titles(self.url, json.dumps(self.headers))


@lru_cache(maxsize=4)
def _clause_titles(url: str, headers_json: str) -> dict[str, str]:
    """从商城的 kb://clauses 资源读取条款编号和标题，用于引用核查。每个进程只读一次。"""
    async def read():
        transport = streamable_http_client(url, http_client=create_mcp_http_client(headers=json.loads(headers_json)))
        async with Client(transport) as c:
            res = await c.read_resource("kb://clauses")
            return json.loads(res.contents[0].text)
    return {c["id"]: c["title"] for c in asyncio.run(read())}


def _signature(user_id: str, action_id: str, decision: str) -> str:
    return hmac.new(CONFIRM_SECRET, f"{user_id}:{action_id}:{decision}".encode(), hashlib.sha256).hexdigest()


def simplify_schema(schema: dict) -> dict:
    """把 MCP（pydantic）生成的参数 schema 整理成更简洁的形式再交给大模型。

    可选参数会被写成 anyOf: [{type: string}, {type: null}]，还带 title、default 等字段。
    部分模型对这种写法的理解不稳定，这里改成"非必填的 string"。
    """
    props = {}
    for name, spec in schema.get("properties", {}).items():
        spec = {k: v for k, v in spec.items() if k not in ("title", "default")}
        if "anyOf" in spec:
            options = [o for o in spec.pop("anyOf") if o.get("type") != "null"]
            if len(options) == 1:
                spec = {**options[0], **spec}
        props[name] = spec
    return {"type": "object", "properties": props, "required": schema.get("required", [])}


# ---------- 本地开发：自动启动商城服务 ----------

_server_process: subprocess.Popen | None = None


def _healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(url.rsplit("/mcp", 1)[0] + "/health", timeout=1) as r:
            return r.status == 200
    except OSError:
        return False


def ensure_server(url: str = DEFAULT_URL):
    """商城服务没运行时，本地开发环境自动在后台启动一个（进程退出时一并关闭）。"""
    global _server_process
    if _healthy(url):
        return
    if not url.startswith("http://127.0.0.1") and not url.startswith("http://localhost"):
        raise SystemExit(f"无法连接商城 MCP 服务：{url}")
    port = url.rsplit(":", 1)[1].split("/")[0]
    log = open(config.RUNS_DIR.parent / "data" / "mall_mcp.log", "a")
    _server_process = subprocess.Popen([sys.executable, "-m", "mall.mcp_server", "--port", port],
                                       cwd=config.ROOT, stdout=log, stderr=log)
    atexit.register(_server_process.terminate)
    for _ in range(50):
        if _healthy(url):
            return
        time.sleep(0.2)
    raise SystemExit("商城 MCP 服务启动失败，详见 data/mall_mcp.log")
