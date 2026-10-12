"""林间优选商城 MCP 服务：把订单、物流、售后、知识库能力以 MCP 协议对外提供。

用法：
    python -m mall.mcp_server                  # HTTP 方式，供客服 Agent 连接（默认 http://127.0.0.1:8765/mcp）
    python -m mall.mcp_server --stdio          # stdio 方式，供 Claude Desktop 等 MCP 客户端直接接入

MCP（Model Context Protocol）是一个开放协议：业务方按协议提供一次工具，
任何支持 MCP 的 Agent 都能自动发现并调用，不需要为每个 Agent 单独对接。

本服务提供：
    工具（给大模型用）   search_policy、list_my_orders、get_order_detail、get_logistics、
                          get_aftersale_status、check_return_eligibility、create_aftersale_request
    工具（只给宿主程序用）get_my_profile、confirm_aftersale_request、cancel_aftersale_request
                          带 meta.host_only 标记，Agent 不会把它们交给大模型；确认类工具还要求签名
    资源                  kb://clauses：知识库全部条款，供 Agent 核对引用
    健康检查              GET /health

身份：HTTP 方式从请求头 Authorization: Bearer <令牌> 验签得到 user_id（见 auth.py）；
stdio 方式没有请求头，使用环境变量 MALL_STDIO_USER 指定的演示用户（默认 u1001）。
"""

import argparse
import json
import os
from typing import Annotated, Literal

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field
from starlette.requests import Request
from starlette.responses import JSONResponse

from mall import BusinessError, aftersale, auth, orders, policy, rules
from mall.knowledge import load_chunks

DEFAULT_PORT = int(os.getenv("MALL_MCP_PORT", "8765"))
HOST_ONLY = {"host_only": True}

app = MCPServer(
    name="linjian-mall",
    title="林间优选商城",
    instructions="林间优选商城的订单、物流、售后与售后政策服务。所有查询都限定为当前登录用户本人的数据。",
)

OrderStatus = Literal["待发货", "已发货", "已签收", "已完成", "已取消", "售后中"]
ReasonType = Literal[tuple(rules.REASON_TYPES)]
RequestType = Literal[tuple(rules.REQUEST_TYPES)]


def _user(ctx: Context) -> str:
    """从请求里取出已验证的用户身份。"""
    headers = ctx.headers
    if headers is None:   # stdio 方式：本地演示用户
        return os.getenv("MALL_STDIO_USER", "u1001")
    value = headers.get("authorization", "")
    if not value.lower().startswith("bearer "):
        raise ToolError("缺少身份令牌")
    try:
        return auth.verify_user_token(value[7:].strip())
    except BusinessError as e:
        raise ToolError(str(e)) from None


def _run(fn, *args, **kwargs):
    """业务错误转成 MCP 工具错误，调用方会收到 isError 的结果。"""
    try:
        return fn(*args, **kwargs)
    except BusinessError as e:
        raise ToolError(str(e)) from None


# ---------- 给大模型用的工具 ----------

@app.tool(description=(
    "检索林间优选的平台规则知识库，返回最相关的几条条款（含编号、标题、全文）。"
    "知识库覆盖：退换货、质量问题与保修、退款与优惠券积分、发货物流、运费险、价格保护、发票、"
    "人工客服服务时间、投诉渠道、支付方式、订单取消。"
    "回答任何关于平台规则、时效、费用、流程、服务安排的问题前都必须调用，不要凭印象或常识回答。"
    "query 用简洁的问题或关键词，如'贴身衣物 无理由退货'。一个问题涉及多个方面时，可以分别检索。"))
def search_policy(query: Annotated[str, Field(description="检索内容，简洁的问题或关键词")],
                  top_k: Annotated[int, Field(description="返回条数，默认 4，最多 6")] = 4) -> dict:
    return policy.search_policy(query, top_k)


@app.tool(description=(
    "列出当前登录用户的订单，按付款时间倒序。当用户没有提供订单号、只描述了商品（如'上周买的耳机'）"
    "或想看最近的订单时调用；可以用 keyword 按商品名筛选。找到唯一匹配的订单后，再用 get_order_detail 查看详情。"))
def list_my_orders(ctx: Context,
                   status: Annotated[OrderStatus | None, Field(description="按订单状态筛选，不传则返回全部")] = None,
                   keyword: Annotated[str | None, Field(description="商品名关键词，如'耳机''保温杯'")] = None,
                   limit: Annotated[int, Field(description="返回条数，默认 10，最多 20")] = 10) -> dict:
    return _run(orders.list_orders, _user(ctx), status, keyword, limit)


@app.tool(description=(
    "查询当前用户某个订单的详情：商品（含是否贴身衣物、生鲜、数码等特殊标签）、金额、支付方式、是否有运费险、"
    "付款/发货/签收时间，以及已签收天数。判断能否退换货、是否在时效内之前，必须先调用它获取真实的签收时间和商品类别，"
    "不要让用户自己报。"))
def get_order_detail(ctx: Context, order_id: Annotated[str, Field(description="订单号，如 LJ10010002")]) -> dict:
    return _run(orders.order_detail, _user(ctx), order_id)


@app.tool(description=(
    "查询当前用户某个订单的物流轨迹，返回快递公司、单号、全部轨迹节点，以及距最近一次更新过去了多少小时。"
    "用户问'到哪了''什么时候到''物流不动了'时调用。"))
def get_logistics(ctx: Context, order_id: Annotated[str, Field(description="订单号")]) -> dict:
    return _run(orders.logistics, _user(ctx), order_id)


@app.tool(description=(
    "查询当前用户某个订单的售后记录：类型、原因、进度状态、仓库签收时间（及已过去的工作日数）、退款金额和时间。"
    "用户问退货退款进度、钱什么时候到账时调用。"))
def get_aftersale_status(ctx: Context, order_id: Annotated[str, Field(description="订单号")]) -> dict:
    return _run(orders.aftersale_status, _user(ctx), order_id)


@app.tool(description=(
    "按平台规则判断某件商品能否申请售后：返回是否符合、可申请的类型（退货退款/换货/仅退款/维修）、申请截止日期、"
    "退回运费由谁承担、需要用户自行确认的前提条件，以及对应的知识库条款。"
    "用户想退货、换货、退款或维修时，先调用它，不要自己推算日期和规则。"
    "reason_type：用户只是不想要了用'无理由'；商品有故障、破损、变质等用'质量问题'。"))
def check_return_eligibility(
        ctx: Context,
        order_id: Annotated[str, Field(description="订单号")],
        reason_type: ReasonType,
        product_id: Annotated[str | None, Field(description="商品编号，订单只有一件商品时可省略")] = None) -> dict:
    return _run(aftersale.check_eligibility, _user(ctx), order_id, reason_type, product_id)


@app.tool(description=(
    "为用户发起售后申请（退货退款/换货/仅退款/维修）。调用后不会立即提交，而是生成一张待确认的申请，"
    "由用户在界面上点击确认后才会真正提交。"
    "只在 check_return_eligibility 显示符合条件、且用户明确表示要办理时调用；"
    "request_type 必须是 check_return_eligibility 返回的 allowed_types 之一。"))
def create_aftersale_request(
        ctx: Context,
        order_id: Annotated[str, Field(description="订单号")],
        request_type: RequestType,
        reason_type: ReasonType,
        reason: Annotated[str, Field(description="用户描述的具体原因，如'尺码不合适''杯盖有裂缝'")],
        product_id: Annotated[str | None, Field(description="商品编号，订单只有一件商品时可省略")] = None) -> dict:
    return _run(aftersale.propose, _user(ctx), order_id, request_type, reason_type, reason, product_id)


# ---------- 只给宿主程序用的工具 ----------

@app.tool(meta=HOST_ONLY, description="宿主程序专用：获取当前登录用户的基本信息。")
def get_my_profile(ctx: Context) -> dict:
    return _run(orders.profile, _user(ctx))


@app.tool(meta=HOST_ONLY, description="宿主程序专用：用户点击「确认提交」后执行售后申请。必须附带确认签名。")
def confirm_aftersale_request(ctx: Context, action_id: str, signature: str) -> dict:
    user_id = _user(ctx)
    _run(auth.verify_confirm_signature, user_id, action_id, "confirm", signature)
    return _run(aftersale.commit, user_id, action_id, "用户点击确认（签名已校验）")


@app.tool(meta=HOST_ONLY, description="宿主程序专用：用户点击「取消」后作废待确认的申请。必须附带确认签名。")
def cancel_aftersale_request(ctx: Context, action_id: str, signature: str) -> dict:
    user_id = _user(ctx)
    _run(auth.verify_confirm_signature, user_id, action_id, "cancel", signature)
    return _run(aftersale.cancel, user_id, action_id, "用户点击取消（签名已校验）")


# ---------- 资源与健康检查 ----------

@app.resource("kb://clauses", name="clauses", title="售后政策条款", mime_type="application/json",
              description="知识库全部条款（编号、标题、所属文档、正文），供 Agent 核对引用是否真实")
def clauses() -> str:
    return json.dumps([c.__dict__ for c in load_chunks()], ensure_ascii=False)


@app.custom_route("/health", methods=["GET"])
async def health(_: Request) -> JSONResponse:
    return JSONResponse({"ok": True, "service": "linjian-mall"})


def main():
    parser = argparse.ArgumentParser(description="林间优选商城 MCP 服务")
    parser.add_argument("--stdio", action="store_true", help="以 stdio 方式运行（Claude Desktop 等客户端）")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    if args.stdio:
        app.run("stdio")
    else:
        print(f"商城 MCP 服务：http://127.0.0.1:{args.port}/mcp")
        app.run("streamable-http", port=args.port)


if __name__ == "__main__":
    main()
