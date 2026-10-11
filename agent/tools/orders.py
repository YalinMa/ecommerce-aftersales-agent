"""订单相关的只读工具。

设计原则：
- 所有查询都按 ctx.user_id 过滤，订单不属于当前用户时与"不存在"返回同样的结果，不泄露订单是否存在。
- 日期计算由程序完成（签收了几天、物流多久没更新），不让模型自己算。大模型做日期加减很容易出错。
- 只返回事实数据，不在工具里替模型下"能不能退"的结论；政策判断由模型结合知识库检索结果完成。
  （V3 会新增带业务规则的资格校验工具，届时可以对比两种做法。）
"""

from datetime import timedelta

from agent import db
from agent.tools.base import Tool, ToolContext, ToolError

TAG_LABELS = {"intimate": "贴身衣物", "fresh": "生鲜", "digital": "数码家电", "custom": "定制", "virtual": "虚拟商品"}


def _days_since(t) -> int | None:
    return (db.now() - t).days if t else None


def _workdays_since(t) -> int | None:
    """从 t 的次日起到今天经过的工作日数（不考虑法定节假日）。"""
    if not t:
        return None
    day, today, n = t.date() + timedelta(days=1), db.now().date(), 0
    while day <= today:
        n += day.weekday() < 5
        day += timedelta(days=1)
    return n


def _owned_order(conn, ctx: ToolContext, order_id: str):
    row = conn.execute("SELECT * FROM orders WHERE id = ? AND user_id = ?",
                       (order_id.strip().upper(), ctx.user_id)).fetchone()
    if not row:
        raise ToolError(f"当前账号下没有找到订单 {order_id}，请让用户核对订单号")
    return row


def _items(conn, order_id: str) -> list[dict]:
    rows = conn.execute(
        """SELECT p.id, p.name, p.category, p.tags, i.quantity, i.unit_price
           FROM order_items i JOIN products p ON p.id = i.product_id WHERE i.order_id = ?""",
        (order_id,),
    ).fetchall()
    return [{
        "product_id": r["id"],
        "name": r["name"],
        "category": r["category"],
        "special_tags": [TAG_LABELS[t] for t in r["tags"].split(",") if t],
        "quantity": r["quantity"],
        "unit_price": r["unit_price"],
    } for r in rows]


# ---------- list_my_orders ----------

def _keyword_match(keyword: str, name: str) -> bool:
    """宽松匹配：关键词的每个字按顺序出现在商品名里即可。

    用户和模型说的商品名往往和商品标题不完全一致，比如"降噪耳机"对"降噪蓝牙耳机 Pro"。
    用 SQL 的 LIKE '%降噪耳机%' 会查不到，模型就会告诉用户"没有这个订单"。
    """
    kw, name = keyword.replace(" ", "").lower(), name.replace(" ", "").lower()
    if kw in name:
        return True
    it = iter(name)
    return all(ch in it for ch in kw)


def list_my_orders(ctx: ToolContext, status: str | None = None, keyword: str | None = None,
                   limit: int = 10) -> dict:
    limit = max(1, min(limit, 20))
    with db.connect() as conn:
        sql, params = "SELECT * FROM orders WHERE user_id = ?", [ctx.user_id]
        if status:
            sql += " AND status = ?"
            params.append(status)
        orders = []
        for r in conn.execute(sql + " ORDER BY paid_at DESC", params):
            items = _items(conn, r["id"])
            if keyword and not any(_keyword_match(keyword, i["name"]) for i in items):
                continue
            orders.append({
                "order_id": r["id"],
                "status": r["status"],
                "items": "、".join(f"{i['name']}×{i['quantity']}" for i in items),
                "total_amount": r["total_amount"],
                "paid_at": r["paid_at"],
                "signed_at": r["signed_at"],
                "days_since_signed": _days_since(db.parse(r["signed_at"])),
            })
            if len(orders) == limit:
                break
    result = {"count": len(orders), "orders": orders}
    if not orders and (keyword or status):
        result["note"] = "没有匹配的订单。可以换更短或更通用的关键词（如'耳机'）、去掉筛选条件再查一次，确认没有后再告诉用户。"
    return result


LIST_MY_ORDERS = Tool(
    name="list_my_orders",
    description=(
        "列出当前登录用户的订单，按付款时间倒序。"
        "当用户没有提供订单号、只描述了商品（如'上周买的耳机'）或想看最近的订单时调用；"
        "可以用 keyword 按商品名筛选。找到唯一匹配的订单后，再用 get_order_detail 查看详情。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["待发货", "已发货", "已签收", "已完成", "已取消", "售后中"],
                       "description": "按订单状态筛选，不传则返回全部"},
            "keyword": {"type": "string", "description": "商品名关键词，如'耳机''保温杯'"},
            "limit": {"type": "integer", "description": "返回条数，默认 10，最多 20"},
        },
        "required": [],
    },
    handler=list_my_orders,
)


# ---------- get_order_detail ----------

def get_order_detail(ctx: ToolContext, order_id: str) -> dict:
    with db.connect() as conn:
        o = _owned_order(conn, ctx, order_id)
        signed = db.parse(o["signed_at"])
        return {
            "order_id": o["id"],
            "status": o["status"],
            "items": _items(conn, o["id"]),
            "total_amount": o["total_amount"],
            "pay_method": o["pay_method"],
            "freight_insurance": bool(o["freight_insurance"]),
            "paid_at": o["paid_at"],
            "shipped_at": o["shipped_at"],
            "signed_at": o["signed_at"],
            "days_since_paid": _days_since(db.parse(o["paid_at"])),
            "days_since_signed": _days_since(signed),
            "has_aftersale": conn.execute(
                "SELECT 1 FROM aftersales WHERE order_id = ?", (o["id"],)).fetchone() is not None,
        }


GET_ORDER_DETAIL = Tool(
    name="get_order_detail",
    description=(
        "查询当前用户某个订单的详情：商品（含是否贴身衣物、生鲜、数码等特殊标签）、金额、支付方式、"
        "是否有运费险、付款/发货/签收时间，以及已签收天数。"
        "判断能否退换货、是否在时效内之前，必须先调用它获取真实的签收时间和商品类别，不要让用户自己报。"
    ),
    parameters={
        "type": "object",
        "properties": {"order_id": {"type": "string", "description": "订单号，如 LJ10010002"}},
        "required": ["order_id"],
    },
    handler=get_order_detail,
)


# ---------- get_logistics ----------

def get_logistics(ctx: ToolContext, order_id: str) -> dict:
    with db.connect() as conn:
        o = _owned_order(conn, ctx, order_id)
        ship = conn.execute("SELECT * FROM shipments WHERE order_id = ?", (o["id"],)).fetchone()
        if not ship:
            return {"order_id": o["id"], "status": o["status"], "shipped": False,
                    "note": "订单尚未发货，暂无物流信息"}
        events = conn.execute(
            "SELECT time, location, description FROM shipment_events WHERE order_id = ? ORDER BY time",
            (o["id"],),
        ).fetchall()
        last = db.parse(events[-1]["time"]) if events else None
        return {
            "order_id": o["id"],
            "status": o["status"],
            "carrier": ship["carrier"],
            "tracking_no": ship["tracking_no"],
            "events": [dict(e) for e in events],
            "hours_since_last_update": round((db.now() - last).total_seconds() / 3600, 1) if last else None,
        }


GET_LOGISTICS = Tool(
    name="get_logistics",
    description=(
        "查询当前用户某个订单的物流轨迹，返回快递公司、单号、全部轨迹节点，以及距最近一次更新过去了多少小时。"
        "用户问'到哪了''什么时候到''物流不动了'时调用。"
    ),
    parameters={
        "type": "object",
        "properties": {"order_id": {"type": "string", "description": "订单号"}},
        "required": ["order_id"],
    },
    handler=get_logistics,
)


# ---------- get_aftersale_status ----------

def get_aftersale_status(ctx: ToolContext, order_id: str) -> dict:
    with db.connect() as conn:
        o = _owned_order(conn, ctx, order_id)
        rows = conn.execute(
            """SELECT a.*, p.name AS product_name FROM aftersales a
               JOIN products p ON p.id = a.product_id WHERE a.order_id = ? ORDER BY a.created_at""",
            (o["id"],),
        ).fetchall()
        records = [{
            "aftersale_id": r["id"],
            "product": r["product_name"],
            "type": r["type"],
            "reason": r["reason"],
            "status": r["status"],
            "created_at": r["created_at"],
            "warehouse_signed_at": r["warehouse_signed_at"],
            "workdays_since_warehouse_signed": _workdays_since(db.parse(r["warehouse_signed_at"])),
            "refund_amount": r["refund_amount"],
            "refunded_at": r["refunded_at"],
        } for r in rows]
    return {"order_id": o["id"], "pay_method": o["pay_method"], "records": records}


GET_AFTERSALE_STATUS = Tool(
    name="get_aftersale_status",
    description=(
        "查询当前用户某个订单的售后记录：类型、原因、进度状态、仓库签收时间（及已过去的工作日数）、退款金额和时间。"
        "用户问退货退款进度、钱什么时候到账时调用。"
    ),
    parameters={
        "type": "object",
        "properties": {"order_id": {"type": "string", "description": "订单号"}},
        "required": ["order_id"],
    },
    handler=get_aftersale_status,
)
