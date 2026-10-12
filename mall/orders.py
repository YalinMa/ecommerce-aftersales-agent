"""订单、物流、售后记录查询（只读）。

设计原则：
- 每个函数都显式接收 user_id，并按它过滤。user_id 由 MCP 服务从已验证的身份令牌里解出，
  调用方（包括大模型）没有办法指定别人的 user_id。
- 订单不属于当前用户时，与"不存在"返回同样的错误，不泄露订单是否存在。
- 日期计算在这里完成（签收了几天、物流多久没更新），不让模型自己算。
- 只返回事实，不替调用方下"能不能退"的结论；资格判断在 aftersale.py 的规则引擎里。
"""

from datetime import timedelta

from mall import BusinessError, db

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


def owned_order(conn, user_id: str, order_id: str):
    row = conn.execute("SELECT * FROM orders WHERE id = ? AND user_id = ?",
                       (order_id.strip().upper(), user_id)).fetchone()
    if not row:
        raise BusinessError(f"当前账号下没有找到订单 {order_id}，请让用户核对订单号")
    return row


def items(conn, order_id: str) -> list[dict]:
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


def profile(user_id: str) -> dict:
    with db.connect() as conn:
        row = conn.execute("SELECT id, nickname FROM users WHERE id = ?", (user_id,)).fetchone()
    if not row:
        raise BusinessError(f"用户 {user_id} 不存在")
    return dict(row)


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


def list_orders(user_id: str, status: str | None = None, keyword: str | None = None, limit: int = 10) -> dict:
    limit = max(1, min(limit, 20))
    with db.connect() as conn:
        sql, params = "SELECT * FROM orders WHERE user_id = ?", [user_id]
        if status:
            sql += " AND status = ?"
            params.append(status)
        orders = []
        for r in conn.execute(sql + " ORDER BY paid_at DESC", params):
            its = items(conn, r["id"])
            if keyword and not any(_keyword_match(keyword, i["name"]) for i in its):
                continue
            orders.append({
                "order_id": r["id"],
                "status": r["status"],
                "items": "、".join(f"{i['name']}×{i['quantity']}" for i in its),
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


def order_detail(user_id: str, order_id: str) -> dict:
    with db.connect() as conn:
        o = owned_order(conn, user_id, order_id)
        return {
            "order_id": o["id"],
            "status": o["status"],
            "items": items(conn, o["id"]),
            "total_amount": o["total_amount"],
            "pay_method": o["pay_method"],
            "freight_insurance": bool(o["freight_insurance"]),
            "paid_at": o["paid_at"],
            "shipped_at": o["shipped_at"],
            "signed_at": o["signed_at"],
            "days_since_paid": _days_since(db.parse(o["paid_at"])),
            "days_since_signed": _days_since(db.parse(o["signed_at"])),
            "has_aftersale": conn.execute(
                "SELECT 1 FROM aftersales WHERE order_id = ?", (o["id"],)).fetchone() is not None,
        }


def logistics(user_id: str, order_id: str) -> dict:
    with db.connect() as conn:
        o = owned_order(conn, user_id, order_id)
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


def aftersale_status(user_id: str, order_id: str) -> dict:
    with db.connect() as conn:
        o = owned_order(conn, user_id, order_id)
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
