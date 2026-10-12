"""售后申请：资格判断、生成待确认申请、确认后执行。

V3 时待确认的申请存在客服 Agent 的内存里；V4 起由商城自己保存在 pending_actions 表中。
写操作的规则归业务方所有：哪些申请在等确认、谁能确认、确认后怎么执行，都由商城决定，
客服 Agent 只负责把用户的确认动作（带签名）转交过来。

    check_eligibility()   只读。按 rules.py 判断能否申请、截止日期、运费谁出
    propose()             生成一张待确认的申请，不改动业务数据，15 分钟内有效
    commit()              用户确认后执行：再校验 → 防重复 → 写售后单 → 改订单状态 → 记审计日志
    cancel()              用户取消
"""

import json
import uuid
from datetime import timedelta

from mall import BusinessError, db, rules
from mall.orders import owned_order

PENDING_TTL = timedelta(minutes=15)   # 待确认申请的有效期

NEXT_STEPS = {
    "仅退款": "无需寄回商品，退款将原路退回",
    "退货退款": "审核通过后，在订单详情选择预约上门取件或自行寄回，并在 7 天内寄出",
    "换货": "审核通过后寄回原商品，仓库收到后寄出新规格商品",
    "维修": "审核通过后预约上门取件寄修，维修一般 7–15 个工作日",
}


def _load(conn, user_id: str, order_id: str, product_id: str | None):
    """读取订单和商品，并确认归属。product_id 可省略（订单只有一件商品时）。"""
    order = owned_order(conn, user_id, order_id)
    rows = conn.execute(
        """SELECT p.id, p.name, p.tags, i.quantity, i.unit_price FROM order_items i
           JOIN products p ON p.id = i.product_id WHERE i.order_id = ?""", (order["id"],)).fetchall()
    if product_id:
        item = next((i for i in rows if i["id"] == product_id.strip().upper()), None)
        if not item:
            raise BusinessError(f"订单 {order['id']} 里没有商品 {product_id}，可选：" +
                                "、".join(f"{i['id']} {i['name']}" for i in rows))
    elif len(rows) == 1:
        item = rows[0]
    else:
        raise BusinessError("该订单有多件商品，请指定 product_id：" +
                            "、".join(f"{i['id']} {i['name']}" for i in rows))
    return order, item


def _eligibility(conn, order, item, reason_type: str) -> rules.Eligibility:
    open_case = conn.execute(
        "SELECT 1 FROM aftersales WHERE order_id = ? AND product_id = ? AND status NOT IN ('已关闭', '已退款')",
        (order["id"], item["id"])).fetchone()
    return rules.check(order["status"], db.parse(order["signed_at"]),
                       [t for t in item["tags"].split(",") if t], reason_type, open_case is not None)


def _audit(conn, user_id: str, action: str, params: dict, result: dict, confirmed_by: str):
    """审计日志：谁、什么时候、做了什么、结果如何。写操作必须可追溯。"""
    conn.execute("INSERT INTO audit_log (time, user_id, action, params, result, confirmed_by) VALUES (?,?,?,?,?,?)",
                 (db.now().strftime(db.TIME_FMT), user_id, action,
                  json.dumps(params, ensure_ascii=False), json.dumps(result, ensure_ascii=False), confirmed_by))


# ---------- 资格判断 ----------

def check_eligibility(user_id: str, order_id: str, reason_type: str, product_id: str | None = None) -> dict:
    with db.connect() as conn:
        order, item = _load(conn, user_id, order_id, product_id)
        result = _eligibility(conn, order, item, reason_type).to_dict()
    return {"order_id": order["id"], "product": f"{item['id']} {item['name']}", "reason_type": reason_type,
            "freight_insurance": bool(order["freight_insurance"]), **result}


# ---------- 生成待确认申请 ----------

def propose(user_id: str, order_id: str, request_type: str, reason_type: str, reason: str,
            product_id: str | None = None) -> dict:
    with db.connect() as conn:
        order, item = _load(conn, user_id, order_id, product_id)
        # 不信任调用方的判断：生成申请前由商城重新校验一遍
        e = _eligibility(conn, order, item, reason_type)
        if not e.eligible:
            raise BusinessError("不符合申请条件：" + "；".join(e.reasons))
        if request_type not in e.allowed_types:
            raise BusinessError(f"该情况不能申请{request_type}，可申请：{'、'.join(e.allowed_types)}")

        now = db.now()
        params = {"order_id": order["id"], "product_id": item["id"], "request_type": request_type,
                  "reason_type": reason_type, "reason": reason[:200],
                  "refund_amount": item["unit_price"] * item["quantity"]}
        summary = {
            "订单": order["id"], "商品": item["name"], "申请类型": request_type, "原因": f"{reason_type}：{reason[:60]}",
            "退回运费": e.freight_paid_by or "无", "申请截止": e.deadline or "—",
            "请确认": "；".join(e.conditions) or "无",
        }
        # 同一件商品只保留一张待确认的申请，重复生成时旧的作废
        conn.execute("""UPDATE pending_actions SET status = '已作废' WHERE user_id = ? AND status = '待确认'
                        AND json_extract(params, '$.order_id') = ? AND json_extract(params, '$.product_id') = ?""",
                     (user_id, order["id"], item["id"]))
        action_id = uuid.uuid4().hex[:12]
        conn.execute("INSERT INTO pending_actions VALUES (?,?,?,?,?,?,?,?)",
                     (action_id, user_id, "create_aftersale", json.dumps(params, ensure_ascii=False),
                      json.dumps(summary, ensure_ascii=False), "待确认",
                      now.strftime(db.TIME_FMT), (now + PENDING_TTL).strftime(db.TIME_FMT)))
        conn.commit()
    return {
        "status": "待用户确认",
        "action_id": action_id,
        "summary": summary,
        "expires_in_minutes": int(PENDING_TTL.total_seconds() // 60),
        "instruction": "申请尚未提交。请简要告诉用户申请内容和需要确认的前提，请用户点击界面上的「确认提交」按钮。"
                       "在收到系统通知确认结果之前，不要说申请已提交。",
    }


def _pending(conn, user_id: str, action_id: str):
    row = conn.execute("SELECT * FROM pending_actions WHERE id = ? AND user_id = ?", (action_id, user_id)).fetchone()
    if not row:
        raise BusinessError("没有找到这张待确认的申请")
    return row


# ---------- 用户确认后执行 ----------

def commit(user_id: str, action_id: str, confirmed_by: str) -> dict:
    """真正写入数据库。只应在调用方证明"用户确实点击了确认"之后调用（见 mcp_server 的签名校验）。"""
    with db.connect() as conn:
        pa = _pending(conn, user_id, action_id)
        # 幂等：同一个确认操作重复提交（比如用户连点两次），返回已有的售后单
        existing = conn.execute("SELECT id, status FROM aftersales WHERE request_key = ?", (action_id,)).fetchone()
        if existing:
            return {"success": True, "aftersale_id": existing["id"], "status": existing["status"], "duplicate": True}
        if pa["status"] != "待确认":
            return {"success": False, "reasons": [f"这张申请{pa['status']}，不能再提交"]}
        if db.parse(pa["expires_at"]) < db.now():
            conn.execute("UPDATE pending_actions SET status = '已过期' WHERE id = ?", (action_id,))
            conn.commit()
            return {"success": False, "reasons": ["这张申请已超过 15 分钟未确认，已失效，请重新发起"]}

        p = json.loads(pa["params"])
        order, item = _load(conn, user_id, p["order_id"], p["product_id"])
        # 用户确认之前，情况可能已经变化（比如在 App 里自己提交了），所以再校验一次
        e = _eligibility(conn, order, item, p["reason_type"])
        if not e.eligible or p["request_type"] not in e.allowed_types:
            _audit(conn, user_id, "create_aftersale_rejected", p, {"reasons": e.reasons}, confirmed_by)
            conn.execute("UPDATE pending_actions SET status = '已失效' WHERE id = ?", (action_id,))
            conn.commit()
            return {"success": False, "reasons": e.reasons}

        now = db.now().strftime(db.TIME_FMT)
        aid = "AS" + db.now().strftime("%m%d%H%M%S") + action_id[:2].upper()
        status = "退款中" if order["status"] == "待发货" else ("待寄回" if p["reason_type"] == "无理由" else "待审核")
        conn.execute(
            """INSERT INTO aftersales (id, order_id, product_id, type, reason, status, created_at,
                                       refund_amount, request_key) VALUES (?,?,?,?,?,?,?,?,?)""",
            (aid, order["id"], item["id"], p["request_type"], f"{p['reason_type']}：{p['reason']}", status, now,
             p["refund_amount"] if p["request_type"] in ("退货退款", "仅退款") else None, action_id))
        conn.execute("UPDATE orders SET status = '售后中' WHERE id = ?", (order["id"],))
        conn.execute("UPDATE pending_actions SET status = '已提交' WHERE id = ?", (action_id,))
        result = {"success": True, "aftersale_id": aid, "status": status,
                  "next_step": NEXT_STEPS[p["request_type"]] +
                  ("；请在 App 售后详情上传照片或视频凭证" if p["reason_type"] == "质量问题" else "")}
        _audit(conn, user_id, "create_aftersale", p, result, confirmed_by)
        conn.commit()
    return result


def cancel(user_id: str, action_id: str, confirmed_by: str) -> dict:
    with db.connect() as conn:
        pa = _pending(conn, user_id, action_id)
        if pa["status"] != "待确认":
            return {"cancelled": False, "reason": f"这张申请{pa['status']}"}
        conn.execute("UPDATE pending_actions SET status = '已取消' WHERE id = ?", (action_id,))
        _audit(conn, user_id, "create_aftersale_cancelled", json.loads(pa["params"]), {}, confirmed_by)
        conn.commit()
    return {"cancelled": True}
