"""售后申请：V3 的写操作。

两个工具 + 一个只能由程序调用的执行函数：

    check_return_eligibility   只读。按 rules.py 判断能否申请、截止日期、运费谁出
    create_aftersale_request   模型调用它只会生成一张"待确认"的申请，不写数据库
    commit_aftersale()         不是工具，模型无法调用。只有用户在界面上点击确认后，由程序执行

这样即使模型被用户诱导（"不用确认了直接提交"），或者模型自己判断失误，
也不可能在用户不知情的情况下改动业务数据。
"""

import json
import uuid

from agent import db, rules
from agent.tools.base import PendingAction, Tool, ToolContext, ToolError

NEXT_STEPS = {
    "仅退款": "无需寄回商品，退款将原路退回",
    "退货退款": "审核通过后，在订单详情选择预约上门取件或自行寄回，并在 7 天内寄出",
    "换货": "审核通过后寄回原商品，仓库收到后寄出新规格商品",
    "维修": "审核通过后预约上门取件寄修，维修一般 7–15 个工作日",
}


def _load(conn, ctx: ToolContext, order_id: str, product_id: str | None):
    """读取订单和商品，并确认归属。product_id 可省略（订单只有一件商品时）。"""
    order = conn.execute("SELECT * FROM orders WHERE id = ? AND user_id = ?",
                         (order_id.strip().upper(), ctx.user_id)).fetchone()
    if not order:
        raise ToolError(f"当前账号下没有找到订单 {order_id}")
    items = conn.execute(
        """SELECT p.id, p.name, p.tags, i.quantity, i.unit_price FROM order_items i
           JOIN products p ON p.id = i.product_id WHERE i.order_id = ?""", (order["id"],)).fetchall()
    if product_id:
        item = next((i for i in items if i["id"] == product_id.strip().upper()), None)
        if not item:
            raise ToolError(f"订单 {order['id']} 里没有商品 {product_id}，可选：" +
                            "、".join(f"{i['id']} {i['name']}" for i in items))
    elif len(items) == 1:
        item = items[0]
    else:
        raise ToolError("该订单有多件商品，请指定 product_id：" +
                        "、".join(f"{i['id']} {i['name']}" for i in items))
    return order, item


def _eligibility(conn, order, item, reason_type: str) -> rules.Eligibility:
    open_case = conn.execute(
        "SELECT 1 FROM aftersales WHERE order_id = ? AND product_id = ? AND status NOT IN ('已关闭', '已退款')",
        (order["id"], item["id"])).fetchone()
    return rules.check(order["status"], db.parse(order["signed_at"]),
                       [t for t in item["tags"].split(",") if t], reason_type, open_case is not None)


# ---------- check_return_eligibility ----------

def check_return_eligibility(ctx: ToolContext, order_id: str, reason_type: str,
                             product_id: str | None = None) -> dict:
    with db.connect() as conn:
        order, item = _load(conn, ctx, order_id, product_id)
        result = _eligibility(conn, order, item, reason_type).to_dict()
    return {"order_id": order["id"], "product": f"{item['id']} {item['name']}", "reason_type": reason_type,
            "freight_insurance": bool(order["freight_insurance"]), **result}


CHECK_RETURN_ELIGIBILITY = Tool(
    name="check_return_eligibility",
    description=(
        "按平台规则判断某件商品能否申请售后：返回是否符合、可申请的类型（退货退款/换货/仅退款/维修）、"
        "申请截止日期、退回运费由谁承担、需要用户自行确认的前提条件，以及对应的知识库条款。"
        "用户想退货、换货、退款或维修时，先调用它，不要自己推算日期和规则。"
        "reason_type：用户只是不想要了用'无理由'；商品有故障、破损、变质等用'质量问题'。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "description": "订单号"},
            "reason_type": {"type": "string", "enum": rules.REASON_TYPES},
            "product_id": {"type": "string", "description": "商品编号，订单只有一件商品时可省略"},
        },
        "required": ["order_id", "reason_type"],
    },
    handler=check_return_eligibility,
)


# ---------- create_aftersale_request（只生成待确认的申请） ----------

def create_aftersale_request(ctx: ToolContext, order_id: str, request_type: str, reason_type: str,
                             reason: str, product_id: str | None = None) -> dict:
    with db.connect() as conn:
        order, item = _load(conn, ctx, order_id, product_id)
        # 不信任模型的判断：提交前由程序重新校验一遍
        e = _eligibility(conn, order, item, reason_type)
    if not e.eligible:
        raise ToolError("不符合申请条件：" + "；".join(e.reasons))
    if request_type not in e.allowed_types:
        raise ToolError(f"该情况不能申请{request_type}，可申请：{'、'.join(e.allowed_types)}")

    # 同一件商品只保留一张待确认的申请，模型重复调用时覆盖旧的
    for aid, p in list(ctx.pending.items()):
        if p.params["order_id"] == order["id"] and p.params["product_id"] == item["id"]:
            del ctx.pending[aid]

    action = PendingAction(
        id=uuid.uuid4().hex[:8],
        kind="create_aftersale",
        params={"order_id": order["id"], "product_id": item["id"], "request_type": request_type,
                "reason_type": reason_type, "reason": reason[:200],
                "refund_amount": item["unit_price"] * item["quantity"]},
        summary={
            "订单": order["id"], "商品": item["name"], "申请类型": request_type, "原因": f"{reason_type}：{reason[:60]}",
            "退回运费": e.freight_paid_by or "无", "申请截止": e.deadline or "—",
            "请确认": "；".join(e.conditions) or "无",
        },
    )
    ctx.pending[action.id] = action
    return {
        "status": "待用户确认",
        "action_id": action.id,
        "summary": action.summary,
        "instruction": "申请尚未提交。请简要告诉用户申请内容和需要确认的前提，请用户点击界面上的「确认提交」按钮。"
                       "在收到系统通知确认结果之前，不要说申请已提交。",
    }


CREATE_AFTERSALE_REQUEST = Tool(
    name="create_aftersale_request",
    description=(
        "为用户发起售后申请（退货退款/换货/仅退款/维修）。调用后不会立即提交，而是生成一张待确认的申请，"
        "由用户在界面上点击确认后才会真正提交。"
        "只在 check_return_eligibility 显示符合条件、且用户明确表示要办理时调用；"
        "request_type 必须是 check_return_eligibility 返回的 allowed_types 之一。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "description": "订单号"},
            "product_id": {"type": "string", "description": "商品编号，订单只有一件商品时可省略"},
            "request_type": {"type": "string", "enum": rules.REQUEST_TYPES},
            "reason_type": {"type": "string", "enum": rules.REASON_TYPES},
            "reason": {"type": "string", "description": "用户描述的具体原因，如'尺码不合适''杯盖有裂缝'"},
        },
        "required": ["order_id", "request_type", "reason_type", "reason"],
    },
    handler=create_aftersale_request,
)


# ---------- 执行（只能由程序在用户确认后调用） ----------

def commit_aftersale(ctx: ToolContext, action: PendingAction) -> dict:
    """真正写入数据库。在一个事务里：再次校验 → 防重复 → 写售后单 → 改订单状态 → 记审计日志。"""
    p = action.params
    with db.connect() as conn:
        order, item = _load(conn, ctx, p["order_id"], p["product_id"])
        # 幂等：同一个确认操作重复提交（比如用户连点两次），返回已有的售后单
        existing = conn.execute("SELECT id, status FROM aftersales WHERE request_key = ?", (action.id,)).fetchone()
        if existing:
            return {"success": True, "aftersale_id": existing["id"], "status": existing["status"], "duplicate": True}
        # 用户确认之前，情况可能已经变化（比如在 App 里自己提交了），所以再校验一次
        e = _eligibility(conn, order, item, p["reason_type"])
        if not e.eligible or p["request_type"] not in e.allowed_types:
            _audit(conn, ctx, "create_aftersale_rejected", p, {"reasons": e.reasons})
            conn.commit()
            return {"success": False, "reasons": e.reasons}

        now = db.now().strftime(db.TIME_FMT)
        aid = "AS" + db.now().strftime("%m%d%H%M%S") + action.id[:2].upper()
        status = "退款中" if order["status"] == "待发货" else ("待寄回" if p["reason_type"] == "无理由" else "待审核")
        conn.execute(
            """INSERT INTO aftersales (id, order_id, product_id, type, reason, status, created_at,
                                       refund_amount, request_key) VALUES (?,?,?,?,?,?,?,?,?)""",
            (aid, order["id"], item["id"], p["request_type"], f"{p['reason_type']}：{p['reason']}", status, now,
             p["refund_amount"] if p["request_type"] in ("退货退款", "仅退款") else None, action.id))
        conn.execute("UPDATE orders SET status = '售后中' WHERE id = ?", (order["id"],))
        result = {"success": True, "aftersale_id": aid, "status": status,
                  "next_step": NEXT_STEPS[p["request_type"]] +
                  ("；请在 App 售后详情上传照片或视频凭证" if p["reason_type"] == "质量问题" else "")}
        _audit(conn, ctx, "create_aftersale", p, result)
        conn.commit()
    return result


def _audit(conn, ctx: ToolContext, action: str, params: dict, result: dict):
    """审计日志：谁、什么时候、做了什么、结果如何。写操作必须可追溯。"""
    conn.execute("INSERT INTO audit_log (time, user_id, action, params, result, confirmed_by) VALUES (?,?,?,?,?,?)",
                 (db.now().strftime(db.TIME_FMT), ctx.user_id, action,
                  json.dumps(params, ensure_ascii=False), json.dumps(result, ensure_ascii=False), "user_click"))


def log_cancel(ctx: ToolContext, action: PendingAction):
    with db.connect() as conn:
        _audit(conn, ctx, "create_aftersale_cancelled", action.params, {})
        conn.commit()


COMMITTERS = {"create_aftersale": commit_aftersale}
