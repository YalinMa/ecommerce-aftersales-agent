"""售后资格规则：用代码实现知识库里的硬性规定。

为什么不让模型自己判断"能不能退"？
- 查询类问题答错了，用户还能自己核对；写操作一旦执行就改变了业务数据，必须确定无误。
- 日期计算、多条规则组合（签收天数 × 商品类别 × 退货原因），恰好是大模型容易出错的地方。

所以分工是：模型负责理解用户想办什么；能不能办、截止哪天、运费谁出，由这里的代码决定。
每条判断都标注了对应的知识库条款，模型可以据此向用户解释。
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

NO_REASON_DAYS = 7      # R-01
QUALITY_DAYS = 15       # Q-01
WARRANTY_DAYS = 365     # Q-03

REASON_TYPES = ["无理由", "质量问题"]
REQUEST_TYPES = ["退货退款", "换货", "仅退款", "维修"]


@dataclass
class Eligibility:
    eligible: bool
    allowed_types: list[str] = field(default_factory=list)  # 可以申请的售后类型
    deadline: str | None = None                             # 申请截止日期（含当天）
    days_left: int | None = None
    freight_paid_by: str | None = None                      # 退回运费由谁承担
    conditions: list[str] = field(default_factory=list)     # 用户需要自行确认的前提
    reasons: list[str] = field(default_factory=list)        # 判断依据（尤其是不符合的原因）
    clauses: list[str] = field(default_factory=list)        # 对应的知识库条款

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v not in (None, [])} | {"eligible": self.eligible}


def _deadline(signed: datetime, days: int, today: date) -> tuple[str, int]:
    """"自签收次日起 N 天内"：签收日 + N 天当天 24:00 截止。"""
    end = signed.date() + timedelta(days=days)
    return end.isoformat(), (end - today).days


def check(order_status: str, signed_at: datetime | None, tags: list[str], reason_type: str,
          has_open_aftersale: bool, today: date | None = None) -> Eligibility:
    today = today or date.today()

    if has_open_aftersale:
        return Eligibility(False, reasons=["该商品已有进行中的售后单，不能重复申请，可查询现有售后进度"])

    # 未发货：只能仅退款，无需审核（M-03）
    if order_status == "待发货":
        return Eligibility(True, ["仅退款"], reasons=["订单未发货，可申请仅退款，24 小时内自动退款，运费一并退还"],
                           clauses=["M-03"])
    if order_status == "已取消":
        return Eligibility(False, reasons=["订单已取消"])
    # 已发货未签收：不能申请退货，可拒收（L-04）
    if order_status == "已发货" or signed_at is None:
        return Eligibility(False, reasons=["订单已发货但未签收，不能直接申请退货；如不需要可在派送时拒收"],
                           clauses=["M-03", "L-04"])

    if reason_type == "无理由":
        blocked = {"custom": "定制类商品", "fresh": "生鲜商品", "virtual": "虚拟商品"}
        for tag, label in blocked.items():
            if tag in tags:
                return Eligibility(False, reasons=[f"{label}不支持七天无理由退货；如有质量问题可按质量问题申请"],
                                   clauses=["R-02"])
        deadline, left = _deadline(signed_at, NO_REASON_DAYS, today)
        if left < 0:
            return Eligibility(False, deadline=deadline,
                               reasons=[f"七天无理由退货已于 {deadline} 截止；如有质量问题，签收 15 天内仍可申请"],
                               clauses=["R-01", "Q-01"])
        conditions = ["商品完好、吊牌未剪、不影响二次销售"]
        if "intimate" in tags:
            conditions.append("贴身衣物必须未拆封，拆封后不支持无理由退货")
        if "digital" in tags:
            conditions.append("数码产品必须未激活（未绑定账号）")
        return Eligibility(True, ["退货退款", "换货"], deadline, left, freight_paid_by="买家",
                           conditions=conditions, reasons=["在七天无理由退货期内"],
                           clauses=["R-01", "R-03", "R-04", "R-07"])

    if reason_type == "质量问题":
        if "fresh" in tags:
            # 生鲜质量问题：按坏果比例或整单退款，无需寄回（Q-06）
            deadline, left = _deadline(signed_at, QUALITY_DAYS, today)
            if left < 0:
                return Eligibility(False, deadline=deadline, reasons=["已超过质量问题申请期限"], clauses=["Q-01"])
            return Eligibility(True, ["仅退款"], deadline, left, freight_paid_by="无需寄回",
                               conditions=["需上传能看清坏果数量和外包装的照片"],
                               reasons=["生鲜质量问题可按坏果比例或整单退款，无需寄回；"
                                        "退款比例由审核根据照片确定，申请时不需要选择"], clauses=["Q-06"])
        deadline, left = _deadline(signed_at, QUALITY_DAYS, today)
        if left >= 0:
            return Eligibility(True, ["退货退款", "换货"], deadline, left, freight_paid_by="平台",
                               conditions=["需上传能体现问题的照片或视频", "人为损坏（摔、进水、私自拆机）不属于质量问题"],
                               reasons=["在 15 天质量问题退换期内"], clauses=["Q-01", "Q-02"])
        if "digital" in tags:
            w_deadline, w_left = _deadline(signed_at, WARRANTY_DAYS, today)
            if w_left >= 0:
                return Eligibility(True, ["维修"], w_deadline, w_left, freight_paid_by="平台",
                                   conditions=["需上传故障照片或视频"],
                                   reasons=["已超过 15 天退换期，数码家电在 1 年保修期内可免费维修"],
                                   clauses=["Q-03", "Q-04"])
        return Eligibility(False, deadline=deadline, reasons=["已超过质量问题退换期限"], clauses=["Q-01", "Q-03"])

    return Eligibility(False, reasons=[f"未知的申请原因 {reason_type}"])
