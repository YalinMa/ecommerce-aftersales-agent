"""生成模拟商城数据（SQLite）。

用法：
    python scripts/seed_data.py

数据分两部分：
  1. 手工设计的「场景订单」：每个订单对应一个测试场景（物流停滞、超过七天、退款审核中……），
     订单号固定，测试对话可以直接引用。
  2. 随机生成的「填充订单」：让数据看起来更真实，都是很久以前已完成的订单，不干扰测试场景。

所有时间都相对于运行脚本的时刻生成，所以测试场景里的"签收 3 天""物流 4 天没动"
永远成立。隔一段时间再跑测试前，先重新执行本脚本。
"""

import random
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.db import DB_PATH  # noqa: E402

SCHEMA = """
CREATE TABLE users (
    id TEXT PRIMARY KEY,
    nickname TEXT NOT NULL,
    phone_masked TEXT NOT NULL
);
CREATE TABLE products (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,          -- 服饰鞋包 / 数码家电 / 家居日用 / 食品生鲜
    price REAL NOT NULL,
    tags TEXT NOT NULL DEFAULT ''    -- 逗号分隔：intimate 贴身衣物, fresh 生鲜, digital 数码, custom 定制, virtual 虚拟
);
CREATE TABLE orders (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    status TEXT NOT NULL,            -- 待发货 / 已发货 / 已签收 / 已完成 / 已取消 / 售后中
    paid_at TEXT,
    shipped_at TEXT,
    signed_at TEXT,
    total_amount REAL NOT NULL,
    pay_method TEXT NOT NULL,        -- 余额 / 银行卡 / 信用卡
    freight_insurance INTEGER NOT NULL DEFAULT 0,
    address_masked TEXT NOT NULL
);
CREATE TABLE order_items (
    order_id TEXT NOT NULL REFERENCES orders(id),
    product_id TEXT NOT NULL REFERENCES products(id),
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL
);
CREATE TABLE shipments (
    order_id TEXT PRIMARY KEY REFERENCES orders(id),
    carrier TEXT NOT NULL,
    tracking_no TEXT NOT NULL
);
CREATE TABLE shipment_events (
    order_id TEXT NOT NULL REFERENCES orders(id),
    time TEXT NOT NULL,
    location TEXT NOT NULL,
    description TEXT NOT NULL
);
CREATE TABLE aftersales (
    id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL REFERENCES orders(id),
    product_id TEXT NOT NULL REFERENCES products(id),
    type TEXT NOT NULL,              -- 退货退款 / 换货 / 仅退款
    reason TEXT NOT NULL,
    status TEXT NOT NULL,            -- 待寄回 / 仓库已签收 / 审核中 / 退款中 / 已退款 / 已关闭
    created_at TEXT NOT NULL,
    warehouse_signed_at TEXT,
    refund_amount REAL,
    refunded_at TEXT
);
"""

USERS = [
    ("u1001", "小满", "138****2071"),
    ("u1002", "阿杰", "139****5582"),
    ("u1003", "Lily", "186****9034"),
]

PRODUCTS = [
    # id, 名称, 分类, 价格, 标签
    ("P001", "降噪蓝牙耳机 Pro", "数码家电", 399, "digital"),
    ("P002", "纯棉牛津纺衬衫", "服饰鞋包", 189, ""),
    ("P003", "316 不锈钢保温杯 500ml", "家居日用", 129, ""),
    ("P004", "纯棉圆领 T 恤", "服饰鞋包", 79, ""),
    ("P005", "4.5L 无油空气炸锅", "数码家电", 359, "digital"),
    ("P006", "羊毛针织围巾", "服饰鞋包", 259, ""),
    ("P007", "1.7L 电热水壶", "数码家电", 149, "digital"),
    ("P008", "轻量缓震跑步鞋", "服饰鞋包", 499, ""),
    ("P009", "智利车厘子 JJ 级 2 斤", "食品生鲜", 168, "fresh"),
    ("P010", "无钢圈舒适文胸", "服饰鞋包", 129, "intimate"),
    ("P011", "11 英寸平板电脑", "数码家电", 2599, "digital"),
    ("P012", "定制刻字钢笔", "家居日用", 299, "custom"),
    ("P013", "视频会员年卡", "家居日用", 198, "virtual"),
    ("P014", "便携蓝牙音箱", "数码家电", 229, "digital"),
    ("P015", "全棉四件套 1.8m", "家居日用", 399, ""),
    ("P016", "每日坚果 30 包", "食品生鲜", 99, ""),
    ("P017", "机械键盘 87 键", "数码家电", 449, "digital"),
    ("P018", "帆布双肩包", "服饰鞋包", 159, ""),
    ("P019", "陶瓷餐具 16 件套", "家居日用", 269, ""),
    ("P020", "挂耳咖啡 50 包", "食品生鲜", 89, ""),
]
PRICE = {p[0]: p[3] for p in PRODUCTS}

CARRIERS = {"顺丰速运": "SF", "中通快递": "ZT", "圆通速递": "YT", "京东物流": "JD"}
CITIES = ["上海", "杭州", "苏州", "南京", "广州", "深圳", "成都", "武汉"]


def ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


class Seeder:
    def __init__(self, conn: sqlite3.Connection, now: datetime):
        self.conn = conn
        self.now = now
        self.rng = random.Random(42)

    def ago(self, days: float = 0, hours: float = 0) -> datetime:
        return self.now - timedelta(days=days, hours=hours)

    def order(self, oid, user_id, items, status, paid, shipped=None, signed=None,
              pay_method="余额", insurance=False, address="上海市浦东新区****"):
        total = sum(PRICE[p] * q for p, q in items)
        self.conn.execute(
            "INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?,?)",
            (oid, user_id, status, ts(paid), shipped and ts(shipped), signed and ts(signed),
             total, pay_method, int(insurance), address),
        )
        for p, q in items:
            self.conn.execute("INSERT INTO order_items VALUES (?,?,?,?)", (oid, p, q, PRICE[p]))
        if shipped:
            self.shipment(oid, shipped, signed)

    def shipment(self, oid, shipped, signed, stall_after: datetime | None = None):
        carrier = self.rng.choice(list(CARRIERS))
        self.conn.execute("INSERT INTO shipments VALUES (?,?,?)",
                          (oid, carrier, f"{CARRIERS[carrier]}{self.rng.randint(10**11, 10**12 - 1)}"))
        origin, dest = self.rng.sample(CITIES, 2)
        end = signed or stall_after or self.now
        events = [(shipped, origin, "快件已揽收")]
        # 在揽收和签收（或停滞）之间均匀插入几个中转节点
        hops = [(f"{origin}转运中心", "已发出"), ("华东转运中心", "已到达"), (f"{dest}转运中心", "已到达")]
        for i, (loc, act) in enumerate(hops, 1):
            t = shipped + (end - shipped) * i / (len(hops) + 2)
            events.append((t, loc, f"快件{act}{loc}"))
        if signed:
            events.append((signed - timedelta(hours=3), dest, "快递员正在派送"))
            events.append((signed, dest, "已签收，签收人：本人"))
        for t, loc, desc in events:
            self.conn.execute("INSERT INTO shipment_events VALUES (?,?,?,?)", (oid, ts(t), loc, desc))

    def seed_scenarios(self):
        """手工设计的场景订单，订单号固定，供测试对话引用。"""
        u = "u1001"
        # 刚付款，待发货
        self.order("LJ10010001", u, [("P001", 1)], "待发货", self.ago(1, 2))
        # 签收 3 天，买了运费险 → 可以七天无理由
        self.order("LJ10010002", u, [("P002", 1)], "已签收",
                   self.ago(6), self.ago(5), self.ago(3), insurance=True)
        # 一单两件，签收 5 天，没有运费险
        self.order("LJ10010003", u, [("P003", 1), ("P004", 1)], "已签收",
                   self.ago(8), self.ago(7), self.ago(5), pay_method="银行卡")
        # 已发货，物流停在转运中心 4 天没更新
        self.conn.execute(
            "INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("LJ10010004", u, "已发货", ts(self.ago(7)), ts(self.ago(6)), None,
             PRICE["P005"], "余额", 0, "上海市浦东新区****"),
        )
        self.conn.execute("INSERT INTO order_items VALUES (?,?,?,?)", ("LJ10010004", "P005", 1, PRICE["P005"]))
        self.shipment("LJ10010004", self.ago(6), None, stall_after=self.ago(3.5))
        # 签收 12 天 → 超过七天无理由，但在 15 天质量问题期内
        self.order("LJ10010005", u, [("P006", 1)], "已签收", self.ago(15), self.ago(14), self.ago(12))
        # 电热水壶，签收 8 天（安全问题场景）
        self.order("LJ10010006", u, [("P007", 1)], "已签收", self.ago(11), self.ago(10), self.ago(8))
        # 退货进行中：仓库 2 天前签收，审核中，信用卡支付
        self.order("LJ10010007", u, [("P008", 1)], "售后中",
                   self.ago(20), self.ago(19), self.ago(15), pay_method="信用卡")
        self.conn.execute(
            "INSERT INTO aftersales VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("AS20010007", "LJ10010007", "P008", "退货退款", "尺码不合适", "审核中",
             ts(self.ago(10)), ts(self.ago(2)), PRICE["P008"], None),
        )
        # 车厘子，今天上午签收（生鲜）
        self.order("LJ10010008", u, [("P009", 1)], "已签收",
                   self.ago(2), self.ago(1, 6), self.ago(0, 3), pay_method="银行卡")
        # 文胸，签收 2 天（贴身衣物）
        self.order("LJ10010009", u, [("P010", 1)], "已签收", self.ago(5), self.ago(4), self.ago(2))
        # 蓝牙音箱，签收 6 天
        self.order("LJ10010010", u, [("P014", 1)], "已签收", self.ago(9), self.ago(8), self.ago(6))

        # 其他用户的订单：用来测试"不能查别人的订单"
        self.order("LJ10020001", "u1002", [("P011", 1)], "已发货",
                   self.ago(2), self.ago(1), address="北京市朝阳区****", pay_method="信用卡")
        self.order("LJ10030001", "u1003", [("P015", 1)], "已签收",
                   self.ago(6), self.ago(5), self.ago(3), address="广州市天河区****")

    def seed_filler(self):
        """随机生成的历史订单：40～300 天前，都已完成，不干扰测试场景。"""
        # 不使用场景订单里出现过的商品，避免"按商品名查订单"时出现意料之外的多个匹配
        scenario_pids = {r[0] for r in self.conn.execute(
            "SELECT DISTINCT product_id FROM order_items WHERE order_id LIKE 'LJ1001%'")}
        pids = [p[0] for p in PRODUCTS if "custom" not in p[4] and p[0] not in scenario_pids]
        counter = {"u1001": 100, "u1002": 100, "u1003": 100}
        for user_id, n in [("u1001", 6), ("u1002", 8), ("u1003", 8)]:
            for _ in range(n):
                counter[user_id] += 1
                oid = f"LJ{user_id[1:]}{counter[user_id]:04d}"
                paid = self.ago(self.rng.uniform(40, 300))
                shipped = paid + timedelta(hours=self.rng.uniform(4, 40))
                signed = shipped + timedelta(days=self.rng.uniform(1, 4))
                items = [(p, self.rng.choice([1, 1, 1, 2])) for p in self.rng.sample(pids, self.rng.choice([1, 1, 2]))]
                self.order(oid, user_id, items, "已完成", paid, shipped, signed,
                           pay_method=self.rng.choice(["余额", "银行卡", "信用卡"]),
                           insurance=self.rng.random() < 0.4)


def main():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    DB_PATH.unlink(missing_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    conn.executemany("INSERT INTO users VALUES (?,?,?)", USERS)
    conn.executemany("INSERT INTO products VALUES (?,?,?,?,?)", PRODUCTS)

    seeder = Seeder(conn, datetime.now().replace(microsecond=0))
    seeder.seed_scenarios()
    seeder.seed_filler()
    conn.commit()

    counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ["users", "products", "orders", "shipment_events", "aftersales"]}
    print(f"已生成 {DB_PATH.relative_to(DB_PATH.parent.parent)}：{counts}")


if __name__ == "__main__":
    main()
