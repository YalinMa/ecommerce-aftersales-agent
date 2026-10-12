"""客服平台的数据：会话、消息、人工队列、用户反馈（V6）。

这是客服平台自己的数据库 data/service.db，和商城的 mall.db 分开：
商城管订单和售后，客服平台管"谁在什么时候聊了什么、有没有转人工、用户满不满意"。

会话状态：
    ai       AI 客服接待中
    queued   已转人工，排队等待坐席接入
    human    人工坐席接待中（AI 不再回复）
    closed   已结束
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Iterator

from agent import config

DB_PATH = config.ROOT / "data" / "service.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    nickname TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    channel TEXT NOT NULL,          -- shop（商城前台）/ debug（开发者调试页）
    status TEXT NOT NULL,           -- ai / queued / human / closed
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    handoff_reason TEXT,
    handoff_summary TEXT,
    handoff_by TEXT,                -- 模型 / 规则兜底
    handoff_at TEXT,
    agent_name TEXT                 -- 接入的人工坐席
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,             -- user / ai / human / system
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    meta TEXT                       -- AI 回复的耗时、工具调用等
);
CREATE TABLE IF NOT EXISTS feedback (
    message_id INTEGER PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    rating INTEGER NOT NULL,        -- 1 有帮助 / -1 没帮助
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id);
"""


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ---------- 会话 ----------

def create_conversation(cid: str, user_id: str, nickname: str, provider: str, model: str, channel: str):
    with connect() as c:
        c.execute("INSERT INTO conversations (id, user_id, nickname, provider, model, channel, status, created_at, "
                  "updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
                  (cid, user_id, nickname, provider, model, channel, "ai", _now(), _now()))


def get_conversation(cid: str) -> dict | None:
    with connect() as c:
        row = c.execute("SELECT * FROM conversations WHERE id = ?", (cid,)).fetchone()
    return dict(row) if row else None


def set_status(cid: str, status: str, **fields):
    sets = ", ".join(["status = ?", "updated_at = ?"] + [f"{k} = ?" for k in fields])
    with connect() as c:
        c.execute(f"UPDATE conversations SET {sets} WHERE id = ?", (status, _now(), *fields.values(), cid))


def mark_handoff(cid: str, handoff: dict):
    set_status(cid, "queued", handoff_reason=handoff.get("reason"), handoff_summary=handoff.get("summary"),
               handoff_by=handoff.get("by"), handoff_at=_now())


def list_conversations(statuses: list[str] | None = None, limit: int = 50) -> list[dict]:
    sql = "SELECT * FROM conversations"
    params: list = []
    if statuses:
        sql += f" WHERE status IN ({','.join('?' * len(statuses))})"
        params = statuses
    sql += " ORDER BY CASE status WHEN 'queued' THEN 0 WHEN 'human' THEN 1 WHEN 'ai' THEN 2 ELSE 3 END, updated_at DESC LIMIT ?"
    with connect() as c:
        rows = c.execute(sql, (*params, limit)).fetchall()
        result = []
        for r in rows:
            last = c.execute("SELECT content, role FROM messages WHERE conversation_id = ? AND role != 'system' "
                             "ORDER BY id DESC LIMIT 1", (r["id"],)).fetchone()
            result.append({**dict(r), "last_message": dict(last) if last else None})
    return result


def queue_position(cid: str) -> int:
    """在排队中的会话里排第几位（按转人工时间）。"""
    with connect() as c:
        row = c.execute("SELECT handoff_at FROM conversations WHERE id = ?", (cid,)).fetchone()
        if not row or not row["handoff_at"]:
            return 0
        return c.execute("SELECT COUNT(*) FROM conversations WHERE status = 'queued' AND handoff_at <= ?",
                         (row["handoff_at"],)).fetchone()[0]


# ---------- 消息 ----------

def add_message(cid: str, role: str, content: str, meta: dict | None = None) -> int:
    with connect() as c:
        cur = c.execute("INSERT INTO messages (conversation_id, role, content, created_at, meta) VALUES (?,?,?,?,?)",
                        (cid, role, content, _now(), json.dumps(meta, ensure_ascii=False) if meta else None))
        c.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (_now(), cid))
        return cur.lastrowid


def messages(cid: str, after: int = 0) -> list[dict]:
    with connect() as c:
        rows = c.execute(
            """SELECT m.*, f.rating FROM messages m LEFT JOIN feedback f ON f.message_id = m.id
               WHERE m.conversation_id = ? AND m.id > ? ORDER BY m.id""", (cid, after)).fetchall()
    return [{**dict(r), "meta": json.loads(r["meta"]) if r["meta"] else None} for r in rows]


# ---------- 反馈 ----------

def set_feedback(message_id: int, rating: int) -> bool:
    with connect() as c:
        msg = c.execute("SELECT conversation_id, role FROM messages WHERE id = ?", (message_id,)).fetchone()
        if not msg or msg["role"] != "ai":
            return False
        c.execute("INSERT OR REPLACE INTO feedback VALUES (?,?,?,?)", (message_id, msg["conversation_id"], rating, _now()))
    return True


# ---------- 运营数据 ----------

def stats(day: str | None = None) -> dict:
    """某一天的运营数据（默认今天）：会话量、转人工率、反馈、AI 回复的平均耗时。"""
    day = day or datetime.now().strftime("%Y-%m-%d")
    like = f"{day}%"
    with connect() as c:
        total = c.execute("SELECT COUNT(*) FROM conversations WHERE created_at LIKE ? AND channel = 'shop'", (like,)).fetchone()[0]
        handed = c.execute("SELECT COUNT(*) FROM conversations WHERE created_at LIKE ? AND channel = 'shop' "
                           "AND handoff_at IS NOT NULL", (like,)).fetchone()[0]
        by_rule = c.execute("SELECT COUNT(*) FROM conversations WHERE created_at LIKE ? AND channel = 'shop' "
                            "AND handoff_by = '规则兜底'", (like,)).fetchone()[0]
        queued = c.execute("SELECT COUNT(*) FROM conversations WHERE status = 'queued'").fetchone()[0]
        up = c.execute("SELECT COUNT(*) FROM feedback WHERE created_at LIKE ? AND rating = 1", (like,)).fetchone()[0]
        down = c.execute("SELECT COUNT(*) FROM feedback WHERE created_at LIKE ? AND rating = -1", (like,)).fetchone()[0]
        metas = [json.loads(r[0]) for r in c.execute(
            "SELECT m.meta FROM messages m JOIN conversations v ON v.id = m.conversation_id "
            "WHERE m.role = 'ai' AND m.meta IS NOT NULL AND m.created_at LIKE ? AND v.channel = 'shop'", (like,))]
    ft = [m["first_token_ms"] for m in metas if m.get("first_token_ms")]
    lat = [m["latency_ms"] for m in metas if m.get("latency_ms")]
    return {
        "day": day, "conversations": total, "handoffs": handed, "handoffs_by_rule": by_rule,
        "handoff_rate": round(handed / total, 3) if total else None,
        "queued_now": queued, "thumbs_up": up, "thumbs_down": down, "ai_replies": len(metas),
        "avg_first_token_ms": int(sum(ft) / len(ft)) if ft else None,
        "avg_latency_ms": int(sum(lat) / len(lat)) if lat else None,
    }
