"""模拟商城数据库。V1 直接读 SQLite；V4 会把这一层拆成独立的 MCP Server。"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Iterator

from agent import config

DB_PATH = config.ROOT / "data" / "mall.db"
TIME_FMT = "%Y-%m-%d %H:%M:%S"


def ensure_exists():
    if not DB_PATH.exists():
        raise SystemExit("未找到模拟数据，请先运行：python scripts/seed_data.py")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    ensure_exists()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def parse(s: str | None) -> datetime | None:
    return datetime.strptime(s, TIME_FMT) if s else None


def now() -> datetime:
    return datetime.now().replace(microsecond=0)
