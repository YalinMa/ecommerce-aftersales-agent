"""商城数据库（SQLite）。只有商城自己的代码能访问，客服 Agent 通过 MCP 服务间接使用。"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Iterator

from mall import ROOT

DB_PATH = ROOT / "data" / "mall.db"
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
