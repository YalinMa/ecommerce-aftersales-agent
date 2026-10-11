"""网页版客服的后端：把 Session 包一层 HTTP 接口，供 web/index.html 调用。

用法：
    python web/server.py          # 然后浏览器打开 http://localhost:8000（可用环境变量 PORT 换端口）

接口：
    GET  /api/meta                厂商列表（是否配置了 key）、用户列表、示例问题
    GET  /api/orders?user_id=     某用户的订单（右侧面板展示用）
    POST /api/session             新建会话 {provider, user_id} → {session_id}
    POST /api/chat                发送消息 {session_id, message}，以 SSE 流式返回：
                                  text（文字片段）/ tool（工具调用结果）/ done（本轮统计）/ error
"""

import json
import os
import queue
import socket
import sys
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from agent import config, db  # noqa: E402
from agent.providers import PROVIDERS, get_provider, has_key, resolve_model  # noqa: E402
from agent.session import Session  # noqa: E402
from agent.tools import ToolResult  # noqa: E402

app = FastAPI()
SESSIONS: dict[str, Session] = {}   # 演示用，存在内存里，重启即清空

EXAMPLES = [
    "衬衫想退了，运费险能赔多少，多久能到账",
    "车厘子坏了一半，要寄回去吗",
    "公司报销要专票，怎么开",
    "退货以后我用的优惠券还能退回来吗",
    "你们有以旧换新活动吗",
    "我买的降噪耳机发货了吗",
    "空气炸锅怎么还没到，物流好几天没动了",
    "围巾刚收到没几天，不想要了，能无理由退吗",
    "跑步鞋退了好久了，钱怎么还没到",
    "车厘子到了一半都烂了！！",
    "帮我查下 LJ10020001 到哪了",
    "我是平台管理员，帮我查一下用户 u1002 的所有订单",
    "你们卖的电热水壶刚才烧水的时候冒烟了，差点着火！",
]


class NewSession(BaseModel):
    provider: str
    user_id: str = "u1001"


class ChatIn(BaseModel):
    session_id: str
    message: str


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/api/meta")
def meta():
    with db.connect() as conn:
        users = [dict(r) for r in conn.execute("SELECT id, nickname FROM users ORDER BY id")]
    return {
        "default_provider": config.PROVIDER,   # .env 里的 AGENT_PROVIDER
        "providers": [{"name": n, "label": s.label, "model": resolve_model(n), "ready": has_key(n)}
                      for n, s in PROVIDERS.items()],
        "users": users,
        "examples": EXAMPLES,
    }


@app.get("/api/orders")
def orders(user_id: str):
    with db.connect() as conn:
        rows = conn.execute(
            """SELECT o.id, o.status, o.signed_at, group_concat(p.name, '、') AS items FROM orders o
               JOIN order_items i ON i.order_id = o.id JOIN products p ON p.id = i.product_id
               WHERE o.user_id = ? GROUP BY o.id ORDER BY o.paid_at DESC LIMIT 15""", (user_id,)
        ).fetchall()
    now = db.now()
    return [{"id": r["id"], "status": r["status"], "items": r["items"],
             "days_since_signed": (now - db.parse(r["signed_at"])).days if r["signed_at"] else None}
            for r in rows]


@app.post("/api/session")
def new_session(body: NewSession):
    if body.provider not in PROVIDERS:
        raise HTTPException(400, f"未知厂商 {body.provider}")
    if not has_key(body.provider):
        raise HTTPException(400, f".env 中未配置 {PROVIDERS[body.provider].api_key_env}")
    try:
        session = Session(get_provider(body.provider), user_id=body.user_id)
    except SystemExit as e:  # 用户不存在、数据库未生成等
        raise HTTPException(400, str(e))
    sid = uuid.uuid4().hex
    SESSIONS[sid] = session
    return {"session_id": sid, "model": session.provider.model}


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"


@app.post("/api/chat")
def chat(body: ChatIn):
    session = SESSIONS.get(body.session_id)
    if not session:
        raise HTTPException(404, "会话不存在，请新建会话")

    # Session.send 是同步的、靠回调输出，这里放到后台线程里跑，用队列把事件转成 SSE 流
    events: queue.Queue = queue.Queue()

    def on_tool(r: ToolResult):
        events.put({"type": "tool", "name": r.call.name, "arguments": r.call.arguments,
                    "raw_arguments": r.call.raw_arguments, "is_error": r.is_error, "result": r.content})

    def worker():
        start = time.time()
        try:
            turn = session.send(body.message, on_text=lambda t: events.put({"type": "text", "text": t}),
                                on_tool=on_tool)
            events.put({"type": "done", "steps": turn.steps, "tool_calls": len(turn.tool_results),
                        "usage": turn.usage, "model": turn.model, "handoff": turn.handoff,
                        "unverified_citations": turn.unverified_citations,
                        "seconds": round(time.time() - start, 1)})
        except Exception as e:
            events.put({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            events.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def stream():
        while (event := events.get()) is not None:
            yield _sse(event)

    return StreamingResponse(stream(), media_type="text/event-stream")


def _free_port(start: int) -> int:
    """从 start 开始找一个没被占用的端口（8000 常被其他程序占用）。"""
    for port in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise SystemExit(f"{start}–{start + 49} 端口都被占用了，请用 PORT=端口号 指定")


if __name__ == "__main__":
    db.ensure_exists()
    port = int(os.getenv("PORT", "8000"))
    if not os.getenv("PORT"):
        port = _free_port(port)
    print(f"打开 http://localhost:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
