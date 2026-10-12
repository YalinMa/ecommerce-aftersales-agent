"""身份令牌与确认签名。

两个问题：
1. MCP 服务怎么知道"是谁在查"？
   不能看工具参数里的 user_id，因为那是大模型生成的，可以被用户诱导篡改。
   也不能直接相信请求头里的 user_id，因为请求头的内容同样可以伪造。
   做法：用户登录商城时拿到一个签名令牌（这里模拟登录服务签发），客服 Agent 把它放在请求头里转交，
   MCP 服务用密钥验签后解出 user_id。没有密钥就造不出有效令牌。

2. MCP 服务怎么知道"用户真的点了确认"？
   提交售后申请的工具不在大模型的工具列表里，但这只是客服 Agent 一侧的约定。
   商城自己再加一道：确认必须附带签名，签名只有客服 Agent 的界面层在用户点击按钮后才会计算，
   大模型拿不到密钥，就算能调用确认接口也会被拒绝。

密钥从环境变量读取。演示环境有默认值，生产环境必须改成随机长字符串并妥善保管。
"""

import base64
import hashlib
import hmac
import json
import os
import time

from mall import BusinessError

_DEV_DEFAULT = "dev-only-change-me"
AUTH_SECRET = os.getenv("MALL_AUTH_SECRET", _DEV_DEFAULT).encode()
CONFIRM_SECRET = os.getenv("MALL_CONFIRM_SECRET", _DEV_DEFAULT + "-confirm").encode()
TOKEN_TTL_SECONDS = 12 * 3600


def _sign(secret: bytes, message: str) -> str:
    return hmac.new(secret, message.encode(), hashlib.sha256).hexdigest()


# ---------- 用户令牌（模拟商城登录服务） ----------

def issue_user_token(user_id: str, ttl: int = TOKEN_TTL_SECONDS) -> str:
    payload = base64.urlsafe_b64encode(json.dumps({"uid": user_id, "exp": int(time.time()) + ttl}).encode()).decode()
    return f"{payload}.{_sign(AUTH_SECRET, payload)}"


def verify_user_token(token: str) -> str:
    """验签并返回 user_id；令牌无效或过期时抛出 BusinessError。"""
    try:
        payload, sig = token.rsplit(".", 1)
    except ValueError:
        raise BusinessError("身份令牌格式错误") from None
    if not hmac.compare_digest(sig, _sign(AUTH_SECRET, payload)):
        raise BusinessError("身份令牌签名无效")
    data = json.loads(base64.urlsafe_b64decode(payload))
    if data["exp"] < time.time():
        raise BusinessError("身份令牌已过期，请重新登录")
    return data["uid"]


# ---------- 确认签名 ----------

def confirm_signature(user_id: str, action_id: str, decision: str) -> str:
    """用户点击「确认提交」或「取消」后，由客服 Agent 的界面层计算。decision 为 confirm / cancel。"""
    return _sign(CONFIRM_SECRET, f"{user_id}:{action_id}:{decision}")


def verify_confirm_signature(user_id: str, action_id: str, decision: str, signature: str):
    if not hmac.compare_digest(signature, confirm_signature(user_id, action_id, decision)):
        raise BusinessError("确认签名无效：这个操作没有经过用户在界面上确认")
