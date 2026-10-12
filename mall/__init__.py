"""林间优选商城：业务方代码（V4 起独立于客服 Agent）。

这个包代表"商城的业务系统"：订单、物流、售后、售后政策知识库、资格规则。
它通过 MCP 服务（mall/mcp_server.py）对外提供能力，不依赖 agent/ 里的任何代码。
客服 Agent 只能通过 MCP 协议调用它，不能直接读写数据库。
"""

from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


class BusinessError(Exception):
    """业务层面的错误（订单不存在、不符合条件……），会作为工具错误返回给调用方。"""
