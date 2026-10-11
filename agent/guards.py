"""转人工的规则兜底。

V0–V2 三轮测试里，"用户投诉、要求赔偿时没有转人工"始终没能靠改 prompt 彻底解决：
模型倾向于"先问清楚再说"。对于必须转人工的高风险情况，不能只靠模型自觉。

做法分两层：
  1. 提示：命中规则时，在用户消息后附一句系统提示，引导模型安抚并调用 transfer_to_human。
  2. 兜底：本轮结束后模型仍然没有转接，程序直接转。

关键词规则的代价是会有误判（比如"我不是要投诉，就问问"）。这里只覆盖安全、投诉、索赔三类
误转代价小、漏转代价大的情况；"用户要求人工"这类模型本身就处理得很好的，仍交给模型判断。
"""

import re

RULES = [
    ("安全问题", re.compile(r"冒烟|起火|着火|爆炸|漏电|触电|烫伤|受伤|划伤|过敏|中毒")),
    ("投诉纠纷", re.compile(r"投诉|12315|消协|律师|起诉|曝光|媒体")),
    ("政策外诉求", re.compile(r"赔偿|赔我|索赔|补偿我|赔钱|精神损失")),
]


def match_handoff(text: str) -> str | None:
    """返回命中的转人工原因，没有命中返回 None。"""
    for reason, pattern in RULES:
        if pattern.search(text):
            return reason
    return None


def hint(reason: str) -> str:
    return (f"\n\n[系统提示：这条消息命中了必须转人工的规则（{reason}）。请先安抚用户，"
            f"能查到相关订单的先查订单，然后调用 transfer_to_human，不要先征求用户同意。]")
