"""查看某个厂商账号下实际可用的模型名，用来核对默认配置是否过时。

用法：
    python scripts/list_models.py -p deepseek
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import anthropic  # noqa: E402
import openai  # noqa: E402

from agent.providers import PROVIDERS  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--provider", required=True, choices=list(PROVIDERS))
    name = parser.parse_args().provider
    spec = PROVIDERS[name]

    api_key = os.getenv(spec.api_key_env)
    if not api_key:
        raise SystemExit(f"请先在 .env 中填写 {spec.api_key_env}")

    if spec.kind == "anthropic":
        client = anthropic.Anthropic(api_key=api_key)
    else:
        base_url = os.getenv(f"{name.upper()}_BASE_URL") or spec.base_url
        client = openai.OpenAI(api_key=api_key, base_url=base_url)

    try:
        ids = sorted(m.id for m in client.models.list())
    except (anthropic.APIError, openai.APIError) as e:
        raise SystemExit(f"该厂商不支持或拒绝了 /models 查询：{e}")

    print(f"{spec.label} 可用模型（当前默认：{spec.default_model}）：")
    for i in ids:
        print(("  * " if i == spec.default_model else "    ") + i)


if __name__ == "__main__":
    main()
