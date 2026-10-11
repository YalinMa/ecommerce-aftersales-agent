# 林间优选 · 售后客服 Agent

一个从零手写、逐版本迭代的电商售后 Agent，用来理解大模型应用的完整工作流程。

| 版本 | 内容 | 状态 |
|---|---|---|
| V0 | 纯对话：system prompt + 多轮对话 + 场景回放 | ✅ |
| V1 | 工具调用：手写 agent loop，查订单、物流、售后进度，工具化转人工 | ✅ |
| V2 | 知识库 RAG：混合检索、回答带出处、引用核查、检索评测 | ✅ |
| V3 | 写操作：发起退货申请，带资格校验和用户确认 | ⏳ |
| V4 | MCP 化：业务工具拆成独立 MCP Server | ⏳ |
| V5 | 评测和可观测：自动评分、调用链路追踪 | ⏳ |
| V6 | 产品化：嵌入模拟商城、流式输出、转人工 | ⏳ |

## 快速开始

以下命令都在本项目目录 `aftersales-agent/` 下运行：

```bash
cd aftersales-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                  # 填入要用的厂商的 API key
python scripts/seed_data.py           # 生成模拟商城数据（时间相对于当前，隔天测试前重新生成）
python scripts/build_index.py         # 构建知识库索引（修改 knowledge/ 后需重新运行）

python web/server.py                  # 网页版演示：打开 http://localhost:8000，左边聊天、右边看工具调用过程
python chat.py -p deepseek            # 命令行对话，输入 /orders 查看当前用户的订单
python chat.py -p deepseek -u u1002   # 以其他用户身份登录
python scripts/run_scenarios.py -p deepseek -p openai -p qwen   # 回放 V2 场景，多厂商对比
python scripts/eval_retrieval.py      # 检索评测：BM25 / 向量 / 混合检索对比
python scripts/run_scenarios.py -p openai --out runs/<已有目录>  # 补跑一家，汇总进同一份对比
python scripts/run_scenarios.py -f scenarios/v1_dialogues.json  # 跑 V1 的场景集
python scripts/list_models.py -p kimi # 查看账号下可用的模型名
```

## 架构

```
用户 ──► chat.py ──► Session（agent loop）──► Provider 适配层 ──► 各家大模型
                        │                                    （Claude / OpenAI / DeepSeek / 通义 …）
                        │ 模型要求调用工具
                        ▼
                     工具执行器 ──► 订单 / 物流 / 售后工具 ──► SQLite 模拟商城
                        │              ▲
                        │              └── user_id 由登录态注入，模型无法指定
                        ├──► search_policy ──► 知识库（BM25 + 向量混合检索）
                        └──► transfer_to_human ──► 转人工（V6 接入真实队列）

                     回复 ──► 引用核查：引用的条款编号必须真的检索过
```

## 支持的模型厂商

| `-p` 参数 | 厂商 | 默认模型 | 接口 |
|---|---|---|---|
| `anthropic` | Anthropic Claude | claude-opus-5-5 | 官方 SDK |
| `openai` | OpenAI | gpt-5.5 | OpenAI SDK |
| `deepseek` | DeepSeek | deepseek-flash | OpenAI 兼容 |
| `minimax` | MiniMax | MiniMax-M3 | OpenAI 兼容 |
| `qwen` | 通义千问 | qwen3.7-plus | OpenAI 兼容 |
| `kimi` | Kimi | kimi-k3 | OpenAI 兼容 |
| `glm` | 智谱 GLM | glm-5.3 | OpenAI 兼容 |

只需在 `.env` 里填写要用的厂商的 key。默认模型名截至 2026-10，可用 `<厂商>_MODEL` 环境变量或 `-m` 参数覆盖。新增一个 OpenAI 兼容厂商，只要在 `agent/providers/__init__.py` 的 `PROVIDERS` 里加一项。

## 目录结构

```
agent/
  prompts/system_prompt.md   客服人设、能力范围、工具使用原则、售后政策、转人工规则
  prompts/history/           历次 prompt 存档
  config.py                  全局配置；拼接 prompt 模板与会话上下文（登录用户、日期）
  knowledge.py               知识库：切分、BM25、向量、混合检索
  session.py                 多轮会话 + 手写的 agent loop
  db.py                      模拟商城数据库连接
  providers/                 模型厂商适配层（含各家 tool calling 格式转换）
    __init__.py              厂商注册表（key、默认模型、接入点）
    anthropic_provider.py    Claude 官方 SDK
    openai_compat.py         OpenAI 兼容接口，覆盖其余厂商
  tools/                     工具层（与厂商无关）
    base.py                  工具定义、执行上下文、参数校验
    orders.py                订单、物流、售后查询
    policy.py                知识库检索工具
    handoff.py               转人工
knowledge/                   售后政策知识库（8 个文档、42 条条款）
chat.py                      命令行客服窗口
web/                         网页版演示（server.py 后端 + index.html 页面）
scenarios/                   测试场景（v0 纯对话、v1 工具调用、v2 知识库，含自动检查规则；v2_retrieval 检索测试集）
scripts/seed_data.py         生成模拟商城数据
scripts/build_index.py       构建知识库向量索引
scripts/eval_retrieval.py    检索评测
scripts/run_scenarios.py     批量回放、自动检查、多厂商对比报告
scripts/list_models.py       查看厂商账号下可用的模型
docs/
  v0-notes.md                V0 学习笔记和实验建议
  v0-round1-findings.md      V0 第一轮多模型对比
  v0-round2-findings.md      V0 第二轮：修改 prompt 后的对比和经验
  v1-notes.md                V1 学习笔记：agent loop、设计决策、厂商差异、踩坑记录
  v1-findings.md             V1 多模型测试结果
  v2-notes.md                V2 学习笔记：RAG 三步、检索对比、引用核查、踩坑记录
  v2-findings.md             V2 多模型测试结果
```
