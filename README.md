# 林间优选 · 售后客服 Agent

一个从零手写、逐版本迭代的电商售后 Agent，用来理解大模型应用的完整工作流程。

| 版本 | 内容 | 状态 |
|---|---|---|
| V0 | 纯对话：system prompt + 多轮对话 + 场景回放 | ✅ |
| V1 | 工具调用：手写 agent loop，查订单、物流、售后进度，工具化转人工 | ✅ |
| V2 | 知识库 RAG：混合检索、回答带出处、引用核查、检索评测 | ✅ |
| V3 | 写操作：代提交售后申请，规则引擎校验、用户确认、幂等、审计日志、转人工规则兜底 | ✅ |
| V4 | MCP 化：商城拆成独立 MCP 服务，签名令牌传递身份、确认签名保护写操作，可接入 Claude Desktop | ✅ |
| V5 | 评测和可观测：模型评分（含评分模型校准）、调用链路追踪、可视化评测报告 | ✅ |
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
python -m mall.mcp_server             # 单独启动商城 MCP 服务（不启动也行，客服 Agent 会自动拉起本机服务）
python chat.py -p deepseek -u u1002   # 以其他用户身份登录
python scripts/run_scenarios.py -p deepseek -p openai -p qwen   # 回放最新版本（V3）场景，多厂商对比
python scripts/eval_retrieval.py      # 检索评测：BM25 / 向量 / 混合检索对比
python scripts/eval.py                # 用评分模型给最近一次回放打分，生成 eval_report.html（网页右上角"评测报告"可直接打开）
python scripts/judge_calibration.py   # 校准评分模型：用已知答案的对话检验它是否可靠
python scripts/run_scenarios.py -p openai --out runs/<已有目录>  # 补跑一家，汇总进同一份对比
python scripts/run_scenarios.py -f scenarios/v2_dialogues.json  # 跑之前版本的场景集（回归测试）
python scripts/list_models.py -p kimi # 查看账号下可用的模型名
```

## 架构

```
┌──────────────────── 客服 Agent（agent/）────────────────────┐          ┌────────── 商城 MCP 服务（mall/）──────────┐
│                                                              │          │                                          │
│ 用户 ─► Session（agent loop）─► Provider 适配层 ─► 各家大模型 │          │ 工具（给大模型）                          │
│            │                                                 │   MCP    │   search_policy ─► 知识库（混合检索）      │
│            │ 模型要求调用工具                                 │ ◄──────► │   订单 / 物流 / 售后查询 ─► SQLite         │
│            ▼                                                 │  HTTP    │   check_return_eligibility ─► 规则引擎   │
│        工具执行器 ─► 商城工具（MCP 自动发现）────────────────┼─────────►│   create_aftersale_request ─► 待确认申请   │
│                  └► transfer_to_human（客服自己的工具）       │ 请求头带  │                                          │
│                                                              │ 签名令牌  │ 工具（只给宿主程序，带 host_only 标记）    │
│ 用户点击「确认提交」─► 计算确认签名 ─────────────────────────┼─────────►│   confirm / cancel：验签 → 再校验 → 幂等   │
│                                                              │          │     → 写库 → 审计日志                     │
│ 转人工规则兜底、引用核查、单号核查、链路追踪                  │          │ 资源 kb://clauses：知识库条款              │
└──────────────────────────────────────────────────────────────┘          └──────────────────────────────────────────┘
                                                                           同一个服务也可以用 stdio 接入 Claude Desktop

评测：回放场景 ──► 自动检查（流程对不对）+ 评分模型（话说得对不对）──► 评测报告
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
agent/                       客服 Agent（不引用 mall/ 的任何代码）
  prompts/system_prompt.md   客服人设、能力范围、工具使用原则、办理流程、转人工规则
  prompts/history/           历次 prompt 存档
  config.py                  全局配置；拼接 prompt 模板与会话上下文（登录用户、日期）
  session.py                 多轮会话 + 手写的 agent loop + 引用 / 单号核查 + 链路追踪
  mall_client.py             商城 MCP 客户端：工具发现、调用转发、确认签名、自动启动本地服务
  guards.py                  转人工规则兜底
  providers/                 模型厂商适配层（含各家 tool calling 格式转换）
  tools/                     工具执行器、参数校验、客服自己的工具（转人工）
mall/                        商城业务方（不引用 agent/ 的任何代码）
  mcp_server.py              MCP 服务：工具、宿主专用工具、知识库资源、健康检查
  auth.py                    身份令牌、确认签名
  orders.py                  订单、物流、售后记录查询
  aftersale.py               资格判断、待确认申请、确认后执行写入、审计日志
  rules.py                   售后资格规则引擎
  knowledge.py               知识库：切分、BM25、向量、混合检索
  policy.py                  知识库检索
  db.py                      数据库连接
knowledge/                   售后政策知识库（8 个文档、42 条条款）
chat.py                      命令行客服窗口
web/                         网页版演示（server.py 后端 + index.html 页面）
evals/judge.py               评分模型：评分标准、对话整理、结果校验
scenarios/                   测试场景（v0–v3 各版本对话场景，含自动检查规则；v2_retrieval 检索测试集；v5_judge_calibration 评分校准集）
scripts/seed_data.py         生成模拟商城数据
scripts/build_index.py       构建知识库向量索引
scripts/eval_retrieval.py    检索评测
scripts/run_scenarios.py     批量回放、自动检查、多厂商对比报告
scripts/eval.py              模型评分 + 可视化评测报告
scripts/judge_calibration.py 评分模型校准
scripts/list_models.py       查看厂商账号下可用的模型
docs/
  v0-notes.md                V0 学习笔记和实验建议
  v0-round1-findings.md      V0 第一轮多模型对比
  v0-round2-findings.md      V0 第二轮：修改 prompt 后的对比和经验
  v1-notes.md                V1 学习笔记：agent loop、设计决策、厂商差异、踩坑记录
  v1-findings.md             V1 多模型测试结果
  v2-notes.md                V2 学习笔记：RAG 三步、检索对比、引用核查、踩坑记录
  v2-findings.md             V2 多模型测试结果
  v3-notes.md                V3 学习笔记：写操作的多层防线、规则引擎、伪造通知攻击、踩坑记录
  v3-findings.md             V3 多模型测试结果
  v4-notes.md                V4 学习笔记：MCP 拆分、身份令牌、跨服务确认签名、接入 Claude Desktop
  v5-notes.md                V5 学习笔记：评分设计、评分模型校准、可观测性、踩坑记录
  v5-findings.md             V5 评测结果
```
