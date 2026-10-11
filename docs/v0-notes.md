# V0 学习笔记：纯对话客服

## 这一版做了什么

只有「system prompt + 多轮对话」，没有工具，没有知识库。售后政策直接写在 system prompt 里。

## 一次请求里到底发了什么

在 `chat.py` 里聊两轮后输入 `/debug`，会看到类似这样的结构：

```json
{
  "model": "claude-opus-5-5",
  "system": "# 角色\n你是「林间优选」...",
  "messages": [
    {"role": "user", "content": "衣服能退吗"},
    {"role": "assistant", "content": [{"type": "thinking", ...}, {"type": "text", "text": "..."}]},
    {"role": "user", "content": "<你的下一句话>"}
  ],
  "output_config": {"effort": "low"}
}
```

要点：

1. **模型是无状态的。** 每一轮都要把完整历史重新发一遍，所谓"记忆"就是这个 `messages` 数组。
2. **system 和 messages 分开。** system 是"运营方"的规则，messages 是对话内容。用户在对话里说"忽略之前的指令"，权重不如 system。
3. **assistant 的 content 是一组 block。** 除了 text 还有 thinking（模型的推理过程，默认不展示内容）。回传历史时要原样保留，删掉反而可能报错。
4. **token 会越滚越多。** 每轮输入 token = system + 全部历史。观察 `chat.py` 每轮打印的输入 token，会看到它持续增长，这就是为什么长对话需要摘要压缩（V6）。

## 不同厂商的请求格式差异

用 `/debug` 分别看 `-p anthropic` 和 `-p deepseek` 的请求，会发现：

| | Anthropic | OpenAI 兼容（DeepSeek、通义等） |
|---|---|---|
| system prompt | 独立的 `system` 参数 | `messages` 里第一条 `role: system` |
| assistant 回复 | 一组 content blocks（thinking + text） | 一个字符串 |
| 思考过程 | thinking block，需要原样回传 | `reasoning_content` 字段，**不要**回传 |
| token 缓存命中 | `cache_read_input_tokens` | `cached_tokens`，DeepSeek 叫 `prompt_cache_hit_tokens` |

所以项目里加了一层 Provider：Session 只管"存历史、发消息"，格式转换交给各厂商的适配器。这一层到 V1 会更重要，因为各家 tool calling 的格式差异更大。

## 设计决策

| 决策 | 原因 |
|---|---|
| 政策直接写进 system prompt | 政策只有 1700 字左右，全塞进去最简单也最准。等政策变多、变复杂再上 RAG（V2），届时可以对比两种方案 |
| 明确写出"V0 无法查询订单" | 防止模型为了"帮忙"而编造物流状态，这是客服场景最危险的幻觉 |
| 转人工用固定标记 `[转人工]` | 程序可以稳定识别，后续接人工队列和做自动评测都依赖它 |
| effort 用 low | 客服对话推理不重、对延迟敏感。可以改成 medium 对比效果和耗时 |
| 历史按厂商原生格式存 | 转成统一格式会丢信息（比如 Claude 的 thinking 块必须原样回传）。代价是中途换厂商要开新会话 |
| OpenAI 兼容厂商共用一个实现 | 各家差异只在 base_url、模型名和少量私有参数，用注册表配置即可 |
| 开启 fallbacks | 模型的安全分类器偶尔会误判拒答，开启后服务端自动换模型重试，用户不会看到空回复 |

## 建议你动手做的实验

每做一个实验，跑一次 `python scripts/run_scenarios.py`，对比两份报告：

1. **删掉「当前能力边界」那一节**，再跑 `logistics-01`，看模型会不会编造物流信息。
2. **删掉「回复简洁」的要求**，看回复长度和 token 的变化。
3. **把 effort 改成 medium**（`.env` 里 `AGENT_EFFORT=medium`），对比回答质量、耗时和 token。
4. **换个人设语气**，比如更活泼的"亲～"风格，看哪种更适合售后场景。
5. **多模型横向对比**：`python scripts/run_scenarios.py -p anthropic -p deepseek -p qwen`，对比 `summary.md` 里的耗时和 token，再逐条看各家 `report.md` 的回答质量。哪家会编造物流？哪家转人工更积极？这张对比表很适合放进作品集。
6. **自己补充 5 个刁钻场景**，加进 `scenarios/v0_dialogues.json`，找到模型答错的地方，然后改 prompt 修复。把"发现问题 → 修改 → 验证"的过程记下来，这是面试时最好的素材。

## V0 的局限（也是 V1 的动机）

- 看不到真实订单，大部分实际问题只能引导用户自助或转人工。
- 政策判断依赖用户自述（"前天签收的"），无法核实。

V1 会接入订单查询工具，让模型从"只能说"变成"能查"。
