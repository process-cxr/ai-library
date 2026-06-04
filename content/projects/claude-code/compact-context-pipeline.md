---
title: "compact Context Pipeline"
created: 2026-06-04
published: 2026-06-04
modified: 2026-06-04
type: project
status: growing
tags:
  - claude-code
  - compact
  - code-agent
  - context-compression
  - context-pruning
---

这篇笔记整理 Claude Code `/compact` 的源码阅读结果。它关注的不是“怎么使用 `/compact`”，而是产品级 coding agent 如何在上下文接近饱和时，用一组预算、缓存、历史折叠、状态恢复和 LLM summary 机制维持会话可继续性。

## 输入与来源

- 参考入口：[小红书 Claude Code compact 相关笔记](https://www.xiaohongshu.com/explore/69fde32d000000003503ba09)。
- 本地读码材料：`agent-compression-related` 中的 Claude Code compact 读码内容。
- 相关主题页：[[application/agents/compression|Agent Compression]]。

> [!note] 来源边界
> 本文的机制描述以本地源码阅读材料为准，外部链接作为选题参考入口。

## 核心判断

如果只看用户界面，`/compact` 很容易被理解成“把旧聊天记录总结一下”。但源码里的实际形态更像一条 **deterministic context budget pipeline**：先用代码逻辑做 token 截断、工具结果清理、缓存编辑、历史折叠、session memory 拼接；只有前面这些手段救不了，才进入真正的 LLM summary。

这次读码主要对齐了几个模块：

| 源码位置 | 读到的机制 |
| --- | --- |
| `query.ts` | 主循环里 context 处理的入口顺序 |
| `services/compact/autoCompact.ts` | auto compact 阈值、递归保护、失败熔断、session memory 优先级 |
| `services/compact/microCompact.ts` | time-based / cached microcompact，两者都先于 auto compact |
| `services/compact/sessionMemoryCompact.ts` | 不调用 LLM 的 session memory compaction 路径 |
| `services/compact/compact.ts` | full compact、cache sharing、prompt-too-long retry、post-compact attachment |
| `services/compact/prompt.ts` | no-tools prompt、9 段 summary schema、`<analysis>` strip |

核心判断是：**LLM 负责回答“我们在做什么”，但路径、阈值、文件内容、skill 内容、plan、agent 状态、MCP/tool schema 这些工程状态，尽量由代码重新构造。**

## 入口顺序

主循环在每次用户输入或工具调用结果回来后，都会重新计算当前 message history 的 token 压力。顺序不是直接 auto compact，而是先走一串更便宜、更确定的路径：

```text
messages after compact boundary
  -> per-message tool result budget
  -> history snip
  -> microcompact
  -> contextCollapse
  -> autoCompact
  -> normal API call
  -> if API 413/media too large: reactiveCompact
```

这个顺序很重要。它说明 Claude Code 的策略不是“满了就总结”，而是“先把明显可删、可替换、可投影的东西处理掉，再判断是否真的需要 LLM summary”。尤其是 `microcompact` 和 `contextCollapse` 都在 `autoCompact` 前面，如果它们已经把上下文压到阈值以下，`autoCompact` 就不会触发。

## Auto Compact 触发线

触发阈值不是文档里常说的“接近 95%”。源码里的核心公式更直接：

```text
effectiveContextWindow = modelContextWindow - reservedSummaryOutputTokens
autoCompactThreshold = effectiveContextWindow - AUTOCOMPACT_BUFFER_TOKENS
```

相关常量是：

| 常量 | 值 | 作用 |
| --- | ---: | --- |
| `MAX_OUTPUT_TOKENS_FOR_SUMMARY` | `20_000` | 给 compact summary 的输出预留空间 |
| `AUTOCOMPACT_BUFFER_TOKENS` | `13_000` | 防止 compact 已经来不及时的额外 buffer |
| `MANUAL_COMPACT_BUFFER_TOKENS` | `3_000` | 手动 compact 的 blocking buffer |
| `MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES` | `3` | auto compact 连续失败后的熔断次数 |

所以在 200K context window 的模型上，默认触发线近似是：

```text
200K - 20K - 13K = 167K
```

也就是约 83.5%。20K 的来源不是拍脑袋，源码注释里写的是 compact summary 输出长度的 p99.99 约为 17,387 tokens，所以预留 20K 覆盖长尾。13K 是固定 buffer，用来防止“刚决定 compact，summary 请求自己又满了”。如果换成 1M context window，固定 buffer 的优势会更明显：summary 长度的 p99.99 不会随窗口线性增长，所以不应该按百分比浪费十几万 token。

auto compact 还有两个工程保护：

- `querySource === 'session_memory'` 或 `querySource === 'compact'` 时直接跳过，避免递归压缩自己。
- 连续失败达到 3 次后熔断。源码注释里提到一次 BigQuery 观察：2026-03-10 有 1,279 个 session 出现 50+ 次连续失败，全球每天浪费约 250K API calls。

## 五层压缩路径

进入 compact 管道后，可以按“是否需要 LLM”分成五层。

| 层级 | 机制 | 是否调用 LLM | 解决的问题 |
| --- | --- | --- | --- |
| 1 | per-message tool result budget / snip | 否 | 单条工具输出过大，例如一次 Bash 产出几十 K 日志 |
| 2 | microcompact | 否 | 旧 tool result 累积、prompt cache TTL 过期或 cache 可编辑 |
| 3 | contextCollapse | 主循环里是投影视图 | 折叠某些历史范围，避免立刻变成单个大 summary |
| 4 | sessionMemoryCompaction | 否 | 已有 session memory 时，用结构化记忆替代重新总结 |
| 5 | compactConversation | 是 | 前面都救不了时，fork 一个 summarizer 生成结构化 summary |

这几层不是冗余，而是各自针对不同“症状”。单条工具输出太大，应该先截断那条输出；旧 tool result 太多，应该先清工具结果；已经有 session memory，就没必要再让 LLM 从头总结；只有这些都不够，才把旧轨迹交给 LLM。

## Microcompact

`microcompact` 针对的是旧工具结果，而不是完整对话。它有两条路径。

第一条是 **time-based microcompact**。当距离上一条 main-loop assistant 消息超过阈值时，源码认为服务端 prompt cache 已经过期，再怎么发也会 cache miss，于是趁这个“注定要重写”的时刻，把旧 tool result 内容替换成占位符：

```text
[Old tool result content cleared]
```

默认配置里，时间阈值是 60 分钟，保留最近 5 个 compactable tool results。60 分钟对应服务端 1h prompt cache TTL：如果 cache 已经冷了，旧前缀无论如何都要重写，此时清理旧 tool result 不会额外制造 cache miss，只会减少这次重写要发送的 token。

第二条是 **cached microcompact**。它不改本地 message content，而是在 API 层发送 `cache_edits`，告诉服务端把指定 tool result 从 prompt cache 中删除。本地消息还保持完整，客户端看起来 cache 前缀仍然完整；服务端则少读一批旧工具结果。

这两条路径背后的共同思想是：**旧 observation 的清理应该尽量在缓存语义允许的地方发生，而不是让 LLM 总结旧日志。**

## Session Memory Compaction

在 auto compact 真正调用 LLM 前，Claude Code 会先尝试 `trySessionMemoryCompaction`。如果 session memory 已经存在且不是空模板，它会直接用 session memory 构造 compact summary message，而不是重新调用 summarizer。

这条路径有几个硬规则：

| 配置 | 默认值 | 含义 |
| --- | ---: | --- |
| `minTokens` | `10_000` | compact 后至少保留这么多尾部消息 token |
| `minTextBlockMessages` | `5` | 至少保留 5 条含文本的消息 |
| `maxTokens` | `40_000` | 尾部保留区硬上限 |

算法从 `lastSummarizedMessageId` 后面开始保留消息，如果保留区太小，就从尾部向前扩展，直到满足 token 和文本消息数量下限，或碰到 40K 上限。它还会调整边界，避免切断 `tool_use` / `tool_result` 对，以及同一个 assistant message id 下的 thinking/tool block。

这解释了为什么 session memory 可用时，auto compact 可以绕开 summarizer：如果后台 session memory 已经把“我们在做什么”抽好了，compact 只是把 memory、最近尾部消息、plan/hook 结果拼成新的 active context。

## LLM Summary 路径

只有前面路径失败或不可用，才进入 `compactConversation`。这时 Claude Code 不是换 Haiku 之类便宜模型，而是默认用主线程同款模型。原因在 `streamCompactSummary` 的 cache-sharing 路径里：compact 通过 `runForkedAgent` 复用主线程的 system prompt、tools、model、thinking config 和消息前缀，以便命中 prompt cache。

这里有一个关键源码注释：forked-agent 路径 **不能设置 `maxOutputTokens`**。因为设置它会改变 thinking config / cache key，导致 prompt cache mismatch。源码注释还提到一次 2026 年 1 月实验：关闭这条 cache sharing 路径会产生 98% cache miss，约占 fleet cache_creation 的 0.76%，大约 38B tokens/day。

compact prompt 也不是普通“请总结”。它前后各有 no-tools 约束：

- 开头 `NO_TOOLS_PREAMBLE` 明确要求 `TEXT ONLY`，禁止 Read/Bash/Grep/Glob/Edit/Write 等任何 tool。
- 结尾 `NO_TOOLS_TRAILER` 再提醒一次工具调用会被拒绝。
- `createCompactCanUseTool` 在执行层面也直接 deny 所有工具调用。

源码注释说明这不是多余的礼貌话。Sonnet 4.6+ 在 compact 任务上比 4.5 更容易误调工具，fallback 率从 0.01% 涨到 2.79%；把 no-tools 警告放在 prompt 最前面，是为了解决模型升级带来的 regression。

真正的 summary 模板是 9 个固定 section：Primary Request and Intent、Key Technical Concepts、Files and Code Sections、Errors and fixes、Problem Solving、All user messages、Pending Tasks、Current Work、Optional Next Step。`All user messages` 要求保留用户原话，这是防止 LLM 把用户反馈改写成错误任务意图的关键。

另外，`<analysis>` 只是 scratchpad。模型可以先写分析，但 `formatCompactSummary` 会在写回上下文前把 `<analysis>...</analysis>` strip 掉，只留下 `<summary>` 内容。

## Post-Compact 状态恢复

LLM 生成 summary 后，Claude Code 不会把 summary 直接当成新历史结束。它还会做大量 deterministic 拼接。

第一件事是清缓存状态：

```text
readFileState.clear()
loadedNestedMemoryPaths.clear()
```

这意味着 compact 后，下次 Read 同一个文件不能靠旧 readFileState 糊弄过去，必须重新从磁盘读。这样即使 summary 里文件路径或内容写错，只要后续逻辑需要文件原文，还是会回到真实磁盘内容。

第二件事是把最近读过的文件原文贴回来。默认最多恢复 5 个文件，每个文件最多 5K tokens，总预算 50K。文件不是从 summary 里“找路径”，而是从 compact 前的 `readFileState` 里取最近访问记录，再用 FileReadTool 重新读磁盘生成 attachment。也就是说，文件恢复是 deterministic 的，不依赖 LLM summary 是否记对。

第三件事是重注入运行状态：

- plan file 和 plan mode attachment；
- async agent 状态和输出路径；
- 已调用过的 skill body，单个 skill 最多 5K，总预算 25K；
- deferred tools / MCP instructions / agent listing 的 delta；
- session start hooks 的结果；
- pre-compact 已发现的 tool names。

这就是为什么 “summary 之后怎么拼接” 比 “summary 怎么写” 更重要。Claude Code 的设计是：LLM 只写任务语义，工程状态由代码重新挂回去。整套后处理可以总结为：**让 LLM 决定高层叙事，不让 LLM 决定路径、文件、工具、plan、skill 和 MCP 状态。**

## 边界和溢出防御

即便做了这些，compact 后仍然可能很大。最坏情况下可以接近：summary 17K + 文件 50K + skill 25K + plan + agent 状态 + MCP/tool delta。源码里因此还有几层防御。

第一层是 prompt-too-long retry。如果 compact 请求本身太长，`truncateHeadForPTLRetry` 会按 API round 从最早历史开始丢，最多重试 3 次。能解析 token gap 时按 gap 丢；解析不了时丢 20% group。为了避免 API 拒绝 assistant-first 消息，必要时还会插入一个 synthetic user marker：

```text
[earlier conversation truncated for compaction retry]
```

第二层是 compact 后立即估计 `truePostCompactTokenCount`，并记录 `willRetriggerNextTurn`。源码注释也很诚实：这个估计还没算 system prompt、tools、userContext，下一轮实际输入可能额外多 20K 到 40K，所以 `willRetriggerNextTurn = false` 不代表绝对安全。

第三层是 reactive compact。当正常请求发出去后 API 返回 prompt-too-long 或 media-size error，系统会尝试本地恢复、contextCollapse recovery，再进入 reactive compact。这里的关键不是“永远救回来”，而是不要让用户陷入“压缩 -> 仍然超限 -> 再压缩”的死循环。多次失败后会 surface 错误，最终兜底仍然是让用户 `/clear` 或放弃当前会话。

## 和 `/clear` 的区别

`/compact` 不是 `/clear` 的温和版。它保留的是“可继续工作的最小工程状态”：summary、最近尾部消息、文件重读结果、plan、skill、agent、MCP/tool delta、session hooks。`/clear` 则是显式承认这段历史已经不值得治理，直接断开。

从源码看，Claude Code 把决策权分得很清楚：

- 能用 deterministic 规则删的，就不要叫 LLM；
- 能从磁盘、cache、plan、skill registry 重新构造的，就不要相信 summary；
- 只有“用户到底要什么、我们做到了哪一步”这种语义状态，才交给 LLM；
- 如果压缩链条持续失败，最终让用户选择 `/clear`，而不是无限递归救上下文。

这比“conversation summarization”这个词本身更有启发。一个产品级 coding agent 的 context compression，不是单个 summarizer，而是一组预算、缓存、状态恢复和失败熔断策略。

## 回补到主题页

- [[application/agents/compression|Agent Compression]]：`/compact` 对应 context saturation 阶段，说明产品级 code agent 的压缩不只是 tool-output pruning，还包括会话轨迹、缓存和工程状态恢复。
- [[application/rag/context-compression|Context Compression]]：`/compact` 展示了通用 context compression 在交互式工程任务里的产品化形态。
