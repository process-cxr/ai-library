---
title: Agent Compression
created: 2026-06-02
published: 2026-06-02
modified: 2026-06-04
type: topic
status: growing
area: application
tags:
  - agent
  - compression
  - context-pruning
  - tool-output-pruning
  - code-agent
---

Agent Compression 指在 agentic system 中，对上下文、工具输出、检索结果、会话历史和中间状态进行选择、裁剪、摘要或结构化保留的机制。它的目标不是简单减少 token，而是在有限 context window、有限预算和长程任务中，让 agent 持续看到对下一步行动真正有用的信息。

在 code agent 场景中，这个问题尤其突出。Agent 会反复读取文件、搜索仓库、运行测试、查看日志、修改代码，再把所有 observation 写回对话历史。如果不加控制，history 很快会被大段源码、grep 输出、traceback、diff 和重复日志填满。上下文膨胀不仅增加成本和延迟，也会让模型在无关信息中迷失，导致重复探索、错误修复或过早结束任务。

## 核心问题

Agent Compression 要回答三个问题：

1. **哪些信息应该进入上下文？**
   例如仓库代码、工具输出、历史消息、计划、错误日志和子代理结果，并不是都值得完整保留。

2. **信息应该以什么形态保留？**
   有些内容必须原文保留，例如 traceback、API 签名、测试失败断言；有些内容可以摘要，例如长时间探索过程；有些内容只需要结构占位，例如“旧日志已清理”。

3. **压缩应该发生在什么位置？**
   可以在检索后、工具输出进入历史前、模型调用前、会话接近上限时，或跨 session 持久化时发生。位置不同，影响也不同。

## 压缩对象

Agent Compression 的对象通常包括：

| 对象 | 常见处理方式 | 主要风险 |
|---|---|---|
| Repository context | 检索、重排、结构化选择、代码压缩 | 漏掉低相似但必要的依赖 |
| Tool observations | 行级剪枝、span extraction、日志截断 | 删除关键报错、路径、数字或断言 |
| Conversation history | snip、summary、compact boundary | 摘要扭曲任务意图或丢失约束 |
| Plans and tasks | 保留当前计划、折叠已完成步骤 | 计划和真实状态不一致 |
| Subagent outputs | summary-only return、sidechain transcript | 父 agent 看不到足够证据 |
| Long-term memory | 文件化记忆、经验沉淀、检索式记忆 | 过期记忆污染当前任务 |

这说明 agent compression 不是单一算法，而是一组贯穿 agent loop 的预算管理和状态治理机制。

## 主要策略

### Retrieval-side selection

最早发生的压缩通常在 retrieval 侧：先决定哪些外部上下文值得召回，再决定是否放入 prompt。

在代码任务中，简单相似度检索经常不够。当前文件里的 unfinished code 可能不能准确表达后续要写什么；真正必要的内容可能是 helper function、配置项、类型定义、调用约定或测试 fixture。好的 retrieval-side selection 需要结合关键词、结构、数据流、调用关系、生成草稿或模型不确定性。

这一类方法的核心思想是：不要把 retrieval 当作默认“多拿一些上下文”，而要把它当作 agent 的 context acquisition policy。无效检索会带来噪声、成本和注意力稀释。

### Context pruning before history

对 agent 来说，工具输出进入 conversation history 之前就可以压缩。这个位置很关键：一旦无关日志、文件内容或 grep 结果进入历史，后续每一轮都可能被重复携带、总结和再压缩。

工具输出剪枝通常更适合 extractive 方法，而不是 abstractive summary。原因是 coding agent 依赖很多字符级证据：

- traceback 的具体行号和异常类型；
- test assertion 的 expected / actual；
- grep 命中的文件路径；
- API 签名、参数名和 import；
- config key、环境变量和版本号；
- git diff 中的精确改动。

这些信息被自然语言摘要改写后，很容易变得不可验证或不可执行。因此，tool-output pruning 更常见的理想形态是 query-conditioned span extraction：只保留与当前目标相关的原文证据块。

### Structure-aware code compression

代码上下文不能像普通文本一样随意删 token。压缩时应尽量保护程序结构：

- 文件和模块边界；
- 类、函数和方法签名；
- import、常量和类型定义；
- 控制流 block；
- helper 调用链；
- 配置对象和测试 fixture。

较稳妥的策略是 coarse-to-fine：先在文件、类、函数层面选候选，再对特别长的函数做 block 级压缩。行级或 token 级删除可以进一步省 token，但更容易破坏语法、依赖和可读性。

### Conversation compaction

当 agent 会话已经变长，系统需要把旧历史折叠成可继续工作的状态。Conversation compaction 通常包含：

- 删除或引用过大的工具结果；
- 保留最近关键消息；
- 对旧阶段生成 summary；
- 保留当前 plan、pending task、最近文件、错误和用户约束；
- 记录 compact boundary，方便 resume 或审计。

产品级系统通常不会把 compaction 简化为“让模型总结聊天记录”。更可靠的做法是：语义状态由 LLM summarizer 概括，工程状态由代码从文件系统、计划文件、工具 registry、skill registry、MCP 状态和 transcript 中重新构造。

### Memory and persistence

长期 agent 不能只依赖当前 context window。跨 session 的压缩更接近 memory management：哪些经验、偏好、项目规则和任务状态应该持久化？哪些只属于一次会话？哪些过期后应该清理？

透明的文件化记忆便于用户审计和版本控制，但检索能力较弱；数据库或 embedding memory 更灵活，但更不透明。对 coding agent 来说，memory 不应替代真实文件读取，也不应把过期结论当成硬事实。

## 设计原则

### 相关不等于必要

Agent 真正需要的是能改变下一步判断或行动的信息。相似、相关、可读，不一定意味着必要。尤其在代码库中，低相似度的配置、类型、测试夹具和 helper 可能比高相似度的示例更关键。

### 越早剪枝，轨迹越干净

如果无关 observation 在进入 history 前被过滤，后续 compaction 的负担会小很多。工具输出剪枝、selective retrieval 和 deferred tool schema 都属于“前置压缩”。

### 原文证据优先于漂亮摘要

对代码、日志和错误信息，压缩应优先保留原文证据。摘要适合解释“做过什么”和“下一步是什么”，不适合替代 traceback、API 签名、测试断言和关键代码。

### 压缩是预算分配，不是等比例缩短

不同信息的价值不同。高价值小片段可以完整保留；大而重复的日志可以截断；长函数可以保留签名、关键 block 和依赖调用；已完成阶段可以摘要。

### 压缩必须可恢复或可审计

长程 agent 需要知道被压缩掉的内容还能否找回。常见做法包括 content references、append-only transcript、compact boundary、sidechain transcript 和重新读取文件。

## 常见失败模式

- 过度剪枝导致关键证据没有进入上下文；
- 摘要改写了用户真实意图或错误细节；
- token-level 删除破坏代码语法和依赖；
- 压缩器和生成器能力不一致，导致保留内容偏离最终模型需要；
- subagent summary 太短，父 agent 无法验证结论；
- 长期 memory 过期，污染当前任务；
- compaction 后丢失当前 plan、pending task 或最近错误状态。

## 发展趋势

近期研究从 repository retrieval 逐步走向 agent observation pruning：早期工作更关注如何召回代码片段，后续开始关注哪些候选对生成模型真正必要，再进一步把压缩位置前移到工具输出和多轮轨迹中。产品级 coding agent 则把压缩扩展为系统工程问题：缓存、会话持久化、summary、文件重读、工具状态和权限边界都会影响最终压缩策略。

这个趋势说明，agent compression 正在从“prompt compression algorithm”变成“agent runtime 的 context governance layer”。

## 相关论文

- [[sources/papers/2023-repocoder-repository-level-code-completion|RepoCoder]]：用生成草稿改进 repository-level retrieval query。
- [[sources/papers/2024-repoformer-selective-retrieval-for-repository-level-code-completion|Repoformer]]：让模型判断是否需要 cross-file retrieval。
- [[sources/papers/2024-longllmlingua-accelerating-and-enhancing-llms-in-long-context-scenarios|LongLLMLingua]]：通用长上下文 prompt compression 的代表方法。
- [[sources/papers/2025-coderag-relevant-and-necessary-knowledge|CodeRAG]]：强调 code knowledge 不仅要 relevant，还要 necessary。
- [[sources/papers/2025-longcodezip-compress-long-context-for-code-language-models|LongCodeZip]]：面向代码的 structure-aware compression。
- [[sources/papers/2026-codepromptzip-code-specific-prompt-compression|CodePromptZip]]：针对 RAG coding prompt 的 code token type compression。
- [[sources/papers/2026-swe-pruner-self-adaptive-context-pruning-for-coding-agents|SWE-Pruner]]：在工具输出进入 agent history 前做目标条件化剪枝。
- [[sources/papers/2026-squeez-task-conditioned-tool-output-pruning-for-coding-agents|Squeez]]：把 tool-output pruning 建模为任务条件化原文 span extraction。

## 相关项目

- [[projects/claude-code/compact-context-pipeline|compact Context Pipeline]]：把 `/compact` 放在 conversation compaction 和 context saturation 阶段看，说明产品级 code agent 的压缩不只是 LLM summary，还包括 prompt cache、session memory、最近文件、plan、skill、agent 状态和 MCP/tool schema 的 deterministic 重建。

## 相关概念

- [[application/rag/context-compression|Context Compression]]：更一般的上下文压缩问题。
- [[application/rag/retrieval|Retrieval]]：决定候选上下文池的召回质量。
- [[application/rag/chunking|Chunking]]：决定检索与压缩可操作的信息单元。
- [[application/agents/agent|Agent]]：工具调用、环境反馈和多轮执行的主体。
- [[application/tool-use/tool-calling|Tool Calling]]：工具输出是 agent compression 的关键入口。
