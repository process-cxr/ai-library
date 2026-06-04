---
title: Context Compression
created: 2026-05-10
published: 2026-05-10
modified: 2026-06-04
type: topic
status: growing
area: application
tags:
  - rag
  - context-compression
---

Context Compression 指在不改变主模型参数的情况下，把原始上下文压缩成更短、更高信息密度的输入。它常用于 RAG、long-context QA、code agent 和 repository-level coding 场景，用来降低 token 成本和推理延迟，并减少无关内容对模型注意力的干扰。

上下文压缩不是简单截断。它本质上是一个 token budget allocation 问题：在固定上下文窗口内，哪些内容应完整保留，哪些应摘要，哪些只保留结构提示，哪些可以丢弃。

## 核心问题

好的 Context Compression 需要回答：

- 当前任务真正需要哪些证据、定义、约束或依赖？
- 哪些内容与 query 表面相似，但不会改善最终回答？
- 哪些内容表面不相似，却提供必要背景？
- 压缩后是否仍保留可验证的原文证据？
- 压缩是否破坏代码、表格、JSON、公式或引用关系？

因此，压缩的目标不是最大化压缩率，而是在更小输入中保留足够的任务可用信息。

## 与 RAG 的关系

RAG 的 retrieval 阶段决定候选上下文池，Context Compression 决定候选池进入 prompt 的形态和预算分配。两者通常串联：

1. [[application/rag/retrieval|Retrieval]] 召回文档、代码、日志或工具结果；
2. [[application/rag/reranking|Reranking]] 对候选内容排序；
3. Context Compression 根据 token budget 选择、裁剪或摘要；
4. LLM 基于压缩后的上下文生成答案。

Compression 不是 retrieval 的替代品。若 retrieval 漏掉关键证据，后续压缩无法恢复。反过来，即使 retrieval 召回了正确内容，如果压缩时删掉关键细节，最终生成仍会失败。

## 主要形态

### Extractive compression

Extractive compression 从原文中选择片段、句子、行或 span，不改写内容。它适合需要精确证据的场景：

- traceback、测试断言和日志；
- API 签名、参数名、路径和版本号；
- 法律、医学、配置、JSON、代码等不能随意改写的文本；
- 需要引用来源的问答。

它的优势是保真，缺点是压缩率有限，且可能保留不够连贯的上下文。

### Abstractive compression

Abstractive compression 用模型生成摘要或重写后的短上下文。它适合压缩长对话、长文档背景、已完成阶段和非精确性说明。

它的风险是 hallucination、遗漏约束、改写数字或把用户意图总结错。因此在高精度任务中，摘要常需要搭配原文证据或 content reference。

### Structural compression

Structural compression 保留文档或代码的结构骨架，例如标题、函数签名、类关系、目录树、调用关系、表格 schema 或 chunk placeholder。它不追求保留全部原文，而是让模型理解信息组织方式，并知道需要时可以进一步检索或读取。

### Soft compression

Soft prompt compression 把上下文压缩成连续向量、prefix embedding 或 latent memory，再交给模型使用。它节省 token，但通常需要访问模型内部或训练适配模块，因此在闭源 API 模型场景中较难使用。

## 相关性信号

压缩需要判断内容价值。常见 signal 包括：

| Signal | 适用场景 | 风险 |
|---|---|---|
| Embedding similarity | 语义相近文档、FAQ、普通问答 | 漏掉低相似但必要的依赖 |
| BM25 / keyword overlap | 术语明确、实体名稳定的检索 | 同义表达和抽象依赖较弱 |
| Reranker score | 精排候选文档或代码片段 | 成本较高，受训练分布影响 |
| Attention / perplexity | 贴近生成模型条件预测行为 | 需要额外前向，延迟更高 |
| Program structure | 代码、配置、API 调用和依赖图 | 解析器和语言支持复杂 |
| Task state | agent 当前目标、计划、错误和下一步 | 状态若不准会放大错误 |

一个重要原则是：similarity 不等于 usefulness。压缩器应尽量估计内容对当前任务的边际贡献，而不是只看词面或向量相似度。

## 代码与工具输出场景

代码上下文压缩比普通文本更脆弱，因为删除局部 token 可能破坏：

- 缩进、括号和语法结构；
- 函数签名、类型、默认参数；
- import、常量和配置对象；
- helper function 与调用链；
- 测试 fixture 和错误复现路径。

因此代码场景更适合 structure-aware compression：先按文件、类、函数和 block 选择，再在必要时做行级或 token 级压缩。

Coding agent 的工具输出也需要特殊处理。工具 observation 可能包含日志、grep 输出、traceback、diff、文件片段和命令结果。对这些内容，原文 span 往往比摘要更可靠。更完整的 agent 视角见 [[application/agents/compression|Agent Compression]]。

## 预算分配

压缩不应给所有 chunk 使用相同保留比例。更合理的策略包括：

- 高价值小片段完整保留；
- 长而重复的内容摘要或截断；
- 代码优先保留结构完整单元；
- 日志和错误信息保留原文关键 span；
- 低相关内容保留标题、路径或 placeholder；
- 对任务仍不确定的部分保留可追溯引用。

在代码和结构化文档中，可以把选择问题看成 knapsack：每个片段有 token cost 和 task value，目标是在预算内最大化总价值。

## Failure Modes

- 过度压缩导致缺少必要证据；
- 摘要式压缩改写数字、路径、API 或异常条件；
- token-level 删除破坏代码、JSON、表格或公式；
- 相似度信号偏向表面相关内容，漏掉抽象依赖；
- 压缩模型与生成模型偏好不一致；
- 上游 retrieval 已漏掉关键信息；
- 压缩过程不可审计，用户不知道哪些内容被丢弃。

## 发展趋势

近期研究从通用 prompt compression 逐步走向 task-aware、structure-aware 和 agent-aware compression。自然语言长上下文任务强调 question-aware compression 和证据密度；代码任务强调结构边界和依赖关系；coding agent 则进一步把压缩位置前移到工具输出进入历史之前。

## 相关论文

- [[sources/papers/2024-longllmlingua-accelerating-and-enhancing-llms-in-long-context-scenarios|LongLLMLingua]]：通用长上下文 prompt compression。
- [[sources/papers/2025-longcodezip-compress-long-context-for-code-language-models|LongCodeZip]]：面向代码的 structure-aware compression。
- [[sources/papers/2026-codepromptzip-code-specific-prompt-compression|CodePromptZip]]：RAG coding prompt 的 code-specific compression。
- [[sources/papers/2026-swe-pruner-self-adaptive-context-pruning-for-coding-agents|SWE-Pruner]]：coding agent 工具输出剪枝。
- [[sources/papers/2026-squeez-task-conditioned-tool-output-pruning-for-coding-agents|Squeez]]：任务条件化 tool-output span extraction。

## 相关概念

- [[application/rag/retrieval|Retrieval]]：决定候选上下文池。
- [[application/rag/reranking|Reranking]]：在压缩前重排候选内容。
- [[application/rag/chunking|Chunking]]：决定压缩可操作的最小信息单元。
- [[application/agents/compression|Agent Compression]]：把压缩放到 agent 的工具输出、轨迹和记忆管理中。
- [[inference/performance/latency-throughput|Latency and Throughput]]：压缩增加预处理开销，但可能降低整体推理成本。
