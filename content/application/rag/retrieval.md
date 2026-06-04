---
title: Retrieval
created: 2026-05-09
published: 2026-05-09
modified: 2026-06-04
type: topic
status: growing
area: application
tags:
  - rag
  - retrieval
---

Retrieval 是 RAG pipeline 中从外部知识库、文档集合、代码库或工具结果里召回候选上下文的阶段。它决定模型最终能看到的信息上限：如果检索阶段漏掉关键证据，后续 reranking、[[application/rag/context-compression|Context Compression]] 和 prompt engineering 通常都无法弥补。

## 常见方法

- Sparse retrieval：BM25、关键词匹配、倒排索引。适合术语明确、实体名稳定、查询包含关键字的场景。
- Dense retrieval：embedding + vector search。适合同义表达、语义相近但词面不同的文档问答。
- Hybrid search：结合 sparse 和 dense，常用于生产 RAG。
- Structure-aware retrieval：利用标题层级、链接、代码结构、metadata、调用图或时间戳约束候选范围。
- Iterative retrieval：根据中间答案、缺失信息或生成过程继续检索。
- Selective retrieval：先判断是否需要检索，再决定是否把外部上下文加入 prompt。

## 评价指标

Retrieval 主要看召回质量，而不是最终生成文本本身。常见指标包括：

- Recall@k：正确证据是否出现在 top-k；
- Precision@k：top-k 中有多少是真正相关内容；
- MRR / NDCG：正确内容在排序中的位置；
- coverage：是否覆盖回答所需的多个证据点；
- downstream quality：加入检索结果后，最终答案是否更正确。

在 RAG 调试中，优先检查 retrieval recall。召回不足时，盲目调 prompt 往往只是在要求模型弥补缺失证据。

## 相似性与相关性

检索系统常用 similarity score，但 similarity 不等于 task relevance。两个片段可能词面或 embedding 很像，却对当前问题没有帮助；另一个片段可能不相似，却包含关键定义、配置、边界条件或依赖关系。

代码场景尤其明显。repository-level code completion 可能需要：

- 目标函数调用的 helper function；
- 配置类或默认参数；
- 类型定义和接口约束；
- 相同项目中的实现模式；
- import 和常量定义。

这些片段不一定和当前 instruction 有高 embedding similarity。因此代码检索常会引入更贴近任务的信号，例如生成草稿、模型不确定性、conditional perplexity、调用关系和 dataflow。相关研究的共同趋势是：retrieval 不只是“搜相似片段”，还包括 query formulation、selective retrieval 和必要性判断。

相关论文可参考 [[sources/papers/2023-repocoder-repository-level-code-completion|RepoCoder]]、[[sources/papers/2024-repoformer-selective-retrieval-for-repository-level-code-completion|Repoformer]]、[[sources/papers/2025-coderag-relevant-and-necessary-knowledge|CodeRAG]] 和 [[sources/papers/2025-longcodezip-compress-long-context-for-code-language-models|LongCodeZip]]。

## 常见失败模式

- top-k 太小，关键证据没有被召回；
- chunk 太小，召回片段缺少上下文；
- chunk 太大，召回结果含有大量无关内容；
- embedding 模型与领域不匹配；
- query rewriting 改写过度，丢失原始约束；
- 只用相似度排序，漏掉低相似但高依赖价值的内容。

## 相关概念

- [[application/rag/chunking|Chunking]]：决定检索候选单位。
- [[application/rag/embedding|Embedding for RAG]]：dense retrieval 的表示基础。
- [[application/rag/reranking|Reranking]]：对初步召回结果重新排序。
- [[application/rag/hybrid-search|Hybrid Search]]：结合关键词和向量检索。
