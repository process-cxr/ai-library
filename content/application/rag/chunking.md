---
title: Chunking
created: 2026-05-03
published: 2026-05-03
modified: 2026-06-04
type: topic
status: growing
area: application
tags:
  - rag
  - chunking
---

Chunking 是把长文档、代码库或多模态资料切成可检索、可排序、可压缩片段的过程。它决定了 RAG 系统能召回什么，也决定了后续 reranking 和 [[application/rag/context-compression|Context Compression]] 能操作的最小信息单元。

切块的目标不是平均分配 token，而是在召回率、语义完整性和上下文预算之间取平衡。

## 常见粒度

- Fixed-size chunk：按 token 或字符长度切分，简单稳定，但容易切断表格、公式、代码块和段落。
- Sliding window：固定窗口加 overlap，能缓解边界截断，但会增加索引量和重复上下文。
- Paragraph / section chunk：按标题、段落或 Markdown 结构切分，更适合文档问答。
- Semantic chunk：按语义相似度、topic shift 或模型打分切分，边界更自然，但实现成本更高。
- Code-aware chunk：按文件、类、函数、方法、block 或 AST 节点切分，适合代码检索和 code agent。

## 关键取舍

chunk 太小会提高局部精度，但可能丢失定义、前提、变量来源和论证链。chunk 太大能保留上下文，却会降低检索精度，并把更多无关 token 带入 prompt。

实际系统常用两级粒度：

1. 用较小 chunk 做索引和召回；
2. 命中后扩展到父级 section、函数或相邻窗口；
3. 再通过 reranking 或 context compression 在预算内选择最终上下文。

这种做法能同时保留 recall 和完整性。

## 代码场景

代码 chunking 要优先保护结构边界。随意按 token 切分代码，容易破坏缩进、函数签名、import、类型定义和变量依赖。

常见策略是：

- 用文件或模块作为粗粒度候选；
- 用类和函数作为主要检索单元；
- 对特别长的函数，再按控制流、空行、语义 block 或 perplexity shift 切分；
- 保留函数签名、类名和必要 import，避免只留下函数体片段。

近期代码检索和压缩研究也在强调这一点：固定长度 chunk 很难表达 repository-level coding 需要的程序结构。更常见的方向是用函数、类、block、AST 节点、调用关系或数据流作为候选单位，再在预算内选择真正进入 prompt 的片段。

相关论文可参考 [[sources/papers/2025-longcodezip-compress-long-context-for-code-language-models|LongCodeZip]]、[[sources/papers/2024-repoformer-selective-retrieval-for-repository-level-code-completion|Repoformer]] 和 [[sources/papers/2025-coderag-relevant-and-necessary-knowledge|CodeRAG]]。

## 相关概念

- [[application/rag/retrieval|Retrieval]]：chunk 是检索系统的基本候选单位。
- [[application/rag/reranking|Reranking]]：对召回 chunk 做更细的相关性排序。
- [[application/rag/context-compression|Context Compression]]：在召回 chunk 之后继续裁剪和预算分配。
