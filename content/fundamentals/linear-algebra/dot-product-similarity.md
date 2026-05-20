---
title: Dot Product and Similarity
created: 2026-01-10
published: 2026-01-10
modified: 2026-01-17
type: topic
status: growing
area: fundamentals
tags:
  - math
  - linear-algebra
  - similarity
aliases:
  - Dot Product
  - Cosine Similarity
---

## 概念界定

内积是一种把两个向量映射为标量的运算，可用于刻画两个向量在方向和尺度上的匹配程度。相似度度量则进一步用几何关系近似表示对象之间的相关性或语义接近程度。

## 背景与问题

大模型中有两类典型问题都依赖相似度：一类是 Attention 中 query 与 key 的匹配，另一类是 RAG 和 embedding 检索中文本向量之间的匹配。理解内积和余弦相似度，可以帮助区分“向量几何接近”和“语义真正相同”之间的差别。

## 定义与记号

两个向量：

```text
x = [x_1, x_2, ..., x_d]
y = [y_1, y_2, ..., y_d]
```

内积定义为：

```text
x · y = Σ_i x_i y_i
```

几何形式：

```text
x · y = ||x|| ||y|| cos(θ)
```

余弦相似度定义为：

```text
cos(x, y) = (x · y) / (||x|| ||y||)
```

## 直观解释

内积既受方向影响，也受长度影响。两个方向接近且长度较大的向量会有较大内积。余弦相似度则通过除以范数去掉长度影响，更强调方向是否一致。

## 基本性质

- 内积越大，通常表示两个向量在当前表示空间中越匹配。
- 余弦相似度范围通常在 `[-1, 1]`，更关注方向相似。
- 如果向量已经归一化，内积和余弦相似度等价。
- 相似度的语义依赖 embedding 模型的训练目标，不是线性代数自动赋予的。

## 示例

Attention score：

```text
score(i, j) = q_i · k_j / sqrt(d_head)
```

其中 `q_i` 表示第 `i` 个 token 的 query，`k_j` 表示第 `j` 个 token 的 key。点积越大，说明当前 query 与该 key 越匹配。

向量检索：

```text
query_embedding:    [D]
document_embedding: [D]
similarity = cos(query_embedding, document_embedding)
```

RAG 系统会优先取相似度较高的文档片段。

## 常见误解

- 误解：内积大就一定表示语义相同。
  - 正确理解：内积还受向量长度影响，且语义取决于 embedding 训练方式。
- 误解：Attention score 就是最终注意力权重。
  - 正确理解：score 还要经过 scale、mask 和 softmax。
- 误解：余弦相似度是唯一正确的检索度量。
  - 正确理解：dot product、cosine、L2 都可能使用，关键要和 embedding 训练目标一致。

## 相关概念

- [[fundamentals/linear-algebra/norm|范数]] — 余弦相似度需要范数归一化。
- [[fundamentals/neural-network-basics/softmax|Softmax]] — Attention score 转为权重的归一化函数。
- [[architecture/attention/attention|Attention 机制]] — 内积相似度的核心应用。
- [[application/rag/rag|RAG 检索增强生成]] — 向量相似度的应用场景。
