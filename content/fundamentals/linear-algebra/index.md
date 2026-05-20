---
title: 线性代数
type: topic
status: growing
area: fundamentals
tags:
  - math
  - linear-algebra
  - deep-learning
aliases:
  - Linear Algebra
---

## 概念界定

线性代数研究向量、矩阵、线性空间、线性变换以及它们之间的代数结构。在深度学习和大模型中，线性代数不是孤立的数学背景，而是模型表示、参数计算、注意力机制、向量检索、低秩微调和数值稳定性的共同语言。

这一节只保留理解大模型主线所必需的线性代数内容，不追求完整覆盖传统线性代数课程。

## 背景与问题

大模型中的文本不会以自然语言形式直接参与计算，而是被转换为数值张量：token id 被映射为 embedding，序列被表示为矩阵，batch 被组织为高维张量，模型参数则表现为大量权重矩阵。模型训练和推理的核心过程，可以看作这些张量在不同线性变换、非线性函数和归一化操作之间流动。

因此，学习线性代数的目标不是背诵定理，而是解决几个实际问题：

- 如何读懂 hidden states、attention scores、logits 等张量的形状？
- 为什么 Transformer 中大量计算都可以归结为矩阵乘法？
- Attention 中的点积相似度到底在衡量什么？
- LoRA、量化、模型压缩为什么会频繁涉及低秩、范数和数值尺度？
- 为什么 shape 正确、数值稳定和表示空间理解对大模型工程同样重要？

## 知识结构

### 表示基础

- [[fundamentals/linear-algebra/vector-matrix|向量、矩阵与张量]] — 深度学习中的基本数据表示。
- [[fundamentals/linear-algebra/shape-dimension|形状与维度]] — batch、sequence length、hidden size、head dimension 等 shape 约定。
- [[fundamentals/linear-algebra/basis-coordinate|基与坐标]] — embedding 空间、坐标表示和分布式语义表示。

### 核心运算

- [[fundamentals/linear-algebra/matrix-multiplication|矩阵乘法]] — 线性层、Q/K/V 投影、Attention 和 MLP 的核心计算。
- [[fundamentals/linear-algebra/dot-product-similarity|内积与相似度]] — 点积、余弦相似度、attention score 和向量检索。
- [[fundamentals/linear-algebra/elementwise-broadcasting|逐元素运算与广播]] — bias、mask、gate、residual 和 normalization 中的基础规则。

### 表示变换与压缩

- [[fundamentals/linear-algebra/linear-transformation|线性变换]] — 权重矩阵如何改变表示空间。
- [[fundamentals/linear-algebra/rank-low-rank|秩与低秩近似]] — LoRA、参数高效微调和模型压缩的基础。
- [[fundamentals/linear-algebra/eigendecomposition-svd|特征分解与 SVD]] — 主方向、降维、低秩分解和表示分析。

### 尺度与稳定性

- [[fundamentals/linear-algebra/norm|范数]] — 向量长度、距离、梯度裁剪、权重衰减和归一化。
- [[fundamentals/linear-algebra/numerical-stability|数值稳定性]] — 溢出、下溢、低精度计算和稳定 softmax。

## 推荐阅读顺序

1. [[fundamentals/linear-algebra/vector-matrix|向量、矩阵与张量]]
2. [[fundamentals/linear-algebra/shape-dimension|形状与维度]]
3. [[fundamentals/linear-algebra/matrix-multiplication|矩阵乘法]]
4. [[fundamentals/linear-algebra/dot-product-similarity|内积与相似度]]
5. [[fundamentals/linear-algebra/linear-transformation|线性变换]]
6. [[fundamentals/linear-algebra/norm|范数]]
7. [[fundamentals/linear-algebra/rank-low-rank|秩与低秩近似]]
8. [[fundamentals/linear-algebra/numerical-stability|数值稳定性]]

## 与大模型主线的关系

- [[architecture/attention/attention|Attention 机制]] 依赖 Q/K/V 投影、点积相似度、矩阵乘法和 shape 变换。
- [[architecture/transformer/transformer|Transformer]] 可以理解为张量表示、线性变换、非线性变换和残差结构的组合。
- [[training/distributed-training/fsdp|FSDP 分布式训练]] 需要理解参数、梯度、优化器状态和激活张量的形状与切分。
- [[inference/quantization/quantization|量化]] 依赖数值范围、范数、误差和低精度表示。
- [[application/rag/rag|RAG]] 的向量检索依赖 embedding 空间、相似度度量和向量归一化。

## 常见误解

- 误解：线性代数只是训练前要补的数学课。
  - 正确理解：它贯穿模型结构、训练、推理和应用系统，是读懂大模型计算过程的基础语言。
- 误解：掌握公式就等于理解线性代数在模型里的作用。
  - 正确理解：更重要的是能把公式、shape、代码和模型模块对应起来。
- 误解：每个线性代数概念都需要深入证明。
  - 正确理解：这个知识库优先服务大模型理解，证明只在能澄清概念时补充。
