---
title: 神经网络基础
type: topic
status: growing
area: fundamentals
tags:
  - neural-network
  - deep-learning
  - transformer
aliases:
  - Neural Network Basics
---

## 概念界定

神经网络基础连接数学概念与 Transformer 实现，关注大模型中反复出现的基础模块：embedding、线性层、激活函数、归一化、softmax、残差连接、dropout 和前馈网络等。

这一节不写传统神经网络完整教材，而是围绕理解大模型架构、训练稳定性和推理行为所必需的基础组件组织。

## 背景与问题

Transformer 看起来是复杂架构，但拆开后仍然由一组基础神经网络模块组成。Token 先通过 embedding 变成向量，hidden states 在多层线性变换、attention、MLP、归一化和残差连接中更新，最后通过输出头和 softmax 变成下一个 token 的概率分布。

理解这些模块可以回答：

- token 如何变成连续向量表示？
- 线性层和 MLP 如何改变 hidden state？
- 为什么需要非线性激活函数？
- LayerNorm、RMSNorm 为什么能稳定深层训练？
- Residual connection 为什么对深层网络重要？
- Softmax 如何把 logits 变成概率分布？
- Dropout、参数共享、输出头等设计在大模型里扮演什么角色？

## 知识结构

### 表示与输出

- [[fundamentals/neural-network-basics/embedding|Embedding]] — token、position 和 vocabulary 的向量表示。
- [[fundamentals/neural-network-basics/logits-output-head|Logits 与输出头]] — hidden state 如何映射到词表分布。
- [[fundamentals/neural-network-basics/softmax|Softmax]] — logits 到概率分布的转换。

### 基础模块

- [[fundamentals/neural-network-basics/linear-layer|线性层]] — 神经网络中最基本的可学习投影。
- [[fundamentals/neural-network-basics/activation-functions|激活函数]] — ReLU、GELU、SiLU、SwiGLU 与非线性表达。
- [[fundamentals/neural-network-basics/feed-forward-network|前馈网络]] — Transformer MLP / FFN 模块。

### 深层稳定性

- [[fundamentals/neural-network-basics/normalization|归一化]] — LayerNorm、RMSNorm、Pre-Norm 与 Post-Norm。
- [[fundamentals/neural-network-basics/residual-connection|残差连接]] — 深层网络中的信息通路与梯度通路。
- [[fundamentals/neural-network-basics/dropout|Dropout]] — 随机失活、正则化与大模型中的使用变化。

### 训练与参数组织

- [[fundamentals/neural-network-basics/parameter-sharing-tying|参数共享与权重绑定]] — embedding 和 lm head 的权重绑定。
- [[fundamentals/neural-network-basics/initialization|参数初始化]] — 初始化尺度与训练稳定性。

## 推荐阅读顺序

1. [[fundamentals/neural-network-basics/embedding|Embedding]]
2. [[fundamentals/neural-network-basics/linear-layer|线性层]]
3. [[fundamentals/neural-network-basics/activation-functions|激活函数]]
4. [[fundamentals/neural-network-basics/feed-forward-network|前馈网络]]
5. [[fundamentals/neural-network-basics/normalization|归一化]]
6. [[fundamentals/neural-network-basics/residual-connection|残差连接]]
7. [[fundamentals/neural-network-basics/softmax|Softmax]]
8. [[fundamentals/neural-network-basics/logits-output-head|Logits 与输出头]]

## 与大模型主线的关系

- [[architecture/transformer/transformer|Transformer]] 由 attention、MLP、normalization、residual 等基础模块堆叠而成。
- [[architecture/model-families/llama|LLaMA 系列]] 使用 RMSNorm、SwiGLU、RoPE 等特定模块组合。
- [[training/pretraining/pretraining|预训练]] 依赖输出头、softmax 和交叉熵构成 next-token prediction 目标。
- [[inference/quantization/quantization|量化]] 会影响线性层、embedding、激活和输出头的数值表示。
- [[application/rag/rag|RAG]] 依赖 embedding 和相似度检索连接外部知识。

## 常见误解

- 误解：Transformer 已经替代了传统神经网络基础。
  - 正确理解：Transformer 仍然由线性层、激活函数、归一化、残差连接和 softmax 等基础模块构成。
- 误解：Embedding 就等于词的最终语义。
  - 正确理解：Embedding 是输入表示，深层 hidden state 才是上下文相关表示。
- 误解：归一化和残差连接只是工程技巧。
  - 正确理解：它们是深层网络可训练性的关键结构。
