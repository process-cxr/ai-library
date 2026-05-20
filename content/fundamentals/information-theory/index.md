---
title: 信息论
type: topic
status: growing
area: fundamentals
tags:
  - math
  - information-theory
  - language-modeling
aliases:
  - Information Theory
---

## 概念界定

信息论研究信息量、不确定性、编码长度、分布差异和信息传递。对大模型而言，信息论是理解语言模型训练目标、交叉熵损失、困惑度评估、KL 约束、对齐方法和表示压缩的重要基础。

这一节不追求完整覆盖通信理论，而是围绕 NLP、深度学习和大模型中反复出现的信息论概念组织。

## 背景与问题

语言模型输出的是一个 token 概率分布。训练时，我们希望模型分布接近真实数据分布；评估时，我们希望衡量模型对真实 token 的平均“不意外程度”；对齐时，我们又常常希望新策略不要偏离参考模型太远。这些问题都可以用信息论语言表达。

信息论在大模型中主要回答：

- 一个事件出现带来的信息量如何度量？
- 一个分布的不确定性如何度量？
- 交叉熵为什么能作为 next-token prediction 的训练损失？
- 困惑度为什么能衡量语言模型预测能力？
- KL 散度如何刻画两个分布的差异？
- 为什么 RLHF、DPO、蒸馏和正则化中会频繁出现 KL 或分布约束？

## 知识结构

### 信息量与不确定性

- [[fundamentals/information-theory/self-information|自信息]] — 单个事件的意外程度。
- [[fundamentals/information-theory/entropy|熵]] — 一个分布的平均不确定性。
- [[fundamentals/information-theory/joint-conditional-entropy|联合熵与条件熵]] — 多变量和条件下的不确定性。
- [[fundamentals/information-theory/mutual-information|互信息]] — 一个变量对另一个变量提供了多少信息。

### 训练目标与评估

- [[fundamentals/information-theory/cross-entropy|交叉熵]] — 用真实分布评估模型分布的平均编码成本。
- [[fundamentals/information-theory/perplexity|困惑度]] — 交叉熵的指数形式，用于语言模型评估。
- [[fundamentals/information-theory/negative-log-likelihood|负对数似然]] — 最大似然训练中的常见损失形式。

### 分布差异与约束

- [[fundamentals/information-theory/kl-divergence|KL 散度]] — 衡量两个分布差异的非对称量。
- [[fundamentals/information-theory/js-divergence|JS 散度]] — 对称化、平滑化的分布差异度量。
- [[fundamentals/information-theory/cross-entropy-kl-relation|交叉熵与 KL 的关系]] — 解释为什么最小化交叉熵等价于靠近真实分布。

### 大模型相关扩展

- [[fundamentals/information-theory/label-smoothing|Label Smoothing]] — 用目标分布平滑缓解过度自信。
- [[fundamentals/information-theory/knowledge-distillation|知识蒸馏的信息论视角]] — 用教师分布指导学生模型。
- [[fundamentals/information-theory/rate-distortion|率失真思想]] — 压缩、量化和信息保留的基础视角。

## 推荐阅读顺序

1. [[fundamentals/information-theory/self-information|自信息]]
2. [[fundamentals/information-theory/entropy|熵]]
3. [[fundamentals/information-theory/cross-entropy|交叉熵]]
4. [[fundamentals/information-theory/negative-log-likelihood|负对数似然]]
5. [[fundamentals/information-theory/perplexity|困惑度]]
6. [[fundamentals/information-theory/kl-divergence|KL 散度]]
7. [[fundamentals/information-theory/cross-entropy-kl-relation|交叉熵与 KL 的关系]]
8. [[fundamentals/information-theory/mutual-information|互信息]]
9. [[fundamentals/information-theory/knowledge-distillation|知识蒸馏的信息论视角]]

## 与大模型主线的关系

- [[training/pretraining/pretraining|预训练]] 通常通过最小化 next-token 交叉熵来训练语言模型。
- [[fundamentals/probability/maximum-likelihood|最大似然估计]] 与负对数似然、交叉熵密切相关。
- [[training/post-training/rlhf|RLHF]] 中常通过 KL 惩罚限制策略模型偏离参考模型。
- [[training/post-training/dpo|DPO]] 可以从偏好分布和参考模型约束的角度理解。
- [[inference/quantization/quantization|量化]] 和 [[inference/compression/model-compression|模型压缩]] 可以借助率失真思想理解信息保留与压缩代价。
- [[application/evaluation/evaluation|评测与 Benchmark]] 中的困惑度只衡量预测分布，不等同于任务能力全貌。

## 常见误解

- 误解：熵越低，模型越好。
  - 正确理解：低熵表示分布更确定，但如果模型确定地给出错误 token，效果并不好。
- 误解：交叉熵只是分类任务的一个经验 loss。
  - 正确理解：它对应真实分布下使用模型分布编码的平均代价，也与最大似然等价。
- 误解：KL 散度是距离。
  - 正确理解：KL 非对称，且不满足距离度量的所有性质。
- 误解：困惑度低就代表模型应用能力强。
  - 正确理解：困惑度主要衡量 next-token prediction，不直接等价于推理、指令跟随或工具使用能力。
