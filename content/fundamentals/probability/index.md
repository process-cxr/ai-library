---
title: Probability
created: 2025-12-13
published: 2025-12-13
modified: 2025-12-13
type: topic
status: growing
area: fundamentals
tags:
  - math
  - probability
  - language-modeling
aliases:
  - Probability
---

## 概念界定

概率论研究不确定事件、随机变量、概率分布及其运算规则。在大模型和 NLP 中，概率论是理解语言建模、next-token prediction、采样生成、评估指标和不确定性的基础。

这一节不追求完整覆盖概率论课程，而是围绕语言模型中反复出现的概率概念建立基础。

## 背景与问题

语言模型本质上是在建模 token 序列的概率分布。给定上下文，模型输出的不是唯一答案，而是下一个 token 的概率分布。训练时，模型通过数据估计分布；推理时，模型从分布中选择或采样 token；评估时，我们用交叉熵、困惑度等指标衡量模型分布与真实数据分布的差异。

因此，概率论需要回答几个核心问题：

- token、序列和预测结果如何被看作随机变量？
- 语言模型为什么写成条件概率 `p(x_t | x_<t)`？
- softmax 输出的概率分布到底表示什么？
- 采样、temperature、top-k、top-p 如何改变生成行为？
- 训练 loss 和评估指标为什么常写成期望形式？
- 模型输出的概率能否直接等同于“置信度”？

## 知识结构

### 基本对象

- [[fundamentals/probability/random-variable|随机变量]] — token、序列、标签和模型输出的随机变量视角。
- [[fundamentals/probability/distribution|概率分布]] — 离散分布、分类分布、模型预测分布。
- [[fundamentals/probability/joint-marginal|联合分布与边缘分布]] — 序列概率、变量关系和边缘化。

### 条件建模

- [[fundamentals/probability/conditional-probability|条件概率]] — 自回归语言模型的核心形式。
- [[fundamentals/probability/chain-rule|概率链式法则]] — 将序列概率分解为逐 token 条件概率。
- [[fundamentals/probability/bayes-rule|贝叶斯公式]] — 后验、先验和证据的基本关系。

### 统计量与估计

- [[fundamentals/probability/expectation|期望与方差]] — loss、风险、评估指标和采样估计的基础。
- [[fundamentals/probability/empirical-distribution|经验分布]] — 训练数据如何近似真实分布。
- [[fundamentals/probability/maximum-likelihood|最大似然估计]] — 语言模型训练目标的概率解释。
- [[fundamentals/probability/monte-carlo|蒙特卡罗方法]] — 用随机样本近似期望、概率和评估指标。

### 生成与不确定性

- [[fundamentals/probability/sampling|采样]] — 从概率分布中生成 token。
- [[fundamentals/probability/temperature-topk-topp|Temperature、Top-k 与 Top-p]] — 常见解码采样控制方法。
- [[fundamentals/probability/calibration-uncertainty|校准与不确定性]] — 模型概率、置信度和可靠性的区别。

## 推荐阅读顺序

1. [[fundamentals/probability/random-variable|随机变量]]
2. [[fundamentals/probability/distribution|概率分布]]
3. [[fundamentals/probability/conditional-probability|条件概率]]
4. [[fundamentals/probability/chain-rule|概率链式法则]]
5. [[fundamentals/probability/expectation|期望与方差]]
6. [[fundamentals/probability/maximum-likelihood|最大似然估计]]
7. [[fundamentals/probability/monte-carlo|蒙特卡罗方法]]
8. [[fundamentals/probability/sampling|采样]]
9. [[fundamentals/probability/calibration-uncertainty|校准与不确定性]]

## 与大模型主线的关系

- [[training/pretraining/pretraining|预训练]] 通常可以理解为最大化训练语料中 token 序列的似然。
- [[fundamentals/information-theory/cross-entropy|交叉熵]] 是最大似然训练在分类分布上的常见损失形式。
- [[fundamentals/information-theory/kl-divergence|KL 散度]] 衡量两个概率分布的差异，在对齐和分布约束中常出现。
- [[inference/kv-cache-and-memory/kv-cache|KV Cache]] 服务于自回归条件概率分解下的高效逐 token 生成。
- [[application/rag/rag|RAG]] 可以看作在外部证据条件下改变模型的生成分布。

## 常见误解

- 误解：语言模型输出的是确定答案。
  - 正确理解：模型输出的是下一个 token 的概率分布，答案是解码策略从分布中得到的结果。
- 误解：概率最高的 token 总是最好。
  - 正确理解：贪心选择可能稳定但缺乏多样性，采样策略会改变生成行为。
- 误解：模型概率等于真实置信度。
  - 正确理解：模型可能过度自信或校准不足，概率需要结合校准和任务场景解释。
