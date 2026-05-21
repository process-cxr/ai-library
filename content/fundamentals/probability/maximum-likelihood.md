---
title: Maximum Likelihood Estimation
created: 2025-12-21
published: 2025-12-21
modified: 2025-12-21
type: topic
status: growing
area: fundamentals
tags:
  - math
  - probability
  - training-objective
aliases:
  - Maximum Likelihood Estimation
  - MLE
---

## 概念界定

最大似然估计是一种参数估计原则：选择使观测数据出现概率最大的模型参数。在语言模型中，预训练通常可以理解为最大化训练语料中 token 序列的似然。

## 背景与问题

模型有大量参数 `θ`，我们需要一个原则来判断哪些参数更好。最大似然的想法是：如果一组参数能让训练数据更可能出现，那么这组参数就更符合数据分布。

## 定义与记号

给定数据集：

```text
D = {x_1, x_2, ..., x_N}
```

最大似然目标：

```text
θ* = argmax_θ Π_i p_θ(x_i)
```

通常取 log：

```text
θ* = argmax_θ Σ_i log p_θ(x_i)
```

等价地，训练中常最小化负 log likelihood：

```text
L(θ) = - Σ_i log p_θ(x_i)
```

## 直观解释

最大似然就是让模型尽量“觉得训练数据合理”。如果训练数据中某个 token 在某个上下文后出现，模型就应该提高这个 token 在该上下文下的概率。

## 基本性质

- log likelihood 把概率乘积转化为求和，数值更稳定、优化更方便。
- 最大似然依赖训练数据的经验分布。
- 对分类分布而言，最大似然常对应交叉熵损失。
- 最大似然不自动保证输出符合人类偏好，因此后续还需要 SFT、RLHF 或 DPO 等对齐方法。

## 示例

自回归语言模型中：

```text
p_θ(x_1, ..., x_T) = Π_t p_θ(x_t | x_<t)
```

负 log likelihood：

```text
L(θ) = - Σ_t log p_θ(x_t | x_<t)
```

这就是 next-token prediction 训练目标的概率解释。

## 常见误解

- 误解：最大似然就是让模型记住训练集。
  - 正确理解：目标是提高训练数据分布下的概率，但模型是否泛化取决于数据、模型和正则化等因素。
- 误解：最大似然训练出的模型一定符合人类偏好。
  - 正确理解：它主要学习数据分布，不直接优化有用性、安全性或偏好。
- 误解：概率越高的回答一定越好。
  - 正确理解：高概率可能意味着常见或保守，不一定最符合任务目标。

## 相关概念

- [[fundamentals/probability/chain-rule|概率链式法则]] — 序列似然的分解基础。
- [[fundamentals/information-theory/cross-entropy|交叉熵]] — 分类分布上的负 log likelihood。
- [[training/pretraining/pretraining|预训练]] — 最大似然在语言模型中的主要应用。
- [[training/post-training/rlhf|RLHF]] — 在最大似然后进一步引入偏好优化。
