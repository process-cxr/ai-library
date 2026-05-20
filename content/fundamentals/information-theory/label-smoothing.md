---
title: Label Smoothing
created: 2026-01-25
published: 2026-01-25
modified: 2026-02-01
type: topic
status: seed
area: fundamentals
tags:
  - information-theory
  - loss-function
  - regularization
aliases:
  - 标签平滑
---

## 概念界定

Label Smoothing 是一种目标分布平滑方法：不再把真实标签视为完全 one-hot，而是给非真实类别分配少量概率质量。它常用于缓解模型过度自信。

## 背景与问题

标准交叉熵训练中，真实标签通常是 one-hot 分布。这会鼓励模型把真实类别概率推到接近 1，把其他类别概率推到接近 0。对于分类任务，这可能导致模型过度自信，影响泛化和校准。

## 定义与记号

假设类别数为 `K`，平滑系数为 `ε`。原本 one-hot 标签：

```text
y_true = 1, y_other = 0
```

平滑后可写作：

```text
y_true = 1 - ε
y_other = ε / (K - 1)
```

也有实现会把 `ε` 分配到所有类别上。

## 直观解释

Label Smoothing 相当于告诉模型：即使正确答案是某个类别，也不要对它过度自信。这会让目标分布更平滑，降低模型把概率全部压到单一类别上的倾向。

## 基本性质

- 可以缓解过度自信。
- 可能改善泛化和校准。
- 对需要极高置信度区分的任务，过强平滑可能损害性能。
- 在大语言模型预训练中是否使用、如何使用，需要结合具体训练配方。

## 常见误解

- 误解：Label Smoothing 会改变真实答案。
  - 正确理解：它改变训练目标分布，不是说真实标签不正确。
- 误解：平滑越强越好。
  - 正确理解：过强平滑会削弱监督信号。

## 相关概念

- [[fundamentals/information-theory/cross-entropy|交叉熵]] — Label Smoothing 修改交叉熵中的目标分布。
- [[fundamentals/probability/calibration-uncertainty|校准与不确定性]] — 平滑常与校准问题相关。
