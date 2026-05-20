---
title: Softmax
type: topic
status: growing
area: fundamentals
tags:
  - neural-network
  - probability
  - logits
aliases:
  - Softmax Function
---

## 概念界定

Softmax 是把一组实数 logits 转换为概率分布的函数。它保证输出非负且总和为 1，是分类、next-token prediction 和 Attention 权重计算中的核心函数。

## 背景与问题

神经网络最后一层通常输出未归一化的分数 logits，而概率分布需要满足非负和总和为 1。Softmax 提供了从任意实数向量到概率分布的转换方式。

## 结构与机制

给定 logits `z`，softmax 定义为：

```text
softmax(z_i) = exp(z_i) / Σ_j exp(z_j)
```

输出满足：

```text
softmax(z_i) >= 0
Σ_i softmax(z_i) = 1
```

带 temperature 的形式：

```text
p_i = softmax(z_i / T)
```

## 直观解释

Softmax 会把较大的 logit 转成更高概率，但不是简单线性缩放。指数函数会放大 logit 差异，因此最高 logit 往往会获得更大概率质量。

## 基本性质

- softmax 对所有 logits 同时加同一个常数不变。
- logits 差异越大，softmax 分布越尖锐。
- temperature 越低，分布越尖锐；temperature 越高，分布越平坦。
- softmax 需要注意数值稳定性，通常会先减去最大 logit。

## 示例

next-token prediction 中：

```text
hidden state -> lm head -> logits -> softmax -> token probabilities
```

Attention 中：

```text
scores = QK^T / sqrt(d_head)
weights = softmax(scores + mask)
```

这里 softmax 把 attention scores 转为对 value 的加权权重。

## 常见误解

- 误解：softmax 输出的概率一定是可靠置信度。
  - 正确理解：softmax 概率可能未校准，不一定等于真实正确率。
- 误解：softmax 会保留 logits 的绝对大小。
  - 正确理解：softmax 主要由相对差异决定，对整体平移不敏感。
- 误解：temperature 改变了模型参数。
  - 正确理解：temperature 只在推理时改变 logits 到概率的转换。

## 相关概念

- [[fundamentals/probability/distribution|概率分布]] — softmax 输出的是离散分布。
- [[fundamentals/probability/temperature-topk-topp|Temperature、Top-k 与 Top-p]] — temperature 直接作用于 softmax。
- [[fundamentals/linear-algebra/numerical-stability|数值稳定性]] — softmax 需要稳定实现。
- [[architecture/attention/attention|Attention 机制]] — attention weights 来自 softmax。
