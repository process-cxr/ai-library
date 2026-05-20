---
title: 负对数似然
type: topic
status: growing
area: fundamentals
tags:
  - math
  - information-theory
  - loss-function
aliases:
  - Negative Log Likelihood
  - NLL
---

## 概念界定

负对数似然是最大似然估计中的常见损失形式，表示模型给真实观测数据分配的概率有多低。概率越高，负对数似然越小；概率越低，负对数似然越大。

## 背景与问题

最大化概率乘积在数值上不方便，多个小概率相乘容易下溢。因此训练中通常最大化 log likelihood，或等价地最小化 negative log likelihood。

## 定义与记号

给定样本 `x`，模型概率为 `p_θ(x)`，负对数似然为：

```text
NLL(x) = -log p_θ(x)
```

对于序列：

```text
NLL(x_1, ..., x_T) = -Σ_t log p_θ(x_t | x_<t)
```

## 直观解释

负对数似然就是“真实样本对模型来说有多意外”。模型越相信真实样本，NLL 越低；模型越不相信真实样本，NLL 越高。

## 基本性质

- NLL 是自信息在模型分布下的形式。
- 对 one-hot 分类任务，NLL 与交叉熵损失形式相同。
- 使用 log 可以把概率乘积变成求和，提升数值稳定性。
- 序列越长，总 NLL 通常越大，因此常看平均 NLL。

## 示例

如果模型给真实 token 的概率为：

```text
p_θ(y | x) = 0.8
```

则：

```text
NLL = -log 0.8
```

如果概率为：

```text
p_θ(y | x) = 0.01
```

则 NLL 明显更大。

## 常见误解

- 误解：NLL 和交叉熵完全无关。
  - 正确理解：one-hot 标签下，交叉熵就是真实类别的 NLL。
- 误解：总 NLL 可以直接比较不同长度文本。
  - 正确理解：长度不同会影响总和，通常需要看平均 token NLL 或困惑度。
- 误解：NLL 低表示模型所有能力都强。
  - 正确理解：NLL 主要衡量概率预测质量，不直接覆盖指令跟随、工具使用或推理能力。

## 相关概念

- [[fundamentals/probability/maximum-likelihood|最大似然估计]] — NLL 是最大似然的最小化形式。
- [[fundamentals/information-theory/cross-entropy|交叉熵]] — one-hot 标签下与 NLL 等价。
- [[fundamentals/information-theory/perplexity|困惑度]] — 平均 NLL 的指数形式。
