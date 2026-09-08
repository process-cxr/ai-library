---
title: Negative Log-Likelihood
created: 2025-12-27
published: 2025-12-27
modified: 2026-09-07
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

Negative Log-Likelihood，NLL，是模型对已观测数据所赋 likelihood 的负对数。模型给观测数据的概率越高，NLL 越小；概率越低，NLL 越大。它是 [[fundamentals/probability/maximum-likelihood|Maximum Likelihood Estimation]] 的最小化形式，也是语言模型 token-level loss 的直接统计解释。

## 背景与问题

最大化概率乘积在数值上不方便，多个小概率相乘容易下溢。因此训练中通常最大化 log likelihood，或等价地最小化 negative log likelihood。

## 定义与记号

给定样本 $x$，模型概率为 $p_\theta(x)$，负对数似然为：

$$
\operatorname{NLL}(x)=-\log p_\theta(x)
$$

对于自回归 token 序列 $x_1,\ldots,x_T$：

$$
\operatorname{NLL}(x_{1:T})
=-\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t})
$$

这里的 sequence NLL 是各 token NLL 之和，因此会随序列长度自然增大。跨不同长度样本比较时，通常使用有效 token 上的平均 NLL。

## 与 Cross Entropy 的边界

Cross Entropy 是数据分布 $p$ 下对模型负对数概率的期望：

$$
H(p,q)=\mathbb{E}_{x\sim p}[-\log q(x)]
$$

NLL 则是在具体观测样本上计算 $-\log q(x)$。训练集平均 NLL 是 Cross Entropy 的 empirical estimate；对于 categorical output 和 one-hot target，单样本 Cross Entropy 与 target NLL 数值相同。

因此，更准确的关系是：

```text
Maximum Likelihood：参数估计原则
  -> minimize dataset NLL
  -> empirical Cross Entropy objective
```

## 直观解释

负对数似然就是“真实样本对模型来说有多意外”。模型越相信真实样本，NLL 越低；模型越不相信真实样本，NLL 越高。

## 基本性质

- NLL 是观测结果在模型分布下的 self-information。
- 对 one-hot categorical target，target NLL 与 Cross Entropy 数值相同。
- 使用 log 可以把概率乘积变成求和，提升数值稳定性。
- 总 NLL、token-average NLL 和 sequence-average NLL 的聚合含义不同。
- loss mask 会决定哪些 target tokens 被纳入 NLL。

## 示例

如果模型给真实 token 的概率为 $p_\theta(y\mid x)=0.8$，则 $\operatorname{NLL}=-\log0.8\approx0.223$。如果概率降至 $0.01$，NLL 增至约 $4.605$。

## 常见误解

- **NLL 和 Cross Entropy 完全相同。** 两者在 one-hot categorical target 下数值相同，但一个描述样本 loss，一个描述分布期望。
- **总 NLL 可以直接比较不同长度文本。** 长度会影响总和，通常需要看平均 token NLL、bits per byte 或 Perplexity。
- **NLL 低表示模型所有能力都强。** NLL 主要衡量目标数据上的概率预测，不直接覆盖 instruction following、tool use 或 reasoning。

## 相关概念

- [[fundamentals/probability/maximum-likelihood|最大似然估计]]：NLL 是最大似然的最小化形式。
- [[fundamentals/information-theory/cross-entropy|交叉熵]]：one-hot 标签下与 NLL 等价。
- [[fundamentals/information-theory/perplexity|困惑度]]：平均 NLL 的指数形式。
