---
title: Expectation and Variance
created: 2025-12-20
published: 2025-12-20
modified: 2025-12-20
type: topic
status: growing
area: fundamentals
tags:
  - math
  - probability
  - statistics
aliases:
  - Expectation
  - Variance
---

## 概念界定

期望描述随机变量在概率分布下的平均取值，方差描述随机变量围绕期望的波动程度。在机器学习中，训练目标、风险、评估指标和采样估计常以期望形式表达。

## 背景与问题

模型训练并不是只优化单个样本，而是希望在数据分布上整体表现更好。理论上，我们希望最小化真实数据分布下的期望损失；实践中，只能用训练集上的经验平均近似这个期望。

## 定义与记号

离散随机变量的期望：

```text
E[X] = Σ_x x p(x)
```

函数的期望：

```text
E[f(X)] = Σ_x f(x) p(x)
```

方差：

```text
Var(X) = E[(X - E[X])^2]
```

机器学习中的期望风险：

```text
R(θ) = E_{(x,y)~P_data}[L(f_θ(x), y)]
```

## 直观解释

期望可以理解为“按概率加权的平均”。如果某些样本更常出现，它们对期望损失的影响就更大。方差则衡量结果是否稳定：方差大说明不同样本或不同采样结果之间波动明显。

## 基本性质

- 期望是线性的：`E[aX + bY] = aE[X] + bE[Y]`。
- 样本平均可以近似期望，但会受到样本数量和采样方式影响。
- 方差越大，估计通常越不稳定。
- 训练 loss 通常是对 batch 内样本损失的经验平均。

## 示例

训练集中一个 mini-batch 的平均 loss：

```text
L_batch = (1/B) Σ_i L_i
```

它可以看作对真实期望损失的随机估计。batch size 越大，估计通常越稳定，但计算和显存成本也越高。

## 常见误解

- 误解：训练集平均 loss 就是真实期望 loss。
  - 正确理解：训练集平均只是对真实数据分布期望的经验近似。
- 误解：batch loss 下降就一定代表泛化变好。
  - 正确理解：batch loss 只是采样估计，可能受数据分布和过拟合影响。
- 误解：方差只和数据有关。
  - 正确理解：采样策略、batch size、模型随机性和解码策略都会影响方差。

## 相关概念

- [[fundamentals/probability/empirical-distribution|经验分布]] — 用数据集近似真实分布。
- [[fundamentals/probability/sampling|采样]] — 用样本估计分布性质。
- [[training/pretraining/pretraining|预训练]] — 训练目标通常是经验平均 loss。
