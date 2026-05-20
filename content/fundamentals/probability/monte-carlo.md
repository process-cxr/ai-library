---
title: Monte Carlo Method
created: 2026-01-22
published: 2026-01-22
modified: 2026-01-29
type: topic
status: growing
area: fundamentals
tags:
  - math
  - probability
  - monte-carlo
  - sampling
aliases:
  - Monte Carlo
  - Monte Carlo Method
---

## 概念界定

蒙特卡罗方法是一类用随机采样近似计算期望、积分、概率或复杂系统行为的方法。它的核心思想是：当精确计算很难时，可以通过大量随机样本的平均结果逼近目标量。

## 背景与问题

在机器学习和大模型中，很多目标都可以写成期望形式，但真实分布往往无法完整枚举。例如训练 loss 是数据分布上的期望，强化学习中的策略回报是轨迹分布上的期望，评测指标也常需要在样本集合上估计。蒙特卡罗方法提供了“用样本估计整体”的基础思想。

## 定义与记号

如果我们希望计算：

```text
E[f(X)]
```

但无法对完整分布精确求和或积分，可以从分布中采样：

```text
x_1, x_2, ..., x_N ~ p(X)
```

然后用样本平均近似期望：

```text
E[f(X)] ≈ (1/N) Σ_i f(x_i)
```

这就是最基本的蒙特卡罗估计。

## 直观解释

蒙特卡罗方法可以理解为“多抽几次，看平均结果”。如果样本足够多，并且采样方式正确，样本平均通常会接近真实期望。

例如想估计一个模型在真实用户问题上的平均表现，但无法枚举所有问题，就可以抽取一批代表性问题进行评测，用平均分近似整体表现。

## 基本性质

- 样本数越多，估计通常越稳定。
- 蒙特卡罗估计的误差通常以 `1 / sqrt(N)` 的速度下降。
- 样本是否来自正确分布非常关键；采样偏差会导致估计偏差。
- 方差越大，需要越多样本才能得到稳定估计。
- 蒙特卡罗方法常用于无法解析计算或枚举空间过大的场景。

## 示例

训练时的 mini-batch loss 可以看作对总体期望 loss 的随机估计：

```text
L(θ) = E_{x~p_data}[loss_θ(x)]
```

实践中用一个 batch 近似：

```text
L_batch(θ) = (1/B) Σ_i loss_θ(x_i)
```

语言模型生成时，从 token 分布中多次采样，也可以用于观察模型输出的多样性和不确定性。

## 和大模型的关系

- [[training/pretraining/pretraining|预训练]]：mini-batch 训练可以看作用样本估计数据分布上的期望损失。
- [[training/post-training/rlhf|RLHF]]：策略回报、偏好样本和轨迹估计中常出现蒙特卡罗思想。
- [[application/evaluation/evaluation|评测与 Benchmark]]：有限测试集上的平均指标是对目标任务分布表现的估计。
- [[fundamentals/probability/sampling|采样]]：蒙特卡罗估计依赖从目标分布中采样。

## 常见误解

- 误解：蒙特卡罗方法就是随机试一试。
  - 正确理解：它是有明确估计目标和收敛性质的随机采样方法。
- 误解：样本越多结果一定越准确。
  - 正确理解：样本数增加能降低随机误差，但无法消除采样分布错误带来的系统偏差。
- 误解：蒙特卡罗只和强化学习有关。
  - 正确理解：只要用随机样本近似期望、概率或指标，都可以看到蒙特卡罗思想。

## 相关概念

- [[fundamentals/probability/expectation|期望与方差]] — 蒙特卡罗估计通常是在估计期望。
- [[fundamentals/probability/sampling|采样]] — 蒙特卡罗方法依赖采样。
- [[fundamentals/probability/empirical-distribution|经验分布]] — 有限样本形成经验近似。
- [[training/post-training/rlhf|RLHF]] — 强化学习和偏好优化中常见采样估计。
