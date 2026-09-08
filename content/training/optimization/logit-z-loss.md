---
title: Logit Z-Loss
created: 2026-09-07
published: 2026-09-07
modified: 2026-09-07
type: topic
status: growing
area: training
tags:
  - optimization
  - loss-function
  - numerical-stability
  - language-modeling
aliases:
  - Z-Loss
  - Logit Z Loss
---

## 目标与问题

Logit Z-Loss 是作用于 output logits 的稳定性 regularization。普通 [[fundamentals/information-theory/cross-entropy|Cross Entropy]] 只由 Softmax probability 决定，而 Softmax 对所有 logits 的共同平移不敏感：

$$
\operatorname{softmax}(z+c\mathbf{1})=\operatorname{softmax}(z)
$$

因此，CE 沿这一共同平移方向没有约束。大规模或低精度训练中，训练配方有时会额外约束 logits 的 log-partition function，避免它在不改变预测概率的方向上自由漂移。

## 定义

令：

$$
\log Z(z)=\log\sum_{j=1}^{V}e^{z_j}
$$

Logit Z-Loss 通常写为：

$$
L_z=\lambda\left(\log Z(z)\right)^2
$$

最终 objective 为：

$$
L=L_{\mathrm{task}}+L_z
$$

其中 $\lambda$ 是较小的 regularization coefficient。$L_{\mathrm{task}}$ 可以是 token Cross Entropy；Z-Loss 并不替代语言建模目标。

## 它约束的是什么

Z-Loss 直接推动 $\log Z(z)$ 接近零，并由此打破 Cross Entropy 的 common-shift invariance。它常被概括为“抑制 logits 变大”，但更准确地说，它约束的是 log-partition function，而不是对每个 logit 独立施加 magnitude penalty。

需要区分：

- 给所有 logits 加常数会改变 $\log Z$，但不改变 Softmax；
- 把 logits 乘以系数会改变相对差距、Softmax entropy 和 CE；
- Z-Loss 对 relative spread 也可能有响应，因为 spread 同样会影响 $\log Z$。

因此，它不是简单的 logit norm，也不能与 temperature scaling 等同。

## 为什么可能改善稳定性

在数值稳定的 Cross Entropy kernel 中，max subtraction 已经能够避免直接计算 $e^{z_j}$ 的多数 overflow；Z-Loss 的作用并不是替代这种稳定实现。它改变的是优化动力学：为 CE 不可辨识的 normalization direction 提供额外梯度，并限制 log-partition 的漂移。

这在大规模、Mixed Precision 或 MoE training 中可能帮助降低极端 logits 和相关数值异常的风险。但收益取决于模型、dtype、optimizer 和 loss implementation，不能把加入 Z-Loss 当成解决所有 loss spike 的通用开关。

## 使用边界

- $\lambda$ 太小可能没有可观察影响，太大则会干扰主任务分布学习。
- 训练日志应分开记录 task loss、raw Z-Loss 与加权后的 Z-Loss contribution。
- 比较实验要同时观察 task metric、held-out CE、logit statistics、gradient norm 和 numerical failures。
- vocabulary-parallel training 中，$\log Z$ 需要基于 global vocabulary 计算，不能只使用单个 TP rank 的 local logits。
- 若根因是错误数据、mask、learning rate、overflow 或 collective mismatch，Z-Loss 只能改变症状，不能替代根因排查。

## 相关概念

- [[fundamentals/information-theory/cross-entropy|Cross Entropy]]：Z-Loss 补充约束 CE 的 common-shift direction。
- [[fundamentals/neural-network-basics/softmax|Softmax]]：共同平移不变性的来源。
- [[fundamentals/neural-network-basics/logits-output-head|Logits and Output Head]]：Z-Loss 的直接作用对象。
- [[training/optimization/training-stability|Training Stability]]：Z-Loss 所服务的工程目标。
- [[training/optimization/mixed-precision|Mixed Precision Training]]：logits 与 reduction 的 dtype 会影响数值行为。
- [[training/distributed-training/tensor-parallel|Tensor Parallel]]：global vocabulary 上的 log-partition 需要 TP collective。
