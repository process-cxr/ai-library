---
title: SGD 与 Momentum
type: topic
status: growing
area: fundamentals
tags:
  - optimization
  - sgd
  - momentum
aliases:
  - SGD
  - Momentum
---

## 概念界定

SGD 是使用随机 mini-batch 梯度估计更新参数的方法，Momentum 则在更新中累积历史梯度方向，使优化过程更平滑、更具惯性。

## 背景与问题

完整数据集上的梯度计算成本很高，因此深度学习通常使用 mini-batch 估计梯度。SGD 引入了随机性，计算更便宜，但梯度噪声较大。Momentum 通过累积历史方向缓解震荡，加速沿一致方向的移动。

## 定义与记号

SGD 更新：

```text
g_t = ∇_θ L_batch(θ_t)
θ_{t+1} = θ_t - η g_t
```

Momentum 更新：

```text
v_t = β v_{t-1} + g_t
θ_{t+1} = θ_t - η v_t
```

其中 `β` 控制历史梯度保留程度。

## 直观解释

SGD 像每次根据一小批样本判断下降方向，因此方向会抖动。Momentum 像给优化过程加上惯性：如果多个 batch 的梯度方向大致一致，就沿这个方向走得更稳定。

## 基本性质

- SGD 的随机性可能帮助逃离某些尖锐区域。
- Momentum 可以降低梯度方向震荡。
- Momentum 需要额外保存速度状态。
- 大模型训练更常使用 Adam/AdamW，但 SGD 和 Momentum 是理解优化器的基础。

## 示例

如果某个方向上梯度长期一致，Momentum 会累积该方向的速度，使更新更快；如果某个方向上梯度来回震荡，Momentum 会部分抵消震荡。

## 常见误解

- 误解：SGD 的随机性只是缺点。
  - 正确理解：随机性也可能影响泛化，并帮助探索参数空间。
- 误解：Momentum 改变了优化目标。
  - 正确理解：它改变更新路径，不改变原始 loss。
- 误解：Adam 完全不需要理解 SGD。
  - 正确理解：Adam 可以看作在 Momentum 基础上加入自适应缩放。

## 相关概念

- [[gradient-descent|梯度下降]] — SGD 是随机近似版本。
- [[adam|Adam]] — 同时使用动量和自适应二阶矩。
- [[batch-size-gradient-noise|Batch Size 与梯度噪声]] — mini-batch 影响梯度估计噪声。
