---
title: Muon Optimizer
created: 2026-10-08
published: 2026-10-08
modified: 2026-10-08
type: topic
status: growing
area: training
tags:
  - training-optimization
  - optimizer
  - muon
  - matrix-optimization
---

Muon（MomentUm Orthogonalized by Newton-Schulz）是一种主要面向二维 hidden-layer weight matrices 的 optimizer。它先积累 momentum，再用 Newton-Schulz iteration 近似正交化 momentum matrix，最后更新参数。

## Muon 的设计目的

Muon 不是为了把 AdamW 的每个标量更新换成另一条公式，而是想解决一个更具体的问题：**Transformer 中的很多参数天然是矩阵，但常规 AdamW 主要按元素独立地调整它们。**

一个 $H\times4H$ 的 MLP projection、一个 $H\times H$ 的 attention projection，本质上都是把一个 hidden space 映射到另一个 hidden space 的线性算子。矩阵的行列方向、奇异值谱和整体几何结构，会影响信息沿哪些方向传播；如果更新长期集中在少数最大奇异值方向，模型可能反复强化已有主方向，而较弱但有用的方向得不到足够探索。

Muon 的设计目标可以概括为三点：

1. **利用矩阵结构，而不是只看单个参数。** 对矩阵级 momentum 做统一的几何变换，让更新方向反映整个 weight matrix 的结构。
2. **抑制少数主方向对更新的支配。** Newton-Schulz 正交化近似把 $U\Sigma V^T$ 变成 $UV^T$，保留方向信息、削弱奇异值大小差异，使更多矩阵方向获得相近的更新机会。
3. **让矩阵参数的更新更接近合适的 operator-norm 几何。** Muon 的理论直觉与 spectral norm 下的 steepest descent 有关；它试图让优化几何更符合 weight matrix 作为线性算子的角色。

所以，Muon 解决的不是“AdamW 不会下降”这个问题，而是“在矩阵参数上，怎样用更合适的方向和尺度进行长期更新”。它仍需要 learning rate、weight decay、混合精度和分布式状态管理；也不意味着所有参数都应该统一改用 Muon。

原始 Muon 主要在小规模模型和训练竞速任务中展示了潜力。要进入 LLM pre-training，还必须额外解决 weight growth、不同矩阵形状的 update RMS 不一致、完整矩阵正交化的通信成本，以及 pre-training 与 SFT optimizer mismatch。这些正是 [[sources/papers/2025-muon-is-scalable-for-llm-training|Muon is Scalable for LLM Training]] 的重点。

## 核心更新

给定矩阵梯度 $G_t$：

$$
M_t=\mu M_{t-1}+G_t,
$$

然后计算：

$$
O_t\approx\operatorname{Ortho}(M_t)=UV^T,
$$

其中 $M_t=U\Sigma V^T$ 是 SVD。参数更新为：

$$
W_t=W_{t-1}-\eta_tO_t.
$$

实际的大模型 recipe 还需要 decoupled weight decay 和按矩阵形状校准 update RMS。

## 适用参数

Muon 通常用于 attention、MLP 和 expert 的矩阵参数；embedding、LM head、RMSNorm、bias 和 vector/scalar 参数仍使用 AdamW。因此它通常是 Muon + AdamW 的混合 optimizer，而不是全参数单一 optimizer。

## 为什么需要形状校准

对形状为 $[A,B]$ 的 full-rank matrix，Muon update RMS 的理论量级约为：

$$
\operatorname{RMS}(O)\approx\frac{1}{\sqrt{\max(A,B)}}.
$$

如果不校准，$[H,H]$ attention matrix 和 $[H,4H]$ MLP matrix 的有效步长不同。Muon is Scalable for LLM Training 采用 Adjusted LR：

$$
W_t=W_{t-1}-\eta_t\left(0.2\sqrt{\max(A,B)}O_t+\lambda W_{t-1}\right).
$$

## 与 AdamW、Shampoo 的区别

- AdamW 按坐标维护一阶/二阶矩，成熟稳定，但不直接利用矩阵结构；
- Shampoo 通过左右 preconditioner 处理矩阵几何，状态和实现更复杂；
- Muon 用 momentum + Newton-Schulz 正交化近似矩阵级更新，额外状态主要是一个 momentum buffer；
- Muon 的高效证据主要集中在从头 pre-training，切换到已有 AdamW checkpoint 的 SFT/RL 仍需单独验证。

## 分布式实现

Muon 需要完整 gradient matrix，不能直接在 ZeRO-1 的局部分片上完成正交化。Distributed Muon 的典型流程是：

```text
reduce-scatter gradient
  -> local momentum update
  -> data-parallel gather full matrix
  -> Newton-Schulz
  -> keep local update partition
  -> local parameter update
  -> all-gather updated parameters
```

这会带来额外 gather 和矩阵运算，但只需要一个 momentum buffer，并且正交化可以用 BF16 执行。

## 相关来源

- [[sources/papers/2025-muon-is-scalable-for-llm-training|Muon is Scalable for LLM Training]]
- [Muon: An Optimizer for Hidden Layers in Neural Networks](https://kellerjordan.github.io/posts/muon/)
- [[fundamentals/linear-algebra/eigendecomposition-svd|SVD]]
- [[training/optimization/optimizer-state|Optimizer State]]
