---
title: Hyper-Connections and mHC
created: 2026-10-08
published: 2026-10-08
modified: 2026-10-08
type: topic
status: growing
area: architecture
tags:
  - transformer
  - residual
  - hyper-connections
  - mhc
  - training-stability
  - deepseek
aliases:
  - Manifold-Constrained Hyper-Connections
  - mHC
  - Hyper-Connections
---

## 先说它要解决什么问题

Transformer 中的 residual connection 通常写成：

$$
h_{l+1}=h_l+F_l(h_l),
$$

其中 $h_l$ 是第 $l$ 层的 hidden state，$F_l$ 可以是 attention 或 FFN。它的价值在于保留一条接近 identity 的信息和梯度路径，让每一层学习增量修正，而不是从零重建表示。

但这种结构也有一个限制：每一层基本只有一条 residual stream，子层输出以固定的加法方式回到这条 stream。随着网络变深，模型可以调整每个 block 的变换 $F_l$，却不能灵活地重组不同层、不同 residual 分支之间的信息流。

Hyper-Connections（HC，超连接）试图放宽这个限制：把 residual stream 扩展成多条，并学习这些 stream 如何混合。问题是，混合矩阵如果完全自由，连续堆叠后可能导致信号和梯度不断放大或衰减。Manifold-Constrained Hyper-Connections（mHC，流形约束超连接）就是在 HC 的自由度上加结构约束，目标是同时获得：

- 比普通 residual 更灵活的层间信息混合；
- 比自由 HC 更稳定的信号传播；
- 在很深、很大的 Transformer 中可实现的训练开销。

因此，mHC 不是 attention、MoE 或 optimizer。它改变的是 **Transformer block 之间 residual information 如何流动**。

## 从普通 Residual 到 Hyper-Connections

### 普通 residual connection

标准 residual block：

$$
h_{l+1}=h_l+F_l(h_l).
$$

Pre-Norm Transformer 中常见的形式是：

$$
h' = h + \operatorname{Attention}(\operatorname{Norm}(h)),
$$

$$
h_{out}=h' + \operatorname{FFN}(\operatorname{Norm}(h')).
$$

这里的 residual path 只有一条。每个子层接收当前 hidden state，并将输出加回原来的 hidden state。

### Hyper-Connections 的多路 residual stream

HC 把 residual state 从一个 $d$ 维向量扩展为 $n_{hc}$ 路 residual stream：

$$
X_l=\begin{bmatrix}x_{l,1}\\x_{l,2}\\\cdots\\x_{l,n_{hc}}\end{bmatrix}
\in\mathbb{R}^{n_{hc}\times d}.
$$

第 $l$ 层不再直接把同一个 $X_l$ 输入给子层，而是使用三个映射：

- $A_l$：把多路 residual stream 混合成实际 block 输入；
- $F_l$：执行 attention 或 FFN 等 Transformer 子层；
- $B_l$：在不经过当前子层的路径上混合 residual streams；
- $C_l$：把当前子层的输出写回多路 residual streams。

整体形式是：

$$
X_{l+1}=B_lX_l+C_lF_l(A_lX_l).
$$

其中：

$$
A_l\in\mathbb{R}^{1\times n_{hc}},\qquad
B_l\in\mathbb{R}^{n_{hc}\times n_{hc}},\qquad
C_l\in\mathbb{R}^{n_{hc}\times 1}.
$$

注意，$F_l$ 内部的 attention 或 FFN 仍然处理 $d$ 维 hidden representation。HC 扩展的是 residual stream 的宽度，不是把每个 attention/FFN 都复制成 $n_{hc}$ 份。因此它增加的是层间 mixing capacity，而不是简单地把 Transformer block 计算量乘以 $n_{hc}$。

可以把它理解成：

```text
普通 residual:
  one stream -> block -> add back to one stream

Hyper-Connections:
  many streams -> A_l -> block -> C_l -> mix back with B_l
```

## 为什么自由的 Hyper-Connections 可能不稳定

如果 $B_l$ 完全不受约束，那么忽略 $F_l$ 分支时，多层 residual mixing 大致包含：

$$
X_L\approx B_{L-1}B_{L-2}\cdots B_0X_0.
$$

即使每个 $B_l$ 单独看起来没有问题，矩阵乘积也可能出现：

- 某些方向被连续放大，导致 activation 或 gradient explosion；
- 某些方向被连续压小，导致 signal vanishing；
- 不同 residual streams 的尺度逐层失衡；
- forward signal 与 backward gradient 的传播条件变差。

这和普通 residual connection 的差异在于：普通 residual 的 identity path 结构非常固定，而 HC 让 residual path 本身也变成了可学习的矩阵变换。自由度更大，同时也引入了更多可能失控的谱行为。

## mHC 的核心：约束 residual mixing matrix

### Doubly stochastic matrix

mHC 将 residual mixing matrix $B_l$ 约束为 **doubly stochastic matrix（双随机矩阵）**。一个矩阵 $B\in\mathbb{R}^{n\times n}$ 如果满足：

$$
B_{ij}\geq 0,
$$

$$
B\mathbf{1}=\mathbf{1},
$$

$$
\mathbf{1}^{T}B=\mathbf{1}^{T},
$$

就称为双随机矩阵。也就是：

- 所有元素非负；
- 每一行的和为 1；
- 每一列的和也为 1。

它可以被看作多路信息的“守恒式混合”：每一路输出是输入 streams 的非负加权组合，同时整体 mixing 不会任意改变总量尺度。

### Birkhoff polytope

所有 $n\times n$ 双随机矩阵组成的集合称为 **Birkhoff polytope（Birkhoff 多面体）**。mHC 令：

$$
B_l\in\mathcal{M},
$$

其中：

$$
\mathcal{M}=\{M\in\mathbb{R}^{n\times n}
\mid M\mathbf{1}=\mathbf{1},\;
\mathbf{1}^{T}M=\mathbf{1}^{T},\;M\geq 0\}.
$$

这不是说 $B_l$ 必须是 permutation matrix。Permutation matrix 是双随机矩阵的一种极端情况；一般双随机矩阵允许多路 residual stream 做平滑混合。

### 为什么这个约束有助于稳定

双随机约束带来三个直观性质：

1. 非负性避免不同 stream 之间出现任意的正负抵消；
2. 行和、列和为 1，使 mixing 更接近守恒式重排，而不是任意放大或缩小；
3. 双随机矩阵的谱行为受到控制，mHC 论文据此将 residual transformation 设计成 non-expansive，更利于深层 signal propagation。

需要谨慎区分：双随机约束改善的是 residual mixing 的稳定性，不意味着整个网络的 activation、梯度或 loss 一定严格有界。attention、FFN、normalization、optimizer 和 mixed precision 仍然可能引入其它数值变化。

## Sinkhorn-Knopp 如何把矩阵投影到约束集合

训练中直接让每个参数更新后都严格满足双随机条件并不方便。mHC 先生成一个 unconstrained raw matrix $\widetilde{B}_l$，然后：

1. 先指数化，保证元素为正：

$$
M^{(0)}=\exp(\widetilde{B}_l);
$$

2. 交替进行行归一化和列归一化：

$$
M^{(t)}=T_r(T_c(M^{(t-1)}));
$$

其中 $T_r$ 和 $T_c$ 分别表示行归一化与列归一化；

3. 经过若干次迭代后，将结果作为 $B_l$。

这个过程称为 **Sinkhorn-Knopp normalization**。它不需要对矩阵做昂贵的通用约束优化，而是通过简单的归一化迭代把正矩阵推向双随机矩阵集合。

DeepSeek-V4 的实现使用最多 20 次 Sinkhorn-Knopp iterations。这里的 20 是 V4 报告中的工程配置，不应理解为所有 mHC 实现都必须使用 20 次。

## mHC 的三个映射如何参数化

mHC 不只约束 $B_l$，也会约束输入映射 $A_l$ 和输出映射 $C_l$。

### 输入映射与输出映射

先由输入相关的 dynamic component 和可学习 static component 生成 raw parameters：

$$
\widetilde{A}_l=\alpha_l^{pre}(\widehat{X}_lW_l^{pre})+S_l^{pre},
$$

$$
\widetilde{C}_l=\alpha_l^{post}(\widehat{X}_lW_l^{post})^T+S_l^{post}.
$$

其中 $\widehat{X}_l$ 是 flatten 后经过 RMSNorm 的 residual state。随后使用 sigmoid：

$$
A_l=\sigma(\widetilde{A}_l),
$$

$$
C_l=2\sigma(\widetilde{C}_l).
$$

这样可以保证输入和输出 mixing 的系数非负且处于受控范围。$A_l$ 和 $C_l$ 同时包含：

- static component：不随输入变化的基础 mixing；
- dynamic component：根据当前 residual state 调整的 mixing。

### Residual mixing matrix

$B_l$ 先通过 dynamic/static parameterization 得到 $\widetilde{B}_l$，再通过：

```text
exp(raw B)
  -> row normalization
  -> column normalization
  -> repeat
  -> doubly stochastic B
```

这意味着 mHC 不是固定的平均混合，而是**输入相关、可学习、但受流形约束的 residual mixing**。

## mHC 与普通 residual、HC、MoE 的区别

| 结构 | 主要作用对象 | 核心机制 | 主要解决的问题 |
|---|---|---|---|
| Residual connection | 一条 residual stream | $h+F(h)$ | 深层信息与梯度直通 |
| Hyper-Connections | 多条 residual streams | 学习 $A,B,C$ 做混合 | 提高层间信息流表达能力 |
| mHC | 多条 residual streams | 对 $B$ 加双随机流形约束 | 防止自由 mixing 破坏深层稳定性 |
| MoE | FFN experts | router 选择部分 experts | 扩大模型容量、控制 active compute |
| Attention | token 间交互 | query-key-value 聚合 | 跨 token 建模 |

因此，mHC 不负责：

- 选择 token 应该访问哪些历史 token；
- 选择 token 应该路由到哪个 expert；
- 更新模型参数；
- 压缩 KV cache。

它只负责在 block 与 block 之间，如何组织和传播多路 residual representations。

## DeepSeek-V4 中的工程代价

mHC 增加了 residual stream 的宽度，因此会带来：

- activation memory 增加；
- pipeline stage 之间通信量增加；
- mHC 本身的矩阵操作和动态参数生成开销；
- 反向传播时需要保存或重算更多中间状态。

DeepSeek-V4 的处理方式包括：

1. 为 mHC 实现 fused kernels；
2. 选择性重算 hidden states 和 normalized layer inputs；
3. 调整 DualPipe 1F1B schedule，让部分 mHC 操作和通信重叠；
4. 在训练框架中使用 tensor-level activation checkpointing，而不是只能对整个 module 粗粒度 checkpoint。

报告中给出的结果是：mHC 的 wall-time overhead 被控制在 overlapped 1F1B pipeline stage 的约 6.7%。这个数字属于 DeepSeek-V4 的具体实现和硬件环境，不代表 mHC 在任意框架中的固定开销。

## 它和 Muon、Hybrid Attention 的分工

DeepSeek-V4 同时使用 mHC、Muon 和 Hybrid Attention，但三者处在不同层次：

```text
mHC              -> residual representations 如何在层间传播
Muon             -> weight matrices 如何根据 gradients 更新
CSA / HCA        -> 长上下文历史信息如何压缩和访问
DeepSeekMoE      -> token 如何选择 FFN experts
```

可以把一层 V4 block 的问题拆成四个方向：

1. 输入表示如何在多路 residual stream 中混合：mHC；
2. 当前 token 访问哪些历史信息：CSA/HCA；
3. 当前 token 使用哪些 FFN capacity：MoE router；
4. 下一步如何改变模型参数：Muon/AdamW。

## 证据与边界

目前可以比较确定的内容来自 mHC 原始论文和 DeepSeek-V4 technical report：

- HC 将 residual stream 扩展为多路并学习 mixing；
- mHC 将 residual mapping 约束到 doubly stochastic matrices；
- Sinkhorn-Knopp 用于构造满足约束的 mixing matrix；
- DeepSeek-V4 对 mHC 做了 fused kernel、recomputation 和 pipeline overlap 工程优化；
- V4 报告给出了约 6.7% 的 mHC wall-time overhead。

仍然需要谨慎的地方包括：

- mHC 对最终模型 benchmark 提升的独立贡献，需要完整 ablation 才能判断；
- 双随机约束改善 signal propagation 的理论性质，不等价于所有训练稳定性问题都被解决；
- 6.7% 开销不能脱离 mHC expansion、pipeline schedule、硬件和 kernel 实现外推；
- 多路 residual stream 的表达收益与额外通信/显存成本之间，需要结合模型规模和部署目标权衡。

## 常见误解

- **误解：mHC 是 DeepSeek 专属的全新基本概念。** 更准确地说，mHC 建立在 residual connection、Hyper-Connections、双随机矩阵和 Sinkhorn-Knopp 等已有思想之上，是一种具体组合与工程实现。
- **误解：mHC 就是把 residual 加法换成一个普通矩阵乘法。** 普通自由矩阵会带来不稳定风险，mHC 的关键正是对 mixing matrix 加流形约束。
- **误解：双随机矩阵意味着 residual stream 不能学习。** 它仍然可以学习混合，只是混合受到非负性和行列归一化约束。
- **误解：mHC 直接增加 Transformer hidden size。** 它扩展的是 residual stream 的宽度，内部 attention/FFN 的 hidden size 不一定按同样倍数扩大。
- **误解：mHC 解决的是 token attention 或 expert routing。** 它解决的是 block 间 residual signal propagation；attention 和 MoE 仍由其它模块负责。

## 相关知识链接

- [[architecture/transformer/residual|Residual in Transformer]]
- [[fundamentals/neural-network-basics/residual-connection|Residual Connection]]
- [[architecture/transformer/transformer|Transformer]]
- [[architecture/sparse-and-efficient/moe|Mixture of Experts]]
- [[architecture/attention/hybrid-attention|Hybrid Attention]]
- [[training/optimization/muon|Muon Optimizer]]
- [[training/optimization/gradient-checkpointing|Gradient Checkpointing]]
- [[sources/reports/deepseek-v4-technical-documentation|DeepSeek V4 Technical Report]]

## 参考资料

- [Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880)
- [DeepSeek V4: Towards Highly Efficient Million-Token Context Intelligence](https://arxiv.org/abs/2606.19348)
