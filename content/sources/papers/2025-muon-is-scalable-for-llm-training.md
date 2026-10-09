---
title: "Muon is Scalable for LLM Training"
created: 2026-10-08
published: 2026-10-08
modified: 2026-10-08
type: source
status: processed
source_type: paper
area: sources
tags:
  - source
  - paper
  - optimizer
  - muon
  - llm-training
  - scaling-law
  - distributed-training
  - moe
aliases:
  - Muon is Scalable for LLM Training
source_url: https://arxiv.org/abs/2502.16982
paper_date: "2025-02"
paper_order: "16982"
---

# Muon is Scalable for LLM Training

## 基本信息

- 标题：[Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982)
- 作者：Jingyuan Liu、Jianlin Su、Xingcheng Yao 等
- 机构/来源：Moonshot AI、University of California, Los Angeles（UCLA）
- 发表：arXiv:2502.16982v1，Technical Report
- 日期：2025-02
- 代码/数据：[论文 arXiv 页面](https://arxiv.org/abs/2502.16982)；报告说明将公开 distributed Muon implementation、Moonlight checkpoints 和中间 checkpoint
- 相关主题：[[training/optimization/muon|Muon Optimizer]]、[[training/optimization/optimizer-state|Optimizer State]]、[[fundamentals/linear-algebra/eigendecomposition-svd|SVD]]、[[training/distributed-training/zero|ZeRO]]、[[training/scaling/scaling-law|Scaling Law]]、[[architecture/sparse-and-efficient/moe|Mixture of Experts]]

## 这篇报告要解决什么

Muon 最初在小规模语言模型和竞速训练任务中表现很好，但它能否用于数十亿参数、数万亿 token 的 LLM 训练，并不是自然成立的结论。作者把问题拆成三层：

1. 原始 Muon 在模型变大、训练 token 变多后，权重 RMS 可能持续增大，数值范围和长期训练稳定性会变差；
2. Muon 的正交化更新依赖完整矩阵，普通 ZeRO-1 那种只在本地分片上完成逐元素 optimizer update 的做法不能直接套用；
3. 即使 pre-training 有收益，Muon 是否适用于 SFT，以及 AdamW 预训练模型能否直接切换到 Muon，也需要单独验证。

报告给出的答案不是简单替换 optimizer，而是一套可扩展 recipe：

```text
Muon on matrix parameters
  -> add decoupled weight decay
  -> correct update RMS according to matrix shape
  -> use AdamW for embeddings, heads, norms, and non-matrix parameters
  -> gather full gradients for Newton-Schulz orthogonalization
  -> apply ZeRO-1-style distributed ownership
  -> validate scaling laws and 3B/16B MoE pretraining
```

作者的主张是：经过这些修正后，Muon 在 compute-optimal scaling law 实验中用约 52% 的 AdamW training FLOPs 达到相当表现，并训练出 3B active / 16B total、5.7T token 的 Moonlight MoE 模型。

## 先把 Muon 放回优化器背景

### AdamW 在做什么

AdamW（Adam with decoupled weight decay）对每个标量参数独立维护一阶矩和二阶矩。给定梯度 $g_t$：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
$$

随后按坐标归一化梯度并更新参数。它的优点是成熟、稳定、对不同参数的梯度尺度具有适应性；缺点是它主要把参数看作一组彼此独立的 scalar，不能直接利用一个 $H\times 4H$ 或 $H\times H$ 权重矩阵的整体几何结构。

### Muon 只针对矩阵参数

Muon 的名字来自 **MomentUm Orthogonalized by Newton-Schulz**。它的核心对象是 hidden layer 中的二维 weight matrix，而不是所有参数：

- Transformer 的 attention / MLP matrix 可以使用 Muon；
- embedding、LM head、RMSNorm、bias 和其他 scalar/vector 参数通常继续使用 AdamW；
- 4D convolution kernel 可以先 flatten 成矩阵再考虑 Muon。

因此，Muon 通常不是“整个模型只使用一种 optimizer”，而是 **Muon + AdamW 的混合 optimizer recipe**。这点对复现很重要：把所有参数无差别切到 Muon，并不是这篇报告验证过的设置。

## Muon 的核心更新：先 momentum，再正交化

给定矩阵参数 $W$ 和梯度 $G_t$，Muon 先维护 momentum：

$$
M_t=\mu M_{t-1}+G_t.
$$

实际实现使用 Nesterov 风格，把 $\mu M_t+G_t$ 送入正交化步骤。若对 momentum 做奇异值分解（SVD，Singular Value Decomposition）：

$$
M=U\Sigma V^T,
$$

理想的正交化更新近似为：

$$
\operatorname{Ortho}(M)=UV^T.
$$

它保留奇异向量方向，但把不同方向的奇异值压到相近尺度。于是更新不再被少数最大奇异值方向完全支配。

最后的参数更新可以抽象写成：

$$
W_t=W_{t-1}-\eta_t O_t,
$$

其中 $O_t$ 是经过 Newton-Schulz 迭代近似得到的正交化 momentum。

### 为什么不直接做 SVD

直接求 SVD 容易理解，但在每个训练 step、对大量 hidden matrices 重复执行代价太高。Muon 采用 Newton-Schulz iteration，以矩阵乘法反复逼近半正交矩阵：

$$
X_0=\frac{M}{\|M\|_F},
$$

$$
X_k=aX_{k-1}+b(X_{k-1}X_{k-1}^T)X_{k-1}
+c(X_{k-1}X_{k-1}^T)^2X_{k-1}.
$$

报告采用的系数是 $(a,b,c)=(3.4445,-4.7750,2.0315)$。这些系数让小奇异值附近的迭代增长更快，实际通常只运行 5 steps；运行 10 steps 会得到更精确的正交化，但没有带来稳定的额外性能收益。

### 正交化到底改变了什么

从 SVD 角度看，普通 momentum 的更新是 $U\Sigma V^T$，Muon 近似把它改成 $UV^T$。因此：

- 大奇异值方向不再按原比例获得特别大的更新；
- 小奇异值方向被相对放大，不容易被主方向长期压制；
- 更新矩阵更接近 semi-orthogonal matrix；
- 训练过程探索的矩阵方向更分散。

报告在 1.2T token checkpoint 上计算 SVD entropy，发现 Muon 训练出的 attention、expert、shared expert、router 和 dense matrices 大多比 AdamW 具有更高的 SVD entropy。这个结果支持“Muon 能让权重谱更分散”的解释，但它仍是机制相关性证据，不等于已经证明更高 SVD entropy 必然带来更高下游能力。

## 方法背景：Muon 与其他正交化优化的关系

### 与 Shampoo 的关系

Shampoo 使用左右预条件矩阵，对梯度做类似 inverse fourth-root 的变换。若去掉累积项，在 SVD 形式下可以得到接近 $UV^T$ 的正交化梯度。因此 Muon 可以被理解为一种更直接的矩阵正交化路线：

- Shampoo 显式维护左右 preconditioner，状态和通信更复杂；
- Muon 先做 momentum，再用 Newton-Schulz 近似正交化；
- Muon 的主要额外状态只有 momentum，作者据此强调它比 AdamW 少一个 moment buffer。

这不是说 Muon 等于 Shampoo。二者的状态维护、更新顺序、数值实现和预条件含义都不同，但它们都利用了矩阵而不是 scalar 参数的几何结构。

### 与 Orthogonal-SGDM 的关系

Orthogonal-SGDM 也正交化梯度，但它先正交化梯度，再对正交化结果使用 momentum。Muon 则把 momentum 放在正交化之前：

```text
Orthogonal-SGDM: gradient -> orthogonalize -> momentum
Muon:            gradient -> momentum -> orthogonalize
```

Muon 报告认为后者经验上更好，并且用 Newton-Schulz 替代 SVD 以降低计算成本。

### 与 spectral norm steepest descent 的关系

把优化看作 norm-constrained steepest descent 时，Adam 的几何约束更接近按坐标自适应的 Max-of-Max norm；Muon 的正交化更新则更接近 spectral norm 下的 steepest descent。

这提供了一个直觉：weight matrix 是作用在 hidden space 上的线性算子，限制其 operator norm 可能比把每个元素单独归一化更符合矩阵层的作用方式。报告也把扩展到一般 Schatten norm 视为后续方向，说明当前 Muon 主要对应 spectral-norm 风格，而不是已经解决所有矩阵范数选择问题。

## 从原始 Muon 到可扩展 Muon：两个关键修正

### 1. 加入 weight decay

原始 Muon 在小规模实验中可以快速下降，但作者在更长、更大的训练中观察到 weight RMS 和 layer output RMS 持续增大，甚至可能超出 BF16 的安全范围。于是将 AdamW 的 decoupled weight decay 加回 Muon：

$$
W_t=W_{t-1}-\eta_t(O_t+\lambda W_{t-1}).
$$

800M 模型、100B token 的实验比较了 AdamW、无 weight decay 的 vanilla Muon 和带 weight decay 的 Muon：

- vanilla Muon 早期下降更快；
- 训练时间拉长后，权重增长开始伤害验证 loss；
- 带 weight decay 的 Muon 后期超过 vanilla Muon，并超过 AdamW。

这说明 Muon 的短期训练速度不能代表长期 scaling 行为。weight decay 在这里不只是常规正则化，而是控制正交化更新与矩阵权重尺度之间的长期平衡。

### 2. 校准不同矩阵的 update RMS

对于形状为 $[A,B]$ 的 full-rank matrix，报告给出 Muon update RMS 的理论量级：

$$
\operatorname{RMS}(O)\approx\frac{1}{\sqrt{\max(A,B)}}.
$$

问题在于，不同矩阵形状会获得不同的 update RMS：

- $[H,4H]$ 的 MLP matrix 与 $[H,H]$ 的 attention matrix 不会自然得到同样的更新尺度；
- 把 GQA 或 MLA 的小 head matrix 单独作为参数时，$\max(A,B)$ 变小，更新可能过大；
- 大的 MLP matrix 则可能得到过小更新。

作者比较了三种处理方式：

1. **Baseline**：所有矩阵都乘同一个 $0.2\sqrt{H}$；
2. **Update Norm**：直接把每个更新矩阵归一化到 RMS = 0.2；
3. **Adjusted LR**：为每个矩阵把 learning rate 乘以 $0.2\sqrt{\max(A,B)}$。

在 800M 规模的短实验中，Update Norm 和 Adjusted LR 都优于 Baseline；作者最终选择 Adjusted LR，因为它不需要额外计算 update RMS，成本更低。这个修正的意义是：让 Muon 的 learning rate 能跨矩阵形状复用，而不是让 optimizer 通过形状差异隐式改变每层的有效步长。

## Distributed Muon：为什么 ZeRO-1 不能直接套用

ZeRO-1 对 AdamW 很自然：每个 data-parallel rank 保存一部分 optimizer states，每个参数的更新是 element-wise 的，owner rank 可以直接更新自己的分片。

Muon 不一样。Newton-Schulz 需要看到完整的 gradient matrix。如果 rank 只拿到一个矩阵分片，就无法对这个矩阵执行正确的 $XX^T$ 和正交化。因此 Distributed Muon 的流程是：

```text
1. DP reduce-scatter full gradients
2. 每个 rank 对自己持有的 gradient partition 做 momentum update
3. DP gather，恢复完整 gradient matrix
4. 对完整矩阵执行 Newton-Schulz
5. 每个 rank 丢弃不属于自己的 update，只保留本地分片
6. owner rank 更新本地 parameter partition
7. DP all-gather 更新后的 parameters
```

### 内存与通信

Muon 只维护一个 momentum buffer，而 AdamW 通常需要一阶矩和二阶矩，因此 optimizer state 更小。作者称 Distributed Muon 的通信量约为 Distributed AdamW 的 $(1,1.25]$ 倍，额外成本主要来自 DP gather 和 Newton-Schulz 前的完整矩阵恢复；实践中由于只在 data-parallel group 内通信，额外开销通常接近较低端。

报告文本同时提到实践中的通信比值可能接近 1.5，这与前面的理论区间并不完全一致，可能对应不同的通信口径或 TP 设置。这里不应把“Muon 通信只增加 25%”当成无条件结论。

如果启用 Tensor Parallelism（TP，张量并行），还需要额外在 TP group 上 gather 矩阵。Muon 的分布式实现因此不是简单把 AdamW 的 optimizer class 换掉，而是要和 TP、PP、EP、DP 的参数布局共同设计。

## 实验与证据

### Update RMS 实验

作者在 800M 模型上修改 MLP 形状，比较 $[H,H]$ 的 attention query matrix 和 $[H,4H]$ 的 MLP matrix。4B token 时：

| 方法 | Training loss | Validation loss | Query weight RMS | MLP weight RMS |
|---|---:|---:|---:|---:|
| Baseline | 2.734 | 2.812 | 3.586e-2 | 2.52e-2 |
| Update Norm | 2.720 | 2.789 | 4.918e-2 | 5.01e-2 |
| Adjusted LR | 2.721 | 2.789 | 3.496e-2 | 4.89e-2 |

两种校准方法都改善了 loss，但 Adjusted LR 计算更便宜，因此被用于后续大规模实验。

### Scaling law：约 52% 的 AdamW FLOPs

作者在 399M 到 1.5B 的 dense Llama-style 模型上，对 AdamW 做 compute-optimal hyperparameter search，再让 Muon 复用 AdamW 的最优学习率和 batch 配置。拟合结果显示，在相同目标 loss 下，Muon 需要约 52% 的 AdamW training FLOPs，约等于 2x compute efficiency。

这个结果的关键前提是 AdamW baseline 经过了系统调参，并且 Muon 已经通过 update RMS calibration 对齐了有效更新尺度。若 AdamW baseline 没调好，或者 Muon 使用不匹配的 update scale，比较就没有意义。

### Moonlight：3B / 16B MoE 预训练

Moonlight 使用接近 DeepSeek-V3-Small 的 MoE 架构：2.24B activated、15.29B total parameters（含 embedding 后约 3B/16B），最大 context 为 8K，总训练 5.7T tokens。训练分三阶段：

1. 0-33B tokens：2K steps warmup 到 $4.2\times10^{-4}$，batch size 为 2048 examples；
2. 33B-5.2T tokens：cosine decay 从 $4.2\times10^{-4}$ 到 $4.2\times10^{-5}$，batch 在 200B tokens 后从 2048 增至 4096；
3. 5.2T-5.7T tokens：cooldown 阶段先升到 $1\times10^{-4}$，再在 500B tokens 中线性衰减到 0，并使用高质量 math/code/reasoning data。

在约 1.2T token checkpoint 上，Muon 版 Moonlight 相比同架构 AdamW 版在 HumanEval、MBPP、GSM8K、MATH 和中文任务上整体更强；例如 HumanEval 为 37.2 对 29.3，MATH 为 19.8 对 16.1。作者还报告，完整训练到 5.7T tokens 后，Moonlight 在 MMLU、MMLU-Pro、BBH、HumanEval、MBPP、MATH、C-Eval 和 CMMLU 上超过 DeepSeek-V2-Lite，并以更少训练 token 达到与更大公开模型相近或更好的结果。

这些结果支持 Muon 在 MoE LLM pre-training 中具有实际收益，但并不能把 Moonlight 的全部提升归因给 Muon：模型 recipe、数据、cooldown 和其它架构设置也共同发生作用。

### Singular spectrum 分析

作者使用 SVD entropy 衡量权重矩阵的奇异值是否集中在少数方向。Muon 在 attention、experts、shared experts、router 和 dense matrices 上的 SVD entropy 通常高于 AdamW，超过 90% 的权重矩阵观察到这一趋势。

这与 Muon 的几何直觉一致：正交化会削弱少数大奇异值方向的支配作用，让更新探索更多方向。它是机制证据，而不是能力因果证明；更高的 SVD entropy 不自动等价于更好的下游性能。

### SFT：Muon 不是简单替换 AdamW 就一定有效

报告专门分析了 pre-training optimizer 和 SFT optimizer 是否可以互换。Moonlight@1.2T 分别使用 Muon 或 AdamW 做 SFT，Moonlight-A@1.2T 也做了对应对照。结果显示，Muon 预训练 + Muon SFT 通常最好；Muon 预训练 + AdamW SFT 仍然较强；AdamW 预训练后换 Muon SFT 并没有稳定优势。

在 Qwen2.5-7B 上直接用 Muon 做 SFT，结果与 Adam-SFT 大体相当，部分 math/code 指标还略低。报告因此更支持这样的结论：**Muon 的主要优势出现在从头 pre-training，SFT 阶段单独切换 Muon 的收益并不稳定。**

## 如何理解这篇报告的结论

Muon 的核心贡献不是“找到一个永远比 AdamW 好的 optimizer”，而是把矩阵正交化更新推进到了一个可以进入 LLM 训练系统的状态：

- 原始 Muon 提供矩阵几何上的更新方向；
- weight decay 解决长期训练中的权重膨胀；
- update RMS calibration 解决矩阵形状导致的有效步长不一致；
- Distributed Muon 解决完整矩阵正交化与 ZeRO-1 分片之间的冲突；
- Moonlight 证明它可以支撑 5.7T token 的 MoE pre-training。

从工程决策看，Muon 适合优先考虑的场景是：从头预训练、矩阵参数占主导、训练规模足够大、团队能够修改 distributed optimizer 和 kernel。对于一个已有 AdamW checkpoint 的短 SFT 或 RL run，直接切换 Muon 的收益没有被这篇报告充分证明。

## 局限与开放问题

- 主要 LLM 证据集中在 Moonlight 这一套 MoE 架构与 recipe，尚不能证明 Muon 对所有 dense/MoE/多模态架构都同样有效；
- scaling law 主要使用 399M-1.5B dense 模型拟合，再外推到 3B/16B Moonlight，仍需要更大规模独立验证；
- Moonlight 与公开模型的比较包含不同数据、token 数、架构和后处理，不能作为严格的 optimizer-only ablation；
- AdamW baseline 虽然做了调参，但其它 optimizer 的公平调参空间仍可能影响结论；
- Muon 对 embedding、LM head、norm 和 bias 的处理仍依赖 AdamW，完整“全参数 Muon”还没有被证明；
- pre-training 与 SFT 之间存在 optimizer mismatch：AdamW 预训练模型切换 Muon，以及 Muon 预训练模型切换 AdamW，都可能损失一部分优势；
- Distributed Muon 的通信量描述存在理论区间与实践表述不完全一致的问题，需要结合代码和具体 TP/DP 拓扑核对；
- 目前报告主要使用 loss、benchmark 和 SVD entropy 作为证据，Muon 为什么能带来更好 scaling 的完整理论解释仍未闭合。

## 关键结论

1. Muon 面向的是二维 hidden-layer weight matrices，不是 AdamW 的全参数无差别替代品。
2. Muon 的核心操作是对 momentum update 做 Newton-Schulz 正交化，近似保留奇异向量方向、压平奇异值尺度。
3. 把 Muon 扩展到 LLM 的关键不在原始公式，而在 weight decay、update RMS calibration 和 distributed full-matrix update。
4. 这篇报告最有说服力的证据是 compute-optimal scaling law、Moonlight 5.7T token pre-training 和 distributed implementation，而不是某一张单点 benchmark 表。
5. Muon 当前更像一种面向大规模 pre-training 的矩阵 optimizer recipe；它在 SFT 上可以工作，但不能简单推导出切换 optimizer 就一定有收益。

## 相关知识链接

- [[training/optimization/muon|Muon Optimizer]]
- [[training/optimization/optimizer-state|Optimizer State]]
- [[fundamentals/linear-algebra/eigendecomposition-svd|SVD]]
- [[training/distributed-training/zero|ZeRO]]
- [[training/scaling/scaling-law|Scaling Law]]
- [[architecture/sparse-and-efficient/moe|Mixture of Experts]]
- [[sources/papers/2024-deepseek-v3|DeepSeek-V3 Technical Report]]
- [[sources/reports/deepseek-v4-technical-documentation|DeepSeek V4 Technical Report]]

## 参考资料

- [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982)
- [Muon: An Optimizer for Hidden Layers in Neural Networks](https://kellerjordan.github.io/posts/muon/)
- [Old Optimizer, New Norm: An Anthology](https://arxiv.org/abs/2409.20325)
- [Orthogonalising Gradients to Speed Up Neural Network Optimisation](https://arxiv.org/abs/2202.07052)
