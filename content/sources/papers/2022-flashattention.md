---
title: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
created: 2026-10-10
published: 2026-10-10
modified: 2026-10-10
type: source
status: processed
source_type: paper
area: sources
tags:
  - source
  - paper
  - attention
  - flash-attention
  - gpu
  - long-context
paper_date: "2022-05"
paper_order: "14135"
source_url: https://arxiv.org/abs/2205.14135
---

# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

## 基本信息

- 标题：[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- 作者：Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré
- 机构/来源：Stanford University；University at Buffalo, SUNY
- 发表：arXiv:2205.14135v2
- 日期：2022-06-23
- 代码/数据：[FlashAttention code](https://github.com/HazyResearch/flash-attention)
- 相关主题：[[inference/attention-acceleration/flash-attention|FlashAttention]]、[[architecture/attention/attention|Attention]]、[[architecture/attention/sliding-window-attention|Sliding Window Attention]]、[[inference/attention-acceleration/attention-kernels|Attention Kernels]]

## 研究问题

标准 self-attention 的算术量和中间矩阵规模都随序列长度呈二次增长。论文指出，许多近似 attention 方法虽然减少了理论 FLOPs，却未必带来 wall-clock 加速，因为它们仍然需要频繁读写 GPU 的高带宽显存（High Bandwidth Memory，HBM）。论文因此把 attention 的 GPU 执行看成一个分层存储问题：片上静态随机访问存储器（Static Random-Access Memory，SRAM）容量小但访问快，HBM 容量大但访问慢，attention 的实际速度取决于二者之间搬运了多少数据。

## 核心主张

FlashAttention 是一个**精确 attention 算法**：它保持

$$
O=\operatorname{softmax}(QK^T)V
$$

的计算目标，不用稀疏或低秩近似换取速度。它通过 tiling、在线 softmax 归一化和反向重计算，避免把完整的 $N\times N$ score/probability 矩阵写入 HBM。

论文给出的主要结论是：

- 前向和反向所需的额外显存从二次规模降为线性规模；
- 在典型 SRAM 大小下，HBM 访问量显著低于标准实现；
- 即使重计算增加了 FLOPs，减少 HBM 访问仍能带来实际加速；
- 同一套 IO-aware 原语可以扩展到预定义 block-sparse attention。

## 方法与机制

### 标准实现的瓶颈

标准实现通常分三步：计算 $S=QK^T$ 并写回 HBM，读取 $S$ 做 softmax 并写回 $P$，再读取 $P$ 和 $V$ 计算 $O=PV$。训练反向还需要保存 $S$ 或 $P$。当 $N\gg d$ 时，$N\times N$ 中间矩阵的读写会成为主要成本。

### Tiling 与在线 softmax

FlashAttention 将 Q、K、V 划分为能放入 SRAM 的 block。一个 K/V block 被加载后，算法依次处理 Q blocks，在片上计算局部 score、局部指数和局部的 value 加权和。

softmax 的归一化跨越整个 key 序列，不能对每个 block 独立归一化。算法因此为每个 query row 维护当前最大值 $m$ 和归一化因子 $\ell$。读入新 block 后，用新的最大值重新缩放旧的累积输出，再合并新 block 的贡献。这个在线聚合保证分块结果与完整 softmax 一致，数值上仍使用 max-shifting 避免指数溢出。

### 反向重计算

反向传播不保存完整 $S$ 和 $P$，而是保存前向输出以及 softmax 归一化统计量。反向再次把对应 Q、K、V block 读入 SRAM，重算局部 score 和 probability，再计算 $dQ$、$dK$、$dV$。

这是一种选择性重计算：额外计算 FLOPs 换取更少的 HBM 往返。论文报告，在其 GPU 设置下，重计算带来的算术增加并没有抵消 IO 收益。

### IO 复杂度

在论文假设的 SRAM 范围内，标准 attention 的 HBM 访问包含二次规模的中间矩阵读写；FlashAttention 的主要 HBM 访问项约为

$$
\Theta\left(\frac{N^2d^2}{M}\right),
$$

其中 $N$ 是序列长度，$d$ 是 head dimension，$M$ 是可用 SRAM 大小。论文还给出下界，说明在一段 SRAM 大小范围内，没有精确 attention 算法能渐近地进一步改善该访问量。

这不是把 attention 的算术复杂度改成线性：QK 的配对仍然是二次规模。变化发生在中间矩阵是否被反复写入慢速显存，以及 block 能否在片上复用。

### Block-sparse 扩展

论文进一步给定一个预定义的 block-sparsity mask，只计算非零 block。若非零 block 占比为 $s$，其主要 IO 项随 $s$ 缩小。作者使用 butterfly pattern 做实验，展示了 block-sparse FlashAttention 在长序列上可以超过 dense FlashAttention，同时保持与 dense 模型接近的任务效果。

## 实验与证据

论文的测量都依赖具体 GPU、序列长度、head dimension、batch 和 baseline，实现结果不能脱离这些条件直接外推。

| 实验 | 报告结果 |
|---|---|
| BERT-large | 在 8 张 A100 上达到 MLPerf 1.1 训练速度记录的 1.15 倍 |
| GPT-2 | 相比 HuggingFace 标准实现最高约 3 倍训练加速；相比 Megatron-LM 最高约 1.8 倍 |
| Long-Range Arena | 最高约 2.4 倍加速；block-sparse 版本最高约 2.8 倍 |
| GPT-2 长上下文 | 4K context 比 1K baseline 更快，并改善约 0.7 perplexity |
| Long document classification | MIMIC-III 在 16K 相比 512 token 提升 4.3 个 micro-F1 点；ECtHR 在 8K 提升 8.5 点 |
| Path-X / Path-256 | dense FlashAttention 在 Path-X 达到 61.4%；block-sparse 版本在 Path-256 达到 63.1% |
| Attention benchmark | 常见长度下最高约 3 倍于 PyTorch 标准实现；额外显存随序列长度线性增长，最高约节省 20 倍 |

论文还观察到：短序列上，减少 IO 的 exact FlashAttention 可以快于许多近似方法；序列足够长时，近似方法的渐近计算优势会产生交叉点，而 block-sparse FlashAttention 继续受益于稀疏模式。

## 关键结论

1. attention 的实际瓶颈不能只用 FLOPs 描述，HBM 与片上 SRAM 之间的 IO 是关键变量。
2. 精确 attention 可以通过 tiling 和在线归一化避免物化完整 attention 矩阵。
3. 反向重计算在 GPU 上可能同时节省显存和时间，因为它减少了昂贵的 HBM 访问。
4. dense exact attention 的算术复杂度仍是二次；FlashAttention 改善的是存储、IO 和实际 kernel 执行。
5. block-sparse FlashAttention 展示了一个组合路径：稀疏访问改变算术量，IO-aware kernel 负责把稀疏性转化为实际速度。

## 局限与疑问

- 论文的实现需要手写 CUDA kernel，迁移到不同 GPU 架构需要重新适配。
- 单 GPU IO 分析不能直接覆盖多 GPU attention 的通信成本。
- block-sparse 实验使用预定义结构化 mask；它不等于动态内容选择。
- 报告的速度依赖 GPU 型号、block size、head dimension、mask、dropout 和 baseline 实现，不能作为所有部署场景的固定倍率。
- 论文提出的 IO-aware 原则可推广到其他模块，但没有在本文中系统证明所有模块都能获得同样收益。

## 相关知识链接

- [[inference/attention-acceleration/flash-attention|FlashAttention]]
- [[architecture/attention/attention|Attention 机制]]
- [[architecture/attention/sliding-window-attention|Sliding Window Attention]]
- [[inference/attention-acceleration/attention-kernels|Attention Kernels]]
- [[sources/papers/2020-longformer|Longformer]]
- [[sources/papers/2020-big-bird|Big Bird]]
