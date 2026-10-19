---
title: FlashAttention
created: 2026-04-05
published: 2026-04-05
modified: 2026-10-10
type: topic
status: growing
area: inference
tags:
  - inference
  - optimization
  - attention
---

FlashAttention 是一种 **IO-aware 的精确 attention 实现**。它保持标准 scaled dot-product attention 的数学结果，但改变 Q、K、V 和中间结果在 GPU 不同层级存储之间的访问方式，从而减少高带宽显存（HBM）读写和中间 attention 矩阵的物化。

## 它解决的是什么问题

标准 attention 的主要算术形式仍然是：

$$
O=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

当序列长度为 $T$ 时，score 矩阵的形状接近 $T\times T$。即使 GPU 的矩阵乘法吞吐足够高，完整写入和读取这个中间矩阵也会带来很大的显存访问成本。训练时还要保存反向传播所需的中间结果；长序列下，attention 往往受到 memory bandwidth 和显存容量限制。

FlashAttention 的目标不是改变哪些 token 可以互相注意，也不是压缩 KV Cache，而是让同一个 attention 计算以更少的 IO 完成。

## 核心机制

### Tiling

FlashAttention 把 Q、K、V 沿序列维度切成适合片上 SRAM 的小块。每次加载一组 Q block 和 K/V block，在片上完成局部矩阵乘法与累积，再继续处理下一个 block。

这种组织方式避免了把完整的 $T\times T$ attention score 或概率矩阵反复写入 HBM。它利用的是 GPU 的存储层次：片上 SRAM 容量更小，但访问速度和带宽通常优于 HBM。

### Online softmax

分块计算不能直接对每个 block 单独做 softmax，因为最终的归一化分母跨越整个 key 序列。FlashAttention 使用在线归一化统计量，在扫描不同 K/V block 时维护每行的最大值和归一化因子，并在需要时重新缩放已经累积的输出。

因此，分块处理仍然可以得到与标准 softmax attention 等价的结果（忽略浮点舍入差异），而不需要先保存完整 score matrix。

### 融合与重计算

attention 的相关步骤可以在一个更紧凑的 kernel 组织中完成，减少 QK 矩阵、softmax 矩阵和中间输出之间的往返读写。训练反向阶段还可以根据少量保存的统计量重新计算部分中间结果，用计算换取显存。

具体的 kernel 划分、并行策略、重计算范围和不同 GPU 架构上的优化，属于 FlashAttention 各版本的实现细节，需要结合原论文和代码核对。

## 复杂度应该怎样理解

FlashAttention **没有把标准 attention 的算术复杂度从二次降为线性**。在一般 dense attention 下，QK 的配对数量仍然随序列长度呈二次增长。

它改变的是实际的存储和 IO 行为：

| 维度 | 标准实现 | FlashAttention 的主要变化 |
|---|---|---|
| attention 数学结果 | dense exact attention | 保持相同的 exact attention 目标 |
| QK 配对数量 | 仍为二次规模 | 通常不改变渐近算术量 |
| 中间 score/probability 矩阵 | 可能完整物化 | 分块计算，避免完整物化 |
| HBM 读写 | 较多中间结果往返 | 通过片上复用降低 IO |
| 中间激活显存 | 随 score 矩阵增长 | 主要保留 block 级状态和归一化统计量 |

因此，FlashAttention 和 sliding window、block sparse、linear attention 解决的不是同一个问题：后几者改变访问范围或计算形式，FlashAttention 主要优化 dense attention 的执行方式。

## 与其他注意力变体的关系

| 机制 | 主要改变 | 是否改变 attention 的访问模式 | 是否直接缩小 KV Cache |
|---|---|---:|---:|
| MHA | 多头 Q/K/V 组织 | 否 | 否 |
| MQA / GQA | K/V head 共享 | 否 | 是 |
| MLA | K/V latent compression | 通常不以稀疏访问为核心 | 是 |
| Sliding / Block Sparse | 只计算部分位置 | 是 | 视实现而定 |
| FlashAttention | tiled、IO-aware kernel | 否 | 否 |
| CSA / HCA | 序列压缩与稀疏/密集组合 | 是 | 是，配合压缩 cache |

在现代模型中，这些设计可以叠加。例如一个模型可以使用 GQA 或 MLA 组织 K/V，再用 FlashAttention 风格的 kernel 执行实际的 dense 或局部 attention。FlashAttention 本身不等于 MQA、GQA 或稀疏 attention。

## 适用阶段与边界

FlashAttention 对训练和 prefill 阶段的长序列 attention 尤其重要，因为这些阶段会同时处理大量 query。decode 阶段每次通常只有很少的新 query，瓶颈会更多地取决于 KV Cache 读取、batch 组织和专门的 decode kernel，因此不能把 FlashAttention 的收益直接外推到所有生成阶段。

它也不能单独解决：

- dense attention 的二次算术量；
- 历史 K/V cache 的容量问题；
- 超长上下文中的信息检索和位置外推；
- GPU 之外的通信、调度和 serving 开销。

## 从算法重排到硬件协同

FlashAttention 的三篇论文可以沿着一个连续的问题来读：**先减少 attention 与显存之间的搬运，再提高 GPU 内部的并行利用率，最后让算法主动配合新硬件的异步执行和低精度能力。** 后一篇论文通常继承前一篇的数学目标，因此它们不是三个彼此独立的 attention 变体，而是同一条 exact attention kernel 路线的逐层推进。

第一篇论文先回答“为什么标准 attention 明明 FLOPs 相近，实际运行却很慢”。它指出，完整写入 $N\times N$ 的 score 和 probability 矩阵会造成大量 HBM 读写；因此通过 tiling、online softmax 和反向重计算，把中间结果尽量留在片上 SRAM，同时保持 exact attention。想看 softmax 如何分块合并、IO 复杂度如何推导，以及 block-sparse 扩展怎样建立，可以继续阅读 [[sources/papers/2022-flashattention|FlashAttention 原文解析]]。

当 HBM 往返不再是唯一瓶颈后，第二篇论文转而处理“GPU 有没有被充分用起来”。[[sources/papers/2023-flashattention-2|FlashAttention-2 原文解析]]说明了它如何减少 non-matmul FLOPs，把 sequence length 纳入 thread-block 并行，并重新安排 warp 对 Q/K/V 的分工，以减少 shared memory 通信。这里的重点不是重新定义 attention，而是让前一篇的分块算法更贴近 A100 的执行结构。

第三篇论文继续追问“如果硬件本身支持异步搬运、异步矩阵乘法和 FP8，kernel 应该怎样重写”。在 [[sources/papers/2024-flashattention-3|FlashAttention-3 原文解析]]中，producer-consumer warp specialization 负责把数据搬运和计算并行起来，GEMM-softmax 流水试图隐藏 softmax 延迟，block quantization 与 incoherent processing 则在使用 FP8 时控制 outlier 带来的误差。它的收益因此高度依赖 Hopper H100 的 TMA、WGMMA 和 Tensor Core，不能简单理解为一个脱离硬件的通用数学变体。

用这个顺序阅读时，可以把三篇论文分别对应到三个层次：**FlashAttention-1 解释 IO-aware 算法为什么成立，FlashAttention-2 解释并行工作如何组织，FlashAttention-3 解释算法如何与具体 GPU 的异步流水和低精度指令协同。** 论文中的完整公式、伪代码、复杂度证明和实验设置分别保留在上述原文解析中；本页只保留它们在整个 attention 知识体系中的连接关系。

## 相关知识链接

- [[architecture/attention/attention|Attention 机制]]
- [[architecture/attention/multi-head-attention|Multi-Head Attention]]
- [[architecture/attention/sliding-window-attention|Sliding Window Attention]]
- [[architecture/attention/multi-query-attention|Multi-Query Attention]]
- [[architecture/attention/grouped-query-attention|Grouped-Query Attention]]
- [[architecture/attention/multi-head-latent-attention|Multi-Head Latent Attention]]
- [[architecture/attention/hybrid-attention|Hybrid Attention]]
- [[inference/attention-acceleration/attention-kernels|Attention Kernels]]
- [[inference/attention-acceleration/flash-decoding|FlashDecoding]]

## 原始论文

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608)
