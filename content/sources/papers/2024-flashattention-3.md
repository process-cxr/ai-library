---
title: "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision"
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
  - low-precision
paper_date: "2024-07"
paper_order: "08608"
source_url: https://arxiv.org/abs/2407.08608
---

# FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision

## 基本信息

- 标题：[FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608)
- 作者：Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao
- 机构/来源：Colfax Research；Meta；NVIDIA；Georgia Tech；Princeton University；Together AI
- 发表：arXiv:2407.08608v2
- 日期：2024-07-12
- 代码/数据：[FlashAttention repository](https://github.com/Dao-AILab/flash-attention)
- 相关主题：[[inference/attention-acceleration/flash-attention|FlashAttention]]、[[inference/attention-acceleration/attention-kernels|Attention Kernels]]、[[inference/quantization/quantization|Quantization]]、[[inference/quantization/fp8|FP8]]

## 研究问题

FlashAttention-2 在 A100 上已经能达到较高利用率，但直接运行在 NVIDIA Hopper H100 上时只有约 35% 的理论吞吐。Hopper 提供了异步的 Tensor Memory Accelerator（TMA）、异步 warpgroup matrix multiply-accumulate（WGMMA）以及 FP8 Tensor Core。论文的问题是：如何重新设计 attention kernel，使数据搬运、矩阵乘法和 softmax 能够重叠，并在低精度下控制数值误差。

## 核心主张

FlashAttention-3 针对 Hopper 提出三组互相配合的技术：

1. producer-consumer warp specialization，让负责数据搬运和负责计算的 warpgroups 并行工作；
2. 将 softmax 隐藏在异步 block-wise GEMM 之后，通过跨迭代流水重叠非矩阵乘法与 Tensor Core 计算；
3. 使用 FP8 GEMM，并通过 block quantization 与 incoherent processing 降低量化误差。

论文报告 FP16 前向相对 FlashAttention-2 提升约 1.5--2.0 倍，最高达到 740 TFLOPs/s、H100 理论峰值的 75%；FP8 前向接近 1.2 PFLOPs/s，并比标准 per-tensor FP8 baseline 有更低数值误差。

## 方法与机制

### Hopper 执行模型

论文围绕 H100 SXM5 的层级组织展开：全局显存、shared memory、register file、thread、warp、warpgroup 和 thread block（CTA）。TMA 可以异步把 Q/K/V tile 搬入 shared memory；WGMMA 可以异步调用 Tensor Core 做矩阵乘法。异步性使数据加载和计算不必严格串行等待。

### Producer-consumer warp specialization

一个 CTA 内的 warps 被分成 producer 和 consumer：producer 负责 TMA 加载 Q/K/V，并通过 barrier 通知 consumer；consumer 使用 shared memory 中的 tile 做 QK^T、局部 softmax 和 PV。多阶段 circular shared-memory buffer 让加载下一块数据时，当前块仍可被计算。

这种分工还允许根据角色重新分配寄存器：执行 WGMMA 的 consumer 获得更多寄存器，producer 只承担数据移动。

### GEMM-softmax 流水与 pingpong scheduling

softmax 的指数、归约和输出重缩放使用的吞吐远低于 Tensor Core 矩阵乘法。FlashAttention-3 通过 pingpong 调度，让一个 warpgroup 计算 GEMM 时，另一个 warpgroup 处理前一块的 softmax；随后两个角色交换。

在单个 warpgroup 内，论文又使用两阶段寄存器流水：当前迭代计算下一块的 QK^T，同时处理上一块的 softmax 和 PV。三阶段流水理论上能增加重叠，但会增加寄存器压力，且编译器不一定按理想顺序排布，因此实验中两阶段方案更好。

### FP8、block quantization 与 incoherent processing

FP8 只有有限的指数和尾数位，LLM activation 中的 outlier 会使 per-tensor scale 产生较大的量化误差。FlashAttention-3 使用：

- **Block quantization**：对每个 Q/K/V tile 使用独立 scale，使 block 内的动态范围更集中；
- **Incoherent processing**：在量化前将 Q、K 乘以随机正交矩阵 $M$。因为 $MM^T=I$，有 $(QM)(KM)^T=QK^T$；随机混合可把 outlier 分散到各维度。

论文实现中使用由随机符号对角矩阵和 Hadamard 矩阵组成的快速正交变换，避免显式执行一般的 $d\times d$ 矩阵乘法。

FP8 WGMMA 对 operand layout 有额外约束，论文还通过 V 的 tile transpose、寄存器重排和 FP32 accumulator layout 适配，让两次 attention GEMM 能在一个 kernel 中衔接。

## 实验与证据

实验在 H100 80GB SXM5 上进行，固定 GPU 时钟，比较标准 PyTorch、FlashAttention-2、Triton、cuDNN 和 FlashAttention-3。

| 实验 | 报告结果 |
|---|---|
| FP16/BF16 attention 前向 | 相比 FlashAttention-2 约 1.5--2.0 倍；相比标准实现最高约 3--16 倍 |
| FP16/BF16 attention 反向 | 相比 FlashAttention-2 约 1.5--1.75 倍 |
| H100 吞吐 | 最高约 740 TFLOPs/s，约 75% 理论峰值 |
| FP8 前向 | 接近 1.2 PFLOPs/s；在部分 head dimension 和 mask 设置下超过或接近 cuDNN |
| Pipeline ablation | 固定设置下，完整两阶段流水 661 TFLOPs/s；去掉 GEMM-softmax overlap 为 582，去掉 warp specialization 为 570 |
| 数值误差 | FP16 FlashAttention-2/3 的 RMSE 约为 $1.9\times10^{-4}$，标准 FP16 baseline 为 $3.2\times10^{-4}$；FP8 FlashAttention-3 相比标准 FP8 baseline 的误差降低约 2.6 倍 |

FP8 结果依赖 block scale、layout、outlier 分布、head dimension、causal mask 和 kernel 的具体实现。论文没有把这些结果解释成任意硬件或任意模型上的固定收益。

## 关键结论

1. 在新一代 GPU 上，attention 的优化重点从单纯减少 HBM 访问扩展到异步执行、warp specialization 和指令级流水。
2. softmax 等非矩阵乘法操作可能成为 Tensor Core attention 的主要限制，跨迭代重叠可以隐藏这部分开销。
3. 低精度 attention 需要同时处理硬件 layout 和数值稳定性，block quantization 与 incoherent processing 是两个不同层面的补偿机制。
4. FlashAttention-3 仍然计算 exact attention；它的主要收益来自 Hopper 执行模型和 kernel 调度，而不是改变 attention 的信息访问范围。

## 局限与疑问

- 论文主要针对 Hopper H100，迁移到其他 GPU 需要相应的异步内存和矩阵乘法能力。
- FP8 kernel 在短序列和 causal masking 下受 persistent-kernel 与 load-balancing 影响，部分场景不如 cuDNN。
- 作者将 LLM inference 优化、FP8 persistent kernel 以及大规模训练中的低精度影响列为后续工作。
- 数值误差实验是合成 outlier 分布和 kernel 对比，不能替代完整模型训练稳定性评估。

## 相关知识链接

- [[inference/attention-acceleration/flash-attention|FlashAttention]]
- [[inference/attention-acceleration/attention-kernels|Attention Kernels]]
- [[sources/papers/2022-flashattention|FlashAttention]]
- [[sources/papers/2023-flashattention-2|FlashAttention-2]]
