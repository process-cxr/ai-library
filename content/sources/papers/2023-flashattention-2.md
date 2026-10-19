---
title: "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
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
  - parallelism
paper_date: "2023-07"
paper_order: "08691"
source_url: https://arxiv.org/abs/2307.08691
---

# FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning

## 基本信息

- 标题：[FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- 作者：Tri Dao
- 机构/来源：Princeton University；Stanford University
- 发表：arXiv:2307.08691v1
- 日期：2023-07-17
- 代码/数据：[FlashAttention repository](https://github.com/Dao-AILab/flash-attention)
- 相关主题：[[inference/attention-acceleration/flash-attention|FlashAttention]]、[[inference/attention-acceleration/attention-kernels|Attention Kernels]]、[[architecture/attention/multi-query-attention|Multi-Query Attention]]、[[architecture/attention/grouped-query-attention|Grouped-Query Attention]]

## 研究问题

FlashAttention-1 已经通过 IO-aware tiling 降低了显存读写，但在 A100 上只达到理论峰值的约 25%--40%。论文将剩余瓶颈定位为 GPU 工作划分：thread blocks 和 warps 之间的任务分配不理想，导致占用率低，或者产生不必要的 shared memory 读写。

## 核心主张

FlashAttention-2 保持 FlashAttention 的精确计算目标和线性额外显存，同时从三个方向重组 kernel：

1. 减少 non-matmul FLOPs，让更多时间用于 GPU Tensor Core 擅长的矩阵乘法；
2. 在 batch、head 之外沿 sequence length 并行，让长序列和小 batch 也能占满 GPU；
3. 在单个 thread block 内重新划分 warp 工作，减少 warp 间同步和 shared memory 往返。

论文报告相比 FlashAttention-1 约 2 倍加速，达到 A100 理论吞吐的 50%--73%。

## 方法与机制

### 减少 non-matmul 工作

FlashAttention-2 在在线 softmax 中维护未归一化的输出累积量，最后一次性除以归一化因子，避免在每个 K/V block 更新时重复缩放两部分输出。反向传播只保存每个 query row 的 logsumexp：

$$
L_i=m_i+\log(\ell_i),
$$

不再同时保存最大值和指数和。这些改动不改变输出，却减少了 GPU 上相对昂贵的逐元素、归约和非矩阵乘法操作。

### 沿序列长度并行

FlashAttention-1 的一个 thread block 通常负责一个 attention head，主要沿 batch 和 head 数量产生并行任务。当 batch 或 head 数较少、但序列很长时，GPU 的 streaming multiprocessor（SM）无法充分占用。

FlashAttention-2 的前向计算把不同 Q row blocks 分配给不同 thread blocks；反向计算把不同 K/V column blocks 分配给不同 thread blocks，并通过 atomic add 累积共享的 $dQ$。这样长序列本身也贡献并行度。

### warp 内工作划分

旧方案把 K/V 切给不同 warps。每个 warp 得到部分结果后，需要把中间输出写入 shared memory，再同步并合并。

FlashAttention-2 改为切分 Q，让每个 warp 生成自己负责的输出片段，同时共享 K/V。这样减少了中间结果的 shared memory 读写和 warp 间通信。block size 仍需根据 head dimension、register pressure 和 shared memory 容量调节。

### MQA 和 GQA

论文沿用多查询注意力（Multi-Query Attention，MQA）和分组查询注意力（Grouped-Query Attention，GQA）的 head indexing，避免在 HBM 中复制共享的 K/V。反向传播时，对隐式复制到多个 query heads 的 K/V 梯度进行求和。

## 实验与证据

实验主要在 A100 80GB SXM4 上进行，序列长度覆盖 512 到 16K，比较 causal/non-causal、head dimension 64/128 和多个实现 baseline。

| 实验 | 报告结果 |
|---|---|
| Attention kernel | 相比 FlashAttention-1 约 1.7--3.0 倍；相比 Triton 版约 1.3--2.5 倍；相比标准 PyTorch attention 约 3--10 倍 |
| A100 吞吐 | 最高约 230 TFLOPs/s，达到理论峰值的 73% |
| GPT-style 端到端训练 | 1.3B/2.7B 模型、2K/8K 序列上，相比 FlashAttention-1 最高约 1.3 倍，相比未使用 FlashAttention 的 baseline 最高约 2.8 倍 |
| A100 单卡训练吞吐 | 最高约 225 TFLOPs/s，约 72% model FLOPs utilization |

论文也确认，FlashAttention-2 仍保持精确输出和线性额外显存；改进来自工作组织与硬件利用率，而不是近似 attention 或缩短访问范围。

## 关键结论

1. 当 IO-aware attention 已经消除主要 HBM 瓶颈后，GPU 占用率、非矩阵乘法比例和 warp 间通信会成为下一层限制。
2. 长序列场景需要把 sequence length 纳入并行划分，否则 batch/head 维度可能提供不了足够的 thread blocks。
3. FlashAttention 的算法顺序和 kernel 工作分配是耦合的：在线 softmax 的统计量设计会影响反向保存内容和并行方式。
4. MQA/GQA 可以在同一个 kernel 中通过索引共享 K/V，无需把共享 head 复制到 HBM。

## 局限与疑问

- block size 仍需要按 head dimension 和 GPU 手动调参，论文没有给出通用自动调优方案。
- 论文的主要验证平台是 A100；Hopper 的专用异步指令和低精度 Tensor Core 没有纳入本版本设计。
- sequence-parallel backward 需要 atomic add 累积 $dQ$，其收益和冲突成本会受形状与硬件影响。
- benchmark 是 kernel 和训练吞吐测量，不能单独代表完整 serving 系统的 decode 延迟。

## 相关知识链接

- [[inference/attention-acceleration/flash-attention|FlashAttention]]
- [[inference/attention-acceleration/attention-kernels|Attention Kernels]]
- [[architecture/attention/multi-query-attention|Multi-Query Attention]]
- [[architecture/attention/grouped-query-attention|Grouped-Query Attention]]
- [[sources/papers/2022-flashattention|FlashAttention]]
