---
title: "GShard"
created: 2026-06-01
published: 2026-08-31
modified: 2026-08-31
type: source
status: processed
source_type: paper
area: sources
tags:
  - source
  - paper
  - moe
  - distributed-training
  - automatic-sharding
  - xla
source_url: https://arxiv.org/abs/2006.16668
paper_date: "2020-06"
paper_order: "16668"
---

# GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding

## 基本信息

- 论文：[GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668)
- 作者：Dmitry Lepikhin、HyoukJoong Lee、Yuanzhong Xu、Dehao Chen、Orhan Firat、Yanping Huang 等
- 发布信息：arXiv:2006.16668，v1 发布于 2020-06-30
- 研究对象：基于 XLA 的自动模型分片系统，以及大规模 multilingual MoE Transformer
- 主要实验：100 个语言到英语的 massively multilingual machine translation
- 相关主题：[[architecture/sparse-and-efficient/moe|Mixture of Experts]]、[[training/distributed-training/torch-distributed|torch.distributed]]、[[training/distributed-training/tensor-parallel|Tensor Parallel]]、[[training/distributed-training/pipeline-parallel|Pipeline Parallel]]

GShard 的重点不是提出一种新的 Transformer block，而是解决一个更基础的系统问题：当模型被拆到数百或数千个加速器上时，模型代码、tensor partition、跨设备通信和编译器如何协同工作，才能让开发者不必为每一种架构手写完整的 model-parallel implementation。

论文给出的答案由两部分组成：

1. 用 sparsely-gated Mixture of Experts（MoE）让模型容量随 expert 数量增长，而每个 token 只激活少数 experts；
2. 用轻量的 `replicate`、`split`、`shard` annotations 描述 tensor 的分布方式，再由 XLA SPMD partitioner 自动生成每个设备执行的程序和必要通信。

因此，GShard 应该被理解为一条完整的 co-design 链路：

```text
conditional computation
  -> larger total capacity at bounded per-token compute
  -> tensor sharding annotations
  -> XLA SPMD partitioner
  -> AllReduce / AllGather / AllToAll / CollectivePermute
  -> scalable execution on thousands of TPU devices
```

论文在 multilingual translation 上展示了这条链路的效果：600B 参数的 MoE Transformer 使用 2,048 个 TPU v3 加速器训练约 4 天，平均 BLEU 为 `44.3`，训练成本为 `22.4 TPU v3 core-years`；相比之下，100 个 bilingual baseline 需要约 `29 TPU core-years`，同一数据上的 2.3B dense Transformer 则需要约 `235.5 TPU core-years`。

## 研究问题

### 模型规模增长如何保持计算可行

如果直接增大 dense Transformer 的 depth 或 width，每个 token 都要经过更多参数，单步计算和显存会近似随模型规模增长。模型并行可以把矩阵切到多个设备，但会带来更多同步和通信，且网络的 sequential dependency 可能导致设备利用率下降。

GShard 关注的是一种不同的规模化方式：让模型拥有很大的 total parameters，但通过 conditional computation 只为每个 token 激活少数 sub-network。这样，模型容量可以继续增加，而单个 token 的主要计算路径不必同步增加。

### 模型代码如何与分片实现解耦

传统的 model parallel implementation 往往把设备数量、切分维度和通信逻辑直接写进模型代码。这样做的问题是：

- 改变模型结构时，底层 partitioning 逻辑也要重写；
- 不同 operator 的通信语义分散在各处，难以维护；
- 设备数量变化会改变 graph representation 和编译成本；
- 手动处理 uneven partition、reshape、padding 和 backward graph 很容易出错。

GShard 希望让开发者仍然用逻辑上的 full-size tensor 写模型，只在少数关键 tensor 上标注期望的 sharding，剩余的 partition propagation、resharding 和通信插入交给 compiler。

### 千卡规模下如何避免编译图爆炸

如果用 MPMD（Multiple Program Multiple Data）为不同 partition 生成不同程序，设备数增加时 graph node 和 communication edge 会快速增长。GShard 采用 SPMD（Single Program Multiple Data）：所有设备运行同一个足够通用的 program，每个设备根据自己的 partition view 处理局部数据。

这使编译复杂度不再直接随 partition 数量复制，同时也要求 partitioner 能处理 static shape、uneven partition、halo exchange 和不同 operator 的通信模式。

## 核心主张

1. **Conditional computation 可以把模型容量与每 token 计算部分解耦。** GShard 用 sparsely-gated MoE 替换部分 Transformer FFN，每个 token 最多路由到两个 experts。
2. **轻量 sharding annotation 足以表达多种并行布局。** `replicate`、`split` 和 `shard` 不改变用户代码中的逻辑 tensor shape，只向 compiler 提供分布信息。
3. **SPMD partitioning 比为每个设备生成专用程序更适合千卡规模。** XLA 将 annotated full-size operator 转成 partition-sized operator，并插入必要的 collective communication。
4. **MoE 的 scale-up 受到 capacity bottleneck、负载均衡和通信的共同约束。** 单纯增加 experts 不会线性带来收益，depth、expert 数量、任务间 transfer 和设备通信必须一起考虑。
5. **在 massively multilingual translation 中，增加 depth 和 experts 都能提升质量，但作用不同。** 增加 depth 对高资源和低资源语言都有较稳定收益；增加 experts 主要缓解高资源任务的 capacity bottleneck，同时可能减少共享子网络带来的 positive transfer。
6. **GShard 能够让 600B MoE 模型在实际集群上运行。** 但论文的 1T 参数深层实验出现 numerical stability 和 trainability issues，说明系统可扩展不等于训练优化已经解决。

## 方法总览

### MoE Transformer

GShard 将 Transformer 中每隔一层的 position-wise FFN 替换为 MoE layer。每个 expert 都是一个与普通 FFN 形状相同的两层网络，router 对每个 token 计算 expert scores，选择最多两个 experts，再把输出按 gating weights 加权合并。

简化形式为：

$$
y_s = \sum_{e=1}^{E} G_{s,e}\, FFN_e(x_s)
$$

其中 $G_{s,e}$ 大部分为零，因此 token 只触发少数 experts。模型的 total parameters 随 $E$ 增加，但单 token 的 active sub-network 大致保持不变。

### Top-2 gating 的四个机制

#### Expert capacity

每个 expert 在一个 batch 或 local group 中能够接收的 token 数存在上限。若某 token 选中的两个 experts 都已经超过 capacity，则这个 token 被视为 overflow token，MoE layer 不再为它执行 expert computation，而是通过 residual connection 直接传递原表示。

假设 batch 有 $N$ 个 tokens、每个 token 最多路由到两个 experts、共有 $E$ 个 experts，则单个 expert 的 capacity 大致与 $O(N/E)$ 成正比。

#### Local group dispatching

GShard 将 batch 均匀划分为 $G$ 个 groups，每个 group 独立进行 routing，并分配每个 expert 的 fractional capacity `2N/(G·E)`。这样既可以并行处理 groups，又能让每个 local group 遵守统一的 expert capacity。

#### Auxiliary loss

只使用 top-k score 容易让 router 长期偏向少数 experts。GShard 用 auxiliary loss 鼓励 token dispatch 比较均衡：

$$
\mathcal{L}=\ell_{nll}+k\ell_{aux}
$$

其中 $c_e/S$ 表示 group 内发送到 expert $e$ 的 token 比例，$m_e$ 是该 expert 的 mean gate。由于 top-2 dispatch 本身不可微，论文用 $m_e(c_e/S)$ 作为可优化的近似项。

#### Random routing

第二候选 expert 的 gate weight 如果很小，可以不必总是发送。GShard 以与第二个 gate 权重成比例的概率随机将 token dispatch 到第二个 expert，在控制 capacity 的同时减少低收益的第二路计算。

### Sharding 与 MoE 的交替布局

GShard 的一个重要场景是：普通 Transformer layers 的 weights 可以 replicate 到所有 devices，而巨大的 MoE expert weights 需要 shard 到多个 devices。模型因此在 replicated layout 和 expert-sharded layout 之间交替。

```text
non-MoE Transformer layer
  weights: replicated
  batch: split across devices

MoE layer
  experts: split across devices
  tokens: dispatch to owning expert partitions
  outputs: combine back to original token layout
```

这种布局是 GShard 的系统难点来源：MoE 不是只把一个大矩阵切开，而是要把按 batch/group 布局的 token 重新组织为按 expert 布局的 buffer，再把 expert output 重新合并回原来的 token 位置。

## GShard Annotation API

### `replicate`、`split` 与 `shard`

GShard 使用三个核心 API：

```python
# 只表达分布方式，不改变逻辑 tensor shape
inputs = split(inputs, split_dimension=0, num_partitions=D)
wg = replicate(wg)

# 在 dispatch 后把 partition 从 group 维切换到 expert 维
dispatched_inputs = split(
    dispatched_inputs,
    split_dimension=0,
    num_partitions=D,
)
```

- `replicate(tensor)`：在各个 partitions 上保留完整副本，适合较小的 gating weights 或普通 Transformer weights；
- `split(tensor, split_dimension, num_partitions)`：沿一个维度切分 tensor；
- `shard(tensor, device_assignment)`：进一步指定多个维度如何切分，以及各 slice 放在哪些 device 上。

annotation 只增加分布约束，不改变用户代码看到的 full logical shape，也不要求用户手动处理 uneven partition。这样模型描述与 parallel implementation 保持分离。

### Automatic propagation 与 manual override

用户不需要标注每个 tensor。GShard 会从关键 operator 的 annotations 出发，通过 iterative data-flow analysis 在相邻 operator 之间传播 sharding，并尽量选择能够减少 resharding 的布局。

在 MoE 实现中，典型逻辑是：

1. inputs 沿 group `G` 维度 split；
2. gating weights replicate；
3. gates 和 dispatch mask 在 group layout 下计算；
4. dispatch 后的 tensor 沿 expert `E` 维度重新 split；
5. expert FFN 在本地执行；
6. outputs 再按 token/group layout combine。

GShard 也允许在自动分片中插入手动 partition。例如 dispatch 可以用 one-hot `einsum` 表达，也可以把索引 tensor 手动切分后用 local `Gather`，再回到自动 SPMD partitioning。原因是 operator 的一般语义可能不足以表达用户掌握的 index bounds 或数据局部性信息。

## XLA SPMD Partitioner

### SPMD 与 MPMD

XLA 将计算表示为 dataflow graph：节点是 operators，边是 tensors。SPMD partitioner 接收带 sharding annotations 的 full-size graph，将每个 operator 转换为 partition-sized operator，并生成一份由所有 devices 执行的统一 program。

```text
logical full-size graph
  -> sharding propagation
  -> per-operator partitioning
  -> insert communication / resharding
  -> one SPMD program for all devices
```

论文强调，GShard 使用 XLA HLO 作为 partitioner 的目标层，而不是直接为 TensorFlow、PyTorch 等前端分别实现所有 operator 规则。前端先把模型 lowering 到较小且规整的 HLO operator set，partitioner 再集中处理分片和通信。

### 四种核心通信原语

GShard 使用的 collective communication 与当前 GPU 训练系统中的概念非常接近，但当时主要面向 TPU/XLA：

| Primitive | 作用 | GShard 中的典型用途 |
|---|---|---|
| `CollectivePermute` | 按 source-destination pairs 发送数据 | 改变 device order、halo exchange、循环切片 |
| `AllGather` | 拼接所有 participants 的 tensor | 将 sharded tensor 变成 replicated tensor |
| `AllReduce` | 对所有 participants 做逐元素 reduction | 合并 contracting dimension 上的 partial results |
| `AllToAll` | 每个 participant 切分输入并互相交换 pieces | 在 group/token layout 与 expert layout 之间 reshard |

因此，GShard 并不是绕开通信原语完成自动分片。它的自动化价值在于：根据 tensor layout 和 operator semantics 自动决定何时需要哪种通信，并把通信插入由 compiler 管理的 SPMD program。

### Einsum 的三种分片情形

GShard 以 Einsum / Dot 为重点案例。一个 Einsum 的维度可以分为 batch dimensions、contracting dimensions 和 non-contracting dimensions。

#### Resharding

MoE dispatch 需要把按 group `G` 分片的结果转换成按 expert `E` 分片的结果。GShard 先在本地完成 batch-parallel Einsum，再用 `AllToAll` 将结果 reshard 到 expert layout。

#### Accumulating partial results

如果两个输入沿 contracting dimension 分片，每个 device 只能得到 partial result，最后需要 `AllReduce` 将这些局部结果逐元素相加，恢复 full result。

#### Sliced computation

如果两个 operands 沿不同 non-contracting dimensions 分片，直接计算可能要求复制一个过大的 operand。GShard 可以使用 loop 一次计算一个 output slice，用 `CollectivePermute` 交换所需的 input slice，从而避免任何 device 持有 full-sized operand。

## 一般 operator 的技术问题

### Static shape 与 uneven partition

XLA 要求 tensor shape 静态，但原始维度未必能被 partition count 整除。GShard 会把 partition shape round up，并对 padding region 填入 operator 所需的 identity value 或用 mask 屏蔽。

例如，一个长度为 15 的维度切成 2 partitions 时，某个 partition 会多出一个无效位置。对于 Reduce-Add，额外位置必须填零；对于其他 operator，则需要使用合适的 padding 和 mask，避免无效数据影响结果。

### Halo exchange

卷积、ReduceWindow、Slice、Pad、Reverse 和某些 Reshape 会让相邻 partition 需要彼此边界附近的数据。GShard 将这种局部数据交换称为 halo exchange，并使用 `CollectivePermute` 在相邻 partitions 之间发送 halo。

难点在于不同 partition 的 halo size 可能不同，而 CollectivePermute 的 shape 需要静态。GShard 的处理方式是先交换最大 halo，再用 `DynamicSlice` 截出每个 partition 实际需要的区域，并用 padding/mask 处理无效部分。

### Compiler optimization

自动分片会引入 slice、padding、concat、mask 和 halo exchange 等 data-formatting operators。GShard 依靠 XLA fusion 和 code motion 尽量隐藏这些操作的额外开销，论文报告在其卷积等实验中，这类 formatting overhead 通常可以做到接近可忽略。

## M4 Multilingual Translation 实验

### 任务与数据

论文使用 massively multilingual machine translation 作为压力测试。训练数据来自 web mining，覆盖 100 种语言与英语之间的双向翻译，共约 25B training examples；主评测关注 100 个语言到英语的方向，约使用 13B training examples。

这个任务同时包含高资源和低资源语言：高资源语言可能拥有数十亿 examples，低资源语言只有数万 examples，语言间数据量呈明显 power-law imbalance。单个 multilingual model 可以通过共享参数给低资源语言带来 positive transfer，但共享容量也可能让高资源任务相互干扰，形成 capacity bottleneck。

作者将每个语言 pair 的 tuned bilingual Transformer 作为 baseline，并报告单一 multilingual model 相对各自 bilingual baseline 的 `Delta BLEU`。另外使用一个在同一数据上训练的 dense 96-layer Transformer 作为 GPipe baseline。

### 模型族

GShard 通过两个轴扩大 MoE Transformer：

- 增加 encoder-decoder depth；
- 增加每个 MoE layer 的 expert 数量。

| Model | Experts / layer | Encoder + decoder layers | Total parameters | TPU v3 cores |
|---|---:|---:|---:|---:|
| MoE(128E, 12L) | 128 | 12 | 12.5B | 128 |
| MoE(128E, 36L) | 128 | 36 | 37B | 128 |
| MoE(512E, 12L) | 512 | 12 | 50B | 512 |
| MoE(512E, 36L) | 512 | 36 | 150B | 512 |
| MoE(2048E, 12L) | 2048 | 12 | 200B | 2048 |
| MoE(2048E, 36L) | 2048 | 36 | 600B | 2048 |

每隔一个 Transformer layer 放置 MoE layer。主要实验使用 float32 weights 和 activations 以保证稳定性；作者还试验过 60-layer、约 1T-parameter 的模型，但遇到 numerical stability 和 trainability issues，因此没有将其作为主要结果。

## 实验结果

### Depth 带来稳定收益

在固定 expert 数量的情况下，将 depth 从 12 layers 增加到 36 layers，会同时改善高资源和低资源语言。平均而言，depth 增加带来约 2 到 3 BLEU points 的 additive gain。

这一结果说明，在 MoE 中增加容量不只有“增加 experts”这一条轴。只增加 experts 主要扩展横向的稀疏容量；增加 depth 则扩大了连续变换链路和跨语言共享路径，两者不能用 total parameters 简单替代。

### Experts 首先解决 capacity bottleneck

在固定 12-layer depth 的情况下，expert 数从 128 增加到 512，平均 BLEU 提升约 `3.3`；从 512 增加到 2048，进一步提升约 `1.3`，出现明显 diminishing returns。

作者推测，对于当前的语言数量、数据规模和模型参数化，capacity bottleneck 主要位于 128 到 512 experts 之间。超过这个区间后，模型继续增加 experts 的边际收益下降。

### 高资源与低资源语言的收益不同

增加 experts 对高资源语言的改善更明显，因为高资源任务需要更多独立容量来减少相互干扰。低资源语言已经能够从 multilingual shared sub-network 获得 positive transfer，继续增加 experts 可能减少共享比例，因此 transfer gain 会趋于饱和。

对低资源任务，deep-dense model 的共享程度更高，通常比 shallow MoE 更有利。论文还指出，一个包含 128 experts、但更深的 36-layer MoE，可以达到与 128-layer dense model 相近的低资源 transfer quality，而参数规模更大但每 token 计算仍具有稀疏性。

### 代表性质量与成本结果

| Model | Parameters | Avg. BLEU | Avg. Delta BLEU | TPU core-years | Training days |
|---|---:|---:|---:|---:|---:|
| MoE(128E, 12L) | 12.5B | 36.7 | 5.9 | 1.9 | 5.4 |
| MoE(128E, 36L) | 37B | 39.0 | 8.2 | 6.1 | 17.3 |
| MoE(512E, 12L) | 50B | 40.0 | 9.2 | 4.9 | 3.5 |
| MoE(512E, 36L) | 150B | 43.7 | 12.9 | 15.5 | 11.0 |
| MoE(2048E, 12L) | 200B | 41.3 | 10.5 | 7.5 | 1.4 |
| MoE(2048E, 36L) | 600B | **44.3** | **13.5** | 22.4 | 4.0 |
| Dense T(96L) | 2.3B | 36.9 | 6.1 | 235.5 | 42.0 |

这里的对照需要同时看 model capacity、训练 core 数和 batch size。600B MoE 并不是用 2.3B dense model 的相同 per-device 计算预算得到的直接 apples-to-apples 对照，但结果清晰展示了 conditional computation 的实际 scale/quality/cost trade-off。

## Training Efficiency

### Sample efficiency

论文用达到指定 training cross-entropy 所需的 token 数来比较 sample efficiency。固定 expert 数后，36-layer 模型通常只需要 12-layer 模型约三分之一到二分之一的 tokens 达到同一 loss threshold：

| Model | Tokens to CE 0.7 | Tokens to CE 0.6 | Tokens to CE 0.5 |
|---|---:|---:|---:|
| MoE(2048E, 36L) | 82B | 175B | 542B |
| MoE(2048E, 12L) | 176B | 484B | 1,780B |
| MoE(512E, 36L) | 66B | 170B | 567B |
| MoE(512E, 12L) | 141B | 486B | - |
| MoE(128E, 36L) | 321B | 1,074B | - |
| MoE(128E, 12L) | 995B | - | - |

在固定 depth 下，expert 从 128 增加到 512 也显著减少达到 CE `0.7` 所需的 token 数；从 512 到 2048 后，sample efficiency 的变化趋于平缓。这与 capacity bottleneck 的质量实验相互印证。

### 600B 模型的 wall-clock scaling

当 expert 数从 128 增加到 2048，即模型 capacity 增加 16 倍时，per-device step execution time 只增加约 `1.7x`。在 128 experts 时，模型达到 roofline 的比例超过 70%；在 2048 experts 时仍达到约 48%。

这不是说 600B 模型的总计算量只增加 1.7 倍，而是说在增加设备、分片 experts 和维持每设备 token 数量的条件下，单设备执行时间的增长远小于 dense capacity 的线性增长。

## Memory 与 Communication

### Per-device memory

SPMD partitioning 后，expert 数量增加时，每个 device 主要保存三类状态：

- replicated weights，例如普通 Transformer FFN；
- distributed weights，例如 MoE experts；
- forward/backward 所需 activations。

在固定 layer depth 下，增加 experts 不会显著增加单 device 的 weight memory 和 activation memory，因此 per-device memory 对 expert 数量近似为 `O(1)`。但 layer 数增加仍会让 weight 和 activation memory 近似线性增长。

当 activation memory 超过设备容量时，GShard 使用 compiler-based rematerialization，在 backward 时重新计算部分 activation。论文报告 36-layer 和 60-layer 模型中约有 28% 和 34% 的 cycles 用于 recomputation；12-layer 和 24-layer 模型不需要 rematerialization。

### MoE dispatch 的通信代价

MoE dispatch/combine 是最重要的通信部分，主要由 `AllToAll` 完成。随着 expert 数从 128 增加到 2048，dispatch/combine execution time 约增加 `3.75x`，其在 MoE 和相邻 Transformer layers 总执行时间中的比例从约 `16%` 增加到 `36%`。

因此，MoE 的稀疏计算收益不能只看 FFN FLOPs。token dispatch、expert output combine、设备拓扑和 collective implementation 会成为规模化的主要瓶颈。

### Communication microbenchmarks

在论文使用的二维 TPU device network 上：

- `AllReduce` 的执行时间对 partition 数量近似不敏感；
- `AllToAll` 随 partition 数量增加，但大致按 `O(sqrt(D))` 增长；
- 当 partition 从 16 增加到 2048，即增长 128 倍时，AllToAll 执行时间约增长 9 倍；
- `AllGather` 在固定每-device input size 时，其通信量约随 partition 数量线性增加；
- `CollectivePermute` 是一对一通信，如果 device assignment 让 source/destination 足够接近，其固定 input size 的成本可以近似为 `O(1)`。

这里的复杂度是针对特定 TPU topology、固定每-device data size 和论文实现的经验分析，不能直接当作任意 GPU/NCCL 集群的普遍复杂度结论。

## 结果如何解读

### 关键收益来自组合，而非某一个 API

GShard 的效果不能只归因于 automatic sharding，也不能只归因于 MoE。MoE 提供了可扩展的 conditional computation；annotations 提供了模型与布局之间的接口；SPMD partitioner 负责 per-operator transformation；collective communication 负责实际的数据移动；XLA fusion 和 topology-aware device assignment 决定运行效率。

其中任何一层失效都会改变结论：

- router 失衡会让 expert capacity 和 device load 崩溃；
- annotation 不准确会产生不必要的 resharding；
- AllToAll 实现不高效会吃掉 MoE 的 sparse compute 收益；
- MPMD graph 复制会让千卡规模的编译成本不可接受；
- 设备拓扑不匹配会让理论通信复杂度无法兑现。

### Total parameters 不是有效容量的完整描述

论文最后强调，单纯统计参数量不能完整解释 massively multilingual model 的有效 capacity。增加 experts 可以减少 task interference，但也会减少参数共享；dense-deep model 虽然 total parameters 更小，却可能在低资源语言上提供更强的 transfer。模型比较必须同时考虑：

- total parameters；
- per-token active computation；
- shared 与 specialized sub-network 的比例；
- 任务间数据不平衡；
- capacity bottleneck；
- positive / negative transfer；
- 训练和推理中的通信成本。

## 局限与疑问

### 主要验证场景是机器翻译

论文的核心结果来自 100-language-to-English translation。这个任务天然适合观察多任务共享和 capacity bottleneck，但不等价于 decoder-only LLM、long-context reasoning 或通用 agent。MoE 路由在不同 token 分布、sequence length 和 reward-driven training 中可能表现不同。

### 1T 模型的训练稳定性仍未解决

论文提到 60-layer、约 1T-parameter 模型可以在仔细诊断下训练，但存在数值稳定性和 trainability issues，因此没有纳入主要可复现实验。这个边界很重要：自动分片解决了“怎么放”和“怎么通信”，不自动解决优化器、precision、初始化和超参数问题。

### GShard 的自动 sharding 不是全局最优求解器

论文使用 iterative data-flow analysis 传播 sharding，并以减少 resharding 为主要倾向。作者明确把 integer programming、machine learning based layout search 等更强的 automatic sharding assignment 留作未来工作。

### Capacity overflow 会跳过 expert computation

当 token 所选 experts 都超过 capacity 时，该 token 通过 residual connection 直接进入下一层。这个机制保证了静态 buffer 和负载上限，但 overflow token 没有得到当前 MoE layer 的 expert transformation。论文的 quality result 依赖其具体 capacity、grouping、routing 和数据分布，不能假设 token dropping 或 overflow 在所有 MoE 中都无害。

### 低精度和其他硬件后端的迁移

主实验使用 float32 weights 和 activations。论文虽然讨论了 bfloat16 的额外实验，但深层 1T 模型仍有稳定性问题。GShard 的分片抽象可以迁移到其他 IR 或 framework，但 communication performance、kernel fusion、device topology 和 numerical behavior 需要重新验证。

## 对当前训练系统学习的启发

### 理解并行训练要同时追踪三种对象

阅读 GShard 或现代 Megatron/FSDP 源码时，不能只看“用了哪个 collective”。至少要同时追踪：

1. **逻辑 tensor shape**：模型代码认为 tensor 是什么形状；
2. **分片 layout**：每个 rank 实际持有哪一段、按哪个维度切分；
3. **通信语义**：为什么需要 AllReduce、AllGather、AllToAll 或 P2P，通信前后 layout 如何变化。

GShard 的 annotation 设计尤其清楚地展示了：通信不是独立的黑盒函数，而是 tensor layout 变化的实现。当前 GPU 训练中由 `torch.distributed` / NCCL 执行的 collective，和 GShard/XLA 中的 collective 在抽象层面是同一类问题，只是 backend、设备拓扑和编译方式不同。

### MoE 系统的重点是 dispatch/combine

如果要理解现代 MoE implementation，应把 attention/FFN 之外的 token dispatch 路径单独画出来：

```text
token hidden states
  -> router scores
  -> top-k expert selection
  -> capacity / token position assignment
  -> AllToAll dispatch
  -> local expert computation
  -> AllToAll combine
  -> residual / next Transformer layer
```

GShard 的 group `G` 到 expert `E` 的 reshard，是后来 expert parallel 中 token dispatch/combine 的重要概念前身。

### 自动化不会取消系统设计

GShard 允许开发者少写 partition code，但仍然需要明确：哪些 weights replicate、哪些 experts shard、哪些 tensor 维度切分、设备如何映射、哪些 operator 会触发 resharding。自动分片降低的是实现负担，不是对 tensor layout 和硬件拓扑的理解要求。

## 关键结论

1. GShard 通过 MoE 的 conditional computation 与 XLA 的 automatic SPMD sharding，展示了千卡级 giant model 的一条可运行路线。
2. `replicate`、`split`、`shard` annotations 将模型逻辑与设备布局解耦，compiler 再负责 sharding propagation、per-operator transformation 和通信插入。
3. MoE 的主要通信不是普通参数同步，而是 token 根据 router 结果在 expert partitions 之间的 dispatch/combine，核心 collective 是 `AllToAll`。
4. `AllReduce` 用于合并 contracting dimension 的 partial results，`AllGather` 用于从 sharded layout 恢复 replicated layout，`CollectivePermute` 用于局部交换、device reorder 和 halo exchange。
5. 增加 depth 和 experts 是不同的 capacity scaling 轴：depth 更有利于全局 transfer，experts 更直接缓解高资源任务的 capacity bottleneck。
6. 在论文的 M4 任务中，600B MoE 模型实现了优于 dense baseline 的质量/成本折中，但 1T 深层模型仍存在 trainability 和 numerical stability 问题。
7. MoE 的扩展性不能只用 total parameters 或 active FLOPs 衡量，还要纳入 routing balance、overflow、device topology、AllToAll 成本和 shared/specialized capacity。
8. GShard 的核心遗产不是某个 TPU API，而是一个仍然适用于现代训练系统的分层方法：逻辑模型、tensor layout、compiler transformation 和 collective communication 必须联合设计。

## 相关知识链接

- [[architecture/sparse-and-efficient/moe|Mixture of Experts]]
- [[training/distributed-training/torch-distributed|torch.distributed]]
- [[training/distributed-training/tensor-parallel|Tensor Parallel]]
- [[training/distributed-training/pipeline-parallel|Pipeline Parallel]]
- [[training/distributed-training/megatron|Megatron 与 3D 并行]]
- [[sources/papers/2021-switch-transformer|Switch Transformer]]
- [[sources/papers/2024-deepseekmoe|DeepSeekMoE]]
- [[sources/papers/2024-deepseek-v3|DeepSeek-V3 Technical Report]]
