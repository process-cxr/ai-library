---
title: "GPipe"
created: 2026-05-31
published: 2026-08-26
modified: 2026-08-26
type: source
status: processed
source_type: paper
area: sources
tags:
  - source
  - paper
  - distributed-training
  - pipeline-parallel
  - model-parallel
  - micro-batch
  - scaling
source_url: https://arxiv.org/abs/1811.06965
paper_date: "2018-11"
paper_order: "06965"
---

# GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism

## 基本信息

| 字段 | 内容 |
|---|---|
| 来源 | arXiv:1811.06965v5，2019-07-25 |
| 标题 | [GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism](https://arxiv.org/abs/1811.06965) |
| 作者 | Yanping Huang, Youlong Cheng, Ankur Bapna, Orhan Firat, Mia Xu Chen, Dehao Chen, HyoukJoong Lee, Jiquan Ngiam, Quoc V. Le, Yonghui Wu, Zhifeng Chen |
| 机构 | Google |
| 首次公开 | 2018-11，arXiv v1 |
| 代码/框架 | [Lingvo](https://github.com/tensorflow/lingvo) |
| 相关 topic | [[training/distributed-training/pipeline-parallel|Pipeline Parallel]]，[[training/distributed-training/megatron|Megatron 与 3D 并行]]，[[training/distributed-training/tensor-parallel|Tensor Parallel]]，[[training/optimization/gradient-checkpointing|Gradient Checkpointing]] |

GPipe 是一个面向大型神经网络的 Pipeline Parallelism library。它把可以表示为 layer sequence 的网络切分成多个连续的 cell，每个 cell 放在一个 accelerator 上，再把一个 mini-batch 拆成多个 micro-batches，使不同 accelerator 可以同时处理不同 micro-batch。

论文的核心设计是：**流水线执行可以提高设备利用率，但参数更新仍然在整个 mini-batch 的所有 micro-batch 完成后同步进行。** 这使 GPipe 在减少 pipeline bubble 的同时保持了 synchronous gradient descent 的语义，不需要为异步更新维护多份 stale weights。

## 研究问题

### 单卡无法容纳不断增大的模型

模型规模增长可以提升图像分类、机器翻译和语言建模能力，但参数、activation 和中间计算也会超过单个 accelerator 的显存容量。简单地把模型放到多张设备上并不能自动得到高吞吐训练，因为不同层之间存在严格的前后依赖。

如果采用最直接的 layer-wise model parallelism：

```text
device 0: layers 0 ... k
device 1: layers k+1 ... m
device 2: layers m+1 ... n
```

一个完整 mini-batch 在任意时刻通常只有一个 device 真正在计算，其他 device 处于等待状态，硬件利用率很低。

### 现有方案的取舍

当时的模型并行方案通常在以下三点之间取舍：

- 能支持的模型规模；
- 对网络架构和任务的通用性；
- 训练效率与工程复杂度。

有些方案通过专用 compiler 或 framework 表达分布式 tensor computation，有些方案通过异步 pipeline 提高利用率，但异步更新可能带来 weight staleness，需要保存多个参数版本，反而增加显存和实现复杂度。

GPipe 希望提供一种更通用的方案：只要网络能表示成一串 layer，就能自动切分、流水执行、重计算 activation，并通过同步梯度更新保持训练结果的一致性。

## 核心主张

论文的核心主张可以概括为以下几项。

1. **Pipeline Parallelism 可以通过 micro-batch splitting 获得较高设备利用率。** 一个 mini-batch 被拆成多个 micro-batches，不同 partition 可以并行处理不同 micro-batch。
2. **同步梯度更新可以与 pipeline 并行结合。** 所有 micro-batch 使用同一组模型参数完成 forward/backward，梯度累积到整个 mini-batch 完成后再统一更新。
3. **Rematerialization 可以显著降低 activation memory。** forward 阶段只保存 partition 边界的 activation，backward 时重新计算 cell 内部的 forward。
4. **Pipeline 的通信量较低。** 设备之间只需要在 partition 边界传递 activation 和 gradient，不需要像通用 distributed tensor computation 那样在每次矩阵运算后频繁 collective communication。
5. **方法具有任务和架构通用性。** 论文在 AmoebaNet 图像分类和多语言 Transformer 翻译上验证了相同的 GPipe 思路。
6. **模型容量可以近似随 accelerator 数量扩展。** 当 micro-batch 数量足够多且各 partition 计算均衡时，pipeline bubble 可以变得很小。

## 方法与机制

### Layer Sequence 与 Cell Partition

GPipe 假设网络可以表示为 $L$ 个连续 layer：

$$
L_i=(f_i,w_i,c_i)
$$

其中 $f_i$ 是 forward function，$w_i$ 是参数，$c_i$ 是可选的计算成本估计函数。

给定 $K$ 个 model partitions，GPipe 把连续 layer 分组成 $K$ 个 cell：

$$
p_k=(L_i,\ldots,L_j)
$$

cell $p_k$ 的 forward function 是内部 layer 的复合函数：

$$
F_k=f_j\circ\cdots\circ f_{i+1}\circ f_i
$$

对应的 backward function $B_k$ 可以由自动微分得到。cell 的估计计算成本是内部 layer cost 的总和：

$$
C_k=\sum_{l=i}^{j}c_l
$$

分区算法尽量让各个 cell 的计算成本接近，以减少某个 accelerator 因为层太多或计算太重而成为整体 pipeline 的瓶颈。

GPipe 对用户暴露的核心配置很少：

- model partitions 数量 $K$；
- micro-batches 数量 $M$；
- layer sequence 和每层定义。

partition 边界的 activation transfer 由 library 自动插入。

### Naive Model Parallelism 的空闲问题

如果不做 batch splitting，整个 mini-batch 只能依次通过各个 cell：

```text
time ->
Cell 0: forward(batch)
Cell 1:                  forward(batch)
Cell 2:                                      forward(batch)
Cell 3:                                                  forward(batch)
```

在 Cell 0 开始工作时，后面的 cell 没有输入；在 Cell 0 进入 backward 时，后面的 cell 也可能处于等待状态。设备之间形成很大的空闲区间，这就是 pipeline bubble。

### Micro-Batch Pipeline

GPipe 将大小为 $N$ 的 mini-batch 均匀拆成 $M$ 个 micro-batches，每个 micro-batch 大小为 $N/M$。不同 micro-batch 可以同时处于不同 cell：

```text
time ->
Cell 0: mb1 F  mb2 F  mb3 F  mb4 F  ...
Cell 1:        mb1 F  mb2 F  mb3 F  ...
Cell 2:               mb1 F  mb2 F  ...
Cell 3:                      mb1 F  ...
```

进入 backward 后，各 cell 继续处理不同 micro-batch 的反向计算。流水线 warmup 和 drain 阶段仍然存在空闲，但当 $M$ 足够大时，空闲部分相对于 steady-state 计算的占比会变小。

### Synchronous Gradient Update

GPipe 的关键不是单纯把 forward 切成流水线，而是保持同步更新语义：

```text
固定模型参数 W
  -> micro-batch 1 forward / backward
  -> micro-batch 2 forward / backward
  -> ...
  -> 累积所有 micro-batch 的梯度
  -> 一次 optimizer update 得到 W'
```

同一个 mini-batch 内所有 micro-batch 使用同一组参数完成计算。只有全部 micro-batch 的梯度都准备好后，才对每个 partition 的参数执行更新。这样，改变 partition 数量或 micro-batch 数量不会引入异步参数版本造成的额外优化差异。

这与异步 pipeline 的差异非常重要。异步 pipeline 可能在前一批数据尚未完成时更新部分参数，后续 forward 看到的是不同版本的 weights；GPipe 用同步更新避免了这种 weight staleness。

### Activation Rematerialization

Pipeline 并行如果保存每个 micro-batch、每层的 activation，activation memory 仍然可能很高。GPipe 使用 rematerialization，也称 activation recomputation：

- forward 时每个 accelerator 只保存 cell 边界的输出 activation；
- backward 需要 cell 内部 activation 时，重新执行该 cell 的 composite forward function $F_k$；
- 使用重算出的 activation 完成该 cell 的 backward。

如果 $N$ 表示整个 mini-batch 大小，$L$ 表示 layer 数，$K$ 表示 partition 数，$M$ 表示 micro-batch 数，论文给出的峰值 activation memory 量级近似为：

$$
O\left(N+\frac{L}{K}\cdot\frac{N}{M}\right)
$$

相比没有 partition 和 rematerialization 时的 $O(NL)$，每个设备需要保存的中间状态显著减少。代价是 backward 阶段增加了 forward 重计算 FLOPs。

### Pipeline Bubble

partition 数量增加会拉长 warmup 和 drain，micro-batch 数量增加则可以摊薄这些固定开销。论文实验中观察到，当：

```text
M >= 4 * K
```

bubble overhead 通常已经接近可以忽略的程度。

因此，$M$ 不能只根据 global batch size 决定，还要和 pipeline depth $K$ 一起设计。micro-batch 太少时，设备会频繁等待；micro-batch 太多时，activation buffering、调度和通信次数增加。

### Partition Boundary Communication

GPipe 的设备间通信主要发生在相邻 partition 边界：

```text
Cell k -- activation --> Cell k+1   (forward)
Cell k <-- gradient --- Cell k+1   (backward)
```

通信量与 micro-batch size、sequence length、hidden size 和边界 tensor shape 有关，但不需要对每个矩阵乘法的 partial output 做全局同步。因此，GPipe 在没有高速 accelerator interconnect 的设备上也可能保持较好的扩展性。

这与 Tensor Parallel 的通信模式不同：Tensor Parallel 在单层内部频繁进行 collective communication；GPipe 的通信更像相邻 stage 之间的 point-to-point activation transfer。

### Batch Normalization 的特殊处理

普通 layer-wise pipeline 中，每个 micro-batch 只看到一部分 batch。若网络包含 BatchNorm，直接在 micro-batch 上计算统计量会改变训练与评估语义。

GPipe 的处理是：

- 训练时在 micro-batch 上计算 BatchNorm 的 sufficient statistics，必要时跨 replicas 处理；
- 同时跟踪整个 mini-batch 的 moving average；
- evaluation 时使用整个 mini-batch 统计量对应的 moving average。

这说明 micro-batch splitting 并不是对所有 layer 都完全透明。任何跨 batch 聚合的算子都需要额外处理，才能保证拆分前后的行为一致。

## 训练设置与容量分析

### 支持的模型容量

论文在 8GB NVIDIA GPU 和 16GB Cloud TPUv3 上比较 naive model parallel、pipeline partition 和 rematerialization 对最大可训练模型规模的影响。

对于 AmoebaNet，单 accelerator 的 naive 设置最多支持约 82M 参数；使用 8 个 pipeline partitions 后，最大支持约 1.8B 参数，约为 25 倍。由于 AmoebaNet 不同 layer 的参数量和计算量不均衡，容量没有完美线性增长。

对于 Transformer，单 accelerator 的 naive 设置约支持 282.2M 参数；128 个 pipeline partitions 后可以支持约 83.9B 参数，约为 298 倍。Transformer 的每一层结构更均匀，因此模型容量随 partition 数量更接近线性增长。

### Throughput 与 micro-batch 数量

论文使用 normalized training throughput 比较不同 partition 数量和 micro-batch 数量。

Transformer 在 $M=1$ 时几乎没有 pipeline parallelism，$K=2,4,8$ 的 throughput 约为 1.0、1.07、1.3；当 $M=32$ 时，三种设置约为 1.8、3.4、6.3，接近随设备数量线性增长。

AmoebaNet 的对应结果约为 1.21、1.84、3.48。它的 scaling 略低于 Transformer，主要原因是不同 layer 的计算量分布不均衡，partition balance 更困难。

论文还在没有 NVLink 的多张 P100 GPU 上测试了通信成本。固定 $M=32$ 后，8-way GPipe 对 AmoebaNet 达到约 2.7 倍 speedup，对 Transformer 达到约 3.3 倍 speedup。这个结果支持论文的判断：只在 partition boundary 传递 activation，能够降低对高速互联的依赖。

## 实验与证据

### AmoebaNet 图像分类

论文首先在 ImageNet-2012 上训练 AmoebaNet，验证 GPipe 对 convolutional architecture 的通用性。

- 输入分辨率：$480\times480$；
- 模型：AmoebaNet-B(18, 512)；
- 参数量：557M；
- pipeline partitions：4；
- top-1 validation accuracy：84.4%；
- top-5 validation accuracy：97%。

这个结果说明 GPipe 并不要求模型必须是 Transformer。只要网络能被表示成连续 layer sequence，就可以使用相同的 partition、micro-batch 和 rematerialization 机制。

预训练模型迁移到 CIFAR-10、CIFAR-100、Stanford Cars、Oxford Pets、Food-101、FGVC Aircraft 和 Birdsnap 等数据集时，也取得了有竞争力的结果。论文把这些实验作为模型规模提升和 ImageNet pretraining 迁移价值的补充证据。

### 多语言 Transformer

论文在一个覆盖 102 种语言到 English 的 massively multilingual NMT 语料上训练 Transformer。训练数据包含约 25B 个 training examples，覆盖从低资源到高资源的语言。

模型从约 400M 参数的标准 Transformer Big 开始，逐步扩展到：

| 模型 | Encoder/Decoder layers | FFN hidden dimension | Attention heads | 参数量 |
|---|---:|---:|---:|---:|
| Standard | 6 / 6 | 8192 | 16 | 400M |
| Deep | 24 / 24 | 8192 | 16 | 1.3B |
| Wide | 12 / 12 | 16384 | 32 | 1.3B |
| Larger | 32 / 32 | 16384 | 32 | 3B |
| Largest | 64 / 64 | 16384 | 32 | 6B |

6B 模型使用 16 个 accelerator partitions。随着模型从 400M 扩大到 6B，所有语言的翻译质量整体提升，低资源语言也出现明显收益，说明大模型的跨语言 transfer 不只服务于高资源语言。

### Depth-Width Trade-off

论文比较了两个参数量都约为 1.3B 的模型：

- deep model：24 层，FFN hidden dimension 8192，16 heads；
- wide model：12 层，FFN hidden dimension 16384，32 heads。

在高资源语言上，二者质量接近；在低资源语言上，deep model 明显更好。这组实验说明，在固定参数量下增加 depth 可能比单纯增加 width 更有利于跨语言泛化。

不过这不是一个普遍的“depth 一定优于 width”的结论。结果与 multilingual data distribution、模型结构、优化设置和语言资源差异绑定，论文只在自身 NMT 设置中观察到这一趋势。

### 深层模型的训练稳定性

扩大 Transformer depth 后，论文观察到 sharp activations、正 kurtosis 和 noisy dataset 共同导致训练不稳定。训练几千步后，模型预测可能变得非常 peaky，随后产生 non-finite 或很大的 gradient，破坏训练过程。

作者使用两种方法缓解：

1. 根据层数缩小 Transformer FFN layers 的 initialization；
2. 当 softmax 前的 logits magnitude 超过阈值时进行 clipping。

这说明 pipeline parallel 解决的是模型如何放到多设备上，但模型能否稳定收敛还需要单独的 initialization 和 activation control。系统扩展与优化稳定性并不是同一个问题。

### Large-Batch Training

论文将 German-English 的 Transformer Big batch 从 260K tokens 增加到 1M 和 4M tokens，观察 validation loss 和 BLEU：

| Tokens per batch | BLEU | NLL loss |
|---:|---:|---:|
| 260K | 30.92 | 2.58 |
| 1M | 31.86 | 2.51 |
| 4M | 32.71 | 2.46 |

在这个实验设置中，增大 batch size 同时改善了 BLEU 和 NLL。论文认为 4M tokens 是当时 NMT 文献中非常大的 batch size。

这个结果需要与具体 learning rate、数据采样和训练步数一起理解，不能简单推出任意任务中 batch 越大越好。它主要说明当 model parallel 和 data parallel 共同扩展时，global batch size 仍然是训练效率和收敛质量的重要变量。

## 设计取舍与对比

### GPipe 与异步 Pipeline

论文重点对比了 GPipe 与 PipeDream 一类异步 pipeline。异步更新可以减少等待，但会产生 weight staleness：不同 stage 可能使用不同版本的参数。为了得到正确梯度，异步方案需要维护多份 versioned model parameters，增加显存开销。

GPipe 选择同步 mini-batch update：

| 维度 | GPipe | 异步 pipeline |
|---|---|---|
| 参数版本 | 一个 mini-batch 内保持一致 | 可能存在 stale weights |
| 梯度更新 | 所有 micro-batch 完成后统一更新 | forward/backward 交错更新 |
| 训练一致性 | 与 partition 数量解耦得更好 | 需要处理版本和时序 |
| 显存 | 需要 activation buffering 与重计算 | 可能需要多份参数版本 |
| 调度 | bubble 和同步等待 | 调度更激进但更复杂 |

GPipe 并不是没有代价，而是把复杂度集中到 micro-batch 调度、activation rematerialization 和同步等待上，以换取训练语义稳定和架构通用性。

### GPipe 与 SPMD / Tensor Parallel

Mesh-TensorFlow 等 SPMD 方法可以沿更多 tensor dimensions 切分矩阵，每次局部矩阵运算后用 collective 合并结果。这对高带宽互联很依赖，也需要框架知道更多算子的分布式布局。

GPipe 只把连续 layer 分配给不同 accelerator，边界上进行 activation transfer。它的通信次数更少，但单个 layer 必须能够放入一个 accelerator；它也不能解决“单层矩阵本身就放不下”的问题。

因此二者的关系是：

```text
GPipe / Pipeline Parallel：切模型层序列
Megatron / Tensor Parallel：切单层内部矩阵和 attention heads
```

后续的大模型系统可以把二者组合：同一 pipeline stage 内使用 Tensor Parallel，不同 stage 之间使用 Pipeline Parallel，再通过 Data Parallel 扩展 batch，形成 3D parallelism。

## 关键结论

### 1. Micro-batch 是 Pipeline Parallel 的利用率机制

Pipeline Parallel 的核心不只是把 layer 分到不同 GPU，而是用 micro-batch 把原本串行的 layer dependency 转化为设备之间的重叠工作。$M$ 越大，越能摊薄 pipeline warmup 和 drain 的固定开销，但也会增加 buffering 与调度成本。

### 2. 同步更新避免了参数时序污染

GPipe 用同步 mini-batch gradient descent 维持一个 batch 内的参数一致性。这个选择牺牲了一部分调度自由度，却让 partition 数量变化不必引入额外的 stale-weight 语义，是论文可靠性的核心来源。

### 3. Rematerialization 用计算换显存

只保存 cell 边界 activation、backward 时重算 cell 内部 forward，是 GPipe 支撑巨型模型的另一个关键机制。它不能减少参数 memory，但能显著降低训练 activation memory，代价是额外 forward FLOPs。

### 4. Pipeline 的通信边界比通用 tensor sharding 更简单

GPipe 只在相邻 partition 间传递 activation 和 gradient；Tensor Parallel 则需要在层内频繁做 AllReduce 等 collective。前者更依赖 stage balance 和 micro-batch 调度，后者更依赖高带宽互联和 tensor layout 设计。

### 5. 模型扩展需要同时考虑容量、吞吐和可训练性

GPipe 的实验同时显示了三件事：partition 可以提升可容纳的模型规模，micro-batch 可以提升吞吐，深层模型又可能需要初始化和 logit clipping 才能稳定训练。模型并行的成功不是单一通信优化的结果。

## 我的理解

我把 GPipe 看作 Pipeline Parallel 的基础范式：它把“模型太深、单卡放不下”转化为“把 layer sequence 切成 stage，再让多个 micro-batch 在 stage 间流水流动”。其中最关键的并不是 pipeline 图看起来有多复杂，而是同步梯度更新和 rematerialization 两个约束共同确定了训练语义。

读 GPipe 时，应该把三个数量区分清楚：$K$ 是 partition/stage 数，决定模型被切成多少段；$M$ 是 micro-batch 数，决定 pipeline bubble 能被摊薄多少；$N$ 是一个 mini-batch 的总大小，决定梯度更新和优化统计的粒度。很多 Pipeline Parallel 配置问题，本质上都是这三个量没有和显存、吞吐、global batch size 一起考虑。

它和上一篇 Megatron-LM 的关系也很清楚。GPipe 主要回答“不同层如何分到不同设备”，Megatron 主要回答“同一层内部的矩阵和 attention heads 如何分片”。实际大模型训练通常不会二选一，而是根据模型规模和硬件拓扑组合 TP、PP、DP，以及后续的 sequence/context parallel。

## 局限与疑问

### 单个 layer 仍需放入单个 accelerator

GPipe 只能把连续 layer 分到不同设备，论文明确假设单层可以放在一张 accelerator 上。如果单个 attention、MLP 或 convolution layer 本身就超过显存，需要 Tensor Parallel 或更细粒度的 distributed tensor computation。

### Stage balance 很重要

如果不同 layer 的计算量、参数量或 activation shape 差异很大，简单平均层数并不能得到均衡 partition。AmoebaNet 的最大模型规模和 throughput 就受到 layer imbalance 影响。

### Micro-batch 不是完全透明的重排

BatchNorm 等跨 batch 统计算子会因为 micro-batch splitting 改变行为，需要专门处理。其他依赖全局 batch、sequence packing 或动态控制流的 layer，也可能需要重新检查训练语义。

### Bubble 与显存存在取舍

增加 micro-batch 数量可以降低 bubble，但会增加同时驻留或需要管理的 activation 数量。Rematerialization 可以缓解 activation memory，却会增加重计算开销。实际最佳点取决于 stage 数、batch size、sequence length 和设备内存。

### 论文硬件与任务范围有限

论文使用 Cloud TPU 和 P100 GPU，实验集中在图像分类和 multilingual NMT。它没有覆盖今天的 decoder-only LLM、长上下文 agent trajectory 或现代 optimizer sharding，因此不能把论文的吞吐数字直接迁移到当前训练系统。

## 相关知识链接

- [[training/distributed-training/pipeline-parallel|Pipeline Parallel]]
- [[training/distributed-training/tensor-parallel|Tensor Parallel]]
- [[training/distributed-training/megatron|Megatron 与 3D 并行]]
- [[training/distributed-training/data-parallel|Data Parallel]]
- [[training/optimization/gradient-checkpointing|Gradient Checkpointing]]
- [[training/optimization/checkpoint-sharding|Checkpoint Sharding]]
- [[sources/papers/2017-attention-is-all-you-need|Attention Is All You Need]]
