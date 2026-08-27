---
title: "Megatron-LM"
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
  - tensor-parallel
  - megatron
  - model-parallel
  - scaling
  - gpt
  - bert
source_url: https://arxiv.org/abs/1909.08053
paper_date: "2019-09"
paper_order: "08053"
---

# Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism

## 基本信息

| 字段 | 内容 |
|---|---|
| 来源 | arXiv:1909.08053v4，2020-03-13 |
| 标题 | [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) |
| 作者 | Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, Bryan Catanzaro |
| 机构 | NVIDIA |
| 首次公开 | 2019-09，arXiv v1 |
| 代码 | [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) |
| 相关 topic | [[training/distributed-training/megatron|Megatron 与 3D 并行]]，[[training/distributed-training/tensor-parallel|Tensor Parallel]]，[[training/distributed-training/torch-distributed|torch.distributed]]，[[training/optimization/training-memory-estimation|Training Memory Estimation]] |

这篇论文提出了一种面向 Transformer 的 intra-layer model parallelism，并以 PyTorch 中少量通信原语的插入实现大规模训练。论文的核心问题不是如何把完整模型复制到更多 GPU，而是：**当单个 Transformer layer 的矩阵、attention heads 和 optimizer state 已经无法放在一张 GPU 上时，如何在不重写编译器和训练框架的前提下，把单层计算切到多张 GPU，并把通信控制在可接受范围内。**

如果只保留一个核心认识：**Megatron-LM 的 Tensor Parallel 不是把每个中间张量都 AllGather 成完整副本，而是让 Column Parallel 和 Row Parallel 线性层成对出现，使大部分中间结果保持分片，只在确实需要合并 partial result 的位置进行 AllReduce。**

## 研究问题

### 大模型训练的显存限制

语言模型参数增长后，单卡显存不仅需要存放模型权重，还需要存放：

- forward 和 backward 的 activation；
- gradient；
- Adam 的 momentum 和 variance 等 optimizer state；
- embedding 和 output projection 的大词表矩阵。

Activation checkpointing 可以减少 activation memory，但不能解决模型参数和 optimizer state 本身无法放入单卡的问题。Data Parallel 也不能解决这个问题，因为每张 GPU 仍然保存完整模型副本。

因此需要 model parallelism，把模型参数、计算和相关状态分布到多张 GPU 上。

### 现有 model parallel 的工程代价

论文将当时的 model parallel 路线分成两类：

- **Layer-wise pipeline model parallelism**：不同 GPU 负责不同层，激活在 stage 之间传递；
- **Distributed tensor computation**：把一个张量运算本身切到不同设备上。

论文认为，当时的 GPipe、Mesh-TensorFlow 等方案虽然能够表达模型并行，但通常需要新的编译器、框架改写或专门的图调度逻辑。Megatron-LM 选择更窄但更容易落地的路径：利用 Transformer 中 attention heads 和连续 GEMM 的结构，在原生 PyTorch 中插入少量 collective communication。

论文希望同时回答三个问题：

1. Transformer 的单层矩阵和 attention heads 能否被高效地切分到多张 GPU？
2. 切分后的 forward/backward 是否只需要少量通信？
3. 这种方式能否在数百张 GPU 上保持较高的 scaling efficiency，并支持多十亿参数模型训练？

## 核心主张

论文的贡献可以概括为以下几项。

1. **提出简单的 intra-layer model parallelism。** 通过 Column Parallel 和 Row Parallel 的组合切分 Transformer 中的 GEMM。
2. **使用 autograd-aware 的 f/g 通信算子。** forward 和 backward 在不同位置触发 AllReduce，使通信语义可以嵌入普通 PyTorch 计算图。
3. **利用 multi-head attention 的天然分解。** Q、K、V projection 按 attention heads 分片，每张 GPU 负责一部分 head。
4. **减少不必要的通信。** dropout、LayerNorm、residual 等操作在 GPU 上复制计算，而不是为了同步中间结果频繁通信；output embedding 使用 fused cross-entropy 避免收集完整 vocabulary logits。
5. **与 data parallel 和 pipeline parallel 正交。** Tensor/model parallel 可以与 data parallel 组合，理论上也可以与 pipeline model parallel 组合。
6. **大规模实验验证可扩展性。** 论文训练了 8.3B 参数 GPT-2 类模型和 3.9B 参数 BERT 类模型，最大使用 512 张 V100 GPU。

论文还观察到，在 BERT 类模型中，LayerNorm 与 residual 的放置顺序会影响模型规模扩大后的稳定性。这个观察不是 Tensor Parallel 的通信机制，但说明大规模训练不仅需要解决显存和吞吐，也需要同时处理架构和优化稳定性。

## 方法与机制

### Transformer layer 中的计算结构

论文将一个 Transformer layer 简化为 self-attention block 和 MLP block。两类 block 都包含连续的线性层：

```text
MLP:
  X -> GEMM -> GeLU -> GEMM

Self-Attention:
  X -> Q/K/V projection -> attention per head -> output projection
```

Megatron-LM 的关键是选择线性层的切分方向，使非线性函数和 attention head 尽量在本地完成。

### MLP 的 Column Parallel + Row Parallel

先看 MLP 的第一层 GEMM：

$$
Y=\mathrm{GeLU}(XA)
$$

如果按 $A$ 的行切分，同时按输入 $X$ 的列切分，则每张 GPU 只能得到部分矩阵乘结果：

$$
XA=X_1A_1+X_2A_2
$$

但由于 GeLU 是非线性的：

$$
\mathrm{GeLU}(X_1A_1+X_2A_2)
\neq
\mathrm{GeLU}(X_1A_1)+\mathrm{GeLU}(X_2A_2)
$$

因此必须在 GeLU 之前同步 partial result，通信会落在非线性函数之前。

Megatron-LM 选择按列切分第一层权重：

$$
A=[A_1,A_2]
$$

于是每张 GPU 可以独立计算：

$$
[Y_1,Y_2]=[\mathrm{GeLU}(XA_1),\mathrm{GeLU}(XA_2)]
$$

GeLU 可以完全在本地执行。第二个 GEMM 再按行切分，直接消费各 GPU 上的 $Y_i$：

$$
Y_1W_1+Y_2W_2
$$

各 GPU 得到一个 partial output，最后通过 AllReduce 求和。于是一个 MLP block 的主通信模式是：

```text
X replicated
  -> Column Parallel first GEMM
  -> local GeLU
  -> Row Parallel second GEMM
  -> AllReduce partial outputs
```

这里的关键不是“第一层列切、第二层行切”这个口诀本身，而是：**把通信放到线性层组合的边界，并避免在 GeLU 之前收集不必要的完整张量。**

### f/g 通信算子

论文用两个互为共轭的 autograd operator 表达通信：

| Operator | Forward | Backward |
|---|---|---|
| $g$ | AllReduce | Identity |
| $f$ | Identity | AllReduce |

其核心思想是：同一个逻辑通信在 forward 和 backward 中不一定出现在同一个方向。$g$ 在 forward 合并多个 GPU 的 partial output；$f$ 在 forward 保持本地张量，在 backward 合并对应梯度。

论文给出的 $f$ 伪代码非常短：

```python
class f(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, gradient):
        all_reduce(gradient)
        return gradient
```

这段代码体现了 Megatron 的一个重要工程思想：通信不是训练循环外部的特殊步骤，而可以通过 autograd Function 直接嵌入计算图，让 PyTorch 自动沿反向图调用相应的通信逻辑。

### Self-Attention 的 head-wise 切分

Multi-Head Attention 本身就允许不同 heads 并行计算。Megatron-LM 将 Q、K、V projection 的 GEMM 按列切分，使每张 GPU 负责一部分 attention heads：

```text
hidden states
  -> column-parallel Q projection -> local Q heads
  -> column-parallel K projection -> local K heads
  -> column-parallel V projection -> local V heads
  -> local attention computation
  -> row-parallel output projection
  -> AllReduce
```

每张 GPU 可以在本地完成所负责 heads 的 attention，不需要在 Q/K/V projection 后立刻 AllGather。attention 输出 projection 按行切分，产生 partial output 后再 AllReduce。

因此，self-attention 和 MLP 都能采用相同的结构原则：

- 第一段 projection 使用 Column Parallel，保留分片输出；
- 中间的非线性或 head-wise attention 在本地执行；
- 第二段 projection 使用 Row Parallel，在 block 边界合并 partial output。

论文据此指出，一个 model-parallel Transformer layer 的 forward 需要 2 次 AllReduce，backward 需要 2 次 AllReduce：MLP 一次，self-attention 一次。Figure 4 展示的就是这 4 个通信点。

### Embedding 与 vocabulary parallelism

语言模型的 output embedding / LM head 矩阵形状约为：

$$
H\times V
$$

其中 $H$ 是 hidden size，$V$ 是 vocabulary size。$V$ 往往是数万甚至更大，因此 output projection 也可能成为显存和通信瓶颈。

Megatron-LM 沿 vocabulary dimension 切分 embedding matrix：

$$
E=[E_1,E_2]
$$

输入 embedding 只在每张 GPU 上保存部分词表。因为 token id 只命中某一段 vocabulary，每张 GPU 可以本地查表，再通过 $g$ operator 合并完整 hidden representation。

输出 embedding 如果先得到每张 GPU 的局部 logits，再 AllGather 完整 logits，通信量约为：

$$
b\times s\times V
$$

其中 $b$ 是 batch size，$s$ 是 sequence length。这个通信量会随着词表增大而迅速变大。

论文的做法是把 output projection 和 cross-entropy loss 融合：每张 GPU 在自己的 vocabulary shard 上直接参与 loss 计算，只通信计算 loss 所需的低维结果，而不是收集完整 logits。这是一个很典型的通信优化：**不要为了计算最终标量 loss 而物化并传输完整的 vocabulary-sized tensor。**

### Replicated computation

并不是所有操作都需要切分。论文选择在每张 GPU 上复制 LayerNorm 参数，并复制 dropout、LayerNorm 和 residual 的计算。原因是这些操作本身的计算量相对较小，而跨 GPU 同步它们的中间 tensor 会引入额外通信。

因此，论文的策略是：

```text
大矩阵 GEMM / attention heads：shard
LayerNorm / dropout / residual：replicate
分片结果必须合并时：AllReduce
```

这说明 model parallelism 不等于所有算子都平均切分。更合理的目标是让主要参数和 FLOPs 分片，同时让小型、逐位置或容易复制的算子保持本地执行。

### 单层通信总结

| 模块 | 主要切分 | 中间计算 | Forward 通信 | Backward 通信 |
|---|---|---|---|---|
| MLP | 第一层 Column Parallel，第二层 Row Parallel | GeLU 在本地执行 | 第二层 partial output 后 AllReduce | 对应梯度 AllReduce |
| Self-Attention | Q/K/V 按 heads 切分，output projection Row Parallel | 每张 GPU 计算本地 heads | output projection 后 AllReduce | 对应梯度 AllReduce |
| Input embedding | 沿 vocabulary 切分 | 本地查表 | 合并 hidden representation | 由 autograd 处理对应梯度 |
| Output embedding | 沿 vocabulary 切分 | fused cross-entropy | 避免 AllGather 完整 logits | 由分片 loss 反传 |

## Model Parallel 与 Data Parallel 的组合

论文区分 model parallelism 和 data parallelism：

- **Model Parallel**：一份模型被多个 GPU 共同保存和计算；
- **Data Parallel**：不同 GPU 或 GPU group 处理不同数据 batch，并同步梯度。

Megatron-LM 可以先建立一个 model-parallel group，在组内切分同一层；再复制多个 model-parallel group，形成 data-parallel replicas。

如果 model-parallel degree 为 $M$，data-parallel degree 为 $D$，总 GPU 数为：

$$
W=M\times D
$$

例如 8-way model parallel 加 64-way data parallel 使用 512 张 GPU。每个 model-parallel group 内负责同一份 batch 的不同参数分片；不同 group 处理不同 batch，并在对应参数之间做 data-parallel gradient synchronization。

论文实验中，model+data parallel 的 scaling efficiency 略低于只做 model parallel，因为还要承担 data-parallel 梯度通信。但它是训练大规模模型和扩大 global batch size 的必要组合。

### 与 Pipeline Parallel 的关系

论文的方案属于 intra-layer model parallelism，主要切同一 Transformer layer 内的 tensor。它与把不同层放到不同 stage 的 pipeline model parallelism 正交：

```text
Tensor Parallel：同一层内部怎么切
Pipeline Parallel：不同层之间怎么分 stage
Data Parallel：不同 replica 怎么分 batch
```

当模型继续增大时，可以先用 Tensor Parallel 解决单层过大的问题，再用 Pipeline Parallel 分摊层数和总参数。后来的 Megatron 体系将这几种方式组合为 3D parallelism，但“TP 可以与 PP 组合”在这篇论文中主要是设计上的兼容性主张，不应把后续版本的完整 pipeline scheduler 细节倒推为本文实验内容。

## 训练数据与优化配置

### 预训练语料

论文构建了一个约 174GB 的去重语料，主要来源包括：

- Wikipedia；
- CC-Stories；
- RealNews；
- OpenWebText；
- BERT 训练额外使用 BooksCorpus。

为了减少评测污染，论文移除了 WikiText103 测试集中的 Wikipedia articles；对 CC-Stories 去除预处理产生的不必要换行；过滤掉内容长度少于 128 tokens 的文档；再使用 LSH 对 Jaccard similarity 大于 0.7 的近重复内容去重。

这部分数据工程对实验很重要。论文要证明的是模型规模和并行训练策略的效果，如果训练语料与测试集有较大重叠，语言建模和下游结果会失去解释力。

### 混合精度与显存管理

论文使用 mixed precision 和 dynamic loss scaling，以利用 V100 Tensor Cores。其他训练稳定性和显存设置包括：

- 权重初始化为 $\mathcal{N}(0,0.02)$；
- residual layer 前的权重按 $1/\sqrt{2N}$ 缩放，其中 $N$ 是 self-attention 和 MLP block 的层数；
- Adam 使用 weight decay $\lambda=0.01$；
- global gradient norm clipping 设置为 1.0；
- dropout 设置为 0.1；
- 每个 Transformer layer 后使用 activation checkpointing。

这些技术并不属于 Tensor Parallel 本身，但它们决定了模型是否能在目标硬件上真正完成训练。参数分片解决 model state memory，activation checkpointing 解决 activation memory，mixed precision 和 Tensor Core 则改善吞吐。

### GPT-2 训练设置

GPT-2 类模型使用长度为 1024 的 subword sequence，batch size 为 512，训练 300K iterations。Learning rate 初始为 $1.5\times10^{-4}$，前 3K iterations warmup，之后做 single-cycle cosine decay，最低降到 $1\times10^{-5}$。

论文的 scaling 配置如下：

| 参数量 | Layers | Hidden size | Attention heads | Total GPUs |
|---:|---:|---:|---:|---:|
| 355M | 24 | 1024 | 16 | 64 |
| 2.5B | 54 | 1920 | 20 | 128 |
| 8.3B | 72 | 3072 | 24 | 512 |

在单独的 scaling study 中，论文保持每个 attention head 的 hidden size 为 96，用 1.2B、2.5B、4.2B 和 8.3B 四个配置测试 model parallel 规模。8.3B 配置需要 8-way model parallel；与 data parallel 组合时扩展到 512 GPU。

### BERT 训练设置

BERT 类模型使用 vocabulary size 30,522，并将 next sentence prediction 替换为 sentence order prediction，同时采用 whole-word n-gram masking。batch size 为 1024，learning rate 为 $1.0\times10^{-4}$，前 10K iterations warmup，随后在 2M iterations 上线性衰减。

论文训练了 336M、1.3B 和 3.9B 三个 BERT 类模型，并重点研究模型规模扩大后 LayerNorm 与 residual 的位置对稳定性的影响。

## 实验与证据

### Scaling efficiency

论文首先用 1.2B 参数模型在单张 V100 32GB GPU 上建立强基线，整体训练吞吐约为 39 TFLOPs，相当于 DGX-2H 配置下单 GPU 理论峰值的 30%。

在 model parallel scaling 中，8.3B 模型使用 8-way model parallel，达到相对于单 GPU baseline 的约 77% linear scaling efficiency。进一步加入 data parallel 后，8.3B 模型在 512 张 GPU 上仍达到约 74% 的 scaling efficiency。论文摘要将整个应用的表现概括为 15.1 PFLOPs sustained throughput 和 76% scaling efficiency。

这里需要区分几个概念：

- **Model parallel scaling**：随着 model parallel GPU 数增加，模型参数规模同步增加；
- **Data parallel scaling**：模型规模相对固定，通过更多 replica 和更大 batch 提高吞吐；
- **Weak scaling**：GPU 增加时问题规模也增加，不是固定模型上简单增加 GPU。

论文的 model parallel scaling 结果说明，Tensor Parallel 带来的通信没有完全吞噬计算收益；但 scaling efficiency 低于 100% 仍然反映了 collective communication、kernel efficiency 和系统调度的代价。

### GPT-2 语言建模结果

论文使用 GPT-2 类模型进行 WikiText103 和 LAMBADA zero-shot evaluation：

| Model | WikiText103 Perplexity | LAMBADA Accuracy |
|---|---:|---:|
| Megatron-355M | 19.31 | 45.18% |
| Megatron-2.5B | 12.76 | 61.73% |
| Megatron-8.3B | 10.81 | 66.51% |
| Previous SOTA | 15.79 | 63.24% |

8.3B 模型在 WikiText103 上达到 10.81 perplexity，在 LAMBADA 上达到 66.51% accuracy。随着模型规模从 355M 增加到 8.3B，perplexity 降低、cloze accuracy 提升，论文据此展示了规模扩大与语言建模效果之间的关系。

需要注意，论文中的 8.3B 模型结果不能单独证明“模型越大在所有任务上都更好”。它使用了特定的训练数据、sequence length、优化器、训练步数和 zero-shot evaluation；论文展示的是这一训练 recipe 下的 scaling evidence。

### BERT 规模扩展与 LayerNorm

论文观察到，直接扩大原始 BERT architecture 时，模型规模超过 BERT-Large 后可能出现性能退化。作者调整了 LayerNorm 与 residual connection 的顺序，把 normalization 放到 attention 和 FFN 的输入侧，再执行 residual add，得到更稳定的训练曲线。

在 3.9B BERT 模型上，论文报告了：

- RACE 单模型 accuracy：89.5%；
- RACE ensemble accuracy：90.9%。

在 MNLI、QQP、SQuAD 1.1、SQuAD 2.0 和 RACE 上，随着模型从 336M 扩展到 1.3B、3.9B，单模型结果整体提升。这个实验支持的是一个更具体的结论：**大模型 scaling 的收益依赖架构稳定性；如果 LayerNorm/residual 的放置导致训练不稳定，单纯增加参数并不会自动带来收益。**

### 训练数据污染检查

论文检查了训练集中的 test set 8-gram overlap。WikiText103 test set 与训练数据最多约有 10.8% 的 8-gram overlap，LAMBADA 最多约 1.4%；同时 WikiText103 test 本身与其训练集也存在约 9.09% overlap。作者认为这些结果与此前工作一致，没有发现整篇测试文档被意外加入训练集的证据。

这一检查不能证明完全没有数据污染，但至少说明论文没有忽略 benchmark overlap 问题，并且对结果解释保留了必要的谨慎。

## 关键结论

### 1. Tensor Parallel 的本质是布局设计

Megatron-LM 的性能关键不是某个单独的 AllReduce API，而是每个 tensor 在通信前后处于什么布局：replicated、column-sharded 还是 row-sharded。Column Parallel 保持输出分片，Row Parallel 汇总 partial output，二者配对后可以把通信推迟到 block 边界。

### 2. 非线性决定切分方向

线性层可以有多种切分方式，但非线性函数不能随意穿插在 partial sum 之前。MLP 先 Column Parallel 再 GeLU，再 Row Parallel，就是为了让 GeLU 在本地作用于完整的局部输出，避免在非线性之前做额外同步。

### 3. Attention heads 提供了天然并行维度

Multi-Head Attention 的 head 维度本来就是相对独立的，因此 Q/K/V projection 可以按 heads 分片。这个结构让 self-attention 适合做 intra-layer model parallel，也解释了为什么 Tensor Parallel 通常需要和 attention head 数、hidden size、MLP intermediate size 一起设计。

### 4. 通信优化通常优先于算子完全切分

LayerNorm、dropout 和 residual 计算量相对小，复制计算往往比同步中间激活更划算。output embedding 则通过 fused cross-entropy 避免传输完整 vocabulary logits。这些选择共同体现了一个工程原则：**优化分布式训练时，需要同时衡量计算量、通信量和 tensor layout，而不是只追求参数平均分片。**

### 5. Model Parallel、Data Parallel、Pipeline Parallel 解决不同问题

Tensor Parallel 主要解决单层太大的问题，Data Parallel 主要扩展 batch 和吞吐，Pipeline Parallel 主要分摊层数和模型深度。它们可以组合，但每一维的通信模式、拓扑要求和性能瓶颈不同。

### 6. 大规模训练是系统与架构的共同问题

论文同时使用 Tensor Parallel、mixed precision、activation checkpointing、gradient clipping、数据去重和 LayerNorm 调整。大模型能否稳定训练，不能只归因于参数切分；模型结构、优化 recipe、数据工程、硬件拓扑和通信实现必须共同成立。

## 我的理解

我把这篇论文看成 Megatron 训练系统的原点。它真正留下来的不是“用 8 张 GPU 训练 8B 模型”这个历史数字，而是一套非常清晰的分层思路：底层通信库只负责 collective，Megatron 负责定义 process group、tensor layout 和并行层语义，训练循环再把这些组件串成可恢复的 pretraining run。

读懂这篇论文后，看到 `ColumnParallelLinear` 或 `RowParallelLinear` 时，不能只把它们理解成两个带参数的 Linear module。更重要的问题是：输入在当前 rank 上是完整的还是分片的，输出为什么可以继续保持 shard，哪个 block 边界必须 AllReduce，反向传播时梯度又应该在哪个 group 中合并。Megatron 的很多代码细节，实际上都是在维护这个 layout invariant。

论文还改变了我对“通信开销”的理解。高效并行不是让通信消失，而是把通信放在不得不合并的地方，并让大部分计算在通信前后保持局部化。MLP 中 Column Parallel + Row Parallel 的组合、attention 中按 heads 切分、词表 projection 中 fused cross-entropy，都是同一个原则在不同位置的具体实现。

从今天的视角看，Megatron-LM 后来发展出了更完整的 Tensor Parallel、Pipeline Parallel、Sequence Parallel、Context Parallel、Expert Parallel、distributed optimizer 和 distributed checkpoint。但这些后续能力都可以回溯到本文建立的两个基础：**Transformer 内部存在适合切分的结构维度，以及通信语义可以通过少量 autograd-aware primitives 嵌入普通训练代码。**

## 局限与疑问

### 论文主要验证的是 Tensor Parallel

论文强调方案与 pipeline model parallelism 正交，但主要实验集中在 intra-layer model parallelism 和 model+data parallelism。后续 Megatron 的 1F1B pipeline schedule、interleaved pipeline、sequence/context parallel 等内容不属于本文实验结论。

### 通信拓扑依赖较强

实验使用 DGX-2H，GPU 内通过 NVSwitch 提供高带宽连接，节点间使用 InfiniBand。论文中的 scaling efficiency 不能直接迁移到低带宽跨节点环境。TP 通信频繁，实际部署中通常希望 TP group 尽量放在高速互联范围内。

### Tensor Parallel degree 不是越大越好

GPU 数增加后，每张 GPU 的矩阵会变小，kernel efficiency 可能下降；同时 AllReduce 的频率和跨节点通信压力会上升。最终性能取决于矩阵规模、GPU 拓扑、通信实现和计算通信重叠。

### 结果与训练 recipe 强绑定

GPT-2 和 BERT 结果使用了特定数据、tokenization、sequence length、优化器和训练步数。WikiText103、LAMBADA、RACE 的结果是对该 recipe 的实验验证，不能直接当作现代模型在所有任务上的 scaling law。

### 与 ZeRO/FSDP 的边界

本文主要切分模型计算和参数布局，并不是今天所说的完整 optimizer state sharding 方案。ZeRO/FSDP 重点处理 data-parallel replicas 之间参数、梯度和 optimizer state 的冗余；Megatron Tensor Parallel 重点处理单层矩阵和 attention heads 的分布。两者解决的问题不同，可以在后续系统中组合。

## 与源码阅读的连接

如果结合 Megatron 源码继续学习，可以按下面的对应关系追踪：

```text
论文中的 Column/Row Parallel GEMM
  -> megatron/core/tensor_parallel/layers.py

论文中的 f/g 与通信映射
  -> megatron/core/tensor_parallel/mappings.py

论文中的 TP/PP/DP process groups
  -> megatron/core/parallel_state.py

论文中的 Transformer layer 组合
  -> megatron/core/transformer/transformer_layer.py

论文中的训练与 checkpoint 系统
  -> megatron/training/training.py
  -> megatron/training/checkpointing.py
```

实际源码版本会比论文多出很多工程能力，因此阅读时应先用本文建立“为什么需要通信、通信前后 tensor 是什么布局”的模型，再回到源码理解具体封装。

## 相关知识链接

- [[training/distributed-training/megatron|Megatron 与 3D 并行]]
- [[training/distributed-training/tensor-parallel|Tensor Parallel]]
- [[training/distributed-training/torch-distributed|torch.distributed]]
- [[training/distributed-training/pipeline-parallel|Pipeline Parallel]]
- [[training/distributed-training/data-parallel|Data Parallel]]
- [[training/distributed-training/sequence-parallel|Sequence Parallel]]
- [[training/distributed-training/context-parallel|Context Parallel]]
- [[training/distributed-training/zero|ZeRO]]
- [[training/optimization/gradient-checkpointing|Activation Checkpointing]]
- [[training/optimization/mixed-precision|Mixed Precision]]
- [[training/optimization/training-memory-estimation|Training Memory Estimation]]
