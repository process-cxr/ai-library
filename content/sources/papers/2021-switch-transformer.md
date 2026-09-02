---
title: "Switch Transformers"
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
  - sparse-model
  - expert-parallelism
  - mixed-precision
  - scaling
  - knowledge-distillation
source_url: https://arxiv.org/abs/2101.03961
paper_date: "2021-01"
paper_order: "03961"
---

# Switch Transformers

## 基本信息

- 标题：[Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)
- 作者：William Fedus、Barret Zoph、Noam Shazeer
- 版本：arXiv:2101.03961v3，2022-06-16
- 发表：Journal of Machine Learning Research 23，2022-04
- 机构：Google
- JAX 实现与 checkpoints：[google-research/t5x](https://github.com/google-research/t5x)
- Mesh TensorFlow 实现：[mesh_tensorflow/transformer/moe.py](https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/moe.py)
- 相关主题：[[architecture/sparse-and-efficient/moe|Mixture of Experts]]，[[training/optimization/mixed-precision|Mixed Precision Training]]，[[training/distributed-training/data-parallel|Data Parallel]]，[[training/distributed-training/tensor-parallel|Tensor Parallel]]，[[training/post-training/knowledge-distillation|Knowledge Distillation]]

Switch Transformer 是现代 MoE 路线中最重要的简化之一。它延续 [[sources/papers/2020-gshard|GShard]] 将 Transformer FFN 替换为 sparse expert layer 的基本结构，却把每个 token 的 top-2 routing 缩减为 top-1 routing。这个变化减少了 expert computation、capacity buffer 与跨设备通信，使“增加 total parameters、保持每 token FLOPs 大致不变”成为更容易实现和扩展的模型设计。

论文的贡献不只是一条 router 规则。作者围绕 top-1 routing 补齐了完整训练系统：

```text
token representation
  -> float32 router + top-1 expert
  -> capacity / overflow control
  -> auxiliary load-balancing loss
  -> AllToAll dispatch
  -> local expert FFN
  -> AllToAll combine + residual

training stability
  -> selective precision
  -> 0.1x initialization scale
  -> expert-specific dropout during fine-tuning
```

在 T5 encoder-decoder 与 span-corruption pre-training 设置下，论文报告 FLOP-matched Switch models 相比 dense T5 获得最高约 7 倍 wall-clock pre-training speedup；通过 data、model 与 expert parallelism 的组合，模型规模扩展到 1.571T parameters。与此同时，论文也系统记录了 sparse model 的边界：overflow tokens 可能跳过 expert computation，最大模型仍会不稳定，upstream perplexity 优势不总能转化为 reasoning downstream gains，蒸馏只能保留约 30% 的 sparse teacher 增益。

## 研究问题

### 参数量能否作为独立于 FLOPs 的 scaling axis

Dense Transformer 增大 width 或 depth 时，total parameters 与每 token computation 通常一起增长。Switch Transformer 提出第四条 scaling axis：增加 experts，使不同 token 使用不同参数，但每个 token 只执行一个 expert FFN。

这并不是让更多参数同时参与一次 forward，而是扩大条件参数池：

- total parameters 随 experts 数量增加；
- 每 token active expert 数保持为 1；
- router 计算随 expert 数量增加，但相对 FFN 较小；
- weight memory、device count 与 communication 仍会增长。

论文要验证的是：在固定 FLOPs per token、固定设备预算和相近 wall-clock 条件下，更多 conditional parameters 是否能稳定改善 sample efficiency 与最终质量。

### MoE routing 能否从 top-2 简化为 top-1

早期 sparsely-gated MoE 通常认为，一个 token 至少进入两个 experts 才能为 router 提供有效训练信号。Top-2 也能在首选 expert overflow 时利用第二候选，但代价是每 token 执行两次 FFN、需要更大 expert capacity，并发送更多跨设备数据。

Switch Transformer 反向提出：只选择概率最高的一个 expert，仍用该 expert 的 router probability 缩放输出，并配合 differentiable load-balancing loss。论文需要证明，这种简化不会损害质量，并能把算法计算减少转化为真实吞吐。

### Sparse expert model 如何在低精度下稳定训练

Hard routing 会让不同 tokens 经过不同参数，expert load 和梯度统计也随 batch 波动。Router softmax 还可能在 BF16 中出现数值问题。论文因此研究：哪些局部计算必须保留 FP32，初始化尺度如何影响早期训练，以及大量 expert parameters 在小规模 downstream data 上如何避免过拟合。

### 大规模 sparse model 如何映射到分布式硬件

Experts 分散在设备上后，token 必须根据 router decision 重新分布。Switch Transformer 需要组合：

- data parallelism 对 batch 的切分；
- expert parallelism 对 experts 的切分；
- model parallelism 对单个 FFN 矩阵的切分；
- AllToAll 与 AllReduce 通信；
- static-shape accelerator 所需的 expert capacity buffer。

因此，模型设计不只由 parameter count 决定，还必须联合考虑每卡显存、tokens per core、capacity factor、network cost 与目标 FLOPs per token。

## 核心主张

1. **Top-1 routing 足以训练有效的 sparse expert Transformer。** 选中 expert 的 gate probability 仍参与输出缩放，router 因而保留梯度；top-1 同时减少 expert computation、capacity 和通信。
2. **Expert 数量是独立于 dense width/depth 的有效 scaling axis。** 在 FLOP-matched Switch-Base 实验中，experts 从 2 增至 256 时，C4 test loss 大体持续改善；256-expert model 达到 14.7B parameters。
3. **Sparse scaling 的收益必须同时用 step 与 wall-clock 衡量。** 64-expert Switch-Base 达到 dense T5-Base 同等质量所需 steps 约少 7.5 倍，wall-clock 约缩短到七分之一；这说明额外 AllToAll 没有完全吞噬 sample-efficiency gain。
4. **Selective precision 是稳定性与吞吐之间的有效折中。** 只在 router 内部把输入、logits 与 softmax 提升为 FP32，再把 dispatch/combine tensors 转回 BF16，可以获得接近纯 BF16 的速度和接近 FP32 的稳定性。
5. **Sparse pre-training gain 可以迁移到多数 downstream tasks，但不是无条件的。** Switch-Base/Large 在 SuperGLUE、Winogrande、summarization 和 closed-book QA 上多数更强，ARC 存在例外；最大模型的 upstream gain 对 knowledge tasks 转化更好，对 reasoning tasks 不稳定。
6. **Sparse model 可以蒸馏为 dense model，但容量收益无法完整压缩。** 论文将 sparse teacher 压缩 95%-99% 时，大约只保留 28%-30% 的 quality gain。
7. **Trillion-parameter sparse model 的主要意义是容量与计算解耦，而不是低部署成本。** 1.571T Switch-C 每 sequence FLOPs 远低于同规模 dense model，但仍需保存、分片和通信全部 expert weights。

## Switch Layer

### 从 MoE Top-k 到 Switch Top-1

给定 token representation $x$ 与 $N$ 个 experts，router 首先计算 logits：

$$
h(x)=W_rx
$$

再通过 softmax 得到对每个 expert 的概率：

$$
p_i(x)=\frac{e^{h(x)_i}}{\sum_{j=1}^{N}e^{h(x)_j}}
$$

传统 top-$k$ MoE 的输出是多个 expert outputs 的加权和：

$$
y=\sum_{i\in\mathcal{T}}p_i(x)E_i(x)
$$

Switch 只取：

$$
i^*=\arg\max_i p_i(x)
$$

并计算：

$$
y=p_{i^*}(x)E_{i^*}(x)
$$

虽然 expert index 来自不可微的 `argmax`，选中 expert 的 gate value $p_{i^*}(x)$ 仍乘在输出上，因此 primary task loss 可以更新 router；load-balancing loss 还通过完整 softmax probability 提供额外的 differentiable signal。Top-1 routing 并没有让 router 失去全部梯度，只是舍弃了对未选中 expert 结果的直接比较。

Top-1 带来三项直接收益：

1. 每个 token 只执行一个 expert FFN；
2. 相比 top-2，相同 token batch 所需 expert capacity 理论上至少减半；
3. dispatch / combine 逻辑与跨设备通信更简单。

### Expert Capacity

动态 routing 与 TPU static tensor shapes 之间存在冲突。系统必须预先为每个 expert 声明固定 buffer：

$$
\text{expert capacity}
=
\frac{\text{tokens per batch or routing group}}
{\text{number of experts}}
\times\text{capacity factor}
$$

`capacity factor > 1` 为 routing imbalance 留出额外空间，但会产生 padded empty slots，增加 memory、computation 与 communication。`capacity factor` 太小则会 overflow：当某 expert 接收的 token 超过容量时，溢出 token 不执行该层 expert FFN，而是通过 residual connection 直接进入下一层。

论文在主要实验中将 dropped-token rate 控制在通常低于 1%，且没有观察到 expert 数量与 drop rate 的明显依赖。这个结果依赖足够强的 balance loss、batch size 和具体 routing group，不能解释为 capacity factor 1.0 在任意 workload 下都不会丢 token。

### Load-Balancing Loss

对一个含 $T$ 个 tokens 的 batch 或 local routing group，定义发送到 expert $i$ 的 token 比例：

$$
f_i=\frac{1}{T}\sum_{x\in B}\mathbf{1}
\{\arg\max p(x)=i\}
$$

以及 router 分给 expert $i$ 的平均 probability mass：

$$
P_i=\frac{1}{T}\sum_{x\in B}p_i(x)
$$

辅助损失为：

$$
\mathcal{L}_{balance}
=\alpha N\sum_{i=1}^{N}f_iP_i
$$

$f_i$ 由 hard assignments 统计，不可微；$P_i$ 来自 softmax，可微。乘以 $N$ 是为了让均匀 routing 时的损失尺度不随 expert count 改变。论文对 $\alpha\in\{10^{-1},\ldots,10^{-5}\}$ 做 sweep，最终统一使用 $\alpha=10^{-2}$，认为它能较快均衡负载而不过度干扰主 cross-entropy objective。

这一 loss 同时关注实际 dispatch frequency 与 router probability，但不保证严格 capacity feasibility，也不直接最小化跨设备 tail latency。后续 MoE 系统进一步把 expert、device 与 communication balance 拆成不同目标。

### Router 与 Expert Parallel Execution

论文附录中的执行逻辑可以压缩为：

```text
local tokens
  -> optional input jitter during training
  -> FP32 router logits and softmax
  -> top-1 expert index and gate
  -> load-balancing loss
  -> assign position within each expert buffer
  -> mask overflow assignments
  -> build dispatch / combine tensors in BF16
  -> AllToAll: token layout -> expert layout
  -> local expert FFN
  -> AllToAll: expert layout -> token layout
  -> multiply by gate and restore original shape
  -> residual connection
```

Router 是 token-level decision，而 expert capacity 是 batch/group-level resource constraint。二者共同决定某个 token 是否真正执行首选 expert：top-1 score 决定目的地，capacity mask 决定该 assignment 能否落入静态 buffer。

### Exploration 与 No-Token-Left-Behind

Hard argmax routing 只能观察被选 expert 的结果，存在 exploration-exploitation 问题。论文比较 argmax、softmax sampling、input dropout 和 multiplicative input jitter；差异整体很小，训练配方最终使用 input jitter。

作者还尝试 No-Token-Left-Behind：把首次 overflow 的 token 继续路由到第二候选 expert，并可迭代直到几乎不丢 token。这种方法没有带来实证收益。论文推测，token 与首选 expert 已形成稳定 association，强制发送到次选 expert 可能抵消“完整处理所有 tokens”的收益。

这项负结果很重要：减少 token dropping 本身不必然改善模型，重新路由还会改变 token-expert specialization。Overflow strategy 应同时评估 drop rate、quality、communication 与 assignment consistency。

## Training Stability

### Selective Precision

纯 BF16 的 32-expert Switch-Base 在早期实验中 divergence；纯 FP32 稳定，但速度较慢。作者只把 router 内部计算提升到 FP32：

- router input cast 为 FP32；
- projection、logits 和 softmax 使用 FP32；
- dispatch / combine tensors 在离开 router 前转回 BF16；
- expert computation 与 AllToAll 继续使用 BF16。

| 配置 | Negative log perplexity | Examples/s |
|---|---:|---:|
| Full FP32 | -1.718 | 1,160 |
| Full BF16 | -3.780，diverged | 1,390 |
| Selective precision | -1.716 | 1,390 |

Selective precision 在该实验中获得与 FP32 近似的 learning dynamics，同时保持纯 BF16 的吞吐。它展示的通用原则是：低精度训练不需要全局回退，只需识别数值敏感且通信边界明确的局部子图；但具体保留哪些 operations 必须由失稳定位和对照实验决定。

### Smaller Initialization

论文的 weight matrix initialization 为截断正态分布：

$$
W\sim\operatorname{TruncatedNormal}
\left(0,\sqrt{\frac{s}{n}}\right)
$$

其中 $n$ 是 fan-in，$s$ 是 scale。作者将 Transformer 默认 $s=1.0$ 缩小 10 倍到 $s=0.1$。32-expert Switch-Base 在 3.5K steps、3 个 random seeds 上的结果为：

| Initialization | 平均 negative log perplexity | 标准差 |
|---|---:|---:|
| 0.1x init | -2.72 | 0.01 |
| 1.0x init | -3.60 | 0.68 |

较小 initialization 同时改善早期质量和 run-to-run variance，并被用于从 223M dense baseline 到 trillion-parameter Switch models。这里的 0.1 是当前 T5/Switch parameterization 下的经验 scale，不应脱离 normalization、optimizer、depth 和 residual design 机械复用。

### Expert Dropout

Sparse model 的 total parameters 远大于 FLOP-matched dense baseline，下游数据较小时更容易过拟合。全局提高 dropout 会伤害非 expert layers；论文因而只在 expert FFN 内提高 dropout：

- non-expert layers：dropout 0.1；
- expert FFN：expert dropout 0.4。

该配置在 GLUE、CNN/DailyMail、SQuAD 和 SuperGLUE 的综合结果优于统一使用 0.1、0.2 或 0.3 dropout。它说明 sparse model 的 regularization 也应尊重参数分布：大量 conditional parameters 与共享主干未必需要同等强度的正则化。

### 未解决的稳定性

这些技术稳定了 Switch-Base、Switch-Large 和 1.571T Switch-C，却没有彻底解决 Switch-XXL。Switch-XXL 只有 395B parameters，但 FLOPs per sequence 约为 Switch-C 的 7 倍，并在训练中偶发不稳定；更大的 total parameters 并不必然意味着更难稳定，active compute、width、model parallel communication 与 optimization dynamics 同样重要。

附录还尝试把 Switch experts 放入 Q/K/V projection。FP32 下 quality 有小幅改善，但 BF16 全部 divergence，因此正式架构只替换 FFN。这个负结果表明，能稳定稀疏化 FFN 不代表 attention projections 也能直接采用同一 router recipe。

## Pre-Training and Scaling

### 实验设置

Switch Transformer 以 T5 encoder-decoder 为基础，在 C4 上进行 span-corruption pre-training。训练随机遮盖 15% tokens，并以 sentinel token 替换连续 masked spans，目标是重建缺失内容。它不是 decoder-only next-token prediction；因此论文证明的是 conditional sparsity 在 T5 denoising objective 下的有效性，对现代 decoder-only NTP 模型具有机制参考，但不是完全同分布的直接证据。

主要 Switch layers 每隔一个 FFN layer 放置。Switch-Base 与 T5-Base 的每 sequence FLOPs 均为 124B，Switch-Large 与 T5-Large 均为 425B。Fine-tuning study 中，各模型以每 batch $2^{20}$ tokens 预训练 550K steps，总计约 576B tokens。

### Switch 与 Top-2 MoE

Table 1 在 32 张 TPUv3 cores、相同训练 steps 和硬件下比较 dense、top-2 MoE 与 top-1 Switch。由于 top-2 MoE 每 token 执行两个 experts，它并不与 T5-Base FLOP-matched；Switch-Base 才与 T5-Base 大致 FLOP-matched。

| 模型 | Capacity factor | 100K-step quality | 达到 -1.50 所需时间 | Examples/s |
|---|---:|---:|---:|---:|
| T5-Base | - | -1.731 | 未达到 | 1,600 |
| T5-Large | - | -1.550 | 131.1 h | 470 |
| MoE-Base top-2 | 1.0 | -1.572 | 80.1 h | 860 |
| Switch-Base top-1 | 1.0 | -1.561 | 62.8 h | 1,000 |
| MoE-Base top-2 | 1.25 | -1.559 | 80.7 h | 790 |
| Switch-Base top-1 | 1.25 | -1.553 | 65.0 h | 910 |
| Switch-Base+ | 1.0 | -1.534 | 67.6 h | 780 |

Switch 在 speed-quality trade-off 上优于对应 top-2 MoE。低 capacity factor 1.0/1.25 的结果尤其重要，因为超大模型中 memory buffer 更紧张。不过 examples/s 是算法与 Mesh TensorFlow/TPU implementation 的共同结果，不能直接视为所有硬件上的固有倍率。

### Expert Count Scaling

作者从 T5-Base 223M 开始，把 experts 从 2、4、8 一直增加到 256，在大致固定 FLOPs per token 下将 total parameters 扩大到 14.7B。C4 test loss 随 experts 数量增加呈总体改善，说明 conditional parameter count 在该范围内是有效 scaling axis，但收益逐渐递减。

64-expert Switch-Base 在约 60K steps 达到 T5-Base 约 450K steps 的质量，相当于 7.5 倍 step/sample-efficiency speedup。在 32 TPUv3 cores 的 wall-clock 对照中，它达到相同 perplexity 约只需 T5-Base 七分之一时间。与每 token FLOPs 高 3.5 倍的 T5-Large 相比，Switch-Base 仍有约 2.5 倍 wall-clock speedup。

这里的“speedup”指达到相同 pre-training quality 所需时间，而不是单步吞吐提高 7 倍。Switch 每 step 仍承担 routing 和 AllToAll；它的优势主要来自更少 steps 达到目标质量。

### Trillion-Parameter Models

论文组合 expert、model 与 data parallelism，构建两种不同方向的大模型：

| 模型 | Parameters | FLOPs/sequence | Experts | Expert frequency | 结构定位 |
|---|---:|---:|---:|---:|---|
| T5-XXL | 11B | 6.3T | - | - | Dense baseline |
| Switch-XXL | 395B | 6.3T | 64 | 1/2 | 与 T5-XXL FLOP-matched，结合 model + expert parallelism |
| Switch-C | 1,571B | 890B | 2,048 | 1 | 主要使用 expert parallelism，窄而高度稀疏 |

在 C4 pre-training 上：

| 模型 | 250K-step negative log perplexity | 500K-step negative log perplexity |
|---|---:|---:|
| T5-XXL | -1.147 | -1.095 |
| Switch-XXL | -1.086 | -1.008 |
| Switch-C | -1.096 | -1.043 |

Switch-XXL 在相同 FLOPs/sequence 下拥有更高 sample efficiency；Switch-C 则用更低 active compute 和更多 experts 扩大 total capacity。论文报告 Switch-C 在相同 compute budget 下达到固定 perplexity 的速度约为 T5-XXL 的 4 倍。

Switch-C 的 1.571T 并不表示每 token 执行 trillion parameters，也不表示部署成本接近 890B FLOPs/sequence 的 dense model。全部 expert weights 仍需存储，2048-way routing 仍需硬件与网络支持；低 active compute 与低 total memory 是两件不同的事。

## Downstream Evidence

### Fine-Tuning

FLOP-matched Switch-Base / Switch-Large 在大多数 downstream tasks 上优于对应 T5 baseline：

| Model | GLUE | SQuAD | SuperGLUE | Winogrande | XSum | ANLI R3 |
|---|---:|---:|---:|---:|---:|---:|
| T5-Base | 84.3 | 85.5 | 75.1 | 66.6 | 18.7 | 51.8 |
| Switch-Base | 86.7 | 87.2 | 79.5 | 73.3 | 20.3 | 54.0 |
| T5-Large | 87.8 | 88.1 | 82.7 | 79.1 | 20.9 | 56.6 |
| Switch-Large | 88.5 | 88.6 | 84.7 | 83.0 | 22.3 | 58.6 |

Closed-book TriviaQA 上，Switch-Base / Large 分别为 30.7 / 36.9，对应 T5 为 24.5 / 29.5。例外出现在 ARC：Switch-Base 的 ARC-Challenge 32.8 低于 T5-Base 的 35.5；Switch-Large 的 ARC-Easy 66.0 低于 T5-Large 的 68.8。这些例外说明，更好的 C4 perplexity 和更多 conditional parameters 并不保证每个 reasoning benchmark 都提升。

### Upstream 与 Downstream 的关系

论文附录比较 C4 negative log perplexity 与 SuperGLUE / TriviaQA：

- 对 SuperGLUE，upstream quality 与 score 大体相关，但在最大规模下，同等 perplexity 的 dense model 有时更好；
- 对 TriviaQA，Switch 在同等 upstream perplexity 下可能有更好的 knowledge scaling relationship；
- 最大 sparse models 的 pre-training gain 对 knowledge-heavy tasks 转化得更稳定，对 reasoning tasks 较弱。

Switch-XXL 只预训练约 503B tokens，少于 T5-XXL 的一半。它在 NaturalQuestions、WebQuestions、TriviaQA 上分别达到 34.4、41.0、47.5，高于不使用 Salient Span Masking 的 T5-XXL 32.8、37.2、42.9；但 SuperGLUE 为 87.5，低于 T5-XXL 的 89.3。论文没有把这种差异归结为单一原因，而是指出 fine-tuning quality 可能同时依赖 FLOPs per token、parameter count、regularization、load balancing 与超参数。

### Distillation

作者将 Switch-Base sparse teacher 蒸馏到 223M T5-Base dense student。最有效的组合是：

1. 用 teacher 的 non-expert weights 初始化 student 中形状一致的共享层；
2. 蒸馏 loss 混合 25% teacher probabilities 与 75% hard ground-truth labels。

将 3.8B Switch-Base 蒸馏到 223M 时，student 保留约 29%-30% 的 teacher quality gain。随着 teacher 从 1.1B 增至 14.7B、压缩率从 82% 增至 99%，保留增益从 37% 下降到 28% 左右。Fine-tuned SuperGLUE teacher 从 7.41B 压缩到 223M 后，也保留约 30% gain。

这说明 sparse pre-training 可以作为 dense student 的 teacher，但 conditional capacity 无法通过小 student 完整压缩。所谓“99% compression”描述参数量，不表示 99% 的能力被保留。

### Multilingual Learning

论文在 mC4 的 101 种语言上比较 FLOP-matched mSwitch-Base 与 mT5-Base。训练 1M steps 后：

- 101 种语言的最终 negative log perplexity 全部改善；
- 达到 mT5-Base 最终质量的平均 step speedup 约为 5 倍；
- 91% 的语言达到至少 4 倍 step speedup。

这组结果说明 top-1 sparse experts 不只适用于英语 C4，也能在多语言 multi-task distribution 中扩大容量。但论文没有充分分析 expert specialization、语言间共享与低资源语言 routing，因此不能由“所有语言都改善”推导出 experts 已形成可解释的语言分工。

## Parallelism and System Design

### Data Parallelism

Pure data parallelism 将 batch 切到 $N$ 个 cores，每个 core 保存相同 weights，只在完整 forward/backward 后聚合 gradients。其主要同步位于 step 末尾，但模型必须能在单 core memory 中容纳。

### Model Parallelism

当 FFN intermediate dimension $d_{ff}$ 被切到 $m$ 个 cores 时，每个 core 保存 weight slice，但需要共同处理 batch。在 contraction over sharded $d_{ff}$ 后，forward/backward 中都需要 AllReduce。增大 dense width 会增加 FLOPs per token，也提高层内通信频率。

### Expert and Data Parallelism

Switch 将 experts 分布在不同 cores，同时沿 data dimension 切分 tokens。Router 先在本地形成 assignment，然后通过 AllToAll 把 tensor layout 从 local-token sharding 改成 expert sharding：

```text
[data shard, local tokens, hidden]
  -> dispatch tensor
  -> AllToAll
  -> [expert shard, source cores, capacity, hidden]
  -> local expert FFN
  -> AllToAll combine
  -> original token layout
```

这与 model parallelism 的 AllReduce 语义不同：model parallel 合并同一 dense operation 的 partial results，expert parallel 重新排列 tokens，使每个 token 到达拥有目标 expert weights 的设备。

### 三种并行的组合

总 cores 数可写为 $N=n\times m$，其中 $n$ 是 data/expert-parallel dimension，$m$ 是 model-parallel dimension。扩大 $d_{ff}$ 会提高 active FLOPs 和单 expert 参数，超过单卡显存后必须增加 $m$；在总 cores 固定时，这会压缩 $n$，继而限制 batch size。组合策略同时承担：

- expert routing 的 AllToAll；
- dense shard contraction 的 AllReduce；
- 更小 data-parallel degree 带来的 batch constraint；
- capacity buffer 与 per-core memory。

Switch-XXL 与 Switch-C 展示了两种不同取舍：前者提高 dense width 和 active compute，并结合 model parallelism；后者保持较窄主干，通过 2048 experts 扩大 conditional capacity，主要依赖 expert parallelism。最佳方案取决于硬件 compute、memory 和 interconnect，不存在只按 total parameters 决定的统一配置。

## 论文局限

### 架构与训练目标具有时代和体系边界

实验基于 T5 encoder-decoder、C4/mC4 span corruption、Mesh TensorFlow 与 TPUv3。现代 decoder-only LLM 常使用 NTP、不同 normalization/activation、GPU/NCCL 和更长 sequence length。Switch 的 routing 机制仍有直接参考价值，但论文中的 learning rate、initialization、capacity factor 和速度倍率不能原样迁移。

### FLOP matching 不等于完整成本 matching

FLOPs per token 相同，只控制 active arithmetic 的一个维度。Sparse model 仍有更大的 total weight memory、router overhead、AllToAll、padding、token dropping 和 checkpoint cost。论文补充了 wall-clock 测量，这是重要优点；但测量依赖 TPU topology、Mesh TensorFlow implementation 和特定 batch shape。

### Largest models 的训练稳定性仍未解决

Selective precision 与 smaller initialization 没有稳定所有模型。Switch-XXL 偶发 instability，论文也没有提供完整的 loss-spike mechanism、gradient statistics 或统一稳定性理论。Trillion parameter 可运行不等于整个 sparse scaling surface 已经稳定。

### Upstream gain 不能稳定预测 reasoning gain

Switch 在 C4 perplexity 与 knowledge-heavy QA 上表现突出，但 SuperGLUE、SQuAD 和 ARC 显示不同程度的转化缺口。1.6T Switch-C 的 SQuAD 87.7 还低于较小 Switch-XXL 的 89.6。Total parameters、active FLOPs 和 downstream adaptation 之间存在尚未解释清楚的关系。

### Token dropping 改变了训练语义

Overflow token 跳过 expert FFN，意味着不同 token 可能经历不同 depth。论文报告 drop rate 通常低于 1%，但没有深入分析 dropped tokens 是否集中在特定语言、稀有 token、长尾领域或难样本。平均 drop rate 不能替代语义分布审计。

### Expert specialization 缺少充分解释

论文证明了更多 experts 能提高质量，但没有系统展示 experts 分别学习了什么，也没有证明每个 expert 对应可解释 task/language/domain。Load balance 只确保使用分布，不等于 specialization 一定有意义。

### Deployment 仍然困难

Distillation 可把 sparse model 压缩到 dense student，却只能保留约 30% quality gain。原始 trillion-parameter model 的权重存储、checkpoint、serving expert placement 和 request-level load balance 仍是巨大成本。

## 对现代 MoE 训练的启发

### Top-1 是强基线，不是最终答案

Switch 证明 top-1 routing 可以有效训练，并显著简化系统。但 top-1、top-2 或更大 top-k 的优劣取决于 expert granularity、shared experts、active compute budget 与网络拓扑。现代 MoE 在更多 fine-grained experts 上重新采用 top-k，并不否定 Switch；它们是在不同 expert size 和系统预算下选择不同组合宽度。

### Router 质量与系统负载必须共同评估

训练至少应记录：

- expert selection frequency 与 probability mass；
- per-expert capacity utilization；
- overflow/drop rate 及其 token 分布；
- AllToAll bytes、time 和 tail latency；
- expert gradient norm 与 dead experts；
- primary loss 与 auxiliary loss 比例。

Switch balance loss 适合做最小基线，但只追求 uniform routing 可能限制不同 domain 的自然 specialization。后续方法应通过 ablation 比较“更均衡的系统”是否同时保留“更合理的内容路由”。

### 稀疏 scaling 需要三套坐标

比较 sparse 与 dense models 时，应同时报告：

1. **Capacity**：total parameters、expert count、checkpoint size；
2. **Active computation**：active parameters、FLOPs/token、tokens/s；
3. **System overhead**：device count、AllToAll、capacity padding、drop rate、wall-clock 与 energy。

只报告 1.6T parameters 会夸大每 token computation；只报告 FLOPs per token 又会忽略存储与通信。Switch 论文最值得保留的方法论，是同时给 step-basis 和 time-basis scaling curve。

### Selective precision 应以敏感子图为单位设计

Router 是离散调度之前的概率计算，logit/softmax 数值异常可能放大为完全不同的 expert assignment。Switch 的局部 FP32 策略说明，precision policy 应根据 operation 的决策敏感性与通信边界设计，而不是只按 tensor 大小统一降精度。类似原则也适用于 normalization、loss reduction、quantization scale 与 optimizer updates。

### Fine-Tuning 需要独立于 Pre-Training 的 Sparse Recipe

Pre-training 更好的 sparse checkpoint，不保证使用 dense model 的 fine-tuning hyperparameters 就能释放能力。Expert dropout、router balance、capacity、learning rate 与 task mixture 都可能需要重新调整。面向现代 mid-training 或 long-trajectory agent data，也应分别观察 shared backbone 与 experts 是否过拟合某类 workflow，不能只看总 validation loss。

### Overflow 评估应进入数据质量分析

如果长轨迹、代码、工具调用或低资源语言 tokens 更容易被热门 expert overflow 丢弃，NTP/denoising objective 实际看到的有效训练路径会产生结构性偏差。除总体 drop rate 外，应按 domain、language、token type、sequence position 与 sample quality 分桶统计 overflow，并验证 capacity policy 没有系统性削弱高价值数据。

## 关键结论

1. Switch Transformer 将 MoE 简化为 top-1 routing，使每个 token 只执行一个 expert FFN，同时保留 gate 与 auxiliary loss 对 router 的训练信号。
2. Expert capacity 将动态 token routing 映射到 static buffers；capacity factor 在 dropped tokens 与 padding/communication 之间形成直接权衡。
3. 在 T5/C4 设置下，增加 experts 能在近似固定 FLOPs per token 下持续改善 sample efficiency，64-expert Switch-Base 达到同等质量的 wall-clock 最多缩短约 7 倍。
4. Selective FP32 router、0.1x initialization scale 与 expert dropout 是论文实现稳定 pre-training 和 downstream adaptation 的关键配方。
5. Switch-C 以 2,048 experts 扩展到 1.571T total parameters，但 active compute、weight storage 和 communication 必须分开理解。
6. Sparse pre-training gain 多数能够迁移，但 reasoning tasks 和最大模型存在明显例外；upstream perplexity 不是充分评测。
7. Sparse teacher 可以压缩为 dense student，但在 95%-99% parameter compression 下只保留约 30% quality gain，部署成本问题没有被完全消除。
8. 现代 MoE 系统仍然沿用 Switch 明确化的核心问题：router、capacity、load balance、AllToAll、precision 和 downstream adaptation 必须联合设计。

## 相关知识链接

- [[architecture/sparse-and-efficient/moe|Mixture of Experts]]
- [[sources/papers/2017-outrageously-large-neural-networks|Outrageously Large Neural Networks]]
- [[sources/papers/2020-gshard|GShard]]
- [[sources/papers/2024-deepseekmoe|DeepSeekMoE]]
- [[sources/papers/2024-deepseek-v2|DeepSeek-V2]]
- [[training/optimization/mixed-precision|Mixed Precision Training]]
- [[training/distributed-training/data-parallel|Data Parallel]]
- [[training/distributed-training/tensor-parallel|Tensor Parallel]]
- [[training/distributed-training/torch-distributed|torch.distributed]]
- [[training/post-training/knowledge-distillation|Knowledge Distillation]]
