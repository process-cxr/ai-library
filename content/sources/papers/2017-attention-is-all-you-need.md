---
title: "Attention Is All You Need"
created: 2026-06-01
published: 2026-08-26
modified: 2026-08-31
type: source
status: processed
source_type: paper
area: sources
tags:
  - source
  - paper
  - transformer
  - attention
  - positional-encoding
source_url: https://arxiv.org/abs/1706.03762
paper_date: "2017-06"
paper_order: "03762"
---

# Attention Is All You Need

## 基本信息

| 字段 | 内容 |
|---|---|
| 来源 | arXiv:1706.03762v7，NIPS 2017 |
| 标题 | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) |
| 作者 | Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin |
| 机构 | Google Brain, Google Research, University of Toronto |
| 首次公开 | 2017-06，arXiv v1 |
| 论文代码 | [tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor) |
| 相关 topic | [[architecture/transformer/transformer|Transformer]]，[[architecture/attention/attention|Attention]]，[[architecture/attention/multi-head-attention|Multi-Head Attention]]，[[architecture/positional-encoding/sinusoidal-position|Sinusoidal Position Encoding]] |

这篇论文提出 Transformer。它没有把 recurrent layer 或 convolutional layer 作为序列建模的主干，而是使用 self-attention 建立序列中不同位置之间的信息交互。论文首先在机器翻译中验证这一架构，之后 Transformer 成为 encoder-only、decoder-only 和 encoder-decoder 大模型的共同基础。

如果只保留一个核心认识：**Transformer 的关键不是单独引入了 attention，而是把 token 间的信息交互、逐位置的非线性变换和自回归约束，组织成了一个适合大规模并行训练的模块化架构。**

## 研究问题

论文针对的是当时 sequence transduction 模型的两个主要限制。

第一，RNN/LSTM 需要按照序列位置递推。即使训练阶段可以做一些优化，当前时间步仍然依赖前一个 hidden state，导致一个样本内部难以充分并行化；序列越长，这个限制越明显。

第二，卷积模型虽然可以并行处理不同位置，但一个位置要和很远的位置建立联系，通常需要堆叠多层卷积或使用较大的 kernel。长距离依赖需要经过较长的路径，增加了学习难度和计算成本。

论文的问题可以概括为：

```text
能否只使用 attention，建立一个不依赖 recurrence 和 convolution 的序列到序列模型？
```

理想的架构需要同时满足三点：

- 训练时能够高度并行化；
- 不同位置之间能够直接建立短路径连接；
- 在机器翻译等 sequence-to-sequence 任务上保持甚至超过当时的效果。

## 核心主张

论文的主张主要有四层。

1. **Self-attention 可以取代 sequence-aligned recurrence。** 每个位置可以直接读取其他位置的信息，不需要通过一个递归 hidden state 逐步传递。
2. **Multi-Head Attention 比单个 attention 更适合表达不同关系。** 不同 head 可以在不同表示子空间中学习不同的依赖模式。
3. **Transformer 在训练效率和翻译质量上同时有优势。** 它能够使用矩阵乘法并行计算，并在 WMT14 English-German 和 English-French 上取得强结果。
4. **attention-only 结构可以迁移到其他 sequence transduction 任务。** 论文在 English constituency parsing 上进行了额外验证。

这里的“attention-only”需要准确理解：论文去掉的是作为主干的 RNN 和 convolution，并不是说模型只包含一个 attention 算子。完整 Transformer 仍然包含 FFN、residual connection、LayerNorm、position encoding、masking 和输出 softmax。

## 方法与机制

### 整体架构

原始 Transformer 是标准的 encoder-decoder 架构：

```text
源序列
  -> token embedding + positional encoding
  -> Encoder stack
  -> memory representations

目标序列的历史 token
  -> token embedding + positional encoding
  -> masked Decoder self-attention
  -> encoder-decoder attention 读取 source memory
  -> FFN
  -> next-token distribution
```

Encoder 将源序列映射为连续表示 $z=(z_1,\ldots,z_n)$。Decoder 在给定 encoder 表示的情况下，自回归生成目标序列 $y=(y_1,\ldots,y_m)$。

论文中的 base model 使用 6 层 encoder 和 6 层 decoder，$d_{model}=512$。每个 encoder layer 包含：

- multi-head self-attention；
- position-wise feed-forward network。

每个 decoder layer 在此基础上增加一个 encoder-decoder attention 子层，因此包含：

- masked multi-head self-attention；
- encoder-decoder attention；
- position-wise feed-forward network。

每个子层外都有 residual connection 和 LayerNorm。论文采用的形式是：

$$
\mathrm{LayerNorm}(x+\mathrm{Sublayer}(x))
$$

这属于后来常称的 Post-Norm 结构。现代深层 decoder-only LLM 经常使用 Pre-Norm，但不能把现代实现的 normalization 顺序直接当成原论文的实现。

### Scaled Dot-Product Attention

Attention 接收 query、key、value，并根据 query 与 key 的兼容性计算 value 的加权和。论文使用的 Scaled Dot-Product Attention 为：

$$
\mathrm{Attention}(Q,K,V)=
\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

计算过程可以拆成三步：

1. $QK^T$ 计算每个 query 与所有 key 的匹配分数；
2. 用 $\sqrt{d_k}$ 缩放分数，再通过 softmax 得到权重；
3. 使用这些权重对 value 做加权求和。

除以 $\sqrt{d_k}$ 是论文中的重要细节。当 query 和 key 的每个维度独立、均值为 0、方差为 1 时，点积的方差会随 $d_k$ 增大。如果不做缩放，分数容易变得很大，softmax 会进入梯度极小的饱和区域。缩放让分数保持在更适合优化的数值范围。

论文将 dot-product attention 与 additive attention 做了比较。二者理论复杂度相近，但 dot-product attention 更适合调用高度优化的矩阵乘法，因此在实际硬件上更快、更节省空间。

### Multi-Head Attention

论文没有只使用一个完整维度的 attention，而是把 query、key、value 分别投影到多个较低维子空间，在这些子空间中并行计算 attention，然后拼接结果：

$$
\mathrm{MultiHead}(Q,K,V)=
\mathrm{Concat}(head_1,\ldots,head_h)W^O
$$

其中：

$$
head_i=\mathrm{Attention}(QW_i^Q,KW_i^K,VW_i^V)
$$

论文的 base model 使用 $h=8$ 个 heads，每个 head 的 $d_k=d_v=64$，而 $d_{model}=512$。因此多个 head 的总计算量与一个完整维度的 single-head attention 大致相近，但表达空间更丰富。

Multi-Head Attention 的作用不是简单地重复同一个 attention，而是允许不同 head 学习不同的关系。例如，它们可能分别关注：

- 局部语法关系；
- 长距离依赖；
- 代词与实体之间的指代关系；
- 句法结构中的特定模式；
- 后续模型中的代码结构、格式 token 或长上下文引用。

这些是对 attention pattern 的合理解释方向，不代表论文已经证明每个 head 都对应一个稳定、可命名的人类语义功能。

### Transformer 中的三种 Attention

论文明确区分了三种使用方式。

**Encoder self-attention** 中，Q、K、V 都来自 encoder 的同一层表示。每个源序列位置可以读取其他所有源位置的信息。

**Decoder self-attention** 中，Q、K、V 都来自 decoder，但必须使用 causal mask。第 $i$ 个位置不能读取 $i$ 之后的目标 token，从而保持自回归生成约束。

**Encoder-decoder attention** 中，query 来自 decoder，key 和 value 来自 encoder 输出。它让 decoder 的每个位置都能读取源序列的编码结果。

因此，self-attention 描述的是 Q/K/V 是否来自同一序列；multi-head 描述的是 attention 是否在多个投影子空间中并行组织。二者不是互斥的概念。

### Causal Mask 与训练并行化

Decoder 的 self-attention 在 softmax 前将非法的未来位置分数设为 $-\infty$：

$$
S_{ij}=-\infty \quad \text{when } j>i
$$

这样 softmax 后未来位置的权重为 0。训练时仍然可以一次性输入完整目标序列，并并行计算所有位置的 hidden states 和 loss；mask 负责保证每个位置只使用合法历史。

这正是自回归 Transformer 的关键工程性质：

```text
训练：完整序列并行计算，使用 causal mask
生成：每次只生成一个新 token，按时间步自回归展开
```

所以“Transformer 可以并行训练”不等于“Transformer 可以并行生成”。论文解决的主要是训练阶段的顺序依赖问题。

### Position-wise Feed-Forward Network

Attention 负责不同位置之间的信息混合，FFN 负责在每个位置上做独立的非线性变换。论文使用的 FFN 是两层线性变换和 ReLU：

$$
\mathrm{FFN}(x)=\max(0,xW_1+b_1)W_2+b_2
$$

它对每个位置使用相同参数，但不同位置之间不直接交互。论文的 base model 使用 $d_{model}=512$、$d_{ff}=2048$。

可以把一个 Transformer layer 理解为两个互补步骤：

```text
Attention：从其他 token 读取上下文
FFN：在当前位置对已融合的上下文表示做非线性变换
```

后来的 SwiGLU、GeGLU 和 MoE 都是在这一 position-wise transformation 位置上发展的变体。它们不是原论文的一部分，但可以从原始 FFN 这个结构位置理解其演化关系。

### Residual Connection 与 LayerNorm

论文在每个子层外使用 residual connection：

$$
x_{out}=\mathrm{LayerNorm}(x+\mathrm{Sublayer}(x))
$$

Residual connection 为信息和梯度提供了更直接的传播通道，使每个子层只需要学习对输入表示的增量修改。LayerNorm 则对单个 token 的 hidden dimensions 做归一化，不依赖 batch 内其他样本的统计量，适合变长序列和 sequence-to-sequence 训练。

原论文使用的是 Post-Norm，即先做子层与 residual 相加，再 LayerNorm。现代大模型常使用 Pre-Norm：

$$
x_{out}=x+\mathrm{Sublayer}(\mathrm{Norm}(x))
$$

这体现的是后续深层训练稳定性方面的工程演化，不应与原始 Transformer 的具体配置混为一谈。

### Positional Encoding

Self-attention 根据内容计算匹配，本身不包含 token 顺序。Transformer 没有 recurrence 和 convolution 后，必须显式加入位置信息。

论文在 embedding 的底部加入固定的 sinusoidal positional encoding：

$$
PE_{(pos,2i)}=\sin\left(pos/10000^{2i/d_{model}}\right)
$$

$$
PE_{(pos,2i+1)}=\cos\left(pos/10000^{2i/d_{model}}\right)
$$

不同维度使用不同频率，波长从 $2\pi$ 到 $10000\cdot2\pi$ 形成几何级数。它的直觉是让模型同时获得短距离和长距离的位置变化信号。

论文选择 sinusoidal encoding 的主要理由是：对于固定偏移量 $k$，$PE_{pos+k}$ 可以表示为 $PE_{pos}$ 的线性函数，因此模型可能比较容易学习相对位置关系；同时固定函数理论上可以生成训练长度之外的位置编码。

论文还用 learned positional embedding 做了对照实验，二者在该实验设置下得到近似结果。需要注意，这只是当时的机器翻译长度和模型规模下的结果，不能据此推出所有长上下文场景中二者完全等价。

## 为什么 Self-Attention 有优势

论文从每层复杂度、可并行的顺序操作数和长距离依赖路径长度三个维度比较 self-attention、recurrent layer 和 convolutional layer。

| Layer type | Per-layer complexity | Sequential operations | Maximum path length |
|---|---:|---:|---:|
| Self-attention | $O(n^2d)$ | $O(1)$ | $O(1)$ |
| Recurrent | $O(nd^2)$ | $O(n)$ | $O(n)$ |
| Convolutional | $O(knd^2)$ | $O(1)$ | $O(\log_k n)$ |
| Restricted self-attention | $O(rnd)$ | $O(1)$ | $O(n/r)$ |

其中 $n$ 是序列长度，$d$ 是表示维度，$k$ 是卷积 kernel size，$r$ 是 restricted attention 的局部窗口大小。

### 计算与路径长度

Self-attention 的最大优势是任意两个位置之间只需要一次信息交互，因此长距离依赖的路径长度是常数。相比之下，RNN 需要经过 $O(n)$ 个顺序步骤，卷积则依赖 kernel size 和层数。

但它也有明确代价：标准 self-attention 需要形成位置之间的两两关系，序列长度方向的计算和中间空间开销是二次的。论文已经指出，如果处理非常长的序列，可以进一步研究 restricted self-attention，以 $O(rnd)$ 的代价换取更长的信息路径。

### 何时比 RNN 更快

论文指出，当序列长度 $n$ 小于表示维度 $d$ 时，self-attention 通常比 recurrent layer 更有计算优势；这在当时机器翻译常用的 word-piece 或 byte-pair 表示下经常成立。

这个判断不能直接外推到任意长文档。对于超长上下文，$O(n^2d)$ 会成为主要瓶颈，后续才出现 sparse attention、linear attention、FlashAttention 和其他长上下文方案。

## 训练设置

### 数据与 batching

论文使用两个 WMT 2014 翻译任务：

- English-German：约 4.5M 个 sentence pairs，共享约 37K 的 BPE vocabulary；
- English-French：约 36M 个 sentence pairs，使用 32K word-piece vocabulary。

样本按照近似序列长度组成 batch。每个 batch 约包含 25K 个 source tokens 和 25K 个 target tokens，而不是只按照样本条数固定 batch size。

### Hardware 与训练时长

实验使用一台配备 8 张 NVIDIA P100 的机器。

- base model：100K steps，约 12 小时，每步约 0.4 秒；
- big model：300K steps，约 3.5 天，每步约 1.0 秒。

这组数据说明论文的优势不只是最终 BLEU，也包括在相同硬件上更高的训练并行度和更低的训练成本。

### Optimizer 与 Learning Rate

论文使用 Adam：

$$
\beta_1=0.9,\quad \beta_2=0.98,\quad \epsilon=10^{-9}
$$

Learning rate 使用带 warmup 的 inverse square root schedule：

$$
\mathrm{lrate}=d_{model}^{-0.5}
\cdot\min(\mathrm{step\_num}^{-0.5},
\mathrm{step\_num}\cdot\mathrm{warmup\_steps}^{-1.5})
$$

其中 `warmup_steps = 4000`。训练初期 learning rate 线性增加，warmup 后按 step 的平方根倒数衰减。这个 schedule 后来常被称为 Noam schedule，是理解早期 Transformer 训练 recipe 的重要参考。

### Regularization 与 decoding

论文使用：

- residual dropout，base model 的 dropout rate 为 0.1；
- embedding 与 positional encoding 相加后的 dropout；
- label smoothing，$\epsilon_{ls}=0.1$。

Label smoothing 会让模型的概率分布更不尖锐，因此 perplexity 可能变差，但翻译 accuracy 和 BLEU 可以提升。

推理时，论文使用 beam search。English-German 的 beam size 为 4，length penalty $\alpha=0.6$；并对最后若干个 checkpoint 做平均，base model 平均最后 5 个，big model 平均最后 20 个。

## 实验与证据

### WMT14 Machine Translation

论文报告的主要翻译结果如下：

| Model | EN-DE BLEU | EN-FR BLEU | Reported training cost |
|---|---:|---:|---:|
| Transformer base | 27.3 | 38.1 | $3.3\times10^{18}$ FLOPs |
| Transformer big | 28.4 | 41.8 | $2.3\times10^{19}$ FLOPs |

Transformer big 在 WMT14 English-German 上达到 28.4 BLEU，超过论文比较的已有方法和 ensemble；在 English-French 上达到 41.8 BLEU，并且训练成本低于当时的强基线。

这组实验支撑的是两个结论：

1. attention-only 的序列模型可以达到很强的 sequence transduction 质量；
2. 训练阶段去除 recurrence 带来的并行化收益，确实能转化为更低的训练成本。

BLEU 数字需要放在 2017 年的 WMT 设置和 tokenization、beam search、checkpoint averaging 共同构成的实验条件下理解，不能直接和今天不同数据、模型和 decoding recipe 的结果横向比较。

### Architecture ablation

论文在 WMT14 English-German development set 上改变 base model 配置，主要得到以下观察。

- 单个 attention head 比最优多头配置低约 0.9 BLEU；
- head 太多也会下降，说明 head 数和每个 head 的维度需要共同选择；
- 减小 attention key 的维度会损害质量，说明 query-key compatibility 对模型表现很重要；
- 增加层数、模型宽度或 FFN 内层维度通常能提升效果，但参数量和计算量也会上升；
- dropout 对防止过拟合很有帮助；
- learned positional embedding 与 sinusoidal encoding 在该设置下几乎持平。

这些 ablation 不意味着某一个配置在所有模型规模上都最优，而是说明 Transformer 的效果来自多个结构选择的组合，不能只归因于“使用了 attention”。

### English Constituency Parsing

为了验证 Transformer 不只适用于翻译，论文还在 Penn Treebank 的 English constituency parsing 上训练了 4 层 Transformer，使用 $d_{model}=1024$。

- WSJ-only 设置约 40K 个训练句子，F1 为 91.3；
- 半监督设置加入约 17M 个高置信度和 BerkeleyParser 语料，F1 为 92.7。

论文认为，Transformer 在这个输出具有较强结构约束、且小数据设置较难的任务上也能取得有竞争力的结果，说明 attention-only encoder-decoder 并不局限于机器翻译。

### Attention visualization

论文展示了 encoder 第 5 层中部分 head 对长距离依赖的 attention visualization。例如在处理 `making` 时，不同 head 关注了与 `making ... more difficult` 结构相关的远处 token。

这个例子说明 attention head 可能学到语法或语义相关的模式，但可视化只能作为行为诊断证据，不能单独证明 attention weight 是模型完整决策过程的解释。

## 与现代 LLM 的关系

### 原始 Transformer 不是 decoder-only LLM

原论文的模型是 encoder-decoder，目标是 sequence-to-sequence translation。GPT、LLaMA、Qwen、DeepSeek 等 decoder-only LLM 继承了 Transformer block 的核心思想，但通常做了以下变化：

- 去掉 encoder；
- 去掉 encoder-decoder cross-attention；
- 只保留 causal self-attention 和 FFN；
- 使用 next-token prediction 进行大规模预训练；
- 使用 RoPE、ALiBi 或其他现代位置机制替代原始 sinusoidal encoding；
- 使用 Pre-Norm、RMSNorm、SwiGLU、GQA 或 MoE 等后续结构变体。

因此，现代 LLM 可以看作 Transformer 家族的一个重要分支，而不是原论文架构的逐项复刻。

### 从论文结构到大模型训练

原始论文已经给出了今天大模型训练仍然成立的抽象骨架：

```text
token embedding
  + position information
  -> attention-based token interaction
  -> position-wise nonlinear transformation
  -> residual / normalization
  -> repeated Transformer blocks
  -> next-token or sequence output objective
```

在 decoder-only LLM 中，训练时用 causal mask 后可以并行计算整段序列的 next-token loss。这解释了为什么几十万甚至更长 token 的训练样本仍能通过矩阵化计算进入 GPU，但推理时仍然需要逐 token 生成。

### 后续工程问题

Transformer 的成功也带来了后续系统问题：

- 标准 attention 的序列长度复杂度是二次的；
- 自回归推理需要维护 KV Cache；
- 多头 K/V 会带来显存和 memory bandwidth 成本；
- 大模型训练需要 Tensor Parallel、Pipeline Parallel、Sequence Parallel 和 distributed optimizer；
- 深层模型需要更稳定的 normalization、activation checkpointing 和 mixed precision。

这些并不是原论文解决的问题，但都可以追溯到它确立的 attention-based block 和大规模矩阵计算范式。

## 分析与判断

这篇论文可以看作大模型架构史上的一个分水岭，但它最重要的地方不只是“用 attention 替代 RNN”。更关键的是，它把一个序列建模问题重新组织成了几类可以被现代硬件高效处理的算子：矩阵乘法完成 token 间信息路由，逐位置 FFN 提供非线性表达，residual 和 normalization 支撑深层堆叠，mask 保留自回归约束。

这也解释了为什么它后来能够扩展到完全不同的模型形态。Encoder-only 模型保留了双向 self-attention，decoder-only 模型保留了 causal self-attention，encoder-decoder 模型则继续使用 cross-attention。外部形式发生了变化，但“每个 token 根据上下文动态读取信息，再进行逐位置变换”的基本计算结构没有消失。

从今天的视角看，论文的限制也同样清楚：标准 attention 的长序列成本是二次的，decoder 的生成仍然是顺序的，原始位置编码和 Post-Norm 也不是现代超深模型的最终答案。Transformer 的意义不是一次性解决了所有问题，而是提供了一个足够简单、可并行、可扩展的架构基座，让后续研究可以围绕计算、显存、位置泛化、稀疏化和并行训练继续演化。

## 局限与疑问

### 标准 attention 的二次复杂度

论文已经在复杂度表中明确了 self-attention 的 $O(n^2d)$ 成本，但实验序列主要是机器翻译句子，无法说明它在今天的超长上下文场景中能否经济运行。长文档、代码仓库和 agent trajectory 会放大这一瓶颈。

### 训练并行不等于生成并行

Transformer 解决了训练时的 recurrence dependency，但 decoder 的推理仍然是 autoregressive。后续的 KV Cache、speculative decoding、continuous batching 和并行解码都属于对这一生成瓶颈的进一步工程处理。

### 架构实验范围有限

论文主要在翻译和 constituency parsing 上验证。它没有覆盖大规模 language modeling、代码生成、工具调用、长程 agent 任务或现代 post-training，因此不能直接从论文结果推断这些场景的最终表现。

### 位置外推仍只是动机

论文认为 sinusoidal encoding 可能有助于长度外推，但实验中 learned embedding 与 sinusoidal encoding 几乎持平，也没有系统验证远超训练长度的长上下文能力。后来的位置编码和 context extension 工作需要单独评估。

### Attention visualization 的解释边界

可视化展示了部分 head 与语法依赖相关的模式，但 attention weight 不是完整的因果解释。FFN、residual、其他 head 和后续层共同决定最终输出，不能把单个 attention map 直接当作模型 reasoning trace。

## 相关知识链接

- [[architecture/transformer/transformer|Transformer]]
- [[architecture/transformer/encoder-decoder-transformer|Encoder-Decoder Transformer]]
- [[architecture/transformer/decoder-only-transformer|Decoder-Only Transformer]]
- [[architecture/attention/attention|Attention]]
- [[architecture/attention/self-attention|Self-Attention]]
- [[architecture/attention/multi-head-attention|Multi-Head Attention]]
- [[architecture/positional-encoding/positional-encoding|Positional Encoding]]
- [[training/pretraining/objective|Training Objective]]
- [[inference/kv-cache-and-memory/kv-cache|KV Cache]]
