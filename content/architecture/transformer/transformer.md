---
title: Transformer
created: 2026-01-24
published: 2026-01-24
modified: 2026-01-24
type: topic
status: growing
area: architecture
tags:
  - architecture
  - transformer
  - attention
aliases:
  - Transformer Architecture
---

## 问题背景

Transformer 是一种以 [[architecture/attention/attention|Attention]] 为核心的序列建模架构，最初在机器翻译任务中提出，后来成为大语言模型的基础结构。它解决的核心问题是：如何在处理序列时，让每个位置都能高效地读取其他位置的信息，并且避免传统 RNN 那种强顺序依赖。

在 RNN / LSTM 里，序列信息通常按时间步逐个传递：第 $t$ 个 token 的表示依赖前面隐藏状态，长距离依赖需要经过很多步传播。这会带来两个问题：

- **并行性差**：训练时很难同时处理所有位置。
- **长距离依赖难**：信息从远处传来需要经过多次状态更新，容易衰减或被覆盖。

Transformer 的关键变化是：不再把序列理解成必须逐步递推的状态链，而是把序列看成一组 token 表示，让每个 token 通过 attention 直接从其他 token 聚合信息。

## 核心思想

Transformer 的核心思想可以概括为一句话：

> 每个 token 先拥有一个向量表示，然后在每一层中通过 attention 读取上下文，再通过前馈网络进行逐位置变换，最终得到上下文相关表示。

这句话里有三个关键点：

1. **token 表示不是固定不变的**：输入 embedding 只是初始表示，经过多层 Transformer 后会变成 context-dependent hidden states。
2. **attention 负责 token 之间的信息交互**：一个 token 可以根据相关性从其他 token 读取信息。
3. **FFN 负责对每个位置做非线性变换**：它不混合不同位置，而是在每个 token 的表示内部增加表达能力。

在大语言模型中，Transformer 通常不是单层结构，而是堆叠很多个相似的 Transformer block。每一层都对 token 表示做一次上下文整合和特征变换。

## 输入与表示

Transformer 的输入通常是一段 token 序列：

$$
(x_1, x_2, \dots, x_n)
$$

每个 token 会先被映射成向量：

$$
E(x_i) \in \mathbb{R}^{d_{model}}
$$

如果 batch size 为 $B$，序列长度为 $n$，模型宽度为 $d_{model}$，那么输入 hidden states 的形状通常是：

$$
H \in \mathbb{R}^{B \times n \times d_{model}}
$$

其中：

- $B$：batch size，一次处理多少条序列。
- $n$：sequence length，序列中 token 的数量。
- $d_{model}$：每个 token 的表示维度，也叫 hidden size。

仅有 token embedding 还不够，因为 attention 本身不天然知道 token 的顺序。为了让模型区分“第一个 token”和“第二个 token”，Transformer 还需要加入位置信息，例如 [[architecture/positional-encoding/positional-encoding|Positional Encoding]]、[[architecture/positional-encoding/rope|RoPE]] 或其他位置编码方法。

## Transformer Block

一个标准 Transformer block 通常包含两个核心子层：

1. **Self-Attention 子层**：负责让 token 之间交换信息。
2. **Feed-Forward Network 子层**：负责对每个 token 的表示做非线性变换。

现代大模型里常见的 decoder-only block 可以抽象成：

$$
H' = H + \mathrm{Attention}(\mathrm{Norm}(H))
$$

$$
H_{out} = H' + \mathrm{FFN}(\mathrm{Norm}(H'))
$$

这是一种 **Pre-Norm** 写法。它表示：

- 先对输入做 normalization。
- 再进入 attention 或 FFN。
- 子层输出通过 residual connection 加回原表示。

更早的 Transformer 论文中常见的是 **Post-Norm**：

$$
H' = \mathrm{Norm}(H + \mathrm{Attention}(H))
$$

Post-Norm 在浅层模型中可用，但在深层大模型训练中更容易不稳定，因此现代 LLM 更常用 Pre-Norm 或其变体。

## Self-Attention 的作用

Self-Attention 是 Transformer 的信息路由机制。对序列中每个 token，它会计算这个 token 应该关注哪些其他 token，并根据 attention weight 聚合它们的信息。

Scaled Dot-Product Attention 的基本形式是：

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

其中：

- $Q$ 是 query，表示“当前位置想找什么信息”。
- $K$ 是 key，表示“每个位置提供什么索引”。
- $V$ 是 value，表示“每个位置真正被读取的内容”。
- $d_k$ 是 key/query 的维度，用 $\sqrt{d_k}$ 缩放可以避免 dot product 过大导致 softmax 饱和。

在 self-attention 中，$Q$、$K$、$V$ 都来自同一组 hidden states，只是经过不同线性投影：

$$
Q = HW_Q, \quad K = HW_K, \quad V = HW_V
$$

直观地说，一个 token 会拿自己的 query 去和所有 token 的 key 做匹配，然后按匹配程度读取 value。

在 decoder-only language model 中，还会使用 causal mask，确保第 $t$ 个位置只能看见 $1 \dots t$ 的 token，不能偷看未来 token。

## Multi-Head Attention

单个 attention head 只能在一个表示子空间中计算相关性。Multi-Head Attention 会把 hidden states 投影到多个 head，让不同 head 学习不同类型的关系。

例如：

- 有的 head 可能关注局部语法关系。
- 有的 head 可能关注实体指代。
- 有的 head 可能关注格式、分隔符或特殊 token。
- 有的 head 可能在长上下文中学习检索式关联。

Multi-Head Attention 的输出会把多个 head 拼接后再做一次线性投影：

$$
\mathrm{MHA}(H) = \mathrm{Concat}(head_1, \dots, head_m)W_O
$$

其中每个 head 都有自己的 $W_Q, W_K, W_V$ 投影。后来的大模型为了降低推理时 KV Cache 的内存占用，又发展出 [[architecture/attention/multi-query-attention|MQA]] 和 [[architecture/attention/grouped-query-attention|GQA]] 等变体。

## Feed-Forward Network 的作用

Attention 负责不同 token 之间的信息混合，但它本身主要是加权求和。Transformer 还需要一个逐位置的非线性模块来提升表达能力，这就是 [[architecture/transformer/feed-forward|Feed Forward Network]]。

标准 FFN 通常写成：

$$
\mathrm{FFN}(x) = W_2 \sigma(W_1 x + b_1) + b_2
$$

其中：

- $W_1$ 把维度从 $d_{model}$ 升到更高的 $d_{ff}$。
- $\sigma$ 是激活函数，例如 ReLU、GELU、SiLU。
- $W_2$ 再把维度投回 $d_{model}$。

对整个序列来说，FFN 对每个 token 独立应用同一套参数，不直接混合 token 位置。它的作用更像是：attention 已经把上下文信息放进每个 token 表示里，FFN 再对这个“上下文后的表示”做非线性特征变换。

现代 LLM 常使用 SwiGLU / GeGLU 等门控 FFN 变体，这些变体通常比普通两层 MLP 更强，但参数量和计算成本也更高。

## Residual 与 Normalization

深层 Transformer 能训练起来，很大程度依赖 [[architecture/transformer/residual|Residual Connection]] 和 normalization。

Residual connection 的形式是：

$$
H_{out} = H + F(H)
$$

它的作用是保留原始信息通道，让每一层只需要学习对表示的增量修改，而不是完全重写表示。这样可以缓解深层网络中的梯度传播问题。

Normalization 的作用是稳定 hidden states 的尺度。Transformer 中常见的是 LayerNorm 或 RMSNorm。它们可以减少不同层之间激活分布漂移，让训练更稳定。

现代 decoder-only LLM 中，常见组合是：

```text
input
  -> RMSNorm / LayerNorm
  -> Causal Self-Attention
  -> Residual Add
  -> RMSNorm / LayerNorm
  -> FFN / MLP
  -> Residual Add
output
```

这就是很多 LLM block 的核心骨架。

## Encoder-Decoder 与 Decoder-Only

原始 Transformer 是 encoder-decoder 架构，主要用于机器翻译等 sequence-to-sequence 任务。

### Encoder

Encoder 读取完整输入序列，每个位置可以看见其他所有输入位置。它适合理解输入，例如翻译任务中的源语言句子。

Encoder block 通常包含：

- bidirectional self-attention
- FFN
- residual connection
- normalization

### Decoder

Decoder 生成输出序列，每个位置只能看见已经生成的历史 token。它还可以通过 cross-attention 读取 encoder 输出。

Decoder block 通常包含：

- causal self-attention
- cross-attention
- FFN
- residual connection
- normalization

### Decoder-Only

现在主流 LLM，如 GPT、LLaMA、Qwen、DeepSeek 等，多数采用 [[architecture/transformer/decoder-only-transformer|Decoder-only Transformer]]。它去掉 encoder 和 cross-attention，只保留 causal self-attention + FFN 堆叠，用统一的 next-token prediction 目标训练。

Decoder-only 架构的优势是简单、可扩展、适合自回归生成；代价是所有任务都要被组织成 prompt + continuation 的形式。

## 为什么 Transformer 适合大模型

Transformer 能成为大模型主流架构，主要因为它同时满足几个条件。

### 并行训练友好

虽然自回归生成时必须一个 token 一个 token 输出，但训练时可以一次性输入完整序列，通过 causal mask 并行计算所有位置的 next-token loss。这比 RNN 的逐步递推更适合 GPU / TPU。

### 表示能力强

Attention 提供跨 token 信息交互，FFN 提供逐 token 非线性变换，多层堆叠后能形成复杂的上下文相关表示。浅层可能更多处理局部模式，深层可能形成更抽象的语义和任务表示。

### Scaling 性好

Transformer 的参数量、数据量、计算量可以较直接地扩展，并且在大规模预训练中表现出稳定的 scaling behavior。这让它成为研究 [[training/scaling/scaling-law|Scaling Law]] 的主要架构。

### 与任务形式兼容

只要任务能被表示成文本序列，decoder-only Transformer 就可以通过 next-token prediction、instruction tuning 或 prompting 来处理。这使它能统一语言建模、问答、代码生成、工具调用和多轮对话等任务。

## 复杂度与瓶颈

标准 self-attention 对序列长度 $n$ 的时间和显存复杂度通常是：

$$
O(n^2)
$$

原因是每个 token 都要和其他 token 计算 attention score，形成 $n \times n$ 的 attention matrix。

这带来几个直接影响：

- 训练长上下文时，attention 计算和显存压力快速上升。
- 推理时，虽然每步只生成一个新 token，但需要缓存历史 token 的 K/V，这会造成 [[inference/kv-cache-and-memory/kv-cache|KV Cache]] 显存占用。
- Serving 系统需要在 batch、sequence length、KV cache、latency 和 throughput 之间权衡。

因此，围绕 Transformer 的很多后续工作都在处理这些瓶颈，例如：

- [[inference/attention-acceleration/flash-attention|FlashAttention]] 降低 attention 的显存访问开销。
- [[architecture/attention/grouped-query-attention|GQA]] 降低 KV Cache 规模。
- [[architecture/sparse-and-efficient/mamba|Mamba]] 和 [[architecture/sparse-and-efficient/linear-attention|Linear Attention]] 尝试降低长序列复杂度。
- [[inference/kv-cache-and-memory/paged-attention|PagedAttention]] 改善 serving 中 KV cache 的内存管理。

## 设计取舍

Transformer 的优势和代价是绑定在一起的。

| 设计点 | 优势 | 代价 |
|---|---|---|
| Self-Attention | 任意 token 可直接交互，长距离建模能力强 | 标准形式是 $O(n^2)$ |
| Multi-Head | 多个关系子空间并行建模 | 参数和计算增加 |
| FFN 升维 | 增强逐 token 表达能力 | 参数量占比很高 |
| Residual | 保留信息通道，利于深层训练 | 需要配合 normalization 控制尺度 |
| Decoder-only | 训练和生成形式统一，扩展简单 | 输入理解和输出生成都压成同一自回归形式 |
| KV Cache | 避免重复计算历史 K/V | 推理显存随上下文长度增长 |

理解 Transformer 时，不应该只记住“attention 很重要”，还要理解它为什么引入新的系统瓶颈。大模型工程中的很多优化，本质上都是在处理 Transformer 结构带来的计算、显存和并发问题。

## 常见误解

### 误解一：Transformer 等于 Attention

Attention 是 Transformer 的核心机制，但 Transformer 不只是 attention。FFN、residual connection、normalization、position encoding、masking、stacking depth 都是结构的一部分。去掉这些组件后，模型的表达能力和训练稳定性都会明显改变。

### 误解二：Self-Attention 天然知道顺序

Self-Attention 本身只根据 token 表示计算相似度，不知道 token 在第几个位置。位置信息必须通过 positional encoding 或 position-dependent attention 机制注入。

### 误解三：每个 attention head 都一定有可解释语义

有些 head 可能表现出可解释模式，但不能假设每个 head 都对应清晰的人类语义功能。head 的作用可能是分布式、冗余或随层数变化的。

### 误解四：Decoder-only 只能做生成，不能做理解

Decoder-only 的训练目标是 next-token prediction，但通过 prompt 组织，它可以完成分类、问答、抽取、推理和工具调用等任务。它不是不能理解，而是把理解任务转化成生成条件下的预测问题。

### 误解五：Transformer 的瓶颈只在训练

Transformer 在推理和 serving 中同样有瓶颈，尤其是 KV Cache、batch 调度、长上下文、TTFT 和 TPOT。训练优化和推理优化关注点不同，但都受架构约束。

## 最小例子

假设输入句子是：

```text
The animal didn't cross the street because it was too tired.
```

当模型处理 `it` 时，self-attention 可以让 `it` 直接关注 `animal`、`street` 等位置，从上下文判断 `it` 更可能指代 `animal`。如果是 RNN，这种信息需要沿时间步逐步传递；而 Transformer 中，`it` 可以在同一层里直接读取前面所有位置的信息。

但 attention 只完成了“从哪里读取信息”的过程。读取后，FFN 会继续处理这个上下文表示，例如把代词指代、语义角色、句法关系等信息编码进更高层 hidden state。多层堆叠后，模型逐渐形成更复杂的上下文理解。

## 与训练的关系

Transformer 通常通过 next-token prediction 训练：

$$
\max_\theta \sum_t \log p_\theta(x_t \mid x_{<t})
$$

这和 [[fundamentals/probability/maximum-likelihood|Maximum Likelihood Estimation]]、[[fundamentals/information-theory/cross-entropy|Cross Entropy]]、[[fundamentals/information-theory/negative-log-likelihood|Negative Log-Likelihood]] 直接相关。

训练时，模型输入一段 token，通过 causal mask 让每个位置只能看到之前 token，然后预测下一个 token。虽然目标看起来简单，但当数据、参数和计算规模足够大时，Transformer 会在这个目标下学习语言、知识、推理模式、代码结构和任务格式。

后训练阶段，如 [[training/post-training/sft|SFT]]、[[training/post-training/rlhf|RLHF]]、[[training/post-training/dpo|DPO]]，并不改变 Transformer 的基本结构，而是改变数据、目标函数和优化信号，使 base model 更符合人类指令和偏好。

## 与推理的关系

推理时，decoder-only Transformer 自回归生成 token：

```text
prompt -> predict next token -> append token -> predict next token -> ...
```

如果每一步都重新计算完整上下文，会非常浪费。因此推理系统会保存历史 token 的 K/V，形成 [[inference/kv-cache-and-memory/kv-cache|KV Cache]]。新 token 只需要计算自己的 Q，并与缓存中的 K/V 做 attention。

这解释了为什么 Transformer serving 中显存很关键：长上下文和大 batch 会让 KV Cache 快速增大。很多 serving 优化，如 PagedAttention、continuous batching、prefix cache，本质上都围绕 Transformer 的自回归 attention 机制展开。

## 相关概念

- [[architecture/attention/attention|Attention]] — Transformer 中 token 间信息交互的核心机制。
- [[architecture/attention/multi-head-attention|Multi-Head Attention]] — 多头并行建模不同关系子空间。
- [[architecture/positional-encoding/positional-encoding|Positional Encoding]] — 向 attention 注入顺序信息。
- [[architecture/transformer/feed-forward|Feed Forward Network]] — Transformer block 中逐 token 的非线性变换。
- [[architecture/transformer/residual|Residual in Transformer]] — 深层 Transformer 训练稳定性的关键结构。
- [[architecture/transformer/decoder-only-transformer|Decoder-only Transformer]] — 当前主流 LLM 的基本架构形态。
- [[training/pretraining/pretraining|Pretraining]] — Transformer 通过大规模 next-token prediction 形成 base model 能力。
- [[inference/kv-cache-and-memory/kv-cache|KV Cache]] — Transformer 自回归推理的核心缓存机制。
