---
title: Encoder-Decoder Transformer
created: 2026-01-24
published: 2026-01-24
modified: 2026-08-26
type: topic
status: mature
area: architecture
tags:
  - transformer
  - encoder-decoder
---

Encoder-Decoder Transformer 是一种把输入理解和输出生成分成两个模块的 sequence-to-sequence 架构。原始 Transformer 论文使用它完成机器翻译：Encoder 读取完整的 source sequence，Decoder 在读取 encoder 表示的同时自回归生成 target sequence。

## 整体数据流

```text
source tokens
  -> source embedding + positional encoding
  -> Encoder stack
  -> source representations

target prefix
  -> target embedding + positional encoding
  -> masked Decoder self-attention
  -> encoder-decoder attention
  -> FFN
  -> next-token distribution
```

Encoder 输出一组上下文表示 $z=(z_1,\ldots,z_n)$。Decoder 在第 $t$ 个位置预测目标 token 时，只能读取已经生成的 target prefix，同时通过 cross-attention 读取 encoder 输出。

## Encoder

Encoder 处理 source sequence 时使用 bidirectional self-attention。每个位置可以关注 source 中的其他位置，从而形成包含完整输入上下文的表示。

原始 Transformer 的每一层 Encoder 包含：

1. Multi-Head Self-Attention。
2. Position-wise Feed-Forward Network。
3. Residual Connection 和 LayerNorm。

Encoder 不需要 causal mask，因为 source sequence 在生成 target 时已经整体可见。

## Decoder

Decoder 每一层包含三个子层：

1. Masked Multi-Head Self-Attention。
2. Encoder-Decoder Attention，也称 cross-attention。
3. Position-wise Feed-Forward Network。

Decoder self-attention 使用 causal mask，使目标位置 $i$ 不能读取 $i$ 之后的 target token。Cross-attention 中，query 来自 decoder 当前层，key 和 value 来自 encoder 输出：

$$
Q=H_{decoder}W_Q,\quad K=H_{encoder}W_K,\quad V=H_{encoder}W_V
$$

这样 decoder 可以根据当前生成状态，有选择地读取 source sequence 的不同位置。

## 与原始 Transformer 的关系

[[sources/papers/2017-attention-is-all-you-need|Attention Is All You Need]] 的 base model 使用 6 层 Encoder 和 6 层 Decoder，$d_{model}=512$。每个 attention 使用 8 个 heads，每个 head 的 key/value 维度为 64；FFN 的内层维度为 2048。

原论文的关键变化是：Encoder 和 Decoder 内部不再依赖 RNN 或 convolution，而是用 self-attention、cross-attention 和 FFN 组成。这样训练时可以并行计算完整 source/target 序列，同时保留 decoder 的自回归生成约束。

## 训练与推理

训练时通常使用 teacher forcing：Decoder 输入右移后的 target sequence，并通过 causal mask 并行计算每个位置的 next-token loss。

```text
target:       y_1   y_2   y_3   y_4
decoder in:   BOS   y_1   y_2   y_3
prediction:    y_1   y_2   y_3   y_4
```

推理时没有完整的 target prefix，需要逐 token 生成：

```text
BOS
  -> generate y_1
  -> append y_1, generate y_2
  -> append y_2, generate y_3
  -> ...
```

原始论文使用 beam search 生成翻译结果。现代实现还会使用 KV Cache，避免每一步重新计算 decoder 历史 token 的 key/value。

## 与 Decoder-Only Transformer 的区别

[[architecture/transformer/decoder-only-transformer|Decoder-Only Transformer]] 去掉了 Encoder 和 cross-attention，只保留 causal self-attention 和 FFN。它把输入、输出、对话、代码或工具调用都组织成一条 token sequence，通过 next-token prediction 统一训练。

两者的主要区别如下：

| 维度 | Encoder-Decoder | Decoder-Only |
|---|---|---|
| 输入处理 | Encoder 双向读取 source | 与 target 共用一条自回归序列 |
| 输出生成 | Decoder 通过 cross-attention 读取 encoder | Decoder self-attention 读取历史 token |
| 典型任务 | 翻译、摘要、seq2seq | 语言建模、对话、代码、工具调用 |
| 核心 attention | Encoder self-attention + decoder self-attention + cross-attention | Causal self-attention |
| 训练目标 | 条件序列生成 | Next-token prediction |

Decoder-only 不是 Encoder-Decoder 的简单删减，而是对输入组织方式和训练目标的重新统一。理解原始 Encoder-Decoder 结构，仍然有助于理解 cross-attention、seq2seq 模型和 T5 等架构。

## 复杂度与取舍

Encoder self-attention 对 source length $n$ 的复杂度为 $O(n^2d)$，decoder self-attention 对 target length $m$ 的复杂度为 $O(m^2d)$，cross-attention 还需要处理 source 与 target 之间的交互，复杂度约为 $O(nmd)$。

Encoder-Decoder 的优点是 source 表示和 target 生成职责清晰，输入理解可以是双向的，适合 source 和 target 结构不同的任务。代价是需要维护两套 stack，并在 decoder 中额外执行 cross-attention。

## 相关概念

- [[sources/papers/2017-attention-is-all-you-need|Attention Is All You Need]] - 原始 Transformer 论文。
- [[architecture/transformer/transformer|Transformer]] - Transformer 的整体结构。
- [[architecture/transformer/decoder-only-transformer|Decoder-Only Transformer]] - 当前主流 LLM 的架构分支。
- [[architecture/attention/multi-head-attention|Multi-Head Attention]] - 多头 attention 的计算方式。
- [[architecture/attention/self-attention|Self-Attention]] - 序列内部的信息交互。
- [[architecture/attention/attention|Attention]] - Q/K/V 与 attention score。
