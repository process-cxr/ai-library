---
title: Packing
created: 2026-03-15
published: 2026-03-15
modified: 2026-07-06
type: topic
status: mature
area: training
tags:
  - data-engineering
  - packing
---

Packing 是把多个 tokenized samples 组织成固定长度训练序列的过程。它的目标是减少 padding 浪费、提高 GPU 利用率，并在不破坏样本边界语义的前提下形成高吞吐的 token stream。

Packing 处在 data pipeline 的最后阶段：[[training/data-engineering/data-cleaning|Data Cleaning]]、[[training/data-engineering/deduplication|Deduplication]]、[[training/data-engineering/quality-filtering|Quality Filtering]] 和 [[training/pretraining/tokenizer|Tokenizer]] 完成后，packing 决定这些 token 如何进入 batch。

## 为什么需要 Packing

Transformer 训练通常使用固定 sequence length $S$。如果每个样本单独 padding 到 $S$，大量短样本会浪费计算：

$$
\mathrm{padding\ waste}
= 1 - \frac{\sum_i l_i}{nS}
$$

其中 $l_i$ 是第 $i$ 个样本长度，$n$ 是 batch 内样本数。若大多数样本远短于 $S$，模型会在 padding token 上浪费显存和 FLOPs。

Packing 将多个短样本拼入同一个长度为 $S$ 的序列，使有效 token 比例更高。

## Constant-length Token Stream

预训练中常见做法是把文档 token 拼成一个长 token stream，再切成固定长度 blocks：

1. 每个文档 tokenized；
2. 在文档之间插入 EOS 或 document boundary token；
3. 拼接成连续 token stream；
4. 按 $S$ 切成训练序列；
5. 每个位置做 next-token prediction。

这种方法吞吐高、实现简单，适合大规模 unsupervised pretraining。但它会让一个 block 内包含多个文档，甚至文档边界附近产生跨文档预测。EOS / boundary token 的设计很重要，它告诉模型文档结束，而不是把不相关文档误认为连续上下文。

## Sample Packing

在 SFT、偏好训练、对话数据或结构化样本中，不能总是简单拼接 token stream，因为不同样本之间不应互相 attention 或共享 loss。Sample packing 会把多个样本放进一个 sequence，但用 attention mask 和 loss mask 保持边界。

常见要求：

- 样本之间不能互相注意，使用 block-diagonal attention mask；
- padding 部分不计算 loss；
- prompt 部分可能不计算 loss，只对 response 计算 loss；
- 每个样本保留 BOS/EOS、role token、tool token 等结构；
- pack 后仍能恢复样本级 metadata。

如果 mask 错误，模型可能学习跨样本信息，或在 prompt/padding 上错误计算 loss。

## Padding、Attention Mask 与 Loss Mask

Packing 需要区分三种 mask：

- **attention mask**：控制 token 可以 attend 到哪些位置；
- **loss mask**：控制哪些位置参与 loss；
- **padding mask**：标记无效填充 token。

在 causal LM 预训练中，attention mask 通常是 causal mask 加 padding mask。对 packed SFT，attention mask 可能需要 block-diagonal causal mask，防止样本之间泄漏。

错误示例：

- 样本 A 的 response 可以 attend 到样本 B 的 prompt；
- padding token 被计算 loss；
- document boundary 被删除，模型学习跨文档连续性；
- assistant response 之外的 system/user token 被错误纳入监督。

这些错误不一定导致训练崩溃，但会悄悄污染训练目标。

## Packing 与 Throughput

训练吞吐通常看 tokens/sec，而 packing 会提高有效 tokens/sec：

$$
\mathrm{effective\ tokens/sec}
= \mathrm{raw\ tokens/sec} \times (1-\mathrm{padding\ waste})
$$

Packing 对短样本和长 context 训练尤其重要。没有 packing 时，SFT 数据中的短对话、问答、分类样本会造成大量 padding；有 packing 后，同样显存和计算可以处理更多有效 token。

但 packing 也会增加数据 pipeline 和 attention mask 的复杂度。对某些高性能 kernels，复杂 block-diagonal mask 可能降低 kernel efficiency。因此需要在 padding savings 和 kernel simplicity 之间权衡。

## 长上下文训练中的 Packing

长上下文训练需要特别注意：

- 为了训练 long-range dependency，不能只把无关短文档拼成长序列；
- 需要真实长文档、书籍、论文、代码仓库、多轮对话或合成长上下文任务；
- packing 可能提高长度利用率，但不等于提供长程依赖信号；
- needle-in-a-haystack 类评测只能验证检索行为的一部分；
- 长序列 attention 和 activation 成本更高，padding waste 的代价也更大。

因此，long context training 的数据构造不仅是 packing 问题，还涉及 [[training/mid-training/long-context-training|Long Context Training]] 和位置编码扩展。

对多轮 agent 轨迹和工具调用数据，packing 还会影响格式学习。许多 agent 样本会在开头定义 system instruction、tool schema、tool call format 和 response wrapper，后续 action 则依赖这些开头约束。如果采用 concat-then-split，把一条轨迹从中间切开，模型可能在没有工具定义的片段中学习 tool call，导致 context hallucination、格式错误或参数边界混淆。

因此，agent / tool-use 长上下文训练更适合记录 fragmentation rate，并优先使用能保留样本或轨迹边界的 packing 策略。Best-fit packing 可以在接近零 padding 的情况下减少文档碎片化，适合长文档、仓库级代码和多轮工具轨迹。[[sources/papers/2026-qwen3-coder-next-technical-report|Qwen3-Coder-Next]] 将 best-fit packing 用于 262K context 的 coding agent mid-training，并强调它对工具格式和长程任务的稳定性价值。

## 实践设计

Packing pipeline 通常需要记录：

- tokenizer version；
- sequence length；
- document boundary token；
- packing algorithm；
- 是否允许跨文档 attention；
- attention mask 类型；
- loss mask 规则；
- padding ratio；
- fragmentation rate；
- pack 后平均样本数；
- domain/language distribution 是否变化。

如果按长度排序或 bucketing，需要检查是否改变 batch 分布。极端按长度聚类可能提高吞吐，但会让某些 batch 语言/领域过于单一，影响优化稳定性。

在多卡训练中，packing 还必须和 [[training/data-engineering/distributed-dataloader|Distributed Dataloader]] 配合。不同 data-parallel ranks 的有效 token 数、padding ratio 和 packed sample 数应尽量接近，否则会出现 straggler。若使用 TP/PP/CP，模型并行组内 ranks 必须看到同一个 packed micro-batch 的不同分片，而不是各自读取不同样本。

## 常见失败模式

- **只关注 padding ratio**：忽略跨样本 attention leakage。
- **错误 loss mask**：在 prompt、padding 或 system token 上计算不该计算的 loss。
- **删除文档边界**：模型把不相关文档学成连续上下文。
- **packing 改变数据分布**：长度 bucketing 导致 batch 内 domain/language 偏斜。
- **复杂 mask 降低 kernel 性能**：理论有效 token 增加，但实际吞吐未提升。
- **把拼接短文档当长上下文数据**：模型未真正学习长程依赖。

## 相关概念

- [[training/data-engineering/data-engineering|Data Engineering]]
- [[training/pretraining/tokenizer|Tokenizer]]
- [[training/pretraining/data-mix|Data Mix]]
- [[training/data-engineering/distributed-dataloader|Distributed Dataloader]]
- [[training/mid-training/long-context-training|Long Context Training]]
- [[training/post-training/sft|SFT]]
- [[training/post-training/chat-template|Chat Template]]
