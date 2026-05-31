---
title: Training Objective
created: 2026-02-22
published: 2026-02-22
modified: 2026-05-31
type: topic
status: mature
area: training
tags:
  - pretraining
  - objective
---

预训练目标定义了模型在大规模数据上“为什么被奖励”。它不仅决定 loss 怎么算，也决定模型会被显式鼓励形成哪些行为、哪些表示和哪些生成习惯。对 LLM 来说，训练目标、数据分布和模型结构共同决定 base model 的能力边界：同样的 Transformer，如果目标函数不同，模型会学习到的条件分布、表示结构和可迁移能力也会不同。

这篇笔记关注 decoder-only LLM 最核心的 next-token prediction，并把它放在更大的 objective family 中理解：maximum likelihood 是主干，masking / denoising / seq2seq objective 是相邻路线，reinforcement-style pretraining 则是近年来为了强化推理和探索而出现的扩展。

## Next-token Prediction

Decoder-only LLM 最常见的预训练目标是 next-token prediction，也叫 causal language modeling。给定 token 序列 $x_1,\dots,x_T$，模型在每个位置根据前缀 $x_{<t}$ 预测真实 token $x_t$：

$$
\mathcal{L}_{NTP}(\theta)=-\sum_{t=1}^{T}\log p_\theta(x_t \mid x_{<t})
$$

从概率角度看，这等价于最大化训练语料的条件似然；从信息论角度看，它是在最小化数据分布 $p^*$ 和模型分布 $p_\theta$ 之间的 [[fundamentals/information-theory/cross-entropy|Cross Entropy]]。

更完整地写，模型把整段文本的概率分解为：

$$
p_\theta(x_1,\dots,x_T)=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t})
$$

训练时通常采用 teacher forcing：第 $t$ 个位置看到的前缀来自真实序列，而不是模型自己上一位置采样出的 token。这使训练稳定、并行且监督信号密集，但也意味着训练分布和自由生成时的分布并不完全一致。

实际训练还会引入 loss masking。例如 instruction 或 chat 数据中，system / user prompt token 可能参与 attention 但不参与 loss，只对 assistant response token 计算 supervised loss。预训练语料中也可能对 padding、document boundary 或特殊控制 token 做 mask。因此“训练目标”不只是公式，还包括哪些 token 被预测、哪些位置被忽略、哪些特殊 token 定义了样本边界。

## 为什么这个目标强大

Next-token prediction 的优势是极强的可扩展性：

- 数据不需要人工标注，只要有文本序列即可。
- 每个 token 都提供一个监督信号，训练信号密集。
- 目标与自回归生成过程一致，适合 decoder-only 模型。
- 同一个目标可以覆盖语法、事实、代码、推理痕迹和对话模式。

这也是现代 base model 能从海量语料中获得通用能力的原因。

NTP 的另一个重要性质是它把许多任务统一成“预测序列延续”。翻译、摘要、代码补全、问答、数学推理、对话风格模仿，都可以作为文本序列中的条件分布被学习。模型并不知道某个片段是“任务标签”还是“答案”，它只是在学习不同上下文下哪些 token 更可能出现。这种统一性降低了任务工程成本，也解释了为什么数据格式和语料分布会强烈影响模型行为。

## 与其他预训练目标的关系

LLM 预训练目标并不只有 NTP。常见相邻目标包括：

| 目标 | 典型结构 | 核心信号 | 适用特点 |
|---|---|---|---|
| Causal LM / NTP | decoder-only | 预测下一个 token | 自回归生成、通用 base model |
| Masked LM | encoder-only | 预测被 mask 的 token | 表示学习、理解任务 |
| Denoising / Span Corruption | encoder-decoder 或 decoder | 从破坏文本恢复原文 | seq2seq、文本到文本任务 |
| Prefix LM | decoder 或 encoder-decoder | 给定前缀预测后缀 | 兼顾条件生成和语言建模 |

这些目标不是简单优劣关系，而是匹配不同模型结构和应用假设。Decoder-only LLM 选择 NTP，是因为它与自回归生成链路完全一致：训练时学 $p(x_t\mid x_{<t})$，推理时也逐 token 采样同一个条件分布。

## 目标的边界

NTP 虽然能隐式学习推理模式，但它并不显式奖励模型“先思考再预测”。如果训练语料中没有清晰推理轨迹，模型只能从最终文本分布中间接学习。即使语料包含 Chain-of-Thought，NTP 也只是模仿这些 token，而不是直接判断某段 thought 是否真的提升预测或解决问题。

因此，很多推理能力通常在后训练阶段通过 [[training/post-training/sft|SFT]]、[[training/post-training/rlhf|RLHF]]、[[training/post-training/grpo|GRPO]] 或 RLVR 被进一步强化。

更具体地说，NTP 的边界主要体现在：

- **目标与任务成功不完全一致**：最大化训练文本 likelihood 不等于最大化真实任务正确率、用户满意度或安全性。
- **可观察 token 与潜在推理不同**：模型可以学会输出推理文本，但这些 token 未必忠实反映内部计算过程。
- **数据分布决定奖励方向**：如果语料包含错误、偏见、过时知识或低质量推理，NTP 会把它们也作为可预测模式学习。
- **长程信用分配弱**：每个 token 都有局部监督，但复杂任务的最终成功往往依赖远距离规划、验证和修正。
- **推理时暴露偏差**：训练时条件是真实前缀，生成时条件可能包含模型自己的错误 token，错误会沿上下文传播。

## Reinforcement-style Pretraining Objective

[[training/pretraining/reinforcement-pretraining|Reinforcement Pretraining]] 是对 NTP 边界的一种探索。[[sources/papers/2025-rlp-reinforcement-as-a-pretraining-objective|RLP]] 把 Chain-of-Thought 看成预测下一 token 前的 action，并用信息增益作为 reward：

$$
r(c_t)=\log p_\theta(x_t \mid x_{<t}, c_t)-\log \bar{p}_\phi(x_t \mid x_{<t})
$$

如果 thought $c_t$ 让模型对真实 next token 的概率高于 no-think EMA baseline，它就获得正奖励。这个目标仍然依赖普通文本和 teacher forcing，但它优化的是“生成有预测价值的 thought”，而不只是直接最大化真实 token likelihood。

这类目标说明：预训练目标不一定只能是 likelihood。更一般地，它可以把普通文本转化为密集 reward，用来激励探索、推理或更好的内部表示。

需要注意的是，这类方法仍处在研究探索阶段。当前大规模 LLM 的主干预训练目标仍是 NTP 或其相近变体；reinforcement-style objective 更适合被理解为对特定能力瓶颈的补充，而不是已经替代 maximum likelihood 的通用范式。

## 评估与诊断

判断预训练目标是否有效，不能只看训练 loss。常见诊断维度包括：

- validation cross entropy / perplexity 是否稳定下降；
- downstream benchmark 是否随 loss 改善而改善；
- 长上下文、代码、数学、知识问答等能力是否出现不均衡退化；
- 生成分布是否出现 repetition、mode collapse 或格式异常；
- 后训练后是否更容易被 SFT / RLHF / RLVR 对齐；
- 对数据混合、tokenizer、sequence length 和 curriculum 是否敏感。

如果训练 loss 很低但下游能力差，常见原因不是 objective 公式本身错误，而是数据质量、数据混合、tokenization、模型容量、训练预算或评估集分布出了问题。

## 相关概念

- [[training/pretraining/pretraining|Pretraining]]
- [[training/pretraining/reinforcement-pretraining|Reinforcement Pretraining]]
- [[fundamentals/information-theory/cross-entropy|Cross Entropy]]
- [[fundamentals/probability/maximum-likelihood|Maximum Likelihood]]
- [[training/post-training/grpo|GRPO]]
