---
title: 困惑度
type: topic
status: growing
area: fundamentals
tags:
  - math
  - information-theory
  - evaluation
aliases:
  - Perplexity
  - PPL
---

## 概念界定

困惑度是语言模型评估中常用的指标，可以理解为模型在平均意义上面对多少个“等可能候选”。它通常是平均交叉熵或平均负对数似然的指数形式。

## 背景与问题

交叉熵和 NLL 是 log 空间中的指标，不一定直观。困惑度把平均 log loss 转换回更接近“候选数量”的尺度，用来衡量模型对真实文本的预测有多不困惑。

## 定义与记号

如果平均交叉熵为：

```text
H = - (1/T) Σ_t log p_θ(x_t | x_<t)
```

使用自然对数时，困惑度定义为：

```text
PPL = exp(H)
```

如果使用以 2 为底的对数：

```text
PPL = 2^H
```

## 直观解释

困惑度越低，表示模型越能准确预测真实 token。困惑度为 10 可以粗略理解为：模型平均每一步像是在 10 个等可能候选中选择。

这只是直觉，不代表模型真的只保留了 10 个 token。

## 基本性质

- 困惑度越低，通常表示语言建模能力越好。
- 困惑度依赖数据集、tokenizer、预处理和上下文长度。
- 不同 tokenizer 或不同评测集上的困惑度不能随意横向比较。
- 困惑度主要衡量 next-token prediction，不直接衡量指令跟随、事实性或推理能力。

## Tokenizer 与词表大小的影响

困惑度是按 token 计算的指标，因此 tokenizer 会直接影响 PPL 的数值。不同 tokenizer 会把同一段文本切成不同数量、不同粒度的 token，导致平均 NLL 和 PPL 不能直接比较。

设一段文本被切成 `T` 个 token：

```text
PPL_token = exp(-(1/T) Σ_t log p_θ(x_t | x_<t))
```

如果 tokenizer 更细，把文本切成更多 token，`T` 会变大；如果 tokenizer 更粗，`T` 会变小。因此两个模型即使在同一段原始文本上评估，只要 tokenizer 不同，token-level PPL 就不一定可比。

词表大小 `V` 也会影响困惑度的解释。一个完全均匀随机的模型在每一步给所有 token 相同概率：

```text
p(x_t) = 1 / V
```

此时平均交叉熵为：

```text
H = -log(1 / V) = log V
```

困惑度为：

```text
PPL = exp(log V) = V
```

所以，在极端均匀分布下，PPL 等于词表大小。这解释了“困惑度像平均候选数”的直觉：如果模型每一步真的在 `V` 个 token 中均匀瞎猜，那么它的困惑度就是 `V`。

但真实模型并不是均匀分布，PPL 通常远小于词表大小。并且词表更大不必然导致模型更差，因为更大的词表可能用更少 token 表示同一段文本；词表更小也不必然 PPL 更好，因为它可能把文本切得更碎。

如果要跨 tokenizer 比较语言建模质量，更稳妥的方式是使用按字符或按字节归一化的指标，例如 bits-per-character 或 bits-per-byte：

```text
BPB = total negative log2 likelihood / number of bytes
```

这样可以减弱不同 tokenizer 粒度带来的不可比问题。

## 示例

如果一个模型在测试集上的平均 NLL 是：

```text
H = 2.0
```

则困惑度为：

```text
PPL = exp(2.0) ≈ 7.39
```

如果平均 NLL 下降，困惑度也会下降。

## 常见误解

- 误解：困惑度低就一定是更好的聊天模型。
  - 正确理解：聊天能力还涉及指令对齐、偏好、安全性和工具使用。
- 误解：不同模型的 PPL 可以无条件比较。
  - 正确理解：必须注意 tokenizer、数据集、上下文长度和评测方式是否一致。
- 误解：PPL 表示模型实际候选 token 数量。
  - 正确理解：这是平均意义上的直观解释，不是实际候选集合大小。
- 误解：词表更大的模型 PPL 一定更高，所以一定更差。
  - 正确理解：PPL 是 token-level 指标，词表大小和 tokenizer 粒度会改变 token 数量与预测单位，不能只凭词表大小判断模型好坏。

## 相关概念

- [[fundamentals/information-theory/cross-entropy|交叉熵]] — 困惑度来自平均交叉熵。
- [[fundamentals/information-theory/negative-log-likelihood|负对数似然]] — PPL 是平均 NLL 的指数形式。
- [[application/evaluation/evaluation|评测与 Benchmark]] — 困惑度只是评测指标之一。
- [[training/pretraining/pretraining|预训练]] — 预训练模型常用语言建模指标评估。
