---
title: Logits 与输出头
type: topic
status: growing
area: fundamentals
tags:
  - neural-network
  - logits
  - language-modeling
aliases:
  - Logits
  - Output Head
  - LM Head
---

## 概念界定

Logits 是模型输出的未归一化分数，输出头是把 hidden state 映射到任务输出空间的线性层。在语言模型中，LM Head 通常把最后一层 hidden state 映射到词表大小的 logits。

## 背景与问题

Transformer 内部 hidden state 的维度通常是 `D`，但语言模型最终需要在 `V` 个词表 token 中预测下一个 token。因此需要一个输出头把 `[D]` 维表示映射到 `[V]` 维 logits。

## 结构与机制

最后一层 hidden state：

```text
h_t: [D]
```

LM Head 权重：

```text
W_vocab: [D, V]
```

输出 logits：

```text
z_t = h_t W_vocab
z_t: [V]
```

再经过 softmax：

```text
p_t = softmax(z_t)
```

得到下一个 token 的概率分布。

## 直观解释

输出头可以理解为对每个候选 token 打分。logit 越高，经过 softmax 后该 token 的概率通常越高。

## 基本性质

- logits 不是概率，可以为负，也不要求总和为 1。
- softmax 把 logits 转换成概率分布。
- LM Head 的参数量通常与词表大小和 hidden size 成正比。
- 有些模型会把 input embedding 和 output head 权重绑定。

## 示例

语言模型预测下一个 token：

```text
final hidden state: [B, T, D]
lm_head:            [D, V]
logits:             [B, T, V]
```

训练时，对每个位置的 logits 和真实下一个 token 计算交叉熵。

## 常见误解

- 误解：logits 就是概率。
  - 正确理解：logits 是未归一化分数，需要 softmax 转成概率。
- 误解：输出头只在分类模型中存在。
  - 正确理解：语言模型也有输出头，只是输出空间是词表。
- 误解：logit 最大的 token 一定总会被输出。
  - 正确理解：贪心解码会选最大 logit，但采样解码可能选择其他 token。

## 相关概念

- [[fundamentals/neural-network-basics/softmax|Softmax]] — logits 到概率分布的转换。
- [[fundamentals/neural-network-basics/parameter-sharing-tying|参数共享与权重绑定]] — input embedding 和 lm head 可能共享权重。
- [[fundamentals/information-theory/cross-entropy|交叉熵]] — logits 参与训练损失计算。
- [[training/pretraining/pretraining|预训练]] — next-token prediction 的输出层。
