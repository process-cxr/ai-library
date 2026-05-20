---
title: 概率链式法则
type: topic
status: growing
area: fundamentals
tags:
  - math
  - probability
  - language-modeling
aliases:
  - Chain Rule of Probability
---

## 概念界定

概率链式法则将多个随机变量的联合概率分解为一系列条件概率的乘积。自回归语言模型正是利用这一法则，把整段文本序列的概率分解为逐 token 预测。

## 背景与问题

直接建模整句话的联合概率非常困难，因为可能的 token 序列数量巨大。链式法则提供了一种通用分解方式：只要逐步建模每个 token 在前文条件下的概率，就可以得到整个序列的概率。

## 定义与记号

对于随机变量序列 `X_1, X_2, ..., X_T`：

```text
p(x_1, x_2, ..., x_T)
= Π_{t=1}^{T} p(x_t | x_<t)
```

展开写作：

```text
p(x_1, x_2, x_3)
= p(x_1) p(x_2 | x_1) p(x_3 | x_1, x_2)
```

## 直观解释

一段文本的概率可以看成“从左到右每一步都选中正确 token 的概率乘积”。这就是 next-token prediction 可以训练语言模型的概率基础。

## 基本性质

- 链式法则本身总是成立，不依赖模型假设。
- 自回归模型选择从左到右的分解顺序。
- 不同分解顺序会对应不同建模方式。
- 序列越长，联合概率通常越小，因此评估时常使用平均 log probability、交叉熵或困惑度。

## 示例

句子：

```text
I love AI
```

可以分解为：

```text
p(I, love, AI)
= p(I) · p(love | I) · p(AI | I, love)
```

训练时，模型在每个位置预测下一个 token，并最大化这些条件概率。

## 常见误解

- 误解：链式法则是语言模型特有假设。
  - 正确理解：链式法则是概率恒等式；语言模型选择用它进行自回归建模。
- 误解：只要能分解就能精确建模。
  - 正确理解：分解成立不代表模型容量、数据和优化足以学到真实分布。
- 误解：序列概率越大，文本就一定越好。
  - 正确理解：序列长度、常见短语偏置和解码策略都会影响概率解释。

## 相关概念

- [[fundamentals/probability/conditional-probability|条件概率]] — 链式法则的基本构件。
- [[training/pretraining/pretraining|预训练]] — next-token prediction 的训练目标。
- [[fundamentals/information-theory/cross-entropy|交叉熵]] — 最大化条件概率对应的常见损失。
- [[inference/kv-cache-and-memory/kv-cache|KV Cache]] — 高效执行逐 token 条件生成。
