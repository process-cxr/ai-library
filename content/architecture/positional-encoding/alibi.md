---
title: ALiBi
created: 2026-02-07
published: 2026-02-07
modified: 2026-05-31
type: topic
status: mature
area: architecture
tags:
  - positional-encoding
  - alibi
---

ALiBi，Attention with Linear Biases，是一种直接在 attention score 上加入距离相关线性偏置的位置机制。它不把位置向量加到 token embedding，也不旋转 Q/K，而是在 softmax 前修改 attention logits，使模型天然带有“距离越远，偏置越低”的归纳偏置。

## 基本思想

标准 attention score 为：

$$
S_{ij}=\frac{q_i^T k_j}{\sqrt{d_k}}
$$

ALiBi 加入一个与相对距离有关的 bias：

$$
S'_{ij}=S_{ij}+b_{ij}
$$

在 causal language model 中，位置 $i$ 只能看 $j\le i$ 的历史 token。ALiBi 通常对更远的历史位置施加更大的负偏置：

$$
b_{ij}=-m_h(i-j)
$$

其中：

- $i-j$ 是当前 token 与历史 token 的距离；
- $m_h$ 是第 $h$ 个 attention head 的 slope；
- 不同 head 可以使用不同 slope，从而覆盖不同距离尺度。

## 直觉

ALiBi 的直觉是：近处 token 默认更相关，远处 token 需要更强的内容匹配才能被关注。它不是禁止远距离 attention，而是给远距离位置一个线性惩罚。

如果远处 token 非常重要，内容项 $q_i^Tk_j$ 仍然可以抵消距离惩罚；如果内容相关性一般，模型会更偏向近处上下文。

## 与其他位置机制的区别

| 方法 | 注入位置 | 参数形式 | 主要特点 |
|---|---|---|---|
| [[architecture/positional-encoding/absolute-position|Absolute Position]] | input embedding | learned table | 简单但长度受限 |
| [[architecture/positional-encoding/sinusoidal-position|Sinusoidal Position]] | input embedding | fixed function | 可计算任意位置 |
| [[architecture/positional-encoding/rope|RoPE]] | Q/K rotation | rotation frequency | 相对位置进入 dot product |
| ALiBi | attention score bias | head-specific slope | 距离偏置直接，外推友好 |

ALiBi 更像给 attention 增加结构化先验：距离越远，基础分数越低。

## 长度外推

ALiBi 的一个重要优势是没有固定位置表，也不需要为每个绝对位置学习向量。推理到更长序列时，只要能计算距离 $i-j$，就可以继续生成 bias。

这使它在形式上更适合长度外推。但仍需注意：模型能处理更长位置，不代表一定能在更长上下文中完成复杂检索或推理。长上下文能力还依赖训练长度、数据分布、attention pattern 和推理系统。

## 设计取舍

| 优势 | 代价 |
|---|---|
| 不需要位置 embedding table | 归纳偏置较强，默认惩罚远距离 |
| 直接作用于 attention score | 不像 RoPE 那样改变 Q/K 几何关系 |
| 形式上易于长度外推 | 效果依赖 slope 设计和训练配置 |
| 实现相对简单 | 现代主流 decoder-only LLM 更多采用 RoPE 路线 |

## 常见误解

- **误解：ALiBi 禁止远距离注意力。** 它只是降低远距离 score，重要远程 token 仍可被关注。
- **误解：ALiBi 和 RoPE 做的是同一件事。** 二者都提供位置信息，但 ALiBi 是 score bias，RoPE 是 Q/K 旋转。
- **误解：ALiBi 外推好就等于长上下文理解强。** 位置机制只是必要条件之一。
- **误解：所有 head 使用同样 bias。** ALiBi 通常让不同 head 有不同 slope，以覆盖不同距离偏好。

## 相关概念

- [[architecture/positional-encoding/positional-encoding|Positional Encoding]]
- [[architecture/positional-encoding/rope|RoPE]]
- [[architecture/attention/attention|Attention]]
- [[architecture/attention/self-attention|Self-Attention]]
- [[training/mid-training/long-context-training|Long-context Training]]

## 经典论文与资料

- [[sources/papers/2021-alibi|ALiBi]]
