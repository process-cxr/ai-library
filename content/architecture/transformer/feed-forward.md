---
title: Feed Forward Network in Transformer
created: 2026-01-24
published: 2026-01-24
modified: 2026-05-31
type: topic
status: mature
area: architecture
tags:
  - transformer
  - ffn
---

Feed Forward Network，简称 FFN 或 MLP，是 Transformer block 中逐 token 的非线性变换模块。[[architecture/attention/attention|Attention]] 负责让不同 token 交换信息；FFN 则在每个 token 的 hidden state 内部做更强的特征变换。二者共同构成 Transformer 的核心计算：attention 混合上下文，FFN 提升表示能力。

## 基本结构

标准 Transformer FFN 通常是两层线性变换加一个非线性激活：

$$
\mathrm{FFN}(x)=W_2 \sigma(W_1x+b_1)+b_2
$$

其中：

- $x \in \mathbb{R}^{d_{model}}$ 是某个 token 的 hidden state；
- $W_1$ 把维度从 $d_{model}$ 升到 $d_{ff}$；
- $\sigma$ 是激活函数，例如 ReLU、GELU、SiLU；
- $W_2$ 再把维度从 $d_{ff}$ 投回 $d_{model}$。

对整个序列而言，FFN 对每个位置独立应用同一组参数：

$$
H \in \mathbb{R}^{B \times T \times d_{model}}
\rightarrow
\mathrm{FFN}(H) \in \mathbb{R}^{B \times T \times d_{model}}
$$

它不会直接混合不同 token 位置。位置之间的信息交换已经由 attention 完成，FFN 只处理每个位置当前聚合后的表示。

## 升维与降维

FFN 通常先升维再降维。早期 Transformer 常用：

$$
d_{ff}=4d_{model}
$$

升维的直觉是给每个 token 一个更大的中间特征空间，让模型可以表达更复杂的非线性组合；降维则保证 block 输出仍回到 $d_{model}$，方便残差连接：

$$
H_{out}=H+\mathrm{FFN}(\mathrm{Norm}(H))
$$

如果没有这个投回步骤，残差相加的维度就不匹配。

## 参数量与计算量

忽略 bias，普通两层 FFN 的参数量约为：

$$
2d_{model}d_{ff}
$$

如果 $d_{ff}=4d_{model}$，则 FFN 参数量约为：

$$
8d_{model}^2
$$

相比之下，标准 attention 中 Q/K/V/O 四个投影的参数量约为：

$$
4d_{model}^2
$$

因此在许多 dense Transformer 中，FFN/MLP 是参数量和 FLOPs 的重要来源，甚至常常超过 attention 投影部分。理解训练和推理成本时，不能只看 attention。

## 激活函数

激活函数决定 FFN 如何引入非线性。常见选择包括：

| 激活 | 特点 | 常见使用 |
|---|---|---|
| ReLU | 简单、稀疏激活 | 原始 Transformer 等早期模型 |
| GELU | 平滑，常用于 BERT/GPT 系模型 | 许多早期 LLM |
| SiLU / Swish | 平滑门控友好 | SwiGLU 相关结构 |
| GeGLU / SwiGLU | 门控 GLU family，表达能力更强 | 现代 decoder-only LLM 常见 |

现代 LLM 很多使用 gated FFN，而不是简单的 $W_2\sigma(W_1x)$。

## Gated FFN 与 SwiGLU

SwiGLU 类结构引入门控分支。一个常见抽象是：

$$
\mathrm{SwiGLU}(x)
= W_o\left(\mathrm{SiLU}(xW_g)\odot xW_u\right)
$$

其中：

- $W_g$ 产生 gate 分支；
- $W_u$ 产生 up-projection 分支；
- $\odot$ 是逐元素乘法；
- $W_o$ 把中间表示投回 $d_{model}$。

门控结构的直觉是：一条分支决定哪些特征应该通过，另一条分支提供候选内容。在相同中间宽度下，门控 FFN 会比普通两层 FFN 增加参数和计算；实际模型常通过调整 intermediate size 来控制总成本。它通常能提升模型质量，因此在 LLaMA、Qwen、DeepSeek 等现代模型中非常常见。

## 与 MoE 的关系

[[architecture/sparse-and-efficient/moe|Mixture of Experts]] 通常替换或扩展 Transformer 的 FFN 部分。Dense FFN 是每个 token 都经过同一套 MLP；MoE FFN 则把 MLP 拆成多个 experts，由 router 为每个 token 选择少数 experts：

```text
dense FFN: every token -> same FFN
MoE FFN: token -> router -> selected experts
```

因此 MoE 的核心不是替代 attention，而是把 FFN 从 dense computation 改成 sparse expert computation。理解 FFN 的参数量和计算占比，是理解 MoE 为什么能提高 total capacity 的前提。

## 设计取舍

| 设计点 | 优势 | 代价 |
|---|---|---|
| 较大的 $d_{ff}$ | 表达能力更强 | 参数和 FLOPs 增加 |
| Gated FFN | 更强非线性和特征选择 | 参数更多，实现略复杂 |
| MoE FFN | total capacity 大、active compute 较低 | routing、负载均衡和部署复杂 |
| 逐 token 共享参数 | 结构简单、并行友好 | 不直接做 token 间交互 |

## 常见误解

- **误解：Transformer 的能力主要来自 attention，FFN 只是附属。** FFN 是参数量和非线性表达的重要来源，不能忽略。
- **误解：FFN 会混合不同 token。** 标准 FFN 对每个位置独立应用；token 间信息混合来自 attention。
- **误解：MoE 是很多 attention heads。** MoE 通常作用于 FFN/MLP，不是 attention head 的简单增加。
- **误解：把 $d_{ff}$ 做大总是更好。** 更大 FFN 提高容量，也增加训练/推理成本和显存压力。

## 相关概念

- [[architecture/transformer/transformer|Transformer]]
- [[architecture/attention/attention|Attention]]
- [[architecture/sparse-and-efficient/moe|Mixture of Experts]]
- [[fundamentals/neural-network-basics/feed-forward-network|Feed-Forward Network]]
- [[fundamentals/neural-network-basics/activation-functions|Activation Functions]]
- [[fundamentals/neural-network-basics/linear-layer|Linear Layer]]

## 经典论文与资料

- [[sources/papers/2020-glu-variants-improve-transformer|GLU Variants Improve Transformer]]
- [[sources/papers/2017-attention-is-all-you-need|Attention Is All You Need]]
