---
title: Positional Encoding
created: 2026-02-01
published: 2026-02-01
modified: 2026-05-31
type: topic
status: mature
area: architecture
tags:
  - architecture
  - positional-encoding
  - attention
aliases:
  - Position Encoding
  - Position Embedding
---

## 问题背景

Positional Encoding 是 Transformer 中用于表示 token 顺序和相对位置关系的机制。它解决的核心问题是：[[architecture/attention/attention|Attention]] 本身主要根据 token 内容做匹配，并不天然知道 token 在序列中的先后顺序。

如果没有位置信息，模型看到的更像是一组 token 集合，而不是有顺序的序列。比如：

```text
猫追狗
狗追猫
```

两句话包含相同 token，但语义完全不同。模型必须知道“谁在前、谁在后”，也需要知道 token 之间相隔多远。

因此，位置编码不是 Transformer 的装饰组件，而是序列建模的基础条件。没有位置机制，attention 很难区分不同排列，也很难稳定建模语法、代码结构、对话轮次和长文档中的距离关系。

## 核心问题

一个好的位置机制通常需要回答三个问题：

1. **绝对位置**：这个 token 在第几个位置？
2. **相对位置**：两个 token 相隔多远？谁在谁之前？
3. **长度泛化**：模型能否处理训练长度之外的位置？

不同位置编码方法对这三个问题的处理方式不同。

- Learned absolute position embedding 强调绝对位置。
- Sinusoidal position encoding 用固定频率函数表达位置。
- [[architecture/positional-encoding/rope|RoPE]] 让相对位置信息进入 Q/K dot product。
- [[architecture/positional-encoding/alibi|ALiBi]] 直接给 attention score 加距离偏置。
- [[architecture/positional-encoding/yarn|YaRN]] 等方法主要服务于 RoPE 长上下文扩展。

理解位置编码时，不能只问“它怎么把位置加进去”，还要问“它把位置信息加到哪里，以及会如何影响 attention score”。

## 为什么 Attention 不天然知道顺序

Scaled Dot-Product Attention 的形式是：

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

如果 $Q, K, V$ 只来自 token embedding，而 token embedding 本身没有位置信息，那么 attention 只能根据 token 内容相似度做匹配。它无法区分同一个 token 出现在第 3 位还是第 300 位。

更形式化地说，如果对输入 token 同时做同样的排列，attention 的输出也会按同样方式排列。这说明 attention 对序列顺序没有内建偏好。要让 Transformer 成为序列模型，必须注入 position information。

## 方法一：Absolute Position Embedding

[[architecture/positional-encoding/absolute-position|Absolute Position Embedding]] 是最直接的位置方法：为每个位置 $i$ 学习一个向量 $p_i$，然后加到 token embedding 上。

$$
h_i^{(0)} = e_i + p_i
$$

其中：

- $e_i$ 是第 $i$ 个 token 的 embedding。
- $p_i$ 是第 $i$ 个位置的 position embedding。
- $h_i^{(0)}$ 是进入 Transformer 的初始表示。

这种方法简单有效，但有几个限制：

- 最大长度通常由 position embedding table 的大小决定。
- 模型更容易记住训练中见过的绝对位置。
- 对超过训练长度的位置不天然具备外推能力。

BERT、早期 GPT 等模型常使用 learned absolute position embedding。它适合固定最大长度的模型，但对长上下文扩展不够灵活。

## 方法二：Sinusoidal Position Encoding

原始 Transformer 论文使用的是固定的 sinusoidal position encoding。它不用学习位置向量，而是用不同频率的正弦和余弦函数生成位置表示。

常见形式是：

$$
PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

$$
PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

其中：

- $pos$ 是位置编号。
- $i$ 是维度对编号。
- $d_{model}$ 是模型隐藏维度。

这种方法的直觉是：用不同频率的波来编码位置。高频维度对短距离变化敏感，低频维度对长距离变化更平滑。

Sinusoidal encoding 的优点是无需学习，可以计算任意位置的编码；但它把位置信息加到输入 embedding 上，而不是直接进入 attention score。现代大模型中更常见的是 RoPE 这类和 Q/K 匹配过程更紧密结合的方法。

## 方法三：RoPE

[[architecture/positional-encoding/rope|RoPE]] 的思路是对 query 和 key 做位置相关旋转。它不直接把位置向量加到输入 embedding 上，而是让位置信息参与 attention score 的计算。

设位置 $m$ 的 query 为 $q_m$，位置 $n$ 的 key 为 $k_n$，RoPE 后：

$$
q_m' = R_m q_m
$$

$$
k_n' = R_n k_n
$$

attention score 变为：

$$
(q_m')^T k_n' = (R_m q_m)^T(R_n k_n)
$$

由于旋转矩阵满足：

$$
R_m^T R_n = R_{n-m}
$$

所以：

$$
(q_m')^T k_n' = q_m^T R_{n-m} k_n
$$

这说明 RoPE 可以让 query-key 匹配自然感知相对距离 $n-m$。因此它非常适合 self-attention，尤其适合 decoder-only LLM。

但要严谨地说：RoPE 并不等于“无限长上下文”。它提供了较好的位置建模形式，但长上下文能力还取决于训练长度、数据分布、RoPE scaling、推理系统和 attention 本身的限制。

## 方法四：ALiBi

[[architecture/positional-encoding/alibi|ALiBi]] 的思路不是修改 embedding，也不是旋转 Q/K，而是直接给 attention score 加一个和距离相关的 bias。

简化理解：距离越远，attention score 会被加上越大的惩罚。这样模型天然更偏向近处 token，同时仍可以在需要时关注远处 token。

它的形式大致可以理解为：

$$
S_{ij}' = S_{ij} + b_{ij}
$$

其中 $b_{ij}$ 依赖位置距离。ALiBi 的一个特点是外推相对友好，因为它不依赖固定长度的 embedding table。

不过 ALiBi 和 RoPE 的归纳偏置不同：

- RoPE 通过旋转 Q/K 改变内容匹配的几何关系。
- ALiBi 通过距离 bias 直接改变 attention score。

二者都作用于 attention，但方式不同。

## 位置编码放在哪里

不同方法把位置注入到 Transformer 的不同位置：

| 方法 | 注入位置 | 影响对象 | 常见用途 |
|---|---|---|---|
| Absolute Position Embedding | token embedding | 初始 hidden states | 早期 Transformer / BERT / GPT |
| Sinusoidal Position Encoding | token embedding | 初始 hidden states | 原始 Transformer |
| RoPE | Q/K | attention score | 现代 decoder-only LLM |
| ALiBi | attention score bias | attention score | 长度外推友好的 attention bias |

这个区别很重要。位置加到 embedding 上，会让位置信息进入每一层的 hidden state；位置作用于 Q/K 或 score，则更直接影响 attention 如何分配权重。

## 绝对位置与相对位置

绝对位置回答的是：“这个 token 在第几个位置？”

相对位置回答的是：“这个 token 和另一个 token 相距多远？谁在前面？”

语言建模中，两者都有用，但相对位置往往更贴近 attention 的需求。因为 attention 关心的是当前 token 和其他 token 的关系，而不只是每个 token 的独立编号。

例如：

```text
A 在位置 10，B 在位置 12
A 在位置 300，B 在位置 302
```

二者的绝对位置不同，但相对距离都为 2。很多局部语法关系、括号匹配、短语结构可能更依赖相对位置。

RoPE 和 ALiBi 都可以看作更强调相对位置关系的方向，但表达方式不同。

## 长上下文扩展

长上下文不是简单把最大长度改大。位置机制必须能在更长位置范围内保持稳定，模型也必须学会利用长距离信息。

常见问题包括：

- 训练时没见过这么长的位置。
- 位置编码在远距离区间的行为不稳定。
- attention 可以看到远处 token，但模型不一定知道何时利用它们。
- KV Cache 显存成本随上下文长度增长。
- serving 系统需要处理更大的 prefill、decode 和 cache 管理成本。

对 RoPE 模型，常见长上下文扩展方法包括：

- 增大或调整 `rope_theta`。
- Position interpolation。
- NTK-aware scaling。
- [[architecture/positional-encoding/yarn|YaRN]]。
- 长上下文继续训练。

要严谨区分：位置扩展方法解决的是“位置表示和 attention score 的可计算性与稳定性”，不直接保证模型拥有强长文档理解能力。后者还需要数据、训练目标和评测验证。

## 与 KV Cache 的关系

位置编码也会影响推理系统。对 RoPE 模型来说，历史 key 通常带有对应 position id 的旋转信息，并保存在 [[inference/kv-cache-and-memory/kv-cache|KV Cache]] 中。

推理系统必须保证：

- 每个 token 的 position id 正确。
- prefix cache 复用时位置偏移一致。
- padding 和 batching 不导致位置错位。
- RoPE scaling 配置和模型权重匹配。

如果这些处理错误，模型可能不会报 shape error，但 attention score 会基于错误位置关系计算，输出质量可能明显下降。

## 设计取舍

| 方法 | 优势 | 局限 |
|---|---|---|
| Absolute Position Embedding | 简单、可学习、实现直接 | 外推弱，受最大位置表限制 |
| Sinusoidal Position Encoding | 无需学习，可计算任意位置 | 与 attention score 结合不如 RoPE 直接 |
| RoPE | 相对位置信息自然进入 Q/K 匹配，现代 LLM 常用 | 长上下文需要 scaling 或继续训练 |
| ALiBi | 直接作用于 score，长度外推友好 | 归纳偏置较强，不同任务/模型下效果不一定统一 |
| YaRN 等扩展 | 可提升 RoPE 长上下文能力 | 需要配置、训练或评测验证，不能盲目套用 |

## 常见误解

### 误解一：Attention 自己就能知道顺序

Attention 可以根据内容计算相关性，但如果输入没有位置机制，它不能可靠地区分 token 顺序。

### 误解二：位置编码只是给 embedding 加一个位置向量

这是 absolute position embedding 的做法，但不是所有位置机制都这样。RoPE 作用于 Q/K，ALiBi 作用于 attention score。

### 误解三：RoPE 是绝对位置编码

RoPE 的旋转由绝对 position id 决定，但 query-key dot product 中体现的是相对位置差。因此更准确地说，它用绝对位置构造旋转，并在 attention score 中表达相对位置关系。

### 误解四：长上下文扩展只改位置编码就够了

不够。位置编码扩展只是必要条件之一，还需要长上下文数据、训练或微调、检索能力、评测、以及推理系统的 KV Cache 管理。

### 误解五：最大上下文长度越长，模型一定越好

更长上下文增加可见信息，但也带来更高成本和更多噪声。模型能否有效利用长上下文，需要用 needle-in-a-haystack、长文档问答、代码仓库理解等任务验证。

## 与 Architecture 的关系

位置编码决定 Transformer 如何把序列顺序注入表示系统。它和 attention 紧密相关：

- 如果位置加到 embedding 上，后续每层都在带位置信息的 hidden states 上计算。
- 如果位置进入 Q/K 或 score，attention 权重本身直接受位置关系影响。

因此，位置编码不是一个孤立细节，而是 Transformer 架构归纳偏置的一部分。

## 与 Inference 的关系

推理系统中，位置编码影响 position id、KV Cache、prefix cache、长上下文 serving 和 batch 调度。尤其是 RoPE 模型，服务端必须确保每个请求、每个 token、每段 prefix 的位置编号一致。

长上下文模型部署时，除了看模型宣称的 context length，还要确认：

- tokenizer 和 chat template 是否正确。
- RoPE scaling 配置是否正确加载。
- KV Cache 显存是否足够。
- prefill latency 是否可接受。
- 模型是否真的能利用远距离上下文。

## 相关概念

- [[architecture/attention/attention|Attention]] — 位置编码最终影响 token 之间的信息路由。
- [[architecture/transformer/transformer|Transformer]] — 需要位置机制才能成为序列模型。
- [[architecture/positional-encoding/absolute-position|Absolute Position]] — learned absolute position embedding。
- [[architecture/positional-encoding/sinusoidal-position|Sinusoidal Position]] — 原始 Transformer 的固定位置编码。
- [[architecture/positional-encoding/rope|RoPE]] — 现代 LLM 常用的 rotary position embedding。
- [[architecture/positional-encoding/alibi|ALiBi]] — 基于 attention score bias 的位置方法。
- [[architecture/positional-encoding/yarn|YaRN]] — RoPE 长上下文扩展方法。
- [[inference/kv-cache-and-memory/kv-cache|KV Cache]] — 推理中需要正确维护位置编号和缓存。

## 经典论文与资料

- [[sources/papers/2017-attention-is-all-you-need|Attention Is All You Need]]
- [[sources/papers/2021-roformer|RoFormer]]
- [[sources/papers/2021-alibi|ALiBi]]
- [[sources/papers/2023-yarn|YaRN]]
