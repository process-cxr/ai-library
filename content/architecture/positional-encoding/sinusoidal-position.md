---
title: Sinusoidal Position Encoding
created: 2026-02-01
published: 2026-02-01
modified: 2026-05-31
type: topic
status: mature
area: architecture
tags:
  - positional-encoding
  - sinusoidal
---

Sinusoidal Position Encoding 是原始 Transformer 论文使用的固定位置编码方法。它不用学习位置表，而是用不同频率的正弦和余弦函数为每个位置生成向量，再加到 token embedding 上。

## 基本形式

设位置为 $pos$，维度索引为 $i$，模型维度为 $d_{model}$。常见形式为：

$$
PE(pos,2i)=\sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

$$
PE(pos,2i+1)=\cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

然后与 token embedding 相加：

$$
h_{pos}^{(0)}=e_{pos}+PE(pos)
$$

偶数维使用 sine，奇数维使用 cosine；不同维度对应不同频率。

## 频率直觉

Sinusoidal encoding 的直觉是用多种波长同时表示位置：

- 高频维度对短距离变化敏感；
- 低频维度变化更慢，可以覆盖更长距离；
- sine/cosine 成对出现，使相位关系可以表达位置差异。

这和 Fourier features 有相似直觉：用一组周期函数把标量位置映射到高维表示中。

## 为什么有一定外推性

因为 sinusoidal position encoding 是固定函数，不依赖有限大小的 learned table，所以可以为超过训练长度的位置计算编码。相比 learned absolute position embedding，它至少在形式上支持任意位置。

但这不等于模型一定能有效利用超长上下文。训练时没有见过的长度、attention 成本、数据分布和任务模式仍然会限制外推效果。

## 与 Absolute Position Embedding 的区别

| 方法 | 参数 | 超出训练长度的位置 | 特点 |
|---|---|---|---|
| Learned absolute position | 可学习位置表 | 通常没有自然定义 | 适配训练分布，外推弱 |
| Sinusoidal position | 固定函数 | 可以计算 | 外推形式更自然，但不一定效果强 |

二者都把位置向量加到输入 embedding 上，因此位置信息进入 hidden states，而不是直接作用在 attention score 上。

## 与 RoPE 的关系

[[architecture/positional-encoding/rope|RoPE]] 也使用频率和旋转思想，但作用位置不同。Sinusoidal encoding 生成一个位置向量加到 token embedding；RoPE 则对 Q/K 做位置相关旋转，使相对位置进入 query-key dot product。

因此 RoPE 与 attention 的结合更直接，也成为现代 decoder-only LLM 中更常见的位置机制。

## 局限

- 位置信息只在输入层相加，后续层如何使用完全由模型学习；
- 周期函数可能在很长位置范围内出现复杂周期行为；
- 形式可外推不代表能力可外推；
- 现代 LLM 中通常被 RoPE、ALiBi 或长上下文位置扩展方法替代。

## 常见误解

- **误解：sinusoidal encoding 是 learned embedding。** 它是固定函数，没有可学习位置表。
- **误解：可以计算任意位置就等于无限上下文。** 模型是否能利用长上下文还取决于训练和系统。
- **误解：它只表示绝对位置。** 它以绝对位置生成向量，但 sine/cosine 结构让相对位移可以被线性关系表达。

## 相关概念

- [[architecture/positional-encoding/positional-encoding|Positional Encoding]]
- [[architecture/positional-encoding/absolute-position|Absolute Position Embedding]]
- [[architecture/positional-encoding/rope|RoPE]]
- [[architecture/attention/attention|Attention]]
