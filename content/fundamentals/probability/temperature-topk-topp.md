---
title: Temperature Top-k and Top-p
created: 2025-12-21
published: 2025-12-21
modified: 2025-12-21
type: topic
status: growing
area: fundamentals
tags:
  - math
  - probability
  - decoding
aliases:
  - Temperature
  - Top-k
  - Top-p
  - Nucleus Sampling
---

## 概念界定

Temperature、Top-k 和 Top-p 是语言模型推理阶段常见的采样控制方法。它们不改变模型参数，也不改变模型已经学到的知识，而是改变从下一个 token 概率分布中选择候选 token 的方式。

## 背景与问题

语言模型每一步先输出 logits，再通过 softmax 得到词表上的概率分布。如果总是选择概率最高的 token，输出会稳定但可能重复、保守、缺乏多样性；如果完全按原始分布采样，又可能选到低概率、不合适的 token。

因此，解码参数要解决的问题是：在确定性、多样性和风险之间做权衡。

- 需要稳定事实回答时，通常希望分布更尖锐、随机性更低。
- 需要创意写作或多样候选时，通常允许更多随机性。
- 需要避免低质量 token 时，通常会截断长尾候选。

## 定义与记号

### Temperature

Temperature 通过缩放 logits 改变 softmax 后的分布形状：

```text
p_i = softmax(z_i / T)
```

其中 `z_i` 是第 `i` 个 token 的 logit，`T` 是 temperature。

- `T = 1`：保持原始分布。
- `T < 1`：放大 logit 差异，使分布更尖锐。
- `T > 1`：缩小 logit 差异，使分布更平坦。
- `T -> 0`：接近贪心解码，几乎只选最高 logit。

### Top-k

Top-k 只保留概率最高的 `k` 个 token，丢弃其他 token，再对保留集合重新归一化后采样：

```text
候选集合 = 概率最高的 k 个 token
p'_i = p_i / Σ_{j in TopK} p_j, if i in TopK
p'_i = 0, otherwise
```

Top-k 控制的是候选集合的数量。

### Top-p

Top-p 也叫 nucleus sampling。它按概率从高到低排序，保留累计概率达到阈值 `p` 的最小 token 集合，再重新归一化采样：

```text
选择最小集合 S，使得 Σ_{i in S} p_i >= p
```

Top-p 控制的是保留的概率质量，而不是固定 token 数量。

## 直观解释

Temperature 像是在调节模型“敢不敢选非最高概率 token”。Temperature 低时，模型更保守；Temperature 高时，模型更发散。

Top-k 像是只允许模型从前 `k` 个候选里选。无论当前分布是集中还是分散，候选数量都固定。

Top-p 像是只保留“主要概率质量”。如果分布很集中，它会保留很少 token；如果分布很分散，它会保留更多 token。

## 基本性质

- Temperature 改变整个概率分布的形状。
- Top-k 截断 token 数量，简单直接，但不适应不同上下文下的分布形状。
- Top-p 截断累计概率质量，候选数量会随上下文动态变化。
- Temperature 可以和 Top-k / Top-p 一起使用。
- 解码参数影响输出风格、多样性、重复率、事实稳定性和评测可复现性。

## 示例

假设某一步模型输出的概率分布是：

```text
A: 0.50
B: 0.20
C: 0.12
D: 0.08
E: 0.05
others: 0.05
```

### Top-k 示例

如果 `top-k = 3`，只保留：

```text
A, B, C
```

然后在这三个 token 上重新归一化采样。`D`、`E` 和 `others` 即使仍有概率，也不会被采样到。

### Top-p 示例

如果 `top-p = 0.8`，按概率从高到低累加：

```text
A: 0.50
A + B: 0.70
A + B + C: 0.82
```

达到 0.8 后停止，因此候选集合是：

```text
A, B, C
```

如果分布更集中：

```text
A: 0.82
B: 0.08
C: 0.04
...
```

`top-p = 0.8` 可能只保留 `A`。这就是 Top-p 比 Top-k 更动态的地方。

### Temperature 示例

如果 logits 对应的原始分布比较接近：

```text
A: 0.35
B: 0.30
C: 0.20
D: 0.15
```

降低 temperature 会让 `A` 相对更突出；升高 temperature 会让 `B/C/D` 更有机会被采样到。

## 使用直觉

- 事实问答、代码生成、结构化输出：倾向使用较低 temperature，降低随机性。
- 创意写作、头脑风暴、多候选生成：可以提高 temperature 或使用较大的 top-p。
- 想避免长尾低质量 token：使用 Top-k 或 Top-p 截断候选集合。
- 需要可复现评测：固定解码参数，并尽量减少随机采样。

这些只是经验直觉，不是固定规则。不同模型、任务和 prompt 对参数的敏感性可能不同。

## 常见误解

- 误解：temperature 越高越聪明。
  - 正确理解：更高 temperature 增加随机性和多样性，不代表模型能力更强。
- 误解：Top-k 和 Top-p 是同一种方法。
  - 正确理解：Top-k 固定候选数量，Top-p 固定累计概率质量。
- 误解：调解码参数能修复模型知识错误。
  - 正确理解：它们只改变输出选择方式，不改变模型已经学到的分布。
- 误解：低 temperature 一定更准确。
  - 正确理解：它更稳定，但可能更保守，也可能固化错误高概率答案。

## 相关概念

- [[fundamentals/probability/sampling|采样]] — 这些方法都是采样控制策略。
- [[fundamentals/probability/distribution|概率分布]] — 解码方法作用于 token 概率分布。
- [[fundamentals/neural-network-basics/softmax|Softmax]] — temperature 作用在 softmax 前的 logits 上。
- [[application/evaluation/evaluation|评测与 Benchmark]] — 解码参数会影响评测结果可复现性。
