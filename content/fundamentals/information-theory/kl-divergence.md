---
title: KL 散度
type: topic
status: growing
area: fundamentals
tags:
  - math
  - information-theory
  - alignment
aliases:
  - KL Divergence
  - Kullback-Leibler Divergence
---

## 概念界定

KL 散度衡量当真实分布为 `p`，却使用另一个分布 `q` 进行近似时产生的额外信息代价。它常用于刻画两个概率分布之间的差异，但它不是严格意义上的距离。

## 背景与问题

大模型训练和对齐经常涉及“让一个分布接近另一个分布”。预训练希望模型分布接近数据分布，RLHF 希望策略模型提升偏好回报但不要偏离参考模型太远，蒸馏希望学生模型接近教师模型。KL 散度是这些问题中常见的数学工具。

## 定义与记号

离散分布 `p` 和 `q` 的 KL 散度定义为：

```text
KL(p || q) = Σ_x p(x) log(p(x) / q(x))
```

也可写成：

```text
KL(p || q) = E_{x~p}[log p(x) - log q(x)]
```

## 直观解释

KL 散度衡量：如果数据实际来自 `p`，但你用 `q` 来编码或预测，会比使用 `p` 多付出多少额外代价。`q` 越接近 `p`，KL 越小。

## 基本性质

- KL 散度非负。
- 当且仅当两个分布相同，KL 为 0。
- KL 非对称：`KL(p || q)` 通常不等于 `KL(q || p)`。
- KL 不满足距离度量的所有性质，因此不应简单叫“距离”。

## 前向 KL 与反向 KL

两个方向的 KL 行为不同：

```text
Forward KL: KL(p || q)
Reverse KL: KL(q || p)
```

直观上：

- `KL(p || q)` 更强调覆盖真实分布 `p` 的高概率区域，倾向 mode-covering。
- `KL(q || p)` 更害怕把概率放到 `p` 很低的位置，倾向 mode-seeking。

这个差异会影响变分推断、生成建模和策略优化中的行为。

## 示例

RLHF 中常见 KL 惩罚形式：

```text
reward_total = reward_model_score - β KL(π_θ || π_ref)
```

其中 `π_θ` 是当前策略模型，`π_ref` 是参考模型。KL 惩罚用于限制模型为了追求奖励而过度偏离原始语言分布。

## 常见误解

- 误解：KL 散度是距离。
  - 正确理解：KL 非对称，也不满足所有距离性质。
- 误解：KL 越小一定越好。
  - 正确理解：如果任务需要模型行为改变，过强 KL 约束可能限制学习。
- 误解：KL 只用于 RLHF。
  - 正确理解：KL 也出现在变分推断、蒸馏、正则化和分布对齐中。

## 相关概念

- [[fundamentals/information-theory/cross-entropy|交叉熵]] — KL 与交叉熵存在分解关系。
- [[fundamentals/information-theory/cross-entropy-kl-relation|交叉熵与 KL 的关系]] — 详细解释二者联系。
- [[training/post-training/rlhf|RLHF]] — KL 惩罚常用于限制策略漂移。
- [[training/post-training/dpo|DPO]] — 偏好优化中也涉及参考策略约束。
- [[fundamentals/information-theory/js-divergence|JS 散度]] — 与 KL 相关的对称分布差异度量。
