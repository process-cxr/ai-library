---
title: Information-Theoretic View of Knowledge Distillation
created: 2026-01-03
published: 2026-01-03
modified: 2026-01-03
type: topic
status: growing
area: fundamentals
tags:
  - information-theory
  - distillation
  - compression
aliases:
  - Knowledge Distillation
  - Distillation
---

## 概念界定

知识蒸馏是让学生模型学习教师模型输出分布的方法。从信息论角度看，它不是只学习 hard label，而是让学生分布接近教师分布，从而继承教师模型对类别或 token 之间关系的软信息。

## 背景与问题

真实标签通常只告诉模型“正确答案是什么”，但教师模型的输出分布还包含其他候选答案的相对概率。例如第二候选和第三候选哪个更合理，这些信息可能有助于学生模型学习更平滑的决策边界。

## 定义与记号

教师分布：

```text
q_T(y | x)
```

学生分布：

```text
q_S(y | x)
```

蒸馏目标通常让学生分布接近教师分布，例如最小化：

```text
KL(q_T || q_S)
```

或使用教师 soft labels 的交叉熵。

## 直观解释

如果 hard label 只告诉学生“答案是 A”，教师分布可能告诉学生：

```text
A: 0.70
B: 0.20
C: 0.08
D: 0.02
```

这比 one-hot 标签包含更多结构信息。学生不仅知道 A 最好，也知道 B 比 D 更接近。

## 基本性质

- 蒸馏可以用于模型压缩、能力迁移和对齐数据生成。
- 温度参数常用于软化教师分布，使暗知识更明显。
- 学生最终能力受教师质量、数据覆盖和学生容量限制。
- 分布匹配不保证学生完全复制教师的推理过程。

## 示例

小模型学习大模型输出：

```text
teacher logits -> softened distribution
student logits -> student distribution
loss = distillation loss + optional hard-label loss
```

这可以让小模型在较低成本下接近大模型部分能力。

## 常见误解

- 误解：蒸馏只是复制最终答案。
  - 正确理解：蒸馏通常学习教师输出分布，包含候选之间的相对关系。
- 误解：学生模型可以无损继承教师模型能力。
  - 正确理解：学生容量、训练数据和目标都会限制蒸馏效果。
- 误解：KL 方向无所谓。
  - 正确理解：不同 KL 方向会带来不同的分布拟合行为。

## 相关概念

- [[fundamentals/information-theory/kl-divergence|KL 散度]] — 蒸馏中常用的分布匹配目标。
- [[fundamentals/information-theory/cross-entropy|交叉熵]] — soft label 训练目标。
- [[inference/compression/model-compression|模型压缩]] — 蒸馏常用于压缩和部署。
