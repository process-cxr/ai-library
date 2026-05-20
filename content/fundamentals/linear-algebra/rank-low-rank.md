---
title: Rank and Low-Rank Approximation
created: 2026-01-13
published: 2026-01-13
modified: 2026-01-20
type: topic
status: growing
area: fundamentals
tags:
  - math
  - linear-algebra
  - low-rank
aliases:
  - Rank
  - Low-rank Approximation
---

## 概念界定

矩阵的秩刻画矩阵中线性独立信息的数量。低秩近似试图用较少的独立方向近似原矩阵，是理解 LoRA、参数高效微调、矩阵压缩和表示冗余的重要基础。

## 背景与问题

大模型包含大量高维权重矩阵，但并不是所有任务都需要修改这些矩阵的全部自由度。经验上，许多下游任务的参数更新可能集中在较低维的子空间中。低秩思想为“用较少参数有效改变模型行为”提供了线性代数视角。

## 定义与记号

矩阵 `W ∈ R^{d_in × d_out}` 的秩可以理解为它包含的独立方向数量。

低秩分解形式：

```text
W ≈ A B
A ∈ R^{d_in × r}
B ∈ R^{r × d_out}
```

当 `r << min(d_in, d_out)` 时，`A` 和 `B` 的参数量远小于原矩阵。

## 直观解释

如果一个矩阵的作用主要集中在少数方向上，就可以用这些主要方向近似原矩阵。低秩近似不是声称原矩阵没有复杂性，而是尝试保留最重要的变化方向。

## 基本性质

- 秩越高，矩阵可以表达的独立方向越多。
- 低秩近似通常会带来信息损失，但可能显著减少参数或计算。
- 低秩分解的效果取决于目标任务所需的更新是否集中在少数方向。
- rank 是一个容量超参数，过小可能表达不足，过大可能增加成本和过拟合风险。

## 示例

LoRA 的核心形式：

```text
W' = W + ΔW
ΔW = A B
```

其中原始权重 `W` 冻结，只训练低秩矩阵 `A` 和 `B`。这样可以用较少参数适配下游任务。

参数量对比：

```text
Full update: d_in * d_out
LoRA update: d_in * r + r * d_out
```

当 `r` 很小时，训练参数大幅减少。

## 常见误解

- 误解：低秩就是低质量。
  - 正确理解：如果任务所需变化集中在少数方向，低秩更新可以非常有效。
- 误解：LoRA 是对原模型做压缩。
  - 正确理解：LoRA 通常是在冻结原权重基础上学习低秩增量。
- 误解：rank 越大一定越好。
  - 正确理解：rank 增大会提高容量，也会增加参数、显存和过拟合风险。

## 相关概念

- [[fundamentals/linear-algebra/eigendecomposition-svd|特征分解与 SVD]] — 低秩近似的重要工具。
- [[fundamentals/linear-algebra/linear-transformation|线性变换]] — 权重矩阵的表示空间作用。
- [[training/post-training/sft|SFT 监督微调]] — LoRA 常用于监督微调场景。
- [[inference/compression/model-compression|模型压缩]] — 低秩近似的工程应用之一。
