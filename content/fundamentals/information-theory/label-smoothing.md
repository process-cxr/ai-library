---
title: Label Smoothing
created: 2025-12-28
published: 2025-12-28
modified: 2026-09-07
type: topic
status: growing
area: fundamentals
tags:
  - information-theory
  - loss-function
  - regularization
aliases:
  - 标签平滑
---

## 概念界定

Label Smoothing 是一种目标分布平滑方法：不再把真实标签视为完全 one-hot，而是给非真实类别分配少量概率质量。它常用于缓解模型过度自信。

## 背景与问题

标准交叉熵训练中，真实标签通常是 one-hot 分布。这会鼓励模型把真实类别概率推到接近 1，把其他类别概率推到接近 0。对于分类任务，这可能导致模型过度自信，影响泛化和校准。

## 定义与记号

假设类别数为 $K$，平滑系数为 $\varepsilon$。一种常见定义是把 $\varepsilon$ 分配给所有类别：

$$
p_j^{\mathrm{smooth}}
=(1-\varepsilon)\mathbf{1}[j=y]+\frac{\varepsilon}{K}
$$

此时 target class 的概率为 $1-\varepsilon+\varepsilon/K$，其他类别为 $\varepsilon/K$。另一种约定会保留 $1-\varepsilon$ 给 target，并将 $\varepsilon$ 仅分给 $K-1$ 个非 target 类别：

$$
p_y=1-\varepsilon,
\qquad
p_{j\ne y}=\frac{\varepsilon}{K-1}
$$

两种定义数值不同，阅读实现时需要确认采用哪一种。平滑后的 Cross Entropy 仍是 $H(p^{\mathrm{smooth}},q)$，其 logit gradient 仍满足：

$$
\frac{\partial L}{\partial z_j}=q_j-p_j^{\mathrm{smooth}}
$$

## 直观解释

Label Smoothing 相当于告诉模型：即使正确答案是某个类别，也不要对它过度自信。这会让目标分布更平滑，降低模型把概率全部压到单一类别上的倾向。

## 基本性质

- 可以缓解过度自信。
- 可能改善泛化和校准。
- 对需要极高置信度区分的任务，过强平滑可能损害性能。
- 在大语言模型预训练中是否使用、如何使用，需要结合具体训练配方。
- 它会改变 calibration、logit margin 和表示几何，收益不能只用 training loss 判断。

## 与 LLM Training 的关系

LLM vocabulary 很大，$\varepsilon/K$ 对单个错误 token 很小，但总体仍有 $\varepsilon$ 的概率质量被移出 hard target。Label Smoothing 可能缓解过度尖锐的 token distribution，却也会削弱 rare token、严格格式 token 或确定性目标上的梯度。

它与 Knowledge Distillation 都使用 soft targets，但来源不同：Label Smoothing 按固定规则分配概率质量；distillation target 来自 teacher logits，包含类别之间的非均匀相似性信息。

## 常见误解

- **Label Smoothing 会改变真实答案。** 它改变训练 target distribution，不是否认 hard label。
- **所有实现的 $\varepsilon$ 含义都相同。** 概率质量可以分给全部类别，也可以只分给非 target 类别。
- **平滑越强越好。** 过强平滑会削弱监督信号，并可能损害确定性生成任务。

## 相关概念

- [[fundamentals/information-theory/cross-entropy|交叉熵]]：Label Smoothing 修改交叉熵中的目标分布。
- [[training/post-training/logits-distillation|Logits Distillation]]：teacher distribution 提供另一类 soft target。
- [[fundamentals/probability/calibration-uncertainty|校准与不确定性]]：平滑常与校准问题相关。
