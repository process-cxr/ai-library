---
title: Annealing
created: 2026-02-28
published: 2026-02-28
modified: 2026-05-31
type: topic
status: mature
area: training
tags:
  - mid-training
  - annealing
---

Annealing 在大模型训练报告中通常指训练末期的收敛整理阶段：降低 learning rate，并常常切换到更高质量、更接近目标能力的数据混合，以改善模型稳定性、领域能力、推理表现或最终 checkpoint 质量。

它不是单一算法，而是一种阶段性训练策略。其核心思想是：早期训练强调覆盖和规模，末期训练强调高质量信号、能力整理和避免继续用噪声数据扰动模型。

## 两种 Annealing

大模型语境中的 annealing 至少包含两层含义：

1. **Learning rate annealing**：学习率逐步下降，例如 cosine decay、linear decay 或 step decay。
2. **Data annealing**：训练末期改变 data mix，提高高质量、目标领域、数学、代码、近期知识或长上下文数据比例。

二者经常同时出现，但应区分。只降低 learning rate 不等于 data annealing；只换高质量数据但 learning rate 不合适，也可能造成 loss spike 或过拟合。

## 为什么在末期换数据

预训练早期需要大规模、多样化数据来学习通用语言和知识。训练后期，模型已经具备基础能力，此时继续用大量低质或重复数据，边际收益可能下降，还可能污染最终能力。

末期 data annealing 的目标包括：

- 提高单位 token 的训练价值；
- 强化数学、代码、多语言、工具、长上下文等目标能力；
- 注入较新知识；
- 降低低质量网页、重复模板和噪声数据影响；
- 选择更适合 post-training 的 base checkpoint。

这解释了为什么一些模型报告会在最后阶段上采样高质量数据或特定能力数据。

## 训练流程

一个典型 annealing 阶段：

1. 从主预训练 checkpoint 开始；
2. 降低 learning rate 或进入 decay 尾段；
3. 切换到 curated data mix；
4. 加强 validation 和 checkpoint 保存；
5. 监控通用能力与目标能力；
6. 从多个 late checkpoints 中选择最终 base。

Annealing 阶段 token 量通常远小于主预训练，但影响可能显著，因为它靠近最终 checkpoint。

## Data Mix

Annealing 数据常见组成：

- 高质量通用文本；
- 代码和技术文档；
- 数学题解和证明；
- 学术文本；
- 多语言高质量数据；
- 近期网页或新闻，若需要知识更新；
- 长上下文文档；
- 少量高质量合成数据。

需要谨慎的是：末期数据比例会强烈影响模型最终风格和能力。如果合成数据、数学题解或 instruction-like 数据比例过高，模型可能在 base 阶段提前形成某种回答模式或偏好。

## 与 CPT 和 SFT 的关系

Annealing 可以看作 CPT 的一种特殊形式：它发生在训练末期，目标是整理能力和选择更好 base checkpoint。与 SFT 不同，它通常仍使用 language modeling objective，不主要训练 assistant 对话格式。

如果 annealing 数据包含大量 prompt-response 或 chat 格式，就会和 SFT 边界变得模糊。此时需要明确：

- 是否使用 chat template；
- 哪些 token 计算 loss；
- 是否希望 base model 学习 assistant 风格；
- 后续 SFT 是否会覆盖这些格式偏置。

## 风险

- **过拟合高质量小数据**：目标 benchmark 上升，但通用能力下降。
- **风格偏移**：模型生成更像某类数据，例如题解、代码注释或论文。
- **污染放大**：末期数据若包含 benchmark，影响比早期更直接。
- **学习率不匹配**：数据切换导致 loss spike。
- **checkpoint 选择偏差**：只按单一 benchmark 选最终模型。

## 评测

Annealing 应观察：

- total validation loss；
- annealing data validation loss；
- 通用 benchmark；
- 目标能力 benchmark；
- contamination；
- 多个 late checkpoints 的能力曲线；
- 后训练后的能力保留。

如果某个 checkpoint 在目标能力上最好，但通用能力退化明显，可能不适合作为最终 base。

## 相关概念

- [[training/mid-training/continued-pretraining|Continued Pretraining]]
- [[training/pretraining/data-mix|Data Mix]]
- [[training/data-engineering/quality-filtering|Quality Filtering]]
- [[training/scaling/training-budget|Training Budget]]
- [[training/optimization/training-stability|Training Stability]]
