---
title: Learning Rate Scheduler
created: 2026-02-11
published: 2026-02-11
modified: 2026-02-18
type: topic
status: growing
area: fundamentals
tags:
  - optimization
  - learning-rate
  - scheduler
aliases:
  - Learning Rate Scheduler
  - LR Scheduler
  - Warmup
---

## 概念界定

学习率调度是让学习率随训练步数变化的策略。常见调度包括 warmup、linear decay、cosine decay 和 constant schedule。

## 背景与问题

训练初期模型参数和优化器状态尚不稳定，直接使用较大学习率容易导致 loss spike。训练后期如果学习率仍然过大，模型可能难以收敛到更好的区域。因此大模型训练通常使用分阶段学习率调度。

## 定义与记号

Warmup：

```text
lr_t = peak_lr · t / warmup_steps
```

Linear decay：

```text
lr_t = peak_lr · (1 - progress)
```

Cosine decay：

```text
lr_t = min_lr + 0.5(peak_lr - min_lr)(1 + cos(π · progress))
```

## 直观解释

Warmup 像训练刚开始时慢慢加速，避免一上来步子太大。Decay 像训练后期逐渐放慢脚步，让参数在较好区域附近稳定下来。

## 基本性质

- Warmup 常用于大模型训练初期稳定优化。
- Cosine decay 平滑下降，常见于预训练。
- Linear decay 简单直接，常见于微调任务。
- 最小学习率是否为 0 取决于训练配方。
- 调度策略需要与总训练步数、batch size 和优化器配合。

## 示例

典型 LLM 训练学习率曲线：

```text
0 -> peak_lr -> gradually decay to min_lr
```

如果 warmup 太短，训练初期可能不稳定；如果 warmup 太长，可能浪费训练步数。

## 常见误解

- 误解：warmup 只是可选小技巧。
  - 正确理解：在大模型训练中，warmup 往往是稳定训练的重要组成。
- 误解：学习率调度只影响训练速度。
  - 正确理解：它也会影响最终 loss、稳定性和泛化。
- 误解：所有任务都适合同一种 schedule。
  - 正确理解：预训练、SFT、LoRA 微调可能使用不同调度。

## 相关概念

- [[learning-rate|学习率]] — 调度控制学习率随时间变化。
- [[adam|Adam]] — 优化器更新仍受学习率调度影响。
- [[training/pretraining/pretraining|预训练]] — 常使用 warmup + decay。
- [[training/post-training/sft|SFT 监督微调]] — 微调常使用更短训练和不同调度。
