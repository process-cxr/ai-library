---
title: Learning Rate
created: 2026-01-11
published: 2026-01-11
modified: 2026-08-11
type: topic
status: mature
area: fundamentals
tags:
  - optimization
  - learning-rate
  - training
aliases:
  - Learning Rate
---

## 概念界定

学习率是控制参数更新步长的超参数。它不决定更新方向，而是决定优化器给出方向后，参数沿该方向移动多远。对大模型训练来说，学习率不是一个孤立数字，而是和 batch size、optimizer state、warmup、训练阶段一起决定实际更新强度。

## 背景与问题

大模型训练通常非常昂贵，学习率设置不当可能导致训练发散、收敛过慢或最终效果不佳。即使使用 AdamW，仍然需要同时看学习率大小、调度策略、梯度噪声和参数更新尺度。

## 定义与记号

基本更新中：

```text
θ_{t+1} = θ_t - η g_t
```

`η` 就是学习率。

对于 Adam 类优化器：

```text
θ_{t+1} = θ_t - η · normalized_update
```

学习率仍然控制全局更新幅度。

更一般地，可以把一层参数的相对更新尺度写成：

```text
R_l = |Δθ_l|_2 / |θ_l|_2
```

其中 `R_l` 描述这一层在一个 step 中相对自身参数规模移动了多少。对 Adam 类优化器，`η` 只是缩放因子，真正的更新还取决于梯度历史、二阶矩归一化和参数本身的尺度。

## 直观解释

学习率像走路步长。步子太大容易越过低谷甚至摔出去；步子太小虽然稳，但训练成本很高。

## 基本性质

- 学习率过大可能导致 loss spike、训练发散或已有能力被破坏。
- 学习率过小会导致训练慢、收敛弱或更新不足。
- 合适学习率与 batch size、模型规模、优化器、数据分布和训练阶段有关。
- 大模型训练通常使用 warmup 后再 decay 的策略。
- 同一个学习率在不同层、不同参数组上的实际更新强度可能不同。

## 与 Batch Size 的关系

Batch size 影响的是梯度估计的噪声，不是平均梯度的期望值。若 batch loss 采用 mean reduction，则：

```text
g_B = (1/B) Σ_i g_i
```

batch 增大后，梯度更稳定，优化器通常可以容忍更激进的学习率，但这不是严格线性关系。更稳妥的理解是：

- batch size 决定梯度噪声；
- learning rate 决定沿估计方向走多远；
- gradient accumulation 还要先确认最后是求平均还是求和。

如果 accumulation 之后梯度是 mean，增加 accumulation steps 不会线性放大梯度；如果实现成 sum，则等价于放大了有效更新尺度，学习率需要相应降低。

## 与训练阶段的关系

不同训练阶段允许的参数位移不同：

- Pretraining：目标是高效学习基础能力，通常使用相对更高但仍稳定的 learning rate。
- Continued Pretraining / Mid-training：模型已经具备基础能力，learning rate 往往比 pretraining 更保守，用来注入新能力而不破坏已有表示。
- SFT：训练目标更多是行为塑形，learning rate 一般进一步降低。
- PPO / GRPO / 其他 RL 阶段：除了学习率，还要看 policy drift 和 KL，训练强度通常更敏感。

所以 learning rate 不是“预训练一个值、微调一个值”这么简单，而是和训练阶段一起决定更新预算。

一个常见的经验参考可以写成：

| 训练阶段 | 主要目标 | 常见 LR 起点 | 相对 pretraining |
|---|---|---|---|
| Pretraining | 学基础能力，尽量高效但稳定 | `1e-4` 左右常作为常见起点 | `1x` |
| Continued Pretraining / Mid-training | 注入新能力，同时保留已有表示 | `1e-5 ~ 5e-5` 常见 | `0.1x ~ 0.5x` |
| SFT | 塑形行为、格式和指令跟随 | `1e-6 ~ 1e-5` 常见 | `0.01x ~ 0.1x` |
| PPO / GRPO / 其他 RL | 控制 policy drift，避免更新过猛 | 往往再低一档，常与 KL 一起调 | 通常更低 |

这个表不是固定常数，只是帮助判断“当前阶段该更激进还是更保守”。

## 示例

一个常见训练阶段：

```text
warmup: 学习率从 0 增加到 peak lr
decay: 学习率从 peak lr 逐步降低
```

warmup 可以避免训练初期参数和优化器状态尚不稳定时更新过大。

对大模型训练，更常见的是：

```text
warmup -> peak lr plateau -> cosine decay / linear decay
```

有些继续预训练或微调阶段会把 peak lr 压得更低，并缩短 warmup，以减少对已有表示的扰动。

## 常用判断信号

- 训练早期 loss spike 频繁：常见原因是 lr 偏大、warmup 不足或 batch 噪声太强。
- 训练推进缓慢：可能是 lr 偏小，也可能是 decay 太早。
- CPT / mid-training 后通用能力下降：常见原因是学习率过高或 replay 不足。
- SFT 之后回答风格变差：有时不是数据问题，而是 lr 把 base 表示扰乱了。

## 常见误解

- 误解：学习率越大训练越快，所以越好。
  - 正确理解：过大学习率会导致不稳定或发散。
- 误解：Adam 会自动选择学习率。
  - 正确理解：Adam 自适应缩放各参数更新，但仍需要全局学习率。
- 误解：同一个学习率适用于所有规模模型。
  - 正确理解：学习率通常需要随模型规模、batch size 和训练 token 数调整。
- 误解：batch 变大，梯度就一定按比例变大。
  - 正确理解：在 mean reduction 下，batch 主要改变梯度方差，不改变平均梯度的量级。

## 相关概念

- [[lr-scheduler|学习率调度]] — 学习率如何随训练步数变化。
- [[adam|Adam]] — 学习率控制 Adam 更新的全局尺度。
- [[batch-size-gradient-noise|Batch Size 与梯度噪声]] — batch size 会影响学习率选择。
- [[training/optimization/training-stability|Training Stability]] — 学习率过大常表现为 loss spike 或发散。
- [[training/mid-training/continued-pretraining|Continued Pretraining]] — 不同训练阶段通常对应不同学习率尺度。
  - [[training/scaling/scaling-law|Scaling Law]] — 规模变化会影响训练配方。
