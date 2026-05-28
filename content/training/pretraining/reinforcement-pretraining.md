---
title: Reinforcement Pretraining
created: 2026-05-28
published: 2026-05-28
modified: 2026-05-28
type: topic
status: growing
area: training
tags:
  - pretraining
  - reinforcement-learning
  - reasoning
---

Reinforcement Pretraining 指把强化学习式目标前移到预训练或 continued pretraining 阶段，而不是只在 SFT 之后作为 RLHF / RLVR 使用。它试图在 base model 形成阶段就塑造探索、推理或更有效的中间表示。

## 目标与问题

标准 [[training/pretraining/objective|Training Objective]] 通常是 next-token prediction：给定 $x_{<t}$，最大化真实下一个 token $x_t$ 的 likelihood。这个目标可扩展、稳定，但它没有显式奖励模型在预测前进行探索或推理。

Reinforcement Pretraining 想解决的问题是：能否在普通文本上构造可扩展的 reward，让模型在预训练阶段就学习“产生有用中间思考”或“选择更有信息量的内部动作”？

## RLP 的代表性机制

[[sources/papers/2025-rlp-reinforcement-as-a-pretraining-objective|RLP: Reinforcement as a Pretraining Objective]] 是一个典型例子。它把 Chain-of-Thought 视为预测下一 token 之前的 action：

1. 给定上下文 $x_{<t}$，模型采样 thought $c_t$。
2. 同一个模型在 $x_{<t}, c_t$ 条件下预测真实 token $x_t$。
3. EMA teacher 在不使用 thought 的条件下预测 $x_t$。
4. reward 是两者 log-likelihood 的差：

$$
r(c_t)=\log p_\theta(x_t \mid x_{<t}, c_t)-\log \bar{p}_\phi(x_t \mid x_{<t})
$$

如果 thought 提高了真实 token 的概率，它获得正奖励；如果没有帮助，则 reward 较低甚至为负。

RLP 的 thought 是训练时由当前模型自己 rollout 出来的，而不是离线固定的 CoT 数据。每个 training step 会用当前策略快照采样多条 thoughts，计算组内 reward 和 advantage，然后更新 thought tokens 的生成概率；下一步再用更新后的模型继续采样新的 thoughts。因此它更接近 on-policy / near on-policy reinforcement pretraining。

## 与 Continued Pretraining 的区别

Continued pretraining 仍然通常使用最大似然目标，只是继续在新数据、领域数据或高质量数据上训练。Reinforcement Pretraining 的关键变化不是“多训练一段”，而是改变训练信号：

| 维度 | Continued Pretraining | Reinforcement Pretraining |
|---|---|---|
| 主要目标 | 最大化真实 token likelihood | 最大化 action / thought 带来的 reward |
| 数据需求 | 普通文本或领域文本 | 取决于 reward，可使用普通文本或 verifier 数据 |
| 信号密度 | 每个 token 的监督 loss | 可以是 dense reward，也可以是 sparse reward |
| 行为激励 | 隐式学习语言模式 | 显式激励探索、推理或策略选择 |

RLP 的特殊之处在于它把 reward 设计成 verifier-free dense information gain，因此可以作用于普通文本流，而不是只依赖数学题、代码题或有标准答案的数据。

## 与后训练 RL 的区别

后训练 RL 通常发生在 instruction-following 或 assistant policy 已经形成之后，目标是优化偏好、可验证正确性或安全行为。Reinforcement Pretraining 则发生得更早，目标是塑造 base model 的基础能力。

这带来两个后果：

- 如果有效，pretraining 阶段形成的能力可能与后续 SFT / RLVR 叠加，而不是被后训练覆盖。
- reward 必须更通用、更密集、更便宜，否则无法承受预训练规模。

## 设计取舍

Reinforcement Pretraining 的优势是把 reasoning signal 提前注入基础模型，可能减少后训练阶段才“补课”的压力。它也能把普通文本转成更丰富的训练信号，而不只是监督下一个 token。

主要代价在于工程复杂度和稳定性：

- rollout 会显著增加计算；
- reward 如果来自模型自身，可能有 reward hacking 或自指偏差；
- thought 是否忠实反映真实推理过程仍然不确定；
- 与 NTP loss 如何混合、何时切换、训练多长，都还需要经验规则。

## 相关概念

- [[training/pretraining/objective|Training Objective]]
- [[training/pretraining/pretraining|Pretraining]]
- [[training/mid-training/continued-pretraining|Continued Pretraining]]
- [[training/post-training/grpo|GRPO]]
- [[training/post-training/rlhf|RLHF]]
