---
title: GRPO
created: 2026-03-07
published: 2026-03-07
modified: 2026-05-28
type: topic
status: growing
area: training
tags:
  - post-training
  - grpo
  - reasoning
---

GRPO，Group Relative Policy Optimization，是一种面向大模型 RL 训练的 policy optimization 思路。它的核心是：对同一个 prompt 采样一组 responses，用组内相对分数估计 advantage，从而减少对单独 value model 的依赖。

## 基本思想

PPO 通常需要 policy、reference policy、reward model 和 value model。GRPO 的简化点在于，它不为每个 response 训练一个显式 value function，而是把同组样本的平均 reward 当作 baseline。

给定同一 prompt 的 $G$ 个候选 $y^{(1)},\dots,y^{(G)}$，如果每个候选都有 reward $r^{(i)}$，可以构造组内相对 advantage：

$$
A^{(i)} = r^{(i)} - \frac{1}{G}\sum_{j=1}^{G}r^{(j)}
$$

这样，模型被鼓励提高组内表现更好的回答概率，降低表现较差回答的概率。

## 为什么适合 reasoning RL

Reasoning 任务往往可以为同一问题采样多条解题路径。即使没有精确 value model，组内比较也能提供有用信号：

- 正确答案或更高 verifier 分数的 response 被增强；
- 错误路径或低分 response 被压低；
- 多样采样可以探索不同推理轨迹；
- 不需要为每个中间状态估计 value。

这也是很多 reasoning model 后训练会使用 GRPO-like 方法的原因。

## 与 PPO 的区别

| 维度 | PPO | GRPO |
|---|---|---|
| baseline | 通常使用 value model | 使用组内平均 reward |
| 训练复杂度 | 需要 value head / value model | 相对更轻 |
| 适用场景 | 通用 RLHF / policy optimization | 多候选 reasoning、verifier reward、relative scoring |
| 风险 | value 估计误差、训练复杂 | 依赖组内样本质量和 reward 方差 |

GRPO 并不意味着不需要稳定策略更新。实践中仍常配合 clipped surrogate、KL / reference 约束、长度控制和 reward normalization。

## 在 Pretraining 中的迁移

[[sources/papers/2025-rlp-reinforcement-as-a-pretraining-objective|RLP]] 把 GRPO-style group-relative 思路从后训练迁移到了预训练阶段。它对同一个 context 采样多条 thoughts，用每条 thought 对 next-token prediction 的 information gain 作为 reward，再用组内相对 advantage 更新 thought tokens。

这说明 GRPO 的思想不只限于 instruction-following 后训练。只要能为同一个输入构造多候选 action 和可比较 reward，就可以形成 group-relative policy update。

## 相关概念

- [[training/post-training/ppo|PPO]]
- [[training/post-training/rlhf|RLHF]]
- [[training/pretraining/reinforcement-pretraining|Reinforcement Pretraining]]
- [[training/pretraining/objective|Training Objective]]
