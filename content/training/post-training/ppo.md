---
title: PPO
created: 2026-03-07
published: 2026-03-07
modified: 2026-05-29
type: topic
status: mature
area: training
tags:
  - post-training
  - rlhf
  - ppo
---

PPO，Proximal Policy Optimization，是一种 policy gradient 强化学习算法。在大模型后训练中，PPO 最著名的用途是 [[training/post-training/rlhf|RLHF]]：语言模型作为 policy 生成回答，[[training/post-training/reward-model|Reward Model]] 提供 reward，PPO 在限制策略更新幅度的同时提高 reward。

PPO 的核心思想是“不要让新策略离旧策略太远”。这对语言模型尤其重要，因为 token 分布一旦被过度更新，模型可能迅速出现重复、乱码、过度迎合 reward、拒答泛化或语言能力退化。

## 目标与问题

普通 policy gradient 可以用 reward 更新 policy：

$$
\nabla_\theta J(\theta)
= \mathbb{E}
\left[
\nabla_\theta \log \pi_\theta(a\mid s) A(s,a)
\right]
$$

但在大模型生成中，action 是 token，trajectory 是完整 response，动作空间巨大，reward 稀疏且 noisy。一次过大的策略更新可能让模型远离原有语言分布。PPO 通过 clipped surrogate objective 控制更新幅度，使训练更稳定。

## RLHF 中的对象映射

| RL 概念 | 语言模型后训练中的对应物 |
|---|---|
| state | prompt + 已生成 tokens |
| action | 下一个 token |
| policy | 当前语言模型 $\pi_\theta$ |
| old policy | rollout 时冻结的旧模型 $\pi_{\theta_{\text{old}}}$ |
| reference policy | SFT/reference model $\pi_{\text{ref}}$ |
| reward | RM score、verifier score、人类偏好代理 |
| trajectory | 完整 generated response |
| advantage | response/token 相对于 baseline 的优势估计 |

PPO 不是专门为语言模型发明的算法，但它的保守更新特性使其适合 RLHF 早期系统。

## Clipped Objective

PPO 使用新旧策略概率比：

$$
r_t(\theta)
= \frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}
$$

核心 clipped objective 为：

$$
L^{\text{CLIP}}(\theta)
= \mathbb{E}_t
\left[
\min
\left(
r_t(\theta) A_t,
\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t
\right)
\right]
$$

直观理解：

- 如果某个 token/action 的 advantage 为正，PPO 鼓励提高它的概率，但超过 $1+\epsilon$ 后收益被截断；
- 如果 advantage 为负，PPO 鼓励降低它的概率，但超过 $1-\epsilon$ 后也被限制；
- clipping 让单步更新更保守，避免 policy collapse。

## Reward 与 KL Penalty

RLHF 中的 reward 通常不是单纯 RM score，而是：

$$
R(x,y)
= r_\phi(x,y)
- \beta \sum_t
\left(
\log \pi_\theta(y_t\mid x,y_{<t})
- \log \pi_{\text{ref}}(y_t\mid x,y_{<t})
\right)
$$

第二项是对 reference model 的 token-level KL penalty。它限制 policy 为了提高 RM 分数而偏离 SFT model。PPO 的 clipping 控制“相对于 old policy 的单次更新”，KL penalty 控制“相对于 reference policy 的整体偏移”。两者解决的问题不同，实践中常同时使用。

## Value Model 与 Advantage

PPO 需要 advantage $A_t$。在 RLHF 中，常训练一个 value head 估计当前 state 的 expected return：

$$
A_t = R_t - V_\psi(s_t)
$$

value model 充当 baseline，降低 policy gradient 方差。问题是，语言模型 response 的 reward 往往在序列末端才出现，如何把 sequence-level reward 分配到 token-level advantage 并不天然清晰。实践中会使用 generalized advantage estimation、reward shaping、KL token penalty 等技巧。

## 训练流程

典型 PPO-RLHF 流程：

1. 从 prompt dataset 采样 batch；
2. 使用当前 policy 生成 responses；
3. 计算每个 token 的 policy log-prob 和 reference log-prob；
4. 使用 reward model / verifier 给完整 response 打分；
5. 加入 KL penalty 得到 token 或 sequence reward；
6. 用 value model 估计 returns 和 advantages；
7. 多个 mini-batch epoch 更新 policy 和 value；
8. 监控 reward、KL、entropy、response length、clip fraction、人工偏好。

这里 rollout 与 update 的分离很重要：rollout 阶段生成的数据来自 old policy，更新阶段用 PPO ratio 评估 new policy 相对 old policy 的变化。

## 关键超参

- **clip range $\epsilon$**：越小越保守，越大更新越激进。
- **KL coefficient $\beta$**：越大越接近 reference，越小越容易 reward hacking。
- **learning rate**：过大导致 policy collapse，过小则 reward 改进慢。
- **rollout batch size**：影响 reward/advantage 估计方差。
- **PPO epochs**：同一批 rollout 重复优化次数过多会过拟合旧样本。
- **value loss coefficient**：平衡 policy 与 value 学习。
- **entropy bonus**：鼓励探索，但语言模型中过强可能降低质量。
- **generation temperature / max length**：影响 rollout 分布和 reward 尺度。

## 训练稳定性指标

PPO-RLHF 需要持续监控：

- mean reward 与 reward model score；
- KL to reference；
- policy entropy；
- response length；
- clip fraction；
- value loss 与 explained variance；
- ratio 分布；
- reward 与人工偏好是否一致；
- benchmark regression；
- unsafe / over-refusal 行为。

只看 reward 上升是不够的。reward 上升同时 KL 暴涨、长度变长、人工偏好下降，通常意味着 reward hacking。

## 为什么后来出现 DPO / GRPO

PPO 强大但工程复杂。它需要 rollout、reward model、value model、reference model、KL controller 和稳定分布式训练系统。[[training/post-training/dpo|DPO]] 用静态 preference pairs 直接优化 policy，减少 RL 工程复杂度。[[training/post-training/grpo|GRPO]] 在 reasoning RL 中去掉显式 value model，用组内相对 reward 估计 advantage。

这些方法并不是简单取代 PPO，而是在不同约束下重新权衡：

- PPO 适合在线优化和复杂 reward；
- DPO 适合离线偏好数据和稳定训练；
- GRPO 适合多候选、可验证 reward 和 reasoning 任务。

## 失败模式

- **Policy collapse**：更新过大导致语言质量退化或重复。
- **Reward hacking**：模型利用 RM 偏差而不真正变好。
- **KL runaway**：KL 快速升高，policy 脱离 reference。
- **Value overfitting**：value model 对 rollout 数据过拟合，advantage 失真。
- **Length exploitation**：更长回答获得更高 reward。
- **Distribution drift**：policy 进入 RM 未覆盖区域。
- **Infrastructure complexity**：rollout 与训练吞吐、显存、同步和 log-prob 计算都很重。

## 经典论文与资料

- [[sources/papers/2017-ppo|Proximal Policy Optimization Algorithms]]
- [[sources/papers/2022-instructgpt|Training language models to follow instructions with human feedback]]
- [[sources/papers/2020-learning-to-summarize-from-human-feedback|Learning to summarize from human feedback]]

## 相关概念

- [[training/post-training/rlhf|RLHF]]
- [[training/post-training/reward-model|Reward Model]]
- [[training/post-training/dpo|DPO]]
- [[training/post-training/grpo|GRPO]]
- [[fundamentals/information-theory/kl-divergence|KL Divergence]]
