---
title: RLHF
created: 2026-03-07
published: 2026-03-07
modified: 2026-05-29
type: topic
status: mature
area: training
tags:
  - training
  - post-training
  - alignment
---

RLHF，Reinforcement Learning from Human Feedback，是用人类偏好数据训练 reward signal，并用强化学习优化语言模型策略的后训练方法。它的目标不是让模型复现单个参考答案，而是在开放生成空间中提高人类更偏好的回答概率。

[[sources/papers/2022-instructgpt|Training language models to follow instructions with human feedback]] 是将这条路线系统应用到 GPT-3 的代表性工作：先用 demonstrations 做 SFT，再用 response rankings 训练 reward model，最后用 PPO 优化 policy，并用 pretraining mix 缓解能力回退。这里的三阶段分别承担行为初始化、偏好建模和 reward-driven policy update，不应混成一个统一的监督目标。

这条路线的早期系统化代表是 [[sources/papers/2017-deep-rl-from-human-preferences|Deep Reinforcement Learning from Human Preferences]]：人类比较两个短 trajectory segments，reward predictor 从 pairwise preferences 中学习，policy 再在持续更新的 predicted reward 上进行 RL。现代 LLM RLHF 将视觉行为片段换成 prompt-response，但仍然保留 preference data、reward model、policy rollout 和在线更新之间的基本闭环。

在现代 assistant 训练中，RLHF 通常指一条 pipeline：先进行 [[training/post-training/sft|SFT]] 得到可用初始 policy，再训练 [[training/post-training/reward-model|Reward Model]]，最后用 [[training/post-training/ppo|PPO]] 或其他 policy optimization 方法更新模型，同时用 KL penalty 限制模型偏离 reference model。

## 目标与问题

语言模型预训练和 SFT 都主要依赖 maximum likelihood。这个目标适合学习数据分布，却无法充分表达“回答 A 比回答 B 更好”：

- 一个 prompt 可以有多个正确回答；
- 标准答案未必比模型生成的其他答案更好；
- 人类偏好涉及帮助性、真实性、安全性、简洁性、语气和上下文适配；
- 开放任务很难构造唯一 label；
- SFT 不能直接惩罚“看似合理但用户不喜欢”的候选。

RLHF 将问题改写为：在给定 prompt 分布下，训练一个 policy $\pi_\theta(y\mid x)$，使生成回答获得更高 reward，同时不要偏离原始语言模型太远。

## InstructGPT Pipeline

经典 InstructGPT 流程包括三步：

1. **Supervised Fine-Tuning**：用人工示范数据训练 SFT model，使 base model 初步学会按照 instruction 回答。
2. **Reward Model Training**：对同一 prompt 采样多个模型回答，由人类标注排序或偏好，训练 reward model。
3. **PPO Policy Optimization**：用 reward model 对 policy rollout 打分，再通过 PPO 更新 policy，并加入 KL penalty 约束 policy 不要过度偏离 SFT reference model。

这条路线的关键思想是分离“示范”和“偏好”：示范数据告诉模型可接受回答长什么样；偏好数据告诉模型多个可接受回答之间谁更好。

## 训练目标

RLHF 常见目标可以写成：

$$
\max_\theta \mathbb{E}_{x\sim \mathcal{D}, y\sim \pi_\theta(\cdot\mid x)}
\left[
r_\phi(x,y)
- \beta \mathrm{KL}(\pi_\theta(\cdot\mid x)\|\pi_{\text{ref}}(\cdot\mid x))
\right]
$$

其中：

- $\pi_\theta$ 是当前 policy；
- $\pi_{\text{ref}}$ 是 reference model，通常是 SFT model；
- $r_\phi$ 是 reward model；
- $\beta$ 控制 reward maximization 与 distribution constraint 的平衡。

KL penalty 的作用是防止 policy 为了追求 RM 分数而偏离语言质量、事实性和安全边界。它可以看作 RLHF 中的正则化项。

## PPO 在 RLHF 中的角色

[[training/post-training/ppo|PPO]] 通过 clipped objective 限制单次 policy update 幅度，提高训练稳定性。语言模型输出空间巨大，response 是长 token 序列，reward 通常只在序列末端给出，因此直接 policy gradient 容易高方差且不稳定。PPO 引入：

- old policy 与 new policy 的 probability ratio；
- advantage estimation；
- clipping；
- value function 或 baseline；
- KL penalty / KL monitoring。

这些机制让 policy 能逐步提高 reward，而不是一次更新就崩坏。

## 数据与采样

RLHF 不只依赖静态数据，还依赖 rollout：

1. 从 prompt dataset 采样 prompts；
2. 当前 policy 生成 responses；
3. reward model 给 responses 打分；
4. 计算 KL penalty 和 advantage；
5. 使用 PPO 更新 policy；
6. 定期评估人工偏好、安全性、事实性和 benchmark。

Prompt distribution 很重要。如果 RL prompts 过窄，policy 会只在这些任务上优化；如果 prompts 与真实用户请求差异大，线上行为不会稳定提升。

## KL Penalty

KL penalty 是 RLHF 稳定性的核心。它限制当前 policy 与 reference model 的差异：

$$
\mathrm{KL}(\pi_\theta \| \pi_{\text{ref}})
= \mathbb{E}_{y\sim\pi_\theta}
\left[
\log \pi_\theta(y\mid x) - \log \pi_{\text{ref}}(y\mid x)
\right]
$$

实践中常用 token-level KL 近似，把每个生成 token 的 log-prob 差作为惩罚。KL 太弱会导致 reward hacking；KL 太强会让 policy 几乎无法改进。很多系统会动态调整 $\beta$，使实际 KL 接近目标区间。

## RLHF 的优势

- 能优化开放式偏好，而不是只模仿参考答案；
- 可以把人类排序信号放大为可训练 reward；
- 能处理多个候选之间的整体质量差异；
- 对写作、总结、对话、安全和复杂偏好任务很有效；
- 可以通过 online rollout 看到当前 policy 的真实错误分布。

## RLHF 的局限

### 训练复杂

RLHF 涉及 SFT model、reward model、reference model、policy model、rollout engine、PPO trainer、KL controller 和评估系统，工程成本远高于纯 SFT 或 DPO。

### Reward Hacking

Policy 会主动寻找 RM 漏洞。越强的优化越容易暴露 RM 的偏差，例如冗长、过度自信、模板化、安全过度保守等。

### 标注成本高

人类偏好标注需要一致规范、标注者培训和质量控制。复杂任务还需要专家标注，否则偏好信号噪声很大。

### 可解释性有限

Reward 是 scalar，不能总是指出回答哪里好或哪里坏。对于 reasoning 任务，ORM 只评价最终答案时尤其如此。

### 评估困难

RLHF 提升可能体现在用户偏好而不是标准 benchmark 上。自动评估与人工评估需要结合。

## 与 DPO / GRPO 的关系

[[training/post-training/dpo|DPO]] 试图保留 preference optimization 的核心收益，同时去掉显式 reward model 和 PPO rollout。它更简单、稳定、便宜，但不能像 RLHF 那样持续基于当前 policy 进行 online exploration。

[[training/post-training/grpo|GRPO]] 常用于可验证 reasoning RL。它保留 policy optimization 与 rollout，但用组内相对 reward 构造 advantage，减少 value model 依赖。对于数学和代码任务，reward 可以来自 verifier，而不一定来自人类偏好 RM。

## 经典论文与资料

- [[sources/papers/2017-deep-rl-from-human-preferences|Deep Reinforcement Learning from Human Preferences]]
- [[sources/papers/2020-learning-to-summarize-from-human-feedback|Learning to summarize from human feedback]]
- [[sources/papers/2022-instructgpt|Training language models to follow instructions with human feedback]]
- [[sources/papers/2022-constitutional-ai|Constitutional AI]]
- [[sources/papers/2017-ppo|Proximal Policy Optimization Algorithms]]

## 相关概念

- [[training/post-training/sft|SFT]]
- [[training/post-training/reward-model|Reward Model]]
- [[training/post-training/ppo|PPO]]
- [[training/post-training/dpo|DPO]]
- [[training/post-training/grpo|GRPO]]
- [[fundamentals/information-theory/kl-divergence|KL Divergence]]
