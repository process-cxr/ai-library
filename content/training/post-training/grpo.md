---
title: GRPO
created: 2026-03-07
published: 2026-03-07
modified: 2026-05-29
type: topic
status: mature
area: training
tags:
  - post-training
  - grpo
  - reasoning
---

GRPO，Group Relative Policy Optimization，是一种面向大模型 RL 后训练的 policy optimization 方法。它的核心思想是：对同一个 prompt 采样一组 responses，用组内相对 reward 估计 advantage，从而减少对单独 value model 的依赖。它在数学、代码和可验证 reasoning 任务中尤其常见，因为这些任务可以对同一问题采样多条解题路径，并用 verifier 或规则 reward 比较优劣。

GRPO 可以看作 [[training/post-training/ppo|PPO]] 在 reasoning RL 场景下的一个简化变体：仍然进行 policy optimization，仍然需要 reference / KL 约束和稳定更新，但不再依赖 token-level value function 来估计 baseline，而是使用同组样本的 reward 统计量。

## 目标与问题

PPO-RLHF 通常需要 policy model、reference model、reward model 和 value model。Value model 的作用是估计每个 state 的 expected return，用来降低 policy gradient 方差。但在 LLM reasoning 任务中，value modeling 存在几个困难：

- response 很长，reward 往往只在最终答案或测试结果处给出；
- 中间 reasoning step 的价值难以标注；
- 训练 value head 增加显存、计算和系统复杂度；
- value estimation error 会影响 advantage，进而影响 policy update；
- 对数学/代码等任务，同一 prompt 多采样本身就能提供相对比较信号。

GRPO 试图用 group baseline 替代 learned value baseline。

## 基本思想

PPO 通常需要 policy、reference policy、reward model 和 value model。GRPO 的简化点在于，它不为每个 response 训练一个显式 value function，而是把同组样本的平均 reward 当作 baseline。

给定同一 prompt 的 $G$ 个候选 $y^{(1)},\dots,y^{(G)}$，如果每个候选都有 reward $r^{(i)}$，可以构造组内相对 advantage：

$$
A^{(i)} = r^{(i)} - \frac{1}{G}\sum_{j=1}^{G}r^{(j)}
$$

这样，模型被鼓励提高组内表现更好的回答概率，降低表现较差回答的概率。

有些实现还会对组内 reward 做标准化：

$$
A^{(i)} =
\frac{r^{(i)} - \mathrm{mean}(\{r^{(j)}\}_{j=1}^{G})}
{\mathrm{std}(\{r^{(j)}\}_{j=1}^{G}) + \epsilon}
$$

标准化可以降低不同 prompt reward 尺度差异造成的不稳定，但也会让组内相对排序比绝对 reward 更重要。

## 训练流程

一个典型 GRPO 训练 batch 包括：

1. 从 prompt dataset 中采样 prompts；
2. 对每个 prompt 用当前 policy 采样 $G$ 个 responses；
3. 用 reward model、rule-based verifier、unit tests 或答案校验器给每个 response 打分；
4. 在同一 prompt 的 group 内计算 relative advantage；
5. 计算当前 policy、old policy 和 reference policy 的 token log-prob；
6. 使用 clipped policy objective 更新 policy；
7. 加入 KL penalty 或 reference regularization；
8. 监控 reward、pass rate、KL、response length、格式有效率和样本多样性。

GRPO 的关键是“同一 prompt 的多样本比较”。如果每个 prompt 只采样一个 response，就无法形成 group-relative advantage。

## Objective 直观形式

GRPO 常保留 PPO-style ratio 与 clipping。对第 $i$ 个 response 的 token，可写成近似形式：

$$
r_{i,t}(\theta)
= \frac{\pi_\theta(y^{(i)}_t \mid x, y^{(i)}_{<t})}
{\pi_{\theta_{\text{old}}}(y^{(i)}_t \mid x, y^{(i)}_{<t})}
$$

$$
L_{\text{GRPO}}
= \mathbb{E}
\left[
\min
\left(
r_{i,t}(\theta)A^{(i)},
\text{clip}(r_{i,t}(\theta),1-\epsilon,1+\epsilon)A^{(i)}
\right)
- \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})
\right]
$$

这里 $A^{(i)}$ 是 sequence-level group advantage，通常会广播到该 response 的 tokens。不同实现会在 token-level KL、sequence reward、长度处理和 advantage normalization 上有所差异。

## 为什么适合 reasoning RL

Reasoning 任务往往可以为同一问题采样多条解题路径。即使没有精确 value model，组内比较也能提供有用信号：

- 正确答案或更高 verifier 分数的 response 被增强；
- 错误路径或低分 response 被压低；
- 多样采样可以探索不同推理轨迹；
- 不需要为每个中间状态估计 value。

这也是很多 reasoning model 后训练会使用 GRPO-like 方法的原因。

更具体地说，数学和代码任务具有两个有利条件：

- **Reward 可验证**：最终答案、单元测试、格式校验、编译结果可以作为自动 reward；
- **多路径可采样**：同一问题可能有多条 reasoning path，组内比较可以奖励更可靠路径。

这使 GRPO 更接近 RLVR，Reinforcement Learning from Verifiable Rewards，而不一定依赖人类偏好 RM。

## 与 PPO 的区别

| 维度 | PPO | GRPO |
|---|---|---|
| baseline | 通常使用 value model | 使用组内平均 reward |
| 训练复杂度 | 需要 value head / value model | 相对更轻 |
| 适用场景 | 通用 RLHF / policy optimization | 多候选 reasoning、verifier reward、relative scoring |
| 风险 | value 估计误差、训练复杂 | 依赖组内样本质量和 reward 方差 |

GRPO 并不意味着不需要稳定策略更新。实践中仍常配合 clipped surrogate、KL / reference 约束、长度控制和 reward normalization。

## 与 DPO 的区别

[[training/post-training/dpo|DPO]] 是离线 preference optimization：给定 chosen/rejected pairs，直接优化 policy。GRPO 是在线或半在线 RL：当前 policy 对每个 prompt 采样一组 responses，再根据 reward/verifier 计算优势并更新。

| 维度 | DPO | GRPO |
|---|---|---|
| 数据 | 静态 preference pairs | 当前 policy group rollouts |
| Reward | 隐式偏好 | 显式 reward/verifier |
| 探索 | 受限于数据 | 可通过采样探索 |
| 工程复杂度 | 较低 | 中高 |
| 适合任务 | 偏好对齐、风格、安全 | 数学、代码、可验证 reasoning |

如果任务没有可靠 verifier，GRPO 需要 reward model，此时也会面对 reward hacking。若有可靠 verifier，GRPO 可以直接强化 correctness 或 pass rate。

## Reward 设计

GRPO 的效果高度依赖 reward。常见 reward 包括：

- final answer correctness；
- code unit test pass rate；
- format validity；
- reasoning step verifier；
- rule-based penalty，如长度、重复、非法字符；
- reward model score；
- 多目标组合 reward。

Reward 设计要避免只奖励容易投机的表面指标。例如只检查最终答案字符串，可能鼓励模型猜答案；只奖励格式，可能牺牲内容质量；只奖励长 CoT，可能导致冗长无效推理。

在 agent 场景中，GRPO-like 更新也可以结合 step-level 信号。除了 episode-level success reward，还可以对中间 world-model span 或 planning span 设计局部 reward，例如预测关键词是否在后续真实轨迹中出现、confidence 是否与最终成败校准、工具调用是否有效。这类设计试图缓解长程任务的稀疏 credit assignment，但需要避免局部 proxy 压过最终任务目标。[[sources/papers/2026-internalizing-the-future-world-model-agentic-training|Internalizing the Future]] 使用 grounding reward、Brier score calibration 和 confidence-based local advantage 作为示例。

## 失败模式与边界

- **Group quality 低**：如果同组样本全错，relative advantage 只能在错误中挑“相对较好”，学习信号弱。
- **Reward 稀疏**：数学/代码任务中 pass/fail reward 可能大多为 0，训练早期需要 warmup 或更强初始模型。
- **Reward hacking**：模型可能利用 verifier 漏洞或格式检查漏洞。
- **Mode collapse**：过度强化某类解题模板，降低探索多样性。
- **长度膨胀**：reasoning reward 若不约束长度，模型可能生成过长 CoT。
- **采样成本高**：每个 prompt 采样 $G$ 个 responses，显著增加推理计算。
- **KL 约束敏感**：KL 太弱会漂移，太强会学不动。

## 在 Pretraining 中的迁移

[[sources/papers/2025-rlp-reinforcement-as-a-pretraining-objective|RLP]] 把 GRPO-style group-relative 思路从后训练迁移到了预训练阶段。它对同一个 context 采样多条 thoughts，用每条 thought 对 next-token prediction 的 information gain 作为 reward，再用组内相对 advantage 更新 thought tokens。

这说明 GRPO 的思想不只限于 instruction-following 后训练。只要能为同一个输入构造多候选 action 和可比较 reward，就可以形成 group-relative policy update。

## 经典论文与资料

- [[sources/papers/2024-deepseekmath|DeepSeekMath]]
- [[sources/papers/2017-ppo|Proximal Policy Optimization Algorithms]]
- [[sources/papers/2025-rlp-reinforcement-as-a-pretraining-objective|RLP: Reinforcement as a Pretraining Objective]]

## 相关概念

- [[training/post-training/ppo|PPO]]
- [[training/post-training/rlhf|RLHF]]
- [[training/post-training/reward-model|Reward Model]]
- [[training/post-training/rejection-sampling|Rejection Sampling]]
- [[training/pretraining/reinforcement-pretraining|Reinforcement Pretraining]]
- [[training/pretraining/objective|Training Objective]]
