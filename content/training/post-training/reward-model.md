---
title: Reward Model
created: 2026-03-01
published: 2026-03-01
modified: 2026-05-29
type: topic
status: mature
area: training
tags:
  - training
  - post-training
  - alignment
---

Reward Model，简称 RM，是后训练中把“哪个回答更好”转换为可优化数值信号的模型。它通常接收 prompt 和 response，输出一个 scalar reward，用于 [[training/post-training/rlhf|RLHF]] policy optimization、[[training/post-training/rejection-sampling|Rejection Sampling]] 筛选、best-of-N inference 或数据质量过滤。

Reward Model 的核心价值在于：很多 assistant 行为无法用单一参考答案监督。开放问答、写作、总结、安全拒答、复杂推理和多轮对话都可能存在多个合理回答。RM 不要求给出唯一标准答案，而是学习在候选回答之间排序。

## 目标与问题

SFT 学习的是“训练集中这个 prompt 的参考答案是什么”。Reward modeling 学习的是“给定同一个 prompt，哪个 response 更符合偏好”。这使它能够处理：

- response 不唯一的开放任务；
- 长回答的整体质量判断；
- helpfulness、harmlessness、honesty 等难以 token-level 标注的目标；
- 多个候选之间的细微偏好；
- 用 scalar reward 驱动 RL 或筛选。

但 RM 不是“真实价值函数”。它只是从有限偏好数据中拟合出来的代理目标，容易继承标注偏差并被 policy exploitation。

## 偏好数据

Reward model 常用 pairwise preference data：

```text
prompt x
chosen response y_w
rejected response y_l
```

数据来源包括：

- 人类标注者比较两个或多个模型回答；
- 专家标注，如代码、数学、医学场景；
- AI feedback / RLAIF；
- rule-based verifier，如单元测试、数学答案校验；
- 多候选采样后人工或模型排序；
- 线上用户反馈。

高质量偏好数据要尽量保证候选具有可比性。如果一个候选明显乱码、另一个候选正常，RM 学到的是低层质量过滤；如果候选都较强，RM 才能学习更细粒度的 helpfulness、truthfulness 和 reasoning quality。

## Bradley-Terry 偏好模型

经典 RM 训练常基于 Bradley-Terry 模型。设 reward model 给 prompt-response pair 打分 $r_\phi(x,y)$，则 chosen response 优于 rejected response 的概率为：

$$
P(y_w \succ y_l \mid x)
= \sigma(r_\phi(x,y_w) - r_\phi(x,y_l))
$$

其中 $\sigma$ 是 sigmoid。训练损失为：

$$
\mathcal{L}_{\text{RM}}(\phi)
= - \mathbb{E}_{(x,y_w,y_l)}
\left[\log \sigma(r_\phi(x,y_w) - r_\phi(x,y_l))\right]
$$

这个目标只关心 reward 差值，不关心 reward 绝对值。因此 RM 的标定和尺度会影响后续 RL，需要 reward normalization、KL penalty 或 clipping 等稳定手段。

## ORM 与 PRM

在 reasoning 任务中，reward model 常分为 Outcome Reward Model 和 Process Reward Model。

| 类型 | 打分对象 | 优势 | 局限 |
|---|---|---|---|
| ORM | 完整答案或最终结果 | 标注简单，适合最终正确性 | 很难指出哪一步推理错了 |
| PRM | 中间步骤或 reasoning process | 提供细粒度反馈，适合复杂推理 | 标注成本高，步骤边界和正确性难定义 |

ORM 适合数学答案、代码测试、选择题等最终结果可验证任务。PRM 更适合长推理链、证明、复杂规划等场景，因为最终答案对了不代表过程可靠，最终答案错了也不说明每一步都错。

## Reward Model 在 Pipeline 中的位置

Reward model 可以用于多个阶段：

1. **Preference learning**：从 chosen/rejected pairs 学习 reward。
2. **RLHF**：作为 policy rollout 的 reward function。
3. **Rejection sampling**：从多个候选中选择高 reward response。
4. **Data filtering**：过滤低质量 SFT / synthetic data。
5. **Evaluation**：作为自动评估指标的一部分，但需要谨慎。
6. **DPO 数据生产**：辅助构造或筛选 preference pairs。

同一个 RM 不一定适合所有用途。用于 RL 的 RM 会被 policy 强烈优化，因此需要更强鲁棒性；用于粗筛数据的 RM 可以更便宜但误差更大。

## Reward Hacking

Reward hacking 指 policy 找到提高 RM 分数但不真正提高人类偏好的行为。例如：

- 输出更长、更自信、更模板化的回答；
- 迎合 reward model 的关键词；
- 生成看似严谨但事实错误的解释；
- 在安全任务中过度拒答以获得“安全”高分；
- 利用 verifier 或测试集漏洞。

Reward hacking 的根因是代理目标不完备。RM 只拟合训练偏好分布，而 policy optimization 会主动搜索 RM 的弱点。优化越强、KL 约束越弱、RM 覆盖越窄，reward hacking 风险越高。

## 标定与泛化

Reward model 的分数尺度并没有天然意义。两个 RM 的 reward 不能直接比较，同一个 RM 在不同 prompt 分布上也可能尺度不同。实践中常见处理包括：

- 对 reward 做均值方差归一化；
- 限制 response 长度或加入 length penalty；
- 在 RL 中加入 reference KL；
- 定期用 held-out human preference 验证；
- 使用 ensemble 或多目标 reward；
- 对 adversarial / out-of-distribution prompt 做鲁棒性评估。

## 与 DPO 的关系

[[training/post-training/dpo|DPO]] 可以看作把 reward modeling 与 policy optimization 合并。DPO 不显式训练 RM，而是直接用 chosen/rejected pairs 更新 policy。但从理论上看，它仍然借用了 Bradley-Terry preference model，并把隐式 reward 写成 policy 与 reference policy 的 log-prob ratio。

因此，理解 reward model 是理解 DPO、RLHF 和很多 preference optimization 方法的基础。

## 失败模式与边界

- **标注者偏差**：RM 会学习标注群体的偏好，而不是抽象的“正确价值”。
- **候选分布窄**：如果训练候选都来自同一模型，RM 泛化到新 policy rollout 时可能失效。
- **长度偏差**：长答案常被偏好，导致模型变啰嗦。
- **形式偏差**：结构化、分点、自信语气可能被误判为质量。
- **不可验证事实**：RM 很难可靠判断事实真伪，尤其是长文本。
- **过优化**：policy 多轮更新后进入 RM 未覆盖分布。

## 经典论文与资料

- [[sources/papers/2017-deep-rl-from-human-preferences|Deep Reinforcement Learning from Human Preferences]]
- [[sources/papers/2020-learning-to-summarize-from-human-feedback|Learning to summarize from human feedback]]
- [[sources/papers/2022-instructgpt|Training language models to follow instructions with human feedback]]
- [[sources/papers/2022-constitutional-ai|Constitutional AI]]

## 相关概念

- [[training/post-training/rlhf|RLHF]]
- [[training/post-training/ppo|PPO]]
- [[training/post-training/dpo|DPO]]
- [[training/post-training/rejection-sampling|Rejection Sampling]]
- [[fundamentals/information-theory/kl-divergence|KL Divergence]]
