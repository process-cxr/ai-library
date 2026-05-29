---
title: DPO
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

DPO，Direct Preference Optimization，是一种直接从 pairwise preference data 优化语言模型的后训练方法。它的关键贡献是：在一定理论假设下，可以绕过显式 [[training/post-training/reward-model|Reward Model]] 训练和 [[training/post-training/ppo|PPO]] 在线强化学习，用一个 supervised-style loss 直接让 policy 更偏向 chosen response、远离 rejected response。

DPO 在工程上很重要，因为它把 RLHF pipeline 中最复杂的 online RL 部分转换成离线 preference learning。训练过程更像 SFT：给定 prompt、chosen answer、rejected answer，计算两条回答在 policy 与 reference model 下的 log-prob，并优化一个 pairwise logistic loss。

## 目标与问题

传统 [[training/post-training/rlhf|RLHF]] pipeline 包括：

1. SFT model；
2. preference data；
3. reward model；
4. PPO policy optimization；
5. KL constraint。

DPO 认为，如果 RLHF 的目标是最大化 reward 同时受 KL 约束，那么最优 policy 与 reward function 之间存在解析关系。于是可以把 reward 隐式表示为 policy 相对 reference policy 的 log-prob ratio，并直接用偏好数据训练 policy。

## 数据与输入

DPO 数据是 pairwise preference triples：

```text
x: prompt
y_w: chosen response
y_l: rejected response
```

训练时需要两个模型：

- 当前 policy $\pi_\theta$，需要更新；
- reference policy $\pi_{\text{ref}}$，通常是 SFT model，冻结。

DPO 的质量高度依赖 pairwise 数据。如果 chosen 和 rejected 差异过大，模型只学到粗糙过滤；如果偏好标签噪声大，模型会学习错误方向；如果 rejected 是低质量采样，优化效果会受限。

## 从 Bradley-Terry 到 DPO

Reward model 常使用 Bradley-Terry 形式：

$$
P(y_w \succ y_l \mid x)
= \sigma(r(x,y_w) - r(x,y_l))
$$

KL-regularized RLHF 的最优 policy 满足：

$$
r(x,y)
= \beta \log \frac{\pi^*(y\mid x)}
{\pi_{\text{ref}}(y\mid x)}
+ \beta \log Z(x)
$$

其中 $Z(x)$ 是只依赖 prompt 的 partition function。在 pairwise reward difference 中，$Z(x)$ 会抵消。因此可以用 policy log-ratio 表示 reward difference：

$$
r(x,y_w)-r(x,y_l)
= \beta
\left[
\log \frac{\pi_\theta(y_w\mid x)}
{\pi_{\text{ref}}(y_w\mid x)}
-
\log \frac{\pi_\theta(y_l\mid x)}
{\pi_{\text{ref}}(y_l\mid x)}
\right]
$$

DPO loss 为：

$$
\mathcal{L}_{\text{DPO}}(\theta)
= - \mathbb{E}_{(x,y_w,y_l)}
\left[
\log \sigma
\left(
\beta
\left[
\log \frac{\pi_\theta(y_w\mid x)}
{\pi_{\text{ref}}(y_w\mid x)}
-
\log \frac{\pi_\theta(y_l\mid x)}
{\pi_{\text{ref}}(y_l\mid x)}
\right]
\right)
\right]
$$

直观上，DPO 鼓励 chosen response 相对 reference 的概率提升更多，rejected response 相对 reference 的概率提升更少或下降。

## Beta 的作用

$\beta$ 是 DPO 中连接 preference loss 与 KL-regularized RL 推导的尺度参数。若从

$$
\max_\pi \mathbb{E}_{y\sim \pi(\cdot\mid x)}[r(x,y)]
- \beta \mathrm{KL}(\pi(\cdot\mid x)\|\pi_{\text{ref}}(\cdot\mid x))
$$

这一形式理解，$\beta$ 对应 reference policy 约束的强度；更大的 $\beta$ 会使最优策略更接近 $\pi_{\text{ref}}$，更小的 $\beta$ 允许策略为了偏好回报产生更大的 log-prob ratio 偏移。

在 DPO loss 中，$\beta$ 也会缩放 chosen/rejected log-ratio difference，因此它同时影响梯度尺度和 loss 饱和速度。实践上不能把它简单理解为“越大更新越强”或“越小越保守”，而应同时观察 KL、胜率、长度分布和泛化评估。

经验上：

- $\beta$ 过大时，reference constraint 更强，policy 可能较难充分吸收偏好数据；
- $\beta$ 过小时，policy 可能更容易偏离 reference，出现格式漂移、长度偏移或过度优化；
- 实际效果还取决于 learning rate、response 长度、log-prob 归一化方式、数据噪声和 reference model 质量。

## 与 RLHF / PPO 对比

| 维度 | RLHF + PPO | DPO |
|---|---|---|
| 是否训练显式 RM | 通常需要 | 不需要 |
| 是否在线 rollout | 需要 | 不需要 |
| 训练形态 | 强化学习 | 离线监督式偏好优化 |
| 工程复杂度 | 高 | 较低 |
| 探索能力 | 可以基于当前 policy 采样 | 受限于静态 preference data |
| 目标灵活性 | 可使用复杂 reward/verifier | 主要依赖 pairwise preference |
| 稳定性 | 需要仔细调 PPO/KL | 通常更稳定 |

DPO 的优势是简单、稳定、成本低；局限是不能自动探索新策略分布。如果偏好数据没有覆盖真实错误，DPO 很难凭空修正。

## 与 SFT 的区别

SFT 只最大化 chosen response：

$$
\mathcal{L}_{\text{SFT}}
= -\log \pi_\theta(y_w\mid x)
$$

DPO 同时比较 chosen 和 rejected，并考虑 reference model：

- chosen 好，不只是因为它在数据里出现；
- rejected 坏，不只是因为它没出现；
- policy 更新要相对 reference 进行，避免无限提高所有训练 response 的概率。

因此 DPO 更接近 preference learning，而不是单纯 imitation。

## 实践细节

### Sequence Log-Probability

DPO 需要计算完整 response 的 log-prob：

$$
\log \pi_\theta(y\mid x)
= \sum_{t=1}^{T}
\log \pi_\theta(y_t\mid x,y_{<t})
$$

有些实现会使用长度归一化，避免长回答天然 log-prob 更低。是否归一化会改变训练偏好，需要与数据和目标一致。

### Reference Model

Reference model 通常冻结为 SFT checkpoint。它的作用是提供“不要偏离太远”的基准。如果 reference 很弱，DPO 的 implicit reward 也会受影响；如果 reference 已经过度安全或过度冗长，DPO 可能继承这些倾向。

### Negative Quality

Rejected response 的质量决定学习信号。高质量 hard negatives 能教模型细微偏好；低质量 negatives 只能教模型避开明显错误。

## 失败模式与边界

- **数据噪声**：偏好标签错误会直接推动 policy 错方向。
- **过度保守**：reference constraint 太强、$\beta$ 过大或训练步数不足，偏好提升有限。
- **过度优化**：$\beta$ 过小、学习率过高或训练太久，导致分布漂移、格式僵化。
- **长度偏差**：chosen response 更长时，模型可能学到 verbosity。
- **静态数据限制**：DPO 不会像 RLHF rollout 那样发现当前 policy 的新错误。
- **多目标混淆**：helpfulness、harmlessness、truthfulness 混在一个 preference 标签里，可能产生不透明权衡。

## 经典论文与资料

- [[sources/papers/2023-dpo|Direct Preference Optimization]]
- [[sources/papers/2022-instructgpt|Training language models to follow instructions with human feedback]]
- [[sources/papers/2017-deep-rl-from-human-preferences|Deep Reinforcement Learning from Human Preferences]]

## 相关概念

- [[training/post-training/rlhf|RLHF]]
- [[training/post-training/reward-model|Reward Model]]
- [[training/post-training/sft|SFT]]
- [[training/post-training/rejection-sampling|Rejection Sampling]]
- [[fundamentals/information-theory/kl-divergence|KL Divergence]]
