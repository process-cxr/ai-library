---
title: PPO
created: 2026-03-07
published: 2026-03-07
modified: 2026-07-14
type: topic
status: mature
area: training
tags:
  - post-training
  - rlhf
  - ppo
---

PPO，Proximal Policy Optimization，是一种 policy gradient 强化学习算法。它的核心目标是：**在提高 reward 的同时，限制新 policy 相对旧 policy 的单次更新幅度**。在大模型后训练中，PPO 最典型的用途是 [[training/post-training/rlhf|RLHF]]：语言模型作为 policy 生成回答，[[training/post-training/reward-model|Reward Model]] 或 verifier 提供 reward，PPO 根据 reward 更新 policy，并通过 clipping、KL penalty、value baseline 等机制控制训练稳定性。

PPO 的直觉可以先压缩成一句话：

```text
如果某个 token/action 比 baseline 更好，就提高它的概率；
如果更差，就降低它的概率；
但每次更新都不能离 rollout 时的旧策略太远。
```

这对语言模型很重要。LLM 的 action space 是整个 vocabulary，trajectory 是一串 tokens，reward 往往稀疏、延迟且带噪声。如果直接用普通 policy gradient 大幅更新，模型可能迅速出现重复、乱码、长度投机、reward hacking、KL runaway 或语言能力退化。PPO 的 clipping 机制就是为了让 policy update 更保守。

## 基本问题

强化学习希望最大化 policy 的期望回报：

$$
J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)]
$$

其中 $\tau$ 是 trajectory，$R(\tau)$ 是整条轨迹的回报。policy gradient 的基本形式是：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}
\left[
\nabla_\theta \log \pi_\theta(a_t\mid s_t) A_t
\right]
$$

这里的 $A_t$ 是 advantage，表示在状态 $s_t$ 下采取动作 $a_t$ 比当前平均水平好多少。

如果 $A_t>0$，应该提高 $a_t$ 的概率；如果 $A_t<0$，应该降低 $a_t$ 的概率。问题在于，普通 policy gradient 没有天然限制更新幅度。一次训练 step 可能把某些 token 概率推得过高或过低，导致 policy 跳出原本可控的语言分布。

PPO 解决的问题不是“如何定义 reward”，而是：**给定一批由旧 policy 采样出来的数据，如何比较稳定地更新当前 policy。**

## LLM 中的 RL 对象映射

在语言模型后训练里，RL 对象可以这样对应：

| RL 概念                                | LLM 后训练中的对应物                                         |
| -------------------------------------- | ------------------------------------------------------------ |
| state $s_t$                            | prompt + 已生成前缀 tokens                                   |
| action $a_t$                           | 下一个 token                                                 |
| policy $\pi_\theta$                    | 当前训练中的语言模型                                         |
| old policy $\pi_{\theta_{\text{old}}}$ | rollout 时用于生成 response 的冻结 policy                    |
| reference policy $\pi_{\text{ref}}$    | SFT model 或 base reference model                            |
| reward                                 | RM score、verifier score、规则 reward、人类偏好代理          |
| trajectory                             | prompt 后生成的一整段 response 或 multi-turn action sequence |
| return $R_t$                           | 从 token $t$ 之后累计得到的回报                              |
| value $V(s_t)$                         | 当前状态未来期望回报的估计                                   |
| advantage $A_t$                        | 当前 token/action 相对 baseline 的优势                       |

这里有两个容易混淆的 policy：

- **old policy** 用于 PPO ratio，控制当前 policy 相对 rollout policy 的单次更新幅度；
- **reference policy** 用于 KL penalty，控制当前 policy 相对 SFT/base reference 的总体偏移。

二者解决的问题不同。old policy 是“这批数据由谁生成”；reference policy 是“模型整体不要偏离谁太远”。

## Rollout 与 Update

PPO 训练天然分成两个阶段：

```text
rollout phase:
  用当前 policy 生成 response
  保存 tokens、attention mask、old logprobs、value estimates 等
  用 reward model / verifier 计算 reward

update phase:
  固定 rollout 数据
  用当前 policy 重新计算 token logprobs
  计算 ratio、advantage、policy loss、value loss
  对同一批 rollout 做一个或多个 PPO epochs
```

这就是为什么 PPO 需要 old logprobs。rollout 时模型生成 token $a_t$ 的概率是 $\pi_{\theta_{\text{old}}}(a_t\mid s_t)$；更新时当前模型给同一个 token 的概率是 $\pi_\theta(a_t\mid s_t)$。两者的比值衡量当前 policy 相对旧 policy 改了多少。

如果没有 old logprobs，就无法计算 PPO ratio，也就失去了 PPO clipping 的基础。

## Probability Ratio

PPO 的核心变量是新旧策略概率比：

$$
r_t(\theta)
=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}
$$

在 LLM 中通常用 logprob 计算：

$$
r_t(\theta)
=
\exp
\left(
\log\pi_\theta(a_t\mid s_t)
-
\log\pi_{\theta_{\text{old}}}(a_t\mid s_t)
\right)
$$

如果 $r_t=1$，说明当前 policy 对这个 token 的概率和 rollout 时一样；如果 $r_t>1$，说明当前 policy 提高了该 token 的概率；如果 $r_t<1$，说明降低了概率。

PPO 不希望 $r_t$ 偏离 1 太多。因为 rollout 数据是旧 policy 生成的，一旦当前 policy 距离旧 policy 太远，用旧数据估计当前 policy 的梯度就会变得不可靠。

## Clipped Surrogate Objective

PPO 的 clipped objective 是：

$$
L^{\text{CLIP}}(\theta)
=
\mathbb{E}_t
\left[
\min
\left(
r_t(\theta)A_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t
\right)
\right]
$$

其中 $\epsilon$ 是 clip range，常见值如 0.1 或 0.2。

直观解释要分 advantage 正负来看。

当 $A_t>0$ 时，这个 token/action 比 baseline 好，训练希望提高它的概率，即让 $r_t$ 变大。但如果 $r_t$ 已经超过 $1+\epsilon$，继续增大不再增加目标函数收益。这样可以防止模型对正 reward token 过度追涨。

当 $A_t<0$ 时，这个 token/action 比 baseline 差，训练希望降低它的概率，即让 $r_t$ 变小。但如果 $r_t$ 已经低于 $1-\epsilon$，继续降低不再带来更多收益。这样可以防止模型对负 reward token 过度打压。

所以 PPO clipping 不是简单把梯度裁掉，而是在 objective 层面限制“从这批 rollout 中获得的收益”。它鼓励 policy 朝 advantage 指向的方向移动，但只允许移动到一个 trust region 附近。

## Reward 与 KL Penalty

在 RLHF 中，reward 通常不是单纯的 reward model 分数。常见形式是：

$$
R(x,y)
=
r_\phi(x,y)
-
\beta
\sum_t
\left(
\log \pi_\theta(y_t\mid x,y_{<t})
-
\log \pi_{\text{ref}}(y_t\mid x,y_{<t})
\right)
$$

第一项 $r_\phi(x,y)$ 是 reward model 或 verifier 给整段 response 的分数。第二项是 token-level KL penalty，惩罚当前 policy 偏离 reference policy。

KL penalty 的作用是约束模型不要为了 reward 走到奇怪分布里。例如 reward model 偏爱长回答，policy 可能学会无限变长；reward model 对某种模板打高分，policy 可能过度套模板。KL penalty 会让这种偏移付出代价。

PPO clipping 和 KL penalty 不是一回事：

- **PPO clipping** 约束当前 policy 相对 old rollout policy 的单次更新；
- **KL penalty** 约束当前 policy 相对 reference policy 的总体偏移。

实践中二者经常同时存在。

## Value Model、Return 与 Advantage

PPO 需要 advantage $A_t$。Advantage 的作用是告诉 policy：当前 token/action 相对于当前状态的平均预期表现是好还是坏。

最简单的形式是：

$$
A_t = R_t - V_\psi(s_t)
$$

其中 $R_t$ 是从当前 token 开始的 return，$V_\psi(s_t)$ 是 value model 对该状态未来回报的估计。

value model 的作用是 baseline。它不改变 reward 本身，但能降低 policy gradient 方差。没有 baseline 时，同一个 reward 会把整条 response 的所有 tokens 都往同一个方向推，噪声很大；有 value baseline 后，只有“比预期更好”的 token/状态才得到正 advantage。

在 LLM 后训练中，value model 常见实现是：

- 在 policy model 上加一个 value head；
- 单独训练一个 critic model；
- actor 和 critic 共享部分 backbone；
- 对每个 token 位置输出 scalar value。

value model 本身也要训练，通常用 value loss 拟合 return：

$$
L_V(\psi)
=
\mathbb{E}_t
\left[
\left(V_\psi(s_t)-R_t\right)^2
\right]
$$

## GAE

GAE，Generalized Advantage Estimation，是 PPO 中常见的 advantage 估计方法。它用 temporal difference residual 的加权和来平衡 bias 和 variance。

单步 TD residual 是：

$$
\delta_t
=
r_t
+
\gamma V(s_{t+1})
-
V(s_t)
$$

GAE advantage 为：

$$
A_t^{\text{GAE}}
=
\sum_{l=0}^{T-t-1}
(\gamma\lambda)^l
\delta_{t+l}
$$

其中：

- $\gamma$ 控制未来 reward 折扣；
- $\lambda$ 控制使用多少步未来 TD residual；
- $\lambda$ 越接近 1，越接近 Monte Carlo return，variance 更高但 bias 更低；
- $\lambda$ 越接近 0，越依赖一步 TD，variance 更低但 bias 更高。

在语言模型 RL 中，GAE 的难点是 reward 经常是 sequence-level 的。很多 token 本身没有即时 reward，最终 RM score 或 verifier score 才在 response 末端出现。因此实现里常把 KL penalty 作为 token-level reward，把最终任务 reward 加到最后一个有效 token 上，再从后往前计算 return 和 advantage。

## LLM PPO 的 Loss 组成

一个典型 PPO 训练目标包含三部分：

```text
policy loss:
  clipped surrogate objective

value loss:
  value model 拟合 returns

entropy bonus:
  可选，用于保持探索和分布多样性
```

若以最小化 loss 表示，常见形式是：

$$
L
=
-
L^{\text{CLIP}}
+
c_v L_V
-
c_e H(\pi_\theta)
$$

其中 $c_v$ 是 value loss coefficient，$c_e$ 是 entropy coefficient。

语言模型实现中还要处理 loss mask：通常只对 assistant response tokens 或 model-generated action tokens 计算 policy loss，不对 prompt、system message、用户输入、工具返回等非模型输出 tokens 计算 policy gradient。

## 训练流程

典型 PPO-RLHF 流程如下：

1. 从 prompt dataset 采样 prompts；
2. 使用当前 policy 生成 responses；
3. 保存 generated tokens、attention mask、old logprobs、values；
4. 用 reward model、verifier 或规则函数对 response 打分；
5. 计算 reference model logprobs，并构造 KL penalty；
6. 组合 token-level KL reward 与最终 sequence reward；
7. 计算 returns 和 advantages；
8. 用当前 policy 重新计算 logprobs；
9. 计算 ratio、clipped policy loss、value loss；
10. 对同一批 rollout 做若干 PPO epochs / minibatches；
11. 更新 policy 和 value model；
12. 监控 reward、KL、clip fraction、entropy、value loss、response length 和评测指标。

这里最容易误解的是：PPO 不是在静态数据上做普通 supervised fine-tuning。它的训练样本来自当前或近当前 policy 的 rollout，policy 更新后，rollout 分布也会变化。因此 PPO 是 online 或 near-on-policy 的训练过程。

## 关键张量与数据字段

LLM PPO 的 batch 通常至少包含：

- `input_ids`：prompt + generated response；
- `attention_mask`：有效 token mask；
- `response_mask` 或 `action_mask`：哪些 token 是 policy 生成并参与 loss；
- `old_logprobs`：rollout policy 生成每个 response token 时的 logprob；
- `ref_logprobs`：reference policy 对同一 token 的 logprob；
- `values`：value model 在 rollout 或 update 时的 value estimate；
- `rewards`：token-level reward 或最终 sequence reward；
- `advantages`：每个 action token 的 advantage；
- `returns`：value model 训练目标；
- `scores`：reward model / verifier 的原始分数。

这些字段决定了 PPO 能不能正确工作。尤其是 `old_logprobs` 和 action mask：前者用于 ratio，后者决定哪些 tokens 真正被当作 actions 更新。

## 关键超参

- **clip range $\epsilon$**：越小越保守，越大更新越激进。
- **KL coefficient $\beta$**：越大越接近 reference，越小越容易 reward hacking。
- **learning rate**：过大容易 collapse，过小 reward 改进慢。
- **rollout batch size**：影响 advantage 估计方差和吞吐效率。
- **PPO epochs**：同一批 rollout 重复优化次数；过多会过拟合旧样本。
- **minibatch size**：影响梯度噪声和显存占用。
- **value loss coefficient**：平衡 policy 与 value 学习。
- **entropy coefficient**：鼓励探索，但过强会损害输出质量。
- **GAE $\gamma,\lambda$**：控制 return / advantage 的时间尺度。
- **generation temperature / top-p**：影响 rollout 多样性。
- **max response length**：影响 reward、KL、显存和训练稳定性。

## 监控指标

PPO-RLHF 需要同时看优化指标、分布指标和任务指标：

- mean reward / verifier pass rate；
- KL to reference；
- policy entropy；
- response length；
- clip fraction；
- approx KL between old and new policy；
- ratio 分布；
- value loss；
- explained variance；
- advantage mean / std；
- invalid format rate；
- reward hacking 样例；
- benchmark regression；
- safety / refusal / hallucination 指标。

只看 reward 上升是不够的。reward 上升同时 KL 暴涨、长度变长、人工偏好下降，通常意味着 reward hacking 或 distribution drift。

## 适用场景

PPO 适合这些情况：

- reward 需要通过当前 policy rollout 在线获得；
- reward 不只是静态 chosen/rejected preference；
- 需要 value model 来降低方差；
- 任务有复杂、多目标或稀疏反馈；
- 希望在训练中持续探索新的 response 分布；
- 可以承担 rollout、reward、value、reference、分布式训练的工程成本。

PPO 不一定适合这些情况：

- 只有静态偏好对，且不需要在线探索；
- reward model 不可靠，容易被 hacking；
- rollout 成本过高；
- value model 很难训练稳定；
- 目标只是让模型模仿高质量回答格式。

## 与 DPO / GRPO 的关系

PPO、DPO、GRPO 不是简单的先后替代关系，而是不同约束下的选择。

[[training/post-training/dpo|DPO]] 用静态 preference pairs 直接优化 policy，不需要在线 rollout、reward model inference 和 value model，工程更简单，但探索能力弱，受限于离线数据覆盖。

[[training/post-training/grpo|GRPO]] 保留 rollout 和 policy optimization，但用同一 prompt 的 group reward 估计 baseline，减少 value model 依赖。它适合多候选、可验证 reward 的 reasoning 任务。

PPO 保留 value model 和 online policy optimization，工程最重，但表达能力也更通用。只要 reward 可以定义，PPO 可以处理更一般的 RLHF / RLVR / agent reward 设置。

## 失败模式

- **Policy collapse**：更新过大导致语言质量退化、重复或模板化。
- **Reward hacking**：模型利用 RM 或 verifier 漏洞获得高分。
- **KL runaway**：policy 快速偏离 reference。
- **Value overfitting**：value model 对 rollout 数据过拟合，advantage 失真。
- **Poor explained variance**：value model 无法解释 returns，policy gradient 方差变大。
- **Length exploitation**：模型通过变长提高 reward。
- **Mode collapse**：模型收敛到单一输出模式，探索下降。
- **Old-policy overfitting**：PPO epochs 太多，当前 policy 过度拟合同一批旧 rollout。
- **Mask 错误**：把 prompt 或 observation tokens 当作 action tokens 更新。
- **Infrastructure complexity**：rollout、reward、reference logprobs、value、训练同步都很重。

## 常见误解

**PPO clipping 不等于 KL penalty。** Clipping 约束的是 current policy 相对 old rollout policy 的 ratio；KL penalty 约束的是 current policy 相对 reference policy 的偏移。

**Value model 不负责给最终答案打分。** Reward model / verifier 给 reward，value model 估计未来 expected return，用来降低 policy gradient 方差。

**PPO 不是普通 SFT。** SFT 模仿数据中的 target tokens；PPO 根据 reward 调整当前 policy 对自己生成 tokens 的概率。

**Reward 高不代表模型真的更好。** 如果 reward 上升但 KL、长度、失败样例和人工偏好变差，通常是 reward hacking。

**Token-level action 不意味着每个 token 都有真实语义动作。** 它是语言模型 RL 的建模方式。对于工具调用或 agent trajectory，还需要额外区分模型生成 tokens 与环境 observation tokens。

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
