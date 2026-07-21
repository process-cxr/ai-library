---
title: GRPO
created: 2026-03-07
published: 2026-03-07
modified: 2026-07-14
type: topic
status: mature
area: training
tags:
  - post-training
  - grpo
  - reasoning
---

GRPO，Group Relative Policy Optimization，是一种面向大模型后训练的 policy optimization 方法。它的核心思想是：**对同一个 prompt 采样一组 responses，用组内相对 reward 构造 advantage，从而避免单独训练 value model。**

GRPO 最常见于数学、代码和可验证 reasoning 任务。原因很直接：这些任务通常可以对同一个问题采样多条解题路径，再用最终答案、单元测试、编译结果、格式检查或 verifier 给每条路径打分。同组样本之间自然形成比较：哪条 response 比同组其他 response 更好，就提高它的概率；哪条更差，就降低它的概率。

可以把 GRPO 先理解成：

```text
PPO-style policy update
  + group-level reward baseline
  - learned value model
```

它不是完全抛弃 PPO，而是保留 policy ratio、clipping、reference / KL 约束等稳定更新思想，把 PPO 中的 learned value baseline 换成了同 prompt 多候选的 group baseline。

## 为什么需要 GRPO

PPO-RLHF 通常需要 policy model、reference model、reward model 和 value model。Value model 用来估计当前 state 的 expected return，从而构造 advantage。但在 LLM reasoning 任务中，value modeling 很难：

- response 很长，reward 通常只在最终答案处出现；
- 中间推理步骤的真实价值很难标注；
- value head / critic 增加显存、计算和训练复杂度；
- value model 估计错误会直接污染 advantage；
- reasoning 任务本身可以通过多采样产生比较信号。

GRPO 的思路是：既然同一 prompt 可以采样多个 responses，而且每个 response 都能被 verifier 打分，那么可以直接用这组 responses 的 reward 分布作为 baseline，而不必训练一个 value model 去预测每个 token 的未来价值。

## Group Sampling

GRPO 的数据基本单元不是单个 prompt-response，而是一个 prompt group：

```text
prompt x
  -> response y(1), reward r(1)
  -> response y(2), reward r(2)
  -> ...
  -> response y(G), reward r(G)
```

其中 $G$ 是 group size。同一个 prompt 下的 $G$ 个 responses 来自当前 policy 的采样，通常会使用非零 temperature 来保持多样性。

这个 group 是 GRPO 的核心。如果每个 prompt 只采一个 response，就没有组内相对比较，也就不能构造 group-relative advantage。

## Group Relative Advantage

最简单的 group advantage 是 reward 减去组内平均值：

$$
A^{(i)}
=
r^{(i)}
-
\frac{1}{G}
\sum_{j=1}^{G}r^{(j)}
$$

其中 $i$ 表示同一 prompt 下第 $i$ 个 response。

如果 $A^{(i)}>0$，说明这条 response 比同组平均更好；如果 $A^{(i)}<0$，说明它比同组平均更差。训练时，前者会被增强，后者会被压低。

很多实现会进一步做组内标准化：

$$
A^{(i)}
=
\frac{
r^{(i)}-\operatorname{mean}(\{r^{(j)}\}_{j=1}^{G})
}{
\operatorname{std}(\{r^{(j)}\}_{j=1}^{G})+\epsilon
}
$$

标准化的作用是减少不同 prompt reward 尺度差异。例如有的问题整体很难，所有 reward 都低；有的问题整体很容易，所有 reward 都高。组内标准化后，训练更关注“同一问题内哪条更好”，而不是不同问题之间 reward 的绝对大小。

但这也带来一个后果：GRPO 更强调相对排序，而不一定保留不同 prompt 的绝对难度信息。

## 方差降低的本质

GRPO 使用 group baseline 的核心作用，是降低 policy gradient 中与 action 好坏无关的 return 波动。Return 的波动通常不只来自 response 质量，还来自 prompt / state 本身难度、reward 噪声、采样随机性和长序列 credit assignment。

可以先把 return 粗略理解为：

```text
return
  = state / prompt 难度带来的平均收益
  + 当前 response / action 质量带来的增减
  + reward / environment / sampling 噪声
```

Policy optimization 真正关心的是第二项：在当前 prompt 或 state 下，这条 response 是否比平均 response 更好。如果直接用 raw return 更新 policy，简单 prompt 天然更容易得到高 reward，困难 prompt 天然更容易得到低 reward，batch 内不同 prompt 的难度差异会混进梯度信号。

PPO 中的 value model 用 $V(s)$ 估计当前 state 的平均收益：

$$
A(s,a)=Q(s,a)-V(s)
$$

这里 $V(s)$ 扣掉的是 state 本身难度带来的平均回报，使 advantage 更关注 action 相对当前 state 的好坏。

GRPO 不训练 value model，而是利用同一个 prompt 下的多条 responses 构造一个经验 baseline：

$$
A^{(i)}
=
r^{(i)}
-
\frac{1}{G}
\sum_{j=1}^{G}r^{(j)}
$$

因为同组 responses 共享同一个 prompt，它们面对的 state 难度基本相同。减去组内平均 reward 后，prompt 本身难度造成的 return 偏移被抵消，剩下的信号更接近“这条 response 相对同 prompt 其他 responses 好不好”。

所以 GRPO 的方差降低不是让某一次 reward 更真实，也不是消除所有噪声，而是把 return 中一部分与 action 无关、主要由 prompt 难度带来的波动去掉。

这也是 group sampling 的价值所在：

```text
同 prompt 多 responses
  -> 共享 prompt 难度
  -> 用组内均值估计该 prompt 的平均收益
  -> reward - group mean
  -> 得到相对 response 质量信号
```

但 group baseline 也有边界。它能较好消掉 prompt-level 难度差异，却不能完全解决 reward 误判、长轨迹 credit assignment、环境随机性或组内样本都很差的问题。如果同组 responses 全错、全对或差异很小，group-relative advantage 仍然会很弱。

## Objective

GRPO 通常保留 PPO-style probability ratio。对第 $i$ 条 response 的第 $t$ 个 token：

$$
\rho_{i,t}(\theta)
=
\frac{
\pi_\theta(y_t^{(i)}\mid x,y_{<t}^{(i)})
}{
\pi_{\theta_{\text{old}}}(y_t^{(i)}\mid x,y_{<t}^{(i)})
}
$$

用 logprob 计算时：

$$
\rho_{i,t}(\theta)
=
\exp
\left(
\log\pi_\theta(y_t^{(i)}\mid x,y_{<t}^{(i)})
-
\log\pi_{\theta_{\text{old}}}(y_t^{(i)}\mid x,y_{<t}^{(i)})
\right)
$$

直观的 clipped objective 可以写成：

$$
L_{\text{GRPO}}
=
\mathbb{E}_{i,t}
\left[
\min
\left(
\rho_{i,t}(\theta)A^{(i)},
\operatorname{clip}(\rho_{i,t}(\theta),1-\epsilon,1+\epsilon)A^{(i)}
\right)
\right]
$$

这里 $A^{(i)}$ 通常是 sequence-level advantage，会广播到该 response 的所有有效 response tokens。也就是说，同一条 response 中参与训练的 token 使用同一个 response-level group advantage。

有些实现会额外加入 reference KL penalty 或在 objective 中加入 token-level KL 项，用来限制 policy 偏离 reference model。

## 与 PPO 的关系

PPO 和 GRPO 的共同点：

- 都是 policy optimization；
- 都使用 rollout 数据，而不是纯静态监督数据；
- 都需要 current policy 与 old policy 的 logprob ratio；
- 都常使用 clipping 控制更新幅度；
- 都常配合 reference policy / KL 约束；
- 都需要监控 reward、KL、长度、clip fraction 和任务指标。

关键区别在于 advantage 的来源。

| 维度           | PPO                              | GRPO                                   |
| -------------- | -------------------------------- | -------------------------------------- |
| baseline       | learned value model / value head | 同 prompt 组内 reward 均值或标准化值   |
| advantage 粒度 | 可 token-level / state-level     | 通常 response-level 后广播到 tokens    |
| critic         | 需要                             | 不需要                                 |
| 工程复杂度     | 更重                             | 相对更轻                               |
| 适合场景       | 通用 online RLHF / 复杂 reward   | 多候选、可验证 reasoning / code / math |
| 主要风险       | value 不稳、工程复杂             | group 质量、reward 稀疏、采样成本      |

所以 GRPO 可以看作在特定任务条件下对 PPO 的简化：当同一 prompt 的多样本比较足够可靠时，可以用 group baseline 替代 value model。

## 训练流程

典型 GRPO 训练流程如下：

1. 从 prompt dataset 采样一批 prompts；
2. 对每个 prompt 用当前 policy 采样 $G$ 个 responses；
3. 保存 response tokens、old logprobs、attention mask、response mask；
4. 用 verifier、unit tests、规则函数或 reward model 给每条 response 打分；
5. 对同一 prompt 内的 rewards 计算 group-relative advantage；
6. 用当前 policy 重新计算每个 response token 的 logprob；
7. 计算 current / old probability ratio；
8. 使用 clipped surrogate objective 更新 policy；
9. 加入 reference KL 或其他 regularization；
10. 监控 reward、pass rate、KL、clip fraction、response length、format validity、group reward variance。

这条流程里最关键的是第 2 和第 5 步：GRPO 依赖同 prompt 多候选。如果 group 内样本没有差异，或者 reward 全都一样，advantage 信号就会很弱。

## Reward 设计

GRPO 的训练质量高度依赖 reward。常见 reward 包括：

- final answer correctness；
- code unit test pass / fail；
- 编译是否通过；
- tool call 格式是否有效；
- 数学答案是否等价；
- verifier score；
- reward model score；
- 规则惩罚，如重复、超长、非法格式；
- 多目标组合 reward。

可验证任务中，reward 可以很简单，例如正确为 1、错误为 0。但简单不等于容易。二值 reward 往往稀疏，如果一个 prompt 的 $G$ 个 responses 全错，则所有 reward 都一样，group advantage 可能接近 0。训练早期常需要更强初始模型、更大的 group size、更多采样多样性，或者加入格式/过程/部分分奖励。

Reward 设计还要防止投机：

- 只奖励格式，会让模型学会空壳格式；
- 只奖励长 reasoning，可能导致冗长无效推理；
- 只检查最终答案字符串，可能鼓励猜测；
- 单元测试覆盖不全时，模型可能过拟合测试漏洞；
- verifier 有偏时，模型可能学会利用 verifier 偏差。

## Group Size

Group size $G$ 是 GRPO 的核心超参之一。

较大的 $G$ 有几个好处：

- 更容易在同一 prompt 下采到好坏不同的 responses；
- group mean / std 更稳定；
- 相对 advantage 更有判别性；
- 探索空间更大。

代价是：

- rollout 成本按 $G$ 倍增加；
- 显存和序列处理压力增加；
- 长 response 下吞吐下降；
- 同组样本长度差异会带来等待和 padding 浪费。

较小的 $G$ 成本低，但 group baseline 方差更大。$G=1$ 时普通 GRPO 失去 group-relative baseline，已经不再是标准 GRPO。

## Reward 标准化的含义

组内 reward 标准化常见但容易误解。它不是单纯为了数值好看，而是改变了训练信号的含义。

未标准化时：

$$
A^{(i)} = r^{(i)}-\bar{r}
$$

不同 prompt 的 reward 差异仍然保留一部分。标准化后：

$$
A^{(i)} = \frac{r^{(i)}-\bar{r}}{\sigma_r+\epsilon}
$$

每个 prompt group 的 reward 尺度被拉到相近范围。这样可以减少高 reward 尺度 prompt 支配训练，但也可能削弱“某些 prompt 本身更有学习价值”的信息。

如果一个 group 的 reward 方差很小，标准化还可能带来数值不稳定，因此实现中通常需要 $\epsilon$、reward clipping 或跳过低方差 group。

## Token-Level Loss 与 Response-Level Reward

GRPO 的 reward 往往是 response-level 的：整道题对或错、代码是否通过测试、回答是否被 verifier 判好。但 policy update 发生在 token-level logprob 上。

因此常见做法是：

```text
response-level reward
  -> group-relative advantage
  -> broadcast 到 response tokens
  -> 对每个有效 token 计算 clipped objective
```

这会带来 credit assignment 问题：同一条 response 中，关键推理 token、普通连接词、格式 token 可能得到相同 advantage。GRPO 的优势是省掉 value model，代价就是中间步骤的精细 credit assignment 较弱。

对于数学和代码任务，这个粗粒度信号仍然能工作，是因为最终 correctness / test reward 比较可靠，并且同一 prompt 下多样本能提供足够比较信号。

## KL 与 Reference Policy

GRPO 通常仍然需要 reference policy 约束。没有 KL 或 reference regularization 时，policy 可能为了 reward 快速偏离原有语言分布，产生过长、格式怪异或 verifier-specific 的输出。

KL 可以以不同方式进入：

- 从 reward 中扣除 token-level KL penalty；
- 在 loss 中加入 reference KL 项；
- 使用近似 KL 监控并动态调节 coefficient；
- 对超过阈值的 batch 降权或跳过。

直观上，reward 负责告诉模型“往哪里优化”，KL 负责告诉模型“不要离原本能力分布太远”。

## 为什么适合 Reasoning RL

GRPO 特别适合 reasoning RL，是因为 reasoning 任务具备两个条件。

第一，reward 相对可验证。数学题可以校验最终答案，代码题可以跑测试，格式可以用规则检查。相比开放式偏好任务，这些 reward 更便宜、更一致。

第二，同一 prompt 可以采样多条路径。推理过程通常存在多种尝试方式，有的路径正确，有的路径错误。Group relative advantage 可以直接强化更可靠的路径，而不必训练 value model 判断每一步中间状态的价值。

这使 GRPO 成为 RLVR，Reinforcement Learning from Verifiable Rewards，中常见的优化方式。

## 适用场景

GRPO 适合这些场景：

- 同一 prompt 可以采样多个 responses；
- reward 可以自动计算或相对可靠地打分；
- 任务允许通过多路径探索提高成功率；
- 不希望训练 value model；
- reward 大多是 sequence-level，但足以区分好坏 responses；
- 训练目标是数学、代码、格式可验证任务或工具调用成功率。

GRPO 不适合或需要谨慎的场景：

- 每个 prompt 只能自然获得一个反馈轨迹；
- rollout 极长，同组等待成本过高；
- reward 极稀疏，group 内样本经常全错或全对；
- reward model 不可靠，容易被 hacking；
- 任务需要精细 step-level credit assignment；
- 环境状态会变化，同 prompt 多次采样不再等价。

这些边界不是说 GRPO 不能用于 agent，而是说使用时要确认 group sampling 和 group-relative baseline 的假设是否成立。

## 与 DPO 的区别

[[training/post-training/dpo|DPO]] 是离线 preference optimization。它给定 chosen / rejected pairs，用 pairwise loss 直接优化 policy，不需要当前 policy rollout。

GRPO 是在线或半在线 policy optimization。它让当前 policy 对 prompt 采样一组 responses，再根据 reward 或 verifier 计算 group advantage。

| 维度             | DPO                  | GRPO                         |
| ---------------- | -------------------- | ---------------------------- |
| 数据来源         | 静态偏好对           | 当前 policy rollouts         |
| Reward           | 隐式偏好             | 显式 reward / verifier       |
| 探索             | 受限于数据           | 可通过采样探索               |
| 是否需要 rollout | 不需要               | 需要                         |
| 是否需要 group   | 不需要               | 需要                         |
| 工程复杂度       | 较低                 | 中高                         |
| 常见任务         | 偏好对齐、风格、安全 | 数学、代码、可验证 reasoning |

## 关键超参

- **group size $G$**：决定组内比较质量和采样成本。
- **clip range $\epsilon$**：控制 policy update 幅度。
- **KL coefficient $\beta$**：限制 policy 偏离 reference。
- **learning rate**：过大易 collapse，过小学习慢。
- **generation temperature / top-p**：控制组内多样性。
- **max response length**：影响计算成本、reward 和长度投机。
- **reward normalization**：影响不同 prompt 与同组样本之间的相对权重。
- **prompt batch size**：与 $G$ 一起决定总 rollout 数量。
- **PPO/GRPO epochs**：同一批 rollout 重复优化次数。
- **loss mask**：决定哪些 tokens 参与 policy loss。

## 监控指标

GRPO 训练中要持续监控：

- mean reward；
- pass rate / solve rate；
- group reward mean / std；
- advantage mean / std；
- KL to reference；
- clip fraction；
- response length；
- entropy；
- format validity；
- invalid tool call rate；
- duplicate / low diversity responses；
- 每个 prompt group 中全错、全对、低方差 group 的比例；
- benchmark regression；
- reward hacking 样例。

尤其要看 group 内 reward 方差。如果大量 group 全部 reward 相同，GRPO 的相对学习信号会很弱。

## 失败模式与边界

- **Group 全错**：同组样本都失败，relative signal 很弱。
- **Group 全对**：同组样本都成功，也缺少区分度。
- **Reward 方差过低**：advantage 接近 0 或标准化不稳定。
- **Reward hacking**：模型利用 verifier 或规则漏洞。
- **格式过拟合**：模型学会满足格式而不提升内容。
- **长度膨胀**：长 reasoning 被间接奖励。
- **Mode collapse**：模型收敛到少数模板，组内多样性下降。
- **采样成本高**：每个 prompt 需要 $G$ 条 response。
- **Credit assignment 粗糙**：sequence-level reward 广播到 tokens，无法精确定位中间错误。
- **KL 敏感**：KL 太弱会漂移，太强会学不动。

## 常见误解

**GRPO 不是没有 RL。** 它仍然是 policy optimization，需要 rollout、reward、logprob ratio 和稳定更新。

**GRPO 不是不需要 KL。** 去掉 value model 不等于可以不约束 reference drift。

**GRPO 不等于只看最终答案。** 最终答案 reward 很常见，但也可以组合格式、过程、工具、测试等多种 reward。

**Group baseline 不是 value model。** 它只提供同 prompt 内的相对比较，不预测任意 state 的未来价值。

**GRPO 的优势依赖多样采样。** 如果 $G$ 条 responses 过于相似，group relative advantage 的信息量会下降。

**GRPO 简化了 value model，但没有消除 credit assignment 问题。** 它把 response-level 相对好坏广播到 tokens，中间推理步骤的归因仍然粗糙。

## 在 Pretraining / Mid-training 中的迁移

GRPO 的 group-relative 思想并不只属于后训练。只要能为同一输入构造多候选 action，并能给候选打分，就可以形成类似 group-relative policy update。

例如，在推理预训练或中训练中，可以对同一 context 采样多条 intermediate thoughts，用它们对后续预测、任务完成或 verifier score 的增益构造 reward。这类方法借用了 GRPO 的相对比较思想，但训练阶段、数据来源和 reward 定义可能不同。

需要注意的是：一旦离开标准 post-training 语境，GRPO 不再是固定 recipe，而是一种“多候选 + 相对 reward + clipped policy update”的训练范式。

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
