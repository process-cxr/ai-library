---
title: On-policy KD
created: 2026-03-08
published: 2026-03-08
modified: 2026-07-22
type: topic
status: mature
area: training
tags:
  - post-training
  - distillation
  - on-policy-kd
---

On-policy KD，On-policy Knowledge Distillation，指 student model 根据当前 policy 生成样本，再由 teacher model、verifier、reward model 或人工系统对这些样本提供反馈、修正、偏好或分数，随后用这些反馈训练 student。与 [[training/post-training/offline-kd|Offline KD]] 相比，它更贴近 student 当前分布，也更像一个持续纠错闭环。

On-policy KD 的核心价值是：student 犯什么错，就让 teacher 在这些错上提供信号。这对小模型、reasoning model、tool-use model 和 safety alignment 都很重要，因为静态 teacher 数据很难覆盖 student 的真实失败模式。

## 目标与问题

Offline KD 假设固定数据足以训练 student。但 student 部署时的输出分布可能偏离 teacher 数据。On-policy KD 直接从 student 当前行为出发：

1. student 对 prompt 生成 response；
2. teacher / verifier 评价 response；
3. 生成 correction、better answer、preference pair 或 reward；
4. student 学习反馈；
5. 更新后的 student 再生成新样本。

这使训练数据随 student 变化，能更针对性地修复当前 policy 的错误。

## 反馈形式

On-policy KD 可以产生多种训练信号：

| Student 输出       | Teacher / verifier 反馈       | 后续训练                |
| ------------------ | ----------------------------- | ----------------------- |
| 错误答案           | 正确答案                      | SFT correction          |
| 多个候选           | 排序或 chosen/rejected        | DPO / RM                |
| reasoning trace    | step-level critique           | process supervision     |
| code solution      | unit test result              | RL / rejection sampling |
| tool call          | schema error / corrected call | tool-use SFT            |
| unsafe answer      | refusal / safe alternative    | safety tuning           |
| low-quality answer | reward score                  | RL / weighted SFT       |

这种方法不要求 teacher 总是生成完整答案；teacher 可以只是 judge、critic、verifier 或 editor。

## 与 RLHF / GRPO 的关系

On-policy KD 与 [[training/post-training/rlhf|RLHF]]、[[training/post-training/grpo|GRPO]] 有重叠：

- RLHF 中 policy rollout 后由 reward model 打分，本质上也是 on-policy feedback；
- GRPO 对同一 prompt 采样一组 student responses，再用 reward/verifier 组内比较；
- On-policy KD 更强调 teacher-student 知识迁移，可以使用 SFT、DPO、KD loss 或 RL loss；
- RL 更强调通过 reward 优化 policy，不一定产生 teacher answer。

因此，On-policy KD 可以是“带 teacher 的 RL”，也可以是“动态生成纠错数据的 SFT/DPO”。

## Token-Level OPD Signal

On-policy distillation 在 reasoning model 后训练中常被写成 RL-like objective：student 根据当前 policy 采样 response，teacher 只对 student 已经采样出的 tokens 提供 dense reward。传统 OPD 的 token reward 可以写成：

$$
R_t^{\mathrm{OPD}}
=
\log \pi^\*(y_t\mid x,y_{<t})
-
\log \pi_\theta(y_t\mid x,y_{<t})
$$

其中 $\pi^\*$ 是 teacher，$\pi_\theta$ 是 student。这个 reward 的含义是：如果 teacher 比 student 更偏好当前 sampled token，就提高它的概率；如果 teacher 更不偏好，就降低它的概率。

这种形式的优点是训练信号密集，并且作用在 student 自己访问到的状态上。它比 sequence-level SFT 更贴近 inference-time distribution，也不需要为每个任务设计 verifier 或 reward model。

它的限制是，teacher 的完整分布不只包含目标能力，也包含 base model 阶段已经形成的语言偏好、格式偏好和常见表达。对于 reasoning distillation，直接模仿 teacher 分布可能会把这些无关 prior 一起转移给 student。

## Delta Signal

[[sources/papers/2026-on-policy-delta-distillation|On-Policy Delta Distillation]] 提出用 teacher 与 teacher-base 的 logprob 差作为主要 distillation reward：

$$
R_t^\Delta
=
\log \pi^\*(y_t\mid x,y_{<t})
-
\log \pi^\*_{\mathrm{base}}(y_t\mid x,y_{<t})
$$

这里 $\pi^\*_{\mathrm{base}}$ 是 teacher 在 instruction / reasoning tuning 前的 base checkpoint。这个差值可以理解为 teacher 经过 reasoning tuning 后相对自身 base model 发生的 token preference shift。

直观上，OPD 学的是：

```text
teacher 相比 student 更喜欢什么
```

Delta signal 学的是：

```text
reasoning-tuned teacher 相比 teacher-base 新增或强化了什么
```

因此，delta signal 更适合表达某一阶段训练注入的能力增量，而不是 teacher 的完整语言分布。

但 delta signal 不能直接裸用。因为它不包含 student probability，直接最大化可能把 student 推向 delta 最大的 token，而不是稳定对齐 teacher。OPD2 的做法是对 reward 做 centering，并且只在 delta advantage 与传统 OPD advantage 方向一致时更新：

$$
A_t^{\mathrm{D2}}
=
\begin{cases}
A_t^\Delta, & A_t^\Delta A_t^{\mathrm{OPD}} > 0 \\
0, & \mathrm{otherwise}
\end{cases}
$$

这个条件保留了 delta signal 的能力增量信息，同时用传统 OPD signal 约束训练方向，避免 student 被 teacher-base 差值过度带偏。

## 典型流程

1. 准备 prompt distribution，最好包含真实用户样本和难例；
2. 当前 student 生成一个或多个 responses；
3. teacher/verifier 评价 responses；
4. 根据反馈构造训练样本：
   - correction pairs；
   - chosen/rejected pairs；
   - reward-labeled rollouts；
   - revised reasoning traces；
5. 更新 student；
6. 监控 student 分布是否漂移；
7. 周期性刷新 prompts 和 teacher feedback。

这里的关键是保持反馈质量和训练稳定性。Student 生成分布会不断变化，数据分布也会随之变化。

## 优势

- 直接覆盖 student 当前错误；
- 能利用 student exploration；
- 对 hard prompts 和长尾失败更有效；
- 可以形成持续改进闭环；
- 适合 verifier-rich 任务，如数学、代码、格式化输出；
- 可以减少无关 teacher 数据浪费。

## 成本与复杂度

On-policy KD 的代价明显高于 offline KD：

- 训练中需要反复调用 teacher 或 verifier；
- 数据不可完全预生成，系统吞吐复杂；
- teacher latency 会拖慢训练；
- feedback 分布随 student 改变，实验可复现性较弱；
- 如果 feedback 噪声大，错误会动态放大；
- 需要防止 student 钻 verifier 漏洞。

因此它适合高价值任务或关键能力提升，不一定适合所有后训练数据。

## Stability 问题

On-policy 训练容易出现反馈循环：

- student 生成某类输出；
- teacher/scorer 偏好这种输出；
- student 进一步强化；
- 分布变窄或走向 reward hacking。

缓解方式包括：

- 保留 reference / KL constraint；
- 混合 offline high-quality data；
- 使用多样 prompt 和多样采样；
- 定期人工 audit；
- 使用多个 verifier / reward；
- 限制单轮更新幅度；
- 保持 held-out evaluation。

## 与 Rejection Sampling 的组合

On-policy KD 常使用 [[training/post-training/rejection-sampling|Rejection Sampling]]：

```text
student samples N responses
teacher / verifier scores them
keep best response or build preference pairs
train student
```

如果只保留 best response 并做 SFT，这是 on-policy sequence distillation；如果构造 chosen/rejected 并做 DPO，这是 on-policy preference distillation；如果直接用 reward 做 policy gradient，则接近 RL。

## 适用场景

On-policy KD 特别适合：

- student 已经有一定基础能力，但存在可观察失败；
- 有可靠 verifier 或强 teacher；
- 任务反馈可以自动化；
- 需要持续适配真实用户分布；
- 希望提升 reasoning、代码、工具调用等动态能力。

不适合：

- teacher 调用成本极高且无法批量；
- feedback 噪声大；
- 没有稳定评估；
- student 初始能力太弱，rollout 几乎全错；
- 安全高风险但缺乏人工审核。

## 经典论文与资料

- [[sources/papers/2020-learning-to-summarize-from-human-feedback|Learning to summarize from human feedback]]
- [[sources/papers/2022-instructgpt|Training language models to follow instructions with human feedback]]
- [[sources/papers/2024-deepseekmath|DeepSeekMath]]
- [[sources/papers/2023-distilling-step-by-step|Distilling Step-by-Step]]
- [[sources/papers/2026-on-policy-delta-distillation|On-Policy Delta Distillation]]

## 相关概念

- [[training/post-training/knowledge-distillation|Knowledge Distillation]]
- [[training/post-training/offline-kd|Offline KD]]
- [[training/post-training/rejection-sampling|Rejection Sampling]]
- [[training/post-training/grpo|GRPO]]
- [[training/post-training/rlhf|RLHF]]
- [[training/post-training/dpo|DPO]]
