---
title: On-policy KD
created: 2026-03-08
published: 2026-03-08
modified: 2026-05-29
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

| Student 输出 | Teacher / verifier 反馈 | 后续训练 |
|---|---|---|
| 错误答案 | 正确答案 | SFT correction |
| 多个候选 | 排序或 chosen/rejected | DPO / RM |
| reasoning trace | step-level critique | process supervision |
| code solution | unit test result | RL / rejection sampling |
| tool call | schema error / corrected call | tool-use SFT |
| unsafe answer | refusal / safe alternative | safety tuning |
| low-quality answer | reward score | RL / weighted SFT |

这种方法不要求 teacher 总是生成完整答案；teacher 可以只是 judge、critic、verifier 或 editor。

## 与 RLHF / GRPO 的关系

On-policy KD 与 [[training/post-training/rlhf|RLHF]]、[[training/post-training/grpo|GRPO]] 有重叠：

- RLHF 中 policy rollout 后由 reward model 打分，本质上也是 on-policy feedback；
- GRPO 对同一 prompt 采样一组 student responses，再用 reward/verifier 组内比较；
- On-policy KD 更强调 teacher-student 知识迁移，可以使用 SFT、DPO、KD loss 或 RL loss；
- RL 更强调通过 reward 优化 policy，不一定产生 teacher answer。

因此，On-policy KD 可以是“带 teacher 的 RL”，也可以是“动态生成纠错数据的 SFT/DPO”。

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

## 相关概念

- [[training/post-training/knowledge-distillation|Knowledge Distillation]]
- [[training/post-training/offline-kd|Offline KD]]
- [[training/post-training/rejection-sampling|Rejection Sampling]]
- [[training/post-training/grpo|GRPO]]
- [[training/post-training/rlhf|RLHF]]
- [[training/post-training/dpo|DPO]]
