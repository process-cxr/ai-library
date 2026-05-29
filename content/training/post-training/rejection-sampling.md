---
title: Rejection Sampling
created: 2026-03-07
published: 2026-03-07
modified: 2026-05-29
type: topic
status: mature
area: training
tags:
  - post-training
  - sampling
  - data
---

Rejection Sampling 在大模型后训练中通常指：对同一个 prompt 生成多个候选回答，再用 reward model、verifier、规则、人工或其他筛选器选择高质量样本。它既可以作为 inference-time best-of-N 技术，也可以作为 training-time data construction 技术，用于构造 [[training/post-training/sft|SFT]]、[[training/post-training/dpo|DPO]]、[[training/post-training/rlhf|RLHF]] 或 distillation 数据。

在后训练语境里，rejection sampling 的价值不是数学采样教科书中的“从目标分布采样”，而是利用模型自身的多样生成能力和外部评价器，把一次采样的不稳定输出转化为更可靠的训练样本。

## 目标与问题

模型对同一 prompt 可能生成多个质量不同的 responses。单次采样可能出错，但多次采样中常包含更好的候选。Rejection sampling 试图解决：

- 如何从模型自身生成中挖出高质量答案；
- 如何把 reward model / verifier 能力转化为训练数据；
- 如何在不直接 RL 的情况下提升 policy；
- 如何为 DPO 构造 chosen/rejected pairs；
- 如何为小模型蒸馏强模型或强采样策略。

它是一种“生成-筛选-再训练”的闭环数据工程方法。

## 基本流程

典型流程：

1. 准备 prompt set；
2. 对每个 prompt 从 policy 或 teacher model 采样 $N$ 个候选；
3. 用 scorer 对候选打分；
4. 选择 top-1、top-k 或满足阈值的候选；
5. 可选：构造 chosen/rejected pairs；
6. 将筛选结果加入 SFT、DPO、蒸馏或评估数据；
7. 重新训练模型并迭代。

Scorer 可以是：

- reward model；
- rule-based verifier；
- unit tests；
- exact answer checker；
- LLM-as-judge；
- 人工标注；
- 多个评估器的组合。

## Best-of-N

Best-of-N 是 rejection sampling 的常见形式：对 prompt 采样 $N$ 个 responses，选择 reward 最高者：

$$
y^* = \arg\max_{y_i \sim \pi(\cdot\mid x)} s(x,y_i)
$$

其中 $s$ 是 scorer。Best-of-N 可以显著提高单次输出质量，但代价是推理成本乘以 $N$。如果把 best-of-N 结果作为训练数据，再用 SFT 训练模型，就相当于把“采样时搜索”蒸馏进单次生成 policy。

## 构造 SFT 数据

Rejection sampling 可以生成高质量 SFT targets：

```text
prompt x
sample y_1, ..., y_N
score each y_i
keep y_best as supervised target
train pi_theta(y_best | x)
```

这种方法适合数学、代码、结构化输出等可验证任务。例如代码生成可以采样多个程序，通过单元测试筛选 passing solution，再把通过测试的代码作为 SFT 数据。

风险是：如果 scorer 有偏，SFT 会把偏差固化；如果只保留 top-1，数据多样性会下降；如果所有候选都错，筛选无法创造正确答案。

## 构造偏好数据

Rejection sampling 也可以构造 DPO / RM 数据：

```text
chosen = high-score response
rejected = low-score response
```

更强的做法是构造 hard preference pairs：chosen 和 rejected 都是看起来合理的回答，但 chosen 在事实性、推理正确性或安全性上更好。Hard pairs 比“好答案 vs 乱码”更有训练价值。

偏好 pair 的质量取决于：

- 候选来源是否多样；
- scorer 是否可靠；
- chosen/rejected 差距是否适中；
- 是否覆盖真实失败模式。

## 与 RL 的关系

Rejection sampling 可以看作比 RL 更保守的数据改进方法。它不直接通过 policy gradient 更新模型，而是先筛选出更好的样本，再用 supervised 或 preference loss 学习。

与 [[training/post-training/grpo|GRPO]] 相比：

- GRPO 使用同组 reward 直接更新当前 policy；
- rejection sampling 通常先离线选择样本，再训练；
- GRPO 更像 online optimization；
- rejection sampling 更像 data curation 和 policy distillation。

两者可以组合：先用 rejection sampling 生成高质量 warmup 数据，再用 GRPO / RLHF 继续优化。

## Scorer 设计

不同 scorer 决定不同能力：

| Scorer | 适用任务 | 风险 |
|---|---|---|
| Reward Model | 开放对话、总结、写作 | reward hacking、偏好偏差 |
| Exact Match | 数学短答案、选择题 | 只看最终答案，忽略过程 |
| Unit Tests | 代码生成 | 测试覆盖不足、投机 |
| Rule Checker | JSON、格式、长度 | 只优化表面格式 |
| LLM Judge | 开放任务自动评估 | judge bias、不可复现 |
| Human Review | 高风险任务 | 成本高、速度慢 |

优秀的 rejection sampling pipeline 通常使用多阶段筛选：先用规则过滤格式，再用 verifier 检查正确性，最后用 RM 或人工评估质量。

## 失败模式与边界

### 多样性下降

只保留最高分样本会让数据分布变窄。模型可能学到单一解题模板或固定写作风格。可以通过 top-k、多样性约束、cluster-based selection 缓解。

### Scorer Bias 固化

如果 reward model 偏爱长答案，rejection sampling 会选择长答案，SFT 后模型会更长；如果 verifier 有漏洞，模型会学习利用漏洞。

### 假阳性样本

代码通过弱测试不代表正确；数学答案 match 不代表推理可靠；LLM judge 高分不代表事实真实。筛选器只验证它能看到的属性。

### 采样成本

每个 prompt 采样 $N$ 个候选会显著增加计算。高 temperature 提高多样性但增加错误率；低 temperature 稳定但候选差异不足。

### 自我训练退化

如果同一个模型生成、筛选、再训练，错误可能在迭代中积累。使用更强 teacher、外部 verifier 或人工 audit 可以降低风险。

## 经典论文与资料

- [[sources/papers/2022-instructgpt|Training language models to follow instructions with human feedback]]
- [[sources/papers/2024-deepseekmath|DeepSeekMath]]
- [[sources/papers/2023-distilling-step-by-step|Distilling Step-by-Step]]

## 相关概念

- [[training/post-training/sft|SFT]]
- [[training/post-training/dpo|DPO]]
- [[training/post-training/reward-model|Reward Model]]
- [[training/post-training/grpo|GRPO]]
- [[training/post-training/knowledge-distillation|Knowledge Distillation]]
