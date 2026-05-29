---
title: Knowledge Distillation
created: 2026-03-08
published: 2026-03-08
modified: 2026-05-29
type: topic
status: mature
area: training
tags:
  - post-training
  - distillation
---

Knowledge Distillation，知识蒸馏，是把 teacher model 的行为、分布、推理过程或偏好信号迁移到 student model 的训练方法。在后训练中，distillation 不只是模型压缩技术，也是一种能力固化技术：把昂贵模型、强采样策略、RL policy、verifier 筛选结果或长 reasoning traces 转化成更便宜、更稳定、可部署的 student 能力。

与预训练不同，后训练蒸馏通常聚焦 assistant behavior：指令跟随、推理路径、工具调用、安全拒答、代码能力、领域风格和偏好对齐。

## 目标与问题

后训练蒸馏常见目标包括：

- 让小模型模仿大模型回答质量；
- 把 best-of-N / rejection sampling 的搜索收益压缩到单次生成；
- 把 [[training/post-training/rlhf|RLHF]] 或 [[training/post-training/grpo|GRPO]] 后的强 policy 迁移给便宜模型；
- 用 teacher reasoning traces 提升 student 推理能力；
- 用 teacher logits 提供比 hard label 更丰富的 token distribution；
- 在部署成本、延迟、显存和质量之间折中。

蒸馏的基本结构是 teacher-student：

```text
teacher model / verifier / reward policy
        ↓ produces
answers / logits / preferences / rationales / traces
        ↓ trains
student model
```

## 蒸馏信号类型

| 类型 | 学习目标 | 代表笔记 |
|---|---|---|
| Logits distillation | 学习 teacher token distribution | [[training/post-training/logits-distillation|Logits Distillation]] |
| Sequence-level distillation | 学习 teacher 完整输出序列 | [[training/post-training/sequence-level-distillation|Sequence-level Distillation]] |
| Rationale / trace distillation | 学习 teacher reasoning steps | [[training/post-training/sequence-level-distillation|Sequence-level Distillation]] |
| Preference distillation | 学习 teacher/judge 偏好排序 | [[training/post-training/dpo|DPO]]、[[training/post-training/reward-model|Reward Model]] |
| Policy distillation | 学习强 policy 的行为分布 | [[training/post-training/offline-kd|Offline KD]]、[[training/post-training/on-policy-kd|On-policy KD]] |
| Data distillation | teacher 生成或筛选训练数据 | [[training/post-training/rejection-sampling|Rejection Sampling]] |

这些类型可以组合。例如 teacher 先采样多条 CoT，经 verifier 选择正确答案，再把 reasoning trace 作为 sequence-level SFT 数据训练 student。

## Logits 与 Hard Labels

普通 SFT 使用 hard label：每个位置只有参考 token 是正确目标。Logits distillation 使用 teacher 的 soft distribution：

$$
\mathcal{L}_{\text{KD}}
= T^2 \cdot
\mathrm{KL}
\left(
p_T^{\text{teacher}}(\cdot\mid x)
\|
p_T^{\text{student}}(\cdot\mid x)
\right)
$$

soft labels 包含 teacher 对其他 token 的相对偏好，能传递“哪些 token 在语义或格式上同样可接受”的信息。但保存全词表 logits 成本极高，因此后训练中更常见的是 sequence-level distillation 或 top-k logits distillation。

## Sequence-level Distillation

Sequence-level distillation 让 teacher 直接生成完整回答，student 对这些回答做 SFT。它简单、便宜、容易扩展，特别适合：

- instruction following；
- reasoning traces；
- domain answers；
- tool-use demonstrations；
- safety refusals；
- long-form writing style。

它的缺点是丢失 teacher 在每一步 token 上的完整分布，只保留了一个或少数采样结果。因此数据质量、采样温度、筛选器和 prompt 覆盖非常关键。

## Offline KD 与 On-policy KD

[[training/post-training/offline-kd|Offline KD]] 指 teacher 预先生成数据，student 在固定数据集上训练。优点是稳定、便宜、可复现；缺点是 teacher 数据未必覆盖 student 当前失败分布。

[[training/post-training/on-policy-kd|On-policy KD]] 指 student 根据当前 policy 生成样本，再由 teacher、verifier 或 reward model 提供反馈。优点是更贴近 student 分布，能针对性纠错；缺点是系统更复杂，成本更高，也更容易训练不稳定。

## 蒸馏与压缩

蒸馏常用于压缩大模型能力，但“压缩”并不只意味着参数更少。它也可以压缩：

- 多次采样成本：best-of-N -> one-shot student；
- 长 reasoning 成本：long CoT teacher -> shorter student answer；
- RL 训练成本：RL policy -> SFT/DPO student；
- 多模型 ensemble：ensemble teacher -> single student；
- 专家流程：工具/检索/评审结果 -> 直接回答模型。

因此，蒸馏是训练成本、推理成本和质量之间的桥梁。

## 与 RLHF / DPO / GRPO 的关系

- RLHF 训练出的高 reward policy 可以作为 teacher；
- DPO 可以看作 preference distillation 的一种训练形式；
- GRPO 生成的高质量 reasoning trajectories 可以被蒸馏到小模型；
- rejection sampling 是蒸馏数据生产的重要方式；
- reward model / verifier 可以过滤 teacher outputs。

在现代 reasoning model 路线中，常见模式是：大模型通过 RL 或多采样获得强推理能力，再把高质量 traces 蒸馏给小模型，使小模型在较低成本下复现部分能力。

## 失败模式与边界

### Teacher 错误继承

Student 会学习 teacher 的错误、偏见、幻觉和格式习惯。Teacher 越强不代表输出总是正确，尤其在长尾知识和复杂推理上。

### 能力不可完全转移

小模型容量有限，无法完整吸收大模型能力。蒸馏可能提高特定任务表现，但不一定提升通用推理、知识覆盖或鲁棒性。

### Distribution Mismatch

Offline teacher data 可能与 student 实际使用分布不一致。Student 在真实 prompt 上犯的错，训练数据里未必出现。

### Reasoning Trace 质量

CoT 或 rationale 不一定真实反映 teacher 内部推理。错误 trace 会误导 student；过长 trace 会增加推理成本；隐藏 trace 与公开回答之间也要有产品和安全边界。

### 过度模仿风格

Student 可能学到 teacher 的口吻、冗长程度和模板，而不是核心能力。蒸馏数据需要关注内容正确性和行为目标。

## 经典论文与资料

- [[sources/papers/2016-sequence-level-knowledge-distillation|Sequence-Level Knowledge Distillation]]
- [[sources/papers/2023-distilling-step-by-step|Distilling Step-by-Step]]
- [[sources/papers/2024-deepseekmath|DeepSeekMath]]
- [[sources/papers/2022-self-instruct|Self-Instruct]]

## 相关概念

- [[fundamentals/information-theory/knowledge-distillation|知识蒸馏的信息论视角]]
- [[training/post-training/logits-distillation|Logits Distillation]]
- [[training/post-training/sequence-level-distillation|Sequence-level Distillation]]
- [[training/post-training/offline-kd|Offline KD]]
- [[training/post-training/on-policy-kd|On-policy KD]]
- [[training/post-training/rejection-sampling|Rejection Sampling]]
