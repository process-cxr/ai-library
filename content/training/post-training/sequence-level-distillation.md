---
title: Sequence-level Distillation
created: 2026-03-08
published: 2026-03-08
modified: 2026-05-29
type: topic
status: mature
area: training
tags:
  - post-training
  - distillation
  - sequence-level
---

Sequence-level Distillation 是让 teacher model 生成完整输出序列，再用这些序列训练 student model 的蒸馏方法。它不同于 [[training/post-training/logits-distillation|Logits Distillation]] 对齐每个 token 的完整概率分布，而是把 teacher 的一个或多个生成结果当作 supervised target。

在后训练中，sequence-level distillation 非常常见，因为它简单、可扩展、对 tokenizer 要求低，并且天然适合 instruction answers、reasoning traces、tool-use demonstrations 和 safety responses。

## 目标与问题

Teacher model 往往比 student 更强、更慢、更贵。Sequence-level distillation 试图把 teacher 的输出行为固化到 student 中：

- 大模型回答 -> 小模型 SFT 数据；
- best-of-N 结果 -> one-shot student；
- long CoT reasoning -> student reasoning trace；
- tool-use trajectory -> student tool policy；
- RL policy outputs -> supervised student；
- expert model 风格 -> domain assistant。

它把“teacher 如何回答”转化为 student 可直接模仿的序列。

## 基本流程

1. 准备 prompt set；
2. 使用 teacher 生成 responses；
3. 可选：多采样并用 verifier / RM / 人工筛选；
4. 可选：保留 reasoning trace、tool call 或中间步骤；
5. 构造 chat-template-aligned SFT 数据；
6. 训练 student；
7. 评估 student 是否复现 teacher 的质量和格式。

Sequence-level distillation 的核心质量控制发生在第 2-4 步。Teacher 输出不是天然正确数据，必须经过过滤、去重和格式校验。

## 与 Logits Distillation 的区别

| 维度 | Logits Distillation | Sequence-level Distillation |
|---|---|---|
| 学习信号 | teacher token distribution | teacher generated sequence |
| 存储成本 | 高 | 低 |
| tokenizer 要求 | 通常需要相同或可映射 | 可以不同 |
| 信息量 | 保留分布信息 | 只保留采样结果 |
| 实现复杂度 | 高 | 低 |
| 后训练常用性 | 模型压缩时常见 | instruction/reasoning KD 常见 |

Sequence-level KD 丢失了 teacher 对替代 token 的概率判断，但换来了可扩展性。

## Reasoning Trace 蒸馏

在 reasoning model 中，teacher 可以生成 Chain-of-Thought、解题步骤、证明草稿或代码调试过程。Student 学习这些 traces 后，可能获得更强的逐步求解能力。

Reasoning trace 蒸馏通常有两种目标：

1. **学习过程**：student 在推理时也输出 CoT 或 scratchpad；
2. **学习结果**：student 内化 teacher 的推理模式，但部署时输出简洁答案。

需要注意：CoT 并不一定是 teacher 真实内部机制的忠实解释。它是一个可训练文本 artifact。错误或伪合理 trace 会让 student 学到错误推理习惯。

## 偏好筛选后的序列

Sequence-level distillation 经常与 [[training/post-training/rejection-sampling|Rejection Sampling]] 结合：

```text
teacher samples N responses
verifier / reward model scores them
keep high-quality responses
train student on selected sequences
```

这可以把多次采样、reward ranking 或 verifier 检查带来的质量提升压缩到 student 的单次生成中。DeepSeek-R1 类 reasoning distillation 的公开讨论中，这种“大模型强推理轨迹 -> 小模型”的路线非常重要。

## Tool-use Trajectory 蒸馏

对于 agent 或 tool-use 模型，sequence-level target 可以包含：

- assistant 的 tool call；
- tool name 和 arguments；
- tool observation；
- assistant 对 observation 的整合回答。

通常 tool observation 不作为模型要生成的目标，而是作为上下文。需要通过 loss mask 区分 assistant-generated tokens 和 external tool tokens。

## 数据质量控制

高质量 sequence-level KD 需要检查：

- teacher answer 是否正确；
- format 是否符合目标 chat template；
- reasoning trace 是否与 final answer 一致；
- 是否包含不可公开的隐藏信息、内部环境信息或私有标识；
- 是否重复、模板化或过长；
- 是否覆盖目标任务分布；
- 是否混入 benchmark 测试集；
- 是否存在 safety policy 冲突。

蒸馏数据越像“teacher 自言自语的原始采样”，风险越大；越经过验证和整理，越适合作为长期训练数据。

## 失败模式与边界

- **单样本偏差**：一个 teacher sample 不能代表完整答案分布。
- **错误 trace 固化**：错误 reasoning 被 student 学成稳定模式。
- **风格过拟合**：student 学到 teacher 口吻而非能力。
- **长度膨胀**：CoT 数据过多导致推理时过度冗长。
- **可验证性不足**：开放任务很难确定 teacher 输出是否真的好。
- **容量限制**：小模型无法完整吸收大模型复杂能力。

## 经典论文与资料

- [[sources/papers/2016-sequence-level-knowledge-distillation|Sequence-Level Knowledge Distillation]]
- [[sources/papers/2023-distilling-step-by-step|Distilling Step-by-Step]]
- [[sources/papers/2024-deepseekmath|DeepSeekMath]]

## 相关概念

- [[training/post-training/knowledge-distillation|Knowledge Distillation]]
- [[training/post-training/logits-distillation|Logits Distillation]]
- [[training/post-training/rejection-sampling|Rejection Sampling]]
- [[training/post-training/offline-kd|Offline KD]]
- [[training/post-training/on-policy-kd|On-policy KD]]
