---
title: Synthetic Data
created: 2026-03-15
published: 2026-03-15
modified: 2026-05-31
type: topic
status: mature
area: training
tags:
  - data-engineering
  - synthetic-data
---

Synthetic Data 是由模型、程序、模拟器、规则系统或人机协作流程生成的数据。它可以补足稀缺能力、控制任务格式、扩大训练覆盖，也可能引入模型偏差、错误放大、模板化和评测污染。

在大模型训练中，synthetic data 同时出现在预训练、中训练和后训练中，但每个阶段的目标不同。它不是“免费数据”，而是一种需要生成、筛选、验证和配比的数据工程方法。

## 主要用途

| 阶段 | 用途 | 例子 |
|---|---|---|
| Pretraining | 补足低资源领域或格式 | 合成代码注释、数学文本、结构化文档 |
| Mid-training | 注入特定能力 | 数学推理轨迹、代码执行数据、长上下文问答 |
| SFT | 构造指令响应数据 | Self-Instruct、teacher-generated answers |
| Preference / RL | 构造候选回答和比较样本 | rejection sampling、verifier scoring |
| Evaluation | 生成测试变体或对抗样本 | synthetic benchmark items、stress tests |

不同阶段应分开记录 synthetic data 的来源、生成模型、prompt、筛选器和采样比例。把所有合成数据混成一个 corpus，会让后续分析非常困难。

## Teacher Generation

Teacher generation 指用强模型生成训练样本。常见流程：

1. 选择 seed tasks、documents、questions 或 prompts；
2. 用 teacher model 生成回答、解释、推理轨迹或代码；
3. 用规则、verifier、unit tests 或 judge model 筛选；
4. 去重和质量过滤；
5. 按能力目标混入训练集。

Teacher generation 的优势是可扩展、格式可控、适合稀缺任务。风险是 student 会继承 teacher 的风格、盲点和错误；如果所有数据来自少数 teacher，模型可能变得模板化。

## Self-Instruct 与指令数据扩展

Self-Instruct 类方法用少量 seed instructions 启动，让模型生成更多 instructions、inputs 和 outputs，再经过过滤形成 SFT 数据。它的核心价值是扩大 instruction diversity，减少人工标注成本。

典型风险：

- 生成任务过于相似；
- 输出看似合理但事实错误；
- 指令分布偏离真实用户需求；
- 格式单一，模型学到模板化回答；
- 生成数据与评测题过度相似；
- teacher model 的安全和价值偏好被隐式蒸馏。

因此，Self-Instruct 数据通常需要 diversity filtering、semantic dedup、人工抽检和多维评测。

## Verifier 与可验证合成数据

对于数学、代码、工具使用和检索类任务，可以用外部 verifier 提高 synthetic data 可靠性：

- 代码：运行 unit tests、type checks、linters；
- 数学：使用 symbolic solver、数值校验、答案一致性；
- tool use：执行工具调用并验证返回值；
- retrieval：检查答案是否由给定文档支持；
- structured output：schema validation。

可验证数据的优势是质量边界更清楚。它可以用于训练 reasoning、code、tool use 和 RLVR。局限是 verifier 覆盖有限，容易让模型过拟合可验证形式，而非真实开放问题。

## Rejection Sampling 与 Best-of-N

Rejection sampling 常用于从多个候选中选出高质量样本：

1. 对同一 prompt 采样 $N$ 个 candidate responses；
2. 用 reward model、verifier、judge 或规则打分；
3. 选择最高分或超过阈值的样本；
4. 用选中样本做 SFT、distillation 或偏好训练。

这会把模型分布向 scorer 偏好的区域移动。质量取决于 scorer 是否可靠。如果 reward model 有漏洞，rejection sampling 会放大这些漏洞。

## 合成数据与数据污染

Synthetic data 很容易产生隐蔽污染：

- teacher 已经见过 benchmark；
- prompt seed 来自公开题库；
- 合成题是 benchmark 的轻微改写；
- verifier 使用了测试集信息；
- 生成过程把答案泄漏到上下文；
- 多轮生成中复制了原始评测内容。

因此，合成数据仍需要 contamination audit。不能因为数据是“新生成”的，就默认它没有泄漏。

## 合成数据的配比

合成数据通常不应无限上采样。过高比例可能导致：

- 模型语言风格趋同；
- 错误被系统性放大；
- 真实世界表达减少；
- teacher bias 被蒸馏；
- benchmark 上升但开放场景退化；
- 模型对 verifier 偏好过拟合。

更稳妥的做法是把 synthetic data 作为显式 domain，记录比例，并在 validation 中单独观察 synthetic-heavy capabilities 与 natural data capabilities。

## 质量控制

合成数据质量控制通常包括：

- prompt diversity；
- semantic dedup；
- answer correctness checks；
- format validation；
- toxicity and safety filtering；
- teacher/judge model version tracking；
- human audit samples；
- downstream ablation；
- contamination matching；
- train/eval split hygiene。

如果合成数据用于后训练，还要检查它是否改变 assistant 风格、拒答边界、冗长程度和安全行为。

## 常见失败模式

- **模型自我污染**：student 学到 teacher 的错误和偏见。
- **模式坍缩**：生成样本格式高度相似，降低多样性。
- **验证器过拟合**：模型学会满足 verifier，而不是真正解决任务。
- **隐蔽 benchmark leakage**：生成数据来源或 teacher 记忆导致污染。
- **合成比例过高**：自然语言和真实用户分布被稀释。
- **缺少版本记录**：无法追踪数据由哪个 teacher、prompt 和筛选器生成。

## 相关概念

- [[training/data-engineering/data-engineering|Data Engineering]]
- [[training/data-engineering/quality-filtering|Quality Filtering]]
- [[training/data-engineering/deduplication|Deduplication]]
- [[training/pretraining/data-mix|Data Mix]]
- [[training/post-training/sft|SFT]]
- [[training/post-training/rejection-sampling|Rejection Sampling]]
- [[training/post-training/knowledge-distillation|Knowledge Distillation]]
- [[sources/papers/2022-self-instruct|Self-Instruct]]
