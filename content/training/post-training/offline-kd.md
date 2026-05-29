---
title: Offline KD
created: 2026-03-08
published: 2026-03-08
modified: 2026-05-29
type: topic
status: mature
area: training
tags:
  - post-training
  - distillation
  - offline-kd
---

Offline KD，Offline Knowledge Distillation，指 teacher model 预先生成数据、标签、logits、偏好或 reasoning traces，student model 再在这个固定数据集上训练。它是后训练中最常见、最稳定的蒸馏形式，尤其适合把大模型的 instruction following、reasoning traces、工具调用格式或领域回答迁移到较小模型。

Offline KD 的关键特征是：teacher generation 与 student training 解耦。Teacher 可以很慢、很贵、需要多采样或人工筛选；一旦蒸馏数据固定，student 训练就变成常规 SFT、DPO、KD loss 或多任务训练。

## 目标与问题

Offline KD 解决的是“如何把昂贵 teacher 能力转化为可复用训练数据”：

- teacher 太贵，不能在每个 student batch 中在线调用；
- 需要人工或 verifier 对 teacher 输出做离线质量控制；
- 需要可复现的数据版本和训练实验；
- 需要把 best-of-N、RL policy、long CoT 等结果固化；
- 需要给多个 student size 复用同一批 teacher 数据。

这种方式非常适合工程化训练，因为数据一旦生产完成，就可以进入标准数据管线。

## 数据类型

Offline KD 可以缓存多种 teacher 信号：

| 数据 | Student 训练方式 | 典型用途 |
|---|---|---|
| Teacher answers | SFT | 指令跟随、领域 QA、风格迁移 |
| Reasoning traces | SFT / trace tuning | 数学、代码、复杂推理 |
| Tool trajectories | SFT with mask | function calling、agent 行为 |
| Preference pairs | DPO / RM training | 偏好对齐 |
| Reward scores | weighted SFT / filtering | 数据筛选、排序训练 |
| Top-k logits | logits KD | 同 tokenizer 模型压缩 |
| Best-of-N outputs | SFT | 压缩多采样质量 |

Offline KD 不等于只能使用 teacher 的最终回答。只要 teacher 或外部系统能产生稳定可保存的训练信号，都可以作为 offline distillation data。

## 典型流程

1. 选择 prompt distribution；
2. 使用 teacher 生成一个或多个候选；
3. 可选：使用 [[training/post-training/rejection-sampling|Rejection Sampling]]、verifier、reward model 或人工筛选；
4. 统一 chat template 和数据 schema；
5. 去重、过滤、质量分层；
6. 固定数据版本；
7. 训练 student；
8. 评估 student 与 teacher、base model、previous checkpoint 的差异；
9. 根据 student 失败样例更新下一轮数据。

Offline KD 的数据版本管理非常重要。否则很难判断 student 改进来自 teacher、prompt set、筛选器还是训练超参。

## 与 On-policy KD 的区别

| 维度 | Offline KD | On-policy KD |
|---|---|---|
| 数据来源 | teacher 预生成 | student 当前 policy 生成 |
| 稳定性 | 高 | 较低 |
| 成本 | teacher 可离线批处理 | teacher/verifier 常在训练环中 |
| 覆盖 student 错误 | 间接 | 直接 |
| 可复现性 | 强 | 相对弱 |
| 训练复杂度 | 低 | 高 |

Offline KD 的最大缺点是 distribution mismatch：teacher 数据未必覆盖 student 在真实使用中会犯的错误。

## Distribution Mismatch

Offline 数据来自固定 prompt set 和 teacher policy。Student 训练后，自己的输出分布可能不同。当 student 在某类 prompt 上犯错时，offline 数据里未必有对应纠错信号。

例如：

- teacher 生成的答案都很规范，student 部署时却常输出半截 JSON；
- teacher reasoning 很长，student 容量不够，学到片段化推理；
- prompt set 过于干净，真实用户输入更混乱；
- teacher 从不暴露某些错误，因此 student 不知道如何避免。

缓解方式包括：收集 student 失败样例，增量生成 teacher corrections；混合真实用户 prompt；加入 hard negatives；周期性进行 on-policy data refresh。

## 质量控制

Offline KD 的质量控制应在入库前完成：

- format validation；
- exact / unit-test verification；
- reward threshold；
- deduplication；
- toxicity / safety filter；
- trace-answer consistency；
- length and verbosity control；
- benchmark contamination check；
- manual audit for high-risk domains。

蒸馏数据如果包含错误，student 会把错误当成监督目标。相比 online KD，offline KD 的错误更容易被大规模重复训练放大。

## 优势

- 工程稳定，容易复现；
- teacher 可以异步批量生成；
- 适合多 student 重复使用；
- 容易加入人工审核和数据治理；
- 训练阶段不依赖 teacher 服务；
- 可以与 SFT、DPO、logits KD 等目标组合。

## 局限

- 固定数据难覆盖 student 当前错误；
- teacher 输出可能过时；
- 多轮迭代需要重新生成数据；
- 难以利用 student exploration；
- 可能过度模仿 teacher 风格；
- 如果 prompt distribution 有偏，student 泛化有限。

## 实践建议

- 用明确的数据版本记录 teacher、prompt set、sampling config、filter；
- 对 reasoning traces 做 correctness 与 consistency 检查；
- 不要只保留最模板化的 high-score answers；
- 对不同能力维度分桶采样；
- 混合人工数据、真实用户数据和 teacher 数据；
- 对 student 失败样例做下一轮 targeted distillation。

## 经典论文与资料

- [[sources/papers/2016-sequence-level-knowledge-distillation|Sequence-Level Knowledge Distillation]]
- [[sources/papers/2023-distilling-step-by-step|Distilling Step-by-Step]]
- [[sources/papers/2022-self-instruct|Self-Instruct]]

## 相关概念

- [[training/post-training/knowledge-distillation|Knowledge Distillation]]
- [[training/post-training/sequence-level-distillation|Sequence-level Distillation]]
- [[training/post-training/logits-distillation|Logits Distillation]]
- [[training/post-training/on-policy-kd|On-policy KD]]
- [[training/post-training/rejection-sampling|Rejection Sampling]]
