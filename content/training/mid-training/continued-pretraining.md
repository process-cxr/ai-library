---
title: Continued Pretraining
created: 2026-02-28
published: 2026-02-28
modified: 2026-06-30
type: topic
status: mature
area: training
tags:
  - mid-training
  - continued-pretraining
---

Continued Pretraining，简称 CPT，是在已有 base model 或 intermediate checkpoint 上继续做预训练。它通常仍使用语言建模目标，但更换数据、阶段、学习率或训练预算，用来注入领域知识、延长训练、改善某类能力或完成预训练末期退火。

CPT 处在 [[training/pretraining/pretraining|Pretraining]] 与 [[training/post-training/sft|SFT]] 之间。它继承 base model 的通用能力，又比 SFT 更接近预训练式的数据和目标，因此常被用作能力塑形、领域迁移和训练末期整理的中间层。

## 目标与使用场景

CPT 常见目标包括：

- 领域适配：继续训练法律、医疗、金融、代码、数学等领域语料。
- 能力注入：提高代码、数学、长上下文、多语言或科学问答能力。
- 数据修正：用更高质量或更干净的数据改善早期训练带来的噪声。
- 训练末期增强：在 base model 接近收敛后，用小学习率和高质量数据做能力整理。
- 上下文扩展：在已有模型上加入长文档、代码仓库、多轮对话或合成长上下文数据。
- 知识更新：补充较新网页、论文、文档和领域资料，但需要控制遗忘和污染。

## 与从零预训练的区别

从零预训练需要从随机初始化开始学习词法、语法、知识和基本模式，训练 token 量巨大。CPT 则继承已有模型能力，通常使用更小学习率、更少 token 和更聚焦的数据。

因此，CPT 的风险不是“学不会基础语言”，而是：

- 领域数据过窄导致灾难性遗忘；
- 数据配比不当导致通用能力下降；
- 学习率过高破坏已有表示；
- 数据污染影响评测可信度。

## 训练目标与数据形态

CPT 通常继续使用 [[training/pretraining/objective|next-token prediction objective]]：

$$
\mathcal{L}_{\mathrm{CPT}}
= -\sum_t \log p_\theta(x_t \mid x_{<t})
$$

数据可以是：

- 原始文档：书籍、网页、论文、教材、手册；
- 代码仓库：源码、README、issue、测试、API 文档；
- 数学数据：题目、解答、证明、推理轨迹；
- 长上下文数据：长文档、多文档 QA、代码仓库、长对话；
- 高质量混合数据：训练末期 annealing 使用的 curated mix；
- 合成数据：teacher-generated reasoning、long-context QA、tool traces。

CPT 不是必须使用 prompt-response 格式。如果数据已经变成 instruction / answer 对，它更接近 SFT。实际项目中二者可以混合，但应清楚区分 loss mask、chat template 和训练目标。

## 关键超参

CPT 对已有模型继续优化，超参通常比从零预训练更保守：

| 超参 | 影响 |
|---|---|
| learning rate | 过高会破坏已有表示，过低则难以注入新能力 |
| warmup steps | 缓解切换数据分布后的 loss spike |
| token budget | 决定新领域注入强度和遗忘风险 |
| data mix | 控制目标能力与通用能力之间的平衡 |
| sequence length | 决定长上下文学习和 activation 成本 |
| replay ratio | 保留通用数据以降低遗忘 |
| checkpoint cadence | 便于选择最佳能力/遗忘折中点 |

常见做法是使用较低 learning rate、较短 warmup、明确的 held-out domain validation，并保留多个 checkpoint 做回退。

在实际训练里，CPT 的学习率不应只抄 pretraining 配方。更合理的起点是把它理解为“允许模型在既有表示上移动多少”，然后结合目标领域大小、数据质量和 replay 比例去缩小搜索范围。

## 遗忘与能力迁移

CPT 的核心风险是 catastrophic forgetting。领域数据过窄时，模型可能在目标领域表现提升，但通用语言、代码、数学、多语言或指令跟随能力下降。

缓解方式包括：

- 在领域数据中混入通用高质量 replay data；
- 控制 CPT token budget；
- 使用较小 learning rate；
- 分阶段逐步提高领域比例；
- 保留多领域 validation；
- 在后续 SFT/RLHF 前评估 base capability；
- 对关键 benchmark 做 contamination 检查。

能力迁移也可能是正向的。例如，代码 CPT 可能提升结构化推理，数学 CPT 可能改善符号操作，长上下文 CPT 可能提升多文档整合。但这类迁移依赖数据质量和模型容量，不能假设自动发生。

## 与 SFT 的区别

SFT 训练的是指令响应格式和 assistant 行为，数据通常是 prompt-response。CPT 训练的是语言建模或预训练式目标，数据可以是原始文档、代码、教材、论文、网页或混合语料。

简单说：

- CPT 更像“继续读书”；
- SFT 更像“学习如何按用户请求回答”。

更形式化地说，CPT 主要改变模型对文本分布的建模能力，SFT 主要改变模型在交互协议中的条件响应行为。CPT 做得好，通常会让 SFT 更容易；但 CPT 不能替代 SFT 的 instruction following、safety policy 和 assistant style。

## Agentic CPT

Agentic CPT 是把 agent 行为分布前移到 continued pretraining 阶段的一类做法。它的目标不是让模型直接学习某个最终 agent 产品的固定交互格式，而是在 SFT/RL 之前注入工具调用、规划、长程决策、轨迹理解和环境反馈利用等基础倾向。

与普通领域 CPT 相比，Agentic CPT 的数据不只是领域文档，而往往包含：

- planning 与 next-action prediction；
- reasoning + tool call 片段；
- 多步 action-observation 轨迹；
- 从既有轨迹重组出的 step-level decision data；
- 长上下文 agent trajectory。

这种路线的核心假设是：如果 base model 已经具备 agentic inductive bias，后续 [[training/post-training/sft|SFT]] 或 RL 就不必同时学习“如何作为 agent 行动”和“如何对齐专家轨迹”，从而降低后训练阶段的优化压力。[[sources/papers/2025-scaling-agents-via-continual-pre-training|Scaling Agents via Continual Pre-training]] 是这一方向的代表案例。

在 code agent 场景中，agentic CPT / mid-training 的关键不只是加入更多代码，而是让数据保留软件工程 agent 的工作流结构。典型数据形态包括从 PR 重构出的 contextually-native trajectories，以及从可执行仓库环境中采集的 environmentally-native trajectories。前者强调 issue、相关文件和 commit edits 的上下文连续性，后者强调工具调用、测试失败和修正反馈的真实性。[[sources/papers/2026-davinci-dev-agent-native-mid-training-for-software-engineering|daVinci-Dev]] 是这一方向的代表案例。

更完整的 coding-agent CPT / mid-training recipe 往往还需要配合可执行任务合成、长上下文仓库级代码、multi-scaffold trajectory、工具模板多样性和后续 SFT/RL。[[sources/papers/2026-qwen3-coder-next-technical-report|Qwen3-Coder-Next]] 展示了这种路线的工程化版本：先用 repository-level code、PR 数据、text-code grounding 与 multi-turn agentic trajectories 塑造 base，再用 verified trajectories、tool-format validation、execution-driven RL 和 expert distillation 对齐部署行为。

另一类 agentic mid-training 关注 internal world model。它不是只训练模型预测下一步 action，而是在轨迹中插入对未来路径的压缩摘要、当前信息缺口和成功概率估计，使模型在行动前形成 look-ahead planning 先验。后续 SFT 再把这种潜在能力结构化外显，RL 则用真实执行结果校准预测和 confidence。相关案例见 [[sources/papers/2026-internalizing-the-future-world-model-agentic-training|Internalizing the Future]]。

### DeepSeekMath：领域能力注入案例

DeepSeekMath 提供了一个非 agent 领域的能力注入案例：模型从 DeepSeek-Coder-Base-v1.5 7B 初始化，继续训练 500B tokens，其中包含 56% DeepSeekMath Corpus、4% AlgebraicStack、10% arXiv、20% GitHub code 和 10% 中英文 Common Crawl natural language。数学训练后，模型不仅在数学 benchmark 上提升，MMLU 和 BBH 也得到改善；同时加入 code tokens 帮助保持 coding performance。

这个案例支持两点较稳妥的认识。第一，NTP 形式的中间训练可以通过数据分布注入结构化能力先验，数据不一定要被包装成 SFT 对话格式。第二，能力迁移依赖初始化模型、数据质量、训练顺序和 mixture：code training 对 program-aided math reasoning 的迁移较明显，但 code/math 混合在小模型上会牺牲部分无工具数学 reasoning。因此，agentic mid-training 的数据设计也应通过跨域能力和 forgetting 的联合评测来确定，而不是预先假定某种固定配方。

## 与 Reinforcement Pretraining 的关系

传统 CPT 通常仍使用 next-token prediction。[[training/pretraining/reinforcement-pretraining|Reinforcement Pretraining]] 则在 continued pretraining 阶段改变目标函数。

例如 [[sources/papers/2025-rlp-reinforcement-as-a-pretraining-objective|RLP]] 会让模型在预测下一 token 前先生成 thought，并根据 thought 对真实 token log-likelihood 的提升获得 reward。它和 CPT 的差异不只是训练数据，而是训练信号：CPT 直接最大化 token likelihood，RLP 优化的是 thought 的信息增益。

## 评测

CPT 至少需要同时评估：

- CPT 目标领域 validation loss；
- 通用 validation loss；
- 关键 benchmark；
- 长上下文或领域专项 benchmark；
- 训练数据 contamination；
- 与原始 base model 的差异；
- 后续 SFT 后能力是否保留。

如果只看目标领域提升，容易忽略通用能力退化。如果只看 benchmark，容易被污染或过拟合误导。

## 常见失败模式

- **领域数据过窄**：模型变得更像领域语料补全器，而不是通用 base。
- **学习率过高**：短期 loss 下降，但已有能力被破坏。
- **没有 replay data**：通用能力和多语言能力快速下降。
- **数据污染**：目标 benchmark 被 CPT 数据泄漏。
- **合成数据比例过高**：模型继承 teacher 风格和错误。
- **把 CPT 当 SFT**：没有训练 assistant 行为，却期待模型会更会对话。

## 相关概念

- [[training/pretraining/pretraining|Pretraining]]
- [[training/pretraining/objective|Training Objective]]
- [[training/pretraining/reinforcement-pretraining|Reinforcement Pretraining]]
- [[training/mid-training/domain-adaptation|Domain Adaptation]]
- [[training/mid-training/capability-injection|Capability Injection]]
- [[training/mid-training/long-context-training|Long Context Training]]
- [[training/mid-training/annealing|Annealing]]
- [[training/pretraining/data-mix|Data Mix]]
