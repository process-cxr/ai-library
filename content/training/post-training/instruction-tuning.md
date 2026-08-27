---
title: Instruction Tuning
created: 2026-03-01
published: 2026-03-01
modified: 2026-05-29
type: topic
status: mature
area: training
tags:
  - post-training
  - instruction-tuning
---

Instruction Tuning 是用自然语言指令和对应回答对模型进行有监督微调，使模型学会按照用户意图执行任务。它通常是 [[training/post-training/sft|SFT]] 的一个重要子集或扩展：SFT 强调有监督训练形式，Instruction Tuning 强调训练数据以 instruction 为核心，并希望模型把“理解指令并执行任务”泛化到未见过的任务。

在大模型发展史中，Instruction Tuning 是从 base language model 走向 assistant model 的关键步骤。它把预训练中隐式存在的任务能力显式化：模型不再只是续写文本，而是根据“请总结”“请翻译”“请写代码”“请比较”“请按 JSON 输出”等语言指令选择合适行为。

## 目标与问题

Instruction Tuning 要解决的是 task interface 问题。预训练模型可能已经见过大量问答、代码、解释、推理文本，但用户在真实交互中不会提供完整任务分布，只会写一条自然语言请求。模型必须从请求中推断：

- 任务类型是什么；
- 输入和输出分别在哪里；
- 输出应该使用什么格式；
- 是否需要推理、检索、计算、工具调用或拒答；
- 多个约束发生冲突时如何排序；
- 用户没有说清楚时是否需要追问。

这使 Instruction Tuning 不只是“多做一些 SFT”，而是在训练模型识别和执行任务 specification。

## 与 SFT 的关系

二者关系可以这样理解：

| 维度 | SFT | Instruction Tuning |
|---|---|---|
| 训练形式 | 有监督 token-level imitation | 通常也是有监督 token-level imitation |
| 关注重点 | 学习目标 response 分布 | 学习自然语言指令到任务行为的映射 |
| 数据形态 | 可以是对话、领域 QA、工具轨迹、拒答等 | 以 instruction / task prompt 为中心 |
| 主要目标 | 建立 assistant 行为和格式 | 提升任务泛化与 zero-shot 指令跟随 |
| 典型论文 | InstructGPT SFT stage | FLAN、T0、Self-Instruct |

因此，Instruction Tuning 可以视为 SFT 的任务泛化版本；而 SFT 可以包含 instruction tuning、chat tuning、tool tuning、安全 tuning 和领域 tuning 等多种数据。

## 数据格式

Instruction 数据通常至少包含三个字段：

```json
{
  "instruction": "Summarize the following article in three bullet points.",
  "input": "...",
  "output": "..."
}
```

在 chat model 中，它会进一步转换成 conversation schema：

```json
[
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "Summarize the following article in three bullet points.\n\n..."},
  {"role": "assistant", "content": "..."}
]
```

Instruction 数据的难点不在字段本身，而在 specification 的准确性。一个好 instruction 应该让目标任务可判定、约束清晰、输入边界明确。模糊 instruction 会迫使模型从训练集风格中猜测意图，从而降低泛化性。

## 任务混合

Instruction Tuning 通常依赖 task mixture。常见任务包括：

- 分类、抽取、匹配、改写、翻译、摘要；
- 开放问答、知识解释、教学式回答；
- 数学解题、代码生成、代码解释、调试；
- 信息结构化，如 JSON、YAML、表格；
- 多轮对话和上下文追踪；
- 安全拒答、价值冲突、敏感内容处理；
- 创意写作、语气转换、角色扮演；
- 工具调用和 function calling。

任务混合的核心原则是平衡覆盖面与质量。过窄的 mixture 会让模型只在少数任务上变好；过宽但质量差的 mixture 会让模型学到浅层模板。FLAN 和 T0 类工作的重要启示是：跨任务、多模板、多数据集训练可以显著提升 zero-shot task generalization，但收益依赖任务表述与答案质量。

## 泛化机制

Instruction Tuning 的泛化来自三个层次：

1. **语义对齐**：模型学会将自然语言动词、约束和输入边界映射到任务操作。
2. **格式对齐**：模型学会按 instruction 要求输出列表、JSON、代码、解释或短答案。
3. **任务抽象**：模型从许多具体任务中归纳出更抽象的“任务族”，例如 classification、generation、ranking、reasoning。

这也是为什么多模板训练有价值。同一个任务如果只用一种 prompt 表达，模型可能记住表面模式；如果用多种自然语言表达，模型更可能学习任务意图本身。

## 合成指令数据

Self-Instruct 一类方法使用少量 seed tasks 启动递归 task pool，让模型自动生成 instruction、input 和 output，再经过去重、冲突处理和启发式过滤构造 instruction tuning 数据。这条路线降低了人工标注成本，也成为很多开放模型构建指令数据的基础思想。它的关键不只是生成更多 instruction：classification task 需要采用 output-first 的生成顺序控制 label coverage，non-classification task 则可以采用 input-first；论文对 200 条抽样数据的检查显示，instruction 有效率为 92%，但 instruction、input、output 全部有效的比例只有 54%，说明实例与答案质量往往是主要瓶颈。

合成数据的优势是覆盖广、成本低、迭代快；风险是 teacher bias、重复模板、事实错误和任务难度虚高。高质量合成 instruction tuning 通常需要：

- 去重和模板多样性检查；
- 事实性和可执行性过滤；
- 难度分层；
- 与人工或高质量公开数据混合；
- 对输出格式和安全边界进行验证。

Self-Instruct 在 GPT-3 上的实验还显示，生成数据规模增加带来的收益会逐渐趋于 plateau，而用更强模型重新生成 output 可以带来约 10% 的额外提升。对合成 instruction data 的处理，应同时记录 seed pool、生成模型、prompt template、过滤规则和数据版本，分别评估任务覆盖、实例正确性与下游 instruction-following，而不能只按生成 token 数衡量数据价值。详见 [[sources/papers/2022-self-instruct|Self-Instruct]]。

## 关键超参与工程点

- **Learning rate**：SFT / instruction tuning 通常比预训练学习率小，避免破坏 base model 能力。
- **Epochs**：小数据多 epoch 容易过拟合风格；大数据单 epoch 或少 epoch 更常见。
- **Sequence length**：长 instruction 和长 response 会改变训练成本和位置分布。
- **Loss mask**：通常只训练 assistant 输出，不训练 user instruction。
- **Data weighting**：高价值任务可以加权，但过度加权会扭曲默认行为。
- **Template consistency**：训练和推理的 [[training/post-training/chat-template|Chat Template]] 必须一致。

## 失败模式与边界

### 指令表面化

模型学会“看到请解释就输出一段解释”，但没有真正遵守具体约束。例如用户要求“三句话以内”，模型仍输出长篇内容；要求“只输出 JSON”，模型附加自然语言解释。

### 多任务负迁移

某些任务格式会污染其他任务。例如大量 chain-of-thought 数据可能让模型在不需要解释时也输出冗长推理；大量拒答数据可能导致正常问题被过度拒答。

### 合成数据回音室

如果 teacher model 生成的数据缺乏多样性，student 会继承 teacher 的措辞、偏见和错误。大量合成数据并不等于大量新信息。

### 评测集污染

Instruction tuning 数据常来自公开 benchmark 或其变体，容易造成评测分数虚高。需要保持 held-out set 和真实用户样本。

## 经典论文与资料

- [[sources/papers/2021-flan|Finetuned Language Models Are Zero-Shot Learners]]
- [[sources/papers/2021-t0|Multitask Prompted Training Enables Zero-Shot Task Generalization]]
- [[sources/papers/2022-self-instruct|Self-Instruct]]
- [[sources/papers/2022-instructgpt|Training language models to follow instructions with human feedback]]

## 相关概念

- [[training/post-training/sft|SFT]]
- [[training/post-training/chat-template|Chat Template]]
- [[training/data-engineering/synthetic-data|Synthetic Data]]
- [[training/post-training/rejection-sampling|Rejection Sampling]]
