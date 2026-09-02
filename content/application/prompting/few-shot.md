---
title: Few-shot Prompting
created: 2026-05-02
published: 2026-08-31
modified: 2026-08-31
type: topic
status: growing
area: application
tags:
  - prompting
  - few-shot
  - in-context-learning
---

Few-shot Prompting 是在输入上下文中提供少量 input-output demonstrations，让模型在不更新参数的情况下，按照示例完成后续任务。它是 [[architecture/model-families/gpt|GPT]] 路线中 `in-context learning` 的典型使用方式，也是把一个通用 decoder-only language model 临时适配到特定任务、格式或风格的最轻量方法。

## 基本形式

```text
Task description

Example 1 input: ...
Example 1 output: ...

Example 2 input: ...
Example 2 output: ...

Query input: ...
Query output:
```

模型只接收这段 token sequence，不进行 gradient update。它需要从上下文中识别任务、推断输入输出关系、模仿输出格式，并继续生成 query 的答案。

可以把一次 few-shot inference 理解为：

```text
pretrained model prior
  + task description
  + demonstrations
  + query
  -> conditional continuation
```

这里的 `learning` 不表示参数发生更新。更谨慎的说法是，模型利用当前 context 完成 task recognition、format induction、已有知识调用以及一定程度的 on-the-fly adaptation。

## 与其他设置的区别

| 设置 | 是否更新参数 | 上下文示例 | 典型特点 |
|---|---:|---:|---|
| Zero-shot | 否 | 0 个 | 只提供 instruction 或 task description |
| One-shot | 否 | 1 个 | 成本低，但单个示例可能不足以确定规则 |
| Few-shot | 否 | 多个 | 通过多个示例明确任务规律和输出格式 |
| Fine-tuning | 是 | 通常大量 | 任务能力写入 model weights，训练成本更高 |

Few-shot 与 fine-tuning 的主要差异不只是 examples 数量，而是适配信号存放的位置不同：few-shot 的适配暂时存在 context 与 activations 中，fine-tuning 则改变 parameters。因而 few-shot 不需要新 checkpoint，却会为每次请求重复支付 demonstration 的输入 token 成本。

## GPT-3 的关键证据

[[sources/papers/2020-language-models-are-few-shot-learners|Language Models are Few-Shot Learners]] 系统比较了 zero-shot、one-shot 与 few-shot。GPT-3 最大为 175B parameters、context window 为 2,048 tokens，不对测试任务进行 gradient update。论文观察到：

- larger models 往往更能利用 demonstrations；
- few-shot performance 的提升通常比 zero-shot 更快；
- 适量 examples 可以同时传递任务规则与答案格式；
- demonstrations 并非总是带来收益，one-shot 甚至可能低于 zero-shot；
- 不同任务对 examples 数量、格式和顺序的敏感性不同。

代表性结果包括：

| Task | Zero-shot | One-shot | Few-shot |
|---|---:|---:|---:|
| LAMBADA accuracy | 76.2 | 72.5 | 86.4 |
| TriviaQA | 64.3 | 68.0 | 71.2 |
| Natural Questions | 14.6 | 23.0 | 29.9 |
| CoQA F1 | 81.5 | 84.0 | 85.0 |
| 2-digit addition | 76.9 | 99.6 | 100.0 |
| 3-digit addition | 34.2 | 65.5 | 80.4 |

这些结果支持 few-shot prompting 具有真实行为价值，但不证明模型完全从零学会了任务。论文指出，模型可能是在识别预训练中已经见过的 task distribution，也可能对部分 synthetic tasks 进行较新的规则适应；不同任务应分别分析。

## Demonstration 的作用

### Task recognition

示例让模型知道当前输入应该被当成什么任务。例如，同一个文本可以被要求做分类、摘要、翻译、改写或抽取；input-output pair 把抽象要求变成可观察的条件分布。

### Format induction

示例不仅说明“答案是什么”，还说明答案应如何写：输出标签是 `A/B/C` 还是自然语言，是否需要 JSON，是否只输出一个 token，是否在答案后继续解释。GPT-3 在 LAMBADA 上的 few-shot 提升说明，示例可以把普通 continuation 重新 framing 成 exact fill-in-the-blank task。

### Knowledge and behavior composition

示例可以把已有知识与临时行为规则组合起来。例如模型可能已经知道一个事实，但需要通过 demonstrations 了解应当输出短答案、引用证据、按字段组织或使用某个 API schema。

### Local adaptation

对于未在训练中明确出现的 synthetic rule，示例可能提供局部变换规则，例如字符重排、虚构词定义或自定义分类标准。但这种适应受 context 长度、规则复杂度和训练分布相似性限制。

## 如何设计 demonstrations

### 先保证格式一致

所有 demonstrations 应尽量共享：

- input/output 分隔符；
- label vocabulary；
- 空格、换行和 punctuation；
- 答案长度与结束边界；
- 是否包含 reasoning 或 explanation。

格式不一致时，模型可能学到多个互相竞争的 continuation pattern，而不是任务规则本身。

### 覆盖任务变化

少量 examples 不应只展示同一种表面模式。分类任务至少覆盖不同 label；抽取任务覆盖字段缺失、长短输入和边界情况；工具调用任务覆盖不同 argument、错误条件和终止形式。否则模型可能依赖 example 的表面词汇，而没有学到可迁移规则。

### 选择有代表性的 examples

示例选择应同时考虑：

- 与 query 的语义或结构相似度；
- 任务类别覆盖；
- 难度与边界案例；
- 输出格式稳定性；
- 是否可能泄漏 query 答案。

“最相似”不一定等于“最有教学价值”。一组高度相似但缺乏变化的 examples 可能让模型过拟合局部模板；一组覆盖边界条件的 examples 可能更能约束任务规则。

### 控制顺序与随机性

模型对 demonstration order 可能敏感。评测时应固定或随机化顺序，并报告多次抽样的均值和方差。若只使用一组人工挑选的顺序，很难判断收益来自任务能力还是 prompt 偶然性。

### 在 context budget 内分配 token

更多 examples 通常会消耗更多 context，减少 query 可用空间。应将 token 预算分成：

```text
system / task description
  + demonstrations
  + query context
  + expected output budget
```

对长文档、长轨迹和 tool-use 任务，不能为了堆 examples 而挤掉真正决定答案的 context。可以比较固定 $K$、token-budget sampling、短示例与长示例等策略。

## 评测注意事项

### Prompt 是评测协议的一部分

Few-shot score 同时受模型和 protocol 影响。至少应记录：

- demonstration 来源与是否随机抽样；
- $K$ 值及总 input tokens；
- example 顺序；
- task instruction 与 delimiters；
- decoding parameters；
- scoring 是 likelihood、exact match、F1 还是 judge model。

如果不同模型使用不同 prompt 或不同 context budget，结果不宜直接比较。

### Multiple-choice 的 label bias

把选项写成 `A/B/C/D` 后，模型的 token prior 可能影响结果。GPT-3 论文通常比较各 completion 的 per-token likelihood，并在 ARC、OpenBookQA 和 RACE 上尝试用 generic answer context 做 unconditional probability normalization。

因此，multiple-choice 评测需要区分：

- completion 的语义匹配；
- 答案长度；
- label token 的先验频率；
- task prompt 与 demonstration 的格式效应。

### 不要只看单次结果

Few-shot demonstrations 常从 training set 随机抽取，单次结果会有 sampling variance。更可靠的评测应使用多个 seeds 或多组 demonstration，报告均值、标准差和最差/最好结果。对于长任务，还应按 context length、任务难度、输出长度和 failure type 分桶。

### 需要防止数据污染

如果 demonstrations 或 query 与预训练数据有重叠，few-shot gain 可能混合了 in-context adaptation 与 memorization。GPT-3 论文用 n-gram overlap 构造 clean subset，但也指出 web-scale corpus 中 contamination 检测存在 false positive、background passage overlap 和 clean subset 分布变化等问题。

## 在 Agent 与长轨迹中的使用

Few-shot prompting 可以把 agent 的工具协议、action schema 和行为约束放入 context：

```text
task / state
  + tool schema
  + prior action -> observation examples
  + current query
  -> next action / tool call
```

但这与训练 agentic capability 不同。示例只在当前请求中提供行为先验，不会把工具选择、错误恢复或长程状态跟踪写入模型参数。对于长轨迹任务，还要面对：

- demonstrations 占用本就有限的 context；
- example trajectory 可能与当前 environment 不同；
- 工具 schema 或 observation 格式变化会造成 distribution shift；
- 过度模仿示例可能复制错误 action；
- 只展示成功轨迹，模型不一定学会 recovery 与 termination。

因此，agent few-shot prompt 更适合作为 inference-time protocol adaptation，训练数据仍需要覆盖多环境、多工具、成功与失败分支，并通过执行结果验证 action 是否有效。

## 常见误解

### Few-shot 等于小规模 fine-tuning

不等于。Few-shot 不更新 model weights；它可能产生类似适配效果，但存储位置、成本、稳定性和可迁移范围都不同。

### Example 越多越好

不一定。examples 会消耗 context、引入噪声与冲突，并可能挤掉 query 的关键证据。应按 token budget 与任务收益选择，而不是只追求最大 $K$。

### Few-shot 答对说明模型掌握了通用规则

不充分。模型可能利用了训练集记忆、表面格式或与测试分布相近的 pattern。需要 synthetic、unseen-format、counterfactual 和 contamination-controlled 评测。

### Few-shot 能替代后训练

不能。Few-shot 可以临时传递 instruction、format 和少量行为样例，但不能稳定形成产品级的安全边界、工具协议、长程策略和任务成功能力。

### In-context learning 一定是真正的学习

机制上仍有开放问题。它可能包含 task recognition、retrieval-like matching、activation-level computation 和有限的 rule induction；仅凭最终输出无法区分这些机制。

## 相关知识链接

- [[architecture/model-families/gpt|GPT]]
- [[architecture/transformer/decoder-only-transformer|Decoder-only Transformer]]
- [[application/prompting/prompt-engineering|Prompt Engineering]]
- [[application/prompting/chain-of-thought|Chain of Thought]]
- [[application/tool-use/tool-calling|Tool Calling]]
- [[application/agents/workflow-agent|Workflow Agent]]
- [[training/pretraining/objective|Training Objective]]
- [[training/scaling/scaling-law|Scaling Law]]
- [[sources/papers/2020-language-models-are-few-shot-learners|Language Models are Few-Shot Learners]]
