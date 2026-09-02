---
title: GPT
created: 2026-02-07
published: 2026-08-31
modified: 2026-08-31
type: topic
status: growing
area: architecture
tags:
  - architecture
  - model-family
  - gpt
  - openai
  - decoder-only
  - in-context-learning
---

GPT（Generative Pre-trained Transformer）是一条以 decoder-only Transformer 和 autoregressive language modeling 为核心的模型路线。它的重要性不只在于模型规模不断增长，还在于逐渐把“预训练一个通用模型”和“为每个任务单独设计模型”分开：同一个模型可以通过 prompt、任务格式和少量 demonstrations 承载多种任务。

## 演进主线

| 阶段 | 代表工作 | 主要变化 |
|---|---|---|
| GPT-1 | Generative Pre-Training | 用无监督 language modeling 预训练，再进行 task-specific fine-tuning |
| GPT-2 | Language Models are Unsupervised Multitask Learners | 扩大模型与数据，展示统一文本 continuation 对多任务的迁移 |
| GPT-3 | [[sources/papers/2020-language-models-are-few-shot-learners|Language Models are Few-Shot Learners]] | 175B dense LM 在不更新参数的情况下，通过 instruction 与 demonstrations 进行 in-context learning |
| GPT-3 之后 | GPT-4、GPT-4o 等 | 在更大模型、多模态、后训练、工具使用和产品化系统上继续演进 |

这条路线可以概括为：

```text
pre-training on broad text
  -> general language and knowledge prior
  -> prompt / demonstration as task interface
  -> instruction and tool-oriented post-training
  -> multimodal and agentic systems
```

## Decoder-only 基础

GPT 使用 decoder-only Transformer，通过 causal self-attention 让每个位置只能看到前缀，并学习：

$$
p_\theta(x_1,\ldots,x_T)=
\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t})
$$

训练时可以并行计算整段序列的 next-token loss，推理时则逐 token 生成。prompt、任务说明、demonstrations、工具描述和模型输出都可以被编码为同一条 token sequence；输入与输出的功能边界来自数据格式、特殊 token 和训练/评估 protocol，而不是来自不同的网络模块。

这使 GPT 路线具有三个长期优势：

- 目标统一：很多任务都能转成 sequence continuation；
- 接口统一：classification、QA、translation、code 和 tool-use 都可以通过文本格式表达；
- 扩展统一：模型规模、数据规模和 context window 可以在相同主干上持续扩大。

代价是，纯 autoregressive objective 对长距离比较、精确结构化推理、事实验证、goal-directed action 和双向读取并不总是最合适；能力仍然依赖数据、prompt、后训练和外部系统。

## GPT-3 的关键转折

GPT-3 训练了 125M 到 175B 的 8 个 dense models，最大模型包含 175B parameters、96 layers、12,288 hidden dimension、96 attention heads 和 2,048-token context。它在约 300B training tokens 上进行 pre-training，数据由 filtered Common Crawl、WebText2、Books1、Books2 和 Wikipedia 组成。

GPT-3 的重要观察是：

- zero-shot 只提供 task description；
- one-shot 提供一个 input-output demonstration；
- few-shot 提供尽可能多的 demonstrations，但不更新权重；
- larger models 往往更能利用 examples，few-shot curve 比 zero-shot curve 更陡；
- 任务格式、答案分隔符、示例顺序和 context budget 会直接影响结果。

这并不等同于模型在 inference 时通过梯度学习新任务。更谨慎的解释是，in-context learning 可能混合了 task recognition、format induction、已有知识调用和有限的 on-the-fly adaptation。不同任务在这个 spectrum 上的位置不同。

## Dense Scaling 与能力边界

GPT-3 延续 [[training/scaling/scaling-law|Scaling Law]] 的观察：language-model loss 随 model size 和 training compute 大体平滑改善；下游任务通常也改善，但曲线由任务结构决定。

规模提升最明显地帮助了：

- broad knowledge retrieval；
- natural language completion；
- translation pattern induction；
- prompt format following；
- 一些 arithmetic 和 synthetic symbolic tasks。

规模不能自动解决：

- sentence comparison 与 NLI；
- 长文档中的结构化对话行为；
- common-sense physics；
- 精确数值和离散推理；
- 长篇文本的一致性；
- factuality、calibration、bias 和安全边界。

因此，GPT 的 scaling 结论不应被理解为“参数量替代了所有任务机制”。它更像是把通用语言 prior 的上限推高，再由 task format、检索、工具、验证器和后训练继续塑形。

## GPT 与后续模型路线

GPT 的 decoder-only 设计成为 GPT、LLaMA、Qwen、Mistral 和许多其他现代 LLM 的共同基础。后续路线主要在几条轴上扩展：

| 扩展方向 | 解决的问题 |
|---|---|
| 更大 model/data/compute | 提升语言、知识和推理先验 |
| Instruction tuning | 让模型更稳定地遵循用户意图和输出格式 |
| Preference optimization / RL | 优化帮助性、安全性、任务成功与交互质量 |
| Multimodality | 为模型提供图像、音频、视频等额外 grounding |
| Tool use / agent systems | 将文本 continuation 连接到外部 action、observation 和环境反馈 |
| Retrieval / memory | 补充参数知识的时效性与长尾覆盖 |
| Inference-time scaling | 通过 sampling、search、verifier 和长思考提高单任务成功率 |

这些扩展不是对 GPT 主干的简单替换。decoder-only LM 提供统一 token interface，但 agent 的状态维护、工具协议、权限、安全、环境执行和长期信用分配仍需要模型之外的系统设计。

## 当前阅读重点

阅读 GPT 系列时，建议始终分开四个问题：

1. 模型是否只是提高了 next-token likelihood，还是在任务成功指标上提高？
2. 任务能力来自预训练分布、prompt demonstrations，还是后训练数据与 reward？
3. 规模带来的收益是 general prior，还是特定 benchmark 的记忆与格式适配？
4. 模型本身的生成能力与外部 retrieval、tool、verifier、memory 和 agent harness 如何分工？

这四个问题能够避免把“更大的 base LM”“更好的 assistant”“更强的 agent system”混成同一个对象。

## 相关概念

- [[architecture/transformer/decoder-only-transformer|Decoder-only Transformer]]
- [[training/pretraining/objective|Training Objective]]
- [[training/scaling/scaling-law|Scaling Law]]
- [[training/scaling/model-data-compute|Model, Data and Compute]]
- [[application/prompting/few-shot|Few-shot Prompting]]
- [[application/prompting/prompt-engineering|Prompt Engineering]]
- [[application/agents/|Agents]]
- [[application/tool-use/|Tool Use]]

## 经典论文与资料

- [[sources/papers/2020-language-models-are-few-shot-learners|Language Models are Few-Shot Learners]]
- [[sources/papers/2022-instructgpt|Training language models to follow instructions with human feedback]]
