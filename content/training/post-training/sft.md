---
title: SFT
created: 2026-03-01
published: 2026-03-01
modified: 2026-05-29
type: topic
status: mature
area: training
tags:
  - training
  - post-training
  - alignment
---

SFT，Supervised Fine-Tuning，指在预训练后的 language model 上，用人工编写、模型生成并筛选、专家标注或任务数据构造的 input-output pairs 继续进行有监督训练。对于大模型后训练来说，SFT 的目标不是让模型“获得全部知识”，而是把 base model 中已经形成的语言、知识和推理能力，重新组织成可交互的 assistant 行为。

SFT 是后训练中最基础但最容易被误解的一步。它既是 behavior shaping，也是 format alignment；既影响模型的回答风格、拒答边界、工具调用格式，也影响后续 [[training/post-training/rlhf|RLHF]]、[[training/post-training/dpo|DPO]]、[[training/post-training/grpo|GRPO]] 和蒸馏阶段的初始策略质量。

## 目标与问题

预训练模型通过 next-token prediction 学到的是“给定上下文，继续生成合理文本”。这种能力很强，但并不自动等于 assistant 能力：

- 用户希望模型回答问题，而不是续写一段网页、论文或聊天记录；
- 对话中有 system、user、assistant、tool 等不同角色，模型必须理解谁在发出约束、谁在提出请求、谁应该输出结果；
- 很多任务要求固定格式，例如 JSON、SQL、代码 patch、表格、工具调用参数；
- 模型需要学习“什么时候回答、什么时候追问、什么时候拒答、什么时候调用工具”；
- 预训练语料中的风格、身份、价值判断和错误内容可能不适合作为部署 assistant 的默认行为。

SFT 用高质量示例给模型一个可模仿的行为分布。它通常回答的问题是：**在这一类 prompt 下，一个理想 assistant 的回答应该长什么样？**

## 数据与输入

SFT 数据常见形态包括：

| 数据类型 | 典型形式 | 作用 |
|---|---|---|
| Single-turn instruction | `instruction -> response` | 建立基本指令跟随 |
| Multi-turn conversation | 多轮 `user/assistant` 对话 | 建立上下文记忆和对话连贯性 |
| System-conditioned chat | `system + user -> assistant` | 学习角色优先级和行为边界 |
| Tool-use traces | `assistant -> tool call -> tool result -> assistant` | 学习函数调用和工具结果整合 |
| Domain data | 医学、法律、代码、数学、客服等领域样本 | 注入专业任务格式和领域规范 |
| Reasoning traces | answer with rationale / CoT / scratchpad | 强化逐步推理或可解释求解路径 |
| Refusal / safety examples | unsafe request -> refusal / safe completion | 训练拒答与安全替代回答 |

后训练 SFT 的关键不是“数据越多越好”，而是数据分布是否覆盖目标产品行为。低质量 SFT 数据会造成非常直接的行为污染：模板错误会变成生成错误，冗长示例会变成冗长偏好，错误 reasoning traces 会变成稳定幻觉。

## Chat Template 与 Loss Masking

SFT 训练时通常先把结构化对话转换成 token 序列。以 chat model 为例，一个样本可能包含：

```text
<bos><system>
You are a helpful assistant.
</system>
<user>
Explain KL divergence.
</user>
<assistant>
KL divergence measures how one probability distribution differs from another...
</assistant><eos>
```

实际 token 格式取决于 tokenizer 和模型家族。关键约束是：训练时的 [[training/post-training/chat-template|Chat Template]] 必须和推理时模板一致。模板不一致会导致模型在错误位置预测 assistant 内容，或者把用户消息、特殊 token、工具结果当作可生成文本。

SFT loss 通常只计算 assistant response tokens：

$$
\mathcal{L}_{\text{SFT}}(\theta)
= - \sum_{t \in \mathcal{A}} \log \pi_\theta(y_t \mid x, y_{<t})
$$

其中 $\mathcal{A}$ 表示 assistant tokens 的位置集合。System prompt、user prompt、tool observation 等上下文 tokens 参与 attention，但通常不参与 loss。这样做的原因是：模型应该利用这些 tokens 生成回答，但不应该被训练去“复述用户消息”或“生成 system 指令”。

## 训练目标

SFT 本质上是 conditional maximum likelihood：

$$
\max_\theta \mathbb{E}_{(x,y)\sim \mathcal{D}}
\left[\log \pi_\theta(y \mid x)\right]
$$

其中 $x$ 是 prompt / conversation context，$y$ 是理想 assistant response。实现上通常使用 cross-entropy loss：

$$
\mathcal{L}_{\text{CE}} = - \frac{1}{N}\sum_{i=1}^{N}\sum_{t=1}^{T_i}
m_{i,t}\log p_\theta(y_{i,t}\mid y_{i,<t}, x_i)
$$

$m_{i,t}$ 是 loss mask，决定该 token 是否计入训练。SFT 的优化信号是 token-level 的：模型被鼓励复现参考回答中的每个 token。因此，SFT 对格式、措辞、长度和风格非常敏感。

## 训练流程

一个标准 SFT 流程通常包括：

1. **收集和定义任务数据**：确定目标 assistant 行为、任务范围、语言、领域和安全边界。
2. **转换为统一 conversation schema**：明确 `system`、`user`、`assistant`、`tool` 等角色字段。
3. **应用 chat template**：将结构化样本序列化为模型 tokenizer 可识别的输入。
4. **构造 loss mask**：只在 assistant 可学习输出上计算 loss。
5. **训练模型**：全参微调或参数高效微调，如 LoRA / QLoRA。
6. **评估行为与格式**：检查指令跟随、拒答、格式稳定性、长上下文、多轮对话和领域任务。
7. **数据迭代**：根据失败样例补充数据，去除污染样本，调整 mixture。

## LoRA / QLoRA 场景

在资源受限时，SFT 常用 LoRA 或 QLoRA。LoRA 冻结 base model 参数，只在部分线性层注入低秩适配矩阵；QLoRA 进一步把 base model 量化后进行低秩训练。它们适合领域微调、个人 assistant、企业私有数据适配和小规模实验。

但 LoRA / QLoRA 并不自动解决数据问题。小规模 SFT 很容易出现：

- 只学到表面措辞，没真正提升任务能力；
- 对少数格式过拟合，泛化到新 prompt 变差；
- 与原模型安全策略冲突；
- 多 adapter 合并后行为不可预测；
- 低质量数据被模型“牢牢记住”。

如果目标是改变核心能力或大规模对齐，全参 SFT、继续预训练和偏好优化仍然更常见。

## 数据混合与质量控制

SFT 数据 mixture 决定模型行为的默认先验。常见混合维度包括：

- **任务类型**：问答、写作、总结、翻译、代码、数学、工具调用；
- **难度层级**：简单指令、多约束指令、长上下文、多步推理；
- **语言与地域**：中文、英文、多语言、专业术语；
- **安全与拒答**：正常回答、安全改写、明确拒答、边界案例；
- **风格**：简洁、学术、工程、客服、代码审查；
- **数据来源**：人工标注、专家写作、teacher model 生成、rejection sampling 筛选。

高质量 SFT 数据通常具备四个特征：instruction 明确，response 正确，格式稳定，行为边界一致。相比之下，来源混乱、模板不一致、答案错误、语言风格摇摆的数据会让 SFT 变成行为噪声注入。

## 与其他后训练方法的关系

SFT 通常是后续偏好优化的起点：

- [[training/post-training/reward-model|Reward Model]] 需要基于一个能生成合理候选的 SFT model 收集 preference data；
- [[training/post-training/rlhf|RLHF]] 如果从太弱的 policy 开始，rollout 质量低，reward learning 和 policy optimization 都会更困难；
- [[training/post-training/dpo|DPO]] 依赖 chosen/rejected pairs，常以 SFT model 作为 reference；
- [[training/post-training/grpo|GRPO]] 在 reasoning 任务中也通常需要一个至少会按格式解题的初始模型；
- [[training/post-training/knowledge-distillation|Distillation]] 可以把强 teacher 生成的 SFT-style answers 固化到 student 中。

因此，SFT 的作用不是取代 RLHF / DPO，而是提供一个可优化、可采样、可评估的 assistant policy。

## 失败模式与边界

### 格式过拟合

如果训练数据总是使用固定标题、固定口吻或固定长度，模型会把这些表面模式当作任务本身。表现为：无论用户要求多简单，模型都输出长篇结构化回答；或者无论任务是否需要，都会添加免责声明。

### 错误答案蒸馏

SFT 是 imitation learning，不知道参考答案是否真实。如果 teacher 生成了错误解释，student 会把错误当作标准答案学习。对数学、代码、医学、法律等高风险领域尤其危险。

### 安全边界不稳定

拒答数据太多会造成 over-refusal；拒答数据太少会造成 unsafe completion。安全 SFT 需要覆盖边界案例，而不是只收集明显有害或明显无害样本。

### 能力上限受 base model 限制

SFT 很难凭空创造 base model 不具备的知识和推理能力。它更擅长“调动”和“格式化”已有能力，而不是替代大规模预训练或 continued pretraining。

### 分布漂移

如果训练 prompt 与真实用户 prompt 差异很大，SFT 后的行为会在真实场景中失效。真实用户常常含糊、多目标、带上下文、带错误假设，而合成 SFT 数据往往过于干净。

## 评估指标

SFT 评估不应只看 validation loss。常见评估包括：

- 指令跟随成功率；
- 格式遵守率，如 JSON validity、tool call schema validity；
- 多轮一致性；
- 专项 benchmark，如代码、数学、知识问答；
- 人工偏好胜率；
- 安全拒答与不过度拒答；
- 长度分布和 verbosity；
- 与 base model 的能力回归检查。

Validation loss 下降但人工偏好变差并不罕见，因为 token-level imitation 与用户偏好并不完全一致。

## 经典论文与资料

- [[sources/papers/2022-instructgpt|Training language models to follow instructions with human feedback]]
- [[sources/papers/2021-flan|Finetuned Language Models Are Zero-Shot Learners]]
- [[sources/papers/2021-t0|Multitask Prompted Training Enables Zero-Shot Task Generalization]]
- [[sources/papers/2022-self-instruct|Self-Instruct]]

## 相关概念

- [[training/post-training/instruction-tuning|Instruction Tuning]]
- [[training/post-training/chat-template|Chat Template]]
- [[training/post-training/rlhf|RLHF]]
- [[training/post-training/dpo|DPO]]
- [[training/post-training/rejection-sampling|Rejection Sampling]]
- [[fundamentals/probability/maximum-likelihood|Maximum Likelihood]]
