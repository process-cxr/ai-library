---
title: "Distilling Step-by-Step"
created: 2026-05-29
published: 2026-08-27
modified: 2026-08-31
type: source
status: processed
source_type: paper
area: sources
tags:
  - source
  - paper
  - distillation
  - reasoning
  - rationale
source_url: https://arxiv.org/abs/2305.02301
paper_date: "2023-05"
paper_order: "02301"
---

# Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes

## 基本信息

- 论文：[Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes](https://arxiv.org/abs/2305.02301)
- 作者：Cheng-Yu Hsieh, Chun-Liang Li, Chih-Kuan Yeh, Hootan Nakhost, Yasuhisa Fujii, Alexander Ratner, Ranjay Krishna, Chen-Yu Lee, Tomas Pfister
- 机构：University of Washington、Google Cloud AI Research、Google Research
- 首次公开：2023-05，本文阅读版本为 arXiv v2（2023-07-05）
- 代码：[google-research/distilling-step-by-step](https://github.com/google-research/distilling-step-by-step)
- 相关知识：[[training/post-training/knowledge-distillation|Knowledge Distillation]]、[[training/post-training/sequence-level-distillation|Sequence-level Distillation]]、[[training/post-training/offline-kd|Offline KD]]、[[training/post-training/instruction-tuning|Instruction Tuning]]

这篇论文研究的是一个很实际的模型压缩问题：如果只把大模型生成的最终 label 当作 pseudo-label，student 需要从输入和标签之间自己推断任务规律，往往需要大量数据。论文提出从大模型中额外提取自然语言 rationale，把 rationale 作为 student 的辅助监督，让小模型在训练时同时学习“答案是什么”和“为什么得到这个答案”。

论文的关键设计不是让 rationale 成为推理时必须提供的额外输入，而是把它变成一个 auxiliary generation task。Student 训练阶段同时学习 `label` 和 `rationale`；部署时只执行 `label` 预测，因此不需要在推理前再次调用 teacher 或先生成 rationale。这种训练时增加监督、推理时保持单任务预测的设计，是全文最值得记住的部分。

## 研究问题

### 大模型能力与部署成本之间的矛盾

Large Language Models 具有较强的 few-shot 能力，但参数规模带来显存、吞吐和延迟成本。实际应用通常希望使用较小的 task-specific model，以降低部署成本；但标准 fine-tuning 或普通 task distillation 往往需要大量训练数据才能追上大模型。

论文比较了两类常见方案：

- **Standard fine-tuning**：使用人工标注的 input-label 数据训练小模型；
- **Standard task distillation**：让大模型为 unlabeled inputs 生成 pseudo-label，再用这些 label 训练小模型。

两种方法都只把最终 label 作为监督。论文的出发点是：大模型生成 label 时，往往也能通过 Chain-of-Thought prompting 给出中间 rationale，而这些 rationale 可能包含小模型需要大量样本才能归纳出的 task knowledge。

### 为什么最终 label 可能不是足够的监督

例如，一个数学题的答案可以是一个数字，但从题目到数字之间还需要面积公式、已知量代入和运算过程。若只训练：

```text
question -> answer
```

student 必须从许多样本中间接学习这些中间规则。若将 teacher 的 rationale 也纳入训练：

```text
question -> rationale
question -> answer
```

student 可以通过更丰富的语言信号建立输入、任务知识和最终标签之间的连接。论文将这种额外信息称为 richer task knowledge，但它仍然是 teacher 生成的文本监督，并不等价于 teacher 内部真实、完整的 reasoning process。

## 核心方法

### 总体流程

论文方法包含两步：先用 LLM 生成 rationale 和 label，再用 rationale 与 label 共同训练小模型。

```text
unlabeled input x
      |
      v
few-shot CoT prompt + x
      |
      v
teacher generates rationale r_hat and label y_hat
      |
      +------------------+
      |                  |
      v                  v
 student predicts     student predicts
 rationale            label
      |                  |
      +--------+---------+
               v
     multi-task training
               |
               v
      inference: label only
```

论文的默认 teacher 是 540B PaLM。为研究 teacher 规模影响，作者还使用 20B GPT-NeoX 生成 rationale。Student 使用 T5-Base（220M）、T5-Large（770M）和 T5-XXL（11B）等不同规模模型。

### Teacher-side rationale extraction

对每个 unlabeled input $x_i$，作者构造包含 rationale demonstration 的 few-shot CoT prompt。每个 demonstration 是一个三元组：

$$
p = (x_p, r_p, y_p)
$$

其中 $x_p$ 是示例输入，$r_p$ 是人工提供的解释该输入为何属于某个类别的 rationale，$y_p$ 是对应 label。将新输入拼接到 prompt 后，teacher 生成：

$$
(\hat r_i, \hat y_i) = \mathrm{Teacher}(p, x_i)
$$

这些 rationale 不是由人工逐条标注，而是由 few-shot CoT 诱导出来。论文要求用户为每个 task 准备少量示例 demonstrations，文中称约 10-shot；这也是方法仍然需要 task-specific prompt engineering 的地方。

### Standard fine-tuning 与 task distillation 基线

论文把标准训练形式统一写成 label prediction loss：

$$
\mathcal{L}_{label} = \frac{1}{N}\sum_{i=1}^{N} \ell(f(x_i), \hat y_i)
$$

当有人工标注时，$\hat y_i$ 实际上是 ground-truth $y_i$；当只有 unlabeled data 时，$\hat y_i$ 是 teacher 生成的 pseudo-label。这样可以直接比较：在 label 监督不变的情况下，增加 rationale supervision 是否带来额外收益。

### Multi-task training with rationales

论文没有把 rationale 作为 student 的输入。那样的形式是：

$$
f(x_i, \hat r_i) \rightarrow \hat y_i
$$

虽然可能提升预测效果，但推理阶段也需要先获得 rationale，仍然依赖 LLM teacher，不能得到真正独立部署的小模型。

论文采用 multi-task learning：让同一个 student 根据不同的 task prefix，分别预测 label 或 rationale：

```text
[label]     + x_i -> y_hat_i
[rationale] + x_i -> r_hat_i
```

总损失为：

$$
\mathcal{L} = \mathcal{L}_{label} + \lambda \mathcal{L}_{rationale}
$$

其中：

$$
\mathcal{L}_{rationale} = \frac{1}{N}\sum_{i=1}^{N} \ell(f(x_i), \hat r_i)
$$

`[label]` 和 `[rationale]` 是 T5 text-to-text 框架中的 task prefixes。训练时，student 需要在同一个输入上完成两个相关但不同的生成任务；推理时只提供 `[label]` prefix，因此无需生成 rationale，也不会增加线上推理链路。

这个结构隐含了一个重要判断：rationale 的价值主要来自训练过程中提供更密集、更有结构的学习信号，而不一定需要在最终预测时显式暴露给模型。

### 为什么不是简单拼接 rationale 与 label

另一种直观方案是把 rationale 和 label 串成一个目标序列：

```text
x_i -> [r_hat_i, y_hat_i]
```

对应损失可以写为：

$$
\mathcal{L}_{single} = \frac{1}{N}\sum_{i=1}^{N} \ell(f(x_i), [\hat r_i, \hat y_i])
$$

论文实验显示，单一序列目标并不稳定，有时甚至低于 standard fine-tuning。原因可能包括：rationale 较长，训练梯度被大量 rationale tokens 占据；label 位于序列后部，模型可能更关注生成流畅解释而不是准确预测；rationale 错误还会直接污染同一条目标序列。

Multi-task 方案通过 task prefix 分离两个目标，使 label prediction 和 rationale generation 共享表示、但分别计算监督，更容易控制各自的训练作用。

## 实验设置

### 数据集

论文在四个 NLP benchmark 上实验：

- **e-SNLI**：带自然语言解释的 natural language inference；
- **ANLI**：对抗式 natural language inference，任务难度更高；
- **CQA**：CommonsenseQA，常识问答；
- **SVAMP**：算术 math word problems。

训练、验证和测试规模如下：

| Dataset | Train | Validation | Test |
|---|---:|---:|---:|
| e-SNLI | 549,367 | 9,842 | 9,824 |
| ANLI | 16,946 | 1,000 | 1,000 |
| CQA | 8,766 | 975 | 1,221 |
| SVAMP | 720 | 80 | 200 |

实验分为两种数据条件：有 human labels 时比较 standard fine-tuning 与 Distilling step-by-step；只有 unlabeled inputs 时比较 standard task distillation 与 Distilling step-by-step。SVAMP 还使用 ASDiv 的 2,305 个 math word problems 做 unlabeled data augmentation。

### 训练配置

实验使用 A100 × 16 GPU。T5-Base 和 T5-Large 使用 learning rate $5\times10^{-5}$、batch size 64、最大 input length 1024，最多 10,000 steps；T5-XXL 使用 batch size 32，最多 4,000 steps。结果在 4 个 random runs 上汇报，并在图中给出 standard error。

## 实验结果

### 更少的训练数据

在固定 220M T5-Base student 的实验中，Distilling step-by-step 在四个数据集上都比 standard fine-tuning 更有效地利用 human-labeled examples。论文给出的代表性结果是：

- e-SNLI：只使用完整训练集的 12.5%，即可超过 standard fine-tuning 使用 100% 数据的结果；
- ANLI：达到超过 standard fine-tuning 全量数据结果所需的数据量减少约 75%；
- CQA：减少约 25%；
- SVAMP：减少约 20%。

在只有 unlabeled data 的条件下，Distilling step-by-step 也在四个数据集上超过 standard task distillation，并且通常只需要完整 unlabeled set 的一小部分。这个结果支持论文的核心解释：rationale 给 student 提供了比最终 pseudo-label 更丰富的 task-specific supervision。

### 更小的部署模型

固定使用完整数据时，论文比较 220M、770M 和 11B T5 student 与 540B PaLM 的 few-shot CoT。主要观察如下：

- 在有 human labels 的设置中，Distilling step-by-step 使用 220M T5 就能在 e-SNLI 上超过 PaLM few-shot CoT；使用 770M T5 可以在 ANLI 和 SVAMP 上超过 PaLM；在 CQA 上需要 11B T5；
- 在只有 unlabeled data 的设置中，Distilling step-by-step 在四个数据集中的三个上超过 PaLM few-shot CoT；
- 在 e-SNLI 上，220M T5 可以超过 PaLM，模型规模超过缩小 2,000 倍；在 ANLI 和 CQA 上，770M T5 或 11B T5 也以远小于 540B 的规模达到或超过 teacher 表现；
- 对 SVAMP，原始数据量较小，11B student 仍有差距。加入 ASDiv 的 unlabeled examples 后，Distilling step-by-step 能够更有效地利用新增数据并追平 PaLM，而 standard task distillation 仍未完全追上。

这里的比较是“task-specific 小模型”与“few-shot prompted 大模型”的任务性能比较，不代表小模型在通用能力上已经等价于 PaLM，也不能推出 rationale distillation 在所有开放任务中都能实现同等压缩比例。

### 不同 teacher 规模

作者使用 20B GPT-NeoX 和 540B PaLM 生成 rationales，并固定 student 为 220M T5、使用全部数据：

| Method | Rationale teacher | e-SNLI | ANLI | CQA | SVAMP |
|---|---:|---:|---:|---:|---:|
| Standard fine-tuning | N/A | 88.38 | 43.58 | 62.19 | 62.63 |
| Distilling step-by-step | 20B GPT-NeoX | 89.12 | 48.15 | 63.25 | 63.00 |
| Distilling step-by-step | 540B PaLM | 89.51 | 49.58 | 63.29 | 65.50 |

两个 teacher 都能带来相对 standard fine-tuning 的提升，但 PaLM 带来的增益更大。合理解释是 teacher 规模和 reasoning quality 影响 rationale 的信息密度；不过这组实验并未证明 teacher 越大就一定线性带来收益。

### Multi-task 优于单一拼接目标

在 220M T5、全部数据和 PaLM rationale 的设置下，论文比较三种方法：

| Method | e-SNLI | ANLI | CQA | SVAMP |
|---|---:|---:|---:|---:|
| Standard fine-tuning | 88.38 | 43.58 | 62.19 | 62.63 |
| Single-task training | 88.88 | 43.50 | 61.37 | 63.00 |
| Multi-task training | 89.51 | 49.58 | 63.29 | 65.50 |

Multi-task training 在四个数据集上都优于 single-task rationale + label joint prediction；single-task 在 ANLI 和 CQA 上甚至低于 standard fine-tuning。这是对方法结构的直接 ablation：rationale 不是只要被附加到 target sequence 就会自动带来收益，监督任务之间的组织方式本身很重要。

## 论文的主要贡献

### 将 rationale 从解释文本变成训练信号

论文没有把 rationale 只当作可解释性输出，也没有把它作为推理阶段的外部上下文，而是把它作为 student 的 auxiliary target。这样，teacher 生成的中间文本可以帮助 student 学习 task knowledge，同时保持部署时只输出最终 label 的简单接口。

### 同时降低模型规模与数据需求

实验在四个 benchmark 上显示，rationale supervision 既可以提高固定 student 的 data efficiency，也可以降低达到 teacher 性能所需的 student model size。论文因此将它定位为一种 resource-efficient training-to-deployment paradigm。

### 证明 teacher rationale 的质量会影响收益

20B GPT-NeoX 仍然有效，但 540B PaLM 更好；重新选择更强的 rationale teacher 会改变 student 结果。这说明 rationale 并不是与 teacher 能力无关的通用格式，内容正确性和 task relevance 仍然决定蒸馏上限。

## 局限与需要谨慎的地方

### 需要 task-specific demonstrations

few-shot CoT rationale extraction 需要用户为每个 task 准备大约 10-shot demonstrations。相比逐条人工标注，这已经大幅降低成本，但仍然需要 task-level prompt construction，不能理解成完全 annotation-free。

### 主要验证的是 task-specific NLP 模型

论文实验覆盖 NLI、commonsense QA 和数学 word problem，student 是 T5 text-to-text 模型。结果不能直接外推到长程 agent、工具调用、代码修改或开放环境中的 trajectory reasoning。Agent 场景还需要额外处理 tool schema、observation grounding、时序一致性和执行结果验证。

### rationale 不是可靠的内部思维解释

论文把 rationale 视为支持 label 的自然语言解释。它能提供有效训练信号，但不等于 teacher 的真实内部计算过程。rationale 可能是错误的、后验合理化的，或者与 label 只具有表面一致性。对事实、数学、代码和工具任务，仍需 verifier、执行测试或其他质量控制。

### 训练成本会增加

训练时要额外学习 rationale generation，计算量和 target tokens 增加；论文只说明推理时可以通过 `[label]` prefix 避免这部分开销，并没有声称训练成本不变。超长 rationale 还可能稀释 label supervision，因此需要控制长度、质量和 loss 权重 $\lambda$。

### Student 容量限制

小模型能够吸收 rationale 中一部分规律，但不一定能够复现 teacher 的完整推理能力。论文展示的是特定 task 上的性能压缩，不代表任意复杂 reasoning 都能被压缩到小模型。

## 对当前 reasoning 与 agent 数据的启发

这篇论文对带 reasoning content 的训练数据有直接参考价值，但需要区分“可迁移的训练思想”和“论文已经验证的范围”。

### Reasoning content 可以提供比最终 action 更密集的监督

如果一条数据只有最终答案或最终 tool action，student 需要从大量样本中归纳任务规律。合理构造的 reasoning、decision explanation 或 action justification，可以把目标、证据、约束和选择依据显式化，让训练信号更密集。

但这并不意味着所有 reasoning 都应该无条件加入训练。论文的 ablation 已经说明，rationale 与 label 的组织方式会影响结果；在 agent 数据中，还要判断 reasoning 是否来自当前可见 context，是否支持当前 action，以及是否泄漏后续 observation。

### 训练目标与部署接口可以分离

Distilling step-by-step 的一个通用设计是：训练阶段可以让模型学习更丰富的 intermediate target，部署阶段仍然只执行实际需要的输出目标。对 agent 模型，这可以对应：

```text
training:
  prefix -> rationale / action / tool arguments

deployment:
  prefix -> action or final response
```

是否在部署时保留 reasoning，需要根据 latency、可审计性和产品协议决定；论文证明的是“训练时增加 rationale supervision，不必强制推理时生成 rationale”这一点。

### Multi-task supervision 比简单串联更值得优先验证

对于 agent trajectory，可以把不同字段拆成不同任务，例如：

- 预测当前局部目标；
- 生成 action justification；
- 预测 tool name；
- 预测 tool arguments；
- 生成下一步 check 或 recovery intent。

这些目标可以共享 backbone，但使用明确的 task prefix 或字段边界，避免让长 reasoning tokens 完全占据训练信号。具体是否有效，仍需要在 agent 数据上做 controlled ablation，而不能直接照搬论文中的 loss 权重。

## 分析与判断

这篇论文最核心的贡献，可以浓缩成一句话：**把大模型生成的 rationale 从“解释 label 的附属文本”变成 student 的辅助学习任务，并通过 multi-task training 让它在训练阶段帮助 label prediction、在推理阶段又可以被移除。**

它解释了为什么小模型有时只学习最终答案会比较低效：最终 label 的信息量有限，而 rationale 能够把 task-specific 规则、输入到输出的中间联系以及类别判断依据显式暴露出来。它也同时提醒我们，rationale 不是天然可靠的知识载体。teacher 质量、prompt demonstrations、rationale 与 label 的一致性、multi-task 组织方式和 student 容量都会影响最终收益。

因此，使用这篇论文指导数据工作时，重点不应是机械地把所有长 reasoning 拼进 target，而应验证三个问题：

1. reasoning 是否提供了 label / action 之外的有效 task information；
2. reasoning 是否与目标输出一致，并且没有引入 teacher 的错误或后验信息；
3. 将 reasoning 作为独立辅助目标，是否比直接拼接到最终输出带来更好的 data efficiency 和 held-out performance。

## 相关知识链接

- [[training/post-training/knowledge-distillation|Knowledge Distillation]]
- [[training/post-training/sequence-level-distillation|Sequence-level Distillation]]
- [[training/post-training/offline-kd|Offline KD]]
- [[training/post-training/instruction-tuning|Instruction Tuning]]
- [[sources/papers/2016-sequence-level-knowledge-distillation|Sequence-Level Knowledge Distillation]]
- [[sources/papers/2022-self-instruct|Self-Instruct]]
- [[sources/papers/2023-minillm-knowledge-distillation-of-large-language-models|MiniLLM: Knowledge Distillation of Large Language Models]]
