---
title: "Self-Instruct"
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
  - instruction-tuning
  - synthetic-data
source_url: https://arxiv.org/abs/2212.10560
paper_date: "2022-12"
paper_order: "10560"
---

# Self-Instruct: Aligning Language Models with Self-Generated Instructions

## 基本信息

- 论文：[Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560)
- 作者：Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, Hannaneh Hajishirzi
- 机构：University of Washington、Tehran Polytechnic、Arizona State University、Johns Hopkins University、Allen Institute for AI
- 发表：ACL 2023
- 首次公开：2022-12，arXiv:2212.10560v1；本文阅读版本为 v2
- 代码与数据：[yizhongw/self-instruct](https://github.com/yizhongw/self-instruct)
- 相关知识：[[training/post-training/instruction-tuning|Instruction Tuning]]、[[training/data-engineering/synthetic-data|Synthetic Data]]、[[training/post-training/knowledge-distillation|Knowledge Distillation]]

这篇论文讨论的是一个很具体、但后来影响很大的数据问题：Instruction Tuning 需要大量任务定义和示范回答，而人工编写这些内容既昂贵，又容易集中在少数熟悉的 NLP 任务上。论文提出让语言模型从少量人工 seed tasks 出发，自己扩展 instructions、inputs 和 outputs，再用生成数据训练原来的模型。

论文的价值不只在于“让模型生成更多 instruction”。真正构成方法的是一条闭环：用 seed task pool 约束任务扩展，用不同策略生成 task instances，用启发式规则过滤重复和明显无效样本，把通过筛选的任务重新加入 task pool，并最终将生成数据用于 instruction tuning。这个闭环使一个小规模人工任务集合能够扩展成规模更大的 instruction dataset。

## 研究问题

### 人工 instruction data 的瓶颈

Instruction-tuned language model 已经表现出对未见任务的 zero-shot generalization，但效果依赖 instruction data 的数量、覆盖范围和表达多样性。人工数据存在三个相互关联的瓶颈：

- 需要人工提出任务定义，任务类型容易集中在研究者熟悉的 benchmark 任务上；
- 每个任务还需要人工准备输入和目标输出，数据制作成本随任务数增长；
- 人工任务通常缺乏足够多的表达形式，模型可能学习到 prompt pattern，而不是更抽象的 task intent。

论文关注的不是给某一个固定任务做 data augmentation，而是能否让 pretrained LM 在任务层面自举，自动提出新的 instruction，并为这些 instruction 生成可用于训练的实例。

### 任务定义与训练目标

论文将一条 instruction data 定义为一个 task $t$ 及其若干 instances：

$$
t = \left(I_t, \{(X_{t,i}, Y_{t,i})\}_{i=1}^{n_t}\right)
$$

其中 $I_t$ 是用自然语言描述的 instruction，$X_{t,i}$ 是输入，$Y_{t,i}$ 是目标输出。模型需要学习：

$$
M(I_t, X_{t,i}) \approx Y_{t,i}
$$

`Input` 可以为空。比如“写一封关于某个主题的信”本身就可以构成完整任务，也可以被拆成 instruction 和具体 topic 两个部分。允许 empty-input instance，是为了保留开放式生成任务，而不是强行把所有任务转换为固定的输入字段。

Self-Instruct 的目标可以概括为：在较少人工 task definitions 的基础上，得到数量更大、任务类型更多、表达方式更丰富的 instruction data，并验证这些数据能否提升模型的 instruction-following 能力。

## 核心方法

### 总体 pipeline

论文的 pipeline 如下：

```text
175 个人工 seed tasks
        |
        v
从 task pool 采样 instructions，生成新 instructions
        |
        v
判断每条新 instruction 是否属于 classification task
        |
        +--> classification: output-first instance generation
        |
        +--> non-classification: input-first instance generation
        |
        v
instruction / input / output 过滤、去重和冲突处理
        |
        v
将有效 task 加回 task pool，继续迭代扩展
        |
        v
用生成的 instruction data 对原始 LM 做 instruction tuning
```

这里有两个容易被忽略的设计点。第一，task pool 会递归吸收先前生成的任务，因此后续生成不只依赖最初的人工 seed。第二，instruction generation 与 instance generation 被拆开处理，后者还会根据 classification / non-classification 的差异采用不同生成顺序。

### Seed task pool 与递归扩展

实验从 175 个人工编写的 seed tasks 开始，每个 task 含 1 条 instruction 和 1 个 instance。生成新 instruction 时，从 task pool 中抽取 8 条 instruction 作为 in-context examples：

- 6 条来自人工编写的 seed tasks；
- 2 条来自前面迭代生成的 tasks。

这种比例让模型既能持续接触人工任务提供的基本覆盖，又能利用已生成任务扩大表达和任务类型。通过将合格的新任务放回 task pool，方法形成了递归式 bootstrapping，而不是一次性从固定 seed 直接生成一个平面数据集。

论文没有把这一过程描述成完全无监督的 task discovery。初始 seed、prompt template、过滤规则和生成模型共同决定了最终 task distribution；“almost annotation-free”指大规模数据生成阶段几乎不需要逐条人工标注，并不意味着没有人工先验或质量控制。

### Instruction generation

第一步让 LM 根据 sampled instructions 生成新的任务描述。生成目标是自然语言 instruction，而不是当前任务的答案。instruction 可以是：

- 需要输入的任务，例如“根据给定地址找出 ZIP code”；
- 不需要额外输入的开放式任务，例如“写一个关于……的故事”；
- classification、generation、rewriting、explanation、code 等不同任务族。

论文 Figure 1 展示的生成结果包括写信、填写申请表、温度转换代码、地址到 ZIP code 等任务。它们与传统 benchmark task 并不完全重合，说明模型能够从已有任务提示中组合出一定数量的面向用户的任务表达。

### Classification task identification

新 instruction 生成后，论文先判断它是不是 classification task。这里的 classification 不是按严格的机器学习任务定义判定，而是采用一个操作性标准：输出空间较小，且标签集合有限。

区分这两类任务是因为 instance generation 的顺序会影响数据分布。对于开放式 generation，先构造输入再生成对应输出通常比较自然；但对于 classification，如果先自由生成输入，模型很容易偏向某一个标签，导致类别不平衡。

论文通过 few-shot prompt 完成分类判断，使用 12 条 classification seed instructions 和 19 条 non-classification seed instructions 作为示例。这一步不需要训练一个单独的分类器，但它会影响后续每条任务采用哪种 instance generation strategy。

### Input-first instance generation

对 non-classification task，论文采用 input-first：

```text
instruction
  -> 生成 input
  -> 基于 instruction + input 生成 output
```

这种顺序接近模型实际使用 instruction 的过程。模型先根据 task definition 确定需要什么输入，再完成该输入对应的任务。例如对于“根据一个主题写名人名言”，先生成 topic，再生成 quote。

这一步同时要求模型完成两个判断：当前 instruction 需要哪些输入字段，以及什么输出才算完成任务。因而 input-first 不只是简单的数据填充，也是在生成 task instance 的过程中测试模型对 instruction semantics 的理解。

### Output-first instance generation

对 classification task，论文提出 output-first：

```text
instruction
  -> 先生成可能的 class labels
  -> 针对每个 label 生成 input
  -> 形成 input-output instance
```

核心目的不是让输出更像答案，而是控制标签覆盖。以 grammar error detection 为例，若先自由生成句子，模型更倾向生成语法正确的句子；先列出标签，再分别为每个标签构造输入，可以降低单一类别占满数据的风险。

这说明生成顺序本身就是数据分布控制手段。仅仅扩大采样数量并不能自动解决 coverage 问题；当任务具有有限标签空间时，需要把标签覆盖显式放进生成流程。

### Filtering 与 postprocessing

论文主要使用启发式规则做自动过滤和后处理：

- 新 instruction 与已有 instruction 的 ROUGE-L 相似度必须低于 0.7，以减少重复任务；
- 过滤包含 `image`、`picture`、`graph` 等通常不能由纯语言模型直接处理的关键词的 instructions；
- 删除完全相同的 instances；
- 删除 input 相同但 output 不同的冲突 instances；
- 删除过长或过短的 instruction；
- 删除 output 只是重复 input 等明显无效的生成。

这些规则覆盖的不是同一个问题。ROUGE-L 主要处理 lexical duplication，关键词规则处理能力边界，冲突检查处理监督一致性，长度与重复检查处理格式异常。论文的 pipeline 并没有依赖一个能够全面判断事实正确性的 verifier，因此 output quality 仍是主要噪声来源。

## Instruction tuning

完成数据生成后，论文将 instruction 与 instance input 拼接成 prompt，让模型生成 instance output：

```text
instruction + input -> output
```

训练对象是生成数据的原始 GPT-3 模型本身。实验使用 OpenAI fine-tuning API，除将 prompt loss weight 设为 0、训练 2 epochs 外采用默认超参数，所得模型记为 `GPT3_SELF-INST`。

为了减少模型对某一种表面格式的依赖，论文使用多个 prompt templates，例如：

- instruction 前是否添加 `Task:`；
- input 前是否添加 `Input:`；
- prompt 末尾是否添加 `Output:`；
- instruction 与 input 之间使用不同数量的换行。

这一设置的作用是把训练重点放在 instruction 与 output 的关系上，同时让模型适应多种常见的任务编码格式。需要注意的是，这仍然是标准的 supervised instruction tuning，并不是在生成阶段直接用 self-generated process 做 online reinforcement learning。

## 生成数据的规模、质量与多样性

### 规模统计

在 vanilla GPT-3 上运行 Self-Instruct 并完成过滤后，论文得到：

| 指标 | 数量或均值 |
|---|---:|
| instructions | 52,445 |
| classification instructions | 11,584 |
| non-classification instructions | 40,861 |
| instances | 82,439 |
| empty-input instances | 35,878 |
| 平均 instruction 长度 | 15.9 words |
| 平均 non-empty input 长度 | 12.7 words |
| 平均 output 长度 | 18.9 words |

任务数和 instance 数不相等，是因为一个 task 可以拥有多个 instances，而部分任务允许 empty input。这个数据形态与“每条 instruction 固定配一个输入输出”的简单模板不同，保留了一部分开放式任务的结构。

### 多样性分析

论文用 verb-noun structure 对 instruction 做粗粒度分析。52,445 条 instructions 中有 26,559 条能解析出这样的结构；其余任务往往包含更复杂的从句，或者采用疑问句表达。最常见的 20 个 root verbs 及其 noun 组合只覆盖约 14% 的全部 instructions。

作者还计算每条生成 instruction 与 175 条 seed instruction 的最高 ROUGE-L overlap，并观察 instruction、input、output 的长度分布。结果表明：

- 生成数据中存在大量与 seed 表面重合度较低的 instruction；
- 任务表达不只集中在少数固定句式；
- input 和 output 长度也具有一定变化。

但这里的“多样”主要是 instruction 文本和任务形式的多样，不能直接等同于真实任务覆盖或能力覆盖。低 lexical overlap 也不能证明两个任务在语义上完全不同。

### 质量评估

作者随机抽取 200 条 instructions，每条随机抽取 1 个 instance，由论文作者进行 expert review，分别检查 instruction、input 和 output：

| 检查项 | 有效比例 |
|---|---:|
| instruction 描述了有效任务 | 92% |
| input 适合该 instruction | 79% |
| output 正确且可接受 | 58% |
| instruction、input、output 全部有效 | 54% |

这组结果是论文最重要的质量证据之一。模型提出任务的能力明显好于它为任务构造可靠实例和答案的能力。即使部分样本的 output 不完全正确，作者认为其中一些仍保留正确格式或部分有用结构，因此可能仍能提供 instruction-following 的训练信号；但这不能替代对事实、逻辑或代码正确性的验证。

## 实验设计与结果

### 对比设置

论文以 vanilla GPT-3 为主要起点，比较未做 instruction tuning 的模型、公开 instruction-tuned models、GPT-3 使用 T0 / SUPERNI 数据进行 tuning 的变体、GPT3_SELF-INST，以及 InstructGPT 不同版本。为了与 Self-Instruct 数据规模大致可比，作者从 T0 和 SUPERNI 数据中各采样约 50K instances，同时覆盖相应 instructions。

### SUPERNI zero-shot generalization

SUPERNI evaluation set 包含 119 个 tasks，每个 task 有 100 个 instances。评测只提供 task definition，不提供 in-context demonstrations，考察模型能否把 instruction-following 能力迁移到未见任务。指标是 ROUGE-L，适合与参考输出做自动比较，但不能完整反映开放式回答质量。

| Model | 参数量 | ROUGE-L |
|---|---:|---:|
| T5-LM | 11B | 25.7 |
| GPT-3 | 175B | 6.8 |
| T0 | 11B | 33.1 |
| GPT-3 + T0 training | 175B | 37.9 |
| GPT3_SELF-INST | 175B | 39.9 |
| InstructGPT001 | 175B | 40.8 |
| T$_k$-INSTRUCT | 11B | 46.0 |
| GPT-3 + SUPERNI training | 175B | 49.5 |
| GPT3_SELF-INST + SUPERNI training | 175B | 51.6 |

论文据此强调三点：

1. GPT3_SELF-INST 相比 vanilla GPT-3 提升 33.1 个 ROUGE-L 点，说明自生成 instruction data 能够显著改变 instruction-following 行为；
2. 在没有直接使用 SUPERNI 训练集的比较中，GPT3_SELF-INST 接近 InstructGPT001，并优于 T0 和 GPT-3 + T0 training；
3. Self-Instruct 与 SUPERNI 混合后仍有额外收益，说明两种数据的任务分布并不完全相同，生成数据可以提供互补覆盖。

SUPERNI training 组在 SUPERNI evaluation 上更强并不意外，因为训练和评测的 task style、format 和任务族更接近。这个结果不能单独证明 Self-Instruct 在所有任务域都优于人工数据。

### 面向用户的新任务评估

作者另外构造了 252 条 user-oriented instructions，每条 instruction 配 1 个 instance，覆盖 email writing、social media、productivity、entertainment、programming 等场景，并有意加入长短不同以及 bullet points、tables、code、equations 等输出形式。

作者让 instruction 的编写者评估模型输出，评分分为四级：`A` 为正确且令人满意，`B` 为基本可接受但有小错误，`C` 为与任务相关但存在明显错误，`D` 为无关或完全无效。四级评分的 Cohen's kappa 为 0.57；附录把 `A/B` 与 `C/D` 合并后得到 kappa 0.75，说明“是否可接受”的判断一致性高于细粒度质量分级。

在这组新任务上，GPT3_SELF-INST 明显优于使用 T0 或 SUPERNI 公共数据训练的 GPT-3 变体，并接近 InstructGPT001。若把 `B` 也视为有效回答，GPT3_SELF-INST 与 InstructGPT001 的差距约为 5 个百分点。这个实验补充了 SUPERNI 的局限：Self-Instruct 的收益不只体现在传统 NLP task 的 zero-shot 指标，也体现在更开放、更接近用户场景的 instruction 上。

### 数据规模与数据质量

作者从生成数据中抽取不同数量的 instructions，分别 fine-tune GPT-3，再在 252 条 user-oriented instructions 上做人工评估。整体趋势是数据量增加带来性能提升，但在这组任务上约 16K instructions 后收益开始趋于 plateau；在 SUPERNI 上更早出现平台期，作者认为这可能与 Self-Instruct 任务分布和典型 NLP task 分布不同有关。

作者还用更强的 InstructGPT003 重新生成全部 instance outputs，在 instruction 和 input 不变的情况下只替换 output，再用改进后的数据训练 GPT-3。改进 output 后的模型比原始 noisy data 训练的模型提升约 10%。这说明扩大 instruction 数量确实有收益，但收益不会无限线性增长；在任务覆盖达到一定规模后，提高 instance / output 的正确性可能更有效。

## 关键结论

1. 少量人工 seed tasks 可以通过 LM generation 扩展为更大的 task pool，递归加入生成任务是 Self-Instruct 能够扩大覆盖的关键。
2. task definition、instance input 和 target output 应分阶段生成与检查；生成 instruction 的能力明显好于生成可靠 output 的能力。
3. generation order 会改变任务分布。classification 任务采用 output-first 能够缓解自由生成输入造成的 label imbalance。
4. 生成数据的 instruction 有效率为 92%，但 instruction、input、output 全部有效的比例只有 54%；数据规模必须和质量控制一起设计。
5. Self-Instruct 数据与已有人工 instruction data 可以互补，而不是只能二选一。
6. 数据质量仍有很大提升空间。用更强模型重新生成 output 带来约 10% 的性能提升，说明只扩大数据规模并不是唯一有效路径。

## 对 agent 数据合成的启发

Self-Instruct 直接处理的是静态 instruction-instance 数据，不是包含 environment observation、tool action、trajectory recovery 的完整 agent trajectory。因此，不能把它直接当作 agent trajectory synthesis 方法。但它对更复杂的 agent 数据生成仍有几条可迁移的原则。

### Seed pool 应承担任务覆盖控制

如果目标从已有 agent 数据中扩展任务分布，seed 不应只按数量采样，而应覆盖不同任务域、工具类型、horizon、成功条件和失败模式。Self-Instruct 的 task pool 说明，生成器看到什么样的 seed，会直接影响后续生成空间；递归加入生成任务也可能放大最初的分布偏差。

### 任务生成、轨迹生成和质量判断要分开

在 agent 场景中，可以把 Self-Instruct 的三层对象对应为：

```text
task / query specification
  -> workflow or trajectory instance
  -> action / observation / final outcome validation
```

但 agent trajectory 的有效性不能只靠文本启发式判断，还需要检查 tool schema、参数合法性、状态转移、执行结果和任务是否完成。对于 code、data analysis 或 tool-use 任务，执行器、unit tests、schema checker 或 environment verifier 往往比纯语言 judge 更可靠。

### 生成顺序可以用来控制能力分布

论文用 output-first 解决 classification label imbalance。类似地，agent 数据生成也不应总是从 query 自由 rollout 到终点。针对 recovery、verification、tool selection 或 long-horizon planning，可以先规定目标能力或失败类型，再生成满足该条件的任务和轨迹，以提高稀缺行为的覆盖率。

### 不能把递归自生成当成可靠性保证

Self-Instruct 的质量评估显示，instruction 有效不等于 output 正确。对 agent 数据而言，后续生成器很可能复制早期轨迹中的错误工具调用、错误假设或后验解释。因此，递归扩展必须配合去重、执行验证、未来信息检查、失败类型统计和独立 holdout evaluation；仅增加 synthetic trajectory 数量，可能只是在放大同一套 teacher bias。

## 局限与疑问

### 生成质量的上限受 base model 约束

Self-Instruct 依赖 base LM 已有的语言、任务和世界知识。它可以重组和扩展模型见过的 task priors，但不保证发现完全超出模型分布的新任务。论文也指出，LM 在高频语言和高频任务上通常更强，低频或创造性任务的收益可能较小。

### 任务池会继承并放大偏差

seed tasks、prompt templates、关键词过滤和递归采样共同决定最终数据分布。如果早期生成中存在刻板印象、不平衡标签或错误格式，后续 task pool 可能把这些模式继续传播。论文特别提到社会偏见、slur 和 classification label imbalance 等风险。

### 自动过滤仍然比较浅

ROUGE-L、关键词、长度和重复检查能处理明显异常，但不能判断复杂事实、逻辑、代码和开放式回答是否正确。论文中只有 54% 的抽样数据做到 instruction、input、output 全部有效，这说明“通过启发式过滤”与“可以作为高置信监督”之间仍有距离。

### 实验外推边界

SUPERNI 采用 ROUGE-L，且训练和评测任务可能存在风格差异；252 条 user-oriented instructions 规模较小，人工评估也带有作者判断。论文证明的是在其数据和模型设置下的 instruction-following 改善，不足以推出 Self-Instruct 对所有领域、所有模型规模和所有任务都同样有效。

## 分析与判断

Self-Instruct 可以看成一种“任务分布自举”方法：它先用少量人工任务给模型一个任务空间的起点，再让模型扩展 instruction，并把新任务重新放回 task pool。真正的训练信号来自后续的 input-output instances，而不是 instruction 文本本身，因此 pipeline 的瓶颈最终落在实例和答案质量上。

这篇论文对今天数据合成工作的启发，需要把重点放在闭环中的控制点，而不是简单复述“用大模型生成数据”：

- seed 决定生成空间的起点；
- generation order 决定类别、格式和能力分布；
- filtering 决定哪些样本进入训练；
- teacher refinement 可以显著改善答案质量；
- 数据量增长需要结合任务覆盖和独立评测观察，不能只看 token 数；
- 递归生成提高规模的同时，也会递归放大 bias、重复和错误。

对于更一般的 agent 数据，这套思路可以迁移到 task / workflow / trajectory 的分层生成，但不能省略真实环境执行与结果验证。静态 instruction data 的“看起来合理”，与 agent trajectory 的“在状态变化中确实完成任务”，是两个不同的质量标准。

## 相关知识链接

- [[training/post-training/instruction-tuning|Instruction Tuning]]
- [[training/data-engineering/synthetic-data|Synthetic Data]]
- [[training/post-training/knowledge-distillation|Knowledge Distillation]]
- [[sources/papers/2021-flan|Finetuned Language Models Are Zero-Shot Learners]]
- [[sources/papers/2021-t0|Multitask Prompted Training Enables Zero-Shot Task Generalization]]
- [[sources/papers/2022-instructgpt|Training language models to follow instructions with human feedback]]
- [[sources/papers/2023-distilling-step-by-step|Distilling Step-by-Step]]
