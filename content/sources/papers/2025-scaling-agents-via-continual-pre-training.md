---
title: "Scaling Agents via Continual Pre-training"
created: 2026-06-29
published: 2026-06-29
modified: 2026-06-29
type: source
status: processed
source_type: paper
area: sources
tags:
  - source
  - paper
  - agent
  - continued-pretraining
  - synthetic-data
  - tool-use
aliases:
  - AgentFounder
  - Agentic CPT
source_url: https://arxiv.org/abs/2509.13310
code_url: https://github.com/Alibaba-NLP/DeepResearch
---

# Scaling Agents via Continual Pre-training

## 基本信息

| 字段 | 内容 |
|---|---|
| 来源 | arXiv:2509.13310v1 |
| 标题 | Scaling Agents via Continual Pre-training |
| 作者/机构 | Liangcai Su, Zhen Zhang, Guangyu Li, Zhuo Chen, Chenxi Wang 等 / Tongyi Lab, Alibaba Group |
| 日期 | 2025-09-17 |
| 链接 | https://arxiv.org/abs/2509.13310 |
| 项目 | https://github.com/Alibaba-NLP/DeepResearch |
| 相关 topic | [[training/mid-training/continued-pretraining|Continued Pretraining]], [[training/data-engineering/synthetic-data|Synthetic Data]], [[training/post-training/sft|SFT]], [[application/agents/agent|Agent]], [[application/tool-use/tool-calling|Tool Calling]] |

这篇论文提出 **Agentic Continual Pre-training, Agentic CPT**，核心观点是：deep research agent 的能力不能只靠在通用 base model 上做 SFT / RL 后训练来获得。对于长程工具使用、多步搜索、环境反馈和复杂决策来说，通用 base model 缺少足够的 agentic inductive bias，导致 post-training 需要同时学习“agent 能力”和“专家轨迹对齐”，形成优化张力。论文因此把 agent 能力训练前移到 [[training/mid-training/continued-pretraining|Continued Pretraining]] 阶段，并基于该路线训练出 AgentFounder。

如果只记一句话：**AgentFounder 试图把 agent behavior 从后训练阶段的示范模仿，变成持续预训练阶段就已经具备的基础能力。**

## 研究问题

传统 LLM 训练范式通常是：

```text
Pretraining → SFT / RLHF / RL
```

在普通对话或静态问答任务中，post-training 可以有效学习指令跟随、偏好对齐和回答风格。但 deep research agent 的行为更复杂：模型需要搜索、访问网页、调用代码解释器、解析文件、多轮规划、在不确定环境中修正路线，并最终生成可信答案或研究报告。

论文认为，直接从通用 foundation model 做 agent SFT / RL 会遇到三类问题：

1. **策略空间过大**：agent 任务包含大量可行搜索路线、工具调用序列和中间推理路径，完整高质量轨迹难以覆盖。
2. **监督信号稀疏且延迟**：agent 任务常以最终成功/失败评价整条轨迹，中间步骤很难精确判定对错。
3. **能力学习与行为对齐混在一起**：通用 base model 本身缺少 tool use、long-horizon planning 和 multi-step decision-making 的基础倾向，post-training 既要补能力，又要对齐专家轨迹，容易变成模仿固定模式。

因此论文提出一个新的训练位置：在 pretraining 与 post-training 之间加入 **Agentic CPT**，先把模型塑造成更适合 agent 任务的 base model，再进行后续 SFT。

## 核心主张

Agentic CPT 的目标不是学习某个具体 agent benchmark 的答案，而是让模型在预训练式目标下吸收大量 agentic behavior。论文把这种模型称为 **pre-aligned agentic foundation model**：它还不是最终产品 agent，但已经具备更自然的工具调用、规划、长程决策和信息整合倾向，因此更容易被后续 SFT / RL 对齐到具体 deep research agent。

论文的训练管线可以概括为：

```text
Qwen3 base model
  → Agentic CPT Stage 1: 200B tokens, 32K context
  → Agentic CPT Stage 2: 100B tokens, 128K context
  → Agentic SFT with general instruction + ReAct-style trajectories
  → AgentFounder-30B
```

其中 Stage 1 使用约 200B tokens 的 agent data 与 knowledge reasoning corpora，让模型初步获得 tool invocation patterns 和 multi-step reasoning chains；Stage 2 使用约 100B tokens 的高质量 agent data，并把 context length 扩展到 128K，以学习更复杂的 action space 和 long-horizon planning。

这里的关键不是“多做一次中训练”本身，而是 **CPT 数据被重新设计成 agentic behavior distribution**。也就是说，模型在继续预训练阶段看到的不再只是普通网页、论文或代码，而是经过合成与重组的 planning、tool-call、reasoning 和 trajectory decision 数据。

## 方法与机制

### First-order Action Synthesis

First-order Action Synthesis，简称 FAS，是论文的第一类 agentic CPT 数据合成方法。它的目标是在不执行真实工具、不依赖完整监督轨迹的情况下，构造大规模 planning action 和 reasoning action 数据。

FAS 的起点是把静态知识转成动态问题。论文先从多种来源收集数据，包括 discarded trajectories、历史工具调用结果、CommonCrawl 等公开语料，然后构造 entity-anchored open-world memory。每个 entity 不是只对应固定 schema 的知识图谱关系，而是对应大量自然语言知识陈述，并保留时间、来源和原始表达风格。

基于这些 entity-knowledge mappings，论文再采样 entity clusters，合成多种风格的问题，包括 factual retrieval、numerical computation、multi-hop reasoning 和 synthesis tasks。这样，静态知识就被转化成需要主动检索、整合和推理的问题场景。

FAS 包含两类 action synthesis。

**Planning Action Synthesis** 关注任务初期的分析和下一步行动。给定问题 `Q`，teacher model 生成问题分析以及 first-step action prediction，例如下一步应该搜索什么、访问什么、是否直接回答。为了避免真实调用搜索 API，FAS 只合成 reasoning chains 和 tool calls，不执行工具，从而显著降低成本。

论文还强调，相比对同一个问题采样多个类似 reasoning-action，更有效的做法是对同一知识 memory 生成多种不同风格问题，再分别合成 reasoning-action 数据。这样能扩大训练 context 和 action space，而不是在同一个 prompt 上产生重复样本。

**Reasoning Action Synthesis** 关注信息已经足够时如何综合事实生成最终答案或报告。论文采用两步设计：

1. 只给问题，让模型基于内部知识拆解子问题、生成推测和初步答案 `A_1`；
2. 再给问题和映射到的必要知识，让模型纠正逻辑错误并生成最终答案 `A_2`。

这种设计试图避免模型在直接看到必要知识时机械地把知识点串成答案，而是模拟更真实的思考过程。随后论文用 LLM-as-Judge 检查 `A_2` 与 ground truth answer 是否对齐，通过 rejection sampling 保留高质量 logical reasoning CoT 数据。

### High-order Action Synthesis

High-order Action Synthesis，简称 HAS，是论文更有意思的一部分。它处理的是 post-training 过程中产生的大量轨迹，尤其是被 reject sampling 或 RL 丢弃、只用一次或没有充分利用的轨迹。

论文指出，完整轨迹的最终成功/失败是粗粒度反馈。即使一条 trajectory 最终失败，其中许多中间状态仍然包含有价值的上下文和局部决策信号；反过来，即使一条轨迹成功，也不意味着每一步都是唯一正确路径。直接把整条轨迹拿来模仿，会让模型学习“复现某条路线”，而不是在关键步骤上做决策。

HAS 的核心思想是把轨迹重新建模为 step-wise decision-making。给定问题 `Q` 和轨迹：

```text
T = {(S_1, R_1), ..., (S_K, R_K)}
```

其中 `S_k` 是第 `k` 步 reasoning + tool invocation，`R_k` 是环境响应。对任意 step `S_k`，其条件上下文是：

```text
C_k = (Q, S_1, R_1, ..., S_{k-1}, R_{k-1})
```

在不真实执行工具的情况下，论文让 LLM 基于 `C_k` 生成 `N` 个 alternative thought-and-invocation candidates：

```text
A_k = {S_k^(1), ..., S_k^(N)}
```

然后把原始步骤 `S_k` 和这些候选混合、打乱，形成一个局部多选决策空间。训练文本中显式枚举这些 options，并插入类似 “I will choose option n_k” 的局部决策语句，后面接真实环境响应 `R_k`，最后再附加整条轨迹的 correct / incorrect judgment。

这种做法的价值在于：它不直接构造不可靠的 step-level reward，也不把 uncertain intermediate feedback 强塞进 RL；而是把已有轨迹转换成多选决策文本，让模型在 NTP 目标下学习局部 action space、decision process 和环境反馈的关系。

### 两阶段 Agentic CPT

论文的两阶段训练策略与数据长度有关：

| 阶段 | 数据与目标 | Context |
|---|---|---|
| Stage 1 | FAS 数据、短 HAS 数据、知识推理语料，学习基础 planning / reasoning / tool patterns | 32K |
| Stage 2 | 高质量 HAS 长轨迹数据，学习长程 agent decision-making | 128K |

这说明 Agentic CPT 不只是数据类型变化，也包含 context curriculum：先用较短上下文学习基本行为分布，再用长上下文数据学习复杂轨迹和 long-horizon planning。

## 实验与证据

### 主结果

论文在 10 个 benchmark 上评估 AgentFounder-30B，包括 BrowseComp-en、BrowseComp-zh、GAIA、Xbench-DeepSearch、WebWalkerQA、HLE、DeepResearch Bench、Frames、SEAL-0 和 AcademicBrowse。

在 general web search benchmarks 上，AgentFounder-30B 的结果包括：

| Benchmark | AgentFounder-30B |
|---|---:|
| BrowseComp-en | 39.9 |
| BrowseComp-zh | 43.3 |
| GAIA | 72.8 |
| Xbench-DeepSearch | 73.0 |
| WebWalkerQA | 71.9 |

在 scenario-targeted benchmarks 上：

| Benchmark | AgentFounder-30B |
|---|---:|
| HLE Pass@1 | 31.5 |
| DeepResearch Bench RACE Overall | 47.9 |
| Frames Pass@1 | 89.6 |
| SEAL-0 Pass@1 | 43.9 |
| AcademicBrowse Pass@1 | 75.3 |

论文声称 AgentFounder-30B 在多个任务上超过现有开源 deep research agents，并在 HLE、Frames、AcademicBrowse 等 benchmark 上也超过若干闭源 deep research 产品报告值。这里需要注意：表中大量 baseline 使用 official sources 或 prior work 结果，不一定完全同一运行环境；但至少说明 Agentic CPT 训练出的 30B agent 在 open-source deep research agent 范围内竞争力很强。

### Agentic CPT 是否提升后训练适配

论文用三种不同 SFT 配方比较 Qwen3-30B-A3B-Base 与 AgentFounder-30B-Base。结果显示，在 SFT-A、SFT-B、SFT-C 三种设置下，以 AgentFounder-Base 作为 base model 都比 Qwen3-Base 更好。

例如 SFT-B 设置下：

| Base | BrowseComp-en | BrowseComp-zh | GAIA | HLE |
|---|---:|---:|---:|---:|
| Qwen3-30B-A3B-Base | 28.6 | 35.6 | 71.8 | 27.0 |
| AgentFounder-30B-Base | 39.9 | 43.3 | 72.8 | 31.5 |

这支持论文的核心主张：Agentic CPT 不是只对某个后训练数据配方有用，而是提升了 base model 对不同 agentic post-training 的适配性。

### 两阶段训练与数据类型消融

两阶段训练消融显示，在相同 50B token 规模下，Stage 1 & 2 比 Stage 1 only 更好：

| Strategy | BrowseComp-en Pass@1 | BrowseComp-zh Pass@1 | GAIA Pass@1 |
|---|---:|---:|---:|
| Stage 1 Only | 31.4 | 34.3 | 69.9 |
| Stage 1 & 2 | 35.5 | 37.2 | 72.8 |

这说明长上下文高质量 HAS 数据确实提供了额外收益，尤其能缓解单阶段训练中长轨迹被截断的问题。

数据类型消融显示，FAS 本身已经有效，FAS+HAS 进一步带来互补收益，尤其在 BrowseComp-zh 的 Pass@1 上从 37.0 提升到 40.1。不过 GAIA Pass@1 在 FAS+HAS 下从 72.8 降到 69.9，论文认为 Pass@3 提升说明这可能是正常波动而非系统退化。更保守的理解是：HAS 对探索多样性有帮助，但不同 benchmark 的即时 Pass@1 可能受数据分布和任务类型影响。

### Scaling Law

论文分别观察 model size 和 data volume 对 agentic performance 的影响。

模型规模上，平均 accuracy 随模型大小上升：

```text
1B: 20.4%
4B: 32.7%
30B-A3B: 48.9%
```

数据规模上，从 0B 到 315B token，平均 Pass@3 从 54.2 提升到 62.2，呈现近似 logarithmic scaling。Stage 2 的 128K context training 在 65B 和 315B checkpoint 上都带来额外收益。

论文还比较了 SFT loss：AgentFounder variants 在同一 SFT-A 数据上比 baseline 收敛到更低 loss，且 CPT data volume 越大，loss 越低。这是对“Agentic CPT 减轻后训练双重负担”的间接证据。

### 工具调用分析

AgentFounder 使用五类工具：Search、Visit、Python Interpreter、Google Scholar 和 File Parser。工具调用分布显示，不同任务会诱发不同工具密度：

- BrowseComp-en 和 HLE 具有 heavy-tailed tool-call distribution，需要更密集探索；
- WebWalker 与 GAIA-text 更集中于较少工具调用，说明模型能对结构化任务采取更保守的工具策略。

此外，AgentFounder-30B 在 ACEBench 上从 Qwen3-30B-A3B 的 67.2 提升到 70.0，说明 Agentic CPT 对 general tool-use 也有一定迁移，而不只是 deep research 专项。

## 关键结论

这篇论文最重要的贡献是提出了一个训练位置上的观点：**agent 能力不应只在 post-training 阶段学习，而可以通过 continual pre-training 形成更强的 agentic foundation model。**

从训练方法看，Agentic CPT 有三个可复用思想：

1. **把静态知识转化为 agentic contexts**：通过 entity-anchored memory 和 multi-style question synthesis，把普通知识语料转成需要检索、规划和推理的问题。
2. **用 FAS 扩大 first-step planning 与 reasoning action 数据**：不执行工具，只合成 reasoning + action，因此成本低、规模大。
3. **用 HAS 重构轨迹为 step-wise decision-making 文本**：从完整轨迹模仿转向局部决策空间学习，减少对单一路径复现的依赖。

从实验看，AgentFounder-30B 在 deep research agent benchmarks 上取得很强结果，并且 Agentic CPT 在多种 SFT 配方下都能提升后训练效果。这说明对 agent 任务来说，中训练阶段的数据形态和能力塑形可能和后训练数据同样关键。

## 局限与疑问

### 1. 评测对比不完全统一

论文表格里很多 baseline 是 official results 或 prior work results，不一定与 AgentFounder 在同一工具、同一检索后端、同一推理参数和同一上下文限制下评测。对于 deep research agents，这一点尤其重要，因为搜索工具质量、网页解析器、最大工具次数和执行环境都会显著影响结果。

因此，主表可以说明 AgentFounder 很强，但不能简单解释为“模型本体一定强于所有闭源系统”。它评估的是 model + tool stack + inference protocol 的整体系统表现。

### 2. 合成数据有效，但也有 judge 和 teacher bias 风险

FAS 和 HAS 都严重依赖 teacher model 与 LLM-as-Judge。虽然论文通过 rejection sampling 提高质量，但 judge 偏好、teacher 风格、prompt 模板和合成问题分布仍可能被模型吸收。尤其在 deep research agent 中，模型可能学到特定搜索风格、报告结构或工具调用偏好，而不一定是一般性的最优 agent policy。

### 3. HAS 的 step-wise decision 文本很巧妙，但并不等价于真实 step reward

HAS 避免了直接使用不可靠 step-level reward，这很稳妥。但它把局部候选 action 和最终 trajectory judgment 拼接成文本，让模型在 NTP 中学习决策过程。这种信号是否能精确对应真实环境中的 policy improvement，还需要更细的 ablation。

更具体地说，如果原始轨迹最终失败，那么其中某些步骤可能仍然局部正确；如果原始轨迹最终成功，也可能有冗余或偶然正确步骤。HAS 用最终 judgment 作为文本信号，仍然可能保留 trajectory-level credit assignment 的模糊性。

### 4. Agentic CPT 的成本和数据工程门槛很高

论文使用 200B + 100B token 的两阶段 CPT，并构造大规模 FAS / HAS 数据。这说明方法有效，但也说明它不是轻量后训练技巧。对一般团队来说，更现实的借鉴可能不是完整复现 AgentFounder，而是吸收其中的数据形态：first-step planning synthesis、trajectory reuse、step-wise decision augmentation 和 long-context agent data curriculum。

### 5. Deep research agent 不等同于所有 agent

AgentFounder 主要面向 web search / deep research / information-seeking 任务。虽然 ACEBench 结果显示 general tool-use 有迁移，但它与 coding agent、GUI agent、robotic agent、database agent 等仍有差异。尤其 coding agent 更依赖代码执行、编辑 diff、测试反馈和 repository state，因此 Agentic CPT 思路可以迁移，但 FAS / HAS 的具体数据构造要按环境重写。

## 分析与启发

这篇论文与近期一批 agent 训练工作形成了清晰分工：许多 deep research agent 论文主要关注 SFT/RL 阶段如何造题、rollout、筛选和强化；AgentFounder 则把问题前移到 base model 阶段，讨论模型在进入 agent post-training 之前是否已经应当具备 agentic behavior distribution。

这篇论文最值得沉淀的并不是某个具体 benchmark 数字，而是 **agentic capability injection 可以发生在 continued pretraining 阶段** 这一训练范式判断。它与普通 domain CPT 的区别在于：domain CPT 主要注入领域知识或文本分布，Agentic CPT 则试图注入“在动态环境中规划、调用工具、基于反馈推进任务”的行为分布。

对 Code Agent 轨迹数据而言，FAS 和 HAS 也提供了可迁移的数据工程思路。FAS 类似于低成本构造“问题分析 + 下一步行动”的 first-step supervision；HAS 则更像把已有 agent trajectory 重新组织成 step-level decision space。对于 coding agent，类似方法可以把 `prefix → gold tool call / edit / verify` 的单一路径监督，扩展为包含 alternative actions、argument grounding 和 feedback-aware decision 的训练文本。

不过，Agentic CPT 不应被理解为后训练的替代品。论文自身的 SFT 适配实验也表明，post-training 数据仍然决定最终能力释放。更准确地说，Agentic CPT 负责把模型塑造成“更容易被训练成 agent 的 base”，SFT/RL 仍然负责具体交互协议、工具格式、任务风格和安全边界。

## 相关知识链接

- [[training/mid-training/continued-pretraining|Continued Pretraining]]
- [[training/mid-training/capability-injection|Capability Injection]]
- [[training/data-engineering/synthetic-data|Synthetic Data]]
- [[training/post-training/sft|SFT]]
- [[training/post-training/rejection-sampling|Rejection Sampling]]
- [[application/agents/agent|Agent]]
- [[application/tool-use/tool-calling|Tool Calling]]
