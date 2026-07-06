---
title: "APTBench: Benchmarking Agentic Potential of Base LLMs During Pre-Training"
created: 2026-06-30
published: 2026-06-30
modified: 2026-06-30
type: source
status: processed
source_type: paper
area: sources
tags:
  - source
  - paper
  - agent
  - benchmark
  - evaluation
  - pretraining
aliases:
  - APTBench
source_url: https://arxiv.org/abs/2510.24397
code_url: https://github.com/TencentYoutuResearch/APTBench
---

# APTBench: Benchmarking Agentic Potential of Base LLMs During Pre-Training

## 基本信息

| 字段 | 内容 |
|---|---|
| 来源 | arXiv:2510.24397v1 |
| 标题 | APTBench: Benchmarking Agentic Potential of Base LLMs During Pre-Training |
| 作者/机构 | Jiarui Qin, Yunjia Xi, Junjie Huang, Renting Rui, Di Yin, Weiwen Liu, Yong Yu, Weinan Zhang, Xing Sun / Tencent Youtu Lab, Shanghai Jiao Tong University |
| 日期 | 2025-10-28 |
| 链接 | https://arxiv.org/abs/2510.24397 |
| 代码 | https://github.com/TencentYoutuResearch/APTBench |
| 相关 topic | [[application/evaluation/benchmark|Benchmark]], [[application/evaluation/evaluation|Evaluation and Benchmark]], [[application/agents/agent|Agent]], [[application/tool-use/tool-calling|Tool Calling]], [[training/mid-training/continued-pretraining|Continued Pretraining]], [[training/data-engineering/synthetic-data|Synthetic Data]] |

APTBench 讨论的是一个在 agent 训练链路中越来越重要、但此前缺少直接度量的问题：**base model 在进入 SFT、RL 或完整 agent scaffold 之前，是否已经具备可被后续训练释放出来的 agentic potential**。传统预训练评测通常关注 MMLU、GSM8K、EvalPlus 等静态任务；真实 agent 任务则要求模型在多轮交互中规划、选择工具、处理外部反馈并持续推进任务。两者之间存在明显断层。

论文提出的解决方案不是让 base model 直接跑完整 agent benchmark，而是把真实 agent 任务和成功轨迹转换成适合 base model 的 multiple-choice question 与 text completion question。这样可以绕开 base model 指令跟随和多轮执行能力不足的问题，同时保留 agent 任务中的 planning、action 和场景相关 atomic ability 信号。

如果只保留一个核心观点：**APTBench 把动态 agent trajectory 转译为静态但轨迹条件化的 base-model evaluation，从而为 agent-oriented pretraining 提供比通用静态 benchmark 更贴近下游 agent 能力的早期反馈。**

## 研究问题

随着 Claude Code、Deep Research、SWE agent 等系统的发展，agent 能力不再只是后训练阶段的应用层包装。近期一些工作开始把 agent-specific data 纳入 pre-training 或 continued pretraining，希望在 base model 阶段就注入规划、工具使用、长程交互和环境反馈处理能力。问题在于：预训练阶段如何判断这种注入是否有效。

现有评测体系存在两类不匹配。

第一，通用 base model benchmark 多为单轮、静态、隔离能力测试。MMLU 主要考察知识与理解，GSM8K 考察数学推理，EvalPlus 考察代码生成。这些任务对于衡量模型基础能力有价值，但不能直接反映模型能否在 agent 场景中基于外部反馈做动态决策。论文在引言和附录中展示，若用这些通用 benchmark 分数去预测 instruct model 在 SWE-bench Verified、Terminal-Bench 或 Tua2 等 agent benchmark 上的表现，相关性较弱，甚至在一些设置下呈负相关。

第二，真实 agent benchmark 通常面向 post-trained model。SWE-bench Verified、Terminal-Bench、web automation benchmark 和 deep research benchmark 需要复杂指令跟随、工具调用格式、多轮状态管理、环境执行和最终提交能力。base model 尚未经过指令微调，直接放入这些环境中执行并不可靠，评测成本也很高。

APTBench 因此试图填补二者之间的空白：在不要求 base model 完整执行 agent task 的前提下，评估它是否已经具备后续 agent post-training 可以利用的能力基础。

## 核心主张

论文的核心主张可以分为三层。

第一，agentic potential 应当在 pre-training 或 base-model 阶段被监控。模型的核心能力和行为分布主要由预训练数据、架构和训练过程决定；如果等到 post-training 后才发现模型缺少 agent 相关基础能力，修正成本会显著升高。因此，agent-oriented pretraining 需要一个能够在训练早期提供反馈的评测工具。

第二，base model 的 agentic potential 可以通过 trajectory-conditioned question 来近似测量。完整 agent 执行虽然是动态过程，但其中许多关键能力可以被重构成 next-token prediction 形式，例如在给定任务与已有轨迹时选择下一步计划、补全下一条命令、判断哪个 patch 更合理、选择哪些报告陈述被网页证据支持。APTBench 将这些子问题组织为 MCQ 或 TC，使 base model 可以在 few-shot prompt 下被稳定评测。

第三，这类 agent-oriented base benchmark 比通用静态 benchmark 更能预测下游 agent 表现。论文将 base model 的 APTBench 分数与对应 instruct model 在 SWE-bench Verified 上的表现做相关性比较，认为 APTBench-SWE 和 APTBench-DR 与下游 agent benchmark 的正相关更强。这一结果支持 APTBench 作为 agentic pretraining 过程中的选择信号，而不是最终 agent 评测的替代品。

## 方法与机制

### 从完整 agent task 到 base-model question

APTBench 的构造原则可以概括为：

```text
真实 agent 场景
  → 任务与成功轨迹收集
  → agent-oriented ability decomposition
  → correct answer extraction
  → negative choice generation 或 text completion formatting
  → human validation
```

论文首先选择与真实 agent 应用对齐的场景，例如 software engineering 和 deep research。然后收集任务及其成功轨迹，轨迹可以来自人类文档、GitHub issue、已有 agent 框架或高性能 deep research agent。对于 agent 生成轨迹，论文使用 rejection sampling 和人工验证，确保保留的轨迹确实完成任务。

在能力拆分上，APTBench 关注三类能力：

| 能力 | 含义 |
|---|---|
| Planning | 高层任务分解、stepwise planning，以及根据外部反馈动态调整计划 |
| Action | 基于当前任务和已有轨迹完成下一步行动，例如工具调用、命令生成或最终结论生成 |
| Atomic Abilities | 与具体场景紧密相关的原子能力，例如 SWE 中的 bug localization、patch selection，或 deep research 中的 citation grounding |

答案生成遵循一个关键约束：尽可能从原始成功轨迹中抽取正确答案。对于简短、明确、形式化强的答案，例如下一条 Bash command 或 tool call，APTBench 使用 text completion；对于较长或存在多种可行表达的问题，APTBench 使用 multiple-choice，并通过 LLM 对正确答案进行扰动，生成看似合理但有缺陷的 distractors。人工验证用于保证正确选项确实是最优答案，避免多选题因为局部动作多解而失去判别性。

这种构造方式本质上把 agent 执行过程中的局部决策点转化为 base model 可处理的条件生成或判别任务。它牺牲了完整 end-to-end execution 的真实性，换取了在预训练阶段可运行、可规模化、可快速反馈的评测信号。

### APTBench-SWE

APTBench-SWE 面向 software engineering agent，包含两个场景：EnvSetup 和 IssueFix，总计 3,727 个问题。

**EnvSetup** 关注代码仓库环境配置。论文从 ICML 2025、ICLR 2025 和 NeurIPS 2024 论文对应的 GitHub 仓库中筛选出 489 个带完整 README 的项目。README 被视为人类完成环境配置的单轮轨迹来源。

在 EnvSetup Planning 中，论文用 LLM 从 README 中抽取 step-by-step setup plan，并通过重排步骤、删除关键步骤、加入有害或冗余步骤生成错误选项。模型需要根据仓库信息选择最佳安装计划。该子任务共 437 题，输入长度约 16K-32K。

在 EnvSetup Action 中，论文从 README 中抽取 Bash commands，并给定前 $T$ 条命令，要求模型补全第 $T+1$ 条命令。由于命令本质上是 terminal tool call，评测主要使用 Exact Match。该子任务共 1,084 题，输入长度通常小于 4K。

在 EnvSetup Atomic 中，论文收集 closed GitHub issues，保留响应数较多且存在高赞解决方案的环境配置问题，再由 LLM 识别是否属于 setup issue 并总结解决计划。模型需要从多个候选方案中选择正确处理方式。该子任务共 147 题。

**IssueFix** 关注代码修复。论文使用 SWE-Smith-Trajectory 中成功解决 SWE-Bench 问题的轨迹作为种子。对于 Planning，输入包含 issue、历史轨迹和环境反馈，正确答案是当前步骤的 thought / next-step plan，负选项来自后续步骤中不合当前状态的计划。对于 Action，模型需要基于历史轨迹补全下一条 Bash command 或 tool call，工具集合包括 bash terminal、`str_replace_editor` 和 `submit`。这两个子任务输入较长，达到 32K-128K。

IssueFix Atomic 进一步使用 SWE-Bench-Lite 构造 bug localization、fix patch selection 和 test patch selection。BugLocate 以 gold patch 修改范围作为正确 buggy snippet，从同一文件中采样干扰片段；FixPatch 和 TestPatch 则使用多个 LLM 生成失败 patch 作为负选项，以 gold patch 或 gold test patch 作为正确答案。

APTBench-SWE 的设计重点在于：它没有要求 base model 真正进入 Docker 环境修 bug，而是从成功修复轨迹和 oracle 信息中抽出局部 planning/action/atomic judgment，使软件工程 agent 能力在 base-model 阶段变得可测。

### APTBench-DR

APTBench-DR 面向 deep research agent，包含 closed-ended question 和 open-ended question，总计 2,255 个中英文问题。

**Closed-ended QA** 来自 InfoDeepSeek。论文使用 InfoDeepSeek 框架生成 Plan-Action-Feedback 格式的搜索/浏览轨迹，agent backbone 包括 DeepSeek-V3 和 Gemini-2.5-Flash。筛选分两步：先由 LLM 比较 agent 输出与参考答案是否一致，再由人工确认正确答案是否能从轨迹本身推出，以降低 agent 内部知识或幻觉对数据的污染。

Closed-ended Planning 主要评估 stepwise planning。给定用户问题和前 $T$ 步搜索/浏览轨迹，模型需要在候选 plan-action pair 中选择最合理的下一步。负选项会根据下一步工具类型定制：如果正确动作是 search，负选项可能是浏览无关文档、提前终止、重复已解决搜索，或使用不匹配的 query；如果正确动作是 browse，负选项可能是浏览无关 URL、跳过当前证据直接进入新搜索，或错误调用工具；如果正确动作是 terminate，负选项则会继续搜索或浏览。Closed-ended Action 则给定完整轨迹，让模型生成最终简短答案，采用 TC 形式并用 EM / ROUGE 评价。

**Open-ended QA** 来自 DeepResearch-Bench 和 Researchy Questions。由于商业 deep research 产品通常不公开内部轨迹，论文主要收集高性能 Deep Research Agent 产出的最终报告。Open-ended Planning 使用 Researchy Questions 的标准 plan 作为正确答案，通过打乱、删除或加入无关子计划生成负选项。Open-ended Action 要求模型从多个候选报告中选择最佳报告，正确选项来自高性能 agent，负选项通过破坏事实准确性、逻辑、可读性和用户需求对齐生成。

Open-ended Atomic 重点考察 citation。给定报告、被引用网页内容和若干报告陈述，模型需要选择哪些陈述确实被该网页支持。由于同一网页可能支持多条陈述，该任务允许多正确选项。这个设计使 deep research 的 evidence grounding 能够被拆成相对稳定的判别问题。

### 评测设置

APTBench 面向 base model，因此评测时使用 few-shot prompting 帮助模型遵守输出格式。MCQ 使用 Accuracy；TC 使用 Exact Match 和 ROUGE。SWE 的 TC 任务更强调 EM，因为 Bash command 或 tool call 要求工具名称和参数精确匹配；DR 的最终答案生成允许一定表达差异，因此同时考虑 EM 和 ROUGE。

论文还强调 APTBench 具有较强 long-context 成分。许多任务输入超过 16K，IssueFix plan/action 和 open-ended citation/action 甚至达到 32K-128K。因此，APTBench 同时混合了 agentic decision-making 与 long-context processing 两类能力信号。

## 实验与证据

### Base model 在 APTBench 上的表现

论文评测了 Qwen3 系列、Llama3.2-3B、Llama4-Scout、DeepSeek-V3 / V3.1、SmolLM3-3B、Seed-OSS-36B、Gemma3-27B、GLM-4.5 / GLM-4.5-Air 和 Kimi-K2 等 base 或 pre-trained models。

APTBench-SWE 的平均分显示：

| Model | APTBench-SWE AVG |
|---|---:|
| Qwen3-1.7B | 24.27 |
| Qwen3-4B | 38.75 |
| Qwen3-8B | 41.62 |
| Qwen3-30B-A3B | 41.60 |
| Seed-OSS-36B | 49.93 |
| DeepSeek-V3 | 49.15 |
| DeepSeek-V3.1 | 50.01 |
| Kimi K2 | 52.56 |

APTBench-DR 的平均分显示：

| Model | APTBench-DR AVG |
|---|---:|
| Qwen3-1.7B | 28.52 |
| Qwen3-4B | 40.50 |
| Qwen3-8B | 42.35 |
| Qwen3-30B-A3B | 45.55 |
| Seed-OSS-36B | 61.56 |
| GLM-4.5 | 58.64 |
| DeepSeek-V3 | 61.47 |
| DeepSeek-V3.1 | 66.42 |
| Kimi K2 | 53.12 |

这些结果支持论文的三点观察。

第一，agent 能力在模型规模上存在门槛效应。Qwen3-1.7B 与 4B、8B、30B-A3B 之间差距明显，说明过小模型很难稳定吸收 planning、action、long context 和场景原子能力。

第二，中等规模模型也可以在 agent potential 上表现很强。Seed-OSS-36B 在 SWE 和 DR 上都接近或超过若干更大 MoE 模型，论文据此认为 30B activated-parameter 区间可能是 agent base model 的一个高性价比规模段。

第三，训练数据比参数规模和架构本身更能解释部分差异。论文比较相近规模模型时发现，Qwen3-4B 明显优于 Llama3.2-3B 和 SmolLM3-3B，GLM4.5-Air 也明显优于 Llama4-Scout。论文将这种差异归因于是否在预训练数据中进行了 agent-centric optimization。

### 与下游 agent benchmark 的相关性

APTBench 最关键的实验不是单个模型排名，而是相关性分析。论文将 base model 在 APTBench 上的表现与对应 instruct model 在 SWE-bench Verified 上的表现对齐比较，并与 MMLU、EvalPlus、GSM8K 等通用 benchmark 做对照。

论文的结论是：通用 benchmark 与 SWE-bench Verified 等下游 agent benchmark 相关性弱，分数区间也容易聚集，难以区分模型的 agent 能力；APTBench-SWE 和 APTBench-DR 与 SWE-bench Verified 呈现更强正相关，因此更适合作为 agent-oriented pretraining 的早期评估信号。

这个结论需要谨慎理解。APTBench 与下游 agent benchmark 的相关性并不意味着 APTBench 可以替代端到端 agent evaluation。它更像是一个 pretraining-stage proxy：当模型还不能稳定运行完整 agent scaffold 时，用轨迹条件化的静态问题估计其未来成为 agent 的潜力。

### Long-context confounding

论文特别指出，APTBench 的得分同时受到 long-context capability 影响。许多任务需要读取长轨迹、长 README、长 issue context、长报告或网页内容。如果模型长上下文处理能力不足，即使具备一定 agent decision-making 能力，也可能在 APTBench 上表现受限。

论文在相关性分析中移除部分超长上下文任务，例如 IssueFix 的 plan/action 子任务，以及 open-ended question 中的 citation/action 子任务。移除后，APTBench 与下游 agent evaluation 的相关性进一步增强。这个结果有两层含义：

1. APTBench 原始分数并不是纯粹的 agentic potential，它混合了 long-context robustness。
2. 对 agent 预训练而言，long-context trajectory data 本身也是重要训练对象，因为真实 agent 任务往往无法脱离长上下文历史。

因此，APTBench 的长上下文因素既是潜在混杂变量，也是 agent 场景的真实组成部分。评测使用时需要根据目的决定是否报告去除长上下文任务后的子集分数。

## 关键结论

APTBench 的贡献主要不在于提出新的 agent 算法，而在于把 agent 能力评测前移到 base-model 阶段。它提供了一种可复用的 benchmark construction pattern：从真实任务和成功轨迹中抽取 planning、action 和 atomic ability，将其转化为 base model 可以完成的 MCQ / TC 任务。

这一范式的价值在于，它为 agent-oriented pretraining 提供了早期反馈信号。对于昂贵的预训练或 continued pretraining，训练者需要在数据配比、长上下文 curriculum、agent trajectory 数据质量和模型规模之间做选择。APTBench 这类评测可以帮助判断模型是否正在获得更适合后续 agent post-training 的行为基础。

与 Agentic CPT 类工作相比，APTBench 更像评测侧的互补工具。Agentic CPT 关注如何把 agentic behavior distribution 注入 continued pretraining；APTBench 则关注如何在 base model 阶段检测这种注入是否真的带来了可观测的 agentic potential。二者共同指向一个趋势：agent 能力不再只是 post-training 后期才考虑的问题，而是逐渐进入 pretraining data design、mid-training curriculum 和 early evaluation 的整体闭环。

## 局限与疑问

APTBench 仍然是代理评测，而不是完整 agent 执行评测。MCQ 和 TC 可以降低 base model 评测难度，但无法覆盖真实 agent 中的探索、错误恢复、工具执行失败、状态污染、长时间任务维护和资源调度问题。一个模型在 APTBench 上选择了正确下一步，不等于它能在开放环境中稳定走完整条轨迹。

多选题构造依赖 negative choice quality。论文通过 LLM degradation 和人工验证保证正确答案最优，但 agent 任务天然存在多条合理路径。若 distractor 过弱，题目可能更像模式识别；若 distractor 与正确答案都局部合理，题目又可能引入标注歧义。这个问题在 planning 和 open-ended report selection 中尤其明显。

APTBench 分数混合了 agentic ability 与 long-context ability。论文已经观察到，移除超长上下文任务后相关性增强。这说明原始分数需要结合上下文长度分布解释，否则可能把长上下文读取失败误判为 agent planning 能力不足。

数据来源可能带来风格偏差。SWE 部分依赖 README、SWE-Smith-Trajectory、SWE-Bench-Lite 和 LLM 生成 patch；DR 部分依赖 InfoDeepSeek 轨迹、高性能 deep research agent 报告和 LLM 生成负例。这些来源会把特定 agent framework、工具格式、报告风格和 teacher model 偏好带入 benchmark。

相关性实验仍然是间接证据。论文用 base model 的 APTBench 分数与对应 instruct model 的 SWE-bench Verified 表现比较，但 post-training recipe、agent scaffold、工具环境和评测框架都会影响最终 agent 分数。APTBench 可以作为训练过程中的 useful signal，但不应被解释为决定性能力证明。

## 分析与启发

APTBench 最值得关注的地方，是它把 agent 评测问题拆成了两个层次：最终 agent 能否完成任务是一类问题；base model 是否已经具备可被后续训练激活的 agentic potential 是另一类问题。前者需要 end-to-end rollout，后者可以通过 trajectory-conditioned static tasks 来近似观察。

这一拆分对 agent 训练体系很重要。随着 agent-specific data 进入 pretraining 或 continued pretraining，训练过程需要比 MMLU、GSM8K、EvalPlus 更贴近 agent 行为分布的监控指标。APTBench 说明，成功轨迹不仅可以作为 SFT 或 RL 数据来源，也可以被重构为评测数据：给定局部历史，考察模型是否能识别合理计划、补全正确行动、判断 patch 或 citation 的有效性。

对 Code Agent 轨迹数据而言，APTBench 提供了一个清晰的数据再利用方向。完整轨迹可以拆成 `prefix -> next plan/action`、`issue context -> bug location`、`problem -> fix patch selection`、`trajectory -> verification command` 等问题。这样既能评估 base model 的软件工程 agent 潜力，也能辅助诊断模型到底缺在规划、工具参数、代码定位、补丁判断还是测试生成。

但这类转化需要严格控制时序边界。正确答案可以来自当前轨迹的下一步，但 prompt 不能泄漏当前工具调用之后的 observation、后续修正、最终测试结果或最终任务成功信号。否则评测会从“是否具备 agentic decision ability”变成“是否利用了未来信息”。

因此，APTBench 更适合被看作 agent 数据工程和 agent 评测之间的桥梁：它不替代完整 benchmark，但可以在训练成本更低、模型阶段更早的时候提供更相关的方向性信号。

## 相关知识链接

- [[application/evaluation/benchmark|Benchmark]]
- [[application/evaluation/evaluation|Evaluation and Benchmark]]
- [[application/agents/agent|Agent]]
- [[application/tool-use/tool-calling|Tool Calling]]
- [[training/mid-training/continued-pretraining|Continued Pretraining]]
- [[training/data-engineering/synthetic-data|Synthetic Data]]
