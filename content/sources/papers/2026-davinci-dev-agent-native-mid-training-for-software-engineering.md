---
title: "daVinci-Dev: Agent-native Mid-training for Software Engineering"
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
  - code-agent
  - mid-training
  - software-engineering
aliases:
  - daVinci-Dev
  - Agent-native Mid-training
source_url: https://arxiv.org/abs/2601.18418
---

# daVinci-Dev: Agent-native Mid-training for Software Engineering

## 基本信息

| 字段 | 内容 |
|---|---|
| 来源 | arXiv:2601.18418v2 |
| 标题 | daVinci-Dev: Agent-native Mid-training for Software Engineering |
| 作者/机构 | Ji Zeng, Dayuan Fu, Tiantian Mi, Yumin Zhuang, Yaxing Huang, Xuefeng Li, Lyumanshan Ye, Muhang Xie, Qishuo Hua, Zhen Huang, Mohan Jiang, Hanning Wang, Jifan Lin, Yang Xiao, Jie Sun, Yunze Wu, Pengfei Liu / SII, SJTU, GAIR |
| 日期 | 2026-01-27 |
| 链接 | https://arxiv.org/abs/2601.18418 |
| 相关 topic | [[training/mid-training/continued-pretraining|Continued Pretraining]], [[training/mid-training/capability-injection|Capability Injection]], [[training/data-engineering/synthetic-data|Synthetic Data]], [[application/agents/agent|Agent]], [[application/tool-use/tool-calling|Tool Calling]], [[application/evaluation/benchmark|Benchmark]] |

daVinci-Dev 讨论的是 code agent 训练中一个比 SFT / RL 更靠前的阶段：**能否在 mid-training 阶段，通过大规模接近真实软件工程流程的数据，把 agentic software engineering 的基础行为注入 base model**。论文认为，当前 code agent 训练过度依赖 post-training：用少量高质量轨迹做 SFT，再用执行反馈或 RL 进一步优化。这样的路线有效，但受限于可执行环境数量、成功轨迹稀缺、闭源强模型生成成本，以及 base model 本身是否已经具备 agentic reasoning 的能力上限。

论文提出的核心概念是 **agent-native data**：训练数据不应只展示最终代码、最终 patch 或孤立的 localization / editing 子任务，而应尽量保留 agent 在真实开发中经历的信息流、操作顺序和环境反馈。围绕这个概念，daVinci-Dev 构造两类互补数据：contextually-native trajectories 提供大规模、多样化的 PR 级上下文与编辑流程；environmentally-native trajectories 提供来自真实 Docker 环境、工具调用和单元测试的执行反馈。

如果只保留一个核心观点：**软件工程 agent 的 mid-training 数据应当从“静态代码或孤立技能样本”转向“保留 localize-read-edit-test-revise 流程的 agent-native trajectories”。**

## 研究问题

传统代码模型训练通常从静态代码语料、函数级题目或仓库级补全任务中学习代码分布。进入 code agent 场景后，模型面对的问题发生了变化：它不再只是生成一个函数，而是需要阅读 issue、定位相关文件、理解跨文件依赖、应用修改、运行测试、根据错误继续修正，并在多轮工具交互中完成真实软件工程任务。

论文将 agentic software engineering task 形式化为三元组：

$$
(R, q, E)
$$

其中 $R$ 是 repository state，$q$ 是自然语言问题描述，$E$ 是 evaluation oracle，通常是测试套件。在第 $t$ 步，agent 根据历史 $h_{t-1}$ 和问题 $q$ 选择 action，并从环境获得 observation：

$$
a_t \sim \pi_\theta(a \mid h_{t-1}, q)
$$

$$
o_t \sim \mathrm{Obs}(a_t, R)
$$

历史由过去的 action-observation pairs 组成：

$$
h_{t-1} = \{(a_1,o_1), \ldots, (a_{t-1},o_{t-1})\}
$$

这一形式化强调：code agent 的能力不是孤立的代码生成，而是 action-observation loop 中的连续决策。典型流程可以概括为：

```text
localize → read → edit → test → revise
```

现有训练数据与这一部署形态存在明显 distribution mismatch。静态代码语料展示最终代码长什么样，却不展示模型如何从 issue 找到文件、如何选择编辑位置、如何根据测试错误修复。即使一些 mid-training 数据包含 localization 或 editing，也常常把它们拆成独立子任务：localization 假设后续编辑已知，editing 假设 oracle files 已经给定。这种 factorized subtask training 隐藏了 localization 与 editing 之间的依赖关系，和真实 agent 推理链不一致。

daVinci-Dev 的研究问题正是：如何在 mid-training 阶段构造更接近 agent 部署分布的大规模训练数据，使 base model 在进入后续 SFT / RL 前已经具备软件工程 agent 的基础行为模式。

## 核心主张

论文的核心主张有三层。

第一，agentic capability 不应完全留给 post-training 学习。SFT 和 RL 可以对齐具体工具协议、轨迹格式和成功行为，但它们依赖可执行环境和成功轨迹，数据规模通常远小于预训练或中训练阶段。若 base model 在 mid-training 前缺少 agentic inductive bias，post-training 需要同时学习软件工程知识、工具行为、长程规划和反馈利用，优化压力会很大。

第二，有效的 agentic mid-training 需要 **agent-native data**。这里的 native 不是指数据一定由 agent 生成，而是指数据组织方式要保留 agent 在真实任务中可见的信息结构。模型应在训练中看到问题描述、相关文件、编辑过程、工具调用、测试反馈和修正路径之间的关系，而不是只看到最终 diff 或单个子任务标签。

第三，contextual breadth 与 environmental authenticity 需要互补。PR 数据规模大、覆盖广，适合学习软件工程语境、文件修改模式和多样化任务；但 PR 数据本身没有真实执行反馈。环境 rollout 数据有真实工具调用、测试失败和修正过程，但数量较小、构造成本更高。因此论文将二者组合：用 contextually-native trajectories 建立广泛的软件工程先验，用 environmentally-native trajectories 注入真实 action-observation 动态。

## 方法与机制

### Agent-native Data

论文用 agent-native data 描述一种训练数据设计原则：监督信号应尽量保留 agent 在部署时经历的完整信息流，而不是把问题拆成静态片段。

这一定义包含两个层次：

| 数据类型 | 主要目标 | 核心特征 |
|---|---|---|
| Contextually-native trajectories | 覆盖与多样性 | 从 GitHub PR 重构任务上下文、相关文件和按时间排列的编辑序列 |
| Environmentally-native trajectories | 交互真实性 | 在真实可执行仓库环境中运行 agent，记录工具调用、测试执行、错误反馈和修正 |

二者分别解决不同问题。Contextually-native data 缓解静态代码语料缺少任务上下文的问题；environmentally-native data 缓解 PR 重构数据缺少真实环境反馈的问题。组合后，模型既能接触大量真实开发历史，也能看到执行环境如何影响后续行动。

### Contextually-native Trajectories

Contextually-native trajectories 以 GitHub Pull Requests 为基础。PR 天然连接 issue、代码修改、review、commit history 和最终合并结果，是从真实软件工程历史中恢复 agent-like workflow 的高价值来源。

论文构造两个子集：

| 子集 | 规模 | 目的 |
|---|---:|---|
| $D^{ctx}_{gen}$ | 26.7B tokens | 来自 top-starred repositories，覆盖多语言、多框架软件工程模式 |
| $D^{ctx}_{py}$ | 41.9B tokens | 来自 Python repositories，对齐 SWE-Bench Verified 等 Python 软件工程评测 |
| $D^{ctx}$ | 68.6B tokens | 两者合并 |

构造流程包含三个关键步骤。

第一，收集 PR 元信息、linked issue、相关文件内容和 commit sequence。相关文件不是由模型猜测，而是通过 base commit 与 head commit 之间的 symmetric diff 确定修改文件集合，再回到 PR 初始状态读取完整文件内容。论文还特别指出，文件内容和 patch 要对齐到第一个 PR commit 的 parent，而不是简单使用 PR metadata 中记录的 base commit，以减少状态错位。

第二，进行内容增强。论文使用 Qwen3-235B-A22B-Instruct-2507 生成 PR summary，并将过短或信息不足的 commit message 改写成更清晰的描述。这些 LLM 生成内容在样本中充当高层计划或局部 reasoning 的近似文本，但核心修改内容仍来自真实 PR。

第三，按模板组织信息。General PR format 使用 XML-like tags，并包含 developer comments 与 review threads；Python PR format 使用 Markdown 结构，并把 diff 重写为许多 agent scaffold 常用的 search-and-replace edit action。一个 Python PR 样本通常包括：

```text
Repository Context
Issue
Pull Request
Relevant Files Found
LLM-generated Summary
Edits
```

这种组织方式对应 code agent 的 localize-read-edit 流程：relevant file paths 模拟 localization，完整文件内容模拟 reading，commit edits 模拟 editing，summary 和 refined commit messages 模拟中间 reasoning。它不是严格意义上的真实 agent rollout，但比静态 diff 更接近 agent 在推理时可见的上下文结构。

论文还做了两类过滤：长度上丢弃超过 32K tokens 的样本；泄漏控制上移除 SWE-Bench Verified 涉及仓库中的 PR，以降低 benchmark contamination。

### Environmentally-native Trajectories

Contextually-native trajectories 保留了开发上下文，却仍然缺少 edit-test-revise 的真实动态。为弥补这一点，论文构造 environmentally-native trajectories，即在可执行仓库环境中运行 agent 并记录真实 observation。

数据构造参考 SWE-REBENCH 的环境构建方法。对每个任务，系统构建 Docker image，复现特定 commit 下的仓库状态，并保留真实 unit tests。随后在 SWE-AGENT scaffold 中使用 GLM-4.6 生成最多 4 条 rollout。每条轨迹记录完整 action-observation sequence，包括文件搜索、文件读取、编辑、shell command、测试结果、运行时错误和 scaffold 反馈。

过滤后，论文将轨迹按最终测试结果分为：

| 数据 | 规模 | 含义 |
|---|---:|---|
| $D^{env}_{pass}$ | 1.85 万条，0.7B tokens | 最终通过测试的轨迹 |
| $D^{env}_{fail}$ | 5.55 万条，2.4B tokens | 最终未通过测试但包含真实错误反馈的轨迹 |
| $D^{env}$ | 约 7.4 万条，3.1B tokens | 全部环境交互轨迹 |

论文认为 passing trajectories 展示完整解决路径，non-passing trajectories 则展示真实 debugging 情境中的错误反馈。训练时 $D^{env}_{pass}$ 会被上采样 3 倍，使 $D^{env}$ 的有效 token 量约为 4.5B。

这个设计的关键在于 observation 不是 retrospective explanation，也不是模拟文本，而是工具和测试系统真实返回的内容。因此它更适合训练模型理解“编辑导致测试失败，失败信息又应该如何引导下一步修正”。

### 训练阶段

论文区分三类阶段：

| 阶段 | 含义 |
|---|---|
| Pre-training | 大规模多样语料上的 next-token prediction |
| Mid-training | 在 curated domain data 上继续训练，用于改变能力分布 |
| Post-training | SFT 或 RL，用于学习具体演示、工具格式和行为对齐 |

daVinci-Dev 的主要训练对象是 Qwen2.5-32B-Base 和 Qwen2.5-72B-Base。Mid-training 使用 next-token prediction，global batch size 为 1024，peak learning rate 为 $8\times 10^{-5}$，warmup ratio 为 0.05，训练 1 epoch，且不使用 loss mask。SFT 使用 global batch size 128，peak learning rate $1\times 10^{-5}$，warmup ratio 0.10，并对 user 和 tool tokens 使用标准 loss mask。

$D^{ctx}$ 的训练采用分阶段方式：先训练 $D^{ctx}_{gen}$ 以建立广泛软件工程基础，再训练 $D^{ctx}_{py}$ 以对齐 Python-centric agent 任务。如果使用 $D^{ctx}+D^{env}$，第一阶段仍是 $D^{ctx}_{gen}$，第二阶段混合 $D^{ctx}_{py}$ 与 $D^{env}$。

## 实验与证据

### SWE-Bench Verified 主结果

论文在 SWE-Bench Verified 上使用 SWE-AGENT scaffold 评测，temperature 为 0，context length 为 128K，最多 100 steps，报告 4 次运行平均 Pass@1。论文手动修复少量 ground truth patch 无法通过的问题，因此结果依赖其 patched evaluation harness，这一点在局限中也被作者承认。

核心消融结果如下：

| Model / Variant | MT Data | Post-training Data | Method | SWE-V |
|---|---|---|---|---:|
| Qwen2.5-32B Baseline (Weak SFT) | - | $D_{SWE-smith}$ | SFT | 34.8 |
| Qwen2.5-32B Ours (Weak SFT) | $D^{ctx}$ | $D_{SWE-smith}$ | SFT | 39.5 |
| Qwen2.5-32B Baseline (Strong SFT) | - | $D^{env}_{pass}$ | SFT | 53.0 |
| Qwen2.5-32B Ours (Strong SFT) | $D^{ctx}$ | $D^{env}_{pass}$ | SFT | 54.1 |
| daVinci-Dev-32B | $D^{ctx}+D^{env}$ | $D^{env}_{pass}$ | SFT | 56.1 |
| Qwen2.5-72B Baseline (Weak SFT) | - | $D_{SWE-smith}$ | SFT | 38.0 |
| Qwen2.5-72B Ours (Weak SFT) | $D^{ctx}$ | $D_{SWE-smith}$ | SFT | 46.4 |
| Qwen2.5-72B Baseline (Strong SFT) | - | $D^{env}_{pass}$ | SFT | 56.6 |
| Qwen2.5-72B Ours (Strong SFT) | $D^{ctx}$ | $D^{env}_{pass}$ | SFT | 58.2 |
| daVinci-Dev-72B | $D^{ctx}+D^{env}$ | $D^{env}_{pass}$ | SFT | 58.5 |

这些结果说明：即使后续 SFT 已经使用强轨迹数据，agent-native MT 仍能继续提升 SWE-Bench Verified 表现。72B 弱 SFT 设置下，$D^{ctx}$ 将分数从 38.0 提升到 46.4；强 SFT 设置下，$D^{ctx}$ 从 56.6 提升到 58.2；加入 $D^{env}$ 后进一步达到 58.5。32B 模型也呈现同样趋势。

与 Kimi-Dev 对比时，论文强调 daVinci-Dev 使用更少 MT tokens。Kimi-Dev 约使用 150B tokens，包括从 PR 得到的数据和上采样 synthetic trajectory / CoT；daVinci-Dev 使用 68.6B $D^{ctx}$，完整配方为 73.1B tokens。72B 上 daVinci-Dev 达到 58.5，高于论文引用的 Kimi-Dev 48.6，以及在作者基础设施中用更强 SFT 数据适配的 Kimi-Dev 56.2。

### 与开放训练配方比较

论文还将 daVinci-Dev 与代表性 open code agent recipes 比较。32B 规模下，daVinci-Dev-32B 达到 56.1，高于 FrogBoss 54.6、SWE-Lego-Qwen3-32B 52.6、SWE-Agent-LM 40.2 等结果。72B 规模下，daVinci-Dev-72B 达到 58.5，高于 Kimi-Dev 的 48.6。

需要注意，这类横向比较仍受 base model、scaffold、评测环境、是否使用 RL、是否使用 instruct model 等因素影响。论文的更强证据来自同一 scaffold 和相近 SFT 设置下的消融，而不是跨系统排行榜式比较。

### 泛化到代码生成与科学 benchmark

论文还报告了 base model 经过 $D^{ctx}_{py}+D^{env}$ mid-training 后在一般代码和科学 benchmark 上的变化。

| Benchmark | 32B Base | 32B MT Mix | Δ | 72B Base | 72B MT Mix | Δ |
|---|---:|---:|---:|---:|---:|---:|
| GPQA-Main | 38.17 | 38.84 | +0.67 | 43.30 | 44.87 | +1.57 |
| SuperGPQA | 33.85 | 35.94 | +2.09 | 37.76 | 39.27 | +1.51 |
| SciBench | 18.46 | 20.49 | +2.03 | 19.33 | 19.77 | +0.44 |
| HumanEval | 58.16 | 81.42 | +23.26 | 64.27 | 76.73 | +12.46 |
| EvalPlus | 50.13 | 71.31 | +21.18 | 56.04 | 69.45 | +13.41 |
| DS-1000 | 12.2 | 21.2 | +9.0 | 21.4 | 24.7 | +3.3 |

代码 benchmark 的提升并不意外，因为训练数据高度相关；科学 benchmark 的小幅提升更像是间接证据，说明 agentic software engineering 数据中的多步问题分解、证据整合和错误修正模式可能对一般复杂推理有迁移，但不能过度解释为通用科学能力的大幅改善。

### 数据组件消融

论文最重要的分析是 $D^{ctx}$ 与 $D^{env}$ 的协同关系。

在 zero-shot / no SFT 设置下，仅使用 $D^{env}$ 做 MT，72B 在 SWE-Bench Verified 上达到 47.1；加入 $D^{ctx}_{py}$ 后提升到 54.8。论文据此认为，环境轨迹可以教会模型如何交互，但 PR 数据提供了更广泛的软件工程知识和代码修改多样性。

在有 SFT 的设置下，72B 使用 $D^{ctx}_{py}$ 做 MT 后再用 $D^{env}_{pass}$ SFT，得分为 56.5；若 MT 阶段加入 $D^{env}$，得分提升到 57.8；进一步扩展到完整 $D^{ctx}+D^{env}$，得分达到 58.5。这说明 trajectory data 不只在 SFT 阶段有用，在 MT 阶段提前暴露给模型也能改善后续 SFT 初始化。

这个消融支持一个比较清晰的结论：**contextually-native PR data 是性能的主要规模来源，environmentally-native rollout data 是交互动态的补充，两者不是替代关系。**

### Scaling Law

论文在 $D^{ctx}_{py}+D^{env}$ 混合数据上观察 MT 训练步数与 SWE-Bench Verified Pass@1 的关系。32B 和 72B 都呈现较强 log-linear 拟合，$R^2$ 约为 0.89 和 0.90。72B 在 MT 过程中提升到 54.9，32B 提升到 49.9，并且曲线没有明显饱和。

这一结果是论文对“agent-native mid-training 可扩展”的主要证据。它说明模型能把额外 agent-native 数据和训练步数转化为 agent benchmark 提升，而不是只在早期获得有限收益。不过该 scaling 仍只在 Qwen2.5 系列和 SWE-Bench Verified 上验证，外推到其他模型族或更广泛 agent 任务仍需要谨慎。

## 关键结论

daVinci-Dev 的主要贡献是把 code agent 训练从 post-training-only 视角推进到 mid-training 数据设计。论文不是简单地增加代码数据，而是强调数据形态必须接近 agent 的真实工作流：问题、定位、读取、编辑、测试和修正应被组织成连续信息流。

论文提出的 agent-native data 概念具有较强可复用性。对于 code agent 而言，高质量训练数据不只是“有没有 patch”，而是是否保留了 patch 产生前的上下文、相关文件、编辑顺序和环境反馈。静态 PR 可以通过重构变成 contextually-native trajectories；可执行仓库可以通过 agent rollout 变成 environmentally-native trajectories。前者提供广度，后者提供深度。

实验上，daVinci-Dev 表明 agent-native MT 能在强 SFT 之外继续带来增益。尤其是 72B 模型在同一 SWE-AGENT scaffold 下达到 58.5，说明在后训练前塑造 base model 的软件工程 agent 行为分布是有效的。这个结果与 AgentFounder 的 Agentic CPT 路线形成呼应：agent 能力可以在 SFT/RL 前被提前注入，只是 daVinci-Dev 更聚焦 code agent 和 PR / execution trajectory 数据。

## 局限与疑问

第一，评测范围仍然偏窄。论文主要在 Qwen2.5-32B / 72B 和 SWE-Bench Verified 上验证，尚未证明 agent-native MT 对其他 base model family、其他编程语言、OpenHands scaffold、Terminal-Bench 或真实企业代码库同样有效。

第二，横向比较存在系统差异。不同 work 使用不同 base model、instruct 状态、SFT 数据、RL 设置、scaffold 和评测 harness。daVinci-Dev 的同配方消融更可信，跨方法表格应作为背景参考，而不是严格同条件排名。

第三，数据隐私与归因需要更细处理。作者承认 $D^{ctx}_{gen}$ 没有显式移除 PR 文本中的开发者标识，可能带来隐私、署名和记忆风险。公共 PR 虽然可访问，但大规模模型训练仍需要考虑个人信息和 license / attribution。

第四，PR 重构数据并不等同于真实 agent 轨迹。Contextually-native trajectories 通过模板恢复 localize-read-edit 结构，但 relevant files 来自最终 diff 的反向确定，PR summary 和 commit message refinement 也由 LLM 后处理生成。这种数据比静态 diff 更接近 agent workflow，但仍可能包含后验信息和 teacher 风格偏差。

第五，environmentally-native 数据规模仍相对较小。$D^{env}$ 只有约 3.1B raw tokens，且由 GLM-4.6 在 SWE-AGENT 中生成。它提供真实 execution feedback，但也继承了特定 scaffold、teacher agent 和测试环境的偏好。未来若扩展到更大规模，需要处理环境构建成本、测试可靠性和失败轨迹质量控制。

第六，patched evaluation harness 会引入额外方差。论文手动修复少量 benchmark 问题以解决 ground truth patch 无法通过的情况，这可能是合理工程处理，但也意味着结果与原始 SWE-Bench Verified 运行条件不完全一致。

## 分析与启发

daVinci-Dev 对 code agent 数据工程的启发非常直接：真实 PR 不应只被当作最终 patch 语料，而可以被重构为近似 agent 工作流的中训练样本。相比 “issue -> patch” 或 “oracle file -> edit” 这种短路径监督，PR 级数据更适合表达软件工程任务中的条件依赖：为什么定位这些文件，为什么读取这些上下文，为什么这些编辑按某个顺序发生。

这篇论文也补充了 Agentic CPT / APTBench 所指向的训练闭环。AgentFounder 说明 agentic behavior 可以前移到 continued pretraining；APTBench 说明 base model 的 agentic potential 可以通过 trajectory-conditioned tasks 早期评测；daVinci-Dev 则给出 code agent 领域更具体的数据答案：用 contextually-native PR trajectories 提供广度，用 environmentally-native executable rollouts 提供真实反馈，再通过 SWE-Bench Verified 检查是否提升后续 agent 能力。

对后续构建 code agent 训练数据而言，最关键的不是简单扩充轨迹数量，而是区分数据的“上下文真实性”和“环境真实性”。静态历史数据通常上下文丰富、规模大，但缺少真实 observation；执行 rollout 观测真实，但成本高、覆盖窄。可扩展路线应当同时利用二者，并在训练阶段明确它们分别承担的角色。

需要保持审慎的是，contextually-native PR 样本存在天然后验性：相关文件由最终 diff 反推，编辑顺序来自 commit history，summary 由 LLM 生成。它适合作为 mid-training 的能力塑形数据，但如果用于严格评测或 step-level reasoning 监督，就必须额外控制 future leakage。对于 agent 训练，这一边界非常重要。

## 相关知识链接

- [[training/mid-training/continued-pretraining|Continued Pretraining]]
- [[training/mid-training/capability-injection|Capability Injection]]
- [[training/data-engineering/synthetic-data|Synthetic Data]]
- [[application/agents/agent|Agent]]
- [[application/tool-use/tool-calling|Tool Calling]]
- [[application/evaluation/benchmark|Benchmark]]
