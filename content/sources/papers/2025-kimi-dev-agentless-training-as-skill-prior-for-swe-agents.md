---
title: "Kimi-Dev: Agentless Training as Skill Prior for SWE-Agents"
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
  - code-agent
  - swe-agent
  - agentless
  - reinforcement-learning
  - mid-training
aliases:
  - Kimi-Dev
source_url: https://arxiv.org/abs/2509.23045
paper_date: "2025-09"
paper_order: "23045"
---

# Kimi-Dev: Agentless Training as Skill Prior for SWE-Agents

## 基本信息

| 字段 | 内容 |
|---|---|
| 来源 | arXiv:2509.23045v3 |
| 标题 | Kimi-Dev: Agentless Training as Skill Prior for SWE-Agents |
| 作者/机构 | Zonghan Yang, Shengjie Wang, Kelin Fu, Wenyang He, Weimin Xiong, Yibo Liu, Yibo Miao, Bofei Gao, Yejie Wang, Yingwei Ma, Yanhao Li, Yue Liu, Zhenxing Hu, Kaitai Zhang, Shuyi Wang, Huarong Chen, Flood Sung, Yang Liu, Yang Gao, Zhilin Yang, Tianyu Liu / Moonshot AI, THU, PKU, UCAS, BUPT, NUS |
| 日期 | 2025-12-08 |
| 链接 | https://arxiv.org/abs/2509.23045 |
| 代码与模型 | https://github.com/MoonshotAI/Kimi-Dev；https://huggingface.co/moonshotai/Kimi-Dev-72B |
| 相关 topic | [[application/agents/agent|Agent]], [[application/agents/workflow-agent|Workflow Agent]], [[training/mid-training/capability-injection|Capability Injection]], [[training/mid-training/continued-pretraining|Continued Pretraining]], [[training/data-engineering/synthetic-data|Synthetic Data]], [[training/post-training/grpo|GRPO]], [[application/evaluation/benchmark|Benchmark]] |

Kimi-Dev 讨论的是 software engineering agent 训练中的一个关键关系：固定流程的 Agentless 方法和多轮交互的 SWE-Agent 方法并不是互斥路线。论文的中心观点是，Agentless workflow 虽然在推理时缺少多轮自由探索能力，但它可以作为训练阶段的结构化能力注入方式，为后续 SWE-Agent 适配提供 localization、code edit、test writing、self-reflection 等 skill priors。

论文发布了一个基于 Qwen2.5-72B-Base 训练的开源 SWE LLM。Kimi-Dev 在 Agentless-like workflow 下达到 SWE-bench Verified 60.4%，并在使用 5,016 条公开 SWE-Agent trajectories 做轻量 SFT 后，在 SWE-Agent 框架下达到 48.6% pass@1，接近论文引用的 Claude 3.5 Sonnet 20241022 版本在同一类 SWE-Agent 设置下的 49.0%。

如果只保留一个核心观点：**Agentless training 不必被看作最终部署形态，而可以被看作面向 SWE-Agent 的 skill-prior induction；固定流程训练出的定位、编辑、测试和反思能力，可以降低后续长程 agent 训练的初始化难度。**

## 研究问题

SWE-bench 类任务要求模型根据真实 GitHub issue 修改代码，并通过单元测试验证修复是否正确。围绕这类任务，社区形成了两种典型范式。

一种是 Agentless / workflow-based 方法。它把 GitHub issue resolution 拆成固定步骤，例如 bug localization、bug repair、test generation。每个步骤通常是 single-turn 问题，便于单独优化，也容易接入 execution-based verifier 或 RLVR。它的弱点是流程刚性高，遇到需要多轮探索、反复阅读、修正和验证的问题时，灵活性不足。

另一种是 SWE-Agent / agentic framework。它让模型在可执行环境中多轮使用工具，自主探索仓库、写复现脚本、编辑代码、运行测试并决定何时提交。它更接近真实开发过程，但训练难度更高：trajectory 往往很长，reward 稀疏，credit assignment 困难，工具使用和循环行为对初始化很敏感。

论文挑战的是二者的二分法。Agentless 方法不一定要和 SWE-Agent 方法竞争最终形态；它也可以作为训练前置阶段，把复杂 agent 任务拆解成可验证、可强化的原子技能，再把这些技能迁移到多轮 agent 框架。

研究问题可以概括为：

```text
能否先用 Agentless workflow 训练 localization、repair、test writing 和 self-reflection 等原子能力，
再用少量 SWE-Agent 轨迹把这些能力迁移到端到端多轮 agent？
```

## 核心主张

论文的核心主张有三层。

第一，Agentless workflow 的价值不只在推理时的固定流程，也在训练时的结构化监督。固定 workflow 将长程 SWE 任务拆成 localization、code edit、test composition 等单步问题，使每一步都更容易构造 verifier 和 outcome reward。这种结构化训练可以把软件工程任务中的关键技能提前注入模型。

第二，Agentless training 诱导的不是普通领域知识，而是 skill priors。论文重点强调的 priors 包括 BugFixer 能力、TestWriter 能力和 self-reflection。它们不是完整 agent policy，但可以作为后续 SWE-Agent SFT 或 RL 的更好初始化。

第三，long CoT 中形成的 self-reflection 可以迁移到多轮交互。论文观察到，经过 RL 的 prior 在 SWE-Agent 适配后，能在更长 turn limit 下继续获益；相比原始 base 或仅 mid-trained 模型，它更能利用多轮行动中的 test rerun、错误反馈和修正机会。

## 方法与机制

### BugFixer 与 TestWriter

Kimi-Dev 的 Agentless 框架由两个互补角色组成：

| 角色 | 目标 | 核心技能 |
|---|---|---|
| BugFixer | 生成修复 GitHub issue 的 patch | file localization、code edit |
| TestWriter | 生成能复现 issue 的测试 | test file localization、test edit |

BugFixer 的成功标准是生成 patch 后通过 ground-truth unit tests。TestWriter 的成功标准更特殊：生成的测试应在未应用 ground-truth bugfix patch 的仓库中失败，并在应用 bugfix 后通过。换言之，TestWriter 不只是写任意测试，而是写能够区分 buggy state 与 fixed state 的 reproduction test。

这种 duo 设计把 SWE issue resolution 分成两个互相校验的能力：修复能力和测试构造能力。后续 test-time self-play 正是利用 TestWriter 生成的测试来筛选 BugFixer 的候选 patch。

### Mid-training 与 Cold-start

Kimi-Dev 从 Qwen2.5-72B-Base 开始，使用约 150B tokens 的高质量真实软件工程数据做 mid-training。论文概述的数据包括：

- 约 50B tokens 的 Agentless 格式数据，由自然 diff patch 派生；
- 约 20B tokens 的 curated PR commit packs；
- 约 20B tokens 的 synthetic data，包含 reasoning 与 agentic interaction patterns，并在训练中上采样 4 倍。

附录进一步说明，mid-training 使用标准 next-token prediction，最大序列长度 32K，global batch size 256，learning rate 为 $2\times 10^{-5}$，minimum learning rate 为 $2\times 10^{-6}$，warmup 约 3B tokens，并在约 150B tokens 内完成衰减。论文强调做了严格 decontamination，排除了 SWE-bench Verified test set 涉及仓库。

其中 agentic interaction data 的构造方式很有代表性。论文使用一组 non-execution tools 模拟 agentic trajectory：`open`、`find_file`、`str_replace`、`insert`、`stop` 等。Stage 1 做 localization；Stage 2 进入 code editing，并允许模型打开错误定位文件。系统会人为注入“读取后意识到不需要修改该文件”这类 self-reflection pattern，用 false-positive contexts 训练模型反思。随后将 ground-truth PR commit pack 转写为 trajectory：commit message 作为 reasoning step，代码修改作为 `str_replace` 或 `insert` action，最后接 `stop`。这部分贡献约 10B tokens。

Cold-start 阶段用于激活 long CoT 能力。论文使用 SWE-Gym 和 SWE-bench-extra 构造 reasoning trajectories，由 DeepSeek R1 20250120 版本扮演 BugFixer 和 TestWriter，生成 file localization、code edit、problem analysis、method sketching、self-refinement 和 alternative solutions。通过这一 SFT cold-start，模型获得更强的 reasoning-intensive 输出模式。

### Agentless RL

在 mid-training 和 cold-start 后，模型已经具备较强 localization 能力，因此 RL 主要聚焦 code edit 阶段。每个 prompt 都配备可执行环境；同时用初始模型的多个 localization rollouts 产生不同文件定位结果，以增加 code-edit RL 的 prompt 多样性。

论文采用 Kimi k1.5 风格的 policy optimization。它基于 REINFORCE，并类似 [[training/post-training/grpo|GRPO]] 用同一 prompt 下多个 rollouts 的平均 reward 作为 baseline 归一化 return。Kimi-Dev 对 SWE 场景强调三点。

第一，只使用 outcome-based reward。BugFixer 生成 patch 后通过 ground-truth unit tests 得 1，否则为 0；TestWriter 生成的测试必须在未修复仓库中失败，并在应用 ground-truth bugfix 后通过。论文明确不加入格式或过程 reward。

第二，采用 adaptive prompt selection。初始时丢弃 pass@16 = 0 的 prompt，因为它们对 batch loss 贡献很小；当当前 prompt set 的成功率超过阈值后，每 100 RL steps 重新引入 500 个之前 pass@16 = 0 但当前模型已有改善的 harder prompts，形成 curriculum。

第三，使用 positive example reinforcement。当训练后期收益趋缓时，把最近 RL iterations 中的 positive samples 加入当前 batch，强化成功模式并加快收敛。附录在 14B 模型上也观察到类似 RL scaling 行为。

论文的 RL 实验使用 SWE-Gym、SWE-bench-extra 和 R2E-Gym-Lite 等来源的 Docker 环境。Appendix B/C/D 说明其 sandbox infrastructure 基于 Kubernetes，可支持超过 10,000 个并发实例，并管理超过 25,000 个 Docker images。SWE-bench-extra 中 6.38k instances 里有 3,846 个环境被成功构造并用于 cold-start 和 RL。

### Test-time Self-play

Kimi-Dev 在 Agentless 推理时使用 test-time self-play 来组合 BugFixer 和 TestWriter。对每个 SWE-bench Verified instance，模型生成 40 个候选 patches 和 40 个 candidate tests。第一个 patch/test 使用 greedy decoding，其余使用 temperature 1.0 以增强多样性。

首先过滤 TestWriter 输出，只保留能在原始未修复仓库中触发失败的 test patches。然后对每个 BugFixer patch $b_i$ 和 TestWriter patch $t_j$ 进行两次执行：一次不应用 $b_i$，一次应用 $b_i$。从第一次执行得到 $t_j$ 下 failed tests 数 $F(j)$ 和 passed tests 数 $P(j)$；比较两次执行得到 fail-to-pass 数 $FP(i,j)$ 与 pass-to-pass 数 $PP(i,j)$。论文用下式为每个 patch 打分：

$$
S_i =
\frac{\sum_j FP(i,j)}{\sum_j F(j)}
+
\frac{\sum_j PP(i,j)}{\sum_j P(j)}
$$

第一项近似衡量 patch 是否修复 reproduction tests；第二项近似衡量 patch 是否不破坏 regression tests。最终选择 $S_i$ 最高的 BugFixer patch。

这个机制的意义在于，它把 TestWriter 从训练辅助角色变成推理时的 patch selector。实验显示，从 $1\times1$ patch-test pair 增加到 $40\times40$ 后，SWE-bench Verified 表现从 48.0% 提升到 60.4%，并且明显优于仅对 BugFixer patches 做 majority voting。

### SWE-Agent 适配

论文进一步验证 Agentless skill priors 是否能迁移到多轮 SWE-Agent。它使用 SWE-smith 发布的 5,016 条公开 SWE-Agent trajectories 对 Kimi-Dev 做 SFT。这些轨迹由 Claude 3.7 Sonnet 在合成环境中收集。训练时最大 context length 为 64K，推理时允许 128K context 和最多 100 turns。

为了分析不同训练阶段的 prior 质量，论文比较四个初始化：

| Prior | 含义 |
|---|---|
| Base | 原始 Qwen2.5-72B |
| MT | Agentless mid-trained model |
| SFT | MT 后用 long CoT cold-start 激活 |
| RL | 完成 Agentless RL 后的 Kimi-Dev |

然后使用不同规模的 SWE-Agent SFT trajectories 对它们做轻量适配，包括 zero-shot、单步梯度、100、200、500、1000、2000 和 5016 trajectories。实验显示，RL prior 在几乎所有 SWE-Agent SFT 数据规模下优于其他 prior；达到 Base prior 最高 pass@1 表现时，RL prior 只需要约 $2^{23}$ SWE-Agent SFT tokens，而 Base prior 需要 $1.5\times 2^{28}$ tokens。

论文还使用 turn limit 分析 long CoT 到 multi-turn interaction 的迁移。RL prior 在适配后超过 70 turns 仍继续受益，而 SFT、MT、Base 分别在约 70、60、50 turns 附近收益趋缓。作者据此认为，Agentless RL 中形成的 self-reflection 能帮助 SWE-Agent 利用更长交互。

### End-to-end SWE-Agent RL

最后，论文用 end-to-end SWE-Agent RL 比较不同 prior。为了减少 SWE-smith 轨迹中 proprietary model pattern 的影响，每个 prior 只做极少量 SWE-Agent SFT cold-start，然后进行 agentic RL。RL 仍使用 outcome reward only，不加入 KL 或 entropy regularization。

结果显示，MT prior 可用训练问题太少，end-to-end RL 很快退化；SFT prior 和 RL prior 可以持续训练 300 steps，而 RL prior 在 pass@1、pass@3、pass@5 上略优于 SFT prior。这说明 Agentless RL 不只改善 Agentless workflow 本身，也提高了后续端到端 agentic RL 的起点。

## 实验与证据

### Agentless SWE-bench Verified

在 Agentless-like workflow 下，Kimi-Dev 在 SWE-bench Verified 上达到 60.4%，高于论文列出的其他 workflow-based 方法与开放模型基线：

| Model | #Params | Resolve Rate |
|---|---:|---:|
| Llama3-SWE-RL | 70B | 41.0 |
| Seed1.5-Thinking | 200B | 47.0 |
| OpenAI-o1 | - | 48.9 |
| DeepSeek-R1-0120 | 671B | 49.2 |
| Claude 3.5 Sonnet 20241022 | - | 50.8 |
| MiniMax-M1 | 456B | 56.0 |
| DeepSeek-R1-0528 | 671B | 57.6 |
| SWE-SWISS | 32B | 58.2 |
| Kimi-Dev | 72B | 60.4 |

这一结果来自标准 40 patch、40 test 设置。论文认为，执行信号比文本相似度 reward 更可靠；同时两阶段 TestWriter 比单一 root-level test 更能反映仓库上下文和真实开发流程。

### Mid-training Scaling

论文用 50B、100B 和约 150B tokens 的 mid-training 子集训练 Qwen2.5-72B-Base，然后用同一组 2,000 BugFixer input-output pairs 做轻量 SFT activation。Figure 2 显示，mid-training token 越多，BugFixer pass@1 越高。这为“Agentless mid-training 可以持续注入 SWE skill prior”提供了规模化证据。

### RL Scaling 与长度增长

Figure 3 显示，BugFixer 和 TestWriter 在 joint code-edit RL 中，pass rate / reproduced rate 随训练持续提升，同时 response length 也增长。论文把增长解释为更深入 problem analysis 和 self-reflection 的出现，类似数学和代码 reasoning RL 中常见的长 CoT 扩展。

这里需要保守理解：长度增长不必然等于质量提升；但在该论文的 setting 中，它与 pass rate 同步提升，说明 RL 至少在训练分布内强化了较长的分析和反思模式。作者也指出 TestWriter 可能出现 false positives，即生成测试没有覆盖关键语义差异，却通过 reward check。

### Self-play 与 TestWriter 价值

Figure 4 表明，test-time self-play 随 patch/test 数量增加而提升。$1\times1$ 为 48.0%，$40\times40$ 为 60.4%；仅 $3\times3$ self-play 就超过 40 个 BugFixer patches 的 majority voting。右图还显示，使用 model-generated TestWriter tests 的 self-play 仍低于用 ground-truth tests 作为选择标准的 pass@N，说明 TestWriter 仍是瓶颈和改进空间。

附录 F 还观察到 emergent parallel scaling：把多个 BugFixer 候选 patch 放入 prompt，让模型分析候选差异并综合生成新 patch，随着候选数增加表现提升。与 majority voting 相比，这种 aggregation 利用了模型对候选 patch 的语义比较能力，而不是只统计频率。论文将其作为初步现象而非主方法。

### SWE-Agent 迁移

在 SWE-Agent 框架下，Kimi-Dev 经 5,016 条公开 SWE-Agent trajectories SFT 后达到 48.6% pass@1。论文强调，这不需要额外收集真实环境 trajectories，也未做多轮 agentic RL。横向看，它高于 SWE-agent-LM 40.2 和 DeepSWE 42.2，接近 Claude 3.5 Sonnet 20241022 的 49.0，但低于更强 proprietary models、Kimi-K2、Qwen3-Coder 等大型 agent 模型。

更重要的证据来自 prior comparison。Figure 5 显示，RL prior 在低数据和较高数据设置下都更适合作为 SWE-Agent adaptation 起点。Figure 6 的 turn-limit 分析进一步显示，RL prior 更能从长交互中获益。阶段标注实验显示，BugFixer skill 从 Base 的 484 个 Stage-3 cutoff 成功案例提升到 RL prior 的 605 个；reflection skill 的从 Stage-3 到最终完成的增量也从 +94 提升到 +113。

论文还在 SWE-bench-Live 和 SWE-bench Multilingual 上做泛化分析。结果显示，在数据稀缺的 SWE-Agent SFT 设置下，SFT/RL priors 比 Base 更好；但当使用更多 SWE-smith trajectories 后，Base/MT 与 SFT/RL 的差距缩小。作者将更强 out-of-distribution 和 task-agnostic generalization 留作未来工作。

## 关键结论

Kimi-Dev 的主要贡献是重新解释 Agentless 与 SWE-Agent 的关系。Agentless workflow 的固定流程确实限制推理时灵活性，但在训练阶段，它提供了结构化、可验证、可强化的分解方式。通过 BugFixer 和 TestWriter，模型可以分别学习定位、修复、构造复现测试和反思验证；这些能力再通过少量 SWE-Agent 轨迹转移到多轮 agent 框架。

这一观点与近期 code agent 训练工作形成互补。[[sources/papers/2026-davinci-dev-agent-native-mid-training-for-software-engineering|daVinci-Dev]] 强调 agent-native mid-training 需要保留 localize-read-edit-test-revise 信息流；[[sources/papers/2026-qwen3-coder-next-technical-report|Qwen3-Coder-Next]] 强调可执行任务规模化、多 scaffold 轨迹和工具模板多样性；Kimi-Dev 则说明，即使训练数据仍是 Agentless / workflow 形式，也可以通过结构化技能训练为后续 agent adaptation 提供 prior。

对 code agent 数据建设而言，论文给出一个可复用分层思路：

- 先用 workflow 把复杂 SWE 任务拆成 localization、repair、test writing、reflection 等可验证技能；
- 再用 long CoT cold-start 和 execution-only RL 强化这些技能；
- 最后用少量真实 agent trajectory 对齐工具协议和多轮交互；
- 在推理时，workflow self-play 和 agentic rollout 可以分别作为不同成本/灵活性折中的部署方式。

## 局限与疑问

第一，Agentless workflow 的高分强依赖 test-time compute。Kimi-Dev 的 60.4% 来自 40 patches × 40 tests 的 self-play，而不是单次生成。它证明了 BugFixer/TestWriter 协同有效，但成本、延迟和执行环境可用性需要单独评估。

第二，TestWriter reward 存在 false-positive 风险。论文附录给出案例，模型生成的测试可能覆盖不完整，导致 reward check 误判。这是所有“用模型生成测试筛选 patch”的方法都需要面对的问题：测试越弱，patch selector 越容易选择过拟合测试的修复。

第三，SWE-Agent 迁移实验仍主要围绕 SWE-bench Verified、SWE-smith trajectories 和 SWE-Agent framework。虽然论文补充了 SWE-bench-Live 与 Multilingual 分析，但更广泛的 IDE/CLI scaffold、不同工具模板、真实用户仓库和企业代码库上的泛化仍未充分验证。

第四，论文的 end-to-end SWE-Agent RL 使用 outcome reward only 且无 KL/entropy regularization，结果显示 RL prior 略优，但也说明 agentic RL 仍相当脆弱。MT prior 很快退化，提示初始化、prompt filtering 和可训练问题数量对长程 RL 至关重要。

第五，部分比较跨不同系统、不同 agent framework 和不同 test-time scaling 预算，不能简单视为同一条件下的模型能力排名。Kimi-Dev 最可靠的证据是阶段消融和 prior adaptation 对比，而不是排行榜式横向比较。

## 相关知识链接

- [[application/agents/agent|Agent]]
- [[application/agents/workflow-agent|Workflow Agent]]
- [[training/mid-training/capability-injection|Capability Injection]]
- [[training/mid-training/continued-pretraining|Continued Pretraining]]
- [[training/data-engineering/synthetic-data|Synthetic Data]]
- [[training/post-training/grpo|GRPO]]
- [[application/evaluation/benchmark|Benchmark]]
