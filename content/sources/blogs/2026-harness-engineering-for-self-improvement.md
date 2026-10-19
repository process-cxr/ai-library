---
title: "Harness Engineering for Self-Improvement"
created: 2026-10-19
published: 2026-10-19
modified: 2026-10-19
type: source
status: growing
area: sources
tags:
  - source
  - blog
  - language-model
  - agent
  - auto-research
  - self-improvement
  - prompting
aliases:
  - Harness Engineering
  - Harness Engineering for Self-Improvement
source_url: https://lilianweng.github.io/posts/2026-07-04-harness/
source_type: personal-blog
publisher: Lil’Log
author: Lilian Weng
source_date: "2026-07-04"
---

# Harness Engineering for Self-Improvement

## 基本信息

- 标题：[Harness Engineering for Self-Improvement](https://lilianweng.github.io/posts/2026-07-04-harness/)
- 作者：Lilian Weng
- 来源：Lil’Log
- 日期：2026-07-04
- 类型：个人技术博客 / 研究综述
- 标签：Language Model、Agent、Auto-Research、Self-Improvement、Prompting
- 阅读时间：约 31 分钟

递归自我改进（recursive self-improvement，RSI）的概念可以追溯到 [I. J. Good（1965）](https://philpapers.org/rec/GOOSCT)。他将“超智能机器”（ultraintelligent machine）定义为一种能够在所有智力活动上超越人类，并设计出更好的机器来改进自身的系统。[Yudkowsky（2008）](https://www.lesswrong.com/posts/JBadX7rwdcRFzGuju/recursive-self-improvement) 用“递归自我改进”来描述一种特定的反馈循环：AI 使用当前的智能来改进产生这种智能的认知机制。

在现代 AI 中，这种反馈循环可能意味着模型直接重写自身的权重；更广义地说，也可能意味着模型改进训练管线和部署系统，而这些改进又使得后继模型能够在具有经济价值的任务上取得更好的性能。已有研究显示，前沿实验室中的 AI 研究进展速度已经大幅加快（[Anthropic](https://www.anthropic.com/institute/recursive-self-improvement)；[OpenAI](https://openai.com/index/how-agents-are-transforming-work/)）。

我特意提到“部署系统”，是因为原始模型与现实世界上下文之间的那一层，似乎与模型本身的智能同样重要（也就是预训练之后立即进行的那些评测）。Claude Code 和 Codex 等成功的 coding agent 产品已经展示了 harness 是 AI 部署中的重要组成部分。**Harness** 是包围基础模型的系统：它编排执行，决定模型如何思考和规划、调用工具并采取行动、感知和管理上下文、存储产物，以及评估结果。

这篇文章将聚焦 harness engineering 相关的研究，以及它如何促进 RSI。近期关于自动化研究、自我改进 Agent 和进化式程序搜索的大量工作，都可以围绕这个问题组织起来。关于模型自博弈、合成数据、测试时训练以及更广泛的持续学习主题的其他工作，也符合 RSI 的愿景（例如 [Yuan et al. 2024](https://arxiv.org/abs/2401.10020)、[Chen et al. 2024](https://arxiv.org/abs/2401.01335)、[Zhao et al. 2025](https://arxiv.org/abs/2505.03335)、[Choi et al. 2026](https://openreview.net/forum?id=lTbBFAoPSA)），但它们不是本文的重点。

# Harness 设计模式（Harness Design Patterns）

与[早期的 Agent 框架](https://lilianweng.github.io/posts/2023-06-23-agent/)相比，“Agent = LLM + memory + tools + planning + action”，harness engineering 还额外包含**工作流设计**（例如循环设计）、评估、权限控制和持久状态管理。它已经不再只是 prompt 模板，而更接近运行时和软件系统设计：模型如何观察、行动、记忆、自检和改进。

设计应当有意保持简单和通用，以便实现泛化；这可能需要参考已有的软件工程实践，从而受益于模型在预训练中获得的知识。操作系统和 harness 之间也存在很强的类比。与操作系统类似，harness 应该封装复杂的逻辑，同时保持接口简单。与此同时，配置、工具接口和其他协议可能会逐渐在整个行业中标准化。

## 模式 1：工作流自动化（Pattern 1: Workflow Automation）

定义一个模型可以在其中操作、测试和迭代的工作流，是实现自动化的关键设计。[Karpathy 的 autoresearch 仓库](https://github.com/karpathy/autoresearch)很好地展示了如何构建这样的工作流。一种常见工作流遵循面向目标的循环：规划、执行、观察/测试、改进，然后再次执行，**直到**目标达成。这个过程可能会主动向用户请求澄清，以明确任务规格或执行偏好。

![简化的 Codex Agent 循环：Agent 调用工具，工具响应影响模型下一次生成](https://lilianweng.github.io/posts/2026-07-04-harness/openai-agent-loop.png)

_简化的 Codex Agent 循环：Agent 调用工具，工具响应影响模型下一次生成。（图片来源：[OpenAI Codex agent post](https://openai.com/index/unrolling-the-codex-agent-loop/)）_

工作流图还强调，模型需要分析自己的轨迹和失败案例，然后通过“Agent runtime”迭代自身进展，而不是依赖静态的 prompt 模板。

## 模式 2：文件系统作为持久记忆（Pattern 2: File System as Persistent Memory）

长程 Agent 系统中反复出现的一种模式，是用简单的方式控制丰富的状态和产物。Harness 不应当把整个工作流和所有日志都携带在上下文中，而应该把持久状态保存在文件里。在长程 Agent rollout 中，实验日志、代码 diff、论文摘要、错误轨迹和过去的 rollout 轨迹等产物，往往会变得远远长于模型训练时所适应的上下文窗口。

学习如何读取、写入和编辑文件系统（通常通过 `bash` 命令完成）是 LLM 的基础技能。因此，以文件这种简单形式管理持久记忆，自然会从核心模型能力的提升中受益。

## 模式 3：子代理和后台任务（Pattern 3: Sub-agent and Backend Jobs）

Harness 可以启动多个子代理并行执行，也可以监控后台任务。当主 Agent 需要搜索多个假设、并发运行实验，或委派彼此隔离的子任务而不污染主上下文时，这种能力非常有用。此时，父 Agent 需要一个小型进程管理器，用来启动任务、检查日志、取消失败的运行，并把结果合并回主 Agent 线程。

关键的设计选择是让并行性变得显式且可检查。如果子代理的输出只存在于临时的聊天上下文中，它们很快就会变得过时并被隐藏；如果输出被存储为文件、日志和状态记录，模型就能在中断后恢复，并对自己的执行历史进行推理。

## 案例研究：Coding Agent Harness（Case study: Coding Agent Harness）

Claude Code、Codex、OpenCode 和 Cursor 风格的 Agent 已经逐渐形成了相对稳定的核心接口。它们通常使用如下循环：

![Coding agent harness 循环](https://lilianweng.github.io/posts/2026-07-04-harness/coding-harness-loop.png)

借助一组工具，coding agent 能够在给定仓库中开发和调试问题，就像人类开发者借助 IDE 一样。

（这不是完整列表，只用于演示。如果感兴趣，可以阅读[这份实现](https://github.com/yasasbanukaofficial/claude-code)。）

| 分组       | 工具定义                                                                                                                                                                                   |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 文件系统   | 文件发现：`glob`、`grep`、`ls`；文件读取：`read`、`read_many`；文件修改：`write`（创建完整的新文件）、`edit`（精确匹配字符串并替换）、`multi_edit`、`apply_patch`（应用结构化 patch/diff） |
| Shell 执行 | 运行命令：`bash`、`PowerShell`                                                                                                                                                             |
| IO         | `lsp`，以及 `git_status`、`git_diff`、`git_commit` 等 git 工具                                                                                                                             |
| 外部上下文 | MCP tools、Skills                                                                                                                                                                          |
| Web 搜索   | `web_search`、`web_fetch`、浏览器工具                                                                                                                                                      |
| 产物       | 读取文档和图片；生成 HTML、图片                                                                                                                                                            |
| 后台进程   | 例如：`CronCreate`、`CronDelete`、`CronList`                                                                                                                                               |
| Agent 委派 | 例如：`spawn_agent`、`resume_agent`、`wait_agent`、`list_agents`、`close_agent`、`interrupt_agent` 等                                                                                      |

## Harness 层与核心智能？（Harness Layer vs Core Intelligence?）

很难预测未来的 RSI 会在多大程度上依赖 harness engineering，但 RSI 的近期路径不太可能从模型直接重写自身权重开始。**我对一条可行的近期路径的预测是：**

1. Harness engineering 将朝向 meta-methodology 发展（也就是改进获得更好答案的机制，而不只是改进答案本身）。Harness 系统本身会成为优化目标，启发式规则会减少，而通用机制会增加。
2. 反过来，成熟的 harness 会为模型自我改进循环提供自动化研究能力；更聪明的模型则会防止 harness 过度工程化，并保持系统可持续。

最终，许多 harness 改进可能会被**内化**到核心模型行为中，但与外部上下文和工具的接口仍应保留。我们已经在 prompt engineering 中看到了这一模式的较弱版本：随着 instruction tuning 和模型推理能力的提升，手工 prompt 技巧变得不那么核心，但**明确指定目标、约束、上下文和评估的需求并没有消失**。

# Harness 优化（Harness Optimization）

Harness 系统中被优化的对象，大致经历了这样的演进：instruction [prompts](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/) → 结构化上下文 → 工作流 → harness 代码 → 优化器代码。随着模型变得更智能、更强大，我们会走向更复杂的目标和更通用的方法。

## 上下文工程（Context Engineering）

随着 Agent 任务的时间跨度显著增加，只是把所有工具响应和模型生成追加到上下文中，很快就会失控。上下文管理是一个用于为 LLM 构造更结构化、更简洁的上下文，并管理持久状态的层。毫无疑问，长上下文研究会继续取得进展，但目前长上下文智能和上下文工程有时是交织在一起的。

**Agentic Context Engineering**（ACE；[Zhang et al. 2025](https://arxiv.org/abs/2510.04618)）将上下文视为不断演化的 playbook，而不是不断变长的 prompt。它包含三个组件，用来维护一个由条目构成的上下文 playbook，每个条目都有一个标识符和一段描述：

1. _Generator_：生成任务轨迹，并参考这些条目。
2. _Reflector_：从成功和失败的轨迹中提炼洞见。
3. _Curator_：用增量式、条目化的内容更新结构化上下文。

![Agentic Context Engineering（ACE）框架](https://lilianweng.github.io/posts/2026-07-04-harness/ace.png)

_Agentic Context Engineering（ACE）框架。（图片来源：[Zhang et al. 2025](https://arxiv.org/abs/2510.04618)）_

为了避免迭代重写过程中的上下文坍缩和简洁性偏差，ACE 的一个关键设计选择是：Curator 不重写完整的 prompt blob，而是输出一组 `(identifier, description)` 形式的结构化条目，再通过确定性逻辑将这些条目合并到结构化上下文 logbook 中。上下文条目会被定期细化和去重。

ACE 能够从 rollout 中学习洞见，这推动我们走向自管理记忆；但它的更新规则和整体工作流仍是手工设计的。为了走向更加自我改进的循环，**Meta Context Engineering**（MCE；[Ye et al. 2026](https://arxiv.org/abs/2601.21557)）将机制（如何管理上下文）与产物内容（上下文中有什么）分开，在元优化层面进行 skill 演化，在基础层面进行上下文优化。

一个 MCE skill $s \in \mathcal{S}$ 定义了一个上下文函数 $c_s=(\rho_s,F_s)$，并将输入 $x$ 映射到上下文 $c = F_s(x;\rho_s)$，其中：

- $\rho_s = \{\rho_1,\dots,\rho_m\}$ 是静态组件（prompt、知识库、代码库）。
- $F_s = \{F_1,\dots,F_k\}$ 是动态算子（搜索、选择、过滤、格式化）。

双层优化的目标，是在训练数据上给定 skill $s$ 时寻找最佳上下文 $c_s^*$，而外层循环则寻找能够在验证集上带来最佳性能的 skill：

$$
\text{Inner: }c_s^*=\arg\max_{c_s}J_\text{train}(c_s;s)\quad
\text{Outer: }s^*=\arg\max_{s\in\mathcal{S}}J_\text{val}(c_s^*)
$$

Skill 数据库记录过去的 skill、上下文函数和评估指标 $\mathcal{H}_{k-1} = \{(s_i,c_i,J_i^\text{train}, J_i^\text{val})\}_{i=1}^{k-1}$。元层 Agent 会对已有 skill 执行 [crossover](<https://en.wikipedia.org/wiki/Crossover_(evolutionary_algorithm)>)，针对任务 $\tau$ 创建新的 skill：$s_k=\text{crossover}(\tau,\mathcal{H}_{k-1})$。

随后，基础层的 context engineer 执行 skill $s_k$，并在 rollout 反馈 $\mathcal{R}_k$ 的基础上学习上下文函数，同时受当前 skill 的引导：$c_k=\text{engineer}(\tau,s_k;c_{k-1}^*,\mathcal{R}_k)$。

![Meta Context Engineering（MCE）框架](https://lilianweng.github.io/posts/2026-07-04-harness/mce.png)

_Meta Context Engineering（MCE）框架：元层 skill 演化搜索上下文管理机制，基础层优化任务上下文。（图片来源：[Ye et al. 2026](https://arxiv.org/abs/2601.21557)）_

MCE 不像 ACE 那样规定组织上下文的启发式规则。它使用*自由形式的 skill*来保存任务中最重要的知识，并让 skill 与受 skill 条件化的上下文一起迭代演化。在实现上，上下文函数 $c$ 被实例化为一个专用目录中的文件集合，其中同时包含静态组件（`skill.md`）和动态组件（上下文及数据 rollout）。元层和基础层优化都在 Agent coding 环境中执行，并使用一组标准工具：

$$
\mathcal{T}=\{\texttt{Read},\texttt{Write},\texttt{Edit},\texttt{Bash},\texttt{Glob},\texttt{Grep},\texttt{TodoWrite}\}
$$

**Meta-Harness**（[Lee et al. 2026](https://arxiv.org/abs/2603.28052)）又向下深入了一层：被优化的对象，是决定和优化哪些信息应该被存储、检索并呈现给模型的**代码**。“Meta-”这个名字表示它是一个用来优化 harness 的 harness。

![Meta-Harness 外层优化算法](https://lilianweng.github.io/posts/2026-07-04-harness/meta-harness-outer-loop.png)

_Meta-Harness 外层优化算法。（图片来源：[Lee et al. 2026](https://arxiv.org/abs/2603.28052)）_

提出新 harness 的 proposer 本身就是一个 coding agent，最终输出是一组位于 Pareto frontier 上的 harness 候选。

- 完整的执行历史都可以通过文件系统访问，因此 coding agent 会使用 `grep` 或 `cat` 等命令读取历史，而不是把所有内容一股脑塞进单个 prompt 上下文。
- 被提出的 harness 是文件系统中的一个字典，其中包含自身的源代码、分数、rollout 轨迹和状态更新。
- Meta-Harness 循环会反复创建新的 harness，只保留合格的候选。

![Meta-Harness 的性能：左图为少量迭代的文本分类，右图为 TerminalBench-2](https://lilianweng.github.io/posts/2026-07-04-harness/meta-harness.png)

_Meta-Harness 的性能：左图为少量迭代的文本分类，右图为 TerminalBench-2。注意，TerminalBench-2 实验中的搜索从 Terminus-KIRA 和 Terminus-2 这两个很强的 harness 初始化。（图片来源：[Lee et al. 2026](https://arxiv.org/abs/2603.28052)）_

不过，重要的启示很清楚：一旦 harness 设计成为一个可执行的搜索空间，强大的 coding agent 就可以利用人类工程师使用的同一个设计空间。

## 工作流设计（Workflow Design）

Harness engineering 中的工作流设计可以由领域专家手工完成。以自动化研究为例，目前已经提出并测试了各种框架。**AI Scientist** 系统（[Lu et al. 2026](https://www.nature.com/articles/s41586-026-10265-5)）构建了一条管线，用于提出研究想法、编写代码、运行实验、分析结果、撰写论文和进行同行评审。[Meng et al.（2026）](https://arxiv.org/abs/2605.26340) 将可验证性作为 **ScientistOne** 的核心设计约束：每一个主张（引用、数值、方法、结论）都必须能够追溯到证据来源，并接受 Chain-of-Evidence 检查。

![AI Scientist 的想法生成、实验、论文写作和评审管线](https://lilianweng.github.io/posts/2026-07-04-harness/ai-scientist.png)

_AI Scientist 的想法生成、实验、论文写作和评审管线。（图片来源：[Lu et al. 2026](https://www.nature.com/articles/s41586-026-10265-5)）_

**Autodata** Agent（[Kulikov et al. 2026](https://arxiv.org/abs/2606.25996)）被设计成一个用于生成训练数据和评估数据的数据科学家。主 Agent 管理一个提出问题的 _challenger_、一个 _weak solver_、一个 _strong solver_ 和一个 _verifier/judge_，目标是合成难度“恰到好处”的数据，也就是 strong solver 能够成功而 weak solver 会失败。

在 Autodata 中，challenger prompt 会根据 solver 和 verifier 的反馈迭代更新。这里的限制是：合成任务被用来微调 weak solver，而不是 strong solver；如果循环无法迭代改进 strong model，那么它更像是围绕生成的 prompt 分布进行间接蒸馏，RSI 的意味就比较弱。

![Autodata 围绕 challenger、solver 和 verifier 角色生成合成训练及评估数据的 Agent 工作流](https://lilianweng.github.io/posts/2026-07-04-harness/autodata.png)

_Autodata 围绕 challenger、solver 和 verifier 角色生成合成训练及评估数据的 Agent 工作流。（图片来源：[Kulikov et al. 2026](https://arxiv.org/abs/2606.25996)）_

工作流的设计空间非常庞大，因此我们自然可以把工作流设计视为一个搜索问题；这样一来，我们就应该能够通过算法找到好的方案，而不只是手工构造它们。沿着这个方向，**Automated Design of Agentic Systems**（ADAS；[Hu et al. 2025](https://arxiv.org/abs/2408.08435)）将 Agent 设计本身表述为一个优化问题，即“meta-agent search”：由一个 meta-agent 提出新的 Agent 工作流设计。

1. 使用 CoT 和 self-refine 等简单 Agent 初始化一个 Agent 工作流档案。
2. 让 meta-agent 参考档案中的已有方案，用**代码**编写新的 Agent。
   - meta-agent 首先生成新工作流的高层描述，然后用代码实现它。
   - 随后，meta-agent 对草稿程序执行两轮 self-refine（也就是让模型提供反馈，再让同一个模型根据反馈修改先前的输出；[Madaan et al. 2023](https://arxiv.org/abs/2303.17651)），以检查其新颖性。
3. 评估每个新候选，将成功的候选重新加入档案。
4. 重复第 2—3 步，直到达到最大迭代次数。

![Automated Design of Agentic Systems（ADAS）示意图](https://lilianweng.github.io/posts/2026-07-04-harness/adas.png)

_Automated Design of Agentic Systems（ADAS）示意图。（图片来源：[Hu et al. 2025](https://arxiv.org/abs/2408.08435)）_

**AFlow**（[Zhang et al. 2025](https://arxiv.org/abs/2410.10762)）将 Agent 工作流表示成一张图，其中节点代表调用 LLM 的动作，边用代码实现逻辑操作。工作流优化依赖 [MCTS](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)（Monte Carlo Tree Search，蒙特卡洛树搜索）：

1. 使用一个模板，在树中初始化起始工作流 $W_0$。
2. 使用分数和均匀探索的软混合来选择工作流节点。
3. 让 LLM 根据工作流的评估性能生成一个修改后的工作流，以扩展该节点。
4. 执行并评估新的工作流。
5. 如果新工作流在 $N$ 轮预算内表现出改进，就把它加入树中。
6. 重复第 2—5 步，直到 top-$k$ 平均分趋于平稳或达到预算上限。

![AFlow 在工作流候选树上的优化过程](https://lilianweng.github.io/posts/2026-07-04-harness/aflow.png)

_AFlow 在工作流候选树上的优化过程。（图片来源：[Zhang et al. 2025](https://arxiv.org/abs/2410.10762)）_

AFlow 在问答、代码和数学任务上的实验显示，与手工设计的工作流和 ADAS 相比，AFlow 取得了不错的改进。

![AFlow 与手工方法、ADAS 的实验对比](https://lilianweng.github.io/posts/2026-07-04-harness/aflow-exp.png)

_AFlow 与手工方法、ADAS 的实验对比。（图片来源：[Zhang et al. 2025](https://arxiv.org/abs/2410.10762)）_

## 自我改进的 Harness（Self-Improving Harness）

上下文工程或工作流设计，都只是 harness 的一部分。我们需要搜索整个设计空间，并同时优化上下文管理逻辑、工作流、权限以及其他许多 harness 组件。正如 Meta-Harness、ADAS 和 AFlow 等工作所展示的，**✨代码✨**是定义程序和系统的**通用语言**。简单地说，harness 是一段代码，它规定 prompt、工具调用、子代理、控制流、记忆和工作流逻辑如何协同工作。如果 LLM 能够优化执行 Agent 的代码，它就可以访问比手写 prompt 大得多的设计空间。

**Self-Taught Optimizer**（STOP；[Zelikman et al. 2023](https://arxiv.org/abs/2310.02304)）是递归改进 scaffold 的早期例子之一。位于 $t=0$ 步的种子 improver $I_0$ 接收一个初始解 $s$、一个效用函数 $u$ 和一个黑盒语言模型 $M$，并返回改进后的解 $s'$，也就是 $s' = I(u, s; M)$。STOP 的目标不是直接改进 $s$，而是**改进 improver $I$ 本身**。

首先，我们将给定 improver 函数 $I$ 在下游任务集合 $\mathcal{D}$ 上的平均效用定义为元效用：

$$
\hat{u}(I) \triangleq \frac{1}{\vert\mathcal{D}\vert}\mathbb{E}_{(u,s)\sim \mathcal{D}}[u(I(u,s; M))]
$$

由于改进 improver 函数本身也是一个优化问题，我们可以根据 $I_{t-1}$ 的性能，通过一次自我改进更新递归地得到 $I_t$：

$$
I_t=I_{t-1}(\hat{u},I_{t-1};M)
$$

![Self-Taught Optimizer（STOP）算法](https://lilianweng.github.io/posts/2026-07-04-harness/STOP-algo.png)

_Self-Taught Optimizer（STOP）算法。（图片来源：[Zelikman et al. 2023](https://arxiv.org/abs/2310.02304)）_

在他们的实验中，改进后的 improver 发现了多种策略，例如遗传算法、分解并改进各个部分、多臂 prompt bandit、模拟退火、改变温度，以及 beam/tree search。这与将 harness 工作流表示为一个可优化对象的方式类似。

![STOP 发现的自我改进策略示例](https://lilianweng.github.io/posts/2026-07-04-harness/STOP-patterns.png)

_STOP 发现的自我改进策略示例。（图片来源：[Zelikman et al. 2023](https://arxiv.org/abs/2310.02304)）_

Zelikman et al.（2023）的发现中有一个需要警惕的结果：STOP 在 GPT-4 上使平均下游性能随着迭代提升，但在 GPT-3.5 和 Mixtral 等较弱模型上却出现退化。仅有递归结构是不够的。基础模型必须**足够有能力**，才能改进这一机制。这意味着 harness 改进能够促进模型更好地部署，但智能仍然是核心。

[Lin et al.（2026）](https://arxiv.org/abs/2605.30621) 更详细地研究了 harness 演化对模型能力的依赖。他们区分了两个维度：（1）_harness-updating_，指产生有用 harness 修改的能力；（2）_harness-benefit_，指利用更新后的 harness 来更好地解决任务的能力。有趣的是，在从 Qwen3.5-9B 到 Claude Opus 4.6 的一系列不同规模、不同核心智能水平的模型中，他们观察到相似的 harness updating 能力；9B 的 harness proposer/evolver 能够编写出在过程上与 Opus 同构的 skill。为了最好地利用 harness，模型需要正确且及时地调用 skill/工具，并且擅长长程指令跟随。

![Harness updating 与 harness benefit 的主要结果](https://lilianweng.github.io/posts/2026-07-04-harness/harness-update.png)

_主要结果：（A）在从 Qwen2-32B 到 Opus 4.6 的模型范围内，harness updating 能力基本持平；（B）harness benefit 能力呈非单调关系，中等规模模型受益最多。（图片来源：[Lin et al. 2026](https://arxiv.org/abs/2605.30621)）_

更近的一项工作 **Self-Harness**（[Zhang et al. 2026](https://arxiv.org/abs/2606.09498)）依靠 LLM Agent，通过 propose-evaluate-accept 循环改进自身的 harness。

![Self-Harness 通过弱点挖掘、受限 harness 提议和验证来更新 harness](https://lilianweng.github.io/posts/2026-07-04-harness/self-harness.png)

_Self-Harness 通过弱点挖掘、受限 harness 提议和验证来更新 harness。（图片来源：[Zhang et al. 2026](https://arxiv.org/abs/2606.09498)）_

Self-Harness 的循环包含三个阶段：

1. _Weakness mining_：将失败聚类为由 verifier 支撑的失败模式。
   - 使用当前 harness $h_t$ 在任务上进行评估，并收集执行轨迹供分析。
   - 注意，两个运行在错误日志表面上可能拥有相同的 verifier 结果，例如超时或缺少产物，但它们可能具有不同的因果机制。因此，我们需要一份包含丰富信息的失败记录，其中包括 verifier 层面的终止原因、相关 Agent 行为的因果状态，以及轨迹所暴露的抽象 Agent 机制，从而找出根因。
2. _Harness proposal_：基于挖掘出的失败模式提出受限的 harness 修改。
   - 在 $h_t$ 下调用同一个模型作为 proposer。
   - 向模型提供受限的提议上下文：（1）当前 harness 的可编辑表面；（2）评估系统给出的、由 verifier 支撑的失败模式；（3）应该保留的成功行为记录；（4）此前尝试过的修改摘要。
   - Harness 修改应当优先针对反复出现且可以处理的错误模式（例如不是任务本身的难度），并且能够通过窄范围修改解决。
   - Harness 修改候选应该彼此不同且具有多样性。
3. _Proposal validation_：验证并合并合格的修改，创建新的 harness $h_{t+1}$。
   - 在 held-in $D_\text{in}$ 和 held-out $D_\text{out}$ 划分上运行回归测试：前者用于检查弱点是否解决，后者用于检查是否引入其他未知问题。
   - 只有在 held-in 和 held-out 数据上都没有回归时，候选修改才会被接受。
   - 被接受的候选会被合并，用来更新 harness 为 $h_{t+1}$；被拒绝的候选则会被记录，但不会改变当前的 harness。

在 Terminal-Bench-2 上运行 `MiniMax M2.5`、`Qwen3.5-35B-A3B` 和 `GLM-5` 时，Self-Harness 被证明能够学习针对不同基础模型不同弱点的、模型特定的 harness 指令，并提升 held-out pass rate。

这类 self-harness 工作确实让我担心：如果一个程序被允许编辑操作系统，抽象边界就会被破坏。可编辑表面需要被恰当地设计，权限控制和安全层必须位于这个循环之外。围绕 [reward hacking](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/) 的所有挑战仍然存在。

**Agentic Harness Engineering**（AHE；[Lin et al. 2026](https://arxiv.org/abs/2604.25850)）认为 harness 演化的瓶颈在于**可观测性**——也就是说，当 rollout 失败时，我们需要知道是哪个组件负责导致失败，并且每次修改都应当有证据支撑。

该框架通过三个可观测性支柱构成闭环：

1. _Component observability_：每个可编辑的 harness 组件在文件系统中都有对应表示，因此动作空间是显式且可追踪的。
   - 一个 harness 包含 7 个组件：system prompt、tool description、tool implementation、middleware、skill、sub-agent configuration 和 long-term memory。
   - 每种失败模式都映射到一个组件，使修改更有针对性。
2. _Experience observability_：将大量原始轨迹分析和总结为一个证据及失败模式的层级结构。
   - 每个 harness 会生成 $k$ 条轨迹。
   - 使用一个 Agent（“Agent debugger”）分析每条保存在单独文件中的轨迹，并生成逐任务的失败或成功根因分析报告。
   - 将所有逐任务报告汇总为下一步使用的 benchmark 总览，并在需要时访问原始轨迹。这种分层访问结构更加节省 token。
3. _Decision observability_：每次修改都配有一个关于下一轮的预测，以便验证。
   - 一个 Agent（“Evolve agent”）读取仓库，决定要修改哪个组件，然后给出修改以及修改背后的理由。
   - 每次修改都是一个文件级、可证伪的主张，可以在下一轮验证，但需要满足两个约束：
     - （1）修改只能应用于 harness workspace。runs 目录、tracer、verifier 和 LLM 配置都是只读的，这会禁用一部分 reward hacking（例如禁用 verifier、替换模型或提高 reasoning budget），因此每一项记录到的收益都可以归因于 harness 修改。
     - （2）修改必须以证据为依据，并包含一条 manifesto 记录：失败证据的名称、推断出的根因、目标修复，以及同时包含预期修复和可能回归风险的预测影响。

在 Terminal-Bench-2 上，AHE 的表现优于人工设计的 harness（OpenCode、Terminus-2、Codex），但 Hard 级别和另外几个 self-evolve 基线（ACE、TF-GRPO）除外。同一个冻结的 harness 在不再继续演化的情况下迁移到了 SWE-bench-verified，这表明演化后的 harness 能够把工程经验编码进 harness 组件，而不是进行针对 benchmark 的优化。

## 进化式搜索（Evolutionary Search）

进化式搜索是一种受自然选择启发的优化方法（参见我之前关于[进化算法](https://lilianweng.github.io/posts/2019-09-05-evolution-strategies/)的文章）。它通过变异来演化一组解，只保留群体中具有较高“适应度”的解。在以下情况下，进化式搜索很有用：（1）搜索空间很大或形状奇怪；（2）很难直接使用梯度进行优化，但很容易评估候选解。Harness 搜索似乎很适合采用这种方法。

过去的研究已经将进化式搜索用于 prompt engineering。**Promptbreeder**（[Fernando et al. 2023](https://arxiv.org/abs/2309.16797)）通过丰富的变异操作优化任务专用 prompt；有趣的是，用于指导 LLM 变异任务 prompt 的变异 prompt 本身也会通过进化得到改进。**GEPA**（[Agrawal et al. 2025](https://arxiv.org/abs/2507.19457)）将基于[反思](https://lilianweng.github.io/posts/2023-06-23-agent/#self-reflection)的 prompting 与进化式搜索结合起来，并使用对试错轨迹的自然语言反思来提出 prompt 更新。

[Novikov et al.（2025）](https://arxiv.org/abs/2506.13131) 将 **AlphaEvolve** 引入为一个 coding-agent 进化式搜索系统。它保存一个候选程序池，并提示冻结的 LLM 生成 diff 来进行改进。系统反复评估子程序并保留成功者，随着时间推移发现更好的解。

![AlphaEvolve 的工作方式](https://lilianweng.github.io/posts/2026-07-04-harness/alphaevolve.png)

_AlphaEvolve 的工作方式。（图片来源：[Novikov et al. 2025](https://arxiv.org/abs/2506.13131)）_

AlphaEvolve 的设计中有一些细节很重要：

- prompt 包含父程序、结果、指令，有时还包含元信息。
- coding agent 可以访问完整仓库，但需要改进的代码区域会用 `# EVOLVE-BLOCK-START` 和 `# EVOLVE-BLOCK-END` 显式标记。
- meta-prompt 会与指令和上下文共同演化，这与我们演化解程序的方式类似。

消融实验展示了进化过程、prompt 中的上下文、meta-prompt、整文件进化以及使用更强 LLM 的价值。

![AlphaEvolve 消融实验展示若干设计的价值](https://lilianweng.github.io/posts/2026-07-04-harness/alphaevolve-plot.png)

_消融实验展示 AlphaEvolve 中若干设计的价值。（图片来源：[Novikov et al. 2025](https://arxiv.org/abs/2506.13131)）_

近期的变体包括：将进化式搜索与 RL 和 in-context learning 结合的 **ThetaEvolve**（[Wang et al. 2025](https://arxiv.org/abs/2511.23473)）；将人类专家示范作为 harness 层诊断和编辑的参考经验、补充到 self-rollout 档案中的 **DemoEvolve**（[Che et al. 2026](https://arxiv.org/abs/2605.24539)）。另一方面，**ShinkaEvolve**（[Lange et al. 2025](https://arxiv.org/abs/2509.19349)）引入了三个新组件来提升 LLM 的采样效率：

- 设计父代采样，在性能排名和子代数量之间取得平衡，从而实现更高的样本效率探索。
- 基于 embedding 余弦相似度，丢弃与现有群体过于相似的候选，进行代码新颖性拒绝采样。
- 在 meta-scratchpad 中识别成功解中的优秀模式，以指导未来的变异。

与上述聚焦于改进解的方法不同，**Darwin Gödel Machine**（DGM；[Zhang et al. 2025](https://arxiv.org/abs/2505.22954)）明确将一个可编辑的 harness 代码仓库的演化作为目标，并由基于 LLM 的 coding agent 执行。更准确地说，这个 Agent 被允许修改自身的 harness。后续的 Hyperagents 工作（[Zhang et al. 2026](https://arxiv.org/abs/2603.19461)）引入了一个 meta-agent，用于控制如何修改已有的任务 Agent，以创建新的 Agent。

1. 从池中只有一个 coding agent 开始。
2. 在每一次迭代中，以与性能成正比、与子代数量成反比的概率选择一个父 Agent，对其进行修改并分支出新的 Agent。
3. 被选中的父 Agent 检查自己的 benchmark 评估日志，然后提出对自身 harness 代码库的改进，从而生成一个新版本的 coding agent。代码编辑通过两个基本工具实现：（1）bash（参数为 `<bash_command>`）；（2）editor（参数为 `view/create/edit <file_path>`）。
4. 评估新的 coding agent，只将性能足够高的 Agent 加回池中。
5. 重复第 2—4 步，直到达到某个停止条件。

DGM 是在固定模型下进行的 harness 演化。在以 `Claude 3.5 Sonnet` 为基础 LLM、使用简单初始 harness 配置的实验中，DGM 发现的 Agent 在 SWE-bench Verified 上（20% 到 50%）与手工设计的 Agent 相当或更好，在 Polyglot 上也取得了 14.2% 到 30.7% 的结果。

当候选解可以自动评估且候选适应度容易量化时，这类方法表现良好，例如矩阵乘法、GPU kernel 优化、算法竞赛和数据中心调度。在评估缓慢、含义模糊或主要依赖启发式方法的领域，它会遇到困难。进化的计算效率和有效性也是需要考虑的问题。

## 与模型权重的联合优化（Joint Optimization with Model Weights）

Harness 演化改变的是模型周围的非参数系统。为了实现完整的自我改进，也可以允许模型同时更新自身的权重。权重更新可以通过改进模型训练管线来实现，也可以通过测试时持续学习来实现。持续学习是一个值得未来单独写一篇文章的主题。

**SIA**（[Hebbar et al. 2026](https://arxiv.org/abs/2605.27276)）是将 harness 改进和模型参数更新结合进同一个优化循环的早期尝试，其设计包含三个组件：

- _Meta-Agent_：提出初始 harness。
- _Task-Specific Agent_：执行任务。
- _Feedback-Agent_：根据最近的轨迹，决定更新 harness 还是模型权重。

![SIA 中的 Feedback-Agent 决定下一轮迭代类型](https://lilianweng.github.io/posts/2026-07-04-harness/SIA.png)

_SIA 中的 Feedback-Agent 决定下一轮迭代类型。（图片来源：[Hebbar et al. 2026](https://arxiv.org/abs/2605.27276)）_

SIA 的实验中存在一些混淆因素，使结果难以解释。例如，task-specific agent 远弱于 Meta-Agent 和 Feedback-Agent 所使用的模型（`gpt-oss-120b` 对比 `Claude Sonnet 4.6`），而且基线太弱，无法与相关方法进行清晰的交叉比较。我认为这个方向很有意思，但证据仍然是初步的。不过，训练稳定性和 Goodhart effect 等许多挑战仍然是开放问题。

**Continual Harness**（[Karten et al. 2026](https://arxiv.org/abs/2605.09998)）在长程游戏环境中进行了 harness 更新和策略模型共同学习的实验：它对低奖励轨迹蒸馏强教师模型的标签。

# 未来挑战（Future Challenges）

AI Scientist 这条路线有力地展示了专家设计的 harness 如何协调自动化研究循环中的很大一部分，并以撰写研究论文的形式进行了实验。但生产论文并不等同于科学发现。一个系统可以写出看起来合理的论文，同时仍然存在伪造的引用、实现漂移或薄弱的实验结果。

[Trehan 与 Chopra（2026）](https://arxiv.org/abs/2601.03315) 测试了 LLM 是否能够在最少 scaffold 和基本工具（即 `read_file`、`write_file`、`llm_search`、`list_files`）的帮助下，从研究想法走到论文。每个想法都有一个专用工作区，Agent 可以在其中生成和读取文档，并将其作为上下文的一部分。他们在三个领域进行了实验（world models、多 Agent RL、AI safety & alignment），每个领域包含 45—50 份高质量种子文档，用于激发新的想法。只有四个想法被人类专家选中并进入完整管线，其中只有一个最终完整执行为论文。他们在实验中观察到六种反复出现的失败模式：

- _Bias toward training-data defaults_：偏向训练数据中的默认模式，使用过时的库、陈旧的命令、标准格式，或不以实际仓库和数据集为依据的假设。
- _Implementation drift under execution pressure_：执行压力下的实现漂移，当实现变得技术上复杂时，模型可能会转向一个常见且更简单的方案，而不是最初提出的方法。
- _Memory and context degradation_：记忆和上下文退化，长程项目会丢失关键细节，除非日志被写成持久产物。
- _Over-optimism_：过度乐观，即使实验充满噪声或已经失败，模型仍然宣布成功。这与 [Bubeck et al.（2025）](https://arxiv.org/abs/2511.16072) 观察到的“p-hacking and eureka-ing”模式类似：模型可能引入“数值胶带”（numerical duct tape），在信号仍然只是噪声时就宣布胜利。
- _Insufficient domain intelligence_：领域智能不足，模型缺少隐性的工艺知识，例如预测实现复杂度、判断实验结果是否合理，或知道哪些 baseline 很重要。
- _Weak scientific taste_：科学品味不足，实验可能能够执行，却没有回答正确的问题。

在走向完整 RSI 的过程中，研究人员已经取得了实质进展，但仍有几个瓶颈。

**1. 弱且模糊的评估器。** 许多研究主张没有快速而精确的验证器，许多现实世界任务也是如此。目前的自我改进循环最适合评估指标可测量且客观的任务，类似于 [RL 的工作方式](https://lilianweng.github.io/posts/2018-02-19-rl-overview/)。

研究品味、新颖性和长期科学价值要难以测量得多。例如，研究品味通常混合了问题表述、实验设计，以及判断哪些出乎意料的结果值得继续追踪、哪些失败案例值得重试。

**2. 上下文和记忆生命周期。** 随着 AI Agent 变得更加自主和独立，记忆会不断增长。一个有用的 harness 需要管理上下文和记忆，以弥补当前长上下文生成的限制，同时最大化长程任务的成功率。既然人类能够在一生中维持记忆，我认为这里存在一个类比：上下文工程将会、也应该成为智能的核心部分，而不是停留在软件系统层。

**3. 负结果。** 研究人员受到激励去发表成功结果，因此文献偏向成功。LLM 在大量数据上训练（至少目前主要是人类创建的数据，哈哈），可能不擅长决定何时放弃一个假设、报告负结果，甚至承认失败，因为数据中成功与失败案例的比例不平衡。研究 harness 应当让失败尝试易于保存，因为从失败中学习是缩小任务搜索空间的最佳方式。

**4. 多样性坍缩。** 进化和 RL 循环倾向于利用已知的高奖励模式。我们需要[机制](https://lilianweng.github.io/posts/2020-06-07-exploration-drl/)来防止种群坍缩成同一种解的不同变体。这对于开放式研究尤其重要，因为在当前评估器下，最好的路径一开始可能看起来更差。

**5. [Reward hacking](https://lilianweng.github.io/posts/2024-11-28-reward-hacking)。** 自我改进循环会优化它得到的任何信号。如果奖励来自单元测试，Agent 可能会对测试过拟合；如果奖励来自 judge model，它可能会学到针对这个 judge 的 reward hacking 技巧；如果奖励来自 benchmark 分数，它可能会利用 benchmark 的缺陷。

评估器和权限控制应该位于演化 harness 的循环之外，并配合 held-out 测试、轨迹审计以及关键决策点的人类复核——监督究竟能在多大程度上扩展并自动化，仍然是一个开放的研究问题。

**6. 长期成功。** 外部优化循环处理的是单次 rollout 之外的奖励，而这些奖励可以在训练沙箱中模拟。

以 coding agent 为例，coding agent 已经提高了软件工程中的日常生产力，但许多优化目标仍然过于短期。它通常能够完成手头的任务，却不太清楚应当如何保护一个由数百或数千名工程师共同维护的仓库的长期健康。标准的、基于沙箱的 RLVR 风格训练，很少捕捉可维护性、所有权边界、迁移成本、向后兼容性或未来的调试负担。

**7. 人类的角色。** 人类应该向更高层移动，而不是被从循环中移除；也就是说，人类应当在正确的时间、正确的抽象层级提供监督，我们的系统设计也应考虑何时、如何设置这些接触点。

上面列出的许多挑战都需要人类的反馈和引导。毕竟，我们正在为人类更好的未来构建这项技术，而不是反过来。

# 引用（Citation）

请按如下方式引用本文：

> Weng, Lilian. “Harness Engineering for Self-Improvement”. Lil’Log (Jul 2026). https://lilianweng.github.io/posts/2026-07-04-harness/

或者使用 BibTeX：

```bibtex
@article{weng2026harness,
  title = {Harness Engineering for Self-Improvement},
  author = {Weng, Lilian},
  journal = {lilianweng.github.io},
  year = {2026},
  month = {July},
  url = "https://lilianweng.github.io/posts/2026-07-04-harness/"
}
```

# 附录：一些有用的 Benchmark（Appendix: Some useful benchmarks）

- **[PaperBench](https://arxiv.org/abs/2504.01848)**：从头开始复现 20 篇 ICML 2024 Spotlight 和 Oral 论文，包括理解论文贡献、开发代码库并成功执行实验。
  - 每个复现任务都会被拆分成更小、可以分别评分的任务。
  - 总共有 8,316 条 rubric，由论文作者共同参与制定。
  - 当时最好的模型（`Claude 3.5 Sonnet`，约 21%）没有超过 ML PhD。
  - 包含 PaperBench、PaperBench Code-Dev（较轻量版本）和 JudgeEval。
- **[CORE-Bench](https://arxiv.org/abs/2409.11363)**：评估已发表研究的计算可复现性。
  - 基于来自计算机科学、社会科学和医学领域 90 篇科学论文的 270 个任务。
  - 任务要求根据提供的代码和数据复现结果。
  - 包含多个难度级别，以及纯语言和视觉语言任务。
  - 当时报告的最佳 Agent（`GPT-4o` 和 `GPT-4o-mini`）在最难任务上只有 21% 的准确率。
- **[ScienceAgentBench](https://arxiv.org/abs/2410.05080)**：评估 LLM Agent 进行数据驱动科学发现的能力。
  - 从四个学科（数学、化学、生物学、地理学）的 44 篇同行评审论文中提取 102 个任务。
  - 覆盖这些领域中的基础数据科学任务：数据处理、模型开发、数据分析和信息可视化。
- **[RE-Bench](https://arxiv.org/abs/2411.15114)**：在现实的 ML research-engineering 环境中，将前沿 AI Agent 与人类专家进行比较。
  - 包含 7 个具有挑战性的开放式 ML research-engineering 环境。
  - 每个环境由 `(scoring function, starting solution, reference solution)` 构成；每个环境都可以使用不超过 8 张 H100 GPU 运行。
  - 示例包括优化 kernel、运行 scaling-law 实验、修复 embedding、为 QA 微调 GPT-2 等。
  - 包含 61 位不同人类专家进行的 71 次八小时尝试的数据。
  - 人类专家在 82% 的八小时尝试中取得了非零分数；24% 的尝试达到了或超过了强参考解。
  - 最佳 AI Agent 在两小时预算下的得分是人类的 4 倍，但人类在更长预算下的收益更高，并在 8 小时和 32 小时设置下超过了 Agent。
- **[MLE-bench](https://arxiv.org/abs/2410.07095)**：在离线 Kaggle 竞赛中评估 ML engineering Agent。
  - 包含从 Kaggle 精选的 75 个 ML engineering 竞赛。
  - 测试训练模型、准备数据集、运行实验和向评分脚本提交预测。
  - 使用 Kaggle 公共排行榜作为人类基线。
  - 论文中的最佳设置（带有 AIDE scaffold 的 `o1-preview`）在至少 16.9% 的竞赛中达到了 Kaggle 铜牌水平。
  - 包含资源扩展和污染分析。
- **[KernelBench](https://arxiv.org/abs/2502.10517)**：评估生成的 GPU kernel 的正确性和速度。
  - 包含 250 个 PyTorch 任务，用来评估 LLM 是否能编写快速且正确的 kernel。
  - 评估指标 `fast_p` 表示生成的 kernel 中既正确又快于 baseline 的比例。

# 参考文献（References）

[1] Good, I. J. [“Speculations Concerning the First Ultraintelligent Machine.”](https://philpapers.org/rec/GOOSCT) _Advances in Computers_, 6:31–88, 1965。

[2] Yudkowsky, Eliezer. [“Recursive Self-Improvement.”](https://www.lesswrong.com/posts/JBadX7rwdcRFzGuju/recursive-self-improvement) LessWrong, 2008。

[3] Choi, et al. [“Anchored Self-Play for Code Repair.”](https://openreview.net/forum?id=lTbBFAoPSA) ICML 2026。

[4] Zhao, et al. [“Absolute Zero: Reinforced Self-play Reasoning with Zero Data.”](https://arxiv.org/abs/2505.03335) arXiv preprint arXiv:2505.03335, 2025。

[5] Yuan, et al. [“Self-Rewarding Language Models.”](https://arxiv.org/abs/2401.10020) arXiv preprint arXiv:2401.10020, 2024。

[6] Chen, et al. [“Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models.”](https://arxiv.org/abs/2401.01335) ICML 2024。

[7] Zhang, et al. [“Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models.”](https://arxiv.org/abs/2510.04618) ICLR 2026。

[8] Ye, et al. [“Meta Context Engineering via Agentic Skill Evolution.”](https://arxiv.org/abs/2601.21557) arXiv preprint arXiv:2601.21557, 2026。

[9] Lee, et al. [“Meta-Harness: End-to-End Optimization of Model Harnesses.”](https://arxiv.org/abs/2603.28052) arXiv preprint arXiv:2603.28052, 2026。

[10] Lu, et al. [“Towards end-to-end automation of AI research.”](https://www.nature.com/articles/s41586-026-10265-5) _Nature_, 651:914–919, 2026。

[11] Meng, et al. [“ScientistOne: Towards Human-Level Autonomous Research via Chain-of-Evidence.”](https://arxiv.org/abs/2605.26340) arXiv preprint arXiv:2605.26340, 2026。

[12] Kulikov, et al. [“Autodata: An agentic data scientist to create high quality synthetic data.”](https://arxiv.org/abs/2606.25996) arXiv preprint arXiv:2606.25996, 2026。

[13] Hu, Lu, and Clune. [“Automated Design of Agentic Systems.”](https://arxiv.org/abs/2408.08435) ICLR 2025。

[14] Madaan, et al. [“Self-Refine: Iterative Refinement with Self-Feedback.”](https://arxiv.org/abs/2303.17651) NeurIPS 2023。

[15] Zhang, et al. [“AFlow: Automating Agentic Workflow Generation.”](https://arxiv.org/abs/2410.10762) ICLR 2025。

[16] Zelikman, et al. [“Self-Taught Optimizer (STOP): Recursively Self-Improving Code Generation.”](https://arxiv.org/abs/2310.02304) COLM 2024。

[17] Zhang, et al. [“Self-Harness: Harnesses That Improve Themselves.”](https://arxiv.org/abs/2606.09498) arXiv preprint arXiv:2606.09498, 2026。

[18] Fernando, et al. [“Promptbreeder: Self-Referential Self-Improvement Via Prompt Evolution.”](https://arxiv.org/abs/2309.16797) arXiv preprint arXiv:2309.16797, 2023。

[19] Agrawal, A. et al. [“GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning.”](https://arxiv.org/abs/2507.19457) arXiv preprint arXiv:2507.19457, 2025。

[20] Novikov, et al. [“AlphaEvolve: A coding agent for scientific and algorithmic discovery.”](https://arxiv.org/abs/2506.13131) arXiv preprint arXiv:2506.13131, 2025。

[21] Lange, Imajuku, and Cetin. [“ShinkaEvolve: Towards Open-Ended And Sample-Efficient Program Evolution.”](https://arxiv.org/abs/2509.19349) arXiv preprint arXiv:2509.19349, 2025。

[22] Wang, et al. [“ThetaEvolve: Test-time Learning on Open Problems.”](https://arxiv.org/abs/2511.23473) arXiv preprint arXiv:2511.23473, 2025。

[23] Zhang, et al. [“Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents.”](https://arxiv.org/abs/2505.22954) arXiv preprint arXiv:2505.22954, 2025。

[24] Zhang, et al. [“Hyperagents.”](https://arxiv.org/abs/2603.19461) arXiv preprint arXiv:2603.19461, 2026。

[25] Yuksekgonul, et al. [“Learning to Discover at Test Time.”](https://arxiv.org/abs/2601.16175) arXiv preprint arXiv:2601.16175, 2026。

[26] Riaz, et al. [“Epistemic Uncertainty for Test-Time Discovery.”](https://arxiv.org/abs/2605.11328) arXiv preprint arXiv:2605.11328, 2026。

[27] Hebbar, et al. [“SIA: Self Improving AI with Harness & Weight Updates.”](https://arxiv.org/abs/2605.27276) arXiv preprint arXiv:2605.27276, 2026。

[28] Trehan and Chopra. [“Why LLMs Aren’t Scientists Yet: Lessons from Four Autonomous Research Attempts.”](https://arxiv.org/abs/2601.03315) arXiv preprint arXiv:2601.03315, 2026。

[29] Bubeck, et al. [“Early science acceleration experiments with GPT-5.”](https://arxiv.org/abs/2511.16072) arXiv preprint arXiv:2511.16072, 2025。

[30] Starace, et al. [“PaperBench: Evaluating AI’s Ability to Replicate AI Research.”](https://arxiv.org/abs/2504.01848) ICML 2025。

[31] Wijk, et al. [“RE-Bench: Evaluating frontier AI R&D capabilities of language model agents against human experts.”](https://arxiv.org/abs/2411.15114) ICML 2025。

[32] Chan, et al. [“MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering.”](https://arxiv.org/abs/2410.07095) arXiv preprint arXiv:2410.07095, 2024。

[33] Chen, et al. [“ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery.”](https://arxiv.org/abs/2410.05080) ICLR 2025。

[34] Siegel, et al. [“CORE-Bench: Fostering the Credibility of Published Research Through a Computational Reproducibility Agent Benchmark.”](https://arxiv.org/abs/2409.11363) TMLR 2024。

[35] Ouyang, et al. [“KernelBench: Can LLMs Write Efficient GPU Kernels?”](https://arxiv.org/abs/2502.10517) arXiv preprint arXiv:2502.10517, 2025。

[36] Lin, et al. [“Harness Updating Is Not Harness Benefit: Disentangling Evolution Capabilities in Self-Evolving LLM Agents.”](https://arxiv.org/abs/2605.30621) arXiv preprint arXiv:2605.30621, 2026。

[37] Lin, et al. [“Agentic Harness Engineering: Observability-Driven Automatic Evolution of Coding-Agent Harnesses.”](https://arxiv.org/abs/2604.25850) arXiv preprint arXiv:2604.25850, 2026。

[38] Karten, et al. [“Continual Harness: Online Adaptation for Self-Improving Foundation Agents.”](https://arxiv.org/abs/2605.09998) arXiv preprint arXiv:2605.09998, 2026。

[39] Che, et al. [“DemoEvolve: Overcoming Sparse Feedback in Agentic Harness Evolution with Demonstrations.”](https://arxiv.org/abs/2605.24539) arXiv preprint arXiv:2605.24539, 2026。
