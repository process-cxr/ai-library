# 基于 Claude Code 可观测轨迹的 Coding Agent 训练数据构建研究报告

## 摘要

本文聚焦 Claude Code 类 Coding Agent 的系统设计、可观测轨迹结构及其面向自研模型训练的数据构建方法。分析对象包括 Claude Code 系统架构、Agent Multi Turn 轨迹样例以及长程 Agent 训练相关研究材料。本文的核心目标不是将原始轨迹简单整理为普通对话数据，而是研究如何将 Claude Code 类系统的真实执行轨迹编译为可用于自研 Coding Agent 模型训练的高质量 Agent Trace Data。

本文围绕三个问题展开分析：

1. Claude Code 这类 coding agent 的系统设计如何影响其可观测行为与轨迹形态；
2. 在真实运行或导出数据中，哪些轨迹信息可以被收集、重建、对齐，并转化为模型训练所需的状态、动作、观察与结果信号；
3. 如何将可观测轨迹转化为生产级 Agent Trace Training Data，用于自研模型的 Agent NTP、Agent SFT、偏好学习、PRM 与 RL 训练，并进一步通过模型 API 合成基于轨迹证据的思考推理轨迹，以增强模型的 Agentic reasoning 能力。

需要首先明确的是，Agent Multi Turn 数据轨迹与可直接用于模型训练的数据之间存在明显差异。Agent Multi Turn 更接近一次 Agent 任务执行过程的原始可观测记录，其核心作用是保存任务、环境、工具定义、消息流、工具调用、工具结果、子代理轨迹、推理信号和任务结果等执行痕迹。它记录的是“系统实际发生了什么”。而可用于训练的数据需要进一步表达“在某个状态下模型应学习什么行为、理解什么证据、生成什么动作、优化什么目标”。因此，Agent Multi Turn 是生产级训练数据的上游原材料，而不是最终训练样本本身。其训练价值需要通过 Action–Observation 对齐、Agent Role 识别、Context Reconstruction、Trajectory IR 构建、长轨迹切分、后验标注、质量过滤和格式导出等数据编译过程释放出来。

本文的核心结论是：基于 Claude Code 可观测轨迹所收集的会话记录、工具调用、子代理执行过程、运行时上下文、环境信息和任务结果等数据，可以构建覆盖 Agent NTP、Agent SFT 与 Agent RL 的多阶段训练数据体系。通过统一的 Trajectory IR，可以将原始轨迹重构为状态—动作—观察—结果形式，并进一步导出用于持续预训练的 Agent NTP Data、用于行为模仿的 Agent SFT Data，以及用于偏好学习、过程奖励建模和强化学习的 Agent RL Data。由此，Claude Code 类轨迹不仅可以用于监督微调，还可以进一步支撑 reward model、PRM、Verifier、Rejection Sampling、Offline RL 与 Online RL 等训练环节。

虽然真实的 provider hidden reasoning 通常不可直接获取，也无法从事后日志中被证明性还原，但这并不妨碍基于可观测轨迹构造高质量的思考推理监督信号。通过对轨迹中的状态、动作、观察和结果进行结构化重建，并额外调用强模型 API 进行后验补标，可以生成与轨迹证据一致的 trajectory-grounded synthetic reasoning。这类合成推理并不等同于真实 hidden reasoning，而是基于真实执行轨迹构造的过程监督信号，用于表达状态理解、证据整合、动作意图、决策依据、失败分析和长期任务推进逻辑。其价值不只是提升模型可解释性，更在于增强自研模型的 Agentic reasoning、工具使用、任务分解、子代理编排、长上下文维护、失败恢复和长期任务优化能力。

需要指出的是，关于生产级 Agent Trace Training Data 的具体合成流程、Trajectory IR 设计、标注策略、质量控制机制以及 RL 数据构建方法，目前仍缺乏公开且被广泛验证的统一最佳实践。本文提出的数据构建 pipeline 更多是一种基于现有可观测信息、Claude Code 类系统机制和长程 Agent 训练研究的可行性方案，其有效性及工程实现细节仍需结合后续实验验证不断迭代完善。

---

# 第一部分：Claude Code 系统设计分析

## 1. Claude Code 的系统定位

Claude Code 不应被理解为单纯的“模型聊天接口”，而应被理解为一个生产级 coding agent harness。其核心执行逻辑实际上相当简洁，本质上是一个持续运行的 agent loop：

```text
用户任务
→ runtime 构造上下文
→ 调用模型
→ 模型生成文本或工具调用
→ runtime 执行工具
→ 工具结果回填上下文
→ 模型继续决策
→ 任务完成或被中断
```

从控制流角度看，这一过程可以被概括为一个反复执行的 while-loop：

```text
while not finished:
    调用模型
    解析工具请求
    执行工具
    回填工具结果
```

因此，Claude Code 的复杂性并不主要来自 agent loop 本身，而是来自围绕该 loop 构建的大量外围系统。相关分析将其概括为一种“model judgment within a deterministic harness”的架构：模型负责局部判断与决策生成，而 deterministic harness 负责执行边界、上下文预算、安全控制、状态管理和可恢复性。

在这一架构中，模型主要承担以下职责：

```text
理解用户意图
理解当前上下文
选择下一步动作
生成工具调用参数
解释工具返回结果
决定任务是否完成
```

外围 harness 则负责：

```text
权限系统与安全控制
上下文组装与压缩
工具注册与调度
会话持久化
子代理编排
文件系统与 Shell 执行
Sandbox 与隔离机制
错误恢复与任务续跑
运行时可观测性
```

因此，Claude Code 的能力并不是单一模型能力的直接外显，而是以下组件共同作用的结果：

```text
frontier model
+ system prompt
+ tool schema
+ project memory
+ workspace context
+ permission system
+ hooks
+ subagents
+ compaction pipeline
+ session persistence
+ execution environment
```

这一定位对于理解 Claude Code 的系统设计至关重要。其核心思想并非通过复杂的显式规划器或状态机控制整个任务流程，而是在一个相对简单的 agent loop 外构建强约束、强状态管理和强可恢复性的运行时框架，使模型能够在受控环境中持续完成复杂的软件工程任务。

---

## 2. 高层架构组件

Claude Code 可被拆分为七类高层组件：

| 组件                    | 功能                                              |
| --------------------- | ----------------------------------------------- |
| User                  | 提交任务、审查结果、批准或拒绝高风险动作                            |
| Interfaces            | CLI、IDE、SDK、headless 模式等入口                      |
| Agent loop            | 调用模型、解析工具请求、调度工具、收集结果                           |
| Permission system     | 决定工具调用是否允许、询问、拒绝或升级                             |
| Tools                 | 文件读写、搜索、命令执行、MCP、Web、子代理等工具                     |
| State & persistence   | transcript、session、history、sidechain、checkpoint |
| Execution environment | shell、文件系统、远程环境、sandbox、外部服务                    |

其中，agent loop 是任务执行的中心，但真正复杂的工程设计大量分布在 loop 外围。生产级 coding agent 的差异化能力，往往来自“如何安全、连续、可恢复地执行任务”，而不仅是“模型是否能生成正确代码”。

---

## 3. 分层系统结构

从系统工程角度，Claude Code 可以进一步拆成五层：

```text
Surface layer
Core layer
Safety / action layer
State layer
Backend layer
```

各层含义如下：

| 层级                    | 内容                                                                  |
| --------------------- | ------------------------------------------------------------------- |
| Surface layer         | CLI、IDE、SDK、renderer、用户交互界面                                         |
| Core layer            | query loop、模型调用、上下文压缩流程                                             |
| Safety / action layer | 权限系统、hooks、工具、MCP、sandbox、subagent spawning                         |
| State layer           | 上下文组装、runtime state、session persistence、memory、sidechain            |
| Backend layer         | shell、filesystem、remote execution、MCP transports、external resources |

这一分层结构对于后续轨迹分析具有重要意义。不同层级产生的数据具有不同的语义属性和作用范围：部分信息属于模型决策过程中可直接访问的上下文，部分信息反映运行时状态与系统行为，另一些则仅用于工程监控、调试或审计。因而，在分析 agent 轨迹时，需要首先明确各类数据所对应的系统层级及其与模型决策过程之间的关系，而不能将所有日志、事件和状态信息视为同等性质的数据。

---

## 4. Agent loop：Reactive loop 与工具闭环

Claude Code 的核心执行方式接近 ReAct 式循环：

```text
模型判断当前状态
→ 选择工具或输出文本
→ runtime 执行动作
→ observation 回到上下文
→ 模型继续下一步
```

该 loop 的关键不是复杂的显式 planner，而是围绕简单循环构建强大的 operational harness。这意味着在训练数据中，最基本的监督单元应当是：

```text
state_t → action_t → observation_t → state_{t+1}
```

而不是：

```text
user → final answer
```

在 coding agent 训练中，真正有价值的监督信号包括：

```text
当前任务状态下应选择什么工具；
工具参数应如何构造；
读到某个文件后下一步应读什么；
测试失败后应如何定位；
何时派出子代理；
何时停止探索并输出总结。
```

因此，Claude Code 的轨迹不应被简单建模为对话数据，而应被建模为一系列状态转移与决策节点。

---

## 5. 工具系统、Agent 工具与执行策略

Claude Code 的执行能力建立在统一的 action interface 之上，但从系统设计角度看，并非所有 action 都属于同一种类型。生产级 coding agent 通常同时包含两类能力：

```text
普通工具（Tools）
Agent 工具（Agentic Tools）
```

二者虽然在模型侧都可能表现为一次 tool call，但其语义、执行方式和训练价值存在本质差异，因此不宜完全混为一体。

### 5.1 普通工具（Tools）

普通工具是直接作用于外部环境的执行接口，其特点是：

```text
一次调用对应一次具体操作；
执行结果由 runtime 直接返回；
通常不包含独立推理过程；
生命周期较短；
结果可立即观察。
```

典型工具包括：

```text
list_dir
read_file
grep_content
glob_path
edit_file
write_file
run_command
read_lints
web_search
web_fetch
```

例如：

```text
read_file(path)
→ 返回文件内容

list_dir(path)
→ 返回目录结构

run_command(cmd)
→ 返回命令输出
```

从 agent 视角看，这类工具属于环境交互接口（environment interaction primitives），主要用于获取信息或产生外部副作用。

---

### 5.2 Agent 工具（Agentic Tools）

Agent 工具表面上也是一次 tool call，但其内部并非执行单一步骤，而是启动一个新的智能体执行过程。

典型代表是：

```text
delegate_subtask
task
explore
research
planner
reviewer
verification agent
```

其执行模式更接近：

```text
主代理
→ 创建子任务
→ 启动子代理
→ 子代理执行完整推理与工具调用过程
→ 返回总结结果
```

因此：

```text
普通工具：
  Action → Observation

Agent 工具：
  Action → Sub-Trajectory → Summary
```

对于主代理而言，调用子代理与调用普通工具在接口层面可能没有区别；但从系统内部看，Agent 工具实际上对应一个新的 agent loop，而不是一次简单函数执行。

---

### 5.3 Agent 工具的特殊性

Agent 工具与普通工具至少存在三个关键差异。

第一，返回结果的来源不同。普通工具返回的是环境状态，例如文件内容、目录结构、命令输出、搜索结果。Agent 工具返回的是另一个智能体经过推理后的工作成果，例如架构分析、代码库探索结果、问题诊断报告、测试结论或实现方案。因此其输出本质上属于 compressed reasoning artifact，而不是原始环境观察。

第二，内部可能包含完整轨迹。普通工具通常只有 tool_call 与 tool_result，而 Agent 工具内部可能包含 system prompt、subtask、tool calls、tool results、intermediate reasoning 和 final summary。也就是说，主代理看到的是一次 delegate_subtask 调用及其 summary，而数据工程视角可以观察到完整子代理轨迹。

第三，Agent 工具承担任务分解功能。普通工具主要用于获取信息、修改环境或执行命令；Agent 工具则用于任务拆解、上下文隔离、并行探索和专业化处理。例如，当用户要求绘制大型 monorepo 架构图时，直接让主代理读取大量文件并不现实，更合理的策略是由主代理创建 Explore 子任务，Explore 子代理完成代码库调研，再由主代理生成最终回答。

---

### 5.4 统一接口与分层建模

尽管普通工具和 Agent 工具存在本质差异，但在 Claude Code 等系统中，它们通常通过统一 tool-calling 接口暴露给模型：

```json
{
  "name": "delegate_subtask",
  "arguments": {...}
}
```

与：

```json
{
  "name": "read_file",
  "arguments": {...}
}
```

在调用格式上并无区别。

因此从模型接口层面看，二者都是 tool call。但从训练数据和系统分析角度看，应进行分层建模：

```text
Tool
├── Environment Tool
│   ├── read_file
│   ├── list_dir
│   ├── grep_content
│   ├── edit_file
│   └── run_command
│
└── Agent Tool
    ├── delegate_subtask
    ├── Explore
    ├── Research
    ├── Planner
    └── Reviewer
```

这种区分对于后续训练数据构建尤为重要，因为普通工具样本主要用于学习环境交互能力，而 Agent 工具样本则用于学习任务分解、子代理编排和长程任务管理能力。

---

### 5.5 工具执行策略

Claude Code 的工具系统不仅是一组可调用函数，更是一套包含工具定义、使用策略、权限约束和执行调度机制的行动空间。模型在执行任务时，并非仅根据工具名称进行选择，而是在工具说明、当前任务状态和运行环境约束共同作用下完成决策。

典型工具可大致分为以下几类：

```text
文件与目录操作：
  list_dir
  read_file
  glob_path
  grep_content

代码修改：
  edit_file
  write_file

执行与诊断：
  run_command
  read_lints

任务管理与协作：
  delegate_subtask
  todo_write

外部信息获取：
  web_search
  web_fetch
  codebase_search
```

值得注意的是，工具 schema 通常不仅定义参数结构，还隐含了一系列行为规范。例如，复杂代码库探索应优先委派 Explore 子代理；已知路径的文件读取应直接使用 `read_file`；精确文本检索应使用 `grep_content`；文件修改通常要求先完成读取和理解；通用命令执行工具也不应替代已有的专用工具。这些约束实际上构成了 agent 的操作策略，并直接影响工具选择行为。

从训练数据构建角度看，工具 schema 因而属于重要的决策上下文。如果仅保留工具名称和参数，而忽略工具说明及其使用规则，模型将难以学习真实系统中的行为偏好和执行规范。

此外，不同工具对应的风险等级和副作用也存在明显差异：

| 工具类型  | 示例                               | 副作用       | 典型用途       |
| ----- | -------------------------------- | --------- | ---------- |
| 只读工具  | list_dir、read_file、grep_content  | 无状态修改     | 信息收集与环境理解  |
| 修改工具  | edit_file、write_file、delete_file | 修改文件系统    | 代码实现与修复    |
| 命令工具  | run_command                      | 可能产生任意副作用 | 构建、测试与环境操作 |
| 子代理工具 | delegate_subtask                 | 启动独立执行链路  | 任务分解与上下文隔离 |

这种工具分层反映了 coding agent 的基本执行模式：优先通过低风险工具收集信息，在获得充分证据后再执行具有副作用的操作；对于需要大量探索或独立上下文的任务，则通过子代理机制完成分解与协作。因此，在生产级训练数据中保留工具类别、风险属性和使用约束，有助于模型学习更符合真实 agent 系统的决策策略。

---

## 6. 权限系统与 hooks

Claude Code 采用保守的权限与安全边界。权限系统不仅决定工具调用是否可以执行，也影响模型可见工具集合、action 是否被阻断、工具结果是否被修改或追加上下文。

Hooks 是 Claude Code runtime 的生命周期回调机制。它们可以出现在：

```text
SessionStart
UserPromptSubmit
PreToolUse
PostToolUse
SubagentStart
SubagentStop
PreCompact
PostCompact
Stop
```

Hooks 在轨迹数据中的作用需要分三类理解。

第一类是只记录的 hook，例如 session start 上报、stop 阶段数据记录、duplicate detector 等。这类 hook 通常不进入模型上下文，只属于 runtime metadata。其价值在于审计、复现和调试。

第二类是阻断或改写执行流程的 hook，例如 `PreToolUse:Bash` 阻止危险命令执行，或要求用户确认。这类 hook 可能不直接进入模型输入，但会改变轨迹结果，应在结构化数据中保留为 runtime intervention。

第三类是注入 additional context 的 hook，例如：

```xml
<system-reminder>
PostToolUse:Edit hook additional context:
<ide_diagnostics>...</ide_diagnostics>
</system-reminder>
```

这类 hook 的输出是模型可见上下文，应进入训练输入。若后续模型基于 IDE diagnostics 清理未使用 import、修复 lint error 或继续修改文件，该 hook 内容就是 action 的因果依据之一。

因此，hook 不应被统一处理。生产级数据构建需要判断每个 hook 是否模型可见、是否影响执行、是否仅为日志。

---

## 7. 上下文管理与 compaction

Claude Code 的核心瓶颈之一是 context window。实际模型调用的上下文可能来自：

```text
system prompt
tool schema
user task
conversation history
workspace env
git status
CLAUDE.md / project memory
skills / rules / MCP descriptions
file contents
tool results
subagent summaries
system reminders
compaction summaries
```

由于长程 coding 任务会快速消耗上下文窗口，Claude Code 需要通过多阶段上下文压缩与信息保留机制控制上下文预算。相关分析材料显示，其上下文管理流程并非单一裁剪策略，而是由多个层次的压缩阶段组成，包括：

```text
budget reduction
snipping
micro-compaction
context collapse
auto-compaction summary
```

这些阶段的共同目标是在有限上下文窗口内保留任务执行所需的关键信息，同时逐步移除低价值历史内容。随着会话长度增长，系统会从局部结果压缩逐步过渡到更高层级的历史摘要与状态重构，从而维持长程任务的连续性与可执行性。

这意味着：

```text
完整 transcript ≠ 某一步模型实际看到的上下文
```

从系统运行机制来看，需要区分两个不同的问题：其一是“模型在某一次调用时实际接收到了什么输入”，其二是“事后能够从哪些数据源中重建这次输入”。

理论上，如果能够获取某次模型调用对应的完整 API Request Body，那么该 Request Body 就是该次调用最完整、最准确的模型输入，不存在比它更接近真实上下文的表示。然而，在实际系统中，研究者通常能够获得的是会话 transcript、工具日志、hook 日志、OTel trace、文件快照等外围记录，而不一定能够获得每一次模型调用的完整 Request Body。此时即使拥有完整 transcript，也只能根据可观测事件去推测当时的上下文组装结果，而无法保证百分之百还原模型实际看到的输入。

另一方面，Claude Code 的上下文构造本身是动态的。系统会从 transcript、工具结果、项目记忆、CLAUDE.md、环境信息、压缩摘要等多个来源中抽取内容，并根据上下文窗口预算进行裁剪和重组。因此，完整 transcript 实际上记录的是“系统经历过什么”，而某次模型调用的 Request Body 记录的是“模型当时看到了什么”。前者通常包含更多历史事件，后者则是经过上下文管理机制处理后的最终输入。

因此，严格来说：

```text
完整 API Request Body
    ≈ 某次模型调用的真实输入（最完整）

完整 transcript
    ≈ 系统运行过程的完整事件历史

仅凭 transcript 重建的上下文
    ≈ 对真实模型输入的近似还原
```

真正存在“不完整”风险的并不是模型实时接收到的输入，而是事后分析时能否拿到当时的完整 Request Body。如果只能获得 transcript 或部分日志，那么重建出的上下文就可能与模型实际看到的内容存在差异。

这一特性表明，在分析 Claude Code 的行为时，需要区分“系统保存的完整历史记录”与“模型在某一时刻实际可见的上下文”。前者反映了任务执行过程的全貌，后者则决定了模型当时的决策依据。仅依据 transcript 通常能够重建可观测的会话历史，但无法严格证明某一时刻模型实际接收到的完整输入内容。

---

## 8. Subagent 与 sidechain 机制

Claude Code 的 subagent 机制是复杂 coding agent 的关键能力之一。子代理具有独立上下文，可以执行多步任务，并最终向父代理返回 summary。父代理通常不会吸收子代理完整消息历史，而只接收最终结果。

这种设计具有两重意义：

1. **上下文隔离**：子代理可以在独立窗口中大量读取文件和探索代码，而不污染父代理上下文；
2. **summary-only return**：父代理只接收压缩后的结果，避免 context explosion。

从公开可观测资料和轨迹样例来看，Claude Code 支持多种 specialized subagent。不同版本和配置下具体名称可能有所变化，但其设计目标通常围绕任务专业化展开。已观察到的典型类型包括：

| Subagent 类型           | 主要职责                          |
| --------------------- | ----------------------------- |
| Explore               | 代码库探索、架构分析、依赖关系梳理、定位实现位置      |
| General               | 通用任务处理，适用于不属于特定专业领域的子任务       |
| Verification / Review | 验证实现结果、检查逻辑正确性、辅助代码审查         |
| Research 类 Agent      | 收集资料、阅读文档、分析外部信息源             |
| Domain-specific Agent | 由系统或扩展定义的特定领域代理，例如测试、文档、迁移等任务 |

其中，Explore 是目前最明确可观测到的专用子代理类型。在相关轨迹样例中，主代理在识别到大型 monorepo 架构分析任务后，会调用 `delegate_subtask(agent_type="Explore")`，并要求其完成代码库结构探索、配置分析、依赖关系梳理和架构总结等工作。这表明 Claude Code 的 subagent 并非简单复制主代理能力，而是通过专门的系统提示词和任务边界强化特定能力方向。

从系统设计角度看，subagent 可以被理解为一种轻量级层次化 Agent Architecture：

```text
Main Agent
  ├── Explore Agent
  ├── Verification Agent
  ├── Research Agent
  └── Other Specialized Agents
```

从可观测轨迹来看，可以确认系统存在不同类型的 specialized agent，并且这些 agent 具有不同的 system prompt、工具权限或任务职责划分。但仅依据当前可观测数据，尚无法确定它们是否对应不同底层模型，还是共享同一模型并通过上下文、系统提示词和工具配置实现角色差异。因此，更稳妥的表述是将其视为不同的 agent role，而非不同的 model。

主代理负责全局任务规划与结果整合，子代理负责局部问题求解。子代理完成任务后，仅返回压缩后的结果摘要，而不是完整执行历史。

从训练角度看，应分别构建两类样本：

```text
main-agent sample:
  用户任务 + 父代理历史 + 子代理返回 summary → 父代理下一步动作或最终回答

subagent sample:
  子任务 query + 子代理工具轨迹 → 子代理最终总结
```

进一步地，不同类型子代理应被视为不同的数据分布：

```text
Explore Agent:
  强调搜索、阅读、归纳和架构理解

Verification Agent:
  强调检查、验证和错误发现

Research Agent:
  强调信息收集与证据整合

General Agent:
  强调通用问题求解
```

因此，Claude Code 的子代理机制本质上是一种上下文隔离与任务分解机制。通过将特定任务委派给独立代理执行，并仅返回压缩后的结果摘要，系统能够在有限上下文预算下完成更复杂的代码库探索、信息收集和分析任务，同时避免不同任务轨迹之间产生不必要的上下文干扰。这种设计体现了 Claude Code 在长程任务执行过程中对上下文资源管理和任务组织方式的系统性考虑。

---

# 第二部分：Agent Multi Turn 轨迹格式分析

## 1. 数据格式定位

`agent_multi_turn` 与传统对话数据集中的单轮或多轮 chat message 不同，其设计目标并非记录自然语言对话，而是记录一次完整 Agent 任务执行过程中的状态变化、工具调用、子代理协作和最终结果。

从结构上看，该格式已经超出了普通 SFT 数据的范畴，更接近于 Agent Trajectory Dataset。其记录对象不是：

```text
User → Assistant
```

而是：

```text
Task
→ Agent Decision
→ Tool Call
→ Tool Observation
→ Subagent Execution
→ Result Integration
→ Final Answer
```

因此，该格式天然适合作为 Agent Training Data 的原始载体。

第一部分讨论的是 Claude Code 为什么会产生这样的轨迹，第二部分讨论的是这些轨迹在数据结构中如何被表达。二者之间的对应关系如下：

| 系统设计机制         | 轨迹数据中的表现                                               |
| -------------- | ------------------------------------------------------ |
| Agent loop     | message trace、assistant decision、tool call、tool result |
| 工具系统           | tools schema、tool_calls、tool observations              |
| Agent 工具 / 子代理 | segments、agent_type、subagent trajectory                |
| 上下文管理          | system prompt、history window、summary、context boundary  |
| 权限与 hooks      | runtime metadata、system-reminder、additional context    |
| 会话持久化          | transcript、segment、message id、tool_call_id             |
| 执行环境           | repo、branch、commit_id、cwd、workspace metadata           |
| 任务结果           | final answer、subagent summary、outcome record           |

因此，`agent_multi_turn` 不应被视为孤立的数据格式，而应被视为 Claude Code 类系统设计在数据层面的映射。

---

## 2. 顶层结构分析

一个标准的 `agent_multi_turn` 样本通常可以抽象为如下顶层结构：

```json
{
  "version": "2026-05-21",
  "data_id": "agent_trace_000123",
  "src": "coding-agent-runtime",
  "dataset_id": "agent_trace_dataset",
  "format": "agent_multi_turn",
  "domain": {
    "primary_domain": "code",
    "secondary_domain": "agent"
  },
  "message": {
    "tools": [
      {
        "name": "list_dir",
        "description": "...",
        "parameters": { "...": "..." }
      },
      {
        "name": "read_file",
        "description": "...",
        "parameters": { "...": "..." }
      },
      {
        "name": "delegate_subtask",
        "description": "...",
        "parameters": {
          "agent_type": "Explore",
          "query": "..."
        }
      }
    ],
    "segments": [
      {
        "segment_id": "segment_0",
        "agent_role": "main_agent",
        "messages": [
          {
            "role": "user",
            "content": "画一下 sporehub 的框架图"
          },
          {
            "role": "assistant",
            "reasoning_content": "...",
            "tool_calls": [
              {
                "id": "toolu_001",
                "name": "list_dir",
                "arguments": {
                  "target_directory": "/repo"
                }
              }
            ]
          },
          {
            "role": "tool",
            "tool_call_id": "toolu_001",
            "content": "packages/\napps/\nREADME.md"
          },
          {
            "role": "assistant",
            "reasoning_content": "...",
            "tool_calls": [
              {
                "id": "toolu_002",
                "name": "delegate_subtask",
                "arguments": {
                  "agent_type": "Explore",
                  "query": "分析仓库结构并总结系统架构"
                }
              }
            ]
          }
        ]
      },
      {
        "segment_id": "segment_1",
        "agent_role": "explore_agent",
        "agent_type": "Explore",
        "messages": [
          {
            "role": "user",
            "content": "分析仓库结构并总结系统架构"
          },
          {
            "role": "assistant",
            "reasoning_content": "...",
            "tool_calls": [
              {
                "id": "toolu_101",
                "name": "list_dir",
                "arguments": {
                  "target_directory": "/repo/packages"
                }
              }
            ]
          },
          {
            "role": "tool",
            "tool_call_id": "toolu_101",
            "content": "core/\nweb/\nserver/"
          },
          {
            "role": "assistant",
            "tool_calls": [
              {
                "id": "toolu_102",
                "name": "read_file",
                "arguments": {
                  "file_path": "/repo/package.json"
                }
              }
            ]
          },
          {
            "role": "tool",
            "tool_call_id": "toolu_102",
            "content": "{...}"
          },
          {
            "role": "assistant",
            "content": "Explore Summary: 仓库采用 monorepo 结构，由 web、server、core 等模块组成..."
          }
        ]
      },
      {
        "segment_id": "segment_2",
        "agent_role": "main_agent",
        "messages": [
          {
            "role": "user",
            "content": "画一下 sporehub 的框架图"
          },
          {
            "role": "assistant",
            "tool_calls": [
              {
                "id": "toolu_002",
                "name": "delegate_subtask",
                "arguments": {
                  "agent_type": "Explore"
                }
              }
            ]
          },
          {
            "role": "tool",
            "tool_call_id": "toolu_002",
            "content": "Explore Summary: 仓库采用 monorepo 结构..."
          },
          {
            "role": "assistant",
            "content": "最终架构分析与框架图说明..."
          }
        ]
      }
    ]
  },
  "environment": {
    "repo": "sporehub",
    "branch": "main",
    "commit_id": "abc123"
  }
}
```

与传统对话数据相比，`agent_multi_turn` 的核心特征并不只是保存用户与模型之间的消息，而是将一次完整 Agent Runtime 的执行过程结构化保存下来。其记录对象已经从传统的“对话（Conversation）”扩展为“任务执行轨迹（Execution Trajectory）”。

从顶层结构看，该格式至少包含六类信息：

| 顶层字段                 | 含义                  | 主要用途                  |
| -------------------- | ------------------- | --------------------- |
| `version`            | 数据格式或构建版本           | 版本管理、兼容性判断            |
| `data_id`            | 单条样本唯一标识            | 去重、追踪、实验复现            |
| `src` / `dataset_id` | 数据来源与数据集编号          | 数据治理、来源审计             |
| `domain`             | 任务领域标注              | 数据筛选、任务分桶             |
| `message.tools`      | 当前任务可用工具定义          | 表达 Agent action space |
| `message.segments`   | 主代理、子代理或不同执行视角的消息轨迹 | 表达任务执行过程              |
| `environment`        | 仓库、分支、commit 等环境信息  | 限定任务上下文与可复现性          |

其中，最重要的结构是 `message`。它通常承载两类核心内容：

```text
message.tools
  → 定义 Agent 在该任务中可以执行哪些动作

message.segments
  → 记录 Agent 在任务执行过程中的多段执行轨迹
```

因此，`agent_multi_turn` 不是简单的：

```text
metadata + messages
```

而更接近：

```text
metadata
+ environment
+ action space
+ execution segments
+ agent messages
+ tool calls
+ tool observations
+ subagent trajectory
+ final outcome
```

结合实际样本可以发现，该格式不仅保存了用户请求和最终回答，还保存了工具定义、工具调用、工具返回结果、子代理执行过程、仓库上下文以及阶段性决策信息，因此更接近真实生产环境中的 Agent Trace，而非传统 Chat Completion 数据。

从结构上看，一个完整样本可以进一步拆解为多个信息层级：任务级元数据层、工具定义层、Segment 编排层、Agent Role 层、Message Trace 层、Action–Observation 对齐层、Repository Context 层、Subagent Trajectory 层、Reasoning Signal 层和 Outcome 层。后续章节将分别分析这些层级的含义及其训练价值。

---

## 3. 任务级元数据层（Task Metadata Layer）

顶层字段主要承担数据治理和样本管理职责：

```json
{
  "version": "2026-05-21",
  "data_id": "...",
  "src": "coding-agent-runtime",
  "dataset_id": "...",
  "format": "agent_multi_turn",
  "domain": {
    "primary_domain": "code",
    "secondary_domain": "agent"
  }
}
```

这些字段本身通常不会进入模型上下文，但对于数据工程系统而言十分重要。其作用包括：

```text
样本唯一标识
数据版本管理
来源追踪
数据集切分
实验复现
质量审计
去重索引
```

尤其是在大规模 Agent 数据集构建过程中，`data_id`、`dataset_id` 和 `src` 往往是后续数据治理系统的核心索引字段。

---

## 4. 工具定义层（Tool Schema Layer）

样本中的 `message.tools` 保存了当前任务可用工具的完整定义。

从 Agent 训练视角来看，这些工具可以进一步划分为两类：

```text
普通执行工具（Execution Tools）
Agent 工具（Agentic Tools）
```

普通执行工具主要负责与环境交互，例如：

```text
list_dir
read_file
grep_content
glob_path
run_command
edit_file
write_file
web_search
web_fetch
```

其作用是获取信息、执行命令或修改环境状态。

Agent 工具则负责任务分解、状态管理或多智能体协作，例如：

```text
delegate_subtask
todo_write
codebase_search
```

其中 `delegate_subtask` 最具代表性，它并不是简单调用某个外部能力，而是在运行时创建新的 Agent 执行单元，并形成独立轨迹。因此从系统设计角度看，它更接近一种控制流操作（Control Action），而不是普通工具调用。

不过在 Claude Code 的实际运行机制中，两类能力最终都会以统一 Tool Schema 的形式暴露给模型：

```text
模型视角：
Tool A
Tool B
Tool C
...

Runtime 视角：
Execution Tool
Agent Tool
Control Tool
...
```

因此在轨迹数据分析阶段，可以统一归类为 Tool Schema；但在后续行为建模与能力分析过程中，仍有必要进一步区分工具类型，因为不同类型工具往往对应不同的决策逻辑与执行模式。

例如：

```json
{
  "name": "read_file",
  "tool_type": "execution"
}
```

```json
{
  "name": "delegate_subtask",
  "tool_type": "agent"
}
```

```json
{
  "name": "todo_write",
  "tool_type": "planning"
}
```

与传统 Function Calling 数据不同，这里的 Tool Schema 不只是函数签名，而是完整的 Agent Action Specification。一个工具定义通常包含：

```text
工具名称
工具描述
参数定义
参数约束
使用场景
推荐策略
风险提示
反模式说明
```

因此：

```text
Tool Schema ≠ Function Signature

Tool Schema = Action Space Definition + Behavior Policy + Decision Guidance
```

其中：

* Action Space Definition 定义模型能够执行哪些动作；
* Behavior Policy 定义工具的使用规则和约束；
* Decision Guidance 定义工具选择时的决策依据。

在 Agent 系统中，Tool Schema 的作用远超传统 API 或 Function Signature。它不仅描述工具“能做什么”，还在很大程度上决定模型“应该如何做决策”。对于 Agent 学习而言，工具定义本身就是上下文的一部分。

---

## 5. Segment 编排层（Execution Segmentation Layer）

与普通聊天数据最大的区别之一，是该格式引入了：

```json
{
  "segments": [...]
}
```

这一结构。

在传统聊天数据中：

```text
一个任务 = 一条连续消息链
```

而在 Agent Multi Turn 中：

```text
一个任务 = 多个执行上下文（Execution Context）
```

实际样本中可以观察到如下结构：

```text
Segment 0
主代理启动阶段

Segment 1
Explore 子代理执行阶段

Segment 2
主代理整合结果阶段
```

需要注意的是，Segment 并不一定构成严格顺序且互不重叠的切分。在实际样本中，Segment 0 与 Segment 2 可能存在明显的上下文重叠：两者都包含用户原始请求、主代理初始工具调用以及委派子代理的过程。区别在于，Segment 0 更接近主代理在委派前后的局部执行片段，而 Segment 2 则记录了从任务开始到最终回答的完整主代理闭环。

因此，Segment 更应理解为不同执行视角（execution view）或上下文边界（context boundary）的记录方式，而不是简单的时间切片。多个 Segment 之间可能存在：

```text
前缀重叠（Prefix Overlap）
共享消息历史
不同粒度的执行记录
父代理与子代理的视角差异
```

因此：

```text
Segment ≠ Message Group
Segment ≠ Strict Timeline Slice
Segment = 独立 Agent Context / Execution View
```

每个 Segment 都可能拥有：

```text
独立 System Prompt
独立工具调用历史
独立上下文窗口
独立任务目标
独立执行结果
```

从系统设计角度看，这实际上反映了 Claude Code 类系统中的 Sidechain / Subagent Architecture，同时也体现了生产级 Agent 系统为了支持上下文隔离、轨迹复用和多视角记录而采用的分段式执行表示。

---

## 6. Agent Role 层（Agent Role Layer）

进一步分析 Segment 可以发现，每个 Segment 实际对应一个 Agent Role，而且不同 Segment 所处的上下文边界并不相同。

以典型样本为例：

```text
Segment 0 → Main Agent（Planning Phase）
Segment 1 → Explore Agent（Exploration Phase）
Segment 2 → Main Agent（Synthesis Phase）
```

Segment 0 对应主代理的任务规划阶段，负责：

```text
理解用户需求
识别任务类型
初步探索项目结构
制定执行策略
决定是否调用子代理
构造子代理任务描述
```

Segment 1 对应 Explore Agent，负责：

```text
代码库探索
文件阅读
配置分析
依赖关系分析
构建流程分析
架构理解
证据收集
总结输出
```

Segment 2 再次回到 Main Agent，但此时已经进入结果整合阶段，负责：

```text
接收 Explore Agent 返回结果
整合探索证据
组织最终答案
生成架构说明
输出用户可读结果
```

因此，从 Agent Role 的角度看，一个完整样本实际上包含两个不同 Agent 的行为轨迹：

```text
Main Agent
Explore Agent
```

如果进一步按照执行阶段划分，则可以识别出三个不同的行为角色：

```text
Task Planner
Repository Explorer
Answer Synthesizer
```

其对应关系为：

```text
Segment 0 → Task Planner
Segment 1 → Repository Explorer
Segment 2 → Answer Synthesizer
```

这意味着训练时不能简单将所有 Segment 拼接成一条长对话，而应首先识别：

```text
当前 Segment 属于哪个 Agent Role
当前 Agent 处于哪个执行阶段
```

否则会破坏真实运行时的上下文边界，并混淆主代理与子代理之间的职责分工。

---

## 7. Message Trace 层（Message Trace Layer）

每个 Segment 内部包含：

```json
{
  "messages": [...]
}
```

这是 Agent 的实际执行轨迹。

在当前样例中，最常见的 Message Role 包括：

```text
user
assistant
tool
```

但生产级 Agent 系统中的 Message Role 并不一定只有这三种。不同 Provider、Agent Framework 或 Runtime 可能引入额外角色，因此训练数据构建时不应假设 Role 集合固定。

常见角色可分为以下几类：

| Role        | 含义                               | 是否通常可见于模型上下文 |
| ----------- | -------------------------------- | ------------ |
| user        | 用户输入                             | 是            |
| assistant   | 模型输出                             | 是            |
| tool        | 工具执行结果                           | 是            |
| system      | 系统提示词、运行时指令                      | 是            |
| developer   | 开发者级指令                           | 是            |
| function    | 早期 Function Calling 返回结果         | 是            |
| environment | 某些 Agent Framework 中的环境反馈        | 视实现而定        |
| observation | 部分框架对 Tool Result 的抽象表示          | 视实现而定        |
| critic      | Critic Agent 或 Verifier Agent 输出 | 视实现而定        |
| planner     | Planner Agent 输出                 | 视实现而定        |

虽然底层实现不同，但本质上都可以归约为三类信息流：

```text
Instruction Message
Action Message
Observation Message
```

---

### 7.1 User Message

用户消息定义任务目标，例如：

```json
{
  "role": "user",
  "content": "画一下 sporehub 的框架图"
}
```

这是整个轨迹的起点。在部分系统中，用户消息之外还可能存在 system、developer、workspace instruction、project memory 等信息，这些消息共同构成任务上下文。

---

### 7.2 Assistant Message

Assistant Message 是 Agent 的决策记录。实际样本中通常包含：

```json
{
  "role": "assistant",
  "content": "",
  "tool_calls": [...],
  "reasoning_content": "...",
  "model_name": "..."
}
```

需要注意的是：

```text
Assistant Message ≠ 最终回答
```

在 Agent 数据中，更准确的理解方式是：

```text
Assistant Message = 当前状态下的决策结果
```

或者说：

```text
Assistant Message = Decision Node
```

这里的 Decision Node 指的是 Agent 在观察当前环境状态后，决定下一步应该做什么。从训练角度看，Agent 学习的核心并不是如何生成漂亮的自然语言，而是如何在不同状态下做出正确决策。因此很多 Agent 训练任务实际上是在学习：

```text
State → Decision
```

而不是：

```text
Question → Answer
```

对于支持 Tool Calling 的系统，这种特征会更加明显。模型输出的内容可能不是一句自然语言，而是：

```json
{
  "tool": "read_file",
  "arguments": {
    "file_path": "package.json"
  }
}
```

此时 Assistant Message 的作用已经不是回答用户，而是在告诉运行时：

```text
我决定调用 read_file 工具，并且参数是 package.json。
```

因此，在 Tool Calling Agent 中，Assistant Message 可能承担以下职责：

```text
Action Selection
Parameter Generation
Subagent Delegation
Plan Update
Final Answer
```

其中只有最后一种情况属于传统意义上的“回复用户”。

---

## 8. Action–Observation 对齐层

Action–Observation 对齐是 Agent 轨迹结构化过程中最基础的步骤之一。由于原始轨迹通常已经完整记录了消息流，因此容易产生一种误解：既然工具调用和工具返回都已经存在于日志中，似乎没有必要再进行额外处理。

实际上，对齐的目的并非修改原始数据，也不是生成新的信息，而是将日志中按时间顺序记录的离散事件重构为符合 Agent 决策过程的状态转移单元。

在原始轨迹中，工具调用与工具返回通常以独立消息形式存在：

```json
{
  "role": "assistant",
  "tool_calls": [
    {
      "id": "toolu_xxx",
      "name": "list_dir",
      "arguments": {
        "target_directory": "/repo"
      }
    }
  ]
}
```

随后出现对应的工具执行结果：

```json
{
  "role": "tool",
  "tool_call_id": "toolu_xxx",
  "content": "packages/\nREADME.md\npackage.json"
}
```

从事件记录角度看，上述两条消息已经完整描述了一次工具交互过程。然而，从 Agent 学习与行为建模角度看，更重要的是恢复决策链路中的因果关系：

```text
当前状态
→ 执行动作
→ 获得环境反馈
→ 更新状态
→ 产生下一步决策
```

因此，需要利用 `tool_call_id` 等关联字段，将原本分散记录的调用事件与返回事件重新组织为统一的动作单元：

```json
{
  "action": {
    "tool": "list_dir",
    "arguments": {
      "target_directory": "/repo"
    }
  },
  "observation": {
    "content": "packages/\nREADME.md\npackage.json"
  }
}
```

这种结构通常不直接存在于原始采集数据中，而是在数据预处理阶段构建的中间表示。其作用是显式表达 Agent 与环境之间的交互过程，使后续训练能够围绕状态转移而非消息序列展开。

---

## 9. Repository Context 层

除消息轨迹与工具调用外，样本中还包含仓库级环境信息，例如：

```json
{
  "repo": "...",
  "branch": "...",
  "commit_id": "..."
}
```

从系统设计角度看，这些字段属于 Agent 所处执行环境的一部分，其作用并非简单的数据标识，而是为模型提供任务语义解释所需的上下文边界。

对于 Coding Agent 而言，同一自然语言任务在不同代码仓库中往往对应完全不同的执行路径。例如，“修复登录问题”在前端仓库中可能涉及页面状态管理与接口调用逻辑，而在后端仓库中则可能涉及认证服务、数据库访问控制或会话管理机制。

因此，仓库上下文不仅是数据管理字段，更是 Agent 状态的重要组成部分。其主要作用包括：

```text
确定任务作用域
提供代码语义背景
约束工具搜索空间
支持结果可复现性
关联具体代码版本
```

在训练数据构建过程中，Repository Context 应被视为环境状态的一部分，并与任务描述、工具定义和执行历史共同构成 Agent 的决策输入。

---

## 10. Subagent Trajectory 层

Subagent Trajectory 是该类 Agent 数据区别于传统对话数据的重要特征之一。

在多数公开 Agent 数据集中，子代理通常仅表现为一次抽象调用：

```text
Main Agent
→ Delegate Task
→ Receive Summary
```

而在当前样本结构中，子代理自身的执行过程被完整保留下来，包括：

```text
子任务定义
工具调用过程
搜索与阅读行为
中间观察结果
阶段性分析
最终总结
```

因此可以同时观察两个层面的执行轨迹：

```text
Main Agent Trajectory
Subagent Trajectory
```

以及两者之间的信息传递关系：

```text
Main Agent
→ Delegate

Subagent
→ Explore
→ Collect Evidence
→ Analyze
→ Summarize

Main Agent
→ Integrate Evidence
→ Produce Final Output
```

这种结构使得任务分解过程成为可训练对象。

从训练角度看，可以分别构建：

```text
Main-Agent Training Sample
Subagent Training Sample
Delegation Decision Sample
Summary Integration Sample
```

从而学习：

```text
Task Decomposition
Agent Orchestration
Subagent Delegation
Multi-Agent Collaboration
Hierarchical Planning
```

与传统单代理轨迹相比，这类数据更接近真实生产环境中的 Agent 协作模式，因此具有更高的研究价值和训练价值。

---

## 11. Reasoning Signal 层

部分 Assistant Message 中包含：

```json
{
  "reasoning_content": "..."
}
```

例如：

```text
I need to understand the project structure before I can create a framework diagram.
```

从数据属性分析，该字段更适合被定义为：

```text
Observable Reasoning Signal
```

而非：

```text
Hidden Chain of Thought
```

原因在于：

```text
内容长度较短
聚焦当前决策
通常对应单步动作
缺乏完整推理展开过程
不覆盖全部状态信息
```

其本质更接近于：

```text
Decision Rationale
Action Intent
State Interpretation
```

而不是模型内部完整推理过程的直接暴露。

因此，在训练数据体系中，该字段更适合作为 Synthetic Reasoning Seed。需要强调的是，Observable Reasoning Signal 与 Hidden Reasoning 在数据定义上应严格区分。前者属于模型显式输出内容，后者对应模型内部计算过程，两者既不属于同一类监督信号，也不能相互替代。

但这并不削弱其推理建模价值。相反，Observable Reasoning Signal 可以作为重要的推理锚点。当其与同一步骤的上下文状态、动作选择、工具观察结果以及任务结果结合时，可用于后验构造更完整的推理轨迹。

具体形式如下：

```text
Context
+ Observable Reasoning Signal
+ Action
+ Observation
+ Outcome
```

在此基础上，可通过后验标注或模型补全生成基于轨迹证据的合成推理：

```text
Trajectory-Grounded Synthetic Reasoning
```

这类合成推理并不等同于模型真实发生的内部思考过程，也不应被视为 Hidden Reasoning 的恢复结果。更准确地说，它是一种基于可观测证据的推理重建，用于解释：

```text
模型当时可能掌握的信息
当前动作的选择依据
支撑决策的关键证据
决策与最终结果之间的关联
```

---

## 12. Outcome 层（Task Outcome Layer）

任何 Agent Trajectory 最终都会收敛到某种任务结果。

在 Coding Agent 场景中，Outcome 通常表现为：

```text
Subagent Summary
+
Main Agent Final Answer
```

或者进一步表现为：

```text
代码修改结果
架构分析结果
Bug 修复结果
设计方案
测试报告
文档生成结果
```

从轨迹建模角度看，Outcome 对应任务执行过程中的终止状态（Terminal State）。

其作用不仅是向用户提供最终结果，同时也是评估整个轨迹质量的重要依据。因此 Outcome 可以承担多种训练与评估功能：

```text
Final Answer Supervision
Trajectory Quality Evaluation
Outcome Verification
Reward Modeling
Preference Construction
```

对于生产级 Agent 数据集而言，仅保存执行过程而缺少 Outcome 将导致轨迹难以评价；而仅保存 Outcome 而缺少执行过程，则无法学习 Agent 行为。因此两者必须同时保留。

---

## 13. Agent Multi Turn 格式的整体结构

综合前述分析，可以将该类 Agent 数据抽象为一个分层执行模型：

```text
Task Metadata
    ↓
Environment Context
    ↓
Tool Schema
    ↓
Execution Segments
    ↓
Agent Roles
    ↓
Message Trace
    ↓
Action–Observation Pairs
    ↓
Subagent Trajectories
    ↓
Reasoning Signals
    ↓
Task Outcome
```

进一步从 Agent 运行机制角度抽象，则可以表示为：

```text
Task
→ Environment
→ Agent State
→ Action
→ Observation
→ Evidence Accumulation
→ Subagent Collaboration
→ Decision
→ Result
```

这一结构已经明显超出了传统聊天数据的范畴。

传统对话数据通常描述：

```text
User
→ Assistant
```

而 Agent Trace 数据描述的是：

```text
Environment
→ State
→ Action
→ Observation
→ State Transition
→ Outcome
```

因此，两者在训练目标上存在本质差异。

从数据工程角度看，样本中的部分元数据字段主要服务于数据治理、索引、检索和审计，其训练价值相对有限。例如：

```text
dataset_id
data_id
timestamp
source
storage metadata
```

这些字段对于数据管理十分重要，但不直接参与 Agent 行为学习。

真正承载 Agent 能力的信息结构主要包括：

```text
Tool Schema
Execution Segments
Messages
Tool Calls
Tool Results
Subagent Traces
Reasoning Signals
Outcome Records
```

这些结构共同描述了 Agent 在任务执行过程中状态、动作与观察之间的动态演化关系。

因此，在后续训练数据构建过程中，更合理的做法并非直接使用原始 JSON 结构，而是将上述核心信息编译为统一的 Trajectory Representation，并进一步转换为：

```text
State–Action Samples
Action–Observation Samples
Subagent Samples
Reasoning Samples
Outcome Samples
Preference Samples
Reward Samples
RL Transition Samples
```

等不同训练任务所需的数据形式。

总体而言，该格式已经具备较完整的 Agent Execution Trace 特征。其不仅保留了工具调用与工具结果之间的对应关系，还显式记录了主代理与子代理的协作过程，并保留了环境上下文、决策信号与任务结果等关键要素。因此，该类数据不仅适用于传统 SFT 数据构建，也适用于长程 Agent 训练、过程监督、层级规划学习、多代理协同训练、奖励建模和强化学习等更复杂的研究方向。

---

# 第三部分：生产级 Agent Trace Training Data 合成方法

## 1. 问题定义：从轨迹记录到训练数据

生产级 Agent Trace Training Data 的目标不是保存一段聊天记录，而是将真实 Agent 执行过程转化为可学习的行为监督信号。其基本建模单元不是传统对话数据中的：

```text
User → Assistant
```

而是 Agent 任务执行过程中的：

```text
State → Action → Observation → Updated State → Outcome
```

其中：

* `State` 表示当前任务状态，包括用户目标、环境信息、工具集合、历史上下文、已获得证据和当前执行阶段；
* `Action` 表示 Agent 在当前状态下做出的决策，可以是普通工具调用、子代理委派、自然语言回答、计划更新或任务终止；
* `Observation` 表示工具执行结果、子代理返回结果、测试输出、文件内容、搜索结果或运行时反馈；
* `Updated State` 表示 observation 被纳入上下文后的新状态；
* `Outcome` 表示任务阶段性或最终结果，可用于质量评估、偏好构造和奖励建模。

因此，Agent Trace Training Data 的核心目标是训练模型学习：

```text
在什么状态下，应该执行什么动作；
执行动作后，如何解释观察结果；
如何将观察结果转化为后续决策；
如何在长程任务中维持目标、组织上下文并收敛到结果。
```

这一定义决定了，原始轨迹数据不能直接作为最终训练数据，而需要经过系统化的数据编译过程。

---

## 2. 总体数据编译框架

生产级数据构建通常会经历多个阶段，具体划分方式会因数据来源、系统架构和训练目标而有所不同。一种较为常见的流程包括：

```text
Raw Trace Collection
→ Parsing
→ Normalization
→ Action–Observation Alignment
→ Trajectory IR (Intermediate Representation，中间表示) Construction
→ Context Reconstruction
→ Data Annotation / Synthesis
→ Training Data Export
→ Evaluation / Feedback Loop
```

各阶段职责如下：

| 阶段                           | 目标                                                         |
| ---------------------------- | ---------------------------------------------------------- |
| Raw Trace Collection         | 收集原始轨迹、工具日志、子代理轨迹、文件变更、环境信息                                |
| Parsing                      | 解析 message、segment、tool_call、tool_result、reasoning_content |
| Normalization                | 统一不同来源数据格式和字段命名                                            |
| Action–Observation Alignment | 将工具调用与工具返回对齐为状态转移单元                                        |
| Trajectory IR Construction   | 构建统一中间表示，支持后续多任务导出                                         |
| Context Reconstruction       | 重建每一步模型可见上下文或近似上下文                                         |
| Data Annotation / Synthesis  | 生成 reasoning、quality、phase、reward、preference 等标签           |
| Training Data Export         | 导出 SFT、DPO、PRM、RL、Eval 等训练格式                               |

这一流程的关键在于：原始数据只记录发生了什么，而训练数据必须表达为什么某个动作在某个状态下是合理的、该动作产生了什么观察结果，以及最终是否带来了更好的任务结果。

---

## 3. 统一 Trajectory IR 设计

在生产级 Agent 数据构建中，Trajectory IR（Intermediate Representation）的核心目标并不是重新发明一种新的数据格式，而是在原始 Agent Multi Turn 数据与最终训练数据之间建立一个统一的数据编译层（data compilation layer）。

对于本文讨论的场景，Agent Multi Turn 本身已经包含了大量关键信息，例如：

```text
任务信息
工具定义（Tool Schema）
消息轨迹（Messages）
工具调用（Tool Calls）
工具返回（Tool Results）
Segment 结构
Agent Role
Subagent Trajectory
Reasoning Signal
Outcome
环境上下文
```

因此，Trajectory IR 的主要作用并不是补充大量额外上游数据，而是将 Agent Multi Turn 中已经存在的信息重新组织为统一的状态—动作—观察表示，使后续能够方便地导出 SFT、Preference、PRM、Verifier、RL 等不同训练格式。

换句话说，对于仅拥有 Agent Multi Turn 数据的情况，绝大部分构建工作实际上属于结构化重编译（re-compilation）而非数据补全（data augmentation）。很多后续训练所需的信息，例如工具调用链、Action–Observation 对齐关系、主代理与子代理轨迹、任务结果以及部分决策信号，本身已经能够直接从 Agent Multi Turn 数据中获得，只是需要转换为更适合训练和评估的统一表示。

例如：

```text
原始 transcript 记录消息流；

tool log 记录工具执行；

OTel trace 记录运行时事件；

subagent log 记录子代理执行过程；

git diff 记录环境变化结果。
```

如果直接基于这些原始数据构建训练集，那么每一种训练任务都需要重新解析不同来源的数据，工程复杂度极高，而且难以保证不同训练任务之间的数据一致性。

因此需要引入统一 Trajectory IR：

```text
Raw Logs → Trajectory IR → SFT / Preference / PRM / Verifier / Reward Model / Offline RL / Evaluation
```

Trajectory IR 的定位类似编译器中的 IR（Intermediate Representation）：

```text
原始日志 ≠ 训练数据
原始日志 → Trajectory IR → 多种训练数据
```

其价值在于：

```text
统一不同来源数据；

统一状态、动作、观察表示；

统一主代理与子代理结构；

统一环境状态表示；

统一结果与奖励表示；

支持后续任意训练任务导出。
```

从 Agent 学习角度看，真正重要的并不是消息本身，而是：

```text
Agent 当时处于什么状态；

Agent 看到了什么信息；

Agent 做出了什么动作；

动作产生了什么观察；

观察如何改变后续决策；

最终是否完成任务。
```

因此 Trajectory IR 的核心设计围绕：

```text
State
Action
Observation
Outcome
```

四类对象展开。

推荐结构如下。

需要强调的是，这里的 IR（Trajectory Intermediate Representation）并不要求与原始 `agent_multi_turn` 数据保持严格的一一映射关系。其主要目标是在统一抽象层上表达不同 Agent 系统的执行过程，从而为后续的数据编译、标注、训练和评估提供一致的数据表示。

从目前获得的 `agent_multi_turn` 数据来看，子代理（subagent）在原始格式中通常以一种特殊工具（Agent Tool）的形式出现。例如：

```json
{
  "name": "delegate_subtask",
  "arguments": {
    "agent_type": "Explore",
    "query": "..."
  }
}
```

对于主代理而言，这类调用与 `read_file`、`list_dir` 等普通工具在接口层面并无本质差异，均表现为一次标准的 tool call。因此，如果目标是尽可能忠实地保留原始 `agent_multi_turn` 数据结构，则可以将 subagent 建模为一种特殊工具：

```text
Tool
├── Environment Tool
│   ├── read_file
│   ├── list_dir
│   ├── grep_content
│   └── run_command
│
└── Agent Tool
    └── delegate_subtask
         ├── Explore Agent
         ├── Verification Agent
         └── Research Agent
```

在轨迹层面，其表现形式可抽象为：

```text
assistant
→ tool_call(delegate_subtask)

tool
→ subagent_summary
```

这种表示方式与当前可观测数据最为一致，也最接近运行时实际记录的事件结构。

然而，从训练数据工程的角度来看，仅保留工具调用层面的表示往往会丢失大量有价值的信息。原因在于，在许多场景下，我们不仅能够观察到：

```text
Main Agent
→ delegate_subtask
→ summary
```

还能够进一步获得：

```text
Subagent
→ read_file
→ grep_content
→ run_command
→ summary
```

等完整的子代理执行轨迹。

因此，在 IR 设计中，更合理的做法是采用“两层表示（dual-level representation）”策略。

第一层保留与原始数据一致的工具调用视角：

```json
{
  "action_type": "tool_use",
  "tool_name": "delegate_subtask",
  "arguments": {
    "agent_type": "Explore"
  }
}
```

这一层主要用于保持与 `agent_multi_turn` 原始格式的直接对应关系，确保轨迹能够被无损映射回原始事件序列。

第二层则显式抽取 Agent 层级结构与执行关系：

```json
{
  "agent_id": "explore_agent_1",
  "agent_role": "explore",
  "parent_agent_id": "main_agent"
}
```

用于表达：

```text
Main Agent
  └── Explore Agent
```

这样的层次化执行结构，并进一步关联对应的子代理轨迹。

因此，更准确地说：

```text
agent_multi_turn 原始数据视角：
  subagent ≈ 一种特殊工具

训练数据 IR 视角：
  subagent = Agent Tool
           + 独立 Agent Trajectory
```

这两种表示方式并不存在冲突，而是分别对应不同抽象层级下的建模需求：前者强调对原始运行时事件的忠实记录，后者强调对 Agent 行为结构和训练信号的充分表达。

基于上述原则，可以将 IR 调整为：

```json
{
  "trace_id": "...",
  "source": "claude_code | coding_agent_runtime | other",
  "format": "agent_multi_turn",

  "task": {
    "user_intent": "...",
    "task_type": "...",
    "language": "..."
  },

  "environment": {
    "repo": "...",
    "branch": "...",
    "commit_id": "...",
    "runtime_context": {}
  },

  "tools": [
    {
      "name": "read_file",
      "tool_type": "execution"
    },
    {
      "name": "delegate_subtask",
      "tool_type": "agent"
    }
  ],

  "agents": [
    {
      "agent_id": "main_agent",
      "agent_role": "main",
      "parent_agent_id": null
    },
    {
      "agent_id": "explore_agent_1",
      "agent_role": "explore",
      "parent_agent_id": "main_agent"
    }
  ],

  "steps": [
    {
      "agent_id": "main_agent",

      "action": {
        "action_type": "tool_use",
        "tool_name": "delegate_subtask",
        "arguments": {
          "agent_type": "Explore"
        }
      },

      "observation": {
        "observation_type": "subagent_summary",
        "content": "..."
      }
    }
  ]
}
```

这样既能够与现有 `agent_multi_turn` 数据结构对齐，又能够在后续训练阶段显式恢复主代理与子代理之间的层级关系。

需要强调的是，这里的字段设计并不是为了完整保存原始日志，而是为了表达 Agent 学习所需的关键结构。对于已经能够观察到完整 subagent segment 的数据，建议同时保留“工具调用视角”和“独立 Agent 轨迹视角”；对于只能看到 `delegate_subtask → summary` 的数据，则仍然可以退化为普通 Tool Calling 轨迹，不影响后续 SFT、Preference Learning、PRM 或 RL 数据构建。

例如：

```text
task:
  描述任务目标

environment:
  描述执行环境

tools:
  描述动作空间

agents:
  描述层级 Agent 结构

state:
  描述当前决策状态

action:
  描述当前决策

observation:
  描述环境反馈

reasoning:
  描述决策解释信号

outcome:
  描述任务结果
```

因此：

```text
Trajectory IR ≠ 原始日志
Trajectory IR ≠ 最终训练样本
Trajectory IR = Agent 执行过程的统一语义表示
```

它既保留了足够的信息用于后续训练数据生成，又避免训练阶段反复解析复杂原始日志。

该 IR 的核心设计原则包括：

```text
保留 Agent 层级结构
显式表达状态、动作、观察和结果
区分普通工具与 Agent 工具
区分模型可见上下文与 runtime metadata
支持长轨迹切分
支持上下文重建
支持后验标注与奖励建模
支持多种训练范式导出
```

---

## 4. Context Reconstruction：从 Trajectory IR 到训练样本

Trajectory IR 建立之后，下一步并不是直接训练，而是为每个决策点重建对应的训练上下文（Context Reconstruction）。

对于大多数 Agent 训练任务，核心问题都是：

```text
Agent 在执行某个动作之前，
实际掌握了哪些信息？
```

从训练角度看，一个决策样本通常表示为：

```text
State_t → Action_t
```

对于基于 LLM 的 Agent，`State_t` 并非不可观测的内部状态，而主要由模型在时刻 `t` 可访问的上下文构成，因此可以近似表示为：

```text
State_t ≈ Context_t
```

其中：

```text
Context_t = 用户任务 + system prompt + tool schema + 环境信息 + 历史 action-observation + subagent summary + hook 注入内容 + 当前阶段摘要 + 其他模型可见信息
```

因此，Context Reconstruction 的目标是在可观测轨迹基础上，尽可能重建模型做出 `Action_t` 时的决策状态：

```text
Trajectory
→ 重建 Context_t
→ 作为训练输入
→ 预测 Action_t
```

对于一次工具调用，其上下文通常包括：

```text
用户原始任务
system prompt 或 agent role instruction
工具 schema
repo / cwd / branch / commit 等环境信息
最近若干轮 action-observation history
当前任务阶段摘要
可见 hook additional context
subagent 返回 summary
compaction summary
重要文件内容或文件摘要
```

需要注意的是：

```text
完整 transcript ≠ 模型实际看到的上下文
```

在 Claude Code 类系统中，上下文通常会经过裁剪、压缩、摘要生成、缓存复用以及子代理隔离等处理。因此，Context Reconstruction 的目标并非恢复完整历史，而是恢复当前决策真正依赖的信息。

如果能够获得完整 API Request Body，则应将其视为最接近真实模型输入的上下文；否则只能利用 transcript、tool result、subagent summary、hook context、runtime metadata 和 compaction summary 等可观测信息进行近似重建。

从数据流水线角度看：

```text
Raw Logs
    ↓
Trajectory IR
    ↓
Context Reconstruction
    ↓
Training Samples
```

Context Reconstruction 因而成为连接轨迹表示与训练数据之间的关键环节。基于重建后的上下文，可以进一步导出多种训练任务：

```text
State → Action
Observation → Action
State → Tool
State → Tool Arguments
State → Summary
State → Reasoning
Evidence → Final Answer
```

因此，Trajectory IR、Context Reconstruction 与 Training Dataset 分别承担不同职责：

```text
Trajectory IR
负责统一表达 Agent 执行过程；

Context Reconstruction
负责恢复决策时的可见状态；

Training Dataset
负责将重建后的上下文转换为具体监督任务。
```

三者共同构成从原始 Agent 轨迹到生产级训练数据的核心编译链路。

---

## 5. Trajectory Splitting：长轨迹切分

Claude Code 类任务往往包含大量工具调用、文件阅读、测试运行和子代理交互，完整轨迹可能远超模型训练上下文窗口。因此需要将长轨迹切分为可训练窗口。

推荐采用：

```text
global prefix + local window + target
```

的切分方式。

### 5.1 Global Prefix

Global Prefix 保留任务级和环境级稳定信息：

```text
用户原始任务；
agent role；
repo / cwd / branch / commit；
工具 schema；
项目规则或 CLAUDE.md 摘要；
当前任务高层目标；
必要的全局状态摘要。
```

### 5.2 Local Window

Local Window 保留最近若干步状态转移：

```text
action_1 → observation_1
action_2 → observation_2
...
action_k → observation_k
```

窗口大小可以基于 token budget、step 数量或任务阶段自适应调整。

### 5.3 Target

Target 根据训练任务不同，可以是：

```text
下一步工具调用；
下一步自然语言回复；
下一步子代理委派；
下一步状态摘要；
最终回答；
过程奖励标签；
偏好选择。
```

这种切分方式可避免直接训练超长完整轨迹，同时保留全局任务目标和局部连续性。

---

## 6. Agent Training Data 分类体系

从训练流程角度看，Agent Trace Data 不应按照具体训练方法进行组织，而应首先按照训练阶段进行划分。

对于 Claude Code 类 Agent，基于可观测轨迹构建的数据大致可以分为三大类：

```text
Pretraining-Style Agent Data（NTP）
Post-training Supervised Data（SFT）
Reinforcement Learning Data（RL）
```

三类数据对应不同训练目标：

| 数据类别           | 目标            | 学习内容             |
| -------------- | ------------- | ---------------- |
| Agent NTP Data | 学习 Agent 轨迹分布 | 状态转移、工具调用模式、轨迹结构 |
| Agent SFT Data | 学习专家行为        | 决策策略、工具使用、任务分解   |
| Agent RL Data  | 优化长期回报        | 长程规划、探索策略、任务成功率  |

因此，生产级 Agent Data Pipeline 更合理的组织方式应为：

```text
Raw Trace
→ Trajectory IR
→ Agent NTP Dataset

Raw Trace
→ Trajectory IR
→ Agent SFT Dataset

Raw Trace
→ Reward Annotation
→ Agent RL Dataset
```

下面分别讨论三类数据。

---

### 6.1 Agent NTP Data（用于持续预训练）

Agent NTP（Next Token Prediction）数据对应模型训练体系中的持续预训练阶段。

其目标不是学习最优决策，而是学习 Agent 世界中的语言分布、工具调用格式、轨迹结构和状态转移模式。

形式上仍然是：

```text
Token_{1:n-1}
→ Token_n
```

但训练语料不再是普通网页文本，而是 Agent Execution Trace。

例如：

```text
User
→ Assistant
→ Tool Call
→ Tool Result
→ Assistant
→ Tool Call
→ ...
```

或者：

```text
State
→ Action
→ Observation
→ State
→ Action
```

这样的轨迹序列。

Agent NTP 数据的主要价值包括：

```text
学习工具调用语法；
学习 JSON 参数结构；
学习 Agent 对话格式；
学习工具结果分布；
学习代码仓库探索模式；
学习长轨迹结构；
学习多 Agent 协作模式。
```

从数据构建角度看，NTP 数据通常不需要额外标注。

只需要：

```text
轨迹完整；
格式统一；
工具调用合法；
上下文可恢复。
```

即可直接用于训练。

因此：

```text
Agent NTP
≈ Agent Corpus
≈ Agent Language Modeling Data
```

其作用类似于代码模型中的代码预训练语料。

---

### 6.2 Agent SFT Data（用于行为学习）

SFT 阶段的目标是学习专家行为。

与 NTP 不同，SFT 不再学习整个轨迹分布，而是学习：

```text
给定状态→应该采取什么动作
```

因此 SFT 数据本质上属于：

```text
State → Expert Action
```

监督数据。

根据训练目标不同，可以进一步划分为若干子类。

---

#### 6.2.1 Action Prediction SFT

形式：

```text
State_t → Action_t
```

训练内容：

```text
工具调用；
自然语言回复；
任务终止；
子代理委派。
```

这是最基础的 Agent Behavior Cloning 数据。

---

#### 6.2.2 Tool Use SFT

形式：

```text
State_t → (Tool, Arguments)
```

即在给定当前状态的情况下，同时生成工具选择结果与对应参数，而不是将两者完全割裂。

对于大多数 Agent 场景，没有必要强行拆成两个独立训练任务。因为工具选择与参数生成本身高度耦合，模型通常需要在理解当前状态后一次性决定：

```text
使用什么工具；
工具应该携带什么参数。
```

因此更符合真实执行过程的训练目标通常是：

```text
State_t → Tool Call
```

其中 Tool Call 同时包含：

```text
Tool Name
+
Tool Arguments
```

训练内容：

```text
选择正确工具；
生成与当前状态匹配的正确参数；
学习工具与参数之间的联合分布；
避免工具正确但参数错误；
避免无效或不可执行调用。
```

---

#### 6.2.3 Observation Response SFT

Observation Response SFT 与前述 State→Action SFT 容易混淆，但二者对应的学习目标存在明显差异。

State→Action 关注基于完整任务状态的全局决策，其输入通常包括：

```text
用户任务；
历史工具调用；
历史观察结果；
当前任务阶段；
环境信息；
工具集合；
上下文摘要。
```

模型需要综合上述信息决定下一步行动，因此更接近 Agent 的整体策略学习（policy learning）：

```text
State_t → Action_t
```

相比之下，Observation→Action 更关注最新观察结果所触发的局部决策过程：

```text
Observation_t → Action_{t+1}
```

其中，Observation 通常来源于工具执行后的反馈，例如：

```text
文件内容；
grep 结果；
测试输出；
报错日志；
子代理返回结果；
命令执行结果。
```

其训练目标在于学习如何解释新获得的证据，并据此选择合理的后续动作，包括：

```text
根据观察结果选择下一步操作；
理解工具返回信息的语义；
利用新增证据推进任务执行。
```

例如：

```text
Observation:
pytest 输出：
test_login FAILED
AssertionError: expected 200 got 401

Action:
read_file auth/login.py
```

或：

```text
Observation:
grep 找到 createRouter 定义位于 src/router/index.ts

Action:
read_file src/router/index.ts
```

因此，两类任务可以概括为：

```text
State → Action
= 基于完整任务状态的全局决策

Observation → Action
= 基于最新观察结果的局部响应
```

Observation Response SFT 主要训练模型：

```text
解释工具执行结果；
理解测试与诊断输出；
分析错误日志；
基于观察结果推进任务。
```

该能力构成 Agent 局部推理与连续决策的重要基础。

---

#### 6.2.4 Subagent SFT

形式：

```text
Complex State → Delegate Action
```

以及：

```text
Subtask → Subagent Trajectory
```

训练内容：

```text
任务分解；
Agent Orchestration；
Explore Agent；
Verification Agent；
Research Agent。
```

对应 Claude Code 中的多 Agent 协作能力。

---

#### 6.2.5 State Management SFT

形式：

```text
Long History → State Summary
```

训练内容：

```text
上下文压缩；
任务状态维护；
长期记忆构建；
阶段总结。
```

对应 Claude Code 的 compaction 与 summary 能力。

---

#### 6.2.6 Outcome Generation SFT

形式：

```text
Evidence → Final Answer
```

训练内容：

```text
架构分析；
Bug 修复报告；
测试报告；
设计文档；
最终回答。
```

要求输出与轨迹证据保持一致。

---

#### 6.2.7 Synthetic Reasoning 数据合成与 SFT

与前面的 Next-action、Tool-selection、Tool-argument 等监督任务不同，Synthetic Reasoning 所需的监督信号通常无法直接从原始 Agent Trace 中获得。

对于 Claude Code 类轨迹，我们能够直接观察到的主要是：

```text
State
Action
Observation
Outcome
```

以及少量可观测的 reasoning signal（如 reasoning_content、decision rationale 片段等）。

但这些信息通常不足以形成完整的 reasoning trace。因此，在构建 Synthetic Reasoning SFT 之前，需要先进行专门的数据合成阶段。

推荐流程如下：

```text
原始 Agent Trace
→ Context Reconstruction
→ State / Action / Observation 对齐
→ 调用强模型 API（Teacher Model）
→ 基于轨迹证据生成 reasoning trace
→ 一致性检查与质量过滤
→ Synthetic Reasoning Dataset
→ Synthetic Reasoning SFT
```

其中，Teacher Model 的输入通常包括：

```text
当前状态；
历史关键轨迹；
工具调用；
工具返回结果；
子代理总结；
任务阶段；
最终结果（Outcome）。
```

Teacher Model 输出：

```text
状态理解；
关键证据；
决策依据；
动作意图；
备选方案分析；
风险与不确定性；
下一步推进逻辑。
```

形式：

```text
State + Action + Observation → Reasoning Trace（训练目标）
```

其中 Outcome 不作为推理时输入，而主要用于后验标注阶段，为 reasoning 合成提供结果约束、质量判断和监督信号。

训练内容：

```text
状态理解；
证据整合；
任务分解；
中间目标规划；
动作选择逻辑；
错误分析与修正策略；
长期任务推进思路。
```

需要强调：

```text
Synthetic Reasoning ≠ Hidden Reasoning

Synthetic Reasoning ≠ 仅用于解释动作

Synthetic Reasoning = 基于真实轨迹证据构造的高质量推理监督信号
```

其核心目标并不是提升模型的可解释性，而是利用比最终动作标签更丰富的信息密度，对模型进行过程层面的监督。通过结合状态、动作、观察结果以及最终任务结果，可以生成质量更高、任务导向更强的 reasoning trace，使模型学习如何分析问题、组织证据、规划步骤、修正错误并推进长期任务执行。从训练角度看，Synthetic Reasoning SFT 本质上是一种 reasoning augmentation。它并不试图恢复真实的 hidden reasoning，而是利用高质量推理轨迹作为额外监督信号，帮助模型学习更强的任务求解策略、更稳定的工具使用行为以及更有效的长程决策能力，从而提升最终任务成功率，而不仅仅是生成看起来更合理的解释。

---

### 6.3 Agent RL Data（用于强化学习）

第三类数据用于强化学习。

与 SFT 最大区别在于：

```text
SFT 学习动作

RL 学习回报
```

因此 RL 数据最终需要能够回答一个问题：

```text
哪些行为更值得被重复？
```

对于 Coding Agent 来说，仅仅模仿历史动作并不足够。模型还需要学会区分：

```text
哪些动作真正推进了任务；
哪些探索是低效的；
哪些修复最终能够通过测试；
哪些轨迹虽然过程不同，但结果更好。
```

因此强化学习阶段通常建立在已经完成 SFT 的基础上。

---

#### 6.3.1 Preference / PRM 数据

这一阶段的目标不是直接训练 RL Policy，而是先构建质量信号（Quality Signal）。

因为真实 Coding Agent 的最终奖励通常非常稀疏：

```text
任务完成；
测试通过；
用户满意；
Benchmark 得分提升。
```

如果只依赖最终结果，模型很难知道：

```text
中间哪些步骤是好的；
哪些步骤是坏的；
哪些行为值得鼓励。
```

因此通常会先利用轨迹构建偏好数据和过程质量数据。

##### Preference Data

形式：

```text
State
→ Chosen Action

State
→ Rejected Action
```

典型训练方法：

```text
DPO
IPO
KTO
```

来源包括：

```text
成功轨迹 vs 失败轨迹；
多次 rollout；
LLM Judge 排序；
人工标注。
```

其作用是让模型学会：

```text
更偏好高质量动作；
减少低效探索；
减少无意义工具调用；
提高任务推进效率。
```

##### Process Quality / PRM Data

形式：

```text
(State, Action, Observation)
→ Step Quality Score
```

训练内容：

```text
步骤质量；
信息增益；
风险水平；
任务推进程度。
```

例如：

```text
读取关键配置文件
→ 高质量步骤

重复读取相同文件
→ 低质量步骤
```

这类数据本质上是在学习：

```text
当前步骤是否有价值。
```

除了用于训练 PRM，本身也可以用于：

```text
轨迹过滤；
候选动作排序；
搜索过程引导；
自动评估。
```

---

#### 6.3.2 Offline RL / Online RL 数据

当已经具备行为模型和质量信号之后，才进入真正的 RL 阶段。

此时需要将轨迹组织成：

```text
(s_t, a_t, r_t, s_{t+1})
```

或者：

```text
Trajectory + Reward
```

其中 reward 可以来自：

```text
测试是否通过；
Benchmark 得分；
Judge 评分；
PRM 分数；
任务完成情况。
```

##### Offline RL Data

形式：

```text
(s_t, a_t, r_t, s_{t+1})
```

来源：

```text
历史 Agent Trace；
自动评测结果；
测试结果；
Judge Score；
PRM Score。
```

用于：

```text
Batch RL；
Offline RL；
Conservative RL。
```

特点是：

```text
不需要在线探索；
训练成本较低；
依赖已有轨迹质量。
```

##### Online RL Data

形式：

```text
Environment Rollout → Reward
```

环境包括：

```text
代码仓库；
测试框架；
Sandbox；
Benchmark。
```

训练过程：

```text
执行任务
→ 调用工具
→ 修改代码
→ 运行测试
→ 获得奖励
→ 更新策略
```

训练目标：

```text
提高长期任务成功率；
提高测试通过率；
提高代码质量；
优化 Agent Policy。
```

因此从工程实践角度看，更常见的路线通常是：

```text
SFT
↓
Preference / PRM
↓
Offline RL / Online RL
```

其中 Preference 和 PRM 负责提供质量信号，而 RL 负责利用这些质量信号进一步优化长期任务表现。

---

## 7. 三类数据之间的关系

从训练体系角度看，三类数据并非并列关系，而是逐层递进关系。

```text
Agent Trace
        │
        ▼
Agent NTP Data
        │
        ▼
Agent SFT Data
        │
        ▼
Agent RL Data
```

对应训练流程：

```text
阶段一：
Agent NTP
学习 Agent 世界模型

↓

阶段二：
Agent SFT
学习专家行为

↓

阶段三：
Preference / PRM / Reward Model
学习质量信号

↓

阶段四：
Offline RL / Online RL
优化长期回报
```

因此，对于 Claude Code 可观测轨迹而言，最合理的数据工程目标并不是直接构造某一种训练集，而是构建统一 Trajectory IR，然后从同一份轨迹中导出：

```text
Agent NTP Dataset
Agent SFT Dataset
Agent RL Dataset
```

三套数据资产。

这种划分方式更符合当前 Agent Foundation Model 的训练范式，也更符合生产级 Agent 数据平台的组织方式。

---

## 8. 结论：统一轨迹，多阶段训练

从训练视角看，Claude Code 可观测轨迹的价值并不局限于 SFT。

同一份 Agent Trace 可以同时支持：

```text
Agent NTP
Agent SFT
Preference Learning
PRM
Verifier
Reward Model
Offline RL
Online RL
```

因此，更合理的研究问题不是：

```text
如何把轨迹变成 SFT 数据？
```

而是：

```text
如何把轨迹编译成统一 Trajectory IR，
并从中导出 NTP、SFT 与 RL 三类训练数据？
```

对于生产级 Coding Agent 而言，这种统一数据编译框架能够最大化利用真实运行轨迹中的状态、动作、观察和结果信息，并形成覆盖预训练、后训练和强化学习阶段的完整数据闭环。
