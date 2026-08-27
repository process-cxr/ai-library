---
title: "Dive into Claude Code: The Design Space of Today's and Future AI Agent Systems"
created: 2026-06-02
published: 2026-06-02
modified: 2026-06-02
type: source
status: processed
source_type: paper
area: sources
tags:
  - source
  - paper
  - agents
  - coding-agent
  - software-architecture
  - context-management
  - tool-use
  - safety
aliases:
  - Dive into Claude Code
  - Claude Code Design Space
source_url: https://arxiv.org/pdf/2604.14228
paper_date: "2026-04"
paper_order: "14228"
---

## 基本信息

- Title: [Dive into Claude Code: The Design Space of Today's and Future AI Agent Systems](https://arxiv.org/pdf/2604.14228)
- Authors: Jiacheng Liu, Xiaohan Zhao, Xinyi Shang, Zhiqiang Shen
- Institutions: VILA Lab, Mohamed bin Zayed University of Artificial Intelligence; University College London
- Date: arXiv v1, 2026-04-14
- Code / project: `https://github.com/VILA-Lab/Dive-into-Claude-Code`
- Related topic notes: [[application/agents/agent|Agent]], [[application/agents/workflow-agent|Workflow Agent]], [[application/agents/planning|Planning]], [[application/agents/memory|Memory]], [[application/agents/multi-agent|Multi-Agent]], [[application/tool-use/tool-calling|Tool Calling]], [[application/tool-use/mcp|MCP]], [[application/rag/context-compression|Context Compression]], [[application/evaluation/benchmark|Benchmark]], [[application/evaluation/llm-as-judge|LLM-as-a-Judge]]

这篇文章不是传统意义上的算法论文，而是一篇面向生产级 coding agent 的 source-grounded architecture survey。作者通过分析公开可获得的 Claude Code TypeScript 源码，并与 OpenClaw 这个独立开源 agent gateway 做对照，试图回答一个更系统的问题：当 coding assistant 从 autocomplete、chat-in-IDE 走向可以运行 shell、编辑文件、调用外部服务、委托子代理并持久化会话的 agentic system 时，架构设计空间到底由哪些问题构成？

论文的核心观察是：Claude Code 的 agent loop 本身非常简单，本质上是一个反复调用模型、解析工具请求、执行工具并把结果喂回上下文的 while-loop。真正复杂的部分在 loop 外围：权限系统、上下文管理、扩展机制、subagent orchestration、session persistence、shell sandbox 和各类恢复机制。作者因此把 Claude Code 解读为一种“model judgment within a deterministic harness”的架构：模型负责局部判断，harness 负责执行边界、上下文预算、安全控制和可恢复性。

## 研究问题

论文关心的不是 Claude Code 在某个 benchmark 上能拿多少分，而是生产级 coding agent 必须反复回答的一组设计问题：

- reasoning 应该放在哪里：由模型自由决策，还是由显式 planner / state graph / workflow engine 约束？
- agent loop 应该是统一执行引擎，还是不同入口使用不同路径？
- 安全默认姿态应该是 allow、deny、ask，还是依赖 sandbox / rollback？
- 扩展面应该是单一 tool API，还是 MCP、plugins、skills、hooks 等多机制分层？
- context window 是如何成为核心稀缺资源的，系统应如何分阶段压缩、截断、懒加载和持久化？
- subagent 应该共享父 agent 的上下文和权限，还是使用隔离上下文与单独 transcript？
- session 应该如何 resume、fork、rewind，又应该避免哪些 trust state 被隐式恢复？

这组问题使文章更接近软件架构案例研究，而不是一个“Claude Code 使用指南”。它的贡献在于把一个具体产品拆解成可比较的 design space，并把每个 subsystem 的实现选择放回到可替代方案中理解。

## 核心框架

### 五个价值与十三个设计原则

论文先抽象出 Claude Code 架构背后的五个 human values / philosophies：

| Value | 含义 |
|---|---|
| Human Decision Authority | 人类保留最终决策权，可以观察、批准、拒绝、中断和审计 agent 行为 |
| Safety, Security, and Privacy | 即使用户疏忽，系统也要保护代码、数据、基础设施和隐私 |
| Reliable Execution | agent 需要正确理解意图，并在长程任务、resume、delegation 中保持一致性 |
| Capability Amplification | 系统应显著放大用户单位时间内可完成的工作，而不只是提供建议 |
| Contextual Adaptability | 系统应适配用户项目、工具、约定和技能水平，并随着关系演化改善 |

这些价值进一步映射为十三个设计原则：

| Design Principle | 主要回答的问题 |
|---|---|
| Deny-first with human escalation | 未识别 action 应该允许、阻断，还是升级给人类？ |
| Graduated trust spectrum | 权限是固定等级，还是用户随信任增长逐步放开？ |
| Defense in depth with layered mechanisms | 单一安全边界是否足够，还是需要多层机制重叠？ |
| Externalized programmable policy | 安全策略应硬编码，还是外部配置并允许 hooks 介入？ |
| Context as scarce resource with progressive management | context 是绑定资源时，是否需要多阶段管理？ |
| Append-only durable state | session state 应可变、快照化，还是 append-only？ |
| Minimal scaffolding, maximal operational harness | 复杂性应放在 planner 里，还是放在可执行 harness 里？ |
| Values over rules | rigid rules 是否足够，还是要依赖模型的 contextual judgment？ |
| Composable multi-mechanism extensibility | 扩展面应统一，还是按上下文成本和能力类型分层？ |
| Reversibility-weighted risk assessment | read-only / reversible action 是否应比 destructive action 低风险？ |
| Transparent file-based configuration and memory | memory 应是用户可见文件，还是 opaque database / embedding store？ |
| Isolated subagent boundaries | subagent 是否继承父上下文和权限？ |
| Graceful recovery and resilience | 出错时应硬失败，还是尽量自动恢复并保留用户注意力？ |

这套框架的价值在于，它不把 Claude Code 的实现当作偶然堆叠，而是把权限、context、工具、subagent、session 等设计都放入同一组价值张力中理解。

### 第六个分析视角：长期人类能力保存

论文还引入一个 cross-cutting evaluative lens：long-term human capability preservation。作者认为 Claude Code 的架构明显服务于短期能力放大，但对长期 human understanding、codebase coherence、developer pipeline 的保护机制有限。

这一点很重要，因为它改变了评估问题。一个 coding agent 不能只问“能否更快完成任务”，还要问：

- 用户是否仍理解代码库？
- agent 生成代码是否增加长期复杂度？
- 开发者是否保留监督 agent 的能力？
- AI-assisted workflow 是否削弱新人培养路径？

论文没有把它列为 Claude Code 显式设计价值，而是作为后文讨论和未来方向的评估镜头。

## Claude Code 架构分析

### 七组件高层结构

论文把 Claude Code 拆为七个功能组件：

| Component | 作用 |
|---|---|
| User | 提交 prompt、审批权限、审阅输出 |
| Interfaces | interactive CLI、headless CLI、Agent SDK、IDE/Desktop/Browser 等入口 |
| Agent loop | `queryLoop()` 实现的模型调用、工具分发、结果收集循环 |
| Permission system | deny-first rule evaluation、auto-mode classifier、hook interception |
| Tools | 内置工具、条件工具、MCP tools、plugins 间接贡献的工具 |
| State & persistence | append-oriented JSONL transcript、history、subagent sidechains |
| Execution environment | shell、filesystem、web fetching、MCP connections、remote execution、sandbox |

关键点是：所有入口最终汇入同一个 shared agent loop。interactive CLI、headless CLI、Agent SDK 和 IDE 集成不是各自维护独立 agent engine，而是共享核心 query path。这降低了行为分叉，也使权限、context、tools 和 recovery 逻辑具有一致性。

### 五层子系统分解

论文进一步把系统拆成五层：

| Layer | 内容 |
|---|---|
| Surface layer | CLI / headless CLI / Agent SDK / IDE / renderer |
| Core layer | agent loop 与 compaction pipeline |
| Safety / action layer | permission system、hooks、extensibility、built-in tools、MCP tools、sandbox、subagent spawning |
| State layer | context assembly、runtime state、session persistence、CLAUDE.md memory、sidechain transcripts |
| Backend layer | shell execution、filesystem、remote execution、MCP transports、external resources |

这个分层最能体现论文的主张：LLM 决策逻辑只占很小一部分，大量代码都在 operational harness 中。论文引用的估算认为，Claude Code 只有约 1.6% 代码属于 AI decision logic，其余约 98.4% 是操作基础设施。这个比例未必能作为精确工程事实，但作为架构信号很有解释力：生产级 agent 的可靠性主要来自外围系统，而不是单纯来自“更聪明的模型”。

### QueryEngine 不是核心 engine

论文特别澄清了一个容易误读的点：`QueryEngine` 类是 headless / SDK conversation wrapper，不是实际 shared execution engine。真正共享的核心路径在 `query()` / `queryLoop()` 中。interactive CLI 也调用 `query()`，并不经过 `QueryEngine`。

这类细节说明作者确实在做 source-level architecture reading，而不是根据产品文档做概念推断。

## Agent Loop 与执行机制

### Reactive while-loop

Claude Code 的核心 loop 接近 [[application/agents/agent|Agent]] 领域常见的 ReAct pattern：模型生成 reasoning / tool-use request，harness 执行工具，tool result 回到上下文，模型继续下一步。每一轮大致包括：

1. 解析 settings、system prompt、user context、permission callback、model config；
2. 初始化或更新 mutable state；
3. 从 compact boundary 之后取 conversation messages；
4. 执行 pre-model context shapers；
5. 调用 Claude model 并流式接收响应；
6. 解析 `tool_use` blocks；
7. 经过 permission gate；
8. 执行工具并收集 result；
9. 若没有 tool use，则 turn 结束。

作者强调，Claude Code 没有把 agent 控制流写成显式 state graph，也没有在 harness 侧实现复杂 planner。相比 LangGraph 这类 graph-based orchestration，Claude Code 更像是“简单循环 + 强 harness”。这种设计降低了控制流复杂度和延迟，但也意味着它不会系统性探索多条 action trajectory，缺少 tree search / backtracking 式的全局规划能力。

### Tool dispatch

工具执行路径支持 streaming execution。`StreamingToolExecutor` 可以在模型 response 仍在流式生成时开始执行已出现的工具请求，以降低多工具场景延迟。系统会区分 concurrent-safe tools 和 exclusive tools：read-only 操作可以并发，修改状态的 shell commands 则串行。

这个并发读、串行写的策略体现了 reversibility-weighted risk assessment：系统没有简单地把所有 tool calls 都串行化，而是按副作用风险区分执行方式。执行结果最终按工具请求顺序输出，避免模型收到乱序 tool result。

### Recovery

query loop 还实现了一组恢复机制：

- output token limit 触发时可提升 max output tokens 并重试；
- near-context-capacity 时可 reactive compact；
- prompt-too-long 时先尝试 context-collapse overflow recovery 和 reactive compaction；
- streaming API 出错时可 fallback；
- primary model 失败时可切换 fallback model；
- PostToolUse hook 可阻止继续执行。

这些机制说明生产级 agent loop 不是“模型调用失败就报错”，而是在很多边界条件下尝试保持会话继续。这也是 reliable execution 在系统层面的具体体现。

## 权限、安全与控制边界

### Deny-first permission model

Claude Code 的安全架构不是单纯依赖用户确认。论文指出，Anthropic auto-mode analysis 发现用户批准约 93% 的 permission prompts，因此 per-action approval 容易产生 approval fatigue，不能作为唯一安全机制。

Claude Code 的 permission system 因此采用 deny-first：

- blanket-denied tools 在工具池组装阶段就从模型可见工具中移除；
- deny rules 优先于 allow rules，即使 allow rule 更具体；
- 未匹配规则时默认 ask，而不是 silent allow；
- shell commands 即使通过 permission，也可能进入 sandbox；
- resume / fork 不恢复 session-scoped permissions。

论文列出七层 safety pipeline：

| Layer | 作用 |
|---|---|
| Tool pre-filtering | 在模型看到工具前移除 blanket-denied tools |
| Deny-first rule evaluation | deny 优先于 allow |
| Permission mode constraints | 当前模式决定默认处理方式 |
| Auto-mode classifier | ML classifier 评估工具调用安全性 |
| Shell sandboxing | 已批准 shell command 仍可被 sandbox 隔离 |
| Not restoring permissions on resume | 防止 stale trust state 跨会话传播 |
| Hook-based interception | PreToolUse / PermissionRequest hooks 可修改或阻断决策 |

### Permission modes

论文识别出七种 permission modes，其中外部可见模式包括 `plan`、`default`、`acceptEdits`、`dontAsk`、`bypassPermissions`；`auto` 由 feature flag 条件启用；`bubble` 是 subagent 内部 escalation 模式。

这些模式构成 autonomy spectrum：从 plan 模式下先计划再执行，到 bypassPermissions 下尽量减少 prompts。关键是，越高 autonomy 越不能把安全寄托在用户审批上，而要转向 classifier、sandbox、deny rules 和 hooks。

### Auto-mode classifier 与 hooks

Auto-mode classifier 会根据 conversation transcript 和 permission template 对 proposed tool invocation 做 allow / deny / manual approval 判断。它针对的风险包括 overeager behavior、honest mistakes、prompt injection 和 model misalignment。

Hooks 则提供 programmable policy。论文统计源码中有 27 种 hook events，其中直接参与权限流的包括：

- `PreToolUse`：可 deny / ask，或修改 tool input；
- `PostToolUse`：可注入 additional context，MCP tools 还可修改 tool output；
- `PostToolUseFailure`：可注入 error-specific guidance；
- `PermissionDenied`：可提供 retry guidance；
- `PermissionRequest`：可异步返回 allow / deny。

值得注意的是，hook allow 不能绕过后续 deny rules 或 safety checks。这个约束体现了 externalized programmable policy 与 deterministic safety invariants 的分离。

### 安全层的局限

论文对安全机制并不只唱赞歌。它指出 defense in depth 依赖一个 independence assumption：如果一层失效，其他层能捕获。但现实中安全层会共享性能和成本约束。例如复杂 shell command 的 AST parsing 可能带来 UI freeze，导致系统为了性能放宽 per-subcommand checking。这样，多个安全层可能在同一资源压力下同时退化。

论文还引用独立安全研究指出，某些漏洞来自 pre-trust initialization ordering：hooks、MCP connections 和 settings resolution 在用户 trust dialog 前已经执行，从而落在 deny-first permission pipeline 之外。这提示 permission architecture 不仅有空间顺序，还有时间顺序；“什么时候安全机制开始生效”本身就是攻击面。

## 扩展机制

### MCP, plugins, skills, hooks

Claude Code 的扩展架构不是单一 tool API，而是四种机制：

| Mechanism | 独特能力 | Context cost | 插入点 |
|---|---|---:|---|
| MCP servers | 外部服务集成、多 transport tool integration | High | model / tool pool |
| Plugins | 多组件打包与分发 | Medium | assemble / model / execute |
| Skills | domain-specific instructions 与 meta-tool invocation | Low | context injection |
| Hooks | lifecycle interception、tool blocking / rewriting / annotation | Zero by default | execute pre/post |

这张表是论文中很有价值的设计抽象。它说明扩展机制的差异不只是“功能不同”，更是 context budget 与 runtime insertion point 的差异。

如果所有扩展都做成 tools，那么每个扩展都要向模型暴露 schema，context 成本会很高。如果所有扩展都做成 prompt instructions，又无法可靠地执行外部服务或拦截生命周期。Claude Code 选择多机制分层，本质上是在表达性、上下文成本和控制能力之间做折中。

### Plugin 作为分发层

Plugins 的特殊性在于它不是一个单独 runtime primitive，而是 packaging / distribution layer。Plugin manifest 可以包含 commands、agents、skills、hooks、MCP servers、LSP servers、output styles、channels、settings 和 user configuration 等组件。

也就是说，plugin 可以同时扩展 prompt context、工具 surface、hook lifecycle、agent definitions 和 response style。这让插件生态更有组合性，但也带来更高配置复杂度和更大的攻击面。

### Skill 与 Agent 的区别

论文把 `SkillTool` 和 `AgentTool` 做了重要区分：

- `SkillTool` 把 skill instructions 注入当前 context window；
- `AgentTool` 启动一个新的 isolated context window。

这一区分也很适合迁移到知识库：skill 更像“当前 agent 的认知模式扩展”，agent 更像“新建一个有自己上下文、工具和权限的执行单元”。两者都能提升能力，但对 context、权限和审计的影响完全不同。

## 上下文构造与记忆

### Context window 是核心稀缺资源

论文反复强调，Claude Code 的关键瓶颈是 context window。文章提到旧模型 200K、新 Claude 4.6 系列 1M context，但即使如此，coding agent 仍会因为长任务、多工具结果、文件读取、subagent summaries 和 session resume 面临上下文压力。

Context window 由多个来源共同构造：

- system prompt 与 output style；
- environment info，如 git status；
- CLAUDE.md hierarchy；
- path-scoped rules；
- auto memory；
- skill descriptions、MCP tool names、deferred tool definitions；
- conversation history；
- file reads、command outputs、tool results；
- compact summaries。

其中 system context 会进入 system prompt，而 CLAUDE.md / date 等 user context 被作为 user-context message prepend 到消息数组。这意味着项目指令不是 deterministic enforcement，而是模型可见的 conversational guidance；真正的硬约束仍靠 permission rules。

### CLAUDE.md hierarchy

论文把 Claude Code 的 memory 设计解读为 transparent file-based configuration and memory。CLAUDE.md 是普通 Markdown 文件，用户可以阅读、编辑、删除、版本控制。相比 embedding-based retrieval 或 database-backed memory，它牺牲了一部分检索灵活性，但换来了 inspectability。

CLAUDE.md 体系包括：

- managed memory：OS-level policy；
- user memory：用户全局私有指令；
- project memory：项目根目录或 `.claude/` 下的项目指令；
- local memory：gitignored 的私有项目指令。

加载顺序还体现 priority：离当前目录更近的文件后加载，从而获得更高注意力。嵌套目录规则可以 lazy load，当 agent 读取特定目录文件时才进入上下文。这是 context efficiency 与 instruction locality 之间的折中。

### 五层 compaction pipeline

Claude Code 在每次模型调用前运行五层 pre-model context shapers：

| Layer | 作用 |
|---|---|
| Budget reduction | 限制单个 tool result 大小，用 content references 替代超大输出 |
| Snip | 轻量裁剪较旧历史 |
| Microcompact | 细粒度、cache-aware 压缩 |
| Context collapse | 对历史做 read-time projection，而不是修改原始 REPL history |
| Auto-compact | 调用模型生成完整 summary，作为最后手段 |

这套 pipeline 的原则是 lazy degradation：先做便宜、局部、低损伤的压缩，再逐渐升级到更强的 summarization。相比简单 sliding window 或 single summarization，它更细腻，也更难预测。用户可能很难知道哪些内容被 budget reduction、snip 或 collapse 影响，context efficiency 与 transparency 因此存在张力。

## Subagent 与多代理编排

### AgentTool 与 built-in subagents

Claude Code 通过 `AgentTool` 实现任务委托。内置 subagent 类型包括：

- Explore：偏阅读与搜索，deny write / edit；
- Plan：生成结构化计划；
- General-purpose：通用任务；
- Claude Code Guide：帮助用户理解 Claude Code；
- Verification：运行测试、lint 等验证；
- Statusline-setup：终端状态栏配置。

用户也可以通过 `.claude/agents/*.md` 定义 custom agents，frontmatter 中可指定 tools、disallowedTools、model、effort、permissionMode、mcpServers、hooks、maxTurns、skills、memory scope、background flag 和 isolation mode 等。这说明 Claude Code 的 custom agent 不是一个简单 prompt，而是一个可配置的子系统。

### 隔离模式

Subagent 支持几类 isolation：

- worktree：创建临时 git worktree，使子代理可以修改独立副本；
- remote：内部用户可用，运行在远程 Claude Code Remote 环境；
- in-process：默认模式，共享 filesystem，但 conversation context 隔离。

这种设计介于 conversation-based multi-agent 和 container-isolated execution 之间。它不一定提供 Docker 级资源隔离，但通过 git worktree 获得轻量文件系统隔离，并保持零外部基础设施依赖。

### Sidechain transcripts 与 summary-only return

每个 subagent 拥有自己的 JSONL transcript 和 metadata file。父会话只接收子代理最终 summary 和 metadata，不接收完整 conversation history。

这是一个关键 context-conservation 设计：如果父 agent 吸收所有子代理的完整历史，多代理并行会迅速导致 context explosion。summary-only return 牺牲了细节可见性，但保留了父上下文可控性。完整 sidechain transcript 仍可用于 debugging 和 auditing。

## Session Persistence 与恢复

Claude Code 的 session persistence 使用 mostly append-only JSONL transcript。持久化通道包括：

- session transcripts：每个 session 一个 project-scoped JSONL 文件，记录 messages、tool results、compact boundaries、metadata；
- global prompt history：用户 prompt 历史，支持命令行导航；
- subagent sidechains：每个子代理单独 transcript 与 metadata。

Append-only 的好处是简单、可读、可审计、可重建。缺点是 query power 弱：如果想查询“哪些 tool call 修改了某文件”，需要事后重建，而不是数据库直接查询。

Resume 和 fork 会重放 transcript 恢复 conversation，但不会恢复 session-scoped permissions。论文认为这是 deliberate safety-conservative design：信任状态属于当前 session，不应默默跨 resume / fork 迁移。这个设计牺牲便利性，但避免把旧权限带入新上下文。

Compaction 也遵守 mostly append-only 原则。compact boundary 通过 UUID metadata 记录 preserved segment，loader 在 read-time patch message chain，而不是直接修改旧 transcript lines。这样 live context 可以被压缩，而 durable history 仍保留审计完整性。

## 与 OpenClaw 的对照

论文选择 OpenClaw 作为对照，不是为了证明 Claude Code 更好，而是说明相同设计问题在不同部署场景中会产生不同答案。

| Dimension | Claude Code | OpenClaw |
|---|---|---|
| System scope | 单仓库、CLI / IDE coding harness、session-scoped process | 持久 WebSocket gateway daemon、多 channel control plane |
| Trust model | model 与执行环境之间做 per-action deny-first evaluation | 单一可信 operator，gateway perimeter 做身份与访问控制 |
| Agent runtime | `queryLoop()` 是系统中心 | agent runner 嵌入 gateway RPC dispatch |
| Extension architecture | MCP、plugins、skills、hooks，按 context cost 分层 | manifest-first plugin system，扩展 gateway capability surface |
| Memory / context | CLAUDE.md hierarchy + 五层 compaction | bootstrap files、MEMORY.md、daily notes、optional dreams、hybrid retrieval |
| Multi-agent / routing | 父 agent 委托 subagents，summary-only return | gateway 托管多个独立 agents，并支持 channel binding 与 nested delegation |

这个对照揭示了三个结论：

第一，agent 系统面对的问题是稳定的：reasoning/harness 分界、安全边界、context 管理、扩展机制、delegation、persistence 都必须回答。

第二，答案取决于 deployment topology。Claude Code 面向 trusted developer machine 上的 untrusted model，因此把边界放在 model 和 execution environment 之间；OpenClaw 面向多 channel gateway，因此先把边界放在 gateway perimeter。

第三，两者并非纯粹替代关系。OpenClaw 可以通过 ACP 托管 Claude Code 作为外部 coding harness。这提示未来 agent 系统可能是 layered composition：gateway-level control plane 与 task-level harness 组合，而不是单一 agent 架构覆盖所有场景。

## 讨论与启发

### Operational harness 优先于 decision scaffolding

论文最重要的理论判断是：Claude Code 更相信“模型在好的执行环境中做局部判断”，而不是“用复杂 planner 严格约束模型”。它没有把主要复杂性放在 state graph、explicit planning module 或 search tree 中，而是放在：

- tool routing；
- permission gates；
- context assembly；
- compaction；
- recovery；
- sandbox；
- session persistence；
- extension lifecycle。

这对 agent builder 有很强启发：随着 frontier model 的 coding 能力变强，差异化可能越来越来自 harness，而不是 prompt engineering 或 planning wrapper。一个 production agent 的核心能力可能不在“能想什么”，而在“能安全、连续、可恢复地做什么”。

### 价值张力不是 bug

论文用 Table 4 总结了几组结构性张力：

- Authority vs. Safety：人类审批给了 authority，但 93% prompt approval 说明审批疲劳削弱了实际安全；
- Safety vs. Capability：深度防御会带来性能与成本，安全层可能因延迟退化；
- Adaptability vs. Safety：plugins、hooks、MCP 提升适配性，也扩大 attack surface；
- Capability vs. Adaptability：更主动的 agent 可完成更多任务，但高频主动行为降低用户偏好；
- Capability vs. Reliability：bounded context 与 subagent isolation 可能提高速度，却降低全局一致性。

这些张力不是实现瑕疵，而是 agent 系统把多个价值同时最大化时必然出现的 trade-off。好的架构不是消除所有张力，而是让张力显式化、可配置、可审计。

### 对长期代码质量的预测

论文从架构属性推导出一个经验预测：bounded context 和 local decision making 可能导致代码重复、局部一致但全局不一致、技术债持久化。Claude Code 的 compaction pipeline、subagent summary isolation 和 file-based memory 都是在缓解这个问题，但源码分析无法证明它们足以解决。

这也是为什么作者把 long-term human capability 和 codebase coherence 放到结论中。Coding agent 的评估不能只看 task success rate 或短期速度，还要看：

- 三个月后 velocity 是否因为复杂度回升而消失；
- AI-authored commits 的缺陷是否持续存在；
- 安全相关技术债是否更难清除；
- 开发者是否理解 agent 生成的结构。

这部分是全文最有“综述报告”性质的地方：它把工具架构与软件工程长期质量、组织学习、开发者培养联系起来。

## 未来方向

论文提出六个开放方向。

### Silent Failure and Observability-Evaluation Gap

生产 agent 的主要失败模式可能不是 crash，而是 silent mistakes。工具调用、session transcript 和 hooks 能提供 observability，但 observability 不等于 evaluation。未来 harness 可能需要把 generator-evaluator separation、post-hoc checks、trajectory anomaly detection、cost-controlled benchmark 等机制嵌入系统，而不是只依赖模型能力提升。

### Persistence and Longitudinal Colleague Relationships

Claude Code 当前有 CLAUDE.md / auto memory 和 session transcript 两层，但中间缺少一种 durable state：既不是静态项目指令，也不是单次 session 历史。未来 agent 可能需要积累策略、偏好、项目经验和人机协作关系，同时保持 file-based transparency 和 permission safety。

### Harness Boundary Evolution

未来 harness 会在四个方向扩展：

- where：session、harness、sandbox 可能被虚拟化成可替换接口；
- when：agent 从 reactive tool use 走向 proactive / background agents；
- what：工具从文本和 shell 扩展到 multimodal / VLA / physical actions；
- with whom：从 parent-subagent 走向更复杂 multi-agent coordination。

这些扩展会重新打开安全、治理、上下文和可逆性问题。

### Horizon Scaling

Claude Code 当前的主要单位是 turn、session 和 subagent。当 autonomous work 扩展到多日、多周、科学项目级别时，现有 context pipeline、summary-only subagent return 和 append-only session transcript 是否足够，仍未解决。这与极长程 agent 研究直接相关。

### Governance and Oversight at Scale

随着 agent 自主性提高，外部监管会要求 logging、transparency、human oversight 和 compliance interfaces。Claude Code 的 transcript 可以内部审计，但未必满足外部合规审计形式。values-over-rules 也可能需要转化成更明确的可审查规则。

### Long-Term Human Capability

作者最后把长期人类能力保存从评估镜头升级为设计问题。未来系统可能需要主动支持 human understanding，而不是只最大化任务完成率。例如，agent 可能需要解释关键改动、保留学习线索、检测用户过度 cognitive offloading，或提供 comprehension-preserving interface。

## 方法论与证据等级

论文附录明确给出 evidence tiers：

| Tier | 来源 | 作用 |
|---|---|---|
| Tier A | 官方 Anthropic 文档和工程文章 | 建立 product intent，但不保证实现细节 |
| Tier B | Claude Code v2.1.88 TypeScript source files / functions | 最强证据层，支持实现结构与控制流判断 |
| Tier C | community analysis、OpenClaw comparison、code pattern inference | 需要用 hedging language |

这一分层让报告比普通产品分析更严谨。它承认 source code 能证明 implemented structure、control flow、dependencies 和 feature gates，但不能证明设计意图、生产 feature flag 是否启用、运行时流行度或未发布行为。

## 局限与疑问

这篇文章的局限也比较清楚。

**静态快照问题**：分析基于 Claude Code v2.1.88。Feature flags 会导致不同 build target 出现功能差异，源码中的分支不等于用户一定能用到。

**反向工程认识论边界**：源码可以揭示实现结构，但不能确认 Anthropic 内部设计意图、生产环境配置、A/B experiment 状态或未发布功能是否真实启用。

**单系统分析的外推边界**：Claude Code 是一个重要 production coding agent，但不能代表所有 agent 系统。论文通过 OpenClaw 做 calibration，但 OpenClaw 也只是另一个特定快照。

**缺少直接实证评估**：论文提出了关于 technical debt、code complexity、human capability 的重要预测，但这些并不是在 Claude Code 上直接实验得出的。它引用相邻工具和外部研究来支持推理，因此相关结论应理解为 architecture-informed hypotheses。

**安全分析依赖公开研究**：对 CVE、pre-trust initialization、permission bypass 等问题的讨论来自公开安全研究。它有助于校准架构风险，但不等于完整安全审计。

**综述密度高但分类可能过拟合 Claude Code**：五价值、十三原则、七组件、五层架构、四扩展机制、六未来方向都很清晰，但这些 taxonomy 也可能受 Claude Code 具体实现强烈塑形。迁移到其他 agent 系统时，需要重新验证哪些分类仍成立。

## 关键结论

这篇文章最值得记住的不是 Claude Code 有多少工具、多少 hook、多少 permission mode，而是它提出了一种理解 production agent 的方式：agent loop 可以很简单，真正决定能力与可靠性的，是 loop 周围的 deterministic harness。

Claude Code 的设计点可以概括为：

- 模型拥有广泛局部决策权；
- harness 负责权限、安全、context、工具、恢复和持久化；
- 扩展机制按 context cost 和 lifecycle insertion point 分层；
- memory 与 session 采用可见、可审计、mostly append-only 的文件化设计；
- subagent 通过上下文隔离和 summary-only return 控制 context explosion；
- 安全不依赖单一边界，而是 deny-first rules、classifier、hooks、sandbox 和 permission reset 的组合。

同时，论文也提醒：短期 capability amplification 并不自动等于长期可靠性。随着 coding agent 更主动、更长程、更可扩展，真正重要的问题会从“能不能完成这次任务”转向“能否长期保持代码库一致性、用户理解、组织学习和可审计治理”。

## 相关知识链接

- Agent 架构：[[application/agents/agent|Agent]], [[application/agents/workflow-agent|Workflow Agent]], [[application/agents/planning|Planning]], [[application/agents/memory|Memory]], [[application/agents/multi-agent|Multi-Agent]]
- 工具与扩展：[[application/tool-use/tool-calling|Tool Calling]], [[application/tool-use/mcp|MCP]], [[application/tool-use/structured-api|Structured API]]
- 上下文管理：[[application/rag/context-compression|Context Compression]], [[training/mid-training/long-context-training|Long Context Training]]
- 评测与安全：[[application/evaluation/benchmark|Benchmark]], [[application/evaluation/llm-as-judge|LLM-as-a-Judge]]
