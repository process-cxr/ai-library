---
title: "Agentic Mid-training Strategy for General-Purpose Agent Capability Injection"
created: 2026-06-30
published: 2026-06-30
modified: 2026-07-01
type: source
status: processed
source_type: report
area: sources
tags:
  - source
  - report
  - agent
  - mid-training
  - agent-training
  - evaluation
aliases:
  - Agentic Mid-training Synthesis
  - Agentic Mid-training Strategies Reading Synthesis
---

# Agentic Mid-training Strategy for General-Purpose Agent Capability Injection

## 基本信息

本报告面向自研模型的 agentic mid-training 能力提升计划，围绕 [[training/mid-training/capability-injection|Mid-training Capability Injection]] 阶段的数据、训练和评测闭环展开，目标是在后训练之前形成可迁移的规划、工具使用、环境反馈理解、错误恢复和长程任务执行能力。

## 1. 研究目标与阶段边界

### 1.1 训练阶段划分

训练流程采用如下阶段划分：

```text
Pre-training
  → Mid-training
  → Post-training
  → Inference / Test-time Scaling
```

后训练已经能够训练出有效的 agent 能力，但随着任务长度、工具复杂度和环境反馈复杂度上升，仅在后续适配阶段集中学习 agent 行为会带来更高的数据、算力和迭代压力。Mid-training 引入 agentic 训练的价值在于提前塑造更适合 agent 化的模型先验：在百 B 级别 agent trajectory 数据上学习任务分解、工具选择、状态跟踪、环境反馈理解和错误恢复等通用行为模式，为后续 agent 适配提供更好的初始化。

Pre-training 提供通用语言、知识、代码和基础推理能力；  
Mid-training 注入可迁移的 agentic priors；  
Post-training 承接具体交互协议、工具格式、任务风格、产品形态和安全边界。

讨论范围限定在 mid-training 阶段的数据组织、训练 mixture 和能力评测。后训练作为后续承接阶段处理，不在本文展开。

### 1.2 Agentic capability injection 目标

Mid-training 阶段面向后续 agent 适配，训练可迁移的 agentic priors。核心能力包括：

- 任务理解与目标分解；
- 长程计划与阶段性推进；
- 工具选择与参数 grounding；
- 对 observation、tool result、错误信息和环境反馈的解释；
- 证据整合、状态跟踪和上下文压缩；
- 反思、错误诊断与恢复；
- 任务终止条件判断；
- 多轮交互中的一致性和鲁棒性。

训练信号应覆盖围绕任务状态演化的 decision process，即模型如何根据目标、工具、上下文和环境反馈持续推进任务。

### 1.3 Mid-training 产出

Mid-training 的主要产出是具备更强 agentic priors 的 base checkpoint。该 checkpoint 应在不同任务域中表现出更好的规划、工具选择、环境反馈理解、错误恢复和长程状态维持能力，并为后续 agent 适配提供更好的初始化。

对应训练与评测目标包括：

- 在不依赖完整 agent rollout 的情况下，快速判断 checkpoint 的 planning、tool selection、observation interpretation 和 recovery 能力是否提升；
- 在相同后训练流程下，比较不同 mid-training 数据策略是否降低后续适配成本；
- 在完整后训练后，验证模型是否在跨任务域、跨工具和跨 scaffold 的 agent 任务上获得稳定收益；
- 在训练过程中监控通用能力回退、格式过拟合、future leakage 和 reward hacking 等风险。

后续数据组织、质量分层、快速评测和长期验证均围绕 mid-training checkpoint 的可适配性展开。

## 2. 数据目标与总体原则

数据目标是构建、获取并利用百 B 级别、带 reasoning content 的多任务 agent trajectory corpus，用于提升自研模型的通用 agentic 能力。数据建设需同时满足规模、结构和质量三类约束。

规模约束要求数据量达到 mid-training 能力注入所需量级，区别于少量后训练示例集。结构约束要求数据落库时保留任务目标、reasoning、action/tool call、observation、反馈结果和最终 outcome 之间的边界。质量约束要求每条数据能够被分到高、中、低或 reject / holdout 等质量层级，并记录其来源、审计方式和适合进入训练的权重。

Agent 轨迹入库保留如下训练结构：

```text
task / user goal
  → reasoning
  → action or tool call
  → observation / feedback
  → updated reasoning
  → next action
  → ...
  → final outcome
```

Reasoning content 来源包括强模型原生生成、已有轨迹记录、后处理合成和环境 rollout 扩展。数据记录需标注 reasoning 的生成方式、可见上下文、质量等级和审计状态。

数据建设遵循四项总体原则。

**时序边界清晰**：每一步使用该步之前真实可见的信息，避免当前 tool result 或后续成功信号泄漏。

**任务域多样**：覆盖 `code`、`research`、`web`、`data_analysis`、`document`、`office_workflow`、`general_tool_use` 等任务域。

**Scaffold 多样**：覆盖不同工具协议、prompt 模板、tool schema 和执行环境，降低单一 agent 框架主导训练分布的风险。

**可评测可归因**：每条轨迹记录任务类型、工具类型、step 类型、结果、质量标签和来源，支持后续 ablation 与误差分析。

## 3. Agentic 数据获取体系

### 3.1 强模型生成完整 reasoning agent trajectories

第一类数据来自更强模型在真实或模拟环境中的完整 agent rollout。给定 seed task 或 user query，强模型在特定 scaffold 下进行多轮推理、工具调用、环境观察和最终回答，形成带 reasoning content 的完整轨迹。

该数据形态保留完整行为链路，呈现 agent 从初始任务进入信息搜索、工具调用、阶段性分析、错误恢复和最终交付的全过程，是 mid-training 阶段最接近目标行为分布的高质量样本，尤其适合训练长程状态维持、工具调用节奏和跨 observation 的任务推进能力。

采集时同步保存结构化 metadata：

- teacher model、版本、温度、采样配置；
- scaffold、system prompt、tool schema、环境版本；
- task domain、task type、难度、来源；
- 每一步 reasoning、visible output、tool name、tool arguments；
- observation、错误信息、执行日志和最终 outcome；
- 成功标准、judge 结果、verifier 结果或人工审核结果；
- reasoning 是否为原生生成、压缩生成或后处理生成；
- 截断、压缩、重写和脱敏记录。

强模型轨迹需控制 teacher bias 与 scaffold overfitting。数据采集阶段应覆盖多任务域、多工具集合、多交互模板和多环境类型，避免训练分布被单一 agent 框架主导。

### 3.2 Reasoning 缺失轨迹的合成补齐

第二类数据来自已有 action-observation 轨迹的 reasoning 补齐。部分轨迹保留了用户目标、assistant visible output、tool call、tool result 和最终 outcome，但缺少显式 reasoning；处理目标是为每个 assistant step 补齐训练用 `reasoning_t`。

关键设定是：

```text
输入：
  prefix_<t
  gold_action_t / gold_tool_call_t
  tool schema
  通用工具与任务知识

输出：
  reasoning_t
```

合成任务以当前 gold action 为条件，为该 action 构造可训练、可审计、与当前可见信息一致的中间监督信号。Teacher model 的目标是解释并支撑既有轨迹中的当前动作，而非重新规划一条新的 agent 路径。

合成上下文严格区分允许信息和禁止信息：

| 类型 | 信息 |
|---|---|
| 允许使用 | 当前 step 之前的 prefix、当前 gold action/tool call、tool schema、通用知识 |
| 禁止使用 | 当前 tool result、后续 observation、后续 assistant 修正、最终成功信号、后验 judge 结论 |

合成 reasoning 的质量控制包含：

- prefix grounding：事实性陈述能回链到当前可见上下文；
- action consistency：reasoning 支持当前 tool/action，并与既有轨迹目标一致；
- argument grounding：tool arguments、文件名、搜索词、URL、表格字段等参数有合理来源；
- future leakage detection：不引用当前执行结果或后续反馈；
- verbosity control：避免生成冗长、空泛、模板化的思考；
- style normalization：保持不同来源数据的 reasoning 风格稳定；
- teacher version tracking：记录合成模型、prompt 和采样配置；
- judge filtering 与人工抽检：对高风险数据进行分层审计。

该类数据用于扩大 agent trajectory 覆盖面。训练 mixture 单独标注 synthetic reasoning 来源，并根据 leakage、grounding、action consistency 和抽检结果进入相应质量层级。

### 3.3 基于 RLE 环境的自举式轨迹扩展

第三类数据来自环境驱动的自举式扩展。具备一定能力水平的 agent model 进入 RLE 环境后，根据 seed query 生成指定任务类型的 agent trajectories，用于定向调整数据多样性和难度分布。

RLE 轨迹扩展主要补齐以下数据缺口：

- 稀缺任务域，例如特定行业文档、复杂网页任务、跨工具办公自动化；
- 长 horizon 任务，例如多阶段 research、复杂 debugging、跨文件工程修改；
- recovery 场景，例如工具失败、搜索失败、执行报错、权限限制、测试失败；
- 特定工具与环境，例如数据库、表格、浏览器、代码仓库、文档系统；
- curriculum 任务，从短程确定性任务逐步过渡到开放式长程任务；
- hard negative 和 near-miss 轨迹，用于训练错误识别与恢复。

RLE 数据需配套 verifier、judge、环境日志和 outcome filter。失败轨迹进入 diagnosis、reflection、recovery 或 contrastive decision 数据集；通过验证的成功轨迹进入高质量或中等质量训练池。

## 4. 数据结构化、分类与训练使用

### 4.1 数据分类

Agentic 数据分类服务于训练 mixture、采样、curriculum、ablation 和评测归因。第一阶段优先控制少数会直接影响训练策略的核心变量。

核心分类维度控制为 3 个：任务域、能力/工作流类型、数据质量等级。数据来源、tool schema、context length、scaffold template、teacher model、judge score 等字段作为辅助 metadata 保留。

三个分类维度分别回答数据建设中的三个核心问题。

**任务域 / 环境域**：标识数据来源的任务环境。典型取值包括 `code`、`research`、`web`、`data_analysis`、`document`、`office`、`general_tool_use`。该维度用于控制领域覆盖、跨域泛化和整体数据 mixture。

**能力 / 工作流类型**：标识数据主要覆盖的 agentic 子能力。典型取值包括 `planning`、`information_seeking`、`tool_selection`、`argument_grounding`、`evidence_grounding`、`transformation`、`verification`、`reflection`、`recovery`、`termination`。该维度用于控制能力分布、组织能力视图，并设计快速评测。

**数据质量等级**：标识数据所处质量层级。质量等级采用 `high`、`medium`、`low`、`reject / holdout` 四类。该维度用于控制采样权重、过滤策略、ablation 设计和是否进入主训练集。

数据质量等级是百 B 级 agent 轨迹建设中的必要主轴。高质量数据包括经过 verifier 或人工审核的成功轨迹、原生 reasoning 完整且时序边界清楚的轨迹、执行结果可验证的 RLE 轨迹；中等质量数据包括通过自动 judge 过滤但缺少强 verifier 的轨迹、经过严格 leakage 检查的 synthetic reasoning 轨迹；低质量数据包括 outcome 不确定、reasoning 较弱、工具结果噪声较大但仍可用于弱监督或鲁棒性训练的样本。存在 future leakage、结构错误、不可脱敏风险或错误行为被标成成功的样本归入 reject / holdout，排除出主训练集。

更细的字段，例如 tool schema、step index、context length、scaffold template、teacher model、judge score、token length 等，在数据工程层面保留，用于审计、过滤和后续误差分析。

### 4.2 质量控制原则

Agentic mid-training 数据设置四层质量控制：

1. **结构合法性检查**：消息边界、tool call、tool result、role、时间顺序和 schema 必须可解析。
2. **时序一致性检查**：当前 step 的 reasoning 不得引用当前 action 之后才出现的信息。
3. **行为一致性检查**：reasoning、visible output、tool call 和 observation 之间应形成合理链条。
4. **结果可信度检查**：成功轨迹需要 verifier、judge 或人工审核支撑；失败轨迹需要明确标注失败点和用途。

通用 agent 数据治理包含脱敏和安全过滤，覆盖账号、密钥、内部路径、个人信息、内部域名、非公开项目名、客户信息和无法公开的业务细节。脱敏过程需保持任务结构；实体名称改写后保持前后引用一致。

### 4.3 完整轨迹作为主训练单位

已有完整 agent trajectory 通过结构化入库、分类标注、质量分层和采样配置服务于 mid-training。完整轨迹是主数据单位；分类标签用于管理数据分布、控制训练 mixture、组织快速评测和发现能力缺口。

入库数据保留完整任务执行链路：

```text
task
  → reasoning / action / observation
  → reasoning / action / observation
  → ...
  → final answer / final artifact / outcome
```

该组织方式使模型在标准 next-token prediction 训练中学习长程状态维持、跨步骤一致性、工具调用节奏、阶段性总结和任务终止判断。数据处理重点放在 trajectory 边界、超长上下文、packing、截断策略和不同长度样本的采样比例上，避免默认拆分为独立 step 样本。

### 4.4 分类标签服务训练 mixture

任务域 / 环境域用于控制训练覆盖面。`code`、`research`、`web`、`data_analysis`、`document`、`office`、`general_tool_use` 等任务在 token 预算中保持合理比例，避免模型过度偏向单一任务环境。

能力 / 工作流类型用于观察能力覆盖。`planning`、`information_seeking`、`tool_selection`、`argument_grounding`、`evidence_grounding`、`verification`、`reflection`、`recovery`、`termination` 等标签用于判断数据是否过度集中在搜索和读取，以及是否缺少验证、恢复、终止判断等关键能力。

数据质量等级用于决定采样权重和过滤策略。高质量数据作为主训练主体；中等质量数据用于覆盖扩展；低质量数据控制比例；reject / holdout 数据进入审计、分析或评测保留集。

### 4.5 分类标签服务评测与补齐

分类标签同时服务快速评测和数据补齐。快速评测从完整轨迹中抽取局部判断题，例如 search query selection、tool argument grounding、evidence grounding、bug localization、error diagnosis、recovery decision 和 termination decision。这些评测样本用于观察 mid-training checkpoint 的 agentic potential，不改变完整轨迹作为主训练数据的地位。

分类统计显示某些能力长期不足时，再进行定向补齐。`recovery`、`verification`、`evidence_grounding`、`termination` 等能力在成功轨迹中容易占比偏低，可通过强模型生成、reasoning 补齐或 RLE 环境扩展进行补齐。补充数据继续纳入同一套任务域、能力类型和质量等级体系，避免形成与完整轨迹割裂的独立数据池。

## 5. Mid-training Recipe 设计

### 5.1 两阶段训练框架

Agentic mid-training 采用两个主要阶段。

第一阶段是 Agentic Skill Prior Warmup：

```text
数据：
  短中程高质量 trajectories
  分类统计后的能力补齐数据
  高置信 native reasoning

目标：
  任务分解
  工具选择
  参数 grounding
  observation 解释
  反思与恢复基础能力

context:
  16K / 32K 为主
```

该阶段重点训练通用 agentic 操作能力。训练中控制工具格式多样性，避免单一 scaffold 在早期过度主导。

第二阶段是 Long-horizon Agent Trajectory Training：

```text
数据：
  完整长轨迹
  complex multi-tool tasks
  多轮 recovery trajectories
  长上下文 research / code / web / data analysis 任务
  质量分层后的 RLE trajectories

目标：
  长程状态维持
  多步骤一致性
  跨 observation 的任务推进
  复杂任务终止判断

context:
  64K / 128K 或更高
```

该阶段重点训练长上下文中的任务状态保持、历史反馈处理、计划延续和循环抑制。评测时单独区分 long-context recall 与 agent decision quality。

### 5.2 数据 mixture 原则

训练 mixture 采用质量与功能双重分层，避免所有 agent 数据等权混合：

- 高质量数据作为主训练主体，但需控制任务域、能力类型和 scaffold 分布；
- 中等质量数据可作为覆盖扩展和鲁棒性补充，采样权重应低于高质量数据；
- 低质量数据控制采样比例，主要用于弱监督、鲁棒性训练、错误分析或后续清洗；
- synthetic reasoning 和 RLE self-generated 数据按 leakage 检查、grounding、verifier、judge 和人工抽检结果进入不同质量层级；
- failure trajectories 主要进入 diagnosis、reflection 和 recovery 数据池，避免作为成功策略模仿样本；
- 保留一定比例通用文本、代码、数学和指令数据，防止模型退化为 agent log completion model；
- 控制 read/search/observe 与 act/verify/recover 的比例，避免模型学会无限收集信息而不执行；
- 控制不同 context bucket 的采样比例，避免训练完全被超长样本占据。

### 5.3 训练目标与格式约定

Agentic mid-training 作为海量 agent 数据上的继续训练，沿用标准 causal LM / next-token prediction objective。该阶段的重点不是引入新的 loss，而是通过数据结构、mixture、packing、context length 和格式多样性改变模型可学习到的行为分布。

数据格式需要清晰保留 user、assistant reasoning、visible output、tool call、observation 和 final response 的边界，便于后续组织能力视图、快速评测和错误分析。格式侧可引入多种 tool chat template、不同 JSON schema、函数调用格式和文本工具格式，以提高跨 scaffold 泛化能力。Qwen3-Coder-Next 的经验表明，scaffold diversity 对 agent 模型非常关键。

训练工程上重点关注三项约束：

- packing 时避免破坏 trajectory 内部顺序和 step 边界；
- 长上下文样本按 context bucket 分层采样，防止训练完全被极长轨迹主导；
- 不同任务域、能力类型和质量等级的数据在 batch 与全局 token 预算中保持可控比例。

## 6. Mid-training Agentic Capability 评测体系

Agentic mid-training 的评测应包含快速验证和长期验证两层。快速验证用于训练中频繁筛选 checkpoint 和数据 recipe；长期验证用于判断 mid-training 产物经过完整后训练后是否提升最终 agent 能力。

### 6.1 快速验证：不依赖完整后训练的 agentic potential eval

快速验证在不运行完整后训练流程的条件下评估 mid-training checkpoint 的 agentic priors。APTBench 提供了可复用的评测范式：将真实 agent trajectory 转换为 base/mid checkpoint 可完成的静态或半静态决策题。

内部 APTBench-style eval 采用 logprob ranking、multiple choice、constrained completion 或短文本生成，减少对完整 agent 框架的依赖。评测维度包括：

| 能力维度 | 示例任务 |
|---|---|
| next-plan ranking | 给定任务 prefix，选择最合理的下一阶段计划 |
| next-tool ranking | 给定历史和可用工具，选择下一步工具 |
| tool-argument ranking | 比较多个 tool arguments，选择与 prefix 最一致的版本 |
| observation interpretation | 解释 tool result、错误日志、网页内容或检索结果的含义 |
| recovery decision | 面对失败 observation，选择修复路径 |
| evidence grounding | 判断回答或行动是否被当前证据支持 |
| task-state tracking | 判断哪些子目标已完成、哪些仍缺失 |
| termination decision | 判断任务结束条件，避免 premature termination |
| scaffold transfer | 在不同工具格式下保持同类决策能力 |

不同任务域配置对应的子域评测：

| 子域 | 快速评测示例 |
|---|---|
| code | bug localization、patch selection、test selection、error log diagnosis |
| research/web | search query selection、page selection、citation grounding、claim verification |
| data analysis | table operation selection、SQL/tool argument grounding、chart/summary validity |
| document/office | extraction plan、document transformation step、cross-document evidence matching |
| general workflow | multi-tool next step、dependency ordering、handoff/summary quality |

快速评测服务于训练迭代：

- 比较不同 mid-training checkpoint；
- 判断某类数据加入后提升的是哪一类能力；
- 区分 long-context recall 改善与 agent decision 改善；
- 提前发现 scaffold overfitting；
- 在昂贵后训练前过滤明显无效的数据 recipe。

### 6.2 长期验证：后训练后完整 agent evaluation

长期验证将后训练后完整评测纳入 mid-training capability evaluation 体系。比较不同 mid-training 策略时，固定后续流程：

```text
mid-training checkpoint
  → same post-training recipe
  → same inference budget
  → full agent evaluation
```

该设置用于隔离 mid-training 数据策略的贡献，避免后训练流程、推理预算或 scaffold 调参混入对比。

长期验证指标包括：

- end-to-end success rate；
- pass@1 / pass@k；
- 平均轮数、平均 token、平均 tool call 数；
- tool-call validity；
- recovery success rate；
- invalid action rate；
- loop rate；
- premature termination rate；
- task-state consistency；
- cross-domain generalization；
- cross-scaffold generalization；
- cost / latency；
- reward hacking 与环境泄漏审计；
- 通用能力回归，例如语言、数学、代码、指令跟随和安全性。

Code 子域使用 SWE-Bench、Terminal-Bench、真实仓库任务或内部可执行任务；research/web 子域使用可审计 citation、网页任务、检索任务和报告质量评测；通用工具任务使用多工具 workflow benchmark 和人工审计样本。

## 7. 对比实验与策略选择

不同训练策略通过系统化 ablation 进行比较。基础实验矩阵如下：

| 实验组 | 数据策略 | 目的 |
|---|---|---|
| Base | 不加入 agentic mid-training | 作为底座对照 |
| Full trajectory only | 只加入完整成功轨迹 | 测试长轨迹本身的收益 |
| Native reasoning only | 只使用原生 reasoning 轨迹 | 测试高质量 reasoning 的上限 |
| Synthetic reasoning added | 加入合成 reasoning 补齐数据 | 测试覆盖扩张是否有效 |
| Ability supplement mix | 加入分类统计后的能力补齐数据 | 测试稀缺能力补齐收益 |
| RLE-generated mix | 加入 RLE 自举轨迹 | 测试定向扩展收益 |
| Multi-domain balanced mix | 跨领域均衡采样 | 测试通用 agent 能力与泛化 |
| Full proposed mix | 使用完整分层 mixture | 测试综合方案 |

每个实验组经过三层评估：

```text
quick mid-training eval
  + small post-training adaptation eval
  + full post-training eval
```

策略选择采用多目标标准：

- mid-training 快速评测是否稳定提升；
- 达到同等后训练表现所需的适配数据量是否减少；
- 最终 full agent evaluation 是否提升；
- 跨任务域、跨工具和跨 scaffold 泛化是否改善；
- invalid action、loop 和 premature termination 是否降低；
- 通用能力是否回退；
- 数据生产、训练和评测成本；
- reward hacking、leakage 或安全风险。

## 相关知识链接

- [[application/agents/agent|Agent]]
- [[application/agents/workflow-agent|Workflow Agent]]
- [[application/agents/planning|Planning]]
- [[application/tool-use/tool-calling|Tool Calling]]
- [[application/evaluation/benchmark|Benchmark]]
- [[training/mid-training/capability-injection|Capability Injection]]
- [[training/data-engineering/synthetic-data|Synthetic Data]]
- [[training/data-engineering/packing|Packing]]

## 参考资料

| 资料 | 对本计划的作用 | 知识库笔记 |
|---|---|---|
| APTBench | 提供无需完整后训练的 agentic potential 快速评测思路 | [[sources/papers/2025-aptbench-benchmarking-agentic-potential-of-base-llms|APTBench]] |
| Scaling Agents via Continual Pre-training | 提供在 mid-training 阶段注入 agent 行为分布的证据 | [[sources/papers/2025-scaling-agents-via-continual-pre-training|Scaling Agents via Continual Pre-training]] |
| Kimi-Dev | 说明 workflow skill prior 可降低后续 agent 适配难度 | [[sources/papers/2025-kimi-dev-agentless-training-as-skill-prior-for-swe-agents|Kimi-Dev]] |
| daVinci-Dev | 提供 agent-native trajectory reconstruction 与环境反馈数据的工程范式 | [[sources/papers/2026-davinci-dev-agent-native-mid-training-for-software-engineering|daVinci-Dev]] |
| Internalizing the Future | 强调 look-ahead planning、reflection 与 confidence 需要在更早训练阶段形成 | [[sources/papers/2026-internalizing-the-future-world-model-agentic-training|Internalizing the Future]] |
| Qwen3-Coder-Next | 展示大规模 agent 数据、可执行环境、多 scaffold 和后训练组合的系统化工程经验 | [[sources/papers/2026-qwen3-coder-next-technical-report|Qwen3-Coder-Next]] |
