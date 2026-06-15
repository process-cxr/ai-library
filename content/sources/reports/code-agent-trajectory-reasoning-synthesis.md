---
title: "Code Agent Trajectory Reasoning Synthesis"
created: 2026-06-12
published: 2026-06-12
modified: 2026-06-12
type: source
status: processed
source_type: report
area: sources
tags:
  - source
  - report
  - code-agent
  - trajectory-data
  - synthetic-reasoning
  - post-training
aliases:
  - Code Agent Reasoning Synthesis
---

# 面向 Code Agent 轨迹 Reasoning 合成的调研

## 摘要

本文调研分析了在已有 Code Agent 多轮可观测轨迹基础上，为每一步 assistant output 合成单一训练用 thinking 的可行性。其目标是在保持既有 `gold_answer_t` 不变的前提下，为每个 assistant step 补齐一段可训练、可审计、与当前目标输出对齐的中间 reasoning，使目标 student model 在监督微调阶段更容易从当前 prefix 生成指定 gold answer。

为了更好地理解 Code Agent 多轮 reasoning 轨迹合成任务，本文将该任务定义为一种面向既有监督目标的中间监督信号构造问题。对第 `t` 个 assistant step，给定当前 step 之前模型真实可见的上下文 `prefix_≤t`，以及当前 step 已经存在的目标输出 `gold_answer_t`，目标是合成一段训练用 reasoning，记为 `reasoning_t`，使其在 student model 视角下对生成 `gold_answer_t` 具有增益。形式化地，该目标可表示为：

```text
reasoning_t^* = argmax_reasoning Utility_student(reasoning; prefix_≤t, gold_answer_t)
```

其中，`gold_answer_t` 在 Code Agent 场景下不仅包括自然语言回答，也包括 assistant visible output、tool name、tool arguments、完整 tool call 或 text + tool call 的组合。`gold_answer_t` 是 reasoning 合成的目标变量，而不是 future leakage。真正需要禁止的是当前 tool call 的返回结果、后续 tool result、后续 assistant 修正、后续测试结果以及最终任务成功信号。

主要结论是：现有 synthetic reasoning、rationale distillation、agent trajectory annotation 与 reasoning effort 相关研究均可提供部分启发，但尚不能直接覆盖当前场景。现有 rationale distillation 工作主要面向数学、逻辑、问答等静态任务；agent trajectory 工作更多记录 action / observation 序列，而较少系统研究如何为既有 gold agent step 补齐 hidden thinking。因此，结合现有研究和当前数据形态，本文提出一条主线方案：首先基于结构化 JSON 构造 step-level `prefix_≤t` 与 `gold_answer_t`，再通过 step 分类、decision atoms 生成、reasoning 文本化生成和多层质量控制获得多个 candidate `reasoning_t`，最后以 student 生成目标输出为导向进行筛选，获得适合指定 student model 的 selected `reasoning_t`。

# 一、任务目标与现有数据形态分析

## 1.1 任务背景

当前研究对象是已有 Code Agent 多轮可观测轨迹数据。该类数据通常包含用户请求、system instruction、assistant visible output、tool call、tool result、代码编辑、命令执行、子任务委派以及阶段性总结等事件。与普通问答数据不同，Code Agent 轨迹中的 assistant 输出并不只是自然语言回答，而是经常包含文件读取、代码搜索、代码修改、命令验证、任务拆分和工具调用等操作性行为。

在当前可获得的既有 Code Agent 数据中，agent 的完整行为路线已经存在，并已被处理为结构化 JSON 与训练 TXT 两种形态。但已有数据中的 assistant thinking 部分仍为空，或者只是压缩后的 reasoning summary。当前任务需要为每一个 assistant step 补齐一段合适的中间 reasoning，使其能够作为监督微调中的训练信号。

因此，本研究的核心任务不是“给定问题生成答案”，而是：

```text
在 gold answer 已知的条件下，为当前 prefix 合成一段 reasoning，使目标 student model 更容易生成该 gold answer。
```

其中，`gold answer` 在 Code Agent 场景下不只表示自然语言答案，也包括 assistant visible output、tool name、tool arguments、tool call sequence 或 text + tool call 的组合。

## 1.2 任务定义

本研究将 thinking 合成定义为一种面向既有监督目标的中间监督信号构造问题。对第 `t` 个 assistant step，给定当前 step 之前模型可见的上下文 `prefix_≤t`，以及当前 step 已经存在的目标输出 `gold_answer_t`，需要合成一段训练用 reasoning，记为 `reasoning_t`。形式化地，目标可表示为：

```text
reasoning_t^* = argmax_reasoning Utility_student(reasoning; prefix_≤t, gold_answer_t)
```

其中：

- `prefix_≤t` 表示当前 assistant step 之前模型真实可见的上下文；
- `gold_answer_t` 表示当前 step 的既有目标输出，可以包括 visible text、tool name、tool arguments 或完整 tool call；
- `reasoning_t` 表示待合成的训练用 thinking；
- `Utility_student` 表示该 reasoning 对目标 student model 生成 `gold_answer_t` 的帮助程度。

`reasoning_t` 的目标不是复原原始模型真实的隐藏思维，也不是让 teacher model 在当前 prefix 下重新自由求解下一步动作，而是构造一段能够在训练时桥接 `prefix_≤t` 与 `gold_answer_t` 的中间监督信号。

换言之，一段高质量的 `reasoning_t` 应使指定 student model 在当前 `prefix_≤t` 下更稳定地生成 `gold_answer_t`，而不是仅仅满足 `reasoning_t` 看起来像一段合理解释。因此，本任务中的 reasoning 合成应以 student 的学习效果为导向，而不是以 teacher 的自然语言解释流畅度为唯一目标。

## 1.3 Gold Answer 的角色与信息边界

在本任务中，`gold_answer_t` 是 reasoning 合成的目标变量，而不是需要隐藏的未来信息。teacher model 在合成阶段可以看到 `gold_answer_t`，因为当前任务的目标并非重新求解下一步动作，而是构造一段能够帮助 student 从 `prefix_≤t` 生成该目标输出的中间监督信号。如果不提供 `gold_answer_t`，teacher model 可能在相同 prefix 下生成另一条合理但不同的 agent 路线，导致合成出的 reasoning 与训练样本中的 gold output / tool call 不一致。

例如，在某个 prefix 下，gold step 可能是：

```text
read_file(target_file="parse_excel.py", offset=200, limit=40)
```

如果仅给出 prefix，teacher model 可能认为下一步应先使用 `grep_content` 搜索关键词，或者直接进入 `edit_file`。这些选择在局部上未必错误，但它们并不服务于当前训练样本所指定的 gold step。因此，在本研究中，`gold_answer_t` 需要作为合成目标提供给 reasoning generator。

同时，`gold_answer_t` 的可见性必须与未来观测严格区分。允许使用的信息仅包括 `prefix_≤t`、`gold_answer_t`、tool schema 和通用代码 / 工具知识；禁止使用的信息包括当前 tool call 的返回结果、后续 tool result、后续 assistant 修正、后续测试结果以及最终任务成功信号，从机制上避免 future leakage。

## 1.4 当前数据形态与 Pipeline 输入选择

当前数据已经具备开展 step-level reasoning 合成所需的基础条件，建议直接以结构化 JSON 作为合成 pipeline 的输入格式。目前的 JSON 数据完整保留了轨迹中的消息边界、角色信息、assistant visible output、tool call、tool arguments、tool result 以及事件顺序等结构化信息。这些信息对于 reasoning 合成至关重要，因为它们能够稳定支持逐 step 构造 `prefix_≤t`、提取 `gold_answer_t`，并进行 action consistency、argument grounding 和 future leakage 检查。

从合成任务本身来看，reasoning 的生成依赖于对轨迹状态的精确理解，而不是对最终训练格式的还原。因此，保留原始结构信息比使用经过扁平化处理的数据形式更有优势。结构化 JSON 不仅能够降低解析误差，还能够为后续质量控制、审计和错误分析提供更可靠的数据基础。

基于 JSON 的 reasoning 合成流程可以概括为：

```text
结构化 JSON
  → step-level prefix / gold_answer 构造
  → reasoning 合成
  → 质量控制与筛选
  → 输出 selected reasoning
```

从数据可用性角度看，当前任务具备以下优势：

1. **Gold trajectory 已经存在**：不需要重新生成 agent 路线，只需为既有路线补齐中间 reasoning。
2. **Assistant step 与 tool event 可对齐**：每个 assistant step 的 visible output、tool name、tool arguments 和后续 tool result 可用于构造训练目标和质量检测信号。
3. **工具行为具有明确语义**：`read_file`、`grep_content`、`edit_file`、`run_command`、`todo_write`、`delegate_subagent` 等工具天然对应不同的局部决策类型，有利于设计 step-specific reasoning grammar。
4. **轨迹结构信息完整保留**：消息顺序、工具调用关系和环境反馈均可直接获取，有利于构造高质量的 step-level context。
5. **代码 agent 场景具有较强可验证性**：文件路径、函数名、命令、编辑片段和工具参数均可作为 grounding 对象，用于检查 reasoning 是否真正支持当前 `gold_answer_t`。

## 1.5 本阶段主目标与非目标

本阶段主目标是构建一套可行的单一 thinking 合成流程，使每个 assistant step 最终获得一段 selected `reasoning_t`，并将其回填至训练数据中。

其中，low / medium / high effort controllable 数据可以作为后续扩展方向。其基本思路是基于同一组 reasoning atoms 派生不同信息预算下的 reasoning 版本，并通过显式 effort marker 条件化训练。但该扩展会引入额外的数据展开、控制 token 设计和评估复杂度，因此不作为当前阶段的核心目标。

综上，当前阶段的任务边界可以概括为：

```text
给定 prefix_≤t 和 gold_answer_t，
合成一段 prefix-grounded、gold-answer-supportive、future-leakage-free、student-oriented 的 reasoning_t，
用于补齐既有 Code Agent 训练轨迹中的 assistant thinking。
```

# 二、可行的合成流程与筛选方案

## 2.1 总体方法

基于上述任务定义，本文设计的主线合成策略是：在当前 `gold_answer_t` 已知的前提下，为当前 `prefix_≤t` 构造一段中间 reasoning，使 student model 更容易生成该 gold answer。该方法与普通 CoT 生成存在关键区别。

普通 CoT 生成通常是：

```text
question → reasoning → answer
```

而当前方法是：

```text
prefix_≤t + gold_answer_t → reasoning_t
```

其中，`gold_answer_t` 不是推理之后才发现的答案，而是训练样本中已经存在的监督目标。reasoning 的作用是服务于该监督目标，使训练样本从单纯的输入输出映射变为带中间监督的轨迹样本。

该方法的核心约束包括：

```text
Target-conditioned:
  reasoning_t 必须服务于 gold_answer_t。

Prefix-grounded:
  reasoning_t 中的事实性声明应能由 prefix_≤t、gold_answer_t 或 tool schema 支撑。

Environment-aware:
  reasoning_t 应反映当前轨迹阶段下的局部决策依据，而非静态问题求解过程。

Future-leakage-free:
  reasoning_t 不得引用 current tool result、future observation 或 final outcome。

Student-oriented:
  reasoning_t 应提升 student 生成 gold_answer_t 的稳定性。
```

## 2.2 Reasoning 生成流程：Step 分类、Decision Atoms 与文本化生成

Reasoning 生成阶段可以被视为一个连续流程，而不是若干彼此独立的策略。其核心链路如下：

```text
Step 分类
  → 确定 reasoning grammar
  → 生成 decision atoms
  → 将 atoms 文本化为 candidate reasoning_t
```

其中，step 分类和 decision atoms 都不是最终产物，而是为了约束 reasoning 生成过程，避免 teacher model 直接自由发挥，生成空泛、后验或与当前 action 不一致的 thinking。

### 2.2.1 Step 分类用于确定 reasoning grammar

在生成 reasoning 之前，应先对 assistant step 进行分类。分类的目的不是训练分类器，而是根据当前 step 的行为类型，确定 reasoning 应该包含哪些要素。

建议采用“结构分类 + 行为分类”两层体系。

结构分类描述 assistant output 的外部形态：

```text
tool_only
text_only
text+tool
multi_tool
```

行为分类描述当前 step 的决策性质：

```text
inspect       信息读取与上下文获取
search        定位目标位置或模式
analyze       基于已有信息进行判断
plan          制定后续执行计划
todo_manage   更新任务列表或状态管理
delegate      委派子任务
edit          修改代码或文件
execute       执行命令、脚本或工具操作
verify        验证修改结果
summarize     汇总阶段成果或最终结果
recover       错误恢复与重新规划
communicate   向用户解释、确认或同步状态
other
```

分类结果会直接决定后续 decision atoms 的生成模板。例如：

- 若当前 step 是 `inspect` 类型，reasoning 应强调当前缺失什么信息、为什么需要读取该文件或区域、读取结果将为后续什么操作提供上下文。
- 若当前 step 是 `edit` 类型，reasoning 应强调当前已确认的上下文、修改目标、修改范围为何足够，以及 tool arguments 中 old / new content 的依据。
- 若当前 step 是 `verify` 类型，reasoning 应强调当前需要验证什么、为什么选择该命令、验证结果将如何决定后续步骤。

因此，step 分类不是独立分析模块，而是 reasoning 生成的路由层。它将同一个通用任务：

```text
为当前 gold_answer_t 生成 reasoning
```

具体化为不同类型的生成要求：

```text
为当前 inspect step 生成 reasoning
为当前 edit step 生成 reasoning
为当前 verify step 生成 reasoning
```

这能够显著降低 teacher generation 的自由度。

### 2.2.2 Decision Atoms 用于承接分类结果

在知道当前 step 的类型后，不应直接让 teacher model 生成完整自然语言 thinking，而应先生成结构化 decision atoms。Decision atoms 是当前 gold answer 的原子化中间依据，可以理解为 reasoning 的最小组成单元。

建议 atom 类型包括：

```text
goal：当前局部目标
evidence：prefix 中已有证据
constraint：系统、用户或工具约束
unknown：当前尚未确认的信息
tool_choice：为什么选择该工具
arg_choice：为什么选择这些参数
check：下一步验证或检查意图
risk：当前操作的风险或失败条件
```

其中，不同 step 类型会启用不同的 atom 组合。例如：

```text
inspect step:
  goal + unknown + tool_choice + arg_choice

edit step:
  goal + evidence + constraint + arg_choice + risk

verify step:
  goal + evidence + tool_choice + check
```

这样，step 分类就实际参与了 atoms 生成，而 atoms 又为后续 reasoning 文本化提供结构化骨架。一个 assistant step 的 reasoning 可表示为：

```text
A_t = {a_1, a_2, ..., a_n}
```

其中，`A_t` 是支持当前 `gold_answer_t` 的 decision atoms 集合。合成目标不是生成尽可能多的 atoms，而是找到足以支持当前 gold answer 的最小充分原子集合。

Decision atoms 的引入具有三项作用。

第一，提升可控性。自然语言 reasoning 由 atoms 实现，而不是由 teacher model 自由发挥。

第二，便于质量控制。可以检查每个 atom 是否有证据、是否支持 tool arguments、是否引入未来信息。

第三，支持未来扩展。若后续需要 low / medium / high effort 数据，可基于同一组 atoms 派生不同信息预算下的 reasoning 版本，而不需要重新生成互相矛盾的 reasoning。

### 2.2.3 Reasoning 文本化生成

在完成 step 分类与 decision atoms 结构化生成后，需要将结构化依据转化为可写入 assistant thinking 槽位的自然语言 reasoning。本阶段称为 reasoning 文本化生成，其目标不是继续扩展新的事实或重新规划 action，而是将已经确定的局部目标、上下文证据、工具选择依据和参数依据组织为一段简洁、连贯、符合 Code Agent 风格的候选 reasoning。

在当前主线中，每个 assistant step 可以生成一个或多个候选 `reasoning_t`，并在后续质量控制阶段选择最终写入训练数据的 selected `reasoning_t`。因此，本阶段的输出不是最终训练数据，而是供筛选模块使用的候选 reasoning 文本。

候选 reasoning 应满足以下约束：

```text
面向当前 gold answer；
短而明确；
不做后验总结；
不引入未来 observation；
不过度展开无关替代路线；
不堆砌泛化原则；
保持 Code Agent 工程风格。
```

推荐默认采用中等信息密度，即包含：

```text
当前局部目标；
关键 prefix 证据；
当前 tool / action 选择理由；
必要的参数依据。
```

例如，当前 gold answer 为：

```text
read_file(target_file="parse_excel.py", offset=200, limit=40)
```

对应分类与 atoms 可以是：

```json
{
  "semantic_class": "inspect",
  "decision_atoms": [
    {
      "type": "goal",
      "text": "需要获取目标函数附近的实现上下文"
    },
    {
      "type": "tool_choice",
      "text": "读取文件片段比直接编辑更安全，因为当前还需要确认旧逻辑"
    },
    {
      "type": "arg_choice",
      "text": "文件路径和 offset 来自当前已定位到的目标区域"
    }
  ]
}
```

较合适的 `reasoning_t` 是：

```text
当前已经定位到目标函数附近，下一步需要先读取其实现，确认旧逻辑和上下文后再构造后续 edit_file 的精确替换片段。
```

不合适的 `reasoning_t` 是：

```text
后续读取结果会显示该函数确实包含 skip-done 逻辑，因此现在读取这里是正确的。
```

后者引用了当前 tool result 才能知道的信息，属于 future leakage，不应进入训练数据。

## 2.3 筛选策略与质量控制

质量控制是该任务的核心。Synthetic reasoning 一旦进入 SFT 数据，会直接影响 student model 的行为。如果 reasoning 存在未来泄漏、动作不一致、参数依据错误或明显偏离 Code Agent 风格等问题，可能导致模型学习到错误的决策桥接方式。

从整体流程来看，筛选可以简化为两个阶段：

```text
第一阶段：Teacher-side Quality Control
第二阶段：Student-side Utility Selection
```

前者负责保证生成结果本身是合法、合理且符合任务约束的；后者负责从多个候选 reasoning 中选择对目标 student 最有帮助的版本。

### 2.3.1 Teacher-side Quality Control

在生成阶段，teacher model 不应只生成单个 reasoning，而应生成一定数量的候选 reasoning，然后通过质量控制筛除明显不合格的样本。这一阶段主要关注 reasoning 本身是否满足基本约束，而不直接考虑其对 student 的训练效果。

建议检查以下内容：

| 检查项 | 目标 |
|---|---|
| Schema validity | 输出结构合法 |
| Prefix grounding | 内容能够回链到当前 prefix |
| Gold-answer consistency | reasoning 支持当前 gold answer |
| Argument grounding | tool arguments 有合理依据 |
| Future leakage | 不引用当前 result 或未来 observation |
| Style compliance | 符合 Code Agent thinking 风格 |

其中，Future Leakage 是最高优先级检查项。

允许使用的信息包括：

```text
prefix_≤t
gold_answer_t
tool schema
通用代码与工具知识
```

禁止使用的信息包括：

```text
current tool result
future observation
future tool result
future assistant correction
final outcome
```

只有通过上述检查的候选 reasoning 才进入下一阶段。

### 2.3.2 Student-side Utility Selection

通过 Teacher-side Quality Control 后，通常仍会保留多个候选 reasoning。此时需要从 student 学习效果的角度选择最终版本。

核心思想是比较 student 在有无 reasoning 条件下对当前 gold answer 的生成倾向变化：

```text
TeachingGain(reasoning_t) =
log P_student(gold_answer_t | prefix_≤t, reasoning_t)
-
log P_student(gold_answer_t | prefix_≤t)
```

若 `TeachingGain` 为正，说明该 reasoning 能够提升 student 生成当前 gold answer 的概率；若多个候选 reasoning 均通过质量控制，则优先选择 TeachingGain 更高的版本。

最终筛选逻辑可以概括为：

```text
Teacher 生成多个候选 reasoning
        ↓
Teacher-side Quality Control
        ↓
保留合格候选
        ↓
Student-side Utility Selection
        ↓
选择 TeachingGain 最高的 reasoning
        ↓
得到 selected_reasoning_t
```

这种两阶段设计能够将“生成质量是否合格”和“是否真正有助于 student 学习”两个问题明确分离，使整体流程更加清晰，也更符合当前任务目标。

# 三、Reasoning 合成输入格式设计

前文讨论了任务定义、合成流程、质量筛选与相关研究。为了将上述方案落地，需要进一步定义能够送入 teacher model 的标准化输入格式。该输入格式不仅要包含 `prefix_≤t` 和 `gold_answer_t`，还应显式包含 step 分类结果、reasoning grammar 和 decision atoms 生成要求。否则，第二部分提出的 step 分类与 atoms 结构就无法真正进入 pipeline。

## 3.1 Step-level Synthesis Record

对于轨迹中的第 `t` 个 assistant step，建议从原始 JSON 中抽取如下结构：

```json
{
  "trajectory_id": "string",
  "step_id": "string",
  "task_id": "string",

  "prefix": {
    "messages": [],
    "tool_calls": [],
    "tool_results": []
  },

  "current_step": {
    "assistant_type": "tool_only|text+tool|text_only|multi_tool",
    "gold_answer": {
      "visible_text": "string",
      "tool_call": {
        "name": "string",
        "arguments": {}
      }
    }
  },

  "tool_schema": [],

  "step_classification": {
    "structural_class": "tool_only|text+tool|text_only|multi_tool",
    "semantic_class": "inspect|search|analyze|plan|todo_manage|delegate|edit|execute|verify|summarize|recover|communicate|other"
  },

  "reasoning_grammar": {
    "required_atoms": ["goal", "tool_choice", "arg_choice"],
    "optional_atoms": ["evidence", "unknown", "risk"],
    "style_constraints": [
      "prefix-grounded",
      "gold-answer-supportive",
      "future-leakage-free",
      "concise"
    ]
  },

  "generation_targets": {
    "generate_decision_atoms": true,
    "generate_candidate_reasoning": true,
    "num_candidates": 3
  }
}
```

其中：

- `prefix` 仅包含当前 step 之前真实可见的信息；
- `gold_answer` 为当前 step 的监督目标；
- `tool_schema` 提供工具能力描述；
- `step_classification` 保存结构分类与行为分类；
- `reasoning_grammar` 由 `semantic_class` 自动派生，用于约束 atoms 类型和文本化方式；
- `generation_targets` 指定 teacher 需要先生成 atoms，再生成候选 reasoning；
- 不包含当前 step 执行后的 tool result；
- 不包含任何 future observation。

对应关系如下：

```text
prefix
    ↓
gold_answer
    ↓
step_classification
    ↓
reasoning_grammar
    ↓
decision_atoms
    ↓
candidate reasoning_t
```

因此，该结构不只是 reasoning synthesis 的标准输入单元，也显式承接了第二部分中的 step 分类与 atoms 机制。

## 3.2 Teacher Generation Prompt 结构

该 prompt 应明确任务性质：teacher model 不是在决定下一步做什么，而是在为当前 gold answer 合成训练用 reasoning。

示例 prompt 可包含以下部分：

```text
[Task]

为当前 assistant step 生成训练用 reasoning。

你不是在决定下一步 action。
当前 gold answer 已经给定。
你的任务是生成一段能够帮助 student model 从 prefix 生成该 gold answer 的 reasoning。

[Step Classification]

structural_class: ...
semantic_class: ...

[Reasoning Grammar]

required_atoms: ...
optional_atoms: ...
style_constraints: ...

[Prefix]

...

[Current Step Target]

gold_answer:
...

tool_schema:
...

[Output Format]

{
  "decision_atoms": [
    {
      "type": "goal|evidence|constraint|unknown|tool_choice|arg_choice|check|risk",
      "text": "...",
      "source": "prefix|gold_answer|tool_schema|general_knowledge"
    }
  ],
  "candidate_reasoning": [
    "..."
  ]
}
```

这种格式能够显式告诉 teacher：

```text
你不是在决定下一步做什么，
而是在生成一段能够帮助 student 生成当前 gold answer 的 reasoning。
```

## 3.3 Reasoning Candidate Record

Teacher 生成后，建议保留候选层，而不是直接覆盖原始数据。候选层用于后续质量控制与 student-side utility selection。

候选记录可设计为：

```json
{
  "trajectory_id": "string",
  "step_id": "string",

  "step_classification": {
    "structural_class": "text+tool",
    "semantic_class": "inspect"
  },

  "gold_answer": {
    "visible_text": "Now let me read the target function:",
    "tool_call": {
      "name": "read_file",
      "arguments": {
        "target_file": "parse_excel.py",
        "offset": 200,
        "limit": 40
      }
    }
  },

  "decision_atoms": [
    {
      "atom_id": "a1",
      "type": "goal",
      "text": "需要读取目标函数附近实现以确认旧逻辑",
      "source": "prefix"
    },
    {
      "atom_id": "a2",
      "type": "tool_choice",
      "text": "read_file 适合获取局部代码上下文",
      "source": "tool_schema"
    },
    {
      "atom_id": "a3",
      "type": "arg_choice",
      "text": "target_file 与 offset 来自当前已定位到的目标区域",
      "source": "gold_answer"
    }
  ],

  "candidates": [
    {
      "candidate_id": "c1",
      "reasoning": "当前已经定位到目标函数附近，需要先读取其实现，确认旧逻辑和上下文后再构造后续精确修改。",
      "generation_metadata": {
        "model": "teacher_model_name",
        "temperature": 0.7,
        "seed": 123
      }
    }
  ]
}
```

这样，后续筛选模块不仅能判断候选 reasoning 是否合理，还能回溯它依赖了哪些 decision atoms、这些 atoms 是否由 prefix / gold answer / tool schema 支撑。

## 3.4 Selected Reasoning 回填

经过 Teacher-side Quality Control 与 Student-side Utility Selection 后，每个 step 最终得到一个 selected `reasoning_t`。

此时再将其回填到训练 TXT：

```text
<|im_start|>assistant
<think>
当前已经定位到目标函数附近，下一步需要先读取其实现，确认旧逻辑和上下文后再构造后续 edit_file 的精确替换片段。
</think>
Now let me read the target function:
<tool_call>
...
</tool_call>
<|im_end|>
```

## 3.5 总体数据流

综合上述设计，完整数据流可以表示为：

```text
原始轨迹 JSON
    ↓
step extraction / alignment
    ↓
step classification
    ↓
reasoning grammar 派生
    ↓
step-level synthesis record
    ↓
decision atoms generation
    ↓
candidate reasoning generation
    ↓
teacher-side quality control
    ↓
student-side utility selection
    ↓
selected reasoning_t
    ↓
训练 TXT 回填
```

该结构能够清晰区分：

```text
原始轨迹层
Step 分类与 grammar 层
Decision atoms 层
Reasoning 合成层
质量控制层
训练数据层
```

并为后续扩展多候选生成、judge 筛选、student-fit 评估以及 effort-controllable reasoning 提供统一的数据接口。

# 四、相关研究与适配性分析

## 4.1 Synthetic Reasoning 与 Rationale Distillation

Synthetic reasoning 与 rationale distillation 研究表明，由强模型生成的解释或推理链可以作为 student model 的附加监督信号。典型工作证明，teacher-generated rationales 不仅能够补充最终标签，还能够为小模型提供额外的中间推理结构，从而提升训练效率和泛化能力。

这类研究对当前任务具有直接启发：如果仅使用 `prefix_≤t → gold_answer_t` 的监督信号，student model 只能学习输入与输出之间的黑箱映射；而加入 `reasoning_t` 后，训练样本变为：

```text
prefix_≤t → reasoning_t → gold_answer_t
```

该结构显式提供了中间监督，使 student model 能够学习“为什么当前 prefix 应导向该 gold answer”。

然而，现有 rationale distillation 工作与当前 Code Agent 场景存在显著差异。多数相关研究面向数学、逻辑、常识问答或分类任务，其样本结构通常是：

```text
question → rationale → answer
```

而 Code Agent 轨迹的结构是：

```text
prefix state → hidden thinking → visible output / tool call → environment observation
```

前者主要处理静态推理问题，后者处理多轮、工具增强、环境交互式决策。Code Agent thinking 还必须面对工具选择、参数 grounding、文件路径、代码片段、命令验证和 future leakage 等问题。因此，rationale distillation 的核心结论可以支持“合成 reasoning 作为监督信号是有价值的”，但不能直接提供当前任务所需的 step-level agent thinking 合成方案。

## 4.2 Agent Trajectory、Tool-use 与 Code Agent 数据

Agent trajectory 研究与当前任务最接近。ReAct 等工作表明，在交互式任务中，将 reasoning 与 action 交织起来可以帮助模型跟踪计划、调用工具、处理异常并提升可解释性。Web agent、tool-use agent 和 code agent 相关 benchmark 也进一步说明，agent 行为不能只通过最终答案评价，而需要关注 action-observation 轨迹。

这些工作对当前任务提供了两个重要启发。

第一，agent 的中间行为具有结构性。一个 tool call 不只是普通 token 输出，而是与环境状态、工具能力、局部目标和后续 observation 相关的决策。

第二，reasoning 与 action 的关系是 agent 学习的核心。若 student 只学习 action 而没有中间 thinking，可能更容易形成表层模仿；若 reasoning 能正确连接 prefix 与 gold answer，则更可能提升 tool choice、argument selection 和错误恢复能力。

然而，现有 agent trajectory 工作通常关注 action 记录、action prediction、benchmark evaluation 或 trajectory generation，较少系统研究如下问题：

```text
给定已有 gold action sequence，如何为每一步补齐 hidden reasoning；
如何保证 reasoning 服务于当前 gold answer；
如何避免 reasoning 引用 action 之后的 observation；
如何检查 reasoning 是否真正支持 tool arguments；
如何将 reasoning 作为 SFT 数据回填到已有训练格式。
```

部分 action-rationale 相关工作会为 action 生成 textual rationale，但通常更接近 posterior explanation，并不严格区分当前可见 prefix、当前 gold answer 与未来 observation。相比之下，本文强调：`gold_answer_t` 可以作为合成目标，但当前 tool result 与 future observation 不应进入 generation context。这一点是当前任务与一般 action rationalization 的关键区别。

## 4.3 Reasoning Effort 与 Budget Control

近年来，reasoning effort、thinking budget 和 adaptive thinking 等机制逐渐出现在模型接口和研究系统中。相关趋势说明，不同任务和不同步骤可能需要不同程度的 reasoning；推理预算不必对所有样本统一分配。

该方向为后续 low / medium / high effort controllable 数据提供了理论背景。若未来需要训练一个能够根据 effort marker 控制思考深度的 student model，可以基于同一组 decision atoms 派生不同信息预算下的 reasoning 版本。

但在当前阶段，effort control 不应作为主目标。原因在于：

1. 当前首要问题是验证单一 thinking 轨迹是否能够被稳定合成并提升 SFT 数据质量；
2. 多 effort 会引入额外条件变量，使同一 prefix 对应多种 reasoning 输出；
3. 若缺少显式 effort marker，多 effort 训练会造成标签歧义；
4. effort controllability 需要额外评估体系，容易稀释主线目标。

因此，本文将 effort control 作为后续扩展方向，而不纳入当前主合成方案。

## 4.4 小结：现有研究的不足与当前任务的研究空白

综合来看，已有研究分别覆盖了以下局部问题：

```text
rationale distillation：证明 teacher rationale 可作为附加监督；
agent trajectory：证明 action-observation 轨迹是 agent 行为的核心；
effort control：说明 reasoning budget 可以作为扩展控制变量。
```

但尚缺少直接面向如下完整设定的系统方法：

```text
给定已有 Code Agent gold trajectory；
对每个 assistant step；
使用 prefix_≤t 和 gold_answer_t；
合成一段 hidden reasoning_t；
使其帮助 student 生成当前 gold answer；
同时保证 reasoning prefix-grounded、gold-answer-supportive、future-leakage-free。
```

该研究空白正是当前方案的价值所在。当前任务不是简单地“生成一段 CoT”，而是为既有 agent gold step 构造一种可训练的中间监督信号。其关键挑战不在于自然语言生成本身，而在于如何同时满足目标输出对齐、上下文 grounding、工具参数支持、时序边界约束和 student 可学习性。

参考资料：

1. **STaR: Self-Taught Reasoner**：该工作提出通过模型自生成 rationale 并在正确答案条件下迭代改进推理数据，为 answer-conditioned rationale generation 提供了重要参考。
2. **Distilling Step-by-Step**：该工作表明，teacher-generated rationales 可以作为小模型训练的中间监督信号，在较少标注数据下提升 student model 表现。
3. **ReAct: Synergizing Reasoning and Acting in Language Models**：该工作提出 reasoning 与 action 交织的 agent 范式，是理解 tool-use agent 轨迹结构的重要基础。
4. **ToolBench / StableToolBench**：该系列工作提供了大规模工具调用数据与稳定工具评测环境，对 tool-use reasoning、API 调用和 action-result 轨迹构造具有参考价值。
5. **SWE-agent / SWE-bench 相关工作**：该方向将 agent 应用于真实软件工程任务，并产生可记录、可回放的代码修改轨迹，对 Code Agent 数据结构和 step-level trajectory 研究具有直接价值。
6. **WebLINX / Mind2Web / WebArena**：这些工作主要面向 web agent 多轮导航与动作预测，提供了多轮 action trajectory、网页状态与用户指令之间的结构化对应关系，可作为非代码 agent 域的补充参考。

## 相关知识链接

- [[training/post-training/sft|SFT]]
- [[training/post-training/knowledge-distillation|Knowledge Distillation]]
- [[application/agents/agent|Agent]]
- [[application/tool-use/tool-calling|Tool Calling]]
- [[application/evaluation/llm-as-judge|LLM-as-a-Judge]]
