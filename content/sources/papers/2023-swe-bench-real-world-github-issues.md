---
title: "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?"
created: 2026-10-20
published: 2026-10-20
modified: 2026-10-20
type: source
status: processed
source_type: paper
area: sources
tags:
  - source
  - paper
  - software-engineering
  - coding-agent
  - evaluation
  - benchmark
  - long-context
aliases:
  - SWE-bench
source_url: https://arxiv.org/abs/2310.06770
paper_date: "2023-10"
paper_order: "06770"
---

# SWE-bench: Can Language Models Resolve Real-World GitHub Issues?

## 基本信息

- 标题：[SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://arxiv.org/abs/2310.06770)
- 作者：Carlos E. Jimenez、John Yang、Alexander Wettig、Shunyu Yao、Kexin Pei、Ofir Press、Karthik Narasimhan
- 机构/来源：Princeton University、Princeton Language and Intelligence、University of Chicago
- 发表：ICLR 2024；arXiv:2310.06770v3
- 日期：2023-10
- 代码/数据：[SWE-bench repository](https://github.com/SWE-bench/SWE-bench)、[项目主页](https://www.swebench.com/)
- 相关主题：[[application/evaluation/benchmark|Benchmark]]、[[application/agents/agent|Agent]]、[[application/tool-use/tool-calling|Tool Calling]]、[[sources/papers/2025-kimi-dev-agentless-training-as-skill-prior-for-swe-agents|Kimi-Dev]]、[[sources/papers/2026-tqs-quality-not-just-outcome-coding-agent-trajectories|Quality, Not Just Outcome]]

## 这篇论文解决什么问题

早期代码生成 benchmark 往往给模型一个短函数和自然语言描述，模型只需要补全一段局部代码。这样的任务适合测代码语法和局部语义，但很难回答一个更接近真实研发的问题：

> 模型能否在一个真实代码仓库中理解 issue，找到相关实现，修改一个或多个文件，并通过原项目的测试？

SWE-bench 把这个问题变成可执行的 benchmark。每个任务来自真实开源 Python 项目的 GitHub issue 和后来解决该 issue 的 merged pull request（PR，拉取请求）。模型看到 issue 和某个历史版本的代码库，输出一个 patch；评测系统将 patch 应用到代码库中，安装依赖并运行与该任务相关的测试。

它的关键变化是：**评测对象从“生成一段代码”变成“完成一次受测试约束的软件修改”。** 这使任务同时涉及问题理解、代码定位、跨文件推理、补丁格式、回归约束和执行验证。

## 核心主张

### 1. 真实仓库级任务会暴露传统代码 benchmark 看不到的能力缺口

SWE-bench 的任务通常需要在数千个文件中找到少数相关代码，并理解函数、类、模块之间的依赖关系。参考 PR 平均修改 1.7 个文件、3.0 个函数和 32.8 行代码；这已经超出单函数补全的范围。

论文所读版本报告，在 BM25 检索条件下，Claude 2 的 resolved rate 只有约 1.96%。这说明当时最强模型也只能解决少数最容易的真实 issue。论文后续更新的表格加入了更晚的模型，但整体结论仍然是：**真实软件工程修复远比短代码生成困难。**

### 2. 检索是必要环节，但“给更多上下文”并不能自动解决定位问题

代码库平均约 438K 行，通常无法完整放进模型上下文。论文比较 BM25 稀疏检索和 oracle 检索，并发现：

- BM25 在较大的上下文预算下可以找回更多参考文件，但模型的 resolved rate 反而可能下降；
- 仅仅扩大上下文，会把更多无关代码带给模型，增加定位和注意力分配难度；
- 把 oracle 文件中未修改部分折叠，只保留修改位置附近的代码后，GPT-4 和 Claude 2 的效果明显提升。

因此，SWE-bench 测到的不只是“代码生成能力”，还测到 **repository localization、context selection 和 long-context distraction**。

### 3. 可执行测试让评测从表面相似度转向行为结果

模型不需要复现原 PR 的具体写法。只要生成的 patch 能应用，并让任务要求的 fail-to-pass 测试和原有 pass-to-pass 测试都通过，就算解决任务。

这种判定允许多个正确实现，同时检查两件事：

1. 模型是否真正修复了 issue 指向的行为；
2. 修改是否破坏了原有功能。

但它仍然不是完整的软件工程质量评测。测试没有覆盖的回归、代码可读性、架构质量、性能和维护成本，不能由 resolved rate 单独判断。

## Benchmark 是怎样构造出来的

### 三阶段筛选流程

论文从 12 个流行的开源 Python 仓库中爬取约 93,139 个 PR，经过三阶段筛选，得到 2,294 个 SWE-bench 任务：

```text
merged PRs from popular Python repositories
  -> issue linkage + test-file changes
  -> apply patches and execute repository tests
  -> keep tasks with reliable fail-to-pass behavior
  -> 2,294 benchmark instances
```

#### 第一阶段：仓库和 PR 收集

作者选择维护较好、测试覆盖相对充分的流行 Python 仓库，收集 merged PR 以及对应的 base commit。base commit 是 issue 被解决之前的代码状态，也是模型任务的起点。

#### 第二阶段：属性筛选

保留同时满足以下条件的 PR：

- PR 关联了一个 GitHub issue；
- PR 修改了仓库中的测试文件。

修改测试文件并不意味着测试一定完善，但它提供了一个可执行的信号：提交者很可能增加或调整了用于验证修复的测试。

#### 第三阶段：执行筛选

作者把测试补丁和非测试代码补丁分别应用到 base commit 上，并检查：

- 依赖是否能安装；
- 测试是否能运行；
- 至少有一个测试从 fail 变成 pass；
- 没有因为环境或运行时错误产生无法解释的结果。

只有通过这些检查的候选才进入最终数据集。93,139 个 PR 最终留下 2,294 个任务，说明“存在 issue、PR 和测试修改”还远远不够，必须经过实际执行才能成为可用 benchmark。

### 一个任务包含什么

可以把任务实例抽象成：

$$
\mathcal T=(C, P, \delta, T, \text{environment}),
$$

其中：

- $C$：base commit 对应的代码库状态；
- $P$：由 issue 标题、正文和相关评论组成的问题描述；
- $\delta$：从原 PR 提取出的参考 solution patch；
- $T$：从 PR 中提取出的测试 patch；
- `environment`：对应仓库版本的 Python、依赖和安装命令。

模型输入不是原始完整仓库的纯文本，而是 issue 加上评测系统选出的相关文件和文档。模型输出的是预测 patch $\hat\delta$。

### 测试标签的作用

作者把测试结果按照 patch 前后状态分成几类：

- **FAIL TO PASS**：应用 solution 后从失败变为通过的测试，直接对应 issue 是否被修复；
- **PASS TO PASS**：solution 前后都通过的测试，用来检查原有功能是否保持；
- **FAIL TO FAIL** 和 **PASS TO FAIL**：在最终任务判定中不作为需要通过的目标集合。

任务只有在所有 FAIL TO PASS 与 PASS TO PASS 测试都通过时才算 resolved。这个设计比只运行新增测试更严格，因为一个 patch 可能“修好”目标行为，却同时破坏旧功能。

## 任务规模与难度

论文给出的微平均统计如下：

| 属性 | 平均值 | 最大值 |
|---|---:|---:|
| issue 描述长度 | 195.1 words | 4,477 |
| 非测试文件数 | 3,010 | 5,890 |
| 非测试代码行数 | 438K | 886K |
| 参考 patch 修改行数 | 32.8 | 5,888 |
| 参考 patch 修改文件数 | 1.7 | 31 |
| 参考 patch 修改函数数 | 3.0 | 36 |
| fail-to-pass 测试数 | 9.1 | 1,633 |
| 测试总数 | 120.8 | 9,459 |

这些数字解释了它为什么不是普通的 code completion：issue 本身通常不算长，但代码库和测试空间很大，真正困难的部分是从大量可见上下文中定位、理解和修改少数关键位置。

论文还提供了 SWE-bench Lite，一个包含 300 个任务的子集，偏向更自包含的功能性 bug fix，用于降低早期系统的评测成本。Lite 更适合快速迭代，但不能替代完整 benchmark 的仓库级难度。

## 评测协议：从模型输出到 resolved

对每个任务，评测流程可以写成：

```text
checkout base commit
  -> activate version-specific environment
  -> install repository
  -> apply test patch
  -> apply model prediction patch
  -> repair patch formatting when possible
  -> run repository test command
  -> compare test statuses with ground truth
```

如果模型 patch 无法应用，系统会尝试删除多余上下文行并重新计算 patch header；修复后仍无法应用，或测试命令失败，任务得分为 0。若测试运行成功，则把结果解析成 test-to-status mapping，与 ground truth 中 FAIL TO PASS 和 PASS TO PASS 的集合比较。

报告中需要区分两个指标：

| 指标 | 含义 |
|---|---|
| `apply` | 预测 patch 能否成功应用到目标代码库并进入测试阶段 |
| `resolved` | patch 应用后，目标测试和需要保持通过的测试是否全部通过 |

`apply` 高而 `resolved` 低，通常说明模型能生成形式上可应用的修改，但修改逻辑不足以完成 issue，或引入了回归。`resolved` 才是论文的主要任务完成指标。

## SWE-Llama：论文提供的开放模型基线

作者还发布了 SWE-bench-train，并用它对 CodeLlama-Python 7B 和 13B 做监督微调，得到 SWE-Llama 7B 和 SWE-Llama 13B。

### 训练数据与防泄漏设计

训练集包含约 19,000 个 issue-PR pairs，来自另外 37 个 Python 仓库，与评测集的 12 个仓库不重合。训练时不要求 PR 修改测试文件，因此规模更大，但任务仍然使用 issue、相关代码和 gold patch 作为监督。

由于长序列训练成本高，作者只保留不超过 30,000 tokens 的训练序列，实际有效训练样本约 10,000 个。为了节省显存，只对 attention sublayer 的权重使用 LoRA（Low-Rank Adaptation，低秩适配），没有更新完整模型参数。

训练目标是生成 gold patch：

```text
issue + relevant code files + instructions
  -> supervised fine-tuning
  -> gold patch
```

这和今天常见的交互式 SWE agent 训练不同。论文的 SWE-Llama 学的是“看到任务和代码后生成补丁”，没有把搜索、编辑、运行测试、读取错误和再次修改组织成多轮环境交互。

## 检索设置与实验结果

### BM25 与 oracle retrieval

BM25（Best Matching 25）是基于词项匹配的经典文本检索方法。论文使用它从代码仓库中选择文件，因为当时的代码库和自然语言 query 很长，作者认为 dense retrieval 不适合直接处理这种设置。

实验比较两种检索方式：

- **Sparse / BM25 retrieval**：根据 issue 和代码文本选择尽可能多的文件，受最大上下文长度限制；
- **Oracle retrieval**：直接提供参考 patch 修改过的文件。它适合分析模型上限，但不符合真实工程师事先不知道修改文件的条件。

在 27K token 的 BM25 设置下，检索结果相对于 oracle 文件的平均 recall 为 44.41%；约 40% 的任务中，BM25 找回了 oracle 文件的超集，但接近一半任务完全没有找回 oracle 文件。这说明文件定位本身已经是主要瓶颈。

### BM25 下的结果

所读 v3 PDF 的 Table 5 纳入了后续模型作为更新结果。`% Resolved` 是任务完成率，`% Apply` 是 patch 应用率：

| 模型 | SWE-bench Resolved | SWE-bench Apply | SWE-bench Lite Resolved | SWE-bench Lite Apply |
|---|---:|---:|---:|---:|
| Claude 3 Opus | 3.79% | 46.56% | 4.33% | 51.67% |
| Claude 2 | 1.97% | 43.07% | 3.00% | 33.00% |
| GPT-4-turbo | 1.31% | 26.90% | 2.67% | 29.67% |
| ChatGPT-3.5 | 0.17% | 26.33% | 0.33% | 10.00% |
| SWE-Llama 7B | 0.70% | 51.74% | 1.33% | 38.00% |
| SWE-Llama 13B | 0.70% | 53.62% | 1.00% | 38.00% |

这里的比较不能简单解释成“模型越大，patch 就越容易通过”。SWE-Llama 的 apply rate 反而较高，但 resolved rate 不高，说明格式上可应用的 patch 与真正完成 issue 之间仍有明显差距。不同模型的 prompt、patch 格式和上下文策略也会影响 apply rate。

在 oracle retrieval 下，Claude 2 的 resolved rate 从 BM25 的约 1.96% 上升到 4.8%，说明文件检索造成了很大损失，但即便给出参考文件，问题仍然没有被解决。后续的 oracle-collapsed 设置只保留参考文件中修改位置附近的代码，把其他部分折叠掉；GPT-4 从 1.3% 提升到 3.4%，Claude 2 从 4.8% 提升到 5.9%，更新表中的 Claude 3 Opus 达到 9.39%。

这组结果支持一个重要判断：**长上下文能力不等于从长上下文中定位关键代码的能力。**

### 上下文越长不一定越好

在 BM25 检索实验中，Claude 2 的 resolved rate 从 13K 到 27K、50K 上限逐步下降。虽然更大的窗口可能提高相关文件 recall，但同时也引入更多无关文件。模型需要在更大的代码空间中判断哪些内容值得修改，容易受到无关上下文干扰。

论文的 oracle-collapsed 消融进一步说明，问题不只是“模型上下文不够大”，也包括：

- 相关代码是否被准确定位；
- 关键代码在输入中的相对位置；
- 未修改代码是否造成干扰；
- 模型是否理解跨文件依赖，而不是只记住局部代码模式。

### 检索分布变化会影响微调模型

SWE-Llama 训练时使用 oracle retrieval，训练输入里的文件都来自 gold patch 修改范围。评测时使用 BM25 后，输入中会出现很多模型不需要修改的文件，导致明显的 context distribution shift。

这说明训练数据不能只包含“正确代码文件已经被提供”的静态样本。若目标是训练真实 code agent，还需要让模型学习：

1. 在不确定时搜索哪些文件；
2. 如何判断检索结果是否相关；
3. 哪些文件应该读取但不应修改；
4. 如何根据测试和错误反馈继续定位。

### 模型生成的 patch 往往更短、更局部

在成功应用的 patch 中，模型修改通常比 gold patch 短，而且很少修改多个文件。作者的定性分析还观察到：

- 模型倾向于写直接、原始的 Python 代码，而不是复用仓库已有的抽象或第三方库；
- 模型更倾向于贪心地修正表面问题，较少考虑代码风格、导入约束和全局逻辑；
- gold patch 往往做更结构化的修改，不只是通过当前测试，还会覆盖更大范围的潜在行为；
- 模型即使改到了正确函数，也可能遗漏配置分支或相关调用路径。

因此，测试通过率能说明行为是否满足当前测试契约，但不能说明模型已经具备人类工程师的维护判断。

## 这篇论文对 Agent 训练和评测的意义

### 它把“软件工程能力”拆成了可观察的组合问题

SWE-bench 的一个重要价值，是把 code agent 的成功从单一的代码生成能力中拆开：

```text
issue understanding
  -> repository localization
  -> relevant context selection
  -> cross-file reasoning
  -> patch generation
  -> patch application
  -> test execution
  -> error-driven repair
```

原论文主要评测最后的 patch 结果，并通过检索设置分析定位瓶颈。后来的 agent 系统可以在这条链上加入搜索工具、编辑工具、终端、测试反馈和多轮 recovery；SWE-bench 则提供了一个真实任务和可执行 verifier。

### 它也说明了 trajectory data 的价值

一条 SWE-bench 任务本身是静态的 issue-to-patch 样本，不能直接告诉我们模型为何失败。但如果把执行过程记录下来，就可以得到更丰富的 workflow skill data：

- 从 issue 提取约束；
- 搜索和读取候选文件；
- 判断检索结果是否相关；
- 修改代码；
- 运行测试；
- 根据错误日志恢复；
- 确认 patch 和测试结果。

这也是完整 agent trajectory 相对于静态 gold patch 的增量：它保留了 state、action、observation 和 recovery，而不是只保留最终答案。需要注意的是，SWE-bench 原论文没有把这些交互轨迹作为主要训练或评测对象，不能把 benchmark 本身误称为 trajectory dataset。

### resolved rate 应该和过程指标一起看

仅看 resolved rate，无法区分失败发生在哪一步。更完整的 code-agent evaluation 至少可以同时报告：

| 过程指标 | 观察的问题 |
|---|---|
| retrieval recall | 是否找到需要修改的文件 |
| context precision | 提供了多少无关代码 |
| patch apply rate | 输出是否符合编辑协议 |
| test execution success | 环境和测试是否真正运行 |
| fail-to-pass rate | 是否修复目标行为 |
| pass-to-pass rate | 是否保持已有功能 |
| recovery success | 测试失败后能否继续修复 |
| resolved rate | 最终任务是否完成 |

## 局限与证据边界

### 1. 任务只覆盖 Python

评测集来自 Python 仓库，结论不能直接外推到 Java、C++、JavaScript、Rust 或更复杂的多语言 monorepo。

### 2. 静态输入输出不等于完整 Agent

原论文的主实验让模型基于一次组织好的上下文直接生成 patch，没有让模型自主搜索、运行测试、读取日志并多轮修复。因此它是很强的 repository-level coding benchmark，但不是完整的 interactive agent benchmark。

### 3. 测试覆盖不等于工程质量

所有测试通过只说明测试契约下的行为满足要求。它不能单独衡量代码可读性、性能、架构一致性、安全性、维护成本和未覆盖路径。

### 4. Oracle retrieval 会高估真实系统上限

oracle 直接提供参考 patch 修改过的文件，绕过了最困难的 repository localization。它适合做诊断，不适合作为真实 Agent 的主结果。

### 5. 公开仓库存在污染与时间切分问题

作者通过训练仓库与评测仓库分离、以及按 issue/PR 时间分析来降低泄漏风险，但公开 GitHub 代码、issue 和 patch 仍可能被预训练语料覆盖。时间较新的任务更适合检验训练后的泛化，但不能仅靠“发布日期”证明没有污染。

### 6. 任务筛选会改变任务分布

从流行仓库、关联 issue、修改测试文件和能够稳定执行的 PR 中筛选，会提高可验证性，但也可能让数据偏向测试较好、维护较规范的 Python 项目。它不是所有真实软件工程工作的无偏样本。

## 关键结论

SWE-bench 最重要的贡献不是给出了一个新的代码生成模型，而是建立了一种**从真实软件协作历史中构造任务，并用真实测试验证 patch 行为**的方法。

它揭示了三件对后续 code agent 训练很重要的事：

1. 软件工程任务的难点集中在仓库级定位、跨文件理解和测试反馈闭环，不只是代码生成；
2. 更长的上下文和更高的检索 recall 不能自动带来更高的任务完成率，无关上下文可能造成明显干扰；
3. 静态 issue-to-patch 数据适合建立结果评测，但要训练长程 Agent，还需要把搜索、编辑、执行、观察和恢复过程显式保留下来。

## 相关知识链接

- [[application/evaluation/benchmark|Benchmark]]
- [[application/agents/agent|Agent]]
- [[application/tool-use/tool-calling|Tool Calling]]
- [[sources/papers/2025-kimi-dev-agentless-training-as-skill-prior-for-swe-agents|Kimi-Dev]]
- [[sources/papers/2026-tqs-quality-not-just-outcome-coding-agent-trajectories|Quality, Not Just Outcome]]
