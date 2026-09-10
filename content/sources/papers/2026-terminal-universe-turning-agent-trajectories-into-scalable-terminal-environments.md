---
title: "Terminal-Universe: Turning Agent Trajectories into Scalable Terminal Environments"
created: 2026-09-10
published: 2026-09-10
modified: 2026-09-10
type: source
status: processed
source_type: paper
area: sources
tags:
  - source
  - paper
  - agent
  - code-agent
  - terminal-agent
  - agent-environment
  - trajectory-synthesis
  - synthetic-data
  - verifier
  - multi-round
aliases:
  - Terminal-Universe
source_url: https://arxiv.org/abs/2609.04148
paper_date: "2026-09"
paper_order: "04148"
---

# Terminal-Universe: Turning Agent Trajectories into Scalable Terminal Environments

## 基本信息

- 标题：[Terminal-Universe: Turning Agent Trajectories into Scalable Terminal Environments](https://arxiv.org/abs/2609.04148)
- 版本：arXiv:2609.04148v1，2026-09-03
- 作者：Jie Wu、Zhenru Zhang、Beichen Zhang、Xuwu Wang、Yuhui Su、Mouxiang Chen、Peng Wang、Zhihai Wang、Que Shen、Hao Zhou、An Yang、Fei Huang、Yujiu Yang、Dayiheng Liu
- 机构：Qwen Team, Alibaba Group；Tsinghua University
- 研究对象：从既有 terminal agent trajectory 中恢复可执行 workspace，再生成可验证的单 workspace、跨 workspace 和多轮任务
- 训练验证：使用 Qwen3.7-Max 生成任务、解答与 verifier，以约 1.42B tokens 的合成轨迹对 Qwen3.5-27B 进行 supervised fine-tuning
- 相关 topic：[[training/data-engineering/synthetic-data|Synthetic Data]]，[[application/agents/agent|Agent]]，[[application/agents/planning|Planning]]，[[application/agents/workflow-agent|Workflow Agent]]，[[training/post-training/sft|SFT]]

Terminal-Universe 讨论的核心资源不是又一批静态 agent demonstrations，而是 demonstrations 背后的可执行环境。作者观察到，terminal agent trajectory 中记录的文件读取、写入、编辑和命令执行，已经暴露了原 workspace 的一部分结构与内容。即使原始 repository 或 container 不再可得，也可以利用这些证据恢复一个近似的初始环境，再在其中重新提出任务、重新 rollout、运行 verifier，并继续扩展新的交互。

论文由此提出一种 trajectory-environment duality：已有方法通常从 environment rollout 出 trajectory，Terminal-Universe 则尝试反转这一映射，从 trajectory 重建 environment。这个反向过程并不完美，因为未访问文件、隐式系统依赖和外部网络资源不会完整留在轨迹中；论文的主要工作，就是通过 deterministic replay、agentic completion 和 task-conditioned sufficiency filtering，把这种有损恢复变成可规模化的数据生产管线。

## 研究问题

### 为什么 environment 比单条 trajectory 更值得扩展

一条 trajectory 是某个 policy 在一个 task 上的一次固定行为记录。它具有几个限制：

- 输出质量受生成它的原始 policy 上限约束；
- 如果没有原环境和 verifier，很难独立判断文件修改是否真正正确；
- 同一个 task 无法由更强 teacher 重新求解；
- 无法自然地在相同 workspace 上提出更多任务；
- 环境反馈已经冻结，不能继续产生新的失败、修复和需求变化。

一个可执行 environment 则可以被多次使用：同一个 task 可以重新 rollout，同一个 workspace 可以派生多个 task，也可以通过新的 verifier、user feedback 或跨 workspace 依赖构造更复杂的交互。因而，两种资源的扩展单位不同：

```text
trajectory
  -> one frozen demonstration

environment
  -> many tasks
  -> many solution trajectories
  -> executable feedback
  -> reusable verification
  -> continued interaction
```

论文并不是否定 trajectory 的训练价值，而是将 trajectory 重新定义为恢复环境的观测证据。它提出的问题是：**能否把已经积累的大量 agent trajectories 转化为可再次采样的 environment supply？**

### 仅凭可观测 tool history 能恢复多少 workspace

trajectory 只记录 agent 实际观察或修改过的内容。一个文件从未被读取，就不会出现在可恢复证据中；读取结果可能只包含局部行，terminal output 也可能被截断。因此，replay 得到的不是原环境的精确副本，而是一个由轨迹暴露内容构成的 partial workspace。

论文需要解决三个连续问题：

1. 如何从读写操作中恢复 agent 修改前的文件状态；
2. 如何补齐缺失 context，同时避免提前实现原任务或泄漏 solution；
3. 如何判断恢复后的 workspace 对某个 task 是否已经足够，而不是只看文件数或能否启动。

### 如何让每个恢复环境产生更大的训练价值

环境恢复后，最直接的做法只是重建原任务并让强 teacher 重做一次。Terminal-Universe 进一步沿两条轴扩展：

- **Breadth，广度**：连接存在依赖关系的多个 workspace，构造跨 codebase 的任务；
- **Depth，深度**：保持 workspace state，继续生成多轮需求、失败反馈和需求变更。

作者还比较了相同数据条数预算下的三种扩展方式：增加 environment、增加每个 environment 的 query，或增加每个 query 的 solution。这个实验直接回答 agent data scaling 中一个重要问题：训练预算应该优先投入新的执行上下文，还是在已有上下文内增加更多采样。

## 核心主张

### 1. Agent trajectory 可以作为 environment reconstruction 的数据源

论文从 359,593 条公开轨迹出发，通过 replay 与 completion 得到 68,263 个 reconstructed environments。经过 contamination filtering、repository-level deduplication 和 workspace sufficiency judging 后，最终保留 37,273 个 task-sufficient environments。

这个结果表明，轨迹中的文件操作历史不仅是行为监督，也包含对执行环境的部分观测。尤其是包含多文件操作、长工具链和丰富读写内容的轨迹，会暴露更多 workspace state，从而更可能恢复成复杂、可复用的环境。

但“可恢复”具有明显选择偏差。原始 359.6k trajectories 中只有 68.3k 进入 reconstructed environment，说明大部分轨迹没有暴露足够文件证据；command-heavy、view-only 或 action format 不受支持的数据尤其难以重建。论文证明的是一条可行且高产的路径，而不是所有 agent trajectory 都能被无损还原。

### 2. 恢复环境后重新求解，比直接模仿原轨迹更有效

论文用相同的 35.8k data size、相同 chat template 和 tool-call schema 比较两种 supervised fine-tuning（SFT，监督微调）数据：

- `Source trajectories`：直接训练原始 agent 的行为记录；
- `Intent Recovery`：恢复原 task 和 environment，再由统一的强 teacher 重新求解。

在 Terminal-Bench 2.1 上，两种 scaffold 的平均结果为：

| 训练数据 | 平均分 |
|---|---:|
| Qwen3.5-27B base | 47.0 |
| Source trajectories | 36.7 |
| Intent Recovery | 52.1 |

直接训练 source trajectories 不仅没有提升，反而比 base model 低 10.3 points；在恢复环境中重新求解则比 base 高 5.1 points，也比 source-trajectory SFT 高 15.4 points。

这个 ablation 的含义很明确：轨迹数量本身不能代表监督质量。旧轨迹可能来自能力较弱、格式不一致或行为质量不稳定的 policy；恢复 task 与 environment 后，由同一个更强 teacher 重新 rollout，可以统一行为分布，并用新的执行过程替换原 policy 的局限。

### 3. Deterministic replay 不足以恢复 task context，agentic completion 有独立价值

Deterministic replay 只能还原轨迹直接暴露的文件。作者在相同 35.8k records 和相同 teacher 设置下比较：

| Environment reconstruction | Terminal-Bench 2.1 |
|---|---:|
| Replay only | 48.7 |
| Replay + agentic completion | 52.9 |

Agentic completion 带来 4.2 points 提升，并将多次运行的波动从约 `±3.5` 降到 `±1.4`。与此相符，Terminal pool 的 task sufficiency rate 从 replay 后的 40.2% 提升到 completion 后的 93.5%；software engineering（SWE，软件工程）pool 则从 20.1% 提升到 77.1%。

Replay-only 数据仍比 base 高 2.5 points，因为 teacher 会在求解过程中先修补 workspace。但这种轨迹的一部分监督被消耗在环境维修，而不是目标任务本身。提前完成 task-safe 的环境补齐，可以让后续 rollout 更集中地学习 task solving。

### 4. Verifier filtering 的收益随任务难度增加

每个新合成的 Single-WS 或 Cross-WS task 都配有一个 agent-authored executable verifier。`WS` 是 workspace 的缩写：Single-WS 表示单 workspace，Cross-WS 表示跨 workspace。

Verifier filtering 对两类任务的影响不同：

| 数据 | 不做 verifier filtering | 仅保留 verifier-passed |
|---|---:|---:|
| Single-WS | 56.0，35.1k records | 56.4，25.4k records |
| Cross-WS | 53.2，7.1k records | 55.4，3.5k records |

Single-WS 中，过滤近 28% records 后得分基本不变，说明通过验证的较小 subset 具有更高数据效率。Cross-WS 中，保留失败轨迹会明显拖低结果；仅保留不到一半 records，得分反而提升 2.2 points。论文据此认为，任务越复杂，teacher solution failure 对训练数据的伤害越大，execution-based selection 也越重要。

需要注意，verifier-passed 不等于语义上完全正确。Verifier 只能覆盖测试中编码的要求，且本文由同一个 teacher family 参与 task、solution 和 verifier 生成，可能存在共享盲点。

### 5. Cross-workspace data 扩展了任务广度，而不只是增加 token

Cross-WS task 将一个 writable target workspace 和一个 read-only reference workspace 挂载到同一 container。solver 必须从 reference 中发现至少三个 task query 没有直接写明的关键事实，再将相关 capability 按 target 的结构与约定进行适配，而不是简单复制文件。

`Pass@1` 表示每个 task 只采一次 solution 时的通过率。与 Single-WS 相比，Cross-WS trajectory 的中位统计为：

| 指标 | Single-WS | Cross-WS |
|---|---:|---:|
| Assistant turns | 14 | 23 |
| Tool calls | 20 | 38 |
| Tokens per record | 30.4k | 46.5k |
| Teacher Pass@1 | 72.3% | 49.2% |

Cross-WS 的 teacher Pass@1 从 Single-WS 的 72.3% 降至 49.2%，并伴随更多工具调用、turn 和 token，说明它构造的不是形式上更长的 prompt，而是确实更难的多代码库理解与迁移任务。

在 Terminal-Bench 2.1 上，Single-WS 为 56.4，Cross-WS 单独训练为 55.4，二者混合达到 58.4。Cross-WS 不适合作为 Single-WS 的替代，而更像补足跨 repository dependency reasoning 的数据成分。

### 6. Multi-Round data 的价值来自有状态需求推进和验证反馈

Multi-Round synthesis 保持 workspace 跨轮次持久化，并由 user agent 在每轮后决定继续、停止或修改需求。系统维护 active、satisfied 和 updated requirements；在 solver 行动前，verifier agent 根据当前 specification 编写 round-level acceptance tests；solver 完成后，测试结果不会直接暴露，而是由 user agent 转写成自然语言中的可观察问题。

保留的 3,079 个 multi-round records 平均包含 4.51 个 rounds，其中 69.6% 出现过失败并在后续成功修复。三类 interaction style 的观测分布为：

- feature extension：62.7%；
- feature revision：29.3%；
- feature conflict：8.0%。

在 EvoCode-Bench v2 上，论文使用 `MT@4` 和 `Case score` 两个指标。论文正文没有展开 `MT` 的英文全称，因此这里不补写未经来源确认的全称；其操作定义是：每个 task 独立运行四次，采用 fail-stop scoring，某一轮只有在至少一个 run 成功到达并通过时才获得 credit，再按 task 内轮次和 task 间平均。它强调模型在首次失败前能连续完成多少轮累积需求。`Case score` 则统计各轮 verifier cases 的平均通过比例，用来反映部分完成程度。

实验结果为：

| 训练数据 | MT@4 | Case score |
|---|---:|---:|
| Single-WS | 18.4 | 71.9 |
| Single-WS + Multi-Round | 21.0 | 76.9 |
| Single-WS + Multi-Round，无 round verifier | 18.8 | 73.2 |

去掉 round-level verifier 后，数据条数保持 28.5k，但 MT@4 下降 2.2 points，Case score 下降 3.7 points。说明多轮数据的价值不只是更长：如果没有 grounded failure signal，续写更容易变成冗长但低质量的需求扩展。

### 7. 相同记录数预算下，增加 environment 比增加 query 或 solution 更有效

作者从 17,558 个 environment、每个一个 query 和一个 solution 的 17.6k base pool 出发，把 records 扩大到约 35k，但分别沿三种轴扩展：

| 扩展方式 | Environments | Queries / env | Solutions / query | 得分 |
|---|---:|---:|---:|---:|
| Base pool | 17,558 | 1 | 1 | 53.2 |
| Environment expansion | 35,116 | 1 | 1 | 56.0 |
| Query expansion | 17,558 | 2 | 1 | 53.8 |
| Solution expansion | 17,558 | 1 | 2 | 53.9 |

三种配置的数据 records 接近，但只有 environment expansion 带来明显收益。论文的解释是，每个新 environment 提供不同的 executable context 和新的状态分布；同一 environment 内增加第二个 query 或同一 query 增加第二个 solution，较多重复了模型已经见过的结构。

这是论文对 agent data scaling 最具直接操作价值的结论：**训练数据的有效多样性不仅来自 query 和 response，也来自 environment state 的多样性。** 统计 trajectory 条数或 token 数不足以描述 agent data 的覆盖范围，至少还要记录 unique environments、每个 environment 的 queries 和每个 query 的 solutions。

## 方法与机制

### Environment Reconstruction

Terminal-Universe 从 trajectory $\tau$ 恢复近似环境 $\widehat E$。作者明确承认恢复是 lossy 的，因此整个流程不是 repository restoration，而是 task-conditioned environment reconstruction：只要求 workspace 对目标 task 足够，而不是与原环境字节级一致。

#### Stage 1：Deterministic Replay

系统将不同 source corpus 中的文件操作统一成按时间排序的 event stream，再处理 read、write 和 edit：

- 对只读文件，保留 trajectory 中观察到的内容；
- 对被修改文件，恢复 agent 第一次修改前可见的最早版本；
- agent 在 rollout 中新建的文件不放入初始 workspace；
- agent 的后续修改单独保存，不作为待求解环境的一部分；
- 局部读取和 truncated output 只能恢复已观察片段。

最终得到 replayed workspace $\widehat E_0$。其设计目标是让环境回到 agent 求解之前的未完成状态，避免把原 solution 直接写回 workspace。

这一过程依赖一个重要假设：tool log 中的文件内容和 edit 语义足以确定 pre-edit state。若工具只返回 hash、摘要、diff 不完整或命令在文件系统上产生隐式 side effect，deterministic replay 无法完全恢复。

#### Stage 2：Agentic Completion

Completion agent 接收 recovered task、partial workspace 和 file inventory，补齐：

- missing project files；
- partial files；
- configuration、fixtures、scripts 和 schemas；
- manifests 与必要 dependencies；
- 支持目标 task 的 surrounding code。

关键约束是 `make the task solvable, but NOT solved`。Completion agent 不得实现 task 要求的 feature、fix 或 output，也不得通过注释或测试暴露 solution location。它应尽量依据已有命名、framework、version 和 style 构造现实 context。

作者人工检查了 30 个 Terminal completions：22 个只加入 task-relevant support files，8 个加入了较多非必要文件或代码，未在这 30 个样本中发现 task solution leakage。这个样本量很小，只能作为初步质量信号，不能证明大规模 completion 无泄漏。

#### Stage 3：Task-conditioned Sufficiency Filtering

Agentic judge 使用 read-only shell 和 file tools 检查 completed workspace，并根据 recovered task 判断：source、configuration、data 和 project structure 是否足以让一个 capable agent 做出 grounded attempt。

这里的 sufficiency 不等于：

- repository 能完整 build；
- 所有第三方 dependency 已经安装；
- 原 workspace 被完全恢复；
- task 一定能被当前 teacher 解出。

它只判断 task 所需的 project-specific context 是否存在。可常规重新生成的 dependency、cache、generated output 和 optional documentation 可以缺失。这种 task-conditioned criterion 比单纯要求 container build success 更宽松，也更适合 partial reconstruction；代价是 judge error 可能把隐性缺失环境判为 sufficient。

#### 标准化执行环境

所有 workspace 运行在带 network access 的 `ubuntu:24.04` container 中，而不是恢复 repository-specific image。这样降低了部署成本，但对依赖特殊 system library、compiler、service 或 hardware 的 task 会损失 fidelity。网络允许安装缺失依赖，也会降低严格可复现性，并引入外部资源变化与潜在数据访问风险。

### 四种 Re-querying 方法

#### Intent Recovery：恢复原始任务意图

Intent Recovery 将 source trajectory 中一个或多个 user requests 合并为 self-contained task。对于 multi-round source，第一条 substantive request 定义主题，后续请求只在澄清、约束或扩展同一任务时被纳入；无关 task shift 被排除。

系统可以利用 agent action 和 file evidence 理解用户意图，但最终 query 只能保留用户明确提出的 requirement，不能把 agent 后验选择的路径、文件或数值偷偷变成任务约束。这一点避免了从 solution 反向恢复 task 时出现 implementation leakage。

#### Single-WS：在单个 workspace 中生成新任务

Generator 使用 read-only command 检查 workspace，一次生成五个候选 task，再随机选一个 valid candidate 做 rollout 和 verification。候选需要满足：

- grounded in existing files and behavior；
- implementation strategy 未被提前给出；
- 目标和 acceptance criteria 可观测；
- 可以离线、确定性验证；
- 不是 documentation-only、trivial、flaky 或脱离该 repository 也能完成的任务。

Single-WS 的作用是把一个 recovered environment 从“原任务的载体”转化成可生成新 task 的 reusable workspace。

#### Cross-WS：挖掘方向性依赖并生成跨 workspace 任务

Cross-WS 分三步：

1. 对每个 workspace 生成 domain、language、framework、capability 和 entity profile；
2. 在相同语言组内使用 term frequency-inverse document frequency（TF-IDF，词频-逆文档频率）nearest-neighbor retrieval 找候选 pair，再用 large language model（LLM，大语言模型）judge 标注 `similar`、`complementary`、`dependency` 或 `unrelated`；
3. 对 directional dependency pair 生成 task：一侧是 writable target，另一侧是 read-only reference。

最终 query 只告诉 solver reference mount root，不直接写出 reference 内部文件、函数或关键实现细节。Generator 需要在 metadata 中保留至少三个 solver 必须自行发现的 hidden dependency facts，并明确 target landing zone、observable behavior 和 executable validation。

这种设计试图保证 reference 是 load-bearing dependency，而不是装饰性 context。Verifier 还要确认 reference workspace 未被修改。

#### Multi-Round：扩展需求深度

Multi-Round 从一个已解 task 开始，最多生成六个 follow-up rounds。User agent 私下读取：

- original request 与 conversation；
- 当前 workspace；
- active requirement ledger；
- 最新 verifier result；
- requirement-change policy。

如果 verifier failure，user agent 只能发起 feature revision，描述用户可观察症状并要求修复，不得泄露 test、pytest、continuous integration（CI，持续集成）或 traceback。如果当前 requirement 全部通过，则可以增加 compatible feature extension；只有 policy 允许时，才可以用 feature conflict 显式替换已有 requirement。

系统在每轮 solver 行动前先构造 acceptance test，避免根据 solver 的实现反向定制 verifier。Training filtering 会删除 session 末尾连续失败的 suffix，至少保留两个 verified passing rounds；成功恢复之前的 intermediate failed rounds 则保留，用于训练 error diagnosis 和 recovery。

### Verifier Construction 与数据筛选

Verifier agent 在 solution rollout 前读取 query 和 initial workspace，并生成一个 self-contained `pytest` suite。主要约束包括：

- 只测试 public interface 和 query 明确约束的 behavior；
- 新功能 expected value 必须独立计算，不能运行 incomplete implementation 后把结果当 oracle；
- preservation test 可以把 initial behavior 作为 baseline；
- 最多 18 个 tests；
- 不允许 network、randomness、wall-clock dependency、skip 或 xfail；
- 不得写入 `/app`，临时文件放在 `/tmp`；
- solver 看不到 test file。

Verifier 必须执行 mandatory red check：在未修改 workspace 上，至少一个针对缺失 capability 的 test 应失败，而 preservation tests 必须通过，且不能出现 collection 或 infrastructure errors。这个要求同时检查 task gap 是否真实存在，以及 verifier 是否在初始环境上可运行。

Single-WS 和 Cross-WS 只保留最终所有 tests 通过的 trajectory。Multi-Round 则按 round 选择，保留最终可形成有效 passing prefix 的 session，并允许中间失败后修复。Intent Recovery 不使用 verifier filtering，以保证与 source-trajectory SFT 的对照不受筛选因素干扰。

## 数据构造规模

### Source trajectories 与重建漏斗

数据来自六个公开 source corpora，覆盖 terminal command-line interface（CLI，命令行界面）和 SWE tasks：

| Source pool | Trajectories | Reconstructed environments |
|---|---:|---:|
| SWE-rebench | 67,074 | 6,118 |
| SWE-smith | 95,851 | 8,476 |
| CoderForge | 32,964 | 3,978 |
| SWE-Gym | 4,152 | 929 |
| LFM2-Terminal | 139,841 | 46,037 |
| LiteCoder-Terminal | 19,711 | 2,725 |
| **Total** | **359,593** | **68,263** |

核心筛选漏斗为：

```text
359,593 source trajectories
  -> 68,263 reconstructed environments
  -> 40,194 after contamination filtering and repository deduplication
  -> 37,273 task-sufficient environments
```

Seed 至少要求 trajectory 末尾观察到的 workspace 包含 5 个 files 和 100 lines。SWE corpora 内按 repository、base commit 和 problem statement 去重，并选择 replay 能暴露最多 workspace content 的 trajectory。Terminal-Bench-derived sources 被排除，SWE-bench Verified repositories 也从 SWE pool 中移除。

### Agentic completion 对 workspace complexity 的影响

| Pool | Files：replay -> completion | Text lines：replay -> completion | Code lines：replay -> completion |
|---|---:|---:|---:|
| Terminal，mean | 2.9 -> 22.4 | 90 -> 5,761 | 43 -> 503 |
| SWE，mean | 5.6 -> 37.9 | 595 -> 6,622 | 487 -> 6,302 |

Terminal workspace 的 text lines 平均增长 64 倍，远高于 code lines 的增长。这反映 completion 可能加入 configuration、data、schema、script 和非源码 context；也提示不能只用总行数判断恢复质量。

最终 Terminal pool 以 Python 为主，占 84.7%；domain 以 data processing 47.4%、DevOps 18.9% 和 security 14.6% 为主。模型收益主要建立在这一分布上，不应外推成所有语言和 software engineering domain 都同等覆盖。

### 最终 SFT 数据

| Variant | Records | Median turns | Median tool calls | Median tokens |
|---|---:|---:|---:|---:|
| Intent Recovery | 35,809 | 12 | 17 | 36.1k |
| Single-WS | 25,386 | 14 | 20 | 30.4k |
| Cross-WS | 3,512 | 23 | 38 | 46.5k |
| Multi-Round | 3,079 | 92 | 102 | 126.1k |

论文用于 Full Mixture 的 verifier-filtered Single-WS、Cross-WS 和 Multi-Round 共 31,977 records，约 1.42B training tokens。Intent Recovery 是独立生成的数据分支，主要用于 task re-solving 与 reconstruction ablation，不属于 Table 3 的 Full Mixture 组成。

## 训练与评测设置

### Teacher rollout

所有 model-driven components，包括 completion、task generation、relation judging、user simulation、solution rollout 和 verifier construction，均使用 Qwen3.7-Max，reasoning effort 为 `xhigh`。

Solution trajectory 在 Claude Code scaffold 内生成，使用：

- temperature 1.0；
- top-p 0.95；
- interleaved thinking；
- 256k context window；
- 单 turn 最多 65,536 tokens；
- 176k tokens 时触发 proactive summarization；
- 最多 500 agent turns；
- 四小时 wall-clock timeout。

这一配置允许非常长的 trajectory，也意味着数据生产成本高，且结果依赖 teacher、scaffold、summarization 和 tool protocol 的联合行为。

### Student training

作者使用 Full Mixture 对 Qwen3.5-27B 进行两 epochs SFT：

- constant learning rate：$7\times10^{-6}$；
- global batch size：256；
- sequence length：256k；
- 训练前做 13-gram contamination check。

论文只验证了 SFT，没有直接评估这些 reconstructed environments 用于 reinforcement learning（RL，强化学习）在线 rollout 的效果。因此，“environment 可以支持 RL”是框架能力和未来用途，而不是本文已经完成的训练实证。

### Single-round evaluation

Terminal-Bench 2.0 和 2.1 使用 Claude Code 与 Terminus2 两种 scaffold。每个 task 独立运行六次，报告 average Pass@1。执行资源为 12 central processing unit（CPU，中央处理器）cores、32 gibibytes（GiB，二进制吉字节）memory，单次 wall-clock 最多四小时。Claude Code 中关闭了 WebFetch、WebSearch、AskUserQuestion 和 plan-mode tools。

### Multi-round evaluation

EvoCode-Bench v2 包含 26 个 coding tasks 和 227 个 rounds，每个 task 有 5 到 15 个连续 requests。Workspace 和 session 跨轮持久化，cumulative verifier 同时检查当前与此前仍然 active 的 requirements。每个 task 运行四次，wall-clock 上限十小时。

这套评测比单轮 benchmark 更能观察需求累积与持久状态，但 task 数只有 26，MT@4 对少数 task 和 run 的变化可能较敏感。

## 实验结果

### Main results

Full Mixture 对 Qwen3.5-27B 的提升为：

| Benchmark / scaffold | Base | Terminal-Universe | 提升 |
|---|---:|---:|---:|
| Terminal-Bench 2.0 / Terminus2 | 41.6 | 52.8 | +11.2 |
| Terminal-Bench 2.1 / Terminus2 | 46.2 | 58.1 | +11.9 |
| Terminal-Bench 2.1 / Claude Code | 47.8 | 58.2 | +10.4 |
| EvoCode-Bench v2 MT@4 | 6.3 | 20.1 | +13.8 |
| EvoCode-Bench v2 Case score | 67.8 | 76.1 | +8.3 |

Terminal-Bench 的提升在两个 scaffold 上方向一致，说明收益不完全依赖 Terminus2 parser 或单一 agent harness。Multi-round 结果同时改善 MT@4 和 Case score，表明模型不仅获得更多 partial progress，也能在首次失败前推进更长的有效请求序列。

不过，Table 3 汇总的其他方法来自不同 base model、data size、scaffold 和 evaluation configuration，不能将排名差异全部归因于数据方法。论文对自身 ablation 的控制更严格，因而 task re-solving、completion、verifier、breadth、depth 和 expansion-axis 对照比跨论文 leaderboard 更有解释力。

### Cross-domain transfer

作者还对 SWE workspace 做 Intent Recovery。1,900 个 SWE repositories 中有 1,464 个被判为 task-sufficient，由于每个 repository 可对应多个 source tasks，最终形成约 10.3k trajectories。

这些 SWE data 将 Terminal-Bench 2.1 的两 scaffold 平均分从 47.0 提升到 50.0；Claude Code 从 47.8 到 50.6，Terminus2 从 46.2 到 49.4。结果说明从 SWE environment 重新求解得到的监督可以迁移到 terminal behavior，但论文只测试 SWE-to-terminal 方向，没有验证 terminal-to-SWE 或其他 agent domain。

## 对 Agent 数据建设的启发

### 从 trajectory corpus 中识别可恢复 environment

对已有大量 agent trajectories 的数据管线，可以新增一层 environment recoverability profiling，而不是只做 trajectory quality scoring。至少记录：

- 文件 read / write / edit 是否包含完整 content；
- pre-edit state 是否可确定；
- agent-created files 和 source files 是否可分离；
- workspace identity、repository、base commit 或 session lineage；
- trajectory 暴露的 unique paths、source lines、config 和 dependency evidence；
- external service、network、system dependency 是否可以重建；
- 多条 session 是否来自同一个 workspace，能否合并 state evidence。

论文指出，多文件操作和长 tool-use chain 会恢复出更复杂的 environment；同一 workspace 上的多个 sessions 还可以聚合，使每条 trajectory 暴露的局部 state 合并为更完整的 reconstruction。这意味着 trajectory richness 不只是训练样本质量，还决定 environment recovery ceiling。

### 数据质量可以通过 re-solving 与 verification 重建

已有 trajectory 中 reasoning 缺失、policy 弱或 outcome 不可信时，不一定只能补写 reasoning 或直接过滤。另一条路线是：

```text
trajectory evidence
  -> recover task and pre-solution workspace
  -> complete missing context without solving
  -> author independent verifier
  -> re-solve with a stronger teacher
  -> keep executable successes
```

它把监督质量从“相信原轨迹”改为“相信重建环境中的新执行结果”。但这条路线要求 trajectory 暴露足够 workspace evidence，也需要严格防止 completion agent 把 solution 写入初始环境。

### Environment diversity 应成为数据 mixture 的一级维度

论文的 matched-budget ablation 显示，增加 unique environment 比同环境增加 query 或同 query 增加 solution 更有效。实际记录 agent corpus 时，可以将下面三个计数分开：

```text
unique environments
queries per environment
solutions per query
```

这与普通 language-model corpus 中的 document diversity 类似，但 environment 还包含可执行 state、dependency、tool surface 和 feedback dynamics。两个 query 即使文本差异很大，只要共享高度相似的 workspace 和 execution pattern，提供的新 supervision 仍可能有限。

### Multi-round failure 不应一律删除

论文对不同位置的 failure 采用不同策略：

- Single-WS / Cross-WS final solution failure：过滤；
- Multi-Round trailing consecutive failures：裁掉；
- Multi-Round 中间失败后成功恢复：保留。

这种选择保留了 error diagnosis 与 recovery supervision，同时避免 session 以未解决状态结束。它比“所有失败轨迹都删掉”或“只要最终成功就保留整条轨迹”更细致，也适合用于长程 agent trajectory 的质量分级。

### Environment reconstruction 可以成为 RLE 的环境供给层

对于依赖 reinforcement learning environment（RLE，强化学习环境）的 agent training，Terminal-Universe 提供的是 environment bootstrapping 方法：从历史轨迹中恢复 seed environments，再围绕 seed query 生成新任务、verifier 和 rollout。

但本文没有完成 RL 实验，落地时仍需额外解决：

- reset 是否确定且便宜；
- verifier 能否转换为稳定 reward；
- episode isolation 和 dependency cache 如何管理；
- network access 是否允许；
- environment 是否会被 policy exploit；
- 多轮 user simulator 是否与 reward state 一致；
- environment mutation 后能否可靠恢复 initial snapshot。

因此，论文验证的是 environment reconstruction 对 SFT data generation 的价值，而不是 RLE 系统已经被完整验证。

## 局限与风险

### Reconstruction fidelity 受 source trajectory 上限约束

未被访问的文件和依赖无法从 trajectory 中直接恢复。Completion agent 可以补充通用 project context，却不能可靠重建未见过的 project-specific implementation。最终环境更接近“由原轨迹证据约束的新 workspace”，不一定是原环境的真实副本。

### Completion 同时带来 context 和 synthetic artifact

Completion 显著提高 sufficiency 与训练效果，但它也会引入 teacher 生成的 code、config 和 data。人工抽检仅覆盖 30 个 Terminal workspaces，其中 8 个包含较多非必要内容。缺少大规模的 solution leakage、semantic fidelity 和 artifact quality audit。

### Sufficiency judge 不是 executable correctness guarantee

Judge 评估“是否有足够 context 做 grounded attempt”，而不是完整 build 或 task solvability proof。Agentic judge 与 completion agent 还可能存在同源偏差。更严格的 pipeline 可以增加 dependency resolution、baseline build、red test、static analysis 和独立 judge。

### Task、solution 和 verifier 使用同一个 teacher

Qwen3.7-Max 同时参与多个环节，可能产生 correlated error：task 中遗漏的要求不会被 verifier 检查，错误 solution 也可能通过不充分 test。论文已将 multi-teacher 和 independent verifier model 列为未来方向。

### 数据分布较窄

最终 terminal pool 中 Python 占 84.7%，主要 domain 集中在 data processing、DevOps 和 security。标准 Ubuntu container 也不适合需要特殊 compiler、service、graphics processing unit（GPU，图形处理器）、graphical user interface（GUI，图形用户界面）或 proprietary dependency 的任务。因此，37.3k environments 代表数量扩展，不代表环境类型已经充分覆盖。

### Contamination 仍难完全排除

论文排除了 Terminal-Bench-derived source，并做 13-gram contamination check，但 teacher 可能在预训练中见过 benchmark，生成的新 query 也可能与公开任务存在语义近似。N-gram filtering 无法发现所有 paraphrase、repository overlap 和 implementation-level leakage。

### 评测与训练验证仍有限

Terminal-Bench 是主要单轮验证，EvoCode-Bench v2 只有 26 个多轮 tasks。论文没有报告更多独立 SWE benchmark、真实用户交互、RL training、长期 environment reuse 成本或安全性结果。现有结论支持 terminal-agent SFT data construction，尚不足以证明对所有 code agent 或 general agent 同样有效。

## 研究判断

Terminal-Universe 最重要的贡献，是把 agent trajectory 的价值从“可直接训练的一次 demonstration”扩展到“可恢复 environment 的观测记录”。这改变了 agent data scaling 的基本单位：当 environment 可被恢复、reset 和验证时，一条旧轨迹可以成为新的 task、solution、failure-recovery session 以及未来 RL rollout 的起点。

论文也提供了三条相对扎实的工程结论：

1. 对质量不稳定的历史轨迹，恢复环境后由统一强 teacher 重新求解，可能比直接 imitation 更有效；
2. Partial replay 需要 task-safe completion 和 sufficiency filtering，否则 rollout 会把大量能力消耗在修环境；
3. 在记录数相同的条件下，优先扩展 unique executable environments，比在相同 workspace 内堆叠更多 query 或 solution 更有价值。

需要保持的边界是：环境并非从 trajectory 中完整还原，而是由 trajectory evidence 与 teacher completion 共同构造；verifier-passed 也只是通过当前 tests。因而，一个严谨的 trajectory-to-environment pipeline 应同时保存 provenance、replay evidence、synthetic completion diff、sufficiency decision、verifier coverage 和 execution result，不能只发布最终 SFT text。

对于已有长轨迹数据，这篇论文提供了一条与 reasoning synthesis 互补的路线：reasoning synthesis 修复单条 trajectory 内部的监督缺口，environment reconstruction 则把 trajectory 转换为可继续产生新 supervision 的基础设施。前者提高每条记录的可训练性，后者扩大同一批历史数据能够支撑的任务与交互空间。

## 相关知识链接

- [[training/data-engineering/synthetic-data|Synthetic Data]]
- [[training/data-engineering/deduplication|Deduplication]]
- [[application/agents/agent|Agent]]
- [[application/agents/planning|Planning]]
- [[application/agents/workflow-agent|Workflow Agent]]
- [[application/agents/memory|Memory]]
- [[application/evaluation/benchmark|Benchmark]]
- [[training/post-training/sft|SFT]]
- [[training/post-training/rejection-sampling|Rejection Sampling]]
- [[sources/reports/code-agent-trajectory-reasoning-synthesis|Code Agent Trajectory Reasoning Synthesis]]
- [[sources/papers/2026-klong-training-llm-agent-for-extremely-long-horizon-tasks|KLong]]
- [[sources/papers/2026-davinci-dev-agent-native-mid-training-for-software-engineering|daVinci-Dev]]
