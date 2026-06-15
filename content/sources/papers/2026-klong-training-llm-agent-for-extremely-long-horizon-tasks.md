---
title: "KLong: Training LLM Agent for Extremely Long-horizon Tasks"
created: 2026-06-01
published: 2026-06-01
modified: 2026-06-02
type: source
status: processed
source_type: paper
area: sources
tags:
  - source
  - paper
  - agents
  - long-horizon
  - reinforcement-learning
  - sft
  - evaluation
aliases:
  - KLong
source_url: https://arxiv.org/pdf/2602.17547v2
venue: Lifelong Agent Workshop @ ICLR 2026
---

## 基本信息

- Title: [KLong: Training LLM Agent for Extremely Long-horizon Tasks](https://arxiv.org/pdf/2602.17547v2)
- Authors: Yue Liu, Yingwei Ma, Yibo Miao, Yanhao Li, Yuchong Xie, Xinlong Yang, Zhiyuan Hu, Flood Sung, Jiaheng Zhang, Bryan Hooi
- Institutions: NUS, HKUST, SJTU, PKU, MIT
- Date: arXiv v2, 2026-04-06
- Venue: Lifelong Agent Workshop @ ICLR 2026
- Related topic notes: [[application/agents/agent|Agent]], [[application/agents/planning|Planning]], [[application/agents/memory|Memory]], [[application/agents/workflow-agent|Workflow Agent]], [[application/tool-use/tool-calling|Tool Calling]], [[application/rag/context-compression|Context Compression]], [[application/evaluation/benchmark|Benchmark]], [[application/evaluation/llm-as-judge|LLM-as-a-Judge]], [[training/post-training/sft|SFT]], [[training/post-training/ppo|PPO]], [[training/post-training/grpo|GRPO]], [[training/post-training/rejection-sampling|Rejection Sampling]], [[training/mid-training/long-context-training|Long Context Training]]

KLong 的核心问题是：如何训练一个 LLM agent 去完成超过常规上下文窗口和常规 rollout 时长的极长程任务。论文把 extremely long-horizon task 定义为同时满足两个条件的任务：如果没有 context management 就会超出 context window；并且任务本身包含长时间运行的实验或环境交互。典型例子是 PaperBench 中的论文复现任务和 MLE-bench 中的机器学习竞赛任务。

论文的答案不是单纯扩大 context window，也不是只改 agent scaffold，而是提出一套训练和基础设施结合的方案：先用 trajectory-splitting SFT 冷启动 agent 的长程行为，再用 progressive RL 在逐步延长的任务超时限制下继续优化，同时用 Research-Factory 自动构造论文复现任务、rubric 和长轨迹数据。实验上，KLong 主要针对 PaperBench 训练，并在 SWE-bench Verified、Terminal-Bench Hard、SEC-bench 和 MLE-bench 上报告一定程度的泛化。

## 研究问题

常见 long-horizon agent 任务通常强调多轮交互、工具使用、规划和环境反馈，例如 bug fixing、terminal coding 或 web navigation。但 KLong 关注的任务更长：论文指出，PaperBench 和 MLE-bench 这类任务的 running time 和 assistant turns 大约是普通 long-horizon benchmark 的 10 倍量级。这样的任务给训练带来几个结构性困难：

1. **上下文长度不可避免地溢出**：完整 trajectory 包含论文阅读、代码编写、实验运行、错误修复和结果迭代，无法直接作为一个 SFT 样本或一个 RL trajectory 放入 context window。
2. **奖励稀疏且延迟严重**：论文复现任务的 reward 往往来自最终代码和 rubric-based judge，早期读论文或中期重构代码的动作很难获得直接 credit。
3. **rollout 成本极高**：完整任务可能超过 12 小时，直接做 end-to-end RL 会造成样本昂贵、训练不稳定、评测拥塞和集群资源空转。
4. **评测本身依赖复杂 judge**：PaperBench 需要根据 rubric tree 对复现实验、代码和结果进行评分，训练阶段如果直接依赖官方闭源 judge，成本和 benchmark hacking 风险都会上升。

因此，这篇论文真正讨论的是一个 agent training system 问题：如何在任务、数据、训练目标、执行环境和 judge 之间建立一套可以规模化的闭环。

## 核心主张

KLong 的核心主张可以概括为两层：

第一，极长程 agent 能力需要先通过 imitation-style 的长轨迹监督进行冷启动。普通 SFT 可以激活基础 agentic abilities，但不足以让模型学会长时间维持任务状态、持续阅读、逐步实现代码和反复运行实验。Trajectory-splitting SFT 通过把极长 trajectory 切成带重叠的子轨迹，并在每个子轨迹开头固定任务说明和 paper-reading prefix，让模型在有限 context window 中学习长程任务的局部决策模式。

第二，冷启动后的 agent 还需要在真实长时环境中通过 RL 扩展能力。Progressive RL 不直接从 12 小时完整任务开始训练，而是用逐步增长的 timeout schedule，例如 2H -> 4H -> 6H，让 policy 先学会在较短 horizon 内获得有效 reward，再逐步适应更长的 delayed reward 和更复杂的实验迭代。

这两个主张共同构成一个比较重要的观点：极长程 agent 能力不是单一组件带来的，而是训练数据形态、trajectory construction、RL horizon curriculum、judge design 和 execution infrastructure 的组合产物。

## 方法与机制

### Task Definition

论文以 PaperBench 风格的 research replication task 作为主要训练任务。给定一篇论文 $P$，agent $A$ 需要在沙箱环境中阅读论文、理解核心贡献、实现复现代码 $\hat{C}$，并在时间限制 $t$ 内通过工具调用和环境交互完成尽可能多的核心结果。过程可以抽象为：

$$
A: P \xrightarrow{\{a_k\}_{k=1}^{N}} \hat{C}, \quad \text{s.t. } \sum_{k=1}^{N} t_k \leq t
$$

其中 $a_k$ 是第 $k$ 步 action，$t_k$ 是该 action 消耗的时间。最终质量由 judge model $J$ 根据 rubric tree $K$ 评分：

$$
Q = J(\hat{C}, K)
$$

这一设定使任务天然不同于单轮推理或短程 coding benchmark。模型不只是输出答案，而是要长期维护目标、读取文件、执行代码、根据报错修复、运行实验并优化结果。

### Research-Factory

Research-Factory 是论文用于规模化构造训练数据的自动管线。它包含两个核心组件：search agent 和 evaluation agent。

Search agent 从 ICML、NeurIPS、ICLR 等顶级会议近五年的论文中收集候选论文及 metadata，并通过质量和影响力筛选保留样本。被选中的 PDF 会转换为 Markdown，便于 agent 读取和训练处理。对应论文的官方 GitHub URL 会写入 blacklist，以减少训练和评测时直接访问官方实现造成的数据泄漏或 benchmark hacking。

Evaluation agent 负责为每篇论文构造 addendum 和 rubric tree。它会分析论文内容和官方代码实现，抽取复现任务中应被评分的核心贡献、实验设置、数据集使用、结果验证方式等。这个过程很重要，因为 PaperBench 类任务的 reward 不可能只用最终 accuracy 或单个单元测试表示，必须把复杂论文复现拆解成可判分的层级结构。

Research-Factory 产出的 prompt set 随后用于构造长轨迹。论文使用 Claude 4.5 Sonnet (Thinking) 蒸馏出数千条 extremely long-horizon trajectories，并通过 trajectory 质量和 judge score 做 rejection sampling。这意味着 KLong 的 SFT 阶段并不是从人类手工演示开始，而是用强 teacher model 在受控论文复现环境中的行为轨迹作为冷启动数据。

### Comprehensive SFT Cold Start

在进入极长轨迹训练之前，论文先对 base model 做综合 SFT，以激活基础 agentic abilities。SFT 数据覆盖 common knowledge、coding、mathematics 和 search。目标函数是标准条件最大似然：

$$
L_{\text{SFT}} = - \sum_{(x_j, y_j)\in D_{\text{SFT}}} \log P_\theta(y_j \mid x_j)
$$

这一步的角色类似 agent policy 的基础行为对齐：模型先要掌握工具使用、代码生成、搜索和一般任务响应能力，之后才有可能从论文复现长轨迹中学习更复杂的时间组织和任务执行策略。

### Trajectory-splitting SFT

极长程任务会产生如下形式的 trajectory：

$$
\tau = (s_1, a_1, s_2, a_2, \ldots, s_N, a_N)
$$

其中 $s_i$ 表示 observation，$a_i$ 表示 action。由于 $N$ 很大，完整 trajectory 往往超过模型最大上下文长度 $L_{\max}$。直接把完整 trajectory 作为 SFT 样本会遇到两个问题：一是无法放入 context window；二是简单截断会丢失任务说明、论文阅读结论和全局目标。

KLong 的 trajectory-splitting SFT 使用三项机制：

1. **固定早期全局信息**：把 task specification 和 paper-reading content 作为 prefix $p$，放在每个子轨迹开头。这样每个局部训练样本都保留任务定义和论文理解，而不是只看到中间某段零散 tool trace。
2. **逐步截断后续上下文**：对过长的历史进行截断，使每个子轨迹长度不超过 $L_{\max}$。截断发生在 later context 中，而不是删除关键的任务和论文阅读 prefix。
3. **子轨迹之间保持 overlap**：相邻子轨迹共享一部分 observation-action 历史，用于保持局部连续性，减少切分边界带来的状态断裂。

形式上，原始轨迹被拆成多个重叠子轨迹：

$$
\tau^{(i)} = (s_{t_i}, a_{t_i}, \ldots, s_{t_i+L-1}, a_{t_i+L-1})
$$

并构造成带 prefix 的输入：

$$
\tau^{(i)}_{\text{input}} = [p, s_{t_i}, a_{t_i}, \ldots, s_{t_i+L-1}, a_{t_i+L-1}]
$$

训练 loss 对子轨迹中的 action tokens 计算：

$$
L_{\text{SFT}} = - \sum_{i=1}^{K} \sum_{t=t_i}^{t_i+L-1}
\log P_\theta(a_t \mid \tau^{(i)}_{<t})
$$

机制上，这是一种训练时的 context construction 方法。它不直接解决推理阶段的无限记忆问题，但让模型在训练中反复见到“全局任务信息 + 局部执行历史”的上下文结构，从而学会在有限窗口中继续长任务。实验中，splitting SFT 让平均 assistant turns 从 114.9 增加到 732.7，说明模型更愿意也更能够持续执行任务，而不是过早结束。

### Progressive Reinforcement Learning

Trajectory-splitting SFT 解决了长轨迹如何进入 SFT 的问题，但它仍然是 imitation learning。要进一步优化真实任务得分，论文使用 progressive RL。

直接对完整论文复现任务做 RL 很困难，因为完整任务可能超过 12 小时，reward 稀疏，credit assignment 高方差，且 rollout 成本过高。KLong 因此定义一组逐步增加的 timeout：

$$
T^{(1)} < T^{(2)} < \cdots < T^{(M)}
$$

在第 $m$ 个阶段，rollout 到达 timeout $T^{(m)}$ 时会被强制截断，得到阶段性 trajectory：

$$
\tau^{(m)} = (s_1, a_1, \ldots, s_{N^{(m)}}, a_{N^{(m)}})
$$

由于这些 trajectory 仍可能超过 context window，训练时继续使用 trajectory-splitting，把每条 rollout 拆成 $K^{(m)}$ 个重叠子轨迹。若每个 sample 做 $n$ 次 rollout，则优化中一共使用 $n \cdot K^{(m)}$ 个子轨迹。

论文的 RL objective 是 PPO-like clipped surrogate，并加入 KL regularization：

$$
L^{(m)}_{\text{RL}}(\theta)
= -\frac{1}{nK^{(m)}} \sum_{j=1}^{n}\sum_{i=1}^{K^{(m)}}\sum_{t\in \tau^{(m,i,j)}}
\min\left(
r_t^{(m,i,j)}(\theta)\hat{A}_t^{(m,i,j)},
\text{clip}(r_t^{(m,i,j)}(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t^{(m,i,j)}
\right)
+ \beta \mathbb{E}_t\left[
\text{KL}\left(\pi_\theta(\cdot\mid s_t)\|\pi_{\theta_{\text{ref}}}(\cdot\mid s_t)\right)
\right]
$$

其中 $\pi_{\theta_{\text{ref}}}$ 是上一轮训练迭代的 reference policy，$r_t(\theta)$ 是当前 policy 相对 reference policy 的 likelihood ratio。Advantage 使用 group-relative 形式：

$$
\hat{A}^{(m,i,j)}_t =
Q^{(m,i,j)}
- \frac{1}{nK^{(m)}}\sum_{j'=1}^{n}\sum_{i'=1}^{K^{(m)}} Q^{(m,i',j')}
$$

这里 $Q^{(m,i,j)} = J(\hat{C}_{m,i,j}, K)$，由 judge model 根据复现代码和 rubric tree 打分。这个设计与 [[training/post-training/grpo|GRPO]] 的组内相对优势思想相近：不显式训练 value model，而是在同一任务的一组 rollout 或子轨迹之间做相对比较。不同之处在于，KLong 的任务 reward 来自复杂 judge 对最终复现产物的评分，credit assignment 仍然非常粗粒度。

Progressive RL 的关键不是某个新 RL 算法，而是 horizon curriculum。2H、4H、6H 阶段逐步增加 agent 可用于探索、实验和修复的时间，使 policy 不必一开始就面对 12 小时级别的稀疏反馈。消融结果显示，RL 2H、RL 4H、RL 6H 的 PaperBench 平均分分别为 57.29、58.65、62.59，支持更长训练 horizon 能提升最终长程任务能力。

### Infrastructure Optimization

KLong 论文的一个重要价值在于把训练 infra 作为方法的一部分，而不是附属实现细节。

**Sandbox**：论文构建了基于 Kubernetes 的统一沙箱，支持 Python-accessible、安全、可扩展的代码执行。沙箱采用 ephemeral use-and-destroy 模式，支持 10,000+ concurrent instances。sidecar container 管理 25,000+ Docker images，并预装 torch、TensorFlow、scikit-learn、einops 等 80+ research-related Python packages，以降低论文复现任务中的环境配置成本。

**Scaffolding**：系统基于 PaperBench scaffold，并做了若干针对长程任务的改造：强制 paper reading 和 tracking；改进 context-length error handling；增强 file-reading progress parsing 的鲁棒性；加入 prompt caching；禁止早期调用 end_task 工具。这些约束本质上是在防止 agent 过早结束、跳过论文阅读或因为上下文/文件读取失败而崩溃。

**Rollout & Training Pipeline**：极长程 RL 会出现 pipeline imbalance。固定 2H 之类 timeout 会让大量 rollout 同时开始、同时结束，随后 judge 阶段拥塞，而 rollout nodes 在等待 evaluation 时空转。KLong 的缓解方式是在当前 evaluation 进行时，让 rollout nodes 提前启动下一轮 partial rollouts；未完成的 rollout 会带到下一轮继续，从而提高资源利用率。

**Judge**：PaperBench 官方 judge 是闭源 o3-mini。KLong 训练阶段改用开源 gpt-oss-120b 结合自动构造的 rubrics 作为 judge，以降低成本，并避免直接针对官方 evaluator 过拟合。由于 judge 请求本身并发高、内部并行复杂，系统还引入 priority queue，优先处理 evaluation set，减少评测失败对训练信号的污染。

## 实验与证据

### PaperBench 主结果

KLong 的主要训练目标是 PaperBench。论文报告 KLong(106B) 在 PaperBench 上达到 62.59 的 average score，是表中 open-source models 里最高的结果，并比 Kimi K2 Thinking(1T) 的 51.31 高 11.28 分。它仍低于 Claude 4.5 Sonnet(Thinking) 的 69.75，但高于 GPT-5 Thinking(High) 的 52.31 和 Grok 4 的 47.20。

| Model | Type / Size | PaperBench Average |
|---|---:|---:|
| Qwen3-Thinking | Open-source, 235B | 28.72 |
| DeepSeek-V3.2 | Open-source, 685B | 47.11 |
| Kimi K2 Thinking | Open-source, 1T | 51.31 |
| KLong | Open-source, 106B | 62.59 |
| Grok 4 | Closed-source | 47.20 |
| GPT-5 Thinking(High) | Closed-source | 52.31 |
| Claude 4.5 Sonnet(Thinking) | Closed-source | 69.75 |

任务级结果显示，KLong 在 test-time-model-adaptation、all-in-one、sequential-neural-score-estimation、lca-on-the-line、lbcs 等任务上表现突出。论文据此认为，KLong 不只是增加了输出长度，而是在需要持续推理、代码实现、实验运行和迭代修复的任务上获得了更强能力。

### Ablation Study

消融实验最清楚地说明了两个核心组件的贡献：

| Stage | PaperBench Average | Assistant Turns | Running Hours |
|---|---:|---:|---:|
| Baseline | 38.63 | 114.9 | 1.52 |
| Baseline + splitting SFT | 55.92 | 732.7 | 8.88 |
| Baseline + SFT + RL 2H | 57.29 | - | - |
| Baseline + SFT + RL 4H | 58.65 | - | - |
| Baseline + SFT + RL 6H | 62.59 | - | - |

Splitting SFT 从 38.63 提升到 55.92，绝对增益 17.29，是最大的一段跃迁。与此同时 assistant turns 和 running hours 大幅增加，说明 SFT 让模型学会了更长时间地参与任务执行。Progressive RL 在 SFT 基础上继续提升，RL 6H 相对 SFT-only 增加 6.67 分，相对 Baseline 增加 23.96 分。

需要注意的是，消融中并不是所有单项任务都随 RL horizon 单调上升。例如一些任务在 RL 2H 或 RL 4H 上已经达到较高分，RL 6H 反而下降。论文的平均分趋势支持 progressive RL，但任务级方差说明极长程 RL 仍然存在 credit assignment、judge noise 和任务分布差异。

### Generalization Experiments

论文在四类额外 benchmark 上报告泛化。

| Benchmark | Baseline + SFT | KLong | 说明 |
|---|---:|---:|---|
| SWE-bench Verified | 304/500, 60.80% | 314/500, 62.80% | 使用 OpenHands scaffold |
| Terminal-Bench Hard | 7/48, 14.58% | 8/48, 16.67% | 使用 Terminus 2 scaffold |
| SEC-bench | 15/300, 5.00% | 23/300, 7.67% | CVE: 7/200 -> 9/200; OSS-Fuzz: 8/100 -> 14/100 |

这些结果说明，KLong 从论文复现任务中学到的长程执行能力可以迁移到 software engineering、terminal coding 和 security tasks，但增益幅度并不一样。SWE-bench 和 Terminal-Bench 的绝对提升较小，SEC-bench 相对提升更明显但绝对成功率仍低。

MLE-bench 的结果更异质。KLong 在 tabular-playground-series-dec-2021 上达到 Gold Medal，在 tabular-playground-series-may-2022、the-icml-2013-whale-challenge-right-whale-redux、spooky-author-identification 等任务上达到 Above Median，在 nomad2018-predict-transparent-conductors 上达到 Bronze Medal。但表中也有多个 competition 没有有效提交，或相对 Baseline+SFT 分数下降。因而，MLE-bench 结果更适合解读为“长程训练带来跨任务泛化潜力”，而不是均匀、稳定地提升所有机器学习竞赛任务。

### Training Dynamics and Judge Analysis

训练曲线显示，SFT loss 平滑下降，为 RL 提供稳定初始化。RL 阶段在 PaperBench 上表现逐步提升，assistant turns 增加，policy entropy 下降。论文把这解释为 policy 更加确定、更有效地执行长程任务。

Judge analysis 比较了官方 o3-mini + rubrics 和 gpt-oss-120b + rubrics 对三个模型的排序：

| Model | o3-mini + rubrics | gpt-oss-120b + rubrics |
|---|---:|---:|
| Qwen3(235B) | 13.20 | 14.59 |
| GPT-4.1 | 29.90 | 33.55 |
| GPT-5 | 52.31 | 60.66 |

两种 judge 设置保持相同相对排序，这支持 gpt-oss-120b 作为训练阶段 reward source 的可用性。不过，这个实验只能说明粗粒度 ranking 一致，不能证明两者在任务级评分、细粒度 rubric credit 或训练 reward shaping 上完全等价。

### Case Study

附录中的 case study 把 KLong 在 PaperBench `lbcs` 任务上的 trajectory 拆解为几个阶段：paper reading、paper analysis、code development、running experiments、code refinement。这与论文主张一致：极长程 agent 的核心能力并不是单次回答更长，而是能在多阶段任务中维持目标、持续读取材料、构建代码、运行实验并修正结果。

论文也展示了 Claude 4.5 Sonnet(Thinking) 的一个 bad case：agent 杀掉了主进程，导致任务终止。这个案例说明，强模型在长程执行中仍可能因工具操作或进程管理失误造成不可恢复失败，长程任务评测不仅考察推理，也考察 execution discipline。

## 关键结论

KLong 对知识库最有价值的结论有四点。

第一，extremely long-horizon agent training 需要把任务时间、上下文长度和环境执行同时纳入训练设计。仅仅扩大 context window 或增加工具调用能力，无法自动解决 12 小时级任务中的 delayed reward、实验等待、状态维护和基础设施瓶颈。

第二，trajectory-splitting SFT 是一种有效的长轨迹蒸馏方式。它把完整长任务拆成可训练子轨迹，同时通过固定 paper-reading prefix 和 overlap 保留全局目标与局部连续性。它尤其适合“完整轨迹超出上下文，但早期任务理解必须反复可见”的场景。

第三，progressive RL 的核心贡献是 horizon curriculum。随着 timeout 从 2H 扩展到 4H、6H，模型逐步学习更长的 delayed reward 问题。这与普通 post-training 中只在固定 response length 上做 RL 不同，KLong 的 RL 时间尺度包含真实实验运行和环境等待。

第四，极长程 agent 评测和训练不可避免地依赖 judge 与 infrastructure。Rubric-based [[application/evaluation/llm-as-judge|LLM-as-a-Judge]]、沙箱、prompt caching、partial rollout、priority evaluation queue 都直接影响训练信号质量和系统吞吐。因此，在这类研究中，algorithm 和 system 很难完全分离。

## 局限与疑问

KLong 的结果很强，但需要注意以下边界。

**训练数据依赖强 teacher**：长轨迹由 Claude 4.5 Sonnet(Thinking) 蒸馏，并经过 judge score 与质量筛选。KLong 的 SFT 增益部分来自高质量 teacher trajectory，而不完全是切分策略本身。若 teacher 行为、rubric 或筛选标准改变，训练效果可能不同。

**reward credit assignment 仍然粗粒度**：RL reward 来自 judge 对最终复现产物的评分，然后用于多个子轨迹的 group-relative advantage。这比完全无监督更强，但并没有真正解决“哪一步阅读、哪一段代码、哪次实验导致成功”的细粒度 credit assignment。

**judge consistency 证据有限**：gpt-oss-120b 与 o3-mini 的模型排序一致，但只覆盖三个模型的总体分数。训练阶段长期优化 gpt-oss-120b reward 是否会引入特定 judge bias，仍需要更多任务级、rubric-level 和 human evaluation 验证。

**泛化证据并非完全均匀**：SWE-bench Verified 和 Terminal-Bench Hard 的绝对增益较小；MLE-bench 中有多个任务没有有效提交或分数下降。KLong 证明了 PaperBench 训练可以带来一定跨 benchmark 迁移，但不能简单推出所有长程 agent 任务都会显著受益。

**基础设施门槛很高**：10,000+ concurrent sandbox instances、25,000+ Docker images、priority judge queue 和大规模 rollout pipeline 说明该方法高度依赖工程系统。对普通研究团队而言，复现完整训练闭环的成本可能远高于复现算法公式。

**open-source 主张需要区分对象**：论文称 KLong 是 open-source LLM agent，但 report 中呈现的方法还包含数据生成、teacher distillation、sandbox、judge pipeline 和 training infrastructure。评估可复现性时，需要分别确认 model weights、training data、rubrics、scaffold 和 infra 组件的可获得性。

## 相关知识链接

- Agent 与执行框架：[[application/agents/agent|Agent]], [[application/agents/planning|Planning]], [[application/agents/memory|Memory]], [[application/tool-use/tool-calling|Tool Calling]]
- 长上下文与上下文管理：[[training/mid-training/long-context-training|Long Context Training]], [[application/rag/context-compression|Context Compression]]
- 后训练方法：[[training/post-training/sft|SFT]], [[training/post-training/rejection-sampling|Rejection Sampling]], [[training/post-training/ppo|PPO]], [[training/post-training/grpo|GRPO]]
- 评测与 judge：[[application/evaluation/benchmark|Benchmark]], [[application/evaluation/llm-as-judge|LLM-as-a-Judge]]
