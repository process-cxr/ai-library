---
title: "Learning from Teacher Continuations at Student States"
created: 2026-11-02
published: 2026-11-02
modified: 2026-11-02
type: source
status: processed
source_type: paper
area: sources
tags:
  - source
  - paper
  - distillation
  - on-policy-kd
  - teacher-intervention
  - sequence-level
  - agent-training
  - asynchronous-training
aliases:
  - OLIVE
  - Online Interventional Distillation
source_url: https://arxiv.org/abs/2609.36246
pdf_url: https://arxiv.org/pdf/2609.36246v3
paper_date: "2026-09"
paper_order: "36246"
---

# Learning from Teacher Continuations at Student States

## 基本信息

- 标题：[Learning from Teacher Continuations at Student States](https://arxiv.org/abs/2609.36246)
- 作者：Haojin Wang、Dylan Zhang、Huaibo Chen、Suhao Yu、Yihang Sun、Zhanyang Jin、Jiaying Ye、Dianqi Li、Prasanna Sattigeri、Kamal Youcef-Toumi、Hao Peng
- 机构/来源：University of Illinois Urbana-Champaign、Massachusetts Institute of Technology、University of Pennsylvania、University of Washington、IBM
- 发表：arXiv:2609.36246v3
- 日期：2026-09
- 代码/数据：[OLIVE 项目主页](https://dylanzsz.github.io/olive/)
- 相关主题：[[training/post-training/on-policy-kd|On-policy KD]]、[[training/post-training/sequence-level-distillation|Sequence-level Distillation]]、[[training/post-training/sft|SFT]]、[[application/agents/agent|Agent]]、[[fundamentals/information-theory/cross-entropy|Cross Entropy]]

## 研究问题：纠正一个 Token，还是示范接下来怎么走

蒸馏中有两个经常被合并讨论的问题：**教师提供什么监督，以及监督出现在什么状态上。** 固定教师回答提供了完整示范，却不一定覆盖学生自己犯错后会走到的状态；逐 token 分布蒸馏覆盖学生状态，却不一定示范接受某个纠正之后的后续路径。

论文提出 Online Interventional Distillation（OLIVE，在线干预式蒸馏），将两者组合：

1. 由不断更新的学生生成一个前缀。
2. 教师接着这个前缀，自回归生成一段后续内容。
3. 学生用 cross-entropy（CE，交叉熵）学习教师生成的后缀；问题和学生前缀只作上下文，不作预测目标。
4. 学生更新后重新采样前缀，不把训练永久绑定到初始学生的轨迹。

在多轮任务中，教师不只是续写文本，还接管学生已经交互过的环境，自己的动作继续执行并获得真实 observation。学生由此学习“从我实际到达的局面，接下来应该怎样做”。

论文在可调难度推理任务、五个 Agent 环境和一个小规模终端任务子集上评测，并展示只获得教师文本时的在线训练。核心贡献不是一个新的复杂 loss，而是**重新组织示范的起点、生成者与刷新时机**。

## 为什么离线 SFT 和 Token-level OPD 仍有缺口

### 离线教师轨迹没有覆盖学生自己造成的上下文

有监督微调（Supervised Fine-Tuning，SFT）或离线序列蒸馏通常预先收集教师回答，再用固定文本训练。训练时模型看到教师的历史 tokens，推理时看到自己的历史 tokens，两者可能产生 sequential covariate shift，即序列决策中的状态分布偏移。

对多轮 Agent，偏移不只发生在语言表达上：学生开错门、调用错工具或重复无效动作，会使环境和历史轨迹都不同于教师从初始状态出发的示范。

### 逐 Token OPD 没有真的沿教师建议继续生成

On-policy distillation（OPD，基于学生自身采样的策略蒸馏）通常在完整学生 rollout 上，让教师计算 token probabilities，再作分布匹配或采样式更新。即使教师在某个位置建议另一种选择，后面评分的上下文仍是**原学生轨迹**，没有沿那个建议重新生成。

一个独立示意例子：

```text
学生轨迹：错误假设 A -> 按 A 推导 -> 沿错误路径得到答案
Token-level OPD：评价 A 和后面的每个学生 token
OLIVE：保留学生前缀 -> 教师指出 A 有误 -> 展示替代推导如何继续
```

普通 OPD 可以提高替代 token 的概率，也并非完全没有纠错能力；差别在于它没有在这条训练样本里实际走出那条替代路径。OLIVE 让教师后续 tokens 依赖自己刚作出的选择，形成连续的纠正示范。

### 学生到达的状态可能已经不可恢复

教师接管不等于回滚。它接受既有前缀和环境状态，能自我纠正或另找路径，但不能保证撤销不可逆操作。方法依赖教师在这类状态上仍能生成有价值的后续行为；若状态已无法完成任务，教师续写也可能失败。

### 与模仿学习的关系

OLIVE 延续了 learner roll-in / expert roll-out 的思路：学生先到达状态，专家示范后续动作。它也遵循 DAgger（Dataset Aggregation，数据集聚合）强调的学习者状态覆盖原则。

不过本文算法是特定的前缀接续与 suffix CE 训练，并不等于复现 DAgger 的全部数据聚合、混合策略或理论保证。机制背景说明方法的来源，不能自动移植原算法的所有结论（§2）。

## 主算法：Student Prefix、Teacher Continuation 与 Suffix CE

### 采样与接续

设当前学生为 $\pi_\theta$，教师为 $\pi_T$，问题 $x\sim\mathcal D$。学生先生成长度为 $k$ 的前缀：

$$
p=y_{1:k}\sim\pi_\theta(\cdot\mid x).
$$

教师接着生成最多 $M$ 个教师 tokens：

$$
\widetilde y\sim\pi_T(\cdot\mid x,p),
\qquad |\widetilde y|_T\le M.
$$

教师不必完成整条推理，也不必生成一个可通过最终 verifier 的答案。主推理实验保留所有采样前缀，不按正确性过滤，用短接续控制教师生成成本（§3.1）。

### 学生只学习教师后缀

将教师文本用学生 tokenizer 重新编码为 $\widetilde y_{1:\ell}$，训练目标为论文式 (1)：

$$
\mathcal L_{\mathrm{suffix}}(\theta;x,p,\widetilde y)
=-\sum_{j=1}^{\ell}
\log\pi_\theta(\widetilde y_j\mid x,p,\widetilde y_{<j}).
$$

教师生成预算 $M$ 以教师 tokens 计，$\ell$ 以学生 tokens 计；tokenizer 不同，两者一般不相等。训练时需要保留实际文本连续性和交接位置，不能把教师 token IDs 直接拼进不同词表的学生输入。

```text
问题 / 系统指令         attention context；不计算 CE
学生生成前缀           attention context；不计算 CE
教师生成后缀           attention context；计算 CE
```

mask 学生前缀意味着不把可能错误的学生动作当作模仿目标，不意味着输入前缀完全不影响梯度。后缀预测仍依赖前缀的模型表示，共享参数通过后缀 loss 更新。

### 优化形式像 SFT，数据获取是在线的

对给定样本，更新就是普通 teacher-forced CE：学生在教师后缀的历史上预测下一个教师 token。没有对离散采样过程求导，也没有在该目标中引入策略梯度 reward、critic 或教师 logits。

理想同步数据分布可写为：

$$
x\sim\mathcal D,
\quad p\sim\pi_{\theta_t}(\cdot\mid x),
\quad \widetilde y\sim\pi_T(\cdot\mid x,p).
$$

采样后把 $(x,p,\widetilde y)$ 固定，再更新学生。这里“在线”指下一批示范起点随学生变化，不是 loss 本身换成了强化学习。

它也不是所有监督位置都严格来自学生状态：接管点之前来自学生；后缀内的后续状态沿教师自己的生成或动作推进。因此可称为**在学生可达起点上学习教师接续**，不应直接等同于全程学生 rollout 上的 reverse-KL OPD。KL 是 Kullback-Leibler divergence（KL 散度），reverse KL 在这里表示从学生分布到教师分布的比较方向。

### 与三类常见训练方式比较

| 方法                   | 前缀来源                 | 后续路径由谁决定 | 学习目标                    |
| ---------------------- | ------------------------ | ---------------- | --------------------------- |
| 离线教师序列 SFT       | 固定教师示范             | 教师             | 固定回答的 CE               |
| Token-level OPD        | 当前学生完整轨迹         | 学生             | 同前缀分布或 log-ratio 监督 |
| 固定学生前缀的教师接续 | 初始或旧学生，一次性收集 | 教师             | 固定教师后缀的 CE           |
| OLIVE                  | 持续刷新的学生前缀       | 接管后由教师     | 在线收集后缀的 CE           |

因此它同时具有文本示范蒸馏和学生状态覆盖的特征，但不需要教师暴露完整分布。异构 tokenizer 和只提供生成文本的教师也能使用，前提是接口允许传递所需上下文和获得连续接续。

## 多轮 Agent：交接的是环境状态，不只是对话文本

### 教师在同一个环境里继续行动

多轮设置中，$k$ 和 $M$ 的单位改为 interaction turns：

```text
初始环境
  -> 学生执行 k turns
  -> 保留学生动作和真实 observations
  -> 教师读取完整历史，接管当前环境
  -> 教师动作执行 -> 新 observation -> 下一个教师动作
  -> 收集最多 M 个教师 turns
```

如果只让教师根据一段静态历史编出后续工具结果，就不等价于该协议。环境中的对象、文件、权限和任务进度必须与学生实际执行后的一致；跨进程接管时则需要可恢复状态。

### Loss Mask 的归属

$$
\mathcal L_{\mathrm{agent}}
=-\sum_{t\in\mathcal T_{\mathrm{teacher}}}
\log\pi_\theta(a_t^T\mid h_t).
$$

上式按教师动作文本作概念化表达：保留完整交互历史作为上下文，mask 学生 turns 和所有环境 observations，只对教师生成的动作/响应 tokens 计算 CE（§3.2）。本文 AgentGym 任务使用 ReAct（Reasoning and Acting，推理与行动结合）的文本接口，具体 assistant 字段的切分仍需服从实现。

这些 observations 是教师动作造成的真实反馈，虽不作为生成目标，仍决定教师下一个动作和学生所学习的条件。方法训练的是响应环境的 policy，而不是学习把所有日志照抄出来。

### 一个具有代表性的失败恢复案例

ScienceWorld 的任务要求依次关注最长寿和最短寿动物。学生在温室里循环执行 `go to outside`、`open outside`、`look around`，错误地把地点当成可打开对象，10 turns 后得分仍为 0。

教师接管后先执行 `open door to the outside`，再进入室外、查看动物并完成两次 focus，5 个 turns 内达到 score 1。示范从学生自己造成的失败局面开始，而不是从一个从未犯错的初始轨迹开始（Appendix B.1）。

这个例子解释了为什么局部接续可以提供恢复技能，但它是选定案例，不能代替对所有失败状态可恢复性的统计。

## 异步实现：用有限 Staleness 换取计算重叠

同步 OLIVE 需要依次等待学生前缀、教师接续、学生更新。教师生成期间学生设备可能空闲。异步版本让教师处理当前批次时，学生先采样后面的前缀，用之前已完成的教师轨迹进行训练（Figure 2）。

论文用 asynchronous depth $d$ 限制“生成前缀到用这条接续训练”之间最多发生的学生更新次数，主推理实验取 $d=3$。这不是同时加载三位教师，也不是把一条数据重复训练三轮。

异步后，训练样本可能来自 $\pi_{\theta_{t-\Delta}}$，而非正在更新的 $\pi_{\theta_t}$，其中 $\Delta\le d$。因此有受控的 prefix-distribution lag，不能仍把每条前缀都描述成完全当前策略采样。

论文仍对这些固定接续执行相同 CE，没有报告 PPO（Proximal Policy Optimization，近端策略优化）式 importance correction。这里可以使用较旧前缀，是因为优化的是给定上下文下的示范似然；它并不自动消除旧状态分布降低监督相关性的风险。

## 推理主实验：RLVE 上的短接续蒸馏

### 数据、模型与预算

RLVE 是提供可调难度问题和可验证结果的合成推理环境集合。作者选出 18 个较难环境，每个 500 道训练题、10 道同难度测试题，共 9,000 道训练题和 180 道测试题（§4.1）。

学生为 thinking-enabled 的 Qwen3-1.7B、Qwen3-4B，统一教师为 Qwen3-4B-Thinking-2507。两种规模分别训练，不是把同一个学生继续缩放。

| 设置                      | OPD 对照                   | 主实验 OLIVE           |
| ------------------------- | -------------------------- | ---------------------- |
| 学习信号                  | Top-16 KL 近似             | 教师后缀 CE            |
| 学生 rollout              | 最多 7,168 response tokens | 前缀 4,096 tokens      |
| 教师新生成                | 不生成接续，仅提供分布监督 | 最多 1,024 教师 tokens |
| 每 prompt 学生 rollout 数 | 4                          | 4                      |
| Batch size                | Global/minibatch 均为 64   | Train batch 为 64      |
| 学习率                    | $10^{-6}$                  | $10^{-5}$              |
| Epochs                    | 1                          | 1                      |
| 异步深度                  | 不适用                     | 3                      |

OPD 的训练 temperature 为 1，附录标注 `KL Coefficient = 0.0`；不能据此认为它没有教师 KL 监督，更合理的区分是额外 reference/trainer KL 与蒸馏本身不是一个概念。所读配置表没有完整展开这个字段的代码语义（Table 3、Table 4）。

离线 Teacher-Gen. baseline 使用未按正确性筛选的教师完整轨迹进行 CE 训练。论文以每 rollout 蒸馏内容上限 7,168 tokens 控制比较预算，而 OLIVE 实际只监督短教师后缀。上限相同不意味着有效监督 tokens 数、生成计算量或学习率完全相同；完整分布 OPD 也没有在这里成为对照。

### Pass@8 与 Avg@8

每道题采样八次：

- **pass@8**：八次里至少一次答对的题目比例；
- **avg@8**：平均这八次回答各自是否正确，反映单次生成正确率的采样估计，不是选八次中的最优答案。

Table 1 的结果为：

| 方法                 | 1.7B pass@8 | 1.7B avg@8 | 4B pass@8 | 4B avg@8 |
| -------------------- | ----------: | ---------: | --------: | -------: |
| Original             |        11.1 |        3.3 |      46.1 |     18.1 |
| Teacher-Gen. 离线 CE |        15.0 |        5.6 |      53.3 |     19.9 |
| OPD                  |        14.4 |        4.6 |      45.6 |     21.3 |
| OLIVE，异步 $d=3$    |        19.4 |        7.8 |      52.2 |     23.5 |

按表中四舍五入值，OLIVE 相对原学生的 avg@8 分别提高 4.5、5.4 点，pass@8 提高 8.3、6.1 点。相对 OPD，avg@8 分别提高 3.2、2.2 点。

**4B 的 pass@8 最高是 Teacher-Gen. 的 53.3，不是 OLIVE 的 52.2。** OLIVE 在 avg@8 上更强，不能因此写成所有推理指标都严格领先。两个指标的差别也提示，平均可靠性与多采样覆盖率不能混为一谈。

论文的 Top-K overlap 诊断在 1.7B OPD 过程中仅从 0.707 到 0.713，作者认为它与教师-学生 thinking 行为差异持续存在相符。这是诊断相关性，不是证明所有 OPD 失败都由一个 overlap 指标决定（Figure 3）。

## 多轮 Agent 主实验

### 环境和训练协议

学生为 Qwen3-1.7B，教师为 Qwen3-32B。五项任务来自 AgentGym：

- **ALFWorld**：文本化家庭环境中的交互任务。
- **ScienceWorld**：需要操作对象和完成实验目标的科学环境。
- **TextCraft**：按合成配方进行文本化物品制作。
- **BabyAI**：根据语言目标完成导航与物体交互。
- **SearchQA**：通过搜索 Wikipedia 回答问题。

学生、教师和评测共用环境 prompts 与 action parsers，使用 ReAct 协议。最大 turns 分别为 30、30、30、20、16。ALFWorld、ScienceWorld、SearchQA 训练 1 epoch，BabyAI 与 TextCraft 为 3 epochs（§4.2、Appendix D）。

OLIVE 在 ALFWorld、ScienceWorld 使用 10 个学生 turns，在较短任务中使用 5 个学生 turns，教师最多接续 5 turns。SearchQA 另构造 6K 训练问题、400 个 held-out 评测问题；其余沿用相关 AgentGym 设置，论文未在同一节逐项列出完整数据规模。

对照包括普通 OPD、TCoD 的 B2F/F2B 版本，以及 GuidedOPD。TCoD 研究 Temporal Curriculum in On-Policy Distillation，即多轮蒸馏中的时间课程；GuidedOPD 使用 curriculum turn-level guidance。所读论文没有把各对照的全部 loss、课程和训练细节重新展开，不能仅凭名称补出严格等价的实现。

### 成功率结果

Table 2 报告 avg@4 的 Success Rate（SR，任务成功率），即对多次尝试平均，而非四次中任一次成功的 pass@4：

| 方法      | ALFWorld | ScienceWorld | TextCraft | BabyAI | SearchQA |
| --------- | -------: | -----------: | --------: | -----: | -------: |
| 原学生    |    19.38 |         0.12 |     23.00 |  38.33 |    30.50 |
| 教师      |    52.12 |        15.62 |     85.50 |  83.33 |    55.06 |
| OPD       |    22.25 |         0.00 |     29.50 |  43.06 |    29.56 |
| TCoD-B2F  |    37.10 |         0.75 |     39.50 |  66.30 |    37.69 |
| TCoD-F2B  |    28.00 |         0.50 |     45.50 |  62.50 |    37.75 |
| GuidedOPD |    27.12 |         0.75 |     45.50 |  67.78 |    37.50 |
| OLIVE     |    40.00 |         7.50 |     55.25 |  67.50 |    39.06 |

OLIVE 在五项中的四项最高；BabyAI 略低于 GuidedOPD 的 67.78。不能按正文概括写成“五项都超过所有基线”。

最显著的困难环境差异是 ScienceWorld：原学生 0.12、OPD 0.00，OLIVE 达到 7.50。不过教师也只有 15.62，绝对能力仍然有限；这个结果不表示环境中的大部分任务已被可靠解决。

相对原学生，各环境增益依次为 20.62、7.38、32.25、29.17、8.56 个百分点。它们与“相对其他蒸馏基线提升”是不同分母，不能混用论文摘要中的笼统百分比表述。

### Turns 的变化不能独立当作成功效率

| 方法   | ALFWorld 平均 turns | ScienceWorld | TextCraft | BabyAI | SearchQA |
| ------ | ------------------: | -----------: | --------: | -----: | -------: |
| 原学生 |               26.97 |        28.63 |     23.89 |  14.52 |    11.91 |
| OPD    |               26.28 |        28.77 |     22.28 |  14.32 |    12.07 |
| OLIVE  |               23.15 |        23.64 |     16.45 |   9.98 |    11.13 |

平均交互长度整体缩短，但这不是在相同成功任务集合上测的条件均值，也并非对所有强对照都更短。例如 BabyAI 的 GuidedOPD 平均为 8.30 turns、TCoD-B2F 为 9.03，均短于 OLIVE 的 9.98。评估恢复与效率应同时保留成功率、终止规则和失败轨迹长度。

## 效率实验：异步重叠带来什么

在同一组 9K RLVE 题目上，以 Qwen3-4B 为学生、Qwen3-4B-Thinking-2507 为教师，用 8 张 H200 GPU（Graphics Processing Unit，图形处理器）比较。Figure 4 标注为总 GPU-hours：

| 配方        | GPU-hours | Pass@8 | Avg@8 |
| ----------- | --------: | -----: | ----: |
| OLIVE async |      29.8 |   52.2 |  23.5 |
| OLIVE sync  |      39.1 |   54.4 |  23.1 |
| OPD         |      41.3 |   45.6 |  21.2 |

异步相对同步 GPU-hours 降低约 23.8%，相对 OPD 约 27.8%，正文取整为 28%。同步 OLIVE 的 pass@8 高 2.2 点，异步 avg@8 却略高；不能简单描述成所有质量指标都只有同方向的轻微下降。

Table 1 的 OPD avg@8 为 21.3，而 Figure 4 标为 21.2，报告分别沿用对应来源，不据此解释 0.1 点差异。

这组结果说明短接续与流水可以抵消在线教师生成的一部分成本，但不保证任何教师/学生比例、响应长度或服务接口下都更便宜。固定 token 上限、不同学习率、Top-16 OPD 近似和设备部署方式也属于对照条件，不能把时间差全部归因于 CE 比 KL 更快。

## 进一步分析：在线刷新、遗忘和可塑性

### 正确筛选的分析实验，不是主实验短接续

§5.1 为隔离前缀来源，重新设计了三组带正确性筛选的训练：

1. **Full SFT**：每题从教师生成 4 条完整推理，保留 verifier 判正确的回答。
2. **OEC**：On-Policy Expert Corrections（基于学习者状态的专家纠正）式离线对照；先从初始学生每题收集 1 个前缀，教师从它接续 4 次，筛出正确结果，然后训练固定数据。
3. **OLIVE 分析变体**：随更新后的学生在线刷新前缀，教师生成完整推理、每题 4 次 rollout，并只对筛选后的正确教师后缀计算 CE。

所以这里不再使用主实验“只接续 1,024 tokens、无需完成正确答案、不筛前缀”的完整配方。它支持在线刷新起点有帮助，但 Figure 5 的 33.3 不能直接当作主实验 19.4 的同条件后续结果。

Figure 5 的 1.7B 结果为：

| 设置                | RLVE pass@8 | RLVE avg@8 |
| ------------------- | ----------: | ---------: |
| Plain student       |        11.1 |        3.3 |
| Full SFT，offline   |        23.9 |        8.1 |
| OEC，固定学生前缀   |        30.0 |       10.7 |
| OLIVE，在线刷新前缀 |        33.3 |       12.2 |

固定学生前缀已经优于纯教师轨迹；持续刷新又进一步提升。这说明“包含学生上下文”与“持续跟随学生变化”是两个不同增量。Figure 6 对 OEC 的图例写作 online，但 §5.1 明确将其定义为一次性收集的离线对照，本文依照实际数据获取协议区分。

### 通用能力平均下降更少，不是没有遗忘

作者额外评估 AIME25 数学、LiveCodeBench v6 代码、IFEval 指令遵循和 GPQA Diamond 科学知识。AIME 是 American Invitational Mathematics Examination（美国数学邀请赛）；GPQA 是 Graduate-Level Google-Proof Q&A，Diamond 为其中的高质量子集。

Figure 6 的四项通用指标平均变化为：Full SFT -3.4、OEC -1.3、OLIVE -0.9 个百分点。它支持当前分析设置中在线刷新对已有能力的破坏较小，而不是每项能力都不退化。

该图紫柱的数值 3.3/8.1/10.7/12.2 与 Figure 5 的 avg@8 一致，虽然图注将其称为 pass@8；本报告按对应数值与图内标识区分，不将两种指标混写。

### Text-only Teacher 与持续学习

ScienceWorld 的另一组实验以 Qwen3-1.7B 为学生、GPT-5.4-mini 为教师。仅使用教师文本，不要求其 token probabilities：

- 离线对照先从教师每个训练任务收集 1 条 episode，反复训练固定轨迹。
- OLIVE 每轮从变化中的学生前缀重新获得教师接续。
- 两者训练 5 epochs，比较各阶段学生独立评测成功率。

Figure 7a 中离线约在第 2 epoch 后停滞，而 OLIVE 早期未领先，之后继续提高，到第 5 epoch 约为 26%，离线约为 13%。正文称超过 13%；从图中成功率轴看应理解为**约 13 个百分点的绝对差异**，不是相对只提高 13%。图中未标完整精确值，因此不把读图近似写成精确表格。

Figure 7b 将多 rollout 在线采样与相应离线重复训练作比较，在线在较大采样量下更好。但 OLIVE 在每次刷新时还要新增教师调用，epoch 或更新数相近并不自动意味着教师生成量、费用完全相同。API（Application Programming Interface，应用程序接口）教师的成本应另行计入。

### 顺序训练与 Held-out 环境

Figure 8 用 Qwen3-4B 按 BabyAI→TextCraft→SearchQA→ALFWorld→ScienceWorld 顺序，每个环境训练 5 epochs。最后评测相对离线 SFT 的 SR 差异为：

| 环境         | OLIVE 相对离线的成功率差异 |
| ------------ | -------------------------: |
| BabyAI       |                     约 0.0 |
| TextCraft    |                       +8.5 |
| SearchQA     |                       -0.6 |
| ALFWorld     |                       +4.8 |
| ScienceWorld |                      +13.0 |
| 五环境平均   |                       +5.1 |

后续任务特别是 ScienceWorld 获益较大，但 SearchQA 略低；这些终点表现是保留学习能力的证据之一，不是参数级 plasticity 的直接测量，也未提供全部学习阶段×任务的遗忘矩阵。

Appendix C.2 的另一组实验则使用 1.7B 学生与 GPT-5.4-mini，只顺序训练 BabyAI、TextCraft、SearchQA、ScienceWorld，将 ALFWorld 留作未训练环境。该 ALFWorld 相对离线提高 2.8 点，已训练的 ScienceWorld 提高 13.6 点。它与 Figure 8 的 4B 五环境设置不同，不能合并成同一次泛化实验。

## 干预时机与终端任务扩展

### 学生前缀不是越长越好

在 ALFWorld 上固定教师接续 5 turns，改变学生 turns（Figure 9）：

- 不使用学生前缀、直接教师 5 turns，成功率从图中约读为 33.4%；
- 加入 5-20 个学生 turns 的结果约为 35.6%-40.0%；
- 10+5 配置最好，为 40.0%；15+5 为 35.6%，20+5 为 39.0%。

前缀太短可能偏离学生真实失败状态，太长可能增加不可恢复错误或教师接管难度。该消融支持干预时机是需要选择的超参数，不证明固定 10 turns 对其他环境也最优。

### 终端 Agent 仍是初步证据

Appendix C.3 从 TMax 终端任务数据采样 500 个训练样本，使用 Qwen3.5-2B 学生与 Qwen3.5-9B 教师：

- OPD：学生执行前 20 turns，教师监督这些学生轨迹。
- OLIVE：学生 10 turns，教师接续 10 turns。
- 两者均为每 prompt 8 rollouts、batch size 16。

随机选择 TBLite 中 50 个任务，每题 4 次尝试，报告 pass@4：Base 10.0%、OPD 10.0%、OLIVE 12.5%（Figure 11）。这次是 best-of-four 型 pass 指标，不能和 AgentGym 的 avg@4 直接比较。

此外，若严格是 50 个任务、每题恰好 4 次，按“至少一次成功”直接统计时，任务比例应以 2 个百分点递增；论文未在该处解释 12.5% 的有效分母或额外聚合方式。因此保留报告值，不据此反推成功任务数。作者也将这个小子集结果定位为 preliminary，不能当作已确立的广泛终端任务提升。

## 方法边界与复现注意事项

### 它改变了监督轨迹，但没有抹掉学生过去的错误

学生前缀中的错误不参与直接模仿，教师后缀却必须在这个上下文上合理继续。接管可以处理可恢复失败，无法保证修复已经销毁资源、永久改变状态或丢失关键信息的前缀。

主算法保留所有前缀且允许不完整、未验证的教师后缀，所以也可能学习教师误判、表面合理的修复和无效动作。论文展示了成功案例与总体任务效果，并未提供教师接管成功率、失败前缀类型覆盖率等完整诊断。

### 只用文本降低接口门槛，不代表普通 Completion API 总能无缝接管

不同 tokenizer 可以经文本重新编码，但仍需保持前缀边界、chat template、thought/action 格式与上下文长度兼容。多轮任务还需要教师接入当前环境并遵守同一 parser。

纯文本 CE 不包含教师对未采样替代 tokens 的显式概率信息。它是一个示范学习目标，不应被写成无需 logits 的精确 reverse-KL 等价实现。黑盒教师不暴露概率时，本文的 distribution-matching OPD 对照无法直接运行，但其他基于文本的在线模仿方法仍然可能适用。

### 比较并非只改变一个 Loss

推理对照同时涉及完整/部分生成、监督 token 数、Top-16 概率近似、学习率 $10^{-6}$ 与 $10^{-5}$、异步程度等差异。Agent 训练设置也没有逐项给出全部 baseline 的优化器和有效监督量。

因此当前结果比较的是所给 recipe，而不是在所有优化配置下证明 CE 必然优于 KL。§5.1 的在线/固定前缀实验更有针对性，但依然使用完整接续和正确性筛选，不能替代所有组件的正交消融。

### Synthetic 不等于能够证明零污染

RLVE 随机生成实例和可调难度能减少固定公开题库记忆带来的影响，但模型仍可能见过任务规则、算法或相近模板。180 道测试题和有限环境集合，也不覆盖全部推理能力。

### 异步与可塑性结论仍有统计范围

主实验固定 $d=3$，没有给出完整不同 lag 的稳定性曲线。异步数据应记录产生前缀的学生版本、教师版本、交接点和使用时的 lag；不能只记一条拼接文本就认为严格 on-policy。

论文部分图包含误差条，但未为所有主表差异报告统一的多训练种子区间。较少遗忘、较好后续任务学习是所测协议下的经验结果，不能当作长期持续学习的普遍保证。

## 关键结论

OLIVE 将蒸馏从“只复述教师的完整轨迹”或“只评价学生已经走出的轨迹”，转为“在学生实际到达的状态上，让教师示范怎样继续”。用教师后缀 CE，使这条训练路径可以通过文本接口实现；持续刷新前缀，使监督跟随学生而变化。

最应区分的三个层次是：**状态起点来自谁，接管后的路径由谁生成，以及哪些 tokens 真正进入 loss。** 理解这三点，才能解释它为什么兼具在线数据收集与 SFT 式优化，也才能判断它在失败恢复、异步训练和黑盒教师上的实际适用性。

## 相关知识链接

- [[training/post-training/on-policy-kd|On-policy KD]]
- [[training/post-training/sequence-level-distillation|Sequence-level Distillation]]
- [[training/post-training/sft|SFT]]
- [[application/agents/agent|Agent]]
- [[fundamentals/information-theory/cross-entropy|Cross Entropy]]
