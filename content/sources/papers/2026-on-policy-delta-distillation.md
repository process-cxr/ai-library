---
title: "On-Policy Delta Distillation"
created: 2026-07-22
published: 2026-07-22
modified: 2026-07-22
type: source
status: processed
source_type: paper
area: sources
tags:
  - source
  - paper
  - distillation
  - on-policy-kd
  - post-training
  - reasoning
  - code
aliases:
  - OPD2
  - On-Policy Delta Distillation
source_url: https://arxiv.org/abs/2607.15161
paper_date: "2026-07"
paper_order: "15161"
---

# On-Policy Delta Distillation

## 基本信息

- 来源：arXiv:2607.15161v1
- 标题：On-Policy Delta Distillation
- 作者/机构：Byeongho Heo, Jaehui Hwang, Sangdoo Yun, Dongyoon Han / NAVER AI Lab
- 日期：2026-07-16
- 链接：https://arxiv.org/abs/2607.15161
- 代码：https://github.com/naver-ai/opd2
- 相关 topic：[[training/post-training/on-policy-kd|On-policy KD]]，[[training/post-training/knowledge-distillation|Knowledge Distillation]]，[[training/post-training/grpo|GRPO]]

这篇论文讨论的是 reasoning model 后训练中的 on-policy distillation。OPD（On-Policy Distillation）的基本做法是：student 先根据当前 policy 生成 rollout，再由 teacher 对 student 已经采样出的 token 提供 token-level reward。这样可以避免完全依赖 reward model / verifier，也比普通 SFT 更贴近 student 自己会访问到的状态分布。

论文提出的问题是：传统 OPD 直接用 teacher 与 student 的 log probability 差作为 reward，但这个信号混合了两类东西：

- teacher 在 pre-training / base 阶段已经具备的自然语言偏好、常见表达和普通 next-token prior；
- teacher 经过 reasoning tuning 后新增或强化的推理能力。

OPD2 的核心想法是把第二类信号单独抽出来。它不用 `teacher - student` 作为主要 reward，而是使用 `teacher - teacher_base` 的 delta signal。这个 delta 被作者解释为 teacher 从 base model 到 reasoning-tuned model 的“学习轨迹”，更直接对应 reasoning capability 的增量部分。

如果只保留一个核心观点：**蒸馏 reasoning 能力时，不一定要模仿 teacher 的完整输出分布；更有针对性的信号可能是 teacher 相对自身 base model 发生了哪些概率偏移。**

## 研究问题

Sequence-level distillation 和 SFT 使用 teacher 生成的完整答案训练 student，优点是简单稳定，缺点是数据是 off-policy 的：student 训练时看到 teacher outputs，但推理时会进入自己的状态分布。On-policy distillation 则让 student 自己生成 response，再让 teacher 对这些 sampled tokens 给出 dense token-level supervision。

传统 OPD 的 token reward 可以写成：

$$
R_t^{\mathrm{OPD}}
=
\log \pi^\*(y_t\mid x,y_{<t})
-
\log \pi_\theta(y_t\mid x,y_{<t})
$$

其中 $\pi^\*$ 是 teacher，$\pi_\theta$ 是 student。这个 reward 来自 KL objective 的 sampled-token 梯度形式：teacher 比 student 更偏好的 sampled token 会被增强，teacher 更不偏好的 sampled token 会被压低。

但论文认为，这个设计没有区分“teacher 作为语言模型本来就喜欢什么”和“teacher 经过 reasoning tuning 后才更喜欢什么”。对于 reasoning post-training，真正想转移的是后者。

因此论文要回答的问题是：

1. teacher 与 teacher-base 的差值是否能成为更好的 on-policy distillation reward？
2. delta signal 在 token 粒度上和传统 OPD reward 有什么不同？
3. 只用 delta signal 是否会带来收敛或稳定性问题？
4. 在 math、science、code 三类 reasoning 任务上，OPD2 是否稳定优于 OPD 和 ExOPD？

## 核心主张

论文的核心主张有三层。

第一，reasoning tuning 的增量信号可以通过 teacher 和 teacher-base 的 logprob 差来近似：

$$
R_t^\Delta
=
\log \pi^\*(y_t\mid x,y_{<t})
-
\log \pi^\*_{\mathrm{base}}(y_t\mid x,y_{<t})
$$

这里 $\pi^\*_{\mathrm{base}}$ 是 teacher 在 instruction / reasoning tuning 之前的 base checkpoint。这个差值不是比较 teacher 和 student，而是比较 teacher 在 tuning 前后的偏好变化。

第二，delta signal 在 token 层面更像 reasoning tuning 的方向信号。论文通过 word cloud、人工构造错误推理样例和词级统计分析观察到：相比传统 OPD，delta 更倾向增强 `hence`、`however`、`thus`、`regardless` 等推理连接词，也更容易压低一些泛化叙述、探索性或错误推理中出现的 token。

第三，纯 delta reward 不足以直接替代 OPD，因为它不包含 student probability，理论收敛点可能走向 maximum-reward token 的 one-hot 分布。OPD2 因此加入 centering 和 joint condition：用 delta 控制梯度幅度，但只在 delta signal 和传统 OPD signal 方向一致时更新。

## 方法与机制

### OPD 的 RL-like 形式

OPD 可以看作一种不显式训练 reward model 的 RL-like post-training。Student 生成 response：

$$
y\sim \pi_\theta(\cdot\mid x)
$$

Teacher 对 sampled token 给 reward，student 用 policy-gradient 风格更新：

$$
\nabla_\theta J_{\mathrm{OPD}}(\theta)
=
\mathbb{E}_{x,y}
\left[
\sum_{t=1}^{T}
R_t
\nabla_\theta \log \pi_\theta(y_t\mid x,y_{<t})
\right]
$$

这个形式和 [[training/post-training/grpo|GRPO]] / PPO 类训练框架很接近，因此论文实现时基于 TRL 的 `GRPOTrainer` 改造：只采单个 completion，打开 token-level signals，关闭 group normalization。

### Delta Signal

OPD2 把主要 token reward 改成：

$$
R_t^\Delta
=
\log \pi^\*(y_t\mid x,y_{<t})
-
\log \pi^\*_{\mathrm{base}}(y_t\mid x,y_{<t})
$$

直觉上，如果 reasoning-tuned teacher 比 base teacher 更喜欢某个 token，这个 token 可能与 reasoning tuning 学到的行为有关；如果 base teacher 更喜欢某个 token，而 reasoning teacher 不再偏好它，那么这个 token 更像旧有语言 prior 或无关表达。

这个设计和 ExOPD 都使用 teacher-base，但目标不同：

- ExOPD 使用 teacher-base 构造 extrapolation signal，相当于放大 teacher 相对 base 的偏移，让 student 往 teacher 之外走得更远；
- OPD2 不做 teacher output extrapolation，而是直接把 `teacher - teacher_base` 当成主 reward。

### Centering

论文指出，on-policy reward 中与 action 无关的 bias 对 policy gradient 没有作用，因此先对 OPD reward 和 delta reward 做 expected reward subtraction：

$$
A_t^{\mathrm{OPD}}
=
R_t^{\mathrm{OPD}}
-
\mathbb{E}_{\tilde{y}_t\sim\pi_\theta}
\left[
\log\pi^\*(\tilde{y}_t)
-
\log\pi_\theta(\tilde{y}_t)
\right]
$$

$$
A_t^\Delta
=
R_t^\Delta
-
\mathbb{E}_{\tilde{y}_t\sim\pi_\theta}
\left[
\log\pi^\*(\tilde{y}_t)
-
\log\pi^\*_{\mathrm{base}}(\tilde{y}_t)
\right]
$$

实现时，为了节省显存，期望只在 student 分布的 top-k tokens 上计算，论文使用 `k=1024`。

Centering 的作用不是论文性能提升的最大来源，但它让 reward direction 更清晰，也避免 delta signal 中整体偏置影响训练解释。

### Joint Condition

纯 delta reward 不包含 student probability，因此如果直接最大化，可能把 student 推向 delta 最大的 token，而不是让 student 对齐 teacher。为避免这个 convergence-point 问题，OPD2 定义最终 advantage：

$$
A_t^{\mathrm{D2}}
=
\begin{cases}
A_t^\Delta, & A_t^\Delta A_t^{\mathrm{OPD}} > 0 \\
0, & \mathrm{otherwise}
\end{cases}
$$

也就是说，只有 delta signal 和传统 OPD signal 方向一致时，token 才参与更新。这样做保留了 delta signal 的 reasoning 增量信息，同时用 OPD signal 作为约束，防止 student 被 teacher-base 差值带偏。

当 student 已经和 teacher 相同，即 $\pi_\theta=\pi^\*$ 时，传统 OPD advantage 会趋于关闭，OPD2 也不会继续把 student 推向无限强化的 delta token。

## 实验设计

论文的实验覆盖三个 reasoning domain：

- Math：OpenMathReasoning 作为训练问题来源
- Science：OpenScienceReasoning-2 作为训练问题来源
- Code：OpenCodeReasoning 作为训练问题来源

训练集按 `1:1:1` 采样，构成 100k questions 的多域集合。训练时每个问题最多使用一次，实际训练少于 30k samples，相当于不到一个 epoch。论文明确说明，只使用这些数据集中的 questions，丢弃原始 reasoning traces 和 answers。

评测覆盖 14 个 benchmarks：

- Math：AIME24、AIME25、AMC23、HMMT25、MATH500、OlympiadBench、ReasoningGym Math
- Code：CodeContests、CodeForces、LiveCodeBench v5、ReasoningGym Algorithm
- Science：GPQA、SuperGPQA、SciBench

模型设置包括：

- Qwen3-1.7B、Qwen3-4B、Qwen3-8B 作为 student
- Qwen3-4B / Qwen3-30B-A3B 的 Instruct 或 Thinking checkpoint 作为 teacher
- 对应 Qwen3 Base checkpoint 作为 teacher-base
- Gemma4-E4B-it 作为跨模型族 student，Gemma4-31B-it 作为 teacher，Gemma4-31B 作为 teacher-base

训练设置比较短：100 optimization steps，maximum generation length 8192，sampling temperature 0.7，learning rate $5\times10^{-6}$，AdamW，gradient clipping 1.0，不额外使用 KL regularization。

## 实验结果

### Qwen3 Non-thinking Mode

Non-thinking mode 中，Qwen3 原始模型的显式推理能力较弱，因此所有 OPD 类方法都有明显提升，但 OPD2 最稳定。

Math 平均分：

| Model      | Original |  OPD | ExOPD | OPD2 |
| ---------- | -------: | ---: | ----: | ---: |
| Qwen3-1.7B |     34.8 | 51.0 |  51.4 | 54.6 |
| Qwen3-4B   |     45.8 | 64.0 |  66.4 | 70.3 |
| Qwen3-8B   |     46.9 | 65.9 |  67.8 | 71.6 |

Code 平均分：

| Model      | Original |  OPD | ExOPD | OPD2 |
| ---------- | -------: | ---: | ----: | ---: |
| Qwen3-1.7B |     10.5 | 21.0 |  24.6 | 29.4 |
| Qwen3-4B   |     22.1 | 31.4 |  37.2 | 40.1 |
| Qwen3-8B   |     25.1 | 35.0 |  38.1 | 39.9 |

Science 平均分：

| Model      | Original |  OPD | ExOPD | OPD2 |
| ---------- | -------: | ---: | ----: | ---: |
| Qwen3-1.7B |     30.8 | 36.5 |  36.5 | 38.8 |
| Qwen3-4B   |     40.0 | 47.2 |  47.8 | 50.5 |
| Qwen3-8B   |     43.5 | 49.7 |  50.2 | 51.6 |

最显著的是 Qwen3-4B non-thinking math：OPD2 的平均分达到 70.3，不仅高于同规模 OPD / ExOPD，也超过 Qwen3-8B 经过 OPD 或 ExOPD 后的结果。这说明 delta signal 在小模型推理能力快速增强上有明显收益。

### Qwen3 Thinking Mode

Thinking mode 更难，因为原始 Qwen3 已经具备较强 reasoning 能力，常规 OPD 反而可能破坏已有能力。

Math 平均分：

| Model      | Original |  OPD | ExOPD | OPD2 |
| ---------- | -------: | ---: | ----: | ---: |
| Qwen3-1.7B |     59.2 | 57.1 |  58.4 | 62.7 |
| Qwen3-4B   |     73.3 | 70.9 |  72.3 | 74.8 |
| Qwen3-8B   |     73.7 | 72.2 |  73.6 | 75.9 |

Code 平均分：

| Model      | Original |  OPD | ExOPD | OPD2 |
| ---------- | -------: | ---: | ----: | ---: |
| Qwen3-1.7B |     29.3 | 32.4 |  37.1 | 40.4 |
| Qwen3-4B   |     48.7 | 46.4 |  49.2 | 50.8 |
| Qwen3-8B   |     50.8 | 50.7 |  54.0 | 57.8 |

Science 平均分：

| Model      | Original |  OPD | ExOPD | OPD2 |
| ---------- | -------: | ---: | ----: | ---: |
| Qwen3-1.7B |     40.1 | 41.7 |  42.2 | 43.4 |
| Qwen3-4B   |     51.3 | 48.4 |  49.0 | 51.5 |
| Qwen3-8B   |     53.0 | 50.9 |  52.0 | 54.6 |

这组结果比 non-thinking mode 更能体现 OPD2 的意义。传统 OPD 在多个 thinking setting 下会降低原始模型表现，说明直接模仿 teacher distribution 未必适合强 reasoning student；OPD2 则在大多数场景下提升或至少更好地保留原有能力。

### Gemma4 跨模型族实验

Gemma4-E4B-it 上，OPD2 也优于 OPD 和 ExOPD，但结果更有边界感。

Math 平均分从原始 60.6 提升到 67.8，高于 ExOPD 的 65.3。Science 平均分从 47.0 提升到 48.8，也优于其他方法。Code 上原始模型平均分 55.2，所有蒸馏方法都没超过原始模型，但 OPD2 的 49.5 明显好于 OPD 的 36.9 和 ExOPD 的 45.1。

这说明 OPD2 不是只对 Qwen3 有效，但当 student 本身某个 domain 已经很强时，短程 on-policy distillation 仍可能造成能力退化。OPD2 的优势在于退化更小、跨域更稳，而不是保证所有 domain 都单调提升。

## Ablation 与训练成本

Ablation 的关键结论是：delta signal 是主要收益来源。以 Qwen3-1.7B 为例，去掉 delta、改回 OPD signal 后，non-thinking math 从 54.6 降到 50.5，thinking code 从 40.4 降到 32.1。相比之下，去掉 joint condition 或 centering 的影响更小，也更不稳定。

训练动态上，OPD、ExOPD 和 OPD2 都在早期快速提升，但 OPD / ExOPD 常较早达到峰值后 plateau 或下降。OPD2 的曲线更高，也更能在 100 steps 末尾保持优势。论文特别说明主表报告的是 final step，不是挑选 peak checkpoint。

成本方面，OPD2 需要额外跑 teacher-base forward pass，因此比 OPD 更慢：

| Method | Qwen3-1.7B | Qwen3-4B | Qwen3-8B | Gemma4-E4B |
| ------ | ---------: | -------: | -------: | ---------: |
| OPD    |       4.4h |     7.3h |     7.6h |      12.7h |
| ExOPD  |       5.3h |     9.1h |     9.4h |      13.8h |
| OPD2   |       5.5h |     9.3h |     9.6h |      13.8h |

相对 OPD，OPD2 的 wall-clock overhead 在 Qwen3 上约 24-28%，Gemma4 上约 8%。但由于 OPD 类训练只需要较短 post-training period，这个成本增加仍在可接受范围内。

## 关键结论

这篇论文最值得沉淀的不是具体 benchmark 数字，而是 on-policy distillation reward 设计的思路。

第一，teacher 不应总被当作一个整体分布来模仿。对于 reasoning distillation，teacher 相对自身 base model 的变化可能更接近“能力注入后新增的行为分布”。

第二，on-policy KD 的 token-level reward 很关键。它不是简单把 teacher answer 拿来 SFT，而是在 student 自己生成的 token 上判断哪些 token 应该被增强或压低。OPD2 的 delta signal 表明，reward 的定义会显著改变 student 学到的 reasoning 行为。

第三，delta signal 不能裸用。因为它不包含 student probability，直接最大化可能导致过强偏置；centering 和 OPD direction consistency 是让 delta signal 进入 on-policy training 的稳定化设计。

第四，OPD2 更像短程、高效、无需 reward design 的 reasoning post-training 方法。它不替代 RLVR / GRPO，但可以成为 reward 设计困难、teacher 可用、student 已有一定基础能力时的有效路线。

## 局限与疑问

论文的实验集中在 math、science、code 三类 reasoning benchmark，仍属于相对容易批量评测的任务。它没有覆盖多轮 tool-use agent、web agent、真实软件工程 environment rollout 或长 horizon 交互任务。因此，OPD2 是否适合 agent trajectory post-training，还需要额外验证。

OPD2 依赖 teacher-base checkpoint。如果只有 closed teacher API，或者 teacher 的 base model 不可获得，就无法直接构造 delta signal。对于内部模型体系，这个条件通常更容易满足；对外部模型蒸馏则是实际限制。

此外，delta signal 被解释为 reasoning tuning 的学习轨迹，但它仍只是 logprob 差，不保证所有变化都来自“有益 reasoning 能力”。Teacher 的 post-training 也可能改变格式、语气、安全偏好、拒答倾向和长度分布。若这些变化和目标任务不一致，delta signal 也可能放大不想要的行为。

## 对训练实践的启发

如果有完整 teacher / teacher-base / student 体系，OPD2 提供了一种很清晰的实验路线：

1. 先用短程 OPD2 验证 student 在目标 domain 上是否能快速吸收 teacher 的 reasoning 增量；
2. 把 OPD、ExOPD、OPD2 和原始模型放在相同 on-policy rollout 设置下对比，避免把收益混入采样或评测差异；
3. 对强 student 单独关注 regression，而不只看平均提升；
4. 记录 token-level reward 分布、正负信号比例、gradient clipping 频率和训练曲线，因为 OPD2 的收益来自 reward 形态，而不只是最终分数；
5. 在 agent / tool-use 任务上使用时，需要重新检查 delta signal 是否真的对应 tool choice、argument grounding、verification、recovery 等行为，而不是只学到 reasoning 文本风格。

## 相关知识链接

- [[training/post-training/on-policy-kd|On-policy KD]]
- [[training/post-training/knowledge-distillation|Knowledge Distillation]]
- [[training/post-training/offline-kd|Offline KD]]
- [[training/post-training/grpo|GRPO]]
