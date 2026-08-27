---
title: "Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning"
created: 2026-07-14
published: 2026-07-14
modified: 2026-07-14
type: source
status: processed
source_type: paper
area: sources
tags:
  - source
  - paper
  - agent
  - reinforcement-learning
  - post-training
  - grpo
  - asynchronous-rl
aliases:
  - SAO
  - Single-Rollout Asynchronous Optimization
source_url: https://arxiv.org/abs/2607.07508
paper_date: "2026-07"
paper_order: "07508"
---

# Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning

## 基本信息

- 来源：arXiv:2607.07508v1
- 标题：Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning
- 作者/机构：Zhenyu Hou, Yujiang Li, Jie Tang, Yuxiao Dong / Tsinghua University；部分工作在 Z.AI 实习期间完成
- 日期：2026-07-08
- 链接：https://arxiv.org/abs/2607.07508
- 相关 topic：[[training/post-training/grpo|GRPO]]，[[training/post-training/reinforcement-learning|Reinforcement Learning]]，[[application/agents/agent|Agent]]，[[application/evaluation/benchmark|Benchmark]]

这篇论文讨论的是 agentic RL 的训练系统问题：当任务变成长 horizon、多轮工具调用、代码修改或 reasoning-with-tool 时，rollout 长度高度不均匀，传统同步 RL 管线会被最慢 trajectory 卡住；异步 RL 可以提高系统利用率，但会引入 policy lag 和 off-policy drift。SAO 的目标就是在异步 rollout / training 解耦的情况下，把 agentic RL 训练做稳。

论文的核心不是“又提出一个 GRPO 变体”，而是指出 **GRPO 的 group-wise sampling 与异步 agentic training 存在结构性冲突**。GRPO 需要同一个 prompt 下的一组 responses 都完成后才能计算 group-relative advantage；但 agent trajectory 的长度分布很长尾，慢样本会拖住整组数据，先完成的样本又会在等待中变得更 off-policy。SAO 选择反过来做：每个 prompt 只采一个 rollout，轨迹完成后立刻进入训练队列，再用 value model 和 token-level clipping 稳住方差与 off-policy。

如果只保留一个核心观点：**面向长程 agent 的异步 RL，不应把 GRPO 的 group-wise baseline 当成默认假设；single-rollout + value model + token-level off-policy control 更贴近真实在线环境与长轨迹训练系统。**

## 研究问题

LLM RL 的常见流程是同步 batch-interleaved：policy 先生成一批 rollouts，等整批采样完成后，再用这批数据做若干轮优化。对于普通短回答任务，这种同步 barrier 还能接受；但 agentic reasoning 和 coding 任务中，一条 trajectory 可能只有几步，也可能有几十轮工具调用、Python 执行、OpenHands 交互或环境反馈。短轨迹很快完成，长轨迹成为 straggler，训练 GPU 和 rollout GPU 的调度效率都会变差。

异步 RL 的直觉很自然：rollout 一完成就进入训练，不再等待同批次或同组样本。但异步会带来两个问题。

第一是 policy lag。某条轨迹开始生成时使用的是 rollout policy，生成结束时 training policy 可能已经更新了很多步。长轨迹内部甚至可能跨越多个 policy 版本。此时再用当前 policy 更新这条轨迹，行为策略和学习策略之间的距离会变大，off-policy 更新更容易不稳定。

第二是 GRPO 的 group-wise sampling 不适合这种异步形态。GRPO 通常对同一个 prompt 采样多个 responses，用组内 reward 均值或归一化结果构造 advantage。这在同步训练里可以减少 variance，但在异步 agentic training 中会产生额外等待：同组中快完成的 trajectory 必须等慢 trajectory，等待时间越长，样本越 stale。对于真实在线环境或复杂交互任务，同一个 prompt 往往也只能自然获得一条反馈轨迹，不一定能稳定采到一组可比较 responses。

SAO 因此要回答的问题是：

- 异步 RL 中，如何在不维护大量历史 old policy checkpoint 的情况下控制 off-policy？
- 如果每个 prompt 只采一个 rollout，如何解决 advantage estimation 方差过大的问题？
- 多轮 agent trajectory 中夹杂 observation tokens，token-level value / GAE 应该如何处理？
- single-rollout RL 是否真的能在 agentic reasoning、coding 和模拟在线环境中优于 GRPO？

## 方法主线

SAO 由四个关键设计组成：

```text
asynchronous rollout
  -> single rollout per prompt
  -> direct double-sided token-level importance sampling
  -> stronger value model training
  -> skip-observation token-level GAE
```

这四个设计是连在一起的。single rollout 解决异步训练中的等待和 group mismatch，但带来更高方差；value model 用来降低 single-rollout advantage 的方差；DIS 负责控制异步 policy lag 带来的 off-policy；skip-observation GAE 则处理 agent trajectory 中 action 和 observation 交错的问题。

## Direct Double-Sided Importance Sampling

异步 RL 的关键不稳定来源是 rollout policy 与当前 training policy 不一致。传统 decoupled PPO 可能维护 current policy、old policy 和 rollout policy 三个角色，用 importance sampling 分别校正 stale old policy 和 rollout-training mismatch。但在异步长轨迹中，一条 trajectory 生成期间 rollout engine 可能已经更新多次，要精确追踪每个 token 对应的 old policy checkpoint 非常重。

SAO 采用更直接的做法：不用单独维护 $\pi_{\theta_{\text{old}}}$，而是直接使用 rollout 阶段保存的 token log-prob 作为行为概率：

$$
r_t(\theta)=\exp(\log\pi_\theta(a_t\mid s_t)-\log\pi_{\text{rollout}}(a_t\mid s_t))
$$

也就是说，它把当前 policy 与当时实际生成 token 的 rollout policy 直接相除。这个做法承认会有一定 off-policy bias，但避免了维护历史 policy ensemble 的工程复杂度。

更关键的是它的 double-sided token-level clipping / masking。标准 PPO clipping 往往只在 advantage 符号对应的一侧裁剪 ratio；SAO 则直接设定一个 token-level trust region：

$$
1-\epsilon_l < r_t(\theta) < 1+\epsilon_h
$$

落在区间内的 token 才参与梯度，区间外的 token 直接 mask 掉。形式上：

$$
f(x;\epsilon_l,\epsilon_h)=
\begin{cases}
x,& 1-\epsilon_l<x<1+\epsilon_h \\
0,& \text{otherwise}
\end{cases}
$$

这是一种比较 aggressive 的稳定性设计。它不是温和地截断 ratio，而是把偏离过大的 token 从更新中移除。对长轨迹 agent 来说，这个设计很有意义：轨迹后段的 tokens 更容易因为 policy lag 变 stale，如果仍然强行用这些 token 更新当前 policy，训练容易被极端 ratio 拉崩。

论文把这一机制称为 DIS。实验中，vanilla GRPO 大约在 160 training steps 附近 collapse；加入 DIS 后的 GRPO 可以明显稳定下来，说明 off-policy token gating 是这篇论文中非常关键的稳定性来源。

## Single Rollout 替代 Group-Wise Sampling

GRPO 的优势是 critic-free：它用同 prompt 下多个 responses 的 reward 分布构造相对 advantage，避免训练 value model。但这个前提依赖 group sampling。

在异步 agentic RL 中，group sampling 有两个问题：

- 系统层面：同组样本必须等待最慢的 trajectory，异步吞吐优势被削弱。
- 算法层面：等待时间越长，先完成的样本越 stale，policy lag 更严重。

SAO 取消 group-wise sampling，每个 prompt 只生成一个 rollout。轨迹一完成就可以送入 training queue，不再等待同 prompt 其他 responses。这个设计更像真实在线学习：环境通常只给当前策略一次交互反馈，模型需要从单条轨迹中学习，而不是假设每个 prompt 都能并行采一组候选。

single rollout 的代价是 variance。没有 group-relative baseline 后，算法需要一个可靠 value model 来估计 advantage。因此 SAO 实际上是从 critic-free GRPO 回到 value-based RL，只是为了适配异步和在线 agent 场景，重新设计 value model 的训练方式。

## Value Model 训练

SAO 把 value model 作为 single-rollout RL 的核心支撑。论文认为，single rollout 最大的问题不是不能训练，而是 value model 如果跟不上 policy 变化，advantage 会很噪，policy update 会被错误信号破坏。

论文做了三件事。

第一，critic 更新频率高于 actor。对每一次 policy update，value model 更新 $K$ 次，实验中 $K=2$。这相当于让 critic 更快贴近当前 policy 分布，减少 advantage estimation 的滞后。

第二，value model 使用 frozen-attention 训练。作者观察到 pilot experiments 中 value model 的梯度范数明显大于 policy model，进一步分析发现不稳定主要来自 attention layers，而 MoE layers 相对稳定。于是 RL 训练期间冻结 value model 的 attention 参数，只优化 MoE projections。论文的解释是：预训练 attention 已经具备较强 token 关联能力，value fitting 更需要调整高层表示和专家投影；冻结 attention 可以正则化 critic，避免 value model 在复杂 reasoning 轨迹上剧烈漂移。

第三，扩大 value pretraining 数据规模。论文强调 value estimation 的 cold start 是瓶颈。如果 value model 初始很差，single-rollout advantage 在训练早期就会非常噪，后续的 TTUR / faster critic update 也难以补救。

从消融看，这几项都不是装饰。SAO 在 AIME2025 / BeyondAIME 上达到 97.3 / 74.8；去掉 faster value update 后降到 95.0 / 69.8；去掉 frozen attention 后为 90.6 / 74.5；running mean baseline 只有 79.8 / 55.3。这个结果说明，single-rollout RL 的关键不是“只采一条”本身，而是要有足够稳定的 state-dependent baseline。

## Skip-Observation Token-Level GAE

多轮 agent trajectory 通常是：

```text
action_0, observation_0, action_1, observation_1, ...
```

其中 action tokens 是模型生成的，observation tokens 来自环境，例如工具返回、测试日志、网页内容、Python 输出、OpenHands scaffold feedback。标准 token-level GAE 会在相邻 token 之间计算 TD residual，但 action 结束到 observation 开始这个边界并不是模型生成过程的一部分。模型没有生成 observation，却要让 value 在 observation tokens 上传播，会引入噪声。

SAO 提出 skip-observation token-level GAE：在 action 边界处，直接从当前 action 的最后一个 token 跳到下一个 action 的第一个 token，中间 observation 不作为模型 action 序列上的 GAE 传播对象。

如果 $a_{i,N}$ 是第 $i$ 个 action 的最后一个 token，$a_{i+1,0}$ 是下一个 action 的第一个 token，则：

$$
\hat{A}(a_{i,N})=\delta+\gamma\lambda\hat{A}(a_{i+1,0})
$$

$$
\delta=r_t+\gamma V(a_{i+1,0})-V(a_{i,N})
$$

这个设计很适合 agent 轨迹：observation 仍然在上下文中影响后续 action，但它不被当作模型主动选择的 action tokens 来计算优势传播。换句话说，模型要学的是“看到 observation 后下一步怎么做”，而不是“把 observation tokens 当成自己生成的动作来优化”。

附录还比较了 step-level action granularity。作者尝试把一个 conversation turn 当作一个 step，并用 step average 或 last-token value 聚合；但在同样 400 steps 下，token-level SAO 在 AIME2025 / BeyondAIME 为 89.8 / 66.8，step-level average 为 85.8 / 60.5，step-level last-token 为 87.3 / 62.8。论文据此认为，复杂 reasoning 轨迹仍需要 token-level 细粒度训练信号，直接把一个 agent step 当成单一 action 会损失过多局部逻辑转移信息。

## 实验结果

论文主要评估两个方向：agentic reasoning with Python tool，以及 coding agent。

reasoning 评测包括 AIME2025、BeyondAIME、HMMT Nov 2025 和 IMOAnswerBench，使用 Pass@1 accuracy。评测时 top-p 为 1.0，temperature 为 1.0，最大生成长度 128K，允许最多 50 轮 reasoning 和 tool calls。

coding 评测使用 SWE-Bench Verified，scaffold 为 OpenHands，最大 300 interaction turns，context budget 为 128K。

论文报告的核心结果如下：

| 方法         | AIME2025 | BeyondAIME | HMMT Nov 2025 | IMOAnswerBench | SWE-Bench Verified |
| ------------ | -------: | ---------: | ------------: | -------------: | -----------------: |
| SFT baseline |     80.4 |       53.3 |          75.2 |           53.3 |               23.0 |
| GRPO         |     84.2 |       54.8 |          76.0 |           55.8 |               27.0 |
| SAO          |     97.3 |       74.8 |          88.3 |           74.0 |               29.8 |

在 math reasoning 上，SAO 相对 SFT 和 GRPO 的提升很明显，尤其 BeyondAIME 和 IMOAnswerBench 增幅较大。SWE-Bench Verified 上提升较小但方向一致，从 baseline 23.0 到 GRPO 27.0，再到 SAO 29.8。这个结果也说明，SAO 对 coding agent 的收益存在，但在复杂软件工程任务上，算法层面的 RL 稳定性不是唯一瓶颈，scaffold、环境、数据和 reward 质量仍然很重要。

论文还报告了两个重要训练动态：

- vanilla GRPO 在约 160 steps collapse；GRPO + DIS 稳定很多，说明 token-level off-policy masking 是必要稳定器。
- SAO 和 GRPO + DIS 在早期表现接近，约 400 steps 后拉开差距，说明 single-rollout + value model 的优势需要训练一段时间后体现。

## Online Learning Simulation

论文还设计了一个模拟在线写作任务。环境 reward 偏好在训练中逐阶段切换，要求模型在 Academic、Cute、Chuunibyou、Classical 等风格之间适应变化。reward 由 GLM-4.7 作为 judge，综合 response quality 和 style adherence 得到二值奖励。

这个实验的意义不在于写作风格本身，而在于它模拟了真实在线环境里的单轨迹反馈和非平稳 reward。GRPO 这类 group-relative 方法天然依赖同 prompt 多样本组，在这种设置中不顺手；SAO 用 value-based critic，可以从单条轨迹更新。

结果显示，SAO 在 reward preference shift 后恢复更快，running-mean baseline 有明显滞后。原因也符合直觉：running mean baseline 的窗口会保留旧分布 reward，环境切换后短期内 baseline 偏旧；value model 则能基于 state-dependent 信息更快贴合新偏好。

对真实 agent 系统来说，这节的启发是：如果后续要把模型放进 RLE 或动态任务环境中，让它根据最新环境反馈持续生成轨迹并更新策略，single-rollout value-based 方法比 group-wise relative baseline 更自然。

## 与 GRPO 的关系

SAO 不是简单地否定 GRPO。GRPO 在同步 RLVR、数学题、可并行采样的短任务中仍然有优势：实现简单，不需要训练 critic，组内相对 reward 能降低部分方差。

SAO 真正指出的是 GRPO 的边界：当训练对象变成长程 agent trajectory，且 rollout 与 training 异步解耦时，group-wise sampling 会同时制造系统等待和算法 stale。此时“没有 critic”不再一定是优点，因为为了避免 critic 而强行保持 group sampling，可能让训练数据更慢、更旧、更不贴近在线交互。

因此，更准确的分工是：

```text
短回答 / 同步采样 / 可重复 prompt:
  GRPO-style group baseline 更自然

长轨迹 / 异步 rollout / 在线环境反馈:
  SAO-style single rollout + value model 更自然
```

这也是这篇论文最值得沉淀的判断。

## 对 Agent 训练系统的启发

这篇论文对你当前关注的 agent 训练计划有几条直接启发。

第一，RLE 环境里生成的 agent trajectory 很可能天然是 single-rollout 数据。真实环境任务通常成本高、长度不均、状态会变，同一个 seed query 不一定适合同时采一组 responses 再算 group-relative reward。SAO 支持一种更贴近实际生产的管线：轨迹完成就入队，训练侧持续消费，降低等待和 stale。

第二，长轨迹训练不能只讨论 reward 和 rollout，还要讨论 token-level behavior probability 的保存。SAO 依赖 rollout log-prob 来计算 $\pi_\theta / \pi_{\text{rollout}}$。如果数据生产系统没有保存 token log-prob，后续想做异步 off-policy correction 会很困难。对 RLE pipeline 来说，trajectory schema 中应当考虑保留 action tokens、token logprobs、model version、tool observation boundary 和 reward。

第三，agent trajectory 的 observation tokens 不应被混同为模型动作。对于 SFT / mid-training，observation 可以作为上下文；对于 RL，observation 不是 policy 生成的 action。SAO 的 skip-observation GAE 给了一个很清楚的训练边界：优势传播应沿着模型 action tokens 和 action-to-action 转移走，环境返回用于影响下一步状态，但不作为 policy token 直接优化。

第四，如果后续采用 single-rollout RL，value model 质量会成为核心瓶颈。论文的结果说明，running mean 或简单历史 baseline 能跑，但效果明显弱；critic 更新频率、critic 参数冻结策略、value pretraining 数据规模都会影响稳定性。也就是说，做 agent RL 不能只准备 actor rollout 数据，还要系统性准备 value model 的初始化和训练方案。

第五，SAO 更偏 post-training / online RL，不是 mid-training 的替代品。它解决的是“已有 agent-capable policy 如何在异步环境中稳定强化”的问题；而 agentic mid-training 解决的是“base model 是否已经具备任务分解、工具使用、状态跟踪和环境反馈理解的先验”。二者在训练链路中的位置不同，但可以衔接：mid-training 注入 agentic priors，post-training / RLE 用 SAO 这类方法在真实环境反馈中优化。

## 局限与保守判断

论文的实验主要基于 Qwen3-30B-A3B backbone，覆盖 agentic reasoning、coding 和模拟在线写作。结论未必能直接迁移到小模型、短回答 RLHF、dense reward 环境或不需要长轨迹交互的任务。

SAO 还依赖较强工程条件：rollout engine 要保存 token-level log-prob；训练系统要能处理异步数据入队；value model 要有较好初始化；action / observation boundary 要在轨迹结构中清晰标注。这些条件如果缺失，算法本身的优势很难兑现。

此外，SWE-Bench Verified 上 SAO 的绝对提升只有 23.0 → 29.8，虽然优于 GRPO，但也显示复杂 coding agent 的瓶颈并不只在 RL objective。环境 scaffold、reward correctness、测试覆盖、trajectory quality 和工具交互格式仍然决定上限。

我对这篇论文的保守判断是：SAO 是一条很有价值的 agentic RL 系统路线，特别适合异步长轨迹和在线环境反馈；但它不是“有了 single rollout 就可以不用高质量数据和后训练工程”。它真正改变的是 RL rollout / training 的组织方式，以及 value model 在 agentic RL 中的地位。

## 可回填的稳定知识

这篇论文最适合沉淀到主题笔记的不是具体 benchmark 数字，而是以下几条命题：

- 长程 agentic RL 的 rollout 长度高度不均，异步 rollout / training 解耦可以提升系统效率，但必须处理 policy lag 与 off-policy drift。
- GRPO 的 group-wise sampling 与异步长轨迹 agent training 存在结构性张力：同组等待会带来 straggler latency 和 stale samples。
- single-rollout RL 更贴近在线 agent 环境，但必须依赖高质量 value model 来降低 advantage variance。
- 对异步 RL 来说，直接使用 rollout log-prob 做 token-level importance sampling，并 mask 掉 trust region 外 token，是一种实用的稳定化策略。
- agent trajectory 中 observation tokens 不是模型 action，RL advantage 传播应显式区分 action tokens 与 environment feedback。
- step-level agent action 看起来更符合 agent 抽象，但在复杂 reasoning 中可能损失 token-level 学习信号；token-level value / policy training 仍有优势。

## 相关链接

- [[training/post-training/grpo|GRPO]]
- [[training/post-training/reinforcement-learning|Reinforcement Learning]]
- [[application/agents/agent|Agent]]
- [[application/evaluation/benchmark|Benchmark]]
- [[sources/papers/2025-rlp-reinforcement-as-a-pretraining-objective|RLP: Reinforcement as a Pretraining Objective]]
- [[sources/papers/2026-davinci-dev-agent-native-mid-training-for-software-engineering|daVinci-Dev: Agent-native Mid-training for Software Engineering]]
