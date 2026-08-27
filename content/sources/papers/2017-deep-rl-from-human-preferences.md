---
title: "Deep Reinforcement Learning from Human Preferences"
created: 2026-05-29
published: 2026-08-27
modified: 2026-08-27
type: source
status: processed
source_type: paper
area: sources
tags:
  - source
  - paper
  - rlhf
  - reward-model
  - preference-learning
  - reinforcement-learning
source_url: https://arxiv.org/abs/1706.03741
paper_date: "2017-06"
paper_order: "03741"
---

# Deep Reinforcement Learning from Human Preferences

## 基本信息

| 字段 | 内容 |
|---|---|
| 论文 | [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) |
| 版本 | arXiv:1706.03741v4，2023-02-17 |
| 作者 | Paul F. Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, Dario Amodei |
| 机构 | OpenAI, DeepMind |
| 首次公开 | 2017-06，arXiv v1 |
| 相关 topic | [[training/post-training/rlhf|RLHF]]，[[training/post-training/reward-model|Reward Model]]，[[training/post-training/ppo|PPO]]，[[fundamentals/information-theory/kl-divergence|KL Divergence]] |

这篇论文讨论一个直到今天仍然重要的问题：当任务目标复杂、主观，或者很难写成程序化 reward 时，能不能让人类只比较 agent 的行为片段，由此学习一个 reward function，再用强化学习优化这个 reward？

论文提出的不是一套只适用于某个环境的标注技巧，而是一条完整 pipeline：agent 产生轨迹片段，人类比较两个片段，reward predictor 从比较结果中学习，policy 再最大化 predictor 给出的 reward。论文在 MuJoCo 机器人和 Atari 游戏中验证了这条路线，并展示了通过少量人类反馈学习新行为的结果。

如果只保留一个核心认识：**人类不必为每一步行为提供绝对分数，也不必亲自示范复杂动作；只要能够比较两个短 trajectory segment，系统就可以把比较偏好转化为可供 RL 优化的 learned reward。** 后来的 LLM RLHF 将“trajectory segment”换成 prompt-response，将视频比较换成文本回答比较，但基本分工仍然来自这里。

## 研究问题

### 程序化 Reward 难以覆盖复杂目标

传统 deep RL 依赖环境提供 reward function。对于游戏得分、机器人位置或速度等目标，reward 可以由程序计算；但很多真实任务的目标并不容易写成传感器状态到数值的函数，例如：

- 清理一张桌子但不损坏物品；
- 把多个物体混合到合适程度；
- 让机器人动作平稳、自然且符合人类意图；
- 让 agent 的行为“有帮助”，而不是只满足某个表面指标。

程序化 reward 往往只能近似表达目标，policy optimization 还可能主动寻找 reward 漏洞。一个看似合理的 reward function，可能鼓励模型优化错误的 proxy，而不是人类真正想要的行为。

### Demonstration、Imitation 与 Preference 的不同

如果人类可以完整示范目标行为，可以使用 imitation learning 或 inverse reinforcement learning。但很多复杂行为有两个困难：

1. 人类未必能亲自完成或演示该行为，例如控制具有很多自由度、形态与人类不同的机器人；
2. 人类可能能够判断哪个行为更好，却无法准确写出完整操作步骤。

论文因此采用更弱但更容易获得的监督形式：人类不需要给出 action，也不需要提供绝对 reward，只需在两个短片段之间选择更喜欢的一个，或者标记两者相同/无法比较。

### 人类反馈成本必须远低于环境交互成本

一个 deep RL agent 可能需要数百万到数十亿次环境交互。如果人类对每一个 timestep 都打分，反馈成本会超过传统 RL。论文的目标是让人类只标注很少的 trajectory comparisons，再训练 reward predictor 为其余大量 agent-environment interactions 提供 dense reward。

论文希望同时满足四点：

1. 解决只能识别目标行为、却不能示范目标行为的任务；
2. 允许非专家用户提供反馈；
3. 能够扩展到大型 deep RL 问题；
4. 将人类反馈量降低到可接受的范围。

## 核心主张

论文的贡献可以概括为以下几层：

1. **从 trajectory segment pair 学习 reward function。** 将人类二选一偏好转化为 Bradley-Terry 风格的概率模型和交叉熵目标。
2. **把 reward learning 与 policy optimization 异步交织。** policy 不断探索新状态，新的片段被提交给人类比较，reward predictor 持续更新，再反馈给 policy。
3. **使用 uncertainty-aware query selection。** 通过 reward predictor ensemble 的分歧优先选择不确定片段，减少低价值标注。
4. **使用短视频片段而不是单个状态或完整 episode。** 片段提供了行为趋势和时间上下文，通常比单帧更容易比较，也比完整轨迹更节省标注时间。
5. **在不暴露真实 reward 的情况下学习已知任务。** MuJoCo 上约 700 个比较可以接近使用真实 reward 的 RL；Atari 上 5,500 次比较也能在多数游戏中产生明显学习。
6. **能够学习原本没有现成 reward 的新行为。** 例如机器人连续后空翻、单腿奔跑，以及在 Enduro 中与其他车辆保持并行。

论文的主张重点是“人类反馈可以被经济地放大成大规模 RL 训练信号”，而不是“learned reward 一定等同于真实 reward”。实验也明确展示了 predictor 非平稳、反馈噪声、短片段歧义和 reward hacking 等边界。

## 问题形式化

### Environment 与 Preference

agent 在每个 timestep $t$ 接收 observation $o_t$，选择 action $a_t$，并继续与环境交互。与传统 RL 不同，环境不直接提供可见的 reward $r_t$，而是存在一个能够比较行为的 human overseer。

一个长度为 $k$ 的 trajectory segment 定义为：

$$
\sigma=((o_0,a_0),(o_1,a_1),\ldots,(o_{k-1},a_{k-1})).
$$

人类可以表达：

- $\sigma_1\succ\sigma_2$：更喜欢片段 1；
- $\sigma_2\succ\sigma_1$：更喜欢片段 2；
- tie：两者同样好；
- incomparable：无法比较，不写入训练数据库。

偏好监督比 absolute score 更适合这类任务，因为人类通常更容易回答“哪一个更好”，而不是为每一个片段稳定地指定一个跨时间、跨状态可比较的数值分数。

### 目标 reward 的两种评价方式

论文区分两种评测场景。

**Quantitative evaluation**：如果实验环境本身有隐藏的真实 reward function，可以用 agent 获得的真实总 reward 评价 learned reward 训练出的 policy。训练时不向 agent 暴露这个真实 reward，它只用于研究者比较。

**Qualitative evaluation**：对于新行为，没有一个可以直接计算的 ground-truth reward。此时给人类一个自然语言目标，由人类观看 agent 行为并判断是否满足目标。

这一区分很重要：真实 reward 适合衡量“是否接近已知任务最优解”，但不能完整衡量人类是否真的喜欢一个新行为；人类反馈本身既是训练信号，也是 qualitative evaluation 的来源。

## 方法与机制

### 三个异步过程

论文维护两个主要神经网络：policy $\pi$ 和 reward function estimate $\hat r$。系统由三个异步过程组成：

```text
1. Policy interaction
   policy 与 environment 交互，产生 trajectories
   用 predicted reward 优化 policy

2. Preference elicitation
   从 trajectories 选择两个 segments
   将视频片段交给人类比较

3. Reward predictor fitting
   用已有 preference database 训练 reward predictor
   将更新后的 predictor 返回 policy process
```

数据流是：

```text
policy -> trajectory segments -> human comparisons -> reward predictor
   ^                                                    |
   |----------------------------------------------------|
```

这不是“先收集完所有偏好，再离线训练一个固定 reward model”。policy 在 reward predictor 的指导下持续改变状态分布，新的数据要继续覆盖 policy 当前访问的区域。因此，在线反馈是方法有效性的组成部分，而不是实现细节。

### Policy Optimization

训练 reward predictor 后，当前问题暂时变成普通 RL：将预测 reward

$$
\hat r_t=\hat r(o_t,a_t)
$$

提供给 policy optimizer，让 policy 最大化预测 reward 的累计和。

论文在不同环境使用不同 RL 算法：

- Atari 使用同步版本的 A3C，即 A2C；
- simulated robotics 使用 TRPO。

这说明论文的核心贡献在 reward learning pipeline，而不是某个固定 policy optimizer。任何适合非平稳 reward 的 RL 方法原则上都可以作为 policy optimization backend。作者偏好 policy gradient / trust-region 方法，是因为 learned reward 会随着新反馈到来而变化，需要一定的更新稳定性。

论文还对 predicted rewards 做 normalization，使 reward prediction 的均值为 0、标准差保持稳定。由于从 preference 中只能确定 reward 的相对关系，reward 的绝对平移和尺度并没有被唯一确定，normalization 是必要的训练处理。

### Preference Elicitation

人类看到两个 1 到 2 秒的 video clips，然后选择：

```text
left segment
right segment
tie
cannot compare
```

如果选择 tie，标签分布在两个选项上均匀分配；如果选择无法比较，则该比较不加入数据库。论文中平均一个 query 的回答时间约 3 到 5 秒，实际实验使用的人工时间约为 30 分钟到 5 小时。

片段不一定从相同初始状态开始。论文没有假设可以把环境 reset 到任意状态，因为高维机器人状态空间中很多随机状态根本不可达，目标 policy 也只占据状态空间中的低维 manifold。这使 segment comparison 的解释更困难，但更接近实际在线交互。

### Bradley-Terry Reward Model

设 reward predictor 对一个 observation-action pair 输出 $\hat r(o,a)$。对片段 $\sigma_i$，定义片段总 reward：

$$
R(\sigma_i)=\sum_{t=0}^{k-1}\hat r(o_t^i,a_t^i).
$$

模型假设人类偏好片段 $\sigma_1$ 的概率为：

$$
\hat P[\sigma_1\succ\sigma_2]
=
\frac{\exp(R(\sigma_1))}
{\exp(R(\sigma_1))+\exp(R(\sigma_2))}.
$$

这就是 Bradley-Terry pairwise preference model。对数据库：

$$
D=\{(\sigma_1,\sigma_2,\mu)\},
$$

其中 $\mu$ 是在两个选项上的标签分布，使用交叉熵训练：

$$
\mathrm{loss}(\hat r)=
-\sum_{(\sigma_1,\sigma_2,\mu)\in D}
\left[
\mu(1)\log\hat P[\sigma_1\succ\sigma_2]
+\mu(2)\log\hat P[\sigma_2\succ\sigma_1]
\right].
$$

这个建模方式只要求预测两个片段的相对偏好，不要求 reward 的绝对数值具有跨任务意义。reward 的平移、尺度和时间位置都存在一定不可辨识性，因此后续的 normalization 和 policy optimization 配置会影响实际训练。

### 人类噪声模型

如果直接使用 softmax，模型会在 reward 差异越来越大时把偏好概率推向 0 或 1，等价于假设人类在 reward 差异足够大时几乎不会犯错。现实中的标注者有一个不会随着差异无限减小的错误概率。

论文因此在 preference model 中加入 10% 的 uniform random response probability。这相当于给标签分布保留噪声下界，避免 reward predictor 对有限的人类判断过度自信。

这项处理的意义比一个具体数字更重要：人类 preference label 不是无噪声 oracle，reward model 需要显式建模 annotator noise，否则后续 RL 会放大错误标签。

### Reward Predictor Ensemble

论文训练多个 reward predictors，而不是只维护一个 predictor。每个 predictor 在从 preference database 有放回抽样得到的数据上训练，并进行独立 normalization，最后对预测结果求平均。

默认使用 3 个 predictors，并对每个 predictor 留出约 $1/e$ 的数据作为 validation set；使用 $l_2$ regularization，必要时结合 dropout，使 validation loss 保持在 training loss 的合理范围内。

ensemble 有两个作用：

1. 平均多个 predictor，降低单一 reward model 的偶然误差；
2. 利用 predictor 之间的 disagreement 近似 epistemic uncertainty，为 query selection 提供信号。

它仍然不能保证 reward model 在分布外行为可靠。ensemble disagreement 只是一个粗粒度 uncertainty proxy，不是严格的 expected value of information。

### Active Query Selection

系统先从 policy 产生的大量候选中采样 segment pairs，再让 ensemble 中每个 predictor 预测哪一个片段更好。优先选择 predictors 对偏好判断分歧最大的 pair：

```text
many candidate segment pairs
  -> each reward predictor gives a preference
  -> calculate disagreement / variance
  -> ask human about the most uncertain pairs
```

这样做的目标是把有限 query 用在最能减少 reward model 不确定性的地方，而不是重复标注模型已经很确定的样本。

论文也发现这个启发式并不总是有效：某些任务中随机 query 反而更好。因此，uncertainty sampling 应理解为一种便宜的主动学习近似，而不是普遍成立的最优 query policy。理想的标准应是 query 的 expected value of information，但论文将其留作未来工作。

### Online Feedback 与分布漂移

policy 优化 predicted reward 后，会访问新的状态区域；reward predictor 只在旧数据上训练时，可能无法正确覆盖这些区域。论文因此让 preference collection 和 RL learning 交织进行：

```text
policy changes
  -> state occupancy changes
  -> collect new behavior clips
  -> obtain new labels
  -> update reward predictor
  -> continue policy optimization
```

这也是论文中 online query ablation 很关键的原因。反馈不是静态数据集，而是跟着 policy 的 occupancy distribution 一起移动。

## 实验设计

### 总体设置

论文在 TensorFlow 中实现系统，通过 OpenAI Gym 接入 MuJoCo 和 Arcade Learning Environment。实验分为三类：

1. 有隐藏真实 reward 的 benchmark task：训练时不提供真实 reward，用它做离线 quantitative evaluation；
2. 与 synthetic oracle 比较：oracle 直接根据隐藏真实 reward 回答 segment preference，用于区分人类噪声与 reward learning 本身的效果；
3. 没有现成程序化 reward 的 novel behavior：通过人类偏好训练，再用人类进行 qualitative evaluation。

为了避免环境自身泄露目标，论文还移除了一些隐含监督：

- robotics 中取消会暴露任务目标的 variable-length episode termination，并改成需要从反馈中学到的 penalty；
- Atari 中不向 agent 提供 life loss 和 episode-end signal，并将环境视为连续 episode；
- Atari 画面中的 score、敌人数量、speedometer 等可能直接泄露 reward 的区域被遮挡。

这一步很重要。如果 agent 通过 episode termination、屏幕分数或环境接口已经知道目标，实验就不能说明它真正从 human preferences 学到了 reward。

## 实验与证据

### MuJoCo Simulated Robotics

论文测试了 8 个 MuJoCo 机器人任务。人类反馈实验通常使用约 700 个 preference queries，并与 350、700、1400 个 synthetic oracle queries 以及直接使用真实 reward 的 RL 进行比较。

主要结果是：

- 约 700 个人类 labels 在大多数任务上已经接近使用真实 reward 的 RL；
- learned reward 训练的曲线方差更大、稳定性更弱，但平均性能相近；
- 1400 个 labels 的设置在一些任务上略好于真实 reward baseline，作者推测 learned reward 可能形成了更有利于学习的 reward shaping；
- 真实人类反馈通常略差于同数量的 synthetic feedback，但在人类偏好提供了额外 shaping 的 Ant 任务上反而明显更好。

这里的结果说明 human feedback 不只是“低质量的真实 reward 替代品”。人类可以偏好“机器人保持直立”这类有助于探索的中间行为，从而提供程序化 reward 没有设计好的 shaping signal。但人类反馈的成本和噪声也直接反映在训练稳定性中。

### Atari

论文在 7 个 Atari 游戏上使用约 5,500 个 human queries，并与不同数量的 synthetic oracle queries 以及真实 reward RL 比较。

结果明显比 MuJoCo 更困难：

- BeamRider 和 Pong 中，较少的 synthetic labels 就能接近真实 reward RL；
- Seaquest 和 Qbert 最终可以接近真实 reward，但学习更慢；
- SpaceInvaders 和 Breakout 没有达到真实 reward RL，但仍然产生了明显学习，例如 SpaceInvaders 经常能通过第一关，Breakout 能达到一定得分；
- 真实人类反馈多数略差于同数量 synthetic feedback，但常接近少 40% synthetic labels 的效果；
- Qbert 的短片段较难判断，真实人类反馈未能学会通过第一关；
- Enduro 中人类对“接近并超过其他车辆”的渐进式行为给出 shaping，使 human feedback 结果优于简单随机探索的 A3C。

这组实验暴露了任务难度、观测上下文和 feedback granularity 的影响：如果一个短片段几乎没有可区分的 progress，human preference 的信息量会很低；如果人类能够看到有意义的中间进展，比较反馈就能提供比 sparse programmatic reward 更密集的指导。

### Novel Behaviors

论文进一步训练了没有现成 reward function 的行为：

1. Hopper 连续完成后空翻并直立落地，每次约 900 queries，不到一小时；
2. Half-Cheetah 单腿向前移动，约 800 queries，不到一小时；
3. Enduro 中与其他车辆保持并行，约 1,300 queries 和 4M frames 的环境交互。

这些结果的意义不在于任务本身，而在于监督形式：人类只需要看行为片段并根据目标描述比较，不需要把“连续后空翻且落地直立”拆解成每一步的程序化 reward。

但这些新行为由论文作者提供反馈，不能与陌生 contractor 的 benchmark 结果完全等同。作者对行为目标和环境有更深理解，反馈一致性可能更高。

### Ablation：哪些组件真正重要

论文对多个组件做了 ablation：

- random queries：不再优先选择 ensemble disagreement 最大的片段；
- no ensemble：只训练一个 predictor；
- no online queries：只使用训练早期收集的 comparisons；
- no regularization：移除 $l_2$ regularization，只保留 dropout；
- no segments：在 robotics 中用单个 frame 而不是 trajectory segment；
- target：让 oracle 提供 segment 总 reward，用 regression 预测绝对目标，而不是做 pairwise comparison。

最值得关注的是以下结论。

**Offline reward predictor training 很差。** 当 policy 的 occupancy distribution 发生变化时，早期数据只覆盖真实 reward 的一部分区域。policy 最大化这个 partial reward 后，可能产生真实目标不喜欢的 bizarre behavior。例如 Pong 中，policy 可能学会避免丢分，却不主动得分，导致球一直重复相似回合。

**Comparison 通常比 absolute score 更适合人类。** 连续控制 reward 的尺度跨任务差异很大，直接回归绝对 reward 会让学习更困难；pairwise comparison 只需要判断相对好坏，训练更平滑。Atari 中将 reward clip 到有限范围并预测 sign 后，两种标签方式的差距没有始终保持同一方向。

**Trajectory clips 比 single frames 更有信息。** 单帧常常需要人类先理解当前状态，无法判断趋势；更长片段提供了行为进展和上下文。论文发现，较长 clips 对每个 clip 的帮助更大，但对每一帧的帮助会下降，因此需要选择一个既能表达行为、又不会显著增加判断时间的片段长度。

**Ensemble disagreement 不是总能提高结果。** 它是便宜的 query prioritization heuristic，在某些任务中有帮助，但也可能把预算集中在高不确定、却不一定最有训练价值的片段上。

## 关键结论

### Reward Model 是把主观偏好放大成 RL 信号的中间层

这篇论文最重要的系统分工是：

```text
human preference
  -> learned reward model
  -> dense predicted reward over massive interaction data
  -> policy optimization
```

人类不需要覆盖 agent 的每次交互，reward predictor 负责把少量 comparisons 泛化到大量未标注 trajectory。policy optimization 再将这个可优化的 scalar signal 转化为行为变化。

后来的 LLM RLHF 基本继承这条结构，只是把：

- observation-action segment 换成 prompt-response；
- 视频片段比较换成文本回答排序；
- reward predictor 换成 response-level reward model；
- MuJoCo/TRPO 或 Atari/A2C 换成 PPO、DPO、GRPO 或其他 policy optimization。

### Online data collection 是必要的分布修正

只在初始 policy 的数据上训练 reward model，再固定 reward model 做长期 RL，容易出现 distribution shift 和 reward hacking。policy 改变后，会遇到训练早期没有覆盖的新状态；reward model 在这些状态上的预测不可靠，policy 又会主动寻找高预测 reward 的漏洞。

因此，online preference collection 的价值不是简单“多收一些数据”，而是让 reward model 跟踪 policy 当前真正访问的行为分布。这一结论对今天的 AI feedback、online judge 和 agent trajectory evaluation 仍然成立。

### Human preference 不等于真实 reward 的无噪声观测

论文中的 human label 具有三个特点：

- 人类可能犯错；
- 不同人可能对同一片段判断不一致；
- 人类偏好可能包含程序化 reward 没有表达的 shaping 信息。

因此，reward model 不应被理解成真实价值函数的直接复制。它是一个有限数据、有限模型和特定标注规范共同定义的 proxy objective。RL 越强，越需要关注 proxy 是否能承受被优化，而不只是 held-out preference accuracy。

## 对现代 LLM RLHF 与 Agent Training 的启发

### 从 segment preference 到 response preference

在 LLM 中，一条 response 可以看作由 token action 组成的 trajectory segment。人类比较两个回答，reward model 学习整体 response 的偏好，再通过 PPO 等算法优化 policy。

但 LLM response 通常比论文中的 1 到 2 秒 clip 长得多，且 reward 往往只在末端评估。因此还需要解决：

- sequence-level reward 如何分配到 token-level action；
- reasoning、tool call 和 environment observation 如何区分；
- 长回答的 length bias 如何处理；
- reward model 是否只学到格式、语气和表面 helpfulness；
- online rollout 如何覆盖 policy 新产生的错误。

### 从 human label 到 AI feedback / verifier

现代系统可以将 human preference 替换或扩展为：

- AI feedback；
- rule-based verifier；
- unit test、compiler、数学答案检查；
- 多模型 judge；
- 用户线上反馈。

但替换标注者并不会消除 proxy risk。AI judge 或 verifier 同样可能存在偏差、分布外错误和可被 policy 利用的漏洞。论文的核心教训仍然适用：reward model 的训练分布、反馈时机、候选片段上下文和在线更新机制需要一起设计。

### 对长程 Agent 的边界

论文的 reward model 以短 segment 的 reward 之和解释人类偏好，适合局部行为评价。对于数百步甚至更长的 agent trajectory，这个假设可能不够：

- 人类未必能一次比较完整长轨迹；
- 局部片段可能看起来很好，但破坏全局任务目标；
- 早期计划的价值要到很久之后才显现；
- 不同片段可能处在不可比较的状态分布中。

因此，长程 agent 可能需要层级化比较、阶段性 outcome、process reward、轨迹摘要或可验证终局信号。论文提供的是 preference-to-reward 的基础框架，不能直接解决 long-horizon credit assignment。

## 局限与疑问

1. **Reward model 可能被 policy exploit。** 论文中的 Pong failure 说明，policy 可以优化 reward predictor 捕捉到的局部目标，而不满足真实目标。
2. **Online feedback 成本仍然存在。** 人类反馈量大幅减少，但复杂任务仍需要持续标注，且标注者培训、一致性和调度会影响训练。
3. **短片段存在上下文不足。** 太短看不出行为趋势，太长则增加理解成本；不同任务的最佳 segment length 不同。
4. **Preference 的可比较性并不总是成立。** 不同初始状态、不同进度阶段或不同策略之间，二选一可能缺少明确标准。
5. **Ensemble uncertainty 只是近似。** predictor disagreement 不一定等价于 query 的真实信息价值，可能把标注预算用在难判断但低收益的样本上。
6. **论文的 novel behavior 反馈来自作者。** 这证明了方法的表达能力，但不能完全代表陌生非专家用户的实际效果。
7. **基础实验与现代 LLM RLHF 有显著差异。** 论文使用低维/视觉控制环境、A2C/TRPO 和短 segment；LLM 还需要处理 token-level logprob、reference KL、长上下文、tool-use、批量 rollout 和大规模推理成本。
8. **Preference accuracy 不是最终指标。** reward predictor 在 held-out comparisons 上准确，并不意味着 policy 优化后仍会产生符合人类偏好的行为；必须评估 policy rollout 和真实任务结果。

## 我的理解

这篇论文真正奠定的是一种“把人类监督放在 reward model，而不是直接放在每个 action 上”的系统思路。人类只做自己擅长的判断：比较两个短行为片段；模型和 RL 系统负责把这个判断扩展到更大规模的交互数据中。

但这条路线的关键风险也同时被论文展示出来：reward model 不是目标本身，而是目标的可优化近似。一旦 policy 开始强力优化它，预测器的盲点就会从普通误差变成系统性行为漏洞。因此，完整 RLHF 系统必须同时维护：

```text
preference quality
  + reward model generalization
  + online data coverage
  + policy update stability
  + downstream behavior evaluation
```

对今天的 LLM 后训练，最值得保留的不是“用人类比较代替 reward”这一表面形式，而是下面这条判断：**反馈形式、reward model、policy rollout 和评估不能被拆成互不相干的模块。policy 会改变数据分布，也会主动寻找 reward 的漏洞；所以 reward learning 必须持续面对 policy 当前产生的行为。**

## 相关知识链接

- [[training/post-training/rlhf|RLHF]]
- [[training/post-training/reward-model|Reward Model]]
- [[training/post-training/ppo|PPO]]
- [[training/post-training/grpo|GRPO]]
- [[fundamentals/information-theory/kl-divergence|KL Divergence]]
- [[sources/papers/2022-instructgpt|Training language models to follow instructions with human feedback]]
- [[sources/papers/2020-learning-to-summarize-from-human-feedback|Learning to summarize from human feedback]]
