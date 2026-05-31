---
title: Reinforcement Pretraining
created: 2026-05-28
published: 2026-05-28
modified: 2026-05-31
type: topic
status: mature
area: training
tags:
  - pretraining
  - reinforcement-learning
  - reasoning
---

Reinforcement Pretraining 指把强化学习式目标前移到预训练或 continued pretraining 阶段，而不是只在 SFT 之后作为 RLHF / RLVR 使用。它试图在 base model 形成阶段就塑造探索、推理或更有效的中间表示。

## 目标与问题

标准 [[training/pretraining/objective|Training Objective]] 通常是 next-token prediction：给定 $x_{<t}$，最大化真实下一个 token $x_t$ 的 likelihood。这个目标可扩展、稳定，但它没有显式奖励模型在预测前进行探索或推理。

Reinforcement Pretraining 想解决的问题是：能否在普通文本上构造可扩展的 reward，让模型在预训练阶段就学习“产生有用中间思考”或“选择更有信息量的内部动作”？

因此它不是简单地把后训练 RL 提前运行一遍，而是重新定义预训练阶段的 credit assignment：哪些中间 token、latent action、tool action 或 reasoning trace 应该因为提升未来预测、验证正确性或任务成功而被强化。

## 设计空间

Reinforcement Pretraining 可以从三个维度理解。

### Action 是什么

action 不一定是最终答案，也不一定必须暴露给用户。它可以是：

- 显式 thought tokens，例如 Chain-of-Thought；
- latent scratchpad 或内部 reasoning trace；
- tool / environment action，例如检索、执行代码、调用 verifier；
- 数据选择或 curriculum 决策；
- 对下一段文本生成策略的控制变量。

如果 action 是可见文本，训练会影响模型后续是否倾向于输出推理过程；如果 action 是内部变量或工具调用，训练目标更接近策略学习或表示学习。

### Reward 来自哪里

reward 可以来自：

- information gain：某个 action 是否提高真实 token 的 likelihood；
- verifier：答案是否可验证正确，例如数学、代码、单元测试；
- learned reward model：人类或 AI preference 训练出的打分器；
- environment feedback：任务环境给出的成功信号；
- mixture reward：把 likelihood、KL、格式约束和任务 reward 合成。

预训练规模要求 reward 足够便宜、密集且通用。过于稀疏或昂贵的 verifier 更适合 mid-training / post-training 的特定能力强化，而不是完整预训练主目标。

### On-policy 还是 Offline

如果 thought 或 action 来自当前模型实时 rollout，训练更接近 on-policy / near on-policy RL，能学习当前策略分布下的行为改进，但计算成本高、稳定性更难控制。如果 action 来自离线数据，例如人工 CoT、强模型生成 trace 或已有 tool trajectory，则工程更简单，但模型学到的是数据分布中的行为，探索能力较弱。

## RLP 的代表性机制

[[sources/papers/2025-rlp-reinforcement-as-a-pretraining-objective|RLP: Reinforcement as a Pretraining Objective]] 是一个典型例子。它把 Chain-of-Thought 视为预测下一 token 之前的 action：

1. 给定上下文 $x_{<t}$，模型采样 thought $c_t$。
2. 同一个模型在 $x_{<t}, c_t$ 条件下预测真实 token $x_t$。
3. EMA teacher 在不使用 thought 的条件下预测 $x_t$。
4. reward 是两者 log-likelihood 的差：

$$
r(c_t)=\log p_\theta(x_t \mid x_{<t}, c_t)-\log \bar{p}_\phi(x_t \mid x_{<t})
$$

如果 thought 提高了真实 token 的概率，它获得正奖励；如果没有帮助，则 reward 较低甚至为负。

RLP 的 thought 是训练时由当前模型自己 rollout 出来的，而不是离线固定的 CoT 数据。每个 training step 会用当前策略快照采样多条 thoughts，计算组内 reward 和 advantage，然后更新 thought tokens 的生成概率；下一步再用更新后的模型继续采样新的 thoughts。因此它更接近 on-policy / near on-policy reinforcement pretraining。

这个机制的关键不是“给语料补一段 CoT”，而是让模型自己提出中间思考，并用该思考对真实 next token 的边际帮助来分配 reward。这样，普通文本流也可以产生密集训练信号：每个预测位置都能比较有 thought 与无 thought 的 likelihood 差异。

## 与 Continued Pretraining 的区别

Continued pretraining 仍然通常使用最大似然目标，只是继续在新数据、领域数据或高质量数据上训练。Reinforcement Pretraining 的关键变化不是“多训练一段”，而是改变训练信号：

| 维度 | Continued Pretraining | Reinforcement Pretraining |
|---|---|---|
| 主要目标 | 最大化真实 token likelihood | 最大化 action / thought 带来的 reward |
| 数据需求 | 普通文本或领域文本 | 取决于 reward，可使用普通文本或 verifier 数据 |
| 信号密度 | 每个 token 的监督 loss | 可以是 dense reward，也可以是 sparse reward |
| 行为激励 | 隐式学习语言模式 | 显式激励探索、推理或策略选择 |

RLP 的特殊之处在于它把 reward 设计成 verifier-free dense information gain，因此可以作用于普通文本流，而不是只依赖数学题、代码题或有标准答案的数据。

两者也可以组合：continued pretraining 提供领域语料和语言建模主损失，reinforcement-style loss 只在特定位置、特定样本或特定 thought budget 下启用。这样可以降低纯 RL 目标导致的分布漂移风险。

## 与后训练 RL 的区别

后训练 RL 通常发生在 instruction-following 或 assistant policy 已经形成之后，目标是优化偏好、可验证正确性或安全行为。Reinforcement Pretraining 则发生得更早，目标是塑造 base model 的基础能力。

这带来两个后果：

- 如果有效，pretraining 阶段形成的能力可能与后续 SFT / RLVR 叠加，而不是被后训练覆盖。
- reward 必须更通用、更密集、更便宜，否则无法承受预训练规模。

后训练 RL 常常有明确 prompt、response、reward 和 KL-constrained policy update；Reinforcement Pretraining 的样本则可能是普通文档流，action 也可能嵌在 token-level 预测过程中。因此它对数据管线、rollout 缓存、baseline 更新和训练稳定性的要求更接近预训练系统工程，而不仅是 alignment pipeline。

## 设计取舍

Reinforcement Pretraining 的优势是把 reasoning signal 提前注入基础模型，可能减少后训练阶段才“补课”的压力。它也能把普通文本转成更丰富的训练信号，而不只是监督下一个 token。

主要代价在于工程复杂度和稳定性：

- rollout 会显著增加计算；
- reward 如果来自模型自身，可能有 reward hacking 或自指偏差；
- thought 是否忠实反映真实推理过程仍然不确定；
- 与 NTP loss 如何混合、何时切换、训练多长，都还需要经验规则。

## 失败模式与边界

- **Reward mis-specification**：信息增益、verifier 或 learned reward 只覆盖目标的一部分，模型可能优化 proxy 而不是任务本身。
- **Compute amplification**：同一位置采样多条 thoughts 或 rollouts，会把预训练成本按采样数放大。
- **Distribution drift**：过强 RL 更新可能让 token 分布偏离自然文本，损害语言建模能力。
- **Thought overfitting**：模型可能学会生成看似有帮助但不可解释、不可忠实或不可迁移的中间文本。
- **Baseline leakage / collapse**：如果 baseline 设计不稳，reward 可能变得噪声过大或过早收缩。
- **评估困难**：预训练阶段的 reward 改善未必直接对应后训练后的用户可见能力，需要跨阶段评估。

因此，Reinforcement Pretraining 目前更适合作为研究方向和特定能力增强策略，而不是默认替代 NTP 的工业标准。使用时应保留 likelihood loss、KL 或其他约束，避免模型过度追逐局部 reward。

## 可沉淀的判断

判断一个 reinforcement-style pretraining objective 是否值得采用，可以问四个问题：

1. reward 是否能在足够大规模上低成本计算；
2. reward 是否与长期能力改善有可验证关联；
3. 额外 rollout 成本是否比等量 NTP token 更划算；
4. 该目标形成的能力能否在 SFT / RLHF / RLVR 后保留下来。

如果这些问题无法回答，优先把它视为实验性 mid-training 或能力注入方法，而不是基础预训练配方。

## 相关概念

- [[training/pretraining/objective|Training Objective]]
- [[training/pretraining/pretraining|Pretraining]]
- [[training/mid-training/continued-pretraining|Continued Pretraining]]
- [[training/post-training/grpo|GRPO]]
- [[training/post-training/rlhf|RLHF]]
