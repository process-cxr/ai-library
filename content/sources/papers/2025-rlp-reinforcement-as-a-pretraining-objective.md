---
title: "RLP: Reinforcement as a Pretraining Objective"
created: 2026-05-28
published: 2026-05-28
modified: 2026-05-28
type: source
status: processed
source_type: paper
area: sources
tags:
  - source
  - paper
  - pretraining
  - reinforcement-learning
  - reasoning
  - chain-of-thought
aliases:
  - RLP
  - Reinforcement Learning Pretraining
source_url: https://arxiv.org/pdf/2510.01265
publisher: NVIDIA
---

## 基本信息

- Title: [RLP: Reinforcement as a Pretraining Objective](https://arxiv.org/pdf/2510.01265)
- Authors: Ali Hatamizadeh, Syeda Nahida Akter, Shrimai Prabhumoye, Jan Kautz, Mostofa Patwary, Mohammad Shoeybi, Bryan Catanzaro, Yejin Choi
- Institutions: NVIDIA, Carnegie Mellon University, Boston University, Stanford University
- Date: 2025-09-26
- Code: `https://github.com/NVlabs/RLP`
- Related topic notes: [[training/pretraining/reinforcement-pretraining|Reinforcement Pretraining]], [[training/pretraining/objective|Training Objective]], [[training/pretraining/pretraining|Pretraining]], [[training/mid-training/continued-pretraining|Continued Pretraining]], [[training/post-training/grpo|GRPO]]

这篇论文提出 Reinforcement Learning Pre-training，简称 RLP。它的核心问题是：大模型的推理能力是否一定要等到 SFT / RLHF / RLVR 等后训练阶段才显式优化？论文的回答是否定的。RLP 试图把 RL 中“探索”的精神前移到预训练末期，让模型在预测下一 token 之前先生成一段内部 Chain-of-Thought，并根据这段思考对下一 token 预测概率的提升获得奖励。

这不是传统意义上的“用 verifier 做数学题 RL”。RLP 的奖励来自普通文本上的 teacher-forced next-token prediction：如果模型在看到 context 和 sampled thought 后，对真实下一个 token 的 log-likelihood 高于不思考的 EMA baseline，这段 thought 就得到正奖励。因此它是一种 verifier-free、dense、position-wise 的 reinforcement pretraining objective。

## 研究问题

主流训练流程通常是：

1. 用 next-token prediction 预训练 base model；
2. 用 SFT 把模型转成 assistant；
3. 用 RLHF / RLAIF / RLVR 等后训练方法强化偏好、推理或可验证任务能力。

这个流程的隐含假设是：预训练阶段只负责语言建模，显式推理行为可以后置到 alignment 或 reasoning RL 阶段。RLP 质疑这个假设：如果推理能力依赖“先想再答”的行为，那么只用最大似然预测下一个 token 可能没有充分激励模型在学习语言时主动整合上下文和世界知识。

论文要回答三个问题：

- 能否在普通预训练文本上构造不依赖外部 verifier 的 RL 奖励？
- 这种奖励是否真的比 continued pretraining 或 RPT 更有效？
- 预训练阶段获得的推理增益，经过相同 SFT + RLVR 后是否还会保留甚至叠加？

## 核心主张

RLP 的核心主张是：Chain-of-Thought 可以被视为预测下一 token 前采取的 exploratory action，而这段 action 的价值可以用 information gain 衡量。

具体地，对上下文 $x_{<t}$ 和真实下一个 token $x_t$，模型先采样 thought $c_t$，再预测 $x_t$。奖励定义为：

$$
r(c_t)=\log p_\theta(x_t \mid x_{<t}, c_t)-\log \bar{p}_\phi(x_t \mid x_{<t})
$$

其中：

- $p_\theta(x_t \mid x_{<t}, c_t)$ 是同一个模型在“想过之后”对真实下一个 token 的概率；
- $\bar{p}_\phi(x_t \mid x_{<t})$ 是 EMA teacher 在“不思考”条件下的 baseline；
- $r(c_t)$ 是 thought 带来的 log-likelihood ratio，也就是信息增益。

这个定义让奖励变得密集：每个文本位置都可以构造一次 thought、一次 no-think baseline、一次 reasoned prediction，而不需要人工答案、偏好标注或任务 verifier。

## 方法机制

### 单模型双角色

RLP 中的 thought policy 和 reasoned predictor 共享同一个网络参数 $\theta$：

- $\pi_\theta(c_t \mid x_{<t})$ 负责生成 thought；
- $p_\theta(x_t \mid x_{<t}, c_t)$ 负责在 context + thought 条件下预测真实 next token。

这意味着模型不是训练一个外部 reasoner，再训练一个 predictor，而是让同一个模型学会“生成有用的内部思考”。

### CoT 如何生成与更新

RLP 中的 CoT 不是数据集中预先写好的推理过程，也不是外部 teacher model 离线生成后固定使用的样本。它是在训练过程中由当前模型自己采样出来的。

更具体地说，每个 training step 会先固定一个行为策略快照 $\theta_{old}$，然后对 minibatch 中的上下文位置采样 $G$ 条 thoughts：

$$
c_t^{(i)} \sim \pi_{\theta_{old}}(\cdot \mid x_{<t})
$$

这些 sampled thoughts 随后被拿去计算 reasoned log-likelihood、EMA no-think baseline 和 information-gain reward。模型更新完成后，下一轮 training step 会基于新的模型参数再次采样新的 thoughts。因此 RLP 的 CoT 是 on-policy / near on-policy rollout，而不是一批静态 CoT 数据反复训练。

这个设计很关键：RLP 学到的不是“模仿某个固定 CoT 数据集”，而是让模型在自己的当前策略空间里探索哪些 thought 对预测真实 next token 有帮助。高 reward 的 thought token 概率会被提高，低 reward 的 thought token 概率会被降低。

### 单步训练流程

可以把一次 RLP 更新理解成下面这条链路：

1. 从普通文本中取一个上下文前缀 $x_{<t}$ 和真实下一个 token $x_t$。
2. 用当前模型快照 $\theta_{old}$ 作为 thought policy，采样一组 CoT：$c_t^{(1)},\dots,c_t^{(G)}$。
3. 对每条 CoT，把 $x_{<t}$ 和 $c_t^{(i)}$ 拼接起来，让同一个模型预测真实 token $x_t$。
4. 用 EMA teacher 在不拼接 CoT 的情况下预测同一个 $x_t$，得到 no-think baseline。
5. 计算每条 CoT 的 information-gain reward：

$$
r(c_t^{(i)})=\log p_\theta(x_t \mid x_{<t}, c_t^{(i)})-\log \bar{p}_\phi(x_t \mid x_{<t})
$$

6. 在同一上下文的多条 CoT 之间计算 group-relative advantage。
7. 只更新 CoT token 的生成概率：高 reward 的 CoT 更可能被生成，低 reward 的 CoT 概率被压低。
8. 更新 EMA baseline；下一次 training step 会重新 rollout 新的 CoT。

### EMA no-think baseline

baseline $\bar{p}_\phi(x_t \mid x_{<t})$ 是当前模型的 EMA teacher，不接收 thought。论文使用 $\tau=0.999$：

$$
\phi \leftarrow \tau\phi + (1-\tau)\theta
$$

这个 baseline 的作用是提供反事实比较：同样上下文下，如果不思考能给真实 token 多高概率？如果 thought 让概率更高，说明 thought 有预测价值。

EMA 的滞后也用于稳定训练。冻结 baseline 会让比较对象逐渐过时；完全同步又会让奖励趋近于零，或诱发退化策略。EMA 是两者之间的折中。

### Group-relative advantage

对同一个上下文，RLP 采样 $G$ 条 thoughts，计算每条 thought 的 reward。为了降低方差，它使用 group-relative baseline：

$$
\bar{r}=\frac{1}{G}\sum_{j=1}^{G}r(c_t^{(j)})
$$

并用修正后的 advantage：

$$
A^{(i)}=\frac{G}{G-1}\left(r(c_t^{(i)})-\bar{r}\right)
$$

这个形式接近 GRPO 的 group relative 思路：不需要单独训练 value model，而是在同一 prompt 的多个候选之间做相对比较。

### 只更新 thought tokens

论文明确说 RLP 不加入标准 NTP loss，优化的是 information-gain objective，并且梯度只作用到 sampled thought tokens；reward 本身被 stop-gradient，不反传穿过 $p_\theta$ 或 EMA baseline。

直观上，RLP 不是直接让 predictor 对 $x_t$ 做更强监督，而是让模型提高那些“能帮助预测真实 token 的 thought”的生成概率。也就是说，RLP 优化的是“产生有用思考”的策略。

### Clipped surrogate

RLP 对 thought tokens 使用 per-token importance ratio 和 clipped surrogate，形式类似 PPO/GRPO：

$$
\rho_u=\exp(\log\pi_\theta(\ell_u\mid prefix_u)-\log\pi_{\theta_{old}}(\ell_u\mid prefix_u))
$$

然后用 clipped objective 控制 policy update 的步幅。这样做是为了避免 thought policy 因高方差 reward 发生过大更新。

## 理论解释

论文给出一个关键恒等式：在固定 context 和 thought 下，reward 对真实数据分布取期望，等于 no-think baseline 与 reasoned predictor 的 cross-entropy 差：

$$
\mathbb{E}_{x_t\sim p^*}[r(c_t)]
=
CE(p^*, \bar{p}_\phi(\cdot\mid x_{<t}))
-
CE(p^*, p_\theta(\cdot\mid x_{<t}, c_t))
$$

因此，最大化 expected reward 等价于鼓励 thought 降低下一 token 的 cross-entropy。这个解释很重要：RLP 的 reward 不是任意启发式分数，而是有明确的预测改进含义。

论文还证明了对 thoughts 做 marginalization 后，CoT-conditioned objective 是一种 computable lower bound；附录进一步说明 position-wise reward 在 teacher forcing 下可以聚合成 sequence-level per-token CE improvement。

## 实验设计

论文主要使用两个模型：

- `qwen3-1.7b-base`：用于主实验、RPT 对比、数据源消融、rollout / completion length / KL 消融。
- `Nemotron-Nano-12B-v2`：12B hybrid Mamba-Transformer，用于验证规模和架构泛化。

对照组包括：

- base model；
- continued pretraining，即 CPT；
- RPT，使用 sparse binary next-token correctness reward；
- 相同 SFT + RLVR 后训练流程下的 base / CPT / RLP。

评测覆盖数学与科学推理：

- Math：GSM8K、MATH500、Minerva、AMC23、AIME25 等；
- Science：MMLU、MMLU-Pro、GPQA-Diamond。

论文还做了多种数据源实验，包括 OmniMath、OpenThoughts、Nemotron-Crossthink、ACAD、Math-Text、Web-Crawl 和 PT Data Mix，用于验证 RLP 是否只适用于 curated reasoning data。

## 关键实验结论

### Qwen3-1.7B 上的主结果

在 `qwen3-1.7b-base` 上，RLP 相比 base 和 CPT 都有明显提升。论文报告 overall average：

| Model | Overall |
|---|---:|
| Base | 30.32 |
| CPT | 30.85 |
| RLP | 36.03 |
| Base + Post | 39.34 |
| CPT + Post | 39.90 |
| RLP + Post | 42.51 |

这里最重要的不是单个 benchmark 的涨幅，而是两个现象：

- RLP 在预训练阶段就优于同 token 的 CPT；
- 相同 SFT + RLVR 后，RLP 的优势没有被后训练抹掉，而是继续保留，最终仍比 CPT + Post 高约 7% 相对提升。

### 12B hybrid 架构上的扩展

在 `Nemotron-Nano-12B-v2` 上，RLP 用一个 19.8T token intermediate checkpoint，再训练 250M tokens；base 对照则训练到 20T tokens。结果：

| Model | Math Avg | Science Avg | Science Avg@1[4] | Avg |
|---|---:|---:|---:|---:|
| Base | 61.38 | 34.51 | 32.54 | 42.81 |
| RLP | 65.33 | 57.26 | 61.37 | 61.32 |

Science Avg 的绝对提升尤其大。论文据此认为 RLP 不只是增强数学题，而是能强化更广义的多步解释型推理。

### 与 RPT 的比较

RLP 对比 RPT 的核心差异：

- RPT 依赖预选 token，reward 是 sparse binary next-token correctness；
- RLP 对所有位置都能计算 continuous information-gain reward；
- RLP 的 reward 直接评价 sampled CoT 对预测真实 token 的帮助，而不是只看最终 token 是否匹配。

matched data and compute 下，RLP 的 overall average 43.35，高于 RPT 的 41.69。

### 数据源泛化

论文的 Table 4 表明，RLP 不只在 SFT-style reasoning corpus 上有效，也能在 academic papers、math textbooks、web-crawl 等普通或半普通文本上提取推理信号。

这点是 RLP 最有价值的主张之一：如果 reward 可以从普通文本中自动构造，那么 RL 不再局限于小规模 curated verifiable datasets，而可以作为 pretraining / continued pretraining 阶段的通用增强目标。

### 算力公平性

RLP 需要 rollout，因此只看 input tokens 会低估计算成本。论文用 170M input tokens 的 RLP 对比 6B tokens 的 FLOP-matched CPT。即便 CPT 见到约 35 倍更多 token，RLP 在 Nemotron-Crossthink 设置下仍有更高 average。

这支持论文的结论：在作者的 FLOP-matched 估算下，RLP 的优势不能只用“额外 rollout 算力”解释，还与 information-gain reward 的目标设计有关。

### 超参数消融

论文的默认设置来自消融：

- rollouts：$G=16$ 最好，$G=32$ 有轻微下降；
- completion length：2048 是主要收益点，4096 只带来极小边际收益；
- KL coefficient：$\beta=0$ 最好，加入 token-level KL 没有净收益，还增加显存和 step time。

这个结论说明 RLP 很依赖足够长的 thought channel。短 thought，例如 64/128 tokens，表现很差；从 512 到 1024 有明显跃迁，2048 进一步提升。

## 局限与疑问

### 成本高于普通预训练

RLP 每个位置要采样多个 thoughts，并用 reasoned predictor 和 EMA baseline 计算 reward。虽然论文做了 FLOP-matched CPT 对比，但工程上仍然比普通 CPT 复杂很多，尤其是 rollout length 到 2048、rollouts 到 16 时。

### 是否会学习到“预测文本的解释”，不一定等价于真实推理

RLP 奖励的是 thought 对下一 token log-likelihood 的提升。这个目标鼓励模型生成有助于预测语料的解释性中间文本，但它不直接保证 thought 是忠实因果推理，也不保证 inference-time CoT 与内部决策过程一致。

### reward 来自模型自身，仍有自指风险

RLP 的 reasoned scorer 和 thought policy 是同一网络，baseline 是 EMA teacher。虽然 EMA lag 和 group-relative advantage 能缓解训练不稳定，但 reward 本质上仍是模型体系内部产生的。长期训练是否会出现 reward gaming、格式化 thought、过度依赖语料风格等问题，还需要更长期的实验。

### 结果集中在 reasoning benchmarks

论文覆盖数学、科学和部分通用知识 benchmark，但对开放式生成质量、安全、事实性、代码能力、长上下文任务、工具调用等方面没有展开。RLP 是否改善或损害这些能力，还不能从本文直接得出结论。

### 与标准 NTP 的关系还需要更多研究

论文写到训练不包含 standard NTP loss，只优化 information-gain objective。这很激进。后续值得追踪的是：RLP 是否适合作为预训练末期替换 NTP 的阶段性目标，还是应该与 NTP 混合使用；不同模型规模、数据质量、训练长度下是否会出现语言建模退化。

## 我的理解

RLP 的意义不在于“又一个 reasoning benchmark 提分方法”，而在于它提出了一个把 RL 前移到预训练阶段的可扩展接口：不需要 verifier，也不需要人工偏好，只要有普通文本，就可以把“先生成 thought，再看 thought 是否提升预测”的过程转成 dense reward。

这使预训练目标从单纯的 next-token likelihood 变成了“探索式预测”：模型不只是压缩上下文到下一个 token，而是被鼓励产生一段能解释、补全或组织上下文的信息。换句话说，RLP 把 CoT 从 inference-time prompting 或 post-training artifact，提前变成 pretraining-time action。

但它也提醒我们：所谓“推理”在这里仍然由预测增益定义。RLP 学到的是对下一 token 有用的 thought，而不是外部世界中严格可验证的 reasoning trace。因此，写入 topic note 时应该把它称为 reinforcement pretraining / verifier-free dense reward，而不是直接说它证明了模型在预训练中获得了真实推理能力。

## 可沉淀到 Topic Note 的内容

- [[training/pretraining/reinforcement-pretraining|Reinforcement Pretraining]]：RLP 可以作为一类方法的核心案例，即把 RL objective 前移到 pretraining / continued pretraining。
- [[training/pretraining/objective|Training Objective]]：next-token prediction 不是唯一可行的预训练目标；RLP 用 information gain reward 改造下一 token 预测。
- [[training/pretraining/pretraining|Pretraining]]：预训练末期可以引入显式 reasoning 行为，而不是只靠后训练补齐。
- [[training/mid-training/continued-pretraining|Continued Pretraining]]：RLP 与 CPT 的区别在于目标函数，而不仅是数据继续喂给模型。
- [[training/post-training/grpo|GRPO]]：RLP 借用了 group-relative advantage 与 clipped surrogate，但把使用位置从后训练搬到预训练。

## Related Notes

- [[training/pretraining/reinforcement-pretraining|Reinforcement Pretraining]]
- [[training/pretraining/objective|Training Objective]]
- [[training/pretraining/pretraining|Pretraining]]
- [[training/mid-training/continued-pretraining|Continued Pretraining]]
- [[training/post-training/grpo|GRPO]]
- [[training/post-training/rlhf|RLHF]]
