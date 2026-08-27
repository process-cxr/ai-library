---
title: "Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation"
created: 2026-08-11
published: 2026-08-11
modified: 2026-08-11
type: source
status: processed
source_type: paper
area: sources
tags:
  - source
  - paper
  - distillation
  - on-policy-kd
  - opd
  - grpo
  - reasoning
  - code
  - rl
aliases:
  - G-OPD
  - ExOPD
  - Generalized On-Policy Distillation
  - Reward Extrapolation
  - Reward Correction
source_url: https://arxiv.org/abs/2602.12125
paper_date: "2026-02"
paper_order: "12125"
---

# Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation

## 基本信息

- 来源：arXiv:2602.12125v2
- 标题：Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation
- 作者/机构：Wenkai Yang, Weijie Liu, Ruobing Xie, Kai Yang, Saiyong Yang, Yankai Lin / Renmin University of China, Tencent
- 日期：2026-02-27
- 链接：https://arxiv.org/abs/2602.12125
- 代码：https://github.com/RUCBM/G-OPD
- 相关 topic：[[training/post-training/on-policy-kd|On-policy KD]]，[[training/post-training/knowledge-distillation|Knowledge Distillation]]，[[training/post-training/grpo|GRPO]]

这篇论文是 OPD 线上的一个重要分支。它不再只把 on-policy distillation 看成“student-generated trajectory 上的 teacher logits supervision”，而是把 OPD 进一步改写成一个 dense KL-constrained RL 目标，并显式引入两个可调旋钮：reference model 和 reward scaling factor `λ`。

论文的核心结论很直接：**标准 OPD 只是 G-OPD 的特例；当 `λ>1` 时，reward extrapolation 往往能超过标准 OPD，甚至在 multi-teacher distillation 中把多个 domain teacher 的能力统一回一个 student；在强到弱蒸馏里，若能拿到 teacher 的 pre-RL 版本，reward correction 还能进一步提升效果。**

## 研究问题

标准 OPD 的基本形式是让 student 在自己采样出的 trajectories 上，去拟合 teacher 的 token-level logit distribution。这已经比 off-policy KD 更贴近 student 的实际状态分布，但它仍然存在两个固定约束：

1. reward 和 KL regularization 的相对权重是固定的；
2. reference model 的选择没有被显式参数化。

论文关注的问题是：这两个固定约束是否限制了 OPD 的上限。

更具体地说，作者想回答三件事：

1. OPD 是否可以严格地写成一个 dense KL-constrained RL 目标？
2. 如果把 reward weight 和 reference model 也纳入训练配方，是否能得到比标准 OPD 更强的 distillation？
3. 在 same-size multi-teacher 和 strong-to-weak 两种 setting 下，什么样的 reward shaping 更有效？

## 核心主张

论文的核心主张可以概括为三句。

第一，OPD 本质上是 dense KL-constrained RL 的一个特例。把 teacher logits 看作 implicit reward，OPD 的 token-level reward 等价于一个 reward term 和 KL regularization 等权重的 KL-constrained RL 目标。

第二，通过引入 reward scaling factor `λ` 和 flexible reference model `π_ref`，可以得到更一般的 G-OPD。`λ=1` 退化为标准 OPD；`0<λ<1` 是 reward interpolation；`λ>1` 是 reward extrapolation，对应 ExOPD。

第三，在 strong-to-weak distillation 中，如果可以访问 teacher 的 pre-RL checkpoint，把 reference model 从 student base 替换为 teacher base 可以得到更干净的 reward signal；这被作者称为 reward correction。

## 方法与机制

### OPD 作为 dense KL-constrained RL

论文先把标准 OPD 重写为：

$$
J_{\mathrm{OPD}}(\theta)
=
\mathbb{E}_{x\sim D, y\sim \pi_\theta(\cdot\mid x)}
\left[
\log \pi^\*(y\mid x)-\log \pi_\theta(y\mid x)
\right]
$$

再引入任意 reference model `π_ref`，得到：

$$
J_{\mathrm{OPD}}(\theta)
=
\mathbb{E}\left[
\log \frac{\pi^\*(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
-
D_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{ref}})
\right]
$$

这说明 OPD 不只是“模仿 teacher”，而是在 student 自采样轨迹上做 dense reward learning，只是 reward 和 KL 的权重被锁死在 1:1。

### G-OPD 目标

G-OPD 将上述形式推广为：

$$
J_{\mathrm{G-OPD}}(\theta)
=
\mathbb{E}_{x\sim D, y\sim \pi_\theta(\cdot\mid x)}
\left[
\lambda \log \frac{\pi^\*(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
-
D_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{ref}})
\right]
$$

其中 `λ` 控制 reward term 相对 KL regularization 的强度。

这个式子最有价值的地方是它给出了最优解的闭式形式：

$$
\log \pi_\theta(y\mid x)
=
\lambda \log \pi^\*(y\mid x) + (1-\lambda)\log \pi_{\mathrm{ref}}(y\mid x)
$$

因此：

- `0<λ<1` 时，student 的行为会落在 reference 与 teacher 之间；
- `λ=1` 时，回到标准 OPD；
- `λ>1` 时，student 会向 teacher 之外继续外推，这就是 reward extrapolation。

### ExOPD

论文把 `λ>1` 的 G-OPD 版本称为 ExOPD。它的直觉不是“更强地模仿 teacher”，而是“把 teacher 相对 reference 的偏移继续放大”。

在 same-size distillation 中，reference 通常取 student base model；在这种设定下，ExOPD 会让 student 不只逼近 teacher，而是继续沿着 teacher 的偏移方向推进。

论文报告，`λ=1.25` 往往是比较稳的点；`λ=1.5` 则开始出现不稳定，表现为 response length 增长过快、潜在 reward hacking 和性能回落。这说明 reward extrapolation 不是越大越好，它同时会放大 implicit reward 的偏差。

### Reward correction

在 strong-to-weak distillation 中，若 reference 取 student base，reward 其实是 `log π* / πstudent_base`。论文认为这会混入 teacher-student capacity gap 带来的噪声。

如果能拿到 teacher 的 pre-RL checkpoint，则把 reference 换成 `πteacher_base` 更合理，因为 `log π* / πteacher_base` 更接近 teacher 经过 RL 后真正学到的 reward shift。这个修正被称为 reward correction。

代价是：

- 需要额外访问 teacher base；
- 需要计算更大 reference model 的 logprob；
- 训练和部署成本更高。

## 实验与证据

### 实验设置

论文主要做了三组实验：

1. same-size single-teacher distillation：Qwen3-4B-Non-Thinking 作为 student，分别蒸馏 math teacher 和 code teacher；
2. multi-teacher distillation：把 math teacher 和 code teacher 的能力合回同一个 student；
3. strong-to-weak distillation：Qwen3-30B-A3B-Instruct-2507 作为 teacher，蒸馏到 Qwen3-1.7B / Qwen3-4B Non-Thinking。

训练数据方面，math 使用约 57K DeepMath 高难样本，code 使用约 25K Eurus-RL-Code 样本。Evaluation 覆盖 AIME24、AIME25、HMMT25、HumanEval+、MBPP+ 和 LiveCodeBench v6 等 benchmark。

### Same-size distillation

在 math reasoning 和 code generation 上，标准 OPD 已经能显著恢复 domain teacher 的行为，但 ExOPD 一致更强。

论文最重要的观察有三点：

1. `0<λ<1` 时，student 的性能和 response length 会落在 base 与 teacher 之间，符合 interpolation 预期；
2. `λ>1` 的 ExOPD 在合适范围内优于标准 OPD，且可能超过 domain teacher；
3. response length 和 entropy 会随着 `λ` 增大而上升，说明 reward extrapolation 同时在放大行为强度和多样性。

### Multi-teacher distillation

在把 math teacher 和 code teacher 的知识合回原始 base model 的设定里，ExOPD 是最强的方法。论文的结果显示：

- SFT 往往不稳定，容易低于 teacher；
- ExPO（weight extrapolation）并不总能得到可控的收益；
- OPD 能恢复部分 teacher 能力；
- ExOPD 是唯一能稳定把 unified student 推到所有 domain teacher 之上的方法。

这个结果很关键，因为它说明 G-OPD 不只是单 teacher 的蒸馏技巧，而是可以作为 multi-expert capability merging 的训练框架。

### Strong-to-weak distillation

在更强 teacher 到更弱 student 的设定里，ExOPD 仍然优于标准 OPD 和 SFT。论文给出的结论是：

- 对 Qwen3-1.7B Non-Thinking，ExOPD 平均分高于 OPD；
- 对 Qwen3-4B Non-Thinking，ExOPD 也稳定优于 OPD；
- 若进一步使用 reward correction，性能还能再提升。

论文同时说明，reward correction 并不是免费的：它需要 teacher pre-RL 模型，且会增加 logprob 计算开销。

## 关键结论

这篇论文的价值在于，它把 OPD 的几个“经验性技巧”结构化成了一个统一框架。

第一，OPD 的 reward 其实可以被显式参数化，而不必默认锁死在 teacher-vs-student 的固定权重上。

第二，reward extrapolation 说明 student 不一定只能在 teacher 边界内学习；在某些 multi-teacher 或 same-size distillation 场景里，向 teacher 之外进一步外推反而更强。

第三，reference model 的选择会显著影响 reward 语义。对 strong-to-weak distillation 来说，teacher base 往往比 student base 更像“干净的 reward anchor”。

## 局限与疑问

这篇论文的实验主要集中在 Qwen3 系列的 math 和 code 任务上，能说明 G-OPD 在 reasoning / coding distillation 中有效，但还不能直接外推到更复杂的长程 agent 轨迹、开放式工具调用或多轮环境反馈场景。

ExOPD 对 `λ` 比较敏感。论文已经显示过大 extrapolation 会带来 instability、length bias 和潜在 reward hacking，因此它更像一个需要仔细调参的框架，而不是通吃配方。

Reward correction 虽然更准确，但它要求额外模型和额外计算。对于大模型蒸馏，这个成本不一定总是可接受。

## 与 OPD 论文链的关系

[[sources/papers/2024-on-policy-distillation-of-language-models|On-Policy Distillation of Language Models]] 提供了 on-policy KD 的原始形式：student 自采样，teacher 在 student prefixes 上提供 token-level supervision。

[[sources/papers/2026-on-policy-delta-distillation|On-Policy Delta Distillation]] 进一步把 teacher signal 改写为 `teacher - teacher_base` 的 delta reward，关注 reasoning tuning 的增量。

这篇 G-OPD / ExOPD 则介于两者之间：它不改动 OPD 的基本 on-policy 结构，但把 reward weight 和 reference model 显式参数化，从而把 OPD 从一个固定配方扩展成可插值、可外推、可校正的 dense RL 形式。

## 相关知识链接

- [[training/post-training/on-policy-kd|On-policy KD]]
- [[training/post-training/knowledge-distillation|Knowledge Distillation]]
- [[training/post-training/grpo|GRPO]]
- [[sources/papers/2024-on-policy-distillation-of-language-models|On-Policy Distillation of Language Models]]
- [[sources/papers/2026-on-policy-delta-distillation|On-Policy Delta Distillation]]
