---
title: "Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe"
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
  - reasoning
  - recipe
  - post-training
aliases:
  - OPD Phenomenology
  - Rethinking OPD
source_url: https://arxiv.org/abs/2604.13016
paper_date: "2026-04"
paper_order: "13016"
---

# Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe

## 基本信息

- 来源：arXiv:2604.13016v2
- 标题：Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe
- 作者/机构：Yaxuan Li, Yuxin Zuo, Bingxiang He, Jinqian Zhang, Chaojun Xiao, Cheng Qian, Tianyu Yu, Huan-ang Gao, Wenkai Yang, Zhiyuan Liu, Ning Ding / Tsinghua University, ShanghaiTech University, UIUC, Renmin University of China
- 日期：2026-04-15
- 链接：https://arxiv.org/abs/2604.13016
- 代码：https://github.com/thunlp/OPD
- 相关 topic：[[training/post-training/on-policy-kd|On-policy KD]]，[[training/post-training/knowledge-distillation|Knowledge Distillation]]，[[training/post-training/grpo|GRPO]]

这篇论文不是在提出一个新的 OPD 公式，而是在系统回答三个更基础的问题：OPD 什么时候会成功或失败，为什么 token-level reward 会起作用，以及当 OPD 失败时该怎么修。

它的价值在于把 OPD 从“有效的经验技巧”推进到“可解释的训练现象”。论文给出的结论很适合放在 OPD 机制链条里：**OPD 是否有效，首先取决于 teacher 和 student 的 thinking pattern 是否兼容；其次还取决于 teacher 是否真的带来了 student 之前没见过的新能力；而真正起作用的梯度主要集中在 student 与 teacher 的高概率 overlap tokens 上。**

## 研究问题

OPD 近来已经成为 reasoning model 后训练中的重要方法，但现有工作更多在展示它“能工作”，较少解释它“为什么工作、为什么失败、失败时怎么修”。

论文要回答的核心问题是：

1. 什么时候 OPD 能把 teacher 的能力有效转移给 student？
2. 为什么更强的 teacher 有时反而不如更弱的 teacher？
3. OPD 的 token-level reward 在学生实际访问到的状态上到底是如何起作用的？
4. 如果 OPD 不工作，能不能通过训练设计把它救回来？
5. OPD 的 dense token-level supervision 在长轨迹上是否存在上限？

## 核心主张

论文的核心主张可以概括成四条。

第一，OPD 不是单纯的“teacher 分数越高越好”，而是要求 teacher 和 student 共享 compatible thinking patterns。否则，即使 teacher benchmark 更高，student 也可能从一开始就落在 teacher support 之外，导致 token-level signal 不可用。

第二，更高的 benchmark score 并不必然意味着对 OPD 更有用。若 teacher 只是同一训练管线下的更大模型，它可能并没有给 student 提供真正新的知识；相反，只有 teacher 带来了 student 训练中尚未见过的新能力，OPD 才能稳定产生收益。

第三，OPD 的有效梯度主要集中在 student 和 teacher 的 shared high-probability tokens 上。有效训练不是在整个 vocab 上均匀推进，而是 student 逐步进入 teacher 的高概率区域，并在该区域内重新分配概率质量。

第四，OPD 的 dense token supervision 不是无代价的。响应越长，teacher 对后续 token 的 reward 越不可靠，最后会出现从后向前传播的 instability，因此 OPD 对长 horizon distillation 存在天然上限。

## 方法与机制

### 现象学：thinking pattern 与新知识

论文将 OPD 的成功与否归结为两个条件。

**Thinking-pattern consistency** 指 student 和 teacher 在 top-k token 分布上应具有兼容的思考模式。作者用 overlap ratio 作为一个直接信号：初始 overlap 越高，OPD 越容易起效。

**Higher scores ≠ new knowledge** 指即便 teacher 的 benchmark 更高，如果 teacher 和 student 是同一训练管线下的不同规模模型，它也可能没有提供 student 未见过的新知识。此时 OPD 训练目标会变得弱，甚至出现几乎没有收益的情况。

论文用 reverse distillation 做了验证：当 student 被蒸馏回自己的 pre-RL checkpoint 时，性能几乎回到原始水平；而更高分但同 family 的 teacher 也不一定带来更好效果。这说明 OPD 学到的不是 benchmark score 本身，而是 teacher 的 thinking pattern 和新增能力。

### 机制：progressive alignment on overlap tokens

论文定义了三个监控量：

- overlap ratio：student 与 teacher top-k token 集合的重叠程度；
- overlap-token advantage：重叠 token 上的局部对齐方向；
- entropy gap：student 与 teacher 在同一 state 上的熵差。

成功的 OPD 会表现为：

```text
overlap ratio 上升
entropy gap 收窄
overlap-token advantage 逐步接近 0
```

作者进一步发现，shared top-k tokens 往往已经承载了绝大多数概率质量，达到 97% - 99%。因此，OPD 的主要梯度信号并不在 non-overlap tokens，而在 overlap 区域内部的 reweighting。

这也解释了为什么只优化 overlap tokens，性能几乎可以接近完整 top-k OPD，而 non-overlap tokens 的收益很有限。

### Recipe：off-policy cold start 与 teacher-aligned prompts

当 student 和 teacher 的 thinking pattern 差距太大时，纯 OPD 可能从一开始就难以学习。论文提出两种修复方式。

**Off-policy cold start**：先用 teacher-generated rollouts 做一阶段 SFT，把 student 拉近 teacher 的 thinking pattern，再切回 OPD。这个做法能显著提高初始 overlap ratio，并让后续 OPD 的轨迹更稳定。

**Teacher-aligned prompt selection**：用 teacher post-training 时见过的 prompt template 或 prompt content 来做 OPD。这样可以提高 teacher supervision 的可用性，但也会压低 student entropy，因此更稳妥的做法是与分布外 prompts 混合。

### 长轨迹成本

论文还专门分析了 dense reward 在长轨迹上的代价。随着 response length 增长，teacher reward 的质量会下降，instability 往往从后半段 token 开始，并逐步向前传播。换言之，token-level OPD 在中短长度 reasoning 上有效，但并不天然适合更长的 chain-of-thought 或 agentic multi-turn setting。

## 实验与证据

### Thinking-pattern compatibility

作者对比了两个 teacher 给同一个 student 蒸馏的效果：一个是同 family 的 RL teacher，一个是更强但 thinking pattern 不完全兼容的 teacher。虽然后者 benchmark 更高，但前者的初始 overlap 更高、训练更稳、最终效果更好。

这说明 teacher quality 不能只看分数，得看它和 student 是否处于相似的 token-level decision space。

### Reverse distillation

论文进一步比较了 weak-to-strong reverse distillation。结果显示：

- 同 family 的 1.5B 和 7B teacher 对 student 的可迁移信号非常接近；
- 更高分的 teacher 不一定提供更多可学习信息；
- 当 teacher 只是更大但没有新的 post-training capability 时，OPD 的收益会明显受限。

### Overlap mechanism

论文在成功和失败的 OPD 运行之间对比发现：

- 成功 run 的 overlap ratio 从约 72% 持续上升到 91% 以上；
- entropy gap 持续缩小；
- overlap-token advantage 向 0 逼近；
- failing run 则几乎没有这种动态。

这组结果把“OPD 学到什么”讲得很清楚：它不是在整个 vocab 上平均学习，而是在 student visited states 上逐步找到 teacher 的高概率区域，并在该区域内部进行对齐。

### Recipe 验证

off-policy cold start 和 teacher-aligned prompts 都能显著改善原本失败的 OPD。前者通过 teacher rollouts 先对齐 thinking pattern，后者通过 prompt 匹配提升高概率 token 的可学习性。

## 关键结论

这篇论文最值得沉淀的结论有三条。

第一，OPD 的成败更像是一个分布对齐问题，而不是单纯的 teacher 强弱问题。thinking pattern 不兼容时，再强的 teacher 也可能没有用。

第二，OPD 的主要优化区域是 overlap tokens。也就是说，OPD 的“dense reward”并不意味着所有 token 都同等重要，真正有效的是 student 和 teacher 在同一 state 上共享的高概率 token。

第三，OPD 的收益随 trajectory depth 递减。短程 reasoning 可以被 dense token supervision 高效塑形，但长 horizon 的 agent 轨迹更可能暴露 reward degradation 和 local optimization geometry 问题。

## 局限与疑问

论文的实验主要集中在数学 reasoning 场景，尽管它明确指出了长轨迹上限，但并没有直接覆盖代码 agent、web agent 或多轮工具交互环境。

它提出的修复策略也有边界。off-policy cold start 需要额外 teacher rollouts；teacher-aligned prompts 可能压低 entropy，影响探索；而更长轨迹上的 reward degradation 说明 OPD 不是天然适合所有 agentic setting。

## 与 OPD 论文链的关系

[[sources/papers/2024-on-policy-distillation-of-language-models|On-Policy Distillation of Language Models]] 给出了 on-policy distillation 的基础定义：student self-generated trajectories 上的 teacher token-level supervision。

[[sources/papers/2026-learning-beyond-teacher-generalized-on-policy-distillation-with-reward-extrapolation|Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation]] 从 reward 设计角度把 OPD 推广为 G-OPD / ExOPD。

这篇 Rethinking OPD 则补上了 phenomenology / mechanism / recipe 三层解释：OPD 为什么会成功，为什么会失败，失败后如何修复，以及它为什么会在长轨迹上遇到瓶颈。

## 相关知识链接

- [[training/post-training/on-policy-kd|On-policy KD]]
- [[training/post-training/knowledge-distillation|Knowledge Distillation]]
- [[training/post-training/grpo|GRPO]]
- [[sources/papers/2024-on-policy-distillation-of-language-models|On-Policy Distillation of Language Models]]
- [[sources/papers/2026-learning-beyond-teacher-generalized-on-policy-distillation-with-reward-extrapolation|Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation]]
- [[sources/papers/2026-on-policy-delta-distillation|On-Policy Delta Distillation]]
