---
title: "Rethinking Entropy Interventions in RLVR: An Entropy Change Perspective"
created: 2026-07-15
published: 2026-07-15
modified: 2026-07-15
type: source
status: processed
source_type: paper
area: sources
tags:
  - source
  - paper
  - rlvr
  - entropy
  - grpo
  - ppo
  - reinforcement-learning
  - reasoning
  - coding
aliases:
  - STEER
  - Stabilizing Token-level Entropy-changE via Reweighting
source_url: https://arxiv.org/abs/2510.10150
paper_date: "2025-10"
paper_order: "10150"
---

# Rethinking Entropy Interventions in RLVR: An Entropy Change Perspective

## 基本信息

- 来源：arXiv:2510.10150v4
- 标题：Rethinking Entropy Interventions in RLVR: An Entropy Change Perspective
- 作者/机构：Zhezheng Hao, Hong Wang, Haoyang Liu, Jian Luo, Jiarui Yu, Hande Dong, Qiang Lin, Can Wang, Jiawei Chen / Zhejiang University, Tencent, Independent Researcher, Hangzhou High-Tech Zone (Binjiang) Institute of Blockchain and Data Security
- 日期：2026-04-29
- 链接：https://arxiv.org/abs/2510.10150
- 相关 topic：[[training/post-training/grpo|GRPO]]，[[training/post-training/ppo|PPO]]，[[training/post-training/rlhf|RLHF]]

这篇论文讨论的是 RLVR 训练中的 entropy collapse 问题。作者的判断很直接：很多 post-training 方法已经意识到 policy entropy 会在训练中快速下降，但现有 entropy intervention 多数只是经验性地调一两个开关，缺少统一的机制解释。STEER 的价值不在于“又加了一个 trick”，而在于它把 token-level entropy change 拆开分析，并据此把干预策略从粗粒度的全局修补，推进到基于理论估计的 token reweighting。

如果只保留一个核心观点：**RLVR 里的 entropy control 不能只盯着整体 entropy 曲线，而要看每个 token 在更新一步之后到底是怎么变的。**

## 研究问题

RLVR（Reinforcement Learning with Verifiable Rewards）已经成为 reasoning model 训练中的主力方法之一，但训练过程常伴随 entropy collapse：policy entropy 在较短训练步数内迅速下降，模型输出变得越来越单一，探索能力也随之变差。对于数学推理和 coding 这类需要持续尝试、回退和发现新路径的任务，这种 collapse 会直接限制性能上限。

作者关注的不是“entropy 是否会降”，而是更细的问题：

1. 具体是哪些 token 在推动 entropy 下降？
2. clipping、advantage、token probability、conditional entropy 这几个因素如何共同决定 entropy change？
3. 现有方法为什么只能部分缓解 collapse？
4. 能否构造一个基于 entropy change 估计的统一 reweighting 机制？

## 核心主张

论文的核心主张有三层。

第一，token-level entropy change 可以被做成一个相当准确的局部近似，并且其变化由四个因素共同控制：clipping strategy、advantage、token probability 和 conditional entropy。也就是说，entropy collapse 不是单一变量造成的，而是更新规则在 token 粒度上的组合效应。

第二，现有 entropy intervention 方法之所以有效，是因为它们都在不同程度上重加权了这四个因素中的一部分，但它们普遍是局部的、经验性的：有的只改 clipping upper bound，有的只改正样本权重，有的只改 advantage 的熵感知形式，却没有同时覆盖全部关键因素。

第三，STEER（Stabilizing Token-level Entropy-changE via Reweighting）把理论估计到的 entropy change 直接转成 token weight，对 entropy 变化过大的 token 降权，从而在训练时更平滑地维持探索。

## 方法与机制

### Token-level Entropy Change

论文先给出一个 first-order approximation，描述 policy 在一次更新后某个 state 上的 token-level entropy 如何变化。近似误差是 $O(\eta^2)$，因此在常见的小学习率下可作为实际估计器使用。

这个估计式的关键不在公式形式本身，而在它揭示的四个 governing factors：

- `I_clip`：裁剪指示项，决定哪些 token 的更新会被截断
- `A`：advantage，决定更新方向和强度
- `\pi_{old} / \pi_\theta`：旧策略与当前策略的相对概率
- `H(s)`：conditional entropy，反映当前状态下的分布不确定性

作者还用实验验证了这个估计器。和此前的 Cov 类估计相比，新的 estimator 在三组模型上都显著更准，MSE 达到 `1e-4` 量级，并且 PCC / SRCC 也明显更高。

### Existing Interventions 的统一解释

论文把已有 entropy intervention 方法分成三类：

- `Clip-Higher`
- `Positive-Reweighting`
- `Entropy-Aware Advantage`

然后用 token-level 视角解释它们分别在调哪些因素。

例如 `Clip-Higher` 通过放宽 upper clip bound，让更多低概率 positive samples 进入更新，因此更容易维持 entropy；`Positive-Reweighting` 则通过重加权正样本，削弱 entropy 下降方向上的贡献；`Entropy-Aware Advantage` 则直接把更高 entropy 的 token 赋予更大 advantage。

但这些方法的共同问题是覆盖不全。论文的 Table 2 说明，大多数已有方法只碰到四个因素里的两到三个，而且往往只控制部分 token，因而难以形成稳定的全局效果。

### STEER

STEER 的做法很朴素：先用理论估计式得到每个 state/token 的 entropy change，再把这个变化量映射为 token-level weight `\lambda(s)`，并写回训练目标。

权重形式是指数衰减：

```text
λ(s) = exp(-α |Ω(s)| / max |Ω(s)|)
```

直觉上，entropy 变化越剧烈的 token，梯度权重越小；entropy 相对平稳的 token，保留更多训练信号。这样做不是去“手工修补某一类样本”，而是把 entropy control 变成一个连续的 token reweighting 问题。

STEER 的定位也比较清楚：它不是替代 GRPO / PPO，而是作为一种可以叠加到这些 RLVR 算法上的稳定器。

## 实验与证据

### 数学推理

在 Qwen2.5-Math-7B 上，论文比较了 GRPO、SimpleRL-Zoo、Eurus-PRIME、OPO、clip-high、entropy loss、fork tokens、W-REINFORCE、Entropy-Aware Advantage、Clip-Cov、KL-Cov 和 STEER。

结果如下：

| Method             | AIME24 | AIME25 | AMC23 | MATH500 | Minerva | Olympiad | Avg. |
| ------------------ | -----: | -----: | ----: | ------: | ------: | -------: | ---: |
| GRPO               |   28.0 |   14.3 |  66.2 |    78.6 |    37.3 |     40.9 | 44.2 |
| OPO                |   32.2 |   13.4 |  71.5 |    82.2 |    38.2 |     41.0 | 46.4 |
| Entropy-Aware Adv. |   27.5 |   13.5 |  70.2 |    79.6 |    36.8 |     42.8 | 45.1 |
| Clip-Cov           |   32.5 |   12.9 |  68.4 |    78.0 |    40.8 |     41.3 | 45.7 |
| STEER              |   36.2 |   16.1 |  72.1 |    82.2 |    41.7 |     43.0 | 48.6 |

在 Qwen2.5-14B 上，STEER 也保持领先，平均分达到 `45.2`，高于 GRPO 的 `42.9`、OPO 的 `42.9` 和 Clip-Cov 的 `41.2`。

### 真实 coding 任务

论文还在三个 real-world coding benchmarks 上比较了 STEER 和 GRPO：

| Dataset  | 3B GRPO | 3B STEER | 7B GRPO | 7B STEER | 14B GRPO | 14B STEER |
| -------- | ------: | -------: | ------: | -------: | -------: | --------: |
| Internal |    39.7 |     41.2 |    40.1 |     41.9 |     42.6 |      45.1 |
| Zeta     |    17.4 |     19.3 |    22.0 |     24.0 |     22.3 |      24.1 |
| LCB-v5   |    24.4 |     24.9 |    28.5 |     29.2 |     29.3 |      31.8 |

这组结果说明，STEER 不只是在数学 benchmark 上有效，在 code benchmark 上也能带来稳定但不夸张的收益。

### 训练动态

论文还给出训练过程中的 entropy dynamics。关键现象有两个：

- vanilla GRPO 会较快出现 collapse
- 加入 DIS / STEER 之后，entropy 曲线更平稳，性能也更持久

作者还验证了 STEER 可以迁移到 `GRPO`、`RLOO` 和 `OPO` 等不同 RLVR 算法上，而不是只对某一个实现有效。

## 关键结论

这篇论文最值得沉淀的知识点是：

1. entropy collapse 不是一个笼统的“探索不足”问题，而是 token-level update dynamics 的结果。
2. 仅调 clipping 或仅调 positive sample 权重，通常只能覆盖部分 entropy-driving factors。
3. 用理论估计的 token-level entropy change 做 reweighting，比经验性 entropy hack 更稳定。
4. RLVR 中真正可复用的不是某个固定 trick，而是“先解释 entropy 怎么变，再决定怎么干预”的方法论。

## 局限与疑问

STEER 解决的是 RLVR 训练中的 entropy stability，不是 open-ended agent learning。它的实验主要覆盖数学推理和 coding 任务，仍然属于 reward 可验证、输出较短、反馈较明确的场景。

另外，这类方法仍然依赖较准的 entropy change 估计和合适的超参数设置。它能改善探索，但不能替代 reward design、data quality、rollout diversity 这些更上游的问题。

## 相关知识链接

- [[training/post-training/grpo|GRPO]]
- [[training/post-training/ppo|PPO]]
- [[training/post-training/rlhf|RLHF]]
