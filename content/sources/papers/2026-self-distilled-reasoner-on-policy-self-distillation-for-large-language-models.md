---
title: "Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models"
created: 2026-08-12
published: 2026-08-12
modified: 2026-08-12
type: source
status: processed
source_type: paper
area: sources
tags:
  - source
  - paper
  - distillation
  - on-policy-kd
  - self-distillation
  - reasoning
  - grpo
aliases:
  - OPSD
  - On-Policy Self-Distillation
  - Self-Distilled Reasoner
source_url: https://arxiv.org/abs/2601.18734
paper_date: "2026-01"
paper_order: "18734"
---

# Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models

## 基本信息

- 来源：arXiv:2601.18734v3
- 标题：Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models
- 作者/机构：Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, Aditya Grover / UCLA, HKU, Meta Superintelligence Labs
- 日期：2026-03-20
- 链接：https://arxiv.org/abs/2601.18734
- 代码：https://github.com/siyan-zhao/OPSD
- 相关 topic：[[training/post-training/knowledge-distillation|Knowledge Distillation]]，[[training/post-training/on-policy-kd|On-policy KD]]，[[training/post-training/grpo|GRPO]]，[[training/post-training/sequence-level-distillation|Sequence-level Distillation]]
- 相关 source：[[sources/papers/2024-on-policy-distillation-of-language-models|On-Policy Distillation of Language Models]]，[[sources/papers/2023-minillm-knowledge-distillation-of-large-language-models|MiniLLM: Knowledge Distillation of Large Language Models]]

这篇论文讨论的是 reasoning model 的一个更激进版本：不依赖外部 teacher，而是让同一个 LLM 在不同上下文下同时扮演 teacher 和 student。Teacher policy 能看到 privileged information，比如 ground-truth reasoning trace 或 reference solution；student policy 只看到问题本身。训练时让 student 按自己的 policy rollout，再让 teacher 在这些 rollout 上提供 token-level supervision。

它的关键点不在于“又一个蒸馏技巧”，而在于把 on-policy distillation 推进到 self-distillation：**模型可以借助正确答案和 privileged reasoning traces 教自己。**

## 研究问题

论文要回答的问题主要有四个：

1. 能否不依赖外部 teacher，而用同一个模型完成 on-policy distillation？
2. 能否直接利用 reasoning dataset 里的 ground-truth solution 作为 privileged information？
3. 如何把这种自蒸馏写成 dense token-level objective，而不是稀疏 sequence-level reward？
4. 这种做法在数学 reasoning 上是否能和 GRPO 相比，甚至更省 token？

## 核心主张

论文的核心主张可以概括为三句。

第一，student 生成自己的 rollout，但 teacher 不必是另一个更大的模型；同一个模型在 privileged context 下就能充当 teacher。

第二，teacher 不生成显式 tokens，而是通过不同上下文下的 forward pass 隐式 rationalize reference solution，并给 student rollout 提供全词表分布监督。

第三，这种 dense token-level self-distillation 在数学 reasoning 上可以达到和 GRPO 相近甚至更好的效果，同时 token efficiency 更高。

## 方法与机制

### Teacher / Student as Contextual Views

论文把同一个参数化模型 `p_θ` 拆成两个条件分布：

- teacher policy：`p_T(. | x, y*)`，能看到问题 `x` 和 reference solution `y*`
- student policy：`p_S(. | x)`，只看到问题 `x`

这里的关键不是两个不同模型，而是两个不同上下文视图。Teacher 利用 privileged information，student 保持和推理时一致的条件。

### On-policy Rollout

Student 先对问题 `x` 生成 rollout：

$$
\hat y \sim p_S(\cdot \mid x)
$$

然后 teacher 和 student 都在同一条 student rollout prefix 上给出 next-token 分布。训练目标是让 student 去拟合 teacher 在这些 prefixes 上的分布，而不是只拟合离线答案。

### Full-vocabulary Distillation

论文的主目标是 full-vocabulary divergence：

$$
L_{\mathrm{OPSD}}(\theta)
=
\mathbb{E}_{(x,y^\*)\sim S}
\mathbb{E}_{\hat y\sim p_S(\cdot\mid x)}
\frac{1}{|\hat y|}
\sum_{n=1}^{|\hat y|}
D\!\left(
p_T(\cdot\mid x,y^\*,\hat y_{<n})
\| 
p_S(\cdot\mid x,\hat y_{<n})
\right)
$$

这里的 divergence 可以是 forward KL、reverse KL 或 JSD。论文最终发现，forward KL 配合其余稳定器效果最好。

### Pointwise KL Clipping

作者观察到，token-level divergence 分布非常不均匀：少量 stylistic tokens 往往比数学相关 token 承担更大的训练信号，导致模型容易学到风格而非真正的 reasoning。于是引入 pointwise clipping，对 vocabulary-level divergence contribution 做截断，稳定训练。

### Sampled-token Alternative

论文也实现了 sampled-token policy-gradient 版本：只在 student 采样到的 token 上计算 teacher/student logprob 差，再把它作为 dense reward。这个版本更接近 on-policy distillation 的 RL 视角，但实验上不如 full-vocabulary logit distillation。

## 实验与证据

论文主要在数学 reasoning 上评估，模型规模包括 Qwen3-1.7B、4B 和 8B，训练数据来自 OpenThoughts 的数学 reasoning 子集，最多 3 万条 problem-solution pairs。

对比方法包括：

- SFT
- GRPO
- OPSD

主要结论是：

- OPSD 在多个 benchmark 上优于 SFT；
- OPSD 和 GRPO 相比，样本效率更高；
- OPSD 在 100 步左右就能收敛到很强的结果；
- GRPO 会遇到 reward std collapse，出现一批样本 reward 全相同，从而没有梯度；
- full-vocabulary objective 优于 sampled-token objective；
- pointwise clipping 对稳定训练非常关键。

## 关键结论

这篇论文的价值主要有两点。

第一，它把 self-distillation 和 on-policy distillation 结合起来，说明 privileged answer / reasoning trace 不一定非要外部 teacher 来提供，同一个模型也可以在不同上下文中自我蒸馏。

第二，它把 dense token-level supervision 和 sparse outcome reward 的差别讲得很清楚：OPSD 能在每个 token 上提供梯度，而 GRPO 依赖组内 reward 差异，一旦组内 reward 坍缩，训练信号就会变弱。

## 局限与疑问

- 论文主要在数学 reasoning 上验证，长程 agent 轨迹和工具调用还没有覆盖；
- privileged information 依赖 ground-truth solution，现实任务里未必总能拿到；
- 同模型双角色的自蒸馏稳定性依赖 prompt 设计和 clipping；
- 文章显示 forward KL 更稳，说明在这类任务里“更 on-policy”不一定自动带来更好效果，目标分布的 shape 同样重要。

## 相关知识链接

- [[training/post-training/knowledge-distillation|Knowledge Distillation]]
- [[training/post-training/on-policy-kd|On-policy KD]]
- [[training/post-training/grpo|GRPO]]
- [[training/post-training/sequence-level-distillation|Sequence-level Distillation]]
