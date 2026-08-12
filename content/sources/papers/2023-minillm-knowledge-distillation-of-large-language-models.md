---
title: "MiniLLM: Knowledge Distillation of Large Language Models"
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
  - post-training
  - generative-model
aliases:
  - MiniLLM
  - MiniLLM: On-Policy Distillation of Large Language Models
  - On-Policy Distillation of Large Language Models
source_url: https://arxiv.org/abs/2306.08543
---

# MiniLLM: Knowledge Distillation of Large Language Models

## 基本信息

- 来源：arXiv:2306.08543v6，ICLR 2024
- 标题：MiniLLM: Knowledge Distillation of Large Language Models
- 作者/机构：Yuxian Gu, Li Dong, Furu Wei, Minlie Huang / Microsoft Research Asia
- 日期：2023-06-14（arXiv 初版）
- 链接：https://arxiv.org/abs/2306.08543
- 相关 topic：[[training/post-training/knowledge-distillation|Knowledge Distillation]]，[[training/post-training/on-policy-kd|On-policy KD]]，[[training/post-training/sequence-level-distillation|Sequence-level Distillation]]
- 相关 source：[[sources/papers/2024-on-policy-distillation-of-language-models|On-Policy Distillation of Language Models]]

这篇论文把生成式 LLM 的知识蒸馏问题，重新写成一个更适合 on-policy 学习的优化问题。它的关键判断是：标准 KD 常用的 forward KL 更偏向覆盖 teacher 的所有模式，但对生成任务来说，student 往往容量有限，强行覆盖 teacher 的低概率区域会把概率质量分配到很多不该重点学习的区域，最后影响输出质量。

MiniLLM 的处理方式是改用 reverse KL，并把蒸馏过程写成基于 student 自身采样轨迹的优化。换句话说，它不是只拿 teacher 的固定答案序列训练 student，而是让 student 在自己的生成分布上接受 teacher 反馈，再把这个反馈转化成可优化的训练信号。

## 研究问题

论文要回答的问题很直接：

1. 标准 sequence-level KD 为什么不够适合生成式 LLM？
2. reverse KL 是否比 forward KL 更适合小模型学习大模型的生成分布？
3. 如果 student 需要在自己会走到的 prefixes 上学习，应该如何稳定地做 on-policy 优化？
4. 这种做法能否在 instruction following、长文本生成和不同模型家族上稳定工作？

## 核心主张

MiniLLM 的核心主张可以概括为三点。

第一，生成式 KD 的目标不应只是“模仿 teacher 的完整输出”，而应更关心 student 自己会访问到的生成状态。因为推理时 student 不是沿着 teacher 的轨迹走，而是沿着自己的轨迹往前生成。

第二，reverse KL 比 forward KL 更符合这个场景。forward KL 倾向 mode-covering，容易把概率质量铺到 teacher 的低概率区域；reverse KL 更 mode-seeking，更强调 teacher 的主要模式，因此更适合容量有限的 student。

第三，reverse KL 不是直接离线算出来就完事了，论文把它进一步写成一个 on-policy 优化过程，并加入若干稳定器，让这个目标可以真正训练起来。

## 方法与机制

### Reverse KL for Generative KD

标准 KD 通常最小化 forward KL：

$$
\mathrm{KL}(p \,\|\, q_\theta)
$$

MiniLLM 改成最小化 reverse KL：

$$
\mathrm{KL}(q_\theta \,\|\, p)
$$

这里 $p$ 是 teacher 分布，$q_\theta$ 是 student 分布。直观上，这样 student 更像是在 teacher 的高概率区域里找一个自己能稳定生成的近似，而不是试图覆盖 teacher 的全部表达空间。

### On-policy Optimization

论文把这个目标进一步改写成基于 student rollout 的优化。Student 先按当前 policy 采样 response，然后 teacher 在这些 student-generated prefixes 上给出 token-level 反馈。这样训练的重点就从“固定参考答案上的拟合”转到了“student 自己会走到的状态上的修正”。

### Stabilizers

为了让这个 on-policy 过程稳定下来，论文用了三个关键设计：

- single-step decomposition：把序列级目标拆成逐步更新，降低方差；
- teacher-mixed sampling：混入 teacher 生成样本，缓解 reward hacking；
- length normalization：消掉对长输出的天然偏置。

此外，论文还保留了预训练语言建模目标作为辅助项，以减少蒸馏过程对基础语言能力的破坏。

## 实验与证据

论文在 instruction-following 场景里做了系统实验，覆盖 summarization、translation、arithmetic reasoning 和 instruction tuning 等任务，并用多个指标和反馈源评估结果，包括 ROUGE-L、GPT-4 feedback 和 human judgment。

实验结论比较稳定：

- MiniLLM 比标准 KD 基线更准确，整体质量更高；
- exposure bias 更低；
- calibration 更好；
- 长文本生成能力更强；
- 从 120M 到 13B 的多个模型家族都能工作。

这说明 reverse KL + on-policy optimization 不只是一个理论重写，而是能在生成任务里形成可重复的收益。

## 关键结论

MiniLLM 的长期价值不在于它是一个“更复杂的 KD 版本”，而在于它明确指出：对生成式 LLM 来说，蒸馏的核心对象不只是 teacher 的答案，还包括 student 自己的生成分布。

它把一个原本偏静态的 KD 问题，变成了“在 student 轨迹上做分布修正”的问题。这个思路后来也很自然地衔接到后续的 on-policy distillation、OPD 和 reasoning distillation 里。

## 局限与疑问

- reverse KL 是 mode-seeking 的，可能牺牲一部分多样性；
- on-policy 优化比普通 SFT 更复杂，对训练稳定性更敏感；
- teacher 反馈质量仍然决定上限，teacher 自己的偏差也会被传递；
- 这类方法更适合“提升生成质量和可靠性”，不适合被理解成万能蒸馏框架。

## 相关知识链接

- [[training/post-training/knowledge-distillation|Knowledge Distillation]]
- [[training/post-training/on-policy-kd|On-policy KD]]
- [[training/post-training/sequence-level-distillation|Sequence-level Distillation]]
- [[sources/papers/2024-on-policy-distillation-of-language-models|On-Policy Distillation of Language Models]]
