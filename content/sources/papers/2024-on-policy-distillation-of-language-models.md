---
title: "On-Policy Distillation of Language Models"
created: 2026-08-10
published: 2026-08-10
modified: 2026-08-10
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
  - gkd
  - post-training
aliases:
  - GKD
  - Generalized Knowledge Distillation
  - On-Policy Distillation
  - Learning from Self-Generated Mistakes
source_url: https://arxiv.org/abs/2306.13649
paper_date: "2023-06"
paper_order: "13649"
---

# On-Policy Distillation of Language Models

## 基本信息

- 来源：arXiv:2306.13649v3，ICLR 2024
- 标题：On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes
- 作者/机构：Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos, Matthieu Geist, Olivier Bachem / Google DeepMind, Mila, University of Toronto
- 日期：2024-01-17
- 链接：https://arxiv.org/abs/2306.13649
- 相关 topic：[[training/post-training/on-policy-kd|On-policy KD]]，[[training/post-training/knowledge-distillation|Knowledge Distillation]]，[[training/post-training/logits-distillation|Logits Distillation]]，[[training/post-training/sequence-level-distillation|Sequence-level Distillation]]

这篇论文提出 Generalized Knowledge Distillation（GKD），是后续讨论 On-Policy Distillation（OPD）时需要回到的源头工作之一。它关注的问题不是现代 reasoning model 中的 token reward shaping，而是 autoregressive language model 蒸馏里的 train-inference distribution mismatch：训练时 student 学的是固定数据或 teacher-generated sequences，推理时却必须在自己生成的 prefixes 上继续生成。

GKD 的核心处理方式是：让 student 用当前 policy 自己生成 output sequence，再让 teacher 在这些 student-generated prefixes 上提供 token-level probability supervision。这样，student 学到的不是 teacher 在离线参考序列上的分布，而是在 student 自己实际会访问到的状态上的 teacher 分布。

如果只保留一个核心观点：**autoregressive KD 不应只看固定答案序列，还应把 student self-generated trajectories 纳入训练分布；teacher 的作用是在这些 on-policy states 上提供 token-level expert signal。**

## 研究问题

传统 KD 在 autoregressive sequence model 上有两种常见形式。

第一种是 supervised KD：给定固定的 ground-truth outputs，teacher 对这些 token prefixes 给出 soft logits，student 最小化 teacher 与 student 的 token distribution 差异。

第二种是 sequence-level KD：teacher 先生成完整输出序列，再用这些 teacher outputs 对 student 做 supervised training。

这两种方法都没有直接解决 student 推理时的状态分布问题。Autoregressive generation 的每一步都依赖前面已经生成的 token。如果 student 在早期生成了偏离 teacher 或参考答案的 token，后续 prefix 就会进入训练数据中没有覆盖的区域。固定数据上的 KD 即使能学到 teacher 在参考 prefixes 上的分布，也不一定能告诉 student 在自己犯错后的 prefixes 上应该如何恢复。

论文把这个问题类比为 imitation learning 中的 exposure bias：如果 learner 只在 expert trajectories 上训练，部署时一旦进入 learner 自己导致的状态，就可能出现误差累积。GKD 的切入点是把 autoregressive KD 改写为 interactive imitation learning：student rollout 产生当前状态，teacher 在这些状态上提供 expert label。

## 核心主张

论文的核心主张可以拆成三层。

第一，KD for autoregressive LMs 可以被视为 imitation learning。输出序列中的 prefix 相当于状态，next-token distribution 相当于 action distribution，teacher 是 expert，student 是 learner。固定数据 KD 只覆盖离线状态；on-policy KD 覆盖 student 当前会访问到的状态。

第二，student-generated data fraction 是 GKD 的关键训练旋钮。GKD 用 $\lambda$ 控制训练 batch 中有多少比例来自 student self-generated outputs。$\lambda=0$ 时退化为 supervised KD；$\lambda=1$ 时变成纯 on-policy KD；中间值则混合固定数据和 student on-policy data。

第三，divergence choice 不必固定为 forward KL。Student 容量通常小于 teacher，强行覆盖 teacher 的完整分布可能导致低质量或幻觉式输出。GKD 允许使用 forward KL、reverse KL 或 generalized JSD，在 mode-covering、mode-seeking、生成多样性和质量之间做任务相关折中。

## 方法与机制

### Token-Level Discrepancy

论文首先把 teacher 与 student 在一条输出序列 $y$ 上的差异定义为 token-level divergence 的平均：

$$
D(p_T\|p_S^\theta)(y\mid x)
=
\frac{1}{L_y}
\sum_{n=1}^{L_y}
D\left(
p_T(\cdot\mid y_{<n},x)
\|
p_S^\theta(\cdot\mid y_{<n},x)
\right)
$$

这个定义很重要，因为 GKD 的 teacher signal 并不是只给最终答案分数，而是在 student 已经走到的每个 prefix 上提供下一 token 分布。

### On-Policy KD Objective

On-policy KD 让 student 先生成序列：

$$
y\sim p_S(\cdot\mid x)
$$

然后 teacher 在这些 student-generated prefixes 上提供 token-level probabilities。对应 objective 是：

$$
L_{\mathrm{OD}}(\theta)
=
\mathbb{E}_{x\sim X}
\left[
\mathbb{E}_{y\sim p_S(\cdot\mid x)}
\left[
D_{\mathrm{KL}}(p_T\|p_S^\theta)(y\mid x)
\right]
\right]
$$

论文强调，这里不对 student sampling distribution 反向传播。训练更接近 supervised learning 或 on-policy imitation learning，而不是高方差 policy-gradient RL。Student 采样只负责产生当前 policy 会访问到的序列；梯度来自 teacher 与 student 在这些 prefixes 上的 token distribution discrepancy。

### Generalized Knowledge Distillation

GKD 把 supervised KD 和 on-policy KD 统一到一个混合目标：

$$
L_{\mathrm{GKD}}(\theta)
=
(1-\lambda)
\mathbb{E}_{(x,y)\sim(X,Y)}
\left[
D(p_T\|p_S^\theta)(y\mid x)
\right]
+
\lambda
\mathbb{E}_{x\sim X}
\left[
\mathbb{E}_{y\sim p_S(\cdot\mid x)}
\left[
D(p_T\|p_S^\theta)(y\mid x)
\right]
\right]
$$

其中 $\lambda$ 是 student data fraction，$D$ 可以是 forward KL、reverse KL 或 generalized JSD。Algorithm 1 的训练过程也很直接：每一步按概率 $\lambda$ 决定是用 student 生成输出，还是从固定数据集中采样输出，然后最小化 teacher-student divergence。

这个统一形式有两个好处。

一是把 SeqKD、supervised KD、mixed imitation KD 和纯 on-policy KD 放在同一坐标系里比较。不同方法不再只是名字不同，而是可以用 output sequence source 与 divergence 两个轴描述。

二是可以独立调节“训练状态分布”和“分布对齐方式”。前者由 $\lambda$ 控制，后者由 divergence 控制。

### Divergence Choice

Forward KL 倾向 mode-covering，希望 student 覆盖 teacher distribution 的支持集。对于容量足够的 student，这可以保留更完整的 teacher 偏好；但当 student 容量明显不足时，强行覆盖 teacher 分布可能让 student 给 teacher 低概率 token 分配过多质量，造成低质量生成。

Reverse KL 更 mode-seeking，倾向集中到 teacher 高概率区域。它可以减少低质量样本，但通常牺牲多样性。

Generalized JSD 用系数 $\beta$ 在 forward KL 与 reverse KL 之间插值，并且是 bounded divergence。论文实验显示，最优 divergence 与任务和 decoding temperature 相关：摘要任务在高温采样下更受 mode-seeking divergence 影响；translation 中 JSD 变体表现较好；instruction tuning 中 reverse KL 更有效。

因此，GKD 的一个稳定结论是：on-policy data source 是核心收益来源，但 divergence 不是固定答案，需要结合任务、student 容量和部署时 decoding 方式调节。

### 与 RL Fine-Tuning 的组合

论文还把 on-policy GKD 与 RL fine-tuning 合并为一个 regularized objective：

$$
\mathbb{E}_{x\sim X}
\left[
(1-\alpha)
\mathbb{E}_{y\sim p_S^\theta(\cdot\mid x)}[r(y)]
-
\alpha
\mathbb{E}_{y\sim p_S(\cdot\mid x)}
\left[
D(p_T\|p_S^\theta)(y\mid x)
\right]
\right]
$$

$\alpha$ 控制 reward optimization 与 distillation regularization 的强度。这个部分的意义在于：如果任务有外部 reward 或 verifier，可以用 RL 优化目标指标，同时用 teacher distribution 保持生成质量和能力不塌缩。

需要注意的是，本文中的 GKD 本身不是 PPO/GRPO 式 RL 算法。它可以和 RL 组合，但基础 GKD 仍是 student 采样 + teacher token-level supervision 的蒸馏训练。

## 实验与证据

### 实验设置

论文主要使用 T5 系列模型做 teacher-student distillation。Teacher 通常是 fine-tuned T5-XL，约 3B 参数；student 包括 T5-small、T5-base、T5-large。任务覆盖四类：

- XSum abstractive summarization，使用 ROUGE-2；
- WMT14 English-German translation，使用 BLEU；
- GSM8K arithmetic reasoning，结合 few-shot CoT 和 calculator 评测 accuracy；
- FLAN instruction tuning，使用 held-out MMLU 和 BBH few-shot accuracy。

对 GKD 变体，论文系统比较了 $\lambda=0,0.5,1$，以及 forward KL、reverse KL、JSD(0.1)、JSD(0.5)、JSD(0.9)。

### XSum

在 XSum 上，on-policy GKD 相比 supervised KD、SeqKD、ImitKD、f-distill 有一致提升。数据效率实验显示，T5-small 只用 5% XSum input 做 on-policy GKD，即不依赖 ground-truth summaries，也能超过使用完整训练集的 supervised KD 和 ImitKD。

Divergence 消融显示，mode-seeking divergence 在 temperature sampling 下能提高质量但降低多样性；当 evaluation 改为 greedy decoding 时，不同 divergence 的性能差距会缩小。这说明 divergence 的选择不能脱离部署时 sampling strategy。

论文还在 XSum 上展示了 RLAIF + on-policy GKD。用 textual entailment score 作为 reward 可以提升 factual consistency，而 GKD regularization 帮助保留摘要质量。实验结果表明，相比只向原 student regularize 的 RLEF，向 teacher distribution regularize 可以在 factual consistency 和 ROUGE-2 之间取得更好的折中。

### WMT Translation

在 WMT English-German translation 上，纯 on-policy data 和 mixed data 变体整体优于只使用固定 supervised dataset 的 GKD。Figure 6 中，$\lambda=1$ 的 on-policy variants 在 T5-small 和 T5-base 上普遍更强，JSD 变体表现突出。

这组实验强化了论文的核心判断：student-generated outputs 不是辅助技巧，而是解决 autoregressive KD distribution mismatch 的关键数据来源。

### GSM8K Reasoning

GSM8K 实验使用 CoT prompting 和 calculator 检查最终答案。Student 从 FLAN-T5 模型开始，在 PaLM-540B 生成的 CoT 数据上先做 supervised fine-tuning，再进行 distillation。

结果显示，on-policy GKD 明显优于 supervised KD、SeqKD、ImitKD 和 f-distill。随着 student-generated data proportion 提高，超过 25% 后性能通常继续改善；只使用固定 CoT 数据或混合固定数据都不如纯 on-policy student-generated CoT。

这对 reasoning distillation 很关键：如果 student 的中间推理 prefix 会偏离 teacher trace，那么只在 teacher 或固定 CoT 上做监督，并不能覆盖 student 自己推理中出现的错误状态。On-policy GKD 的优势就在于 teacher 可以直接在 student 自己生成的 reasoning prefixes 上给出 token distribution feedback。

### Task-Agnostic Instruction Distillation

在 FLAN instruction tuning setting 中，论文用 FLAN2021 数据蒸馏 FLAN T5-XL 到 FLAN T5-base，并在 held-out MMLU 和 BBH 上评估。On-policy GKD with reverse KL 明显优于 supervised KD 和 ImitKD。

论文给出的解释是，instruction tuning 更关注抓住 instruction 的主意图和核心行为，reverse KL 的 mode-seeking 特性可能更有利于 student 聚焦主要响应模式，而不是覆盖 teacher 的完整表达分布。

## 关键结论

GKD 对 OPD 相关研究的价值主要不在某个 benchmark 数字，而在它确立了一个清晰训练范式：

```text
student rollout
  -> teacher token-level feedback on student prefixes
  -> student update without backprop through sampling
  -> repeat with refreshed student distribution
```

这个范式把 KD 从固定数据上的 teacher imitation，推进到 student current distribution 上的 expert feedback。后续 OPD、ExOPD、OPD2 等工作虽然目标、reward 形式和模型时代已经变化，但核心问题仍然延续自这里：如何在 student 自己访问到的状态上，把 teacher 的可用信号转化成稳定的训练更新。

对现代 reasoning / agent / tool-use distillation 来说，这篇论文提供了三个可复用原则。

第一，训练数据的 prefix distribution 很重要。只蒸馏 teacher trajectories 可能无法覆盖 student 自己会走到的错误状态。

第二，teacher signal 可以是 token-level distribution，不一定必须是完整 correction 或最终 reward。Dense token signal 能在每个 prefix 上提供更细粒度的学习方向。

第三，on-policy distillation 不必等同于 RL。只要采样来自当前 student，反馈来自 teacher，并且更新作用在 student 访问到的状态上，就已经具备 on-policy KD 的核心结构。

## 局限与疑问

论文的实验主要基于 T5 / FLAN-T5 时代的 encoder-decoder models，模型规模、训练配方和任务形态与当前 decoder-only reasoning model、agent model 有明显差异。它证明了 on-policy data 对 autoregressive KD 有价值，但不能直接推出在长程 agent trajectory、tool call、多轮环境反馈场景中的最佳做法。

GKD 需要频繁调用 teacher logits。对于大 vocab、大 batch、长输出，teacher forward 成本和 logits 存储/传输成本都很高。论文在 GSM8K 上报告 student sampling 相比固定数据采样带来约 1.8x、2x、2.2x 计算开销，实际大模型训练中还需要重新评估吞吐瓶颈。

论文默认 student 已经具备“adequate quality”的生成能力。若 student 初始能力太弱，on-policy samples 质量过低，teacher feedback 虽然可以覆盖错误状态，但训练可能浪费在大量无意义 prefixes 上。因此，GKD 更适合已有一定能力的 student，而不是完全从零训练。

最后，divergence 选择缺少统一准则。论文给出了 task-dependent 的经验结果，但没有提供可以跨任务预测最优 divergence 的理论或自动选择方法。

## 与 OPD2 的关系

[[sources/papers/2026-on-policy-delta-distillation|On-Policy Delta Distillation]] 讨论的是 reasoning model 后训练中的 token-level OPD reward。它认为传统 OPD 使用 teacher 与 student 的 log probability 差，会混入 teacher 在 base 阶段已有的语言 prior，因此提出 teacher 与 teacher-base 的 delta signal。

GKD 与 OPD2 的关系可以这样理解：

- GKD 定义了 on-policy distillation 的基础训练形态：student 生成，teacher 在 student prefixes 上给 token-level distribution signal；
- OPD2 继承了“在 student sampled tokens 上给 teacher signal”的方向，但把问题推进到 reasoning-tuned teacher 的能力增量如何提取；
- GKD 的重点是解决 train-inference distribution mismatch，OPD2 的重点是从 teacher distribution 中分离 reasoning tuning 带来的 delta。

因此，GKD 是 OPD 论文链条中的基础机制论文，OPD2 是面向 reasoning 能力蒸馏的信号改造论文。两者不是互相替代，而是处在不同层次：前者回答“为什么要 on-policy distillation”，后者回答“on-policy signal 应该蒸馏 teacher 的完整分布还是能力增量”。

## 相关知识链接

- [[training/post-training/on-policy-kd|On-policy KD]]
- [[training/post-training/knowledge-distillation|Knowledge Distillation]]
- [[training/post-training/logits-distillation|Logits Distillation]]
- [[training/post-training/sequence-level-distillation|Sequence-level Distillation]]
- [[sources/papers/2026-on-policy-delta-distillation|On-Policy Delta Distillation]]
