---
title: "MOPD: Multi-Teacher On-Policy Distillation for Capability Integration in LLM Post-Training"
created: 2026-08-26
published: 2026-08-26
modified: 2026-08-26
type: source
status: processed
source_type: paper
area: sources
tags:
  - source
  - paper
  - distillation
  - on-policy-kd
  - grpo
  - rl
  - capability-integration
aliases:
  - MOPD
  - Multi-Teacher On-Policy Distillation
source_url: https://arxiv.org/abs/2606.30406
paper_date: "2026-06"
paper_order: "30406"
---

# MOPD: Multi-Teacher On-Policy Distillation for Capability Integration in LLM Post-Training

## 基本信息

- 来源：arXiv:2606.30406v1
- 标题：MOPD: Multi-Teacher On-Policy Distillation for Capability Integration in LLM Post-Training
- 作者/机构：Wenhan Ma, Jianyu Wei, Liang Zhao, Hailin Zhang, Bangjun Xiao, Lei Li, Qibin Yang, Bofei Gao, Yudong Wang, Rang Li, Jinhao Dong, Zhifang Sui, Fuli Luo / Peking University, Xiaomi, HKU, RUC
- 日期：2026-06-29
- 链接：https://arxiv.org/abs/2606.30406
- 相关 topic：[[training/post-training/on-policy-kd|On-policy KD]]，[[training/post-training/grpo|GRPO]]，[[training/post-training/knowledge-distillation|Knowledge Distillation]]

这篇论文讨论的是 post-training 里一个很实际的问题：当 math、instruction following、software engineering、search、tool use 等不同能力各自已经通过专门 RL 训练出来后，如何把这些能力整合进同一个模型，而不是让它们在联合训练里互相打架。

MOPD 的答案是把“能力生产”和“能力整合”拆开。Stage 2 先在各自域上独立训练 domain teachers；Stage 3 再让 student 在自己的 rollouts 上，接受被路由到对应 domain teacher 的 token-level distillation。这样既保留了 on-policy 的分布对齐，又避免了多域联合 RL 的干扰。

## 研究问题

论文要回答的问题很直接：

1. 多个 domain-specific RL teachers 能否比 joint RL 更稳地整合到一个模型里？
2. 能否在不做 weight merge 的情况下，把各域能力合并到 policy space？
3. on-policy distillation 是否能同时保留 dense supervision、低 exposure bias 和并行化训练流程？
4. 在工业级多域模型上，这种路线是否真的比 Mix-RL、Cascade RL、Off-Policy Finetune 和 Param-Merge 更强？

## 核心主张

论文的核心主张可以概括为三点。

第一，能力整合不必发生在 weight space。把能力写进一个统一模型时，可以直接在 policy space 里完成路由式 distillation。

第二，student 应该在自己的 rollouts 上学习，而不是在 teacher 的静态 completions 上学习。这样既有 on-policy 分布，又有 token-level dense supervision。

第三，domain teachers 可以并行、独立地训练，互不耦合；最终只在 distillation 阶段对齐到一个 student。这使多域 post-training 的工程组织方式更清晰。

## 方法与机制

### 三阶段流程

MOPD 把 post-training 分成三步：

1. **General SFT**：先用覆盖面足够广的 SFT 数据得到共享初始化；
2. **Domain-specialized RL**：每个 domain 独立从同一个 SFT checkpoint 出发，训练自己的 RL teacher；
3. **MOPD distillation**：student 从同一个 SFT checkpoint 初始化，按 prompt domain 路由到对应 teacher，在自己的 rollout 上做 per-token distillation。

这种设计的关键不是“再训练一次”，而是把域内能力学习和域间能力整合解耦。

### Per-token reverse KL

Stage 3 的目标是 student 与被路由 teacher 在 student rollout 上做 per-token reverse KL：

$$
L_{\text{rev-KL}}
=
\mathbb{E}_{x,y\sim \pi_\theta}
\left[
\frac{1}{|y|}
\sum_t
\sum_v
\pi_\theta(v)\log\frac{\pi_\theta(v)}{\pi_{\phi_d}(v)}
\right]
$$

其中 `π_{φ_d}` 是与当前 prompt 域匹配的 teacher。

论文给了两种实现：

- **Policy-gradient form**：把 `log π_teacher - log π_student` 当作 token-level advantage，再做 advantage clipping；
- **Top-k distillation form**：只蒸馏 teacher 的 top-k tokens，并加校正项，避免截断后目标不再以 `π_theta = π_teacher` 为最优。

这两种写法本质上都保留了 on-policy distillation 的结构，只是一个更像 RL trainer 里的 advantage update，一个更像轻量 distillation loss。

### Asynchronous teacher prefill

MOPD 的工程设计也很重要。Teacher prefill 被拆成独立服务，student rollout 完成后异步请求 teacher 的 per-token logprob 或 top-k logits。这样 teacher 计算可以被 student sampling 覆盖，额外 wall-clock 开销几乎被隐藏。

这点很适合大规模 post-training：distillation 不需要阻塞训练主循环。

## 实验与证据

### Qwen3-30B-A3B 上的多域整合

论文在三个域上评估：

- Math：AIME25 / AIME26
- Instruction following：IFBench / IFEval
- Software engineering：SWE-bench Verified

对比方法包括：

- Mix-RL
- Cascade RL
- Off-Policy Finetune
- Param-Merge
- MOPD

结果上，MOPD 的 normalized score 为 **0.9373**，高于次优的 Mix-RL（0.8818），并且三域表现更均衡。它在 IF 上接近 teacher，在 Math 和 SWE 上也能闭合大部分 headroom。

### MiMo-V2-Flash

论文还把完整流程部署到工业规模模型 MiMo-V2-Flash，上到 Math、Code、IF、SWE、Tool Use 多个 benchmark。结果显示 MOPD 在多数 benchmark 上匹配或超过 teacher，说明这条路线不只是小规模实验技巧。

### 关键消融

论文的分析结果很稳定：

- policy-gradient 版和 top-k 版整体接近；
- same-origin teacher 很关键，也就是 teacher 与 student 最好来自同一个 SFT 初始化并经域内 RL 得到；
- 如果换成分布差异很大的外部强 teacher，训练会变不稳定，甚至 collapse；
- multi-round student-teacher evolution 还能继续挖出 headroom，说明一轮 MOPD 不一定吃满所有能力。

## 关键结论

MOPD 最值得沉淀的不是某个 benchmark 数字，而是一个训练组织原则：

```text
domain RL 负责把各自能力做强
MOPD 负责把这些能力整合进同一个 student
```

它把 multi-domain post-training 从“一个训练过程同时解决所有问题”改成“先并行生产，再统一整合”。这比 joint RL 更稳，也比 weight merge 更可控。

## 局限与疑问

- 这条路线依赖已经存在足够强的 domain teachers；
- prompt 路由必须足够准确，否则 teacher 选择会偏；
- same-origin teacher 对稳定性非常重要，说明 teacher-student distribution gap 仍然是核心问题；
- 论文主要验证的是 post-training capability integration，还没有覆盖更长程的 agent 轨迹整合。

## 相关知识链接

- [[training/post-training/on-policy-kd|On-policy KD]]
- [[training/post-training/grpo|GRPO]]
- [[training/post-training/knowledge-distillation|Knowledge Distillation]]
