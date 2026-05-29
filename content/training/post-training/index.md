---
title: Post-training
created: 2026-03-01
published: 2026-03-01
modified: 2026-05-29
type: topic
status: growing
area: training
tags:
  - training
  - post-training
---

Post-training 指 base model 完成大规模预训练之后，为了使其成为可交互、可控、符合任务与人类偏好的 assistant 而进行的一组训练与数据工程流程。它并不是单一算法，而是一条训练路线：先用 [[training/post-training/sft|SFT]] 和 [[training/post-training/instruction-tuning|Instruction Tuning]] 建立指令跟随与对话格式，再用 [[training/post-training/reward-model|Reward Model]]、[[training/post-training/rlhf|RLHF]]、[[training/post-training/dpo|DPO]]、[[training/post-training/grpo|GRPO]] 或可验证 reward 强化偏好与推理能力，最后常通过 [[training/post-training/knowledge-distillation|Knowledge Distillation]] 把大模型或强策略的能力迁移到更小、更便宜、更稳定的部署模型中。

后训练的核心问题可以概括为三类：

1. **格式与行为塑形**：模型不仅要会续写文本，还要理解 system / user / assistant / tool 等角色，遵守 [[training/post-training/chat-template|Chat Template]]，在合适位置回答、调用工具或拒答。
2. **偏好与目标对齐**：模型要在多个可能回答之间偏向更有帮助、更真实、更安全、更符合任务约束的回答，而这通常不能只靠 next-token maximum likelihood 完成。
3. **能力迁移与成本约束**：高质量后训练往往依赖昂贵 teacher、reward model、verifier、rollout 或人工偏好数据，因此需要拒绝采样、蒸馏、数据筛选和更稳定的优化目标来降低训练与部署成本。

## 学习路线

### 1. 从格式开始：SFT 与 Chat Template

[[training/post-training/sft|SFT]] 通常是 post-training 的第一个核心阶段。它把 base model 训练成能够按照 prompt-response 或 multi-turn conversation 数据回答的 assistant。这里最容易被低估的是 [[training/post-training/chat-template|Chat Template]]：训练时使用的角色标记、特殊 token、loss masking 和推理时模板必须一致，否则模型会学到一种格式，在部署时却被要求使用另一种格式。

### 2. 从数据到泛化：Instruction Tuning

[[training/post-training/instruction-tuning|Instruction Tuning]] 可以看成 SFT 在任务空间上的扩展。它不只训练“回答某个问题”，而是通过多任务、多格式、多领域指令数据，让模型学会把自然语言指令映射到任务行为。FLAN、T0、Self-Instruct 等路线都说明：指令数据的覆盖范围、任务多样性和响应质量，往往比单纯增大某一类数据更重要。

### 3. 从模仿到偏好：Reward Model 与 RLHF

SFT 本质上仍是 imitation learning：模型学习数据里出现的回答。偏好对齐进一步问：当同一个 prompt 有多个可行回答时，哪个回答更好？[[training/post-training/reward-model|Reward Model]] 用人类或 AI 的 preference data 学习一个打分函数，[[training/post-training/rlhf|RLHF]] 再用这个 reward 优化 policy。经典 InstructGPT 流程就是先 SFT，再训练 reward model，最后用 [[training/post-training/ppo|PPO]] 进行 policy optimization。

### 4. 从 RL 到直接偏好优化：DPO / GRPO

[[training/post-training/dpo|DPO]] 把 reward model 与 RL policy optimization 合并成一个更直接的 pairwise preference loss。它避免显式训练 reward model 和在线 rollout，工程上更简单，但依赖偏好数据质量、reference model 和 $\beta$ 所控制的 reference 约束尺度。

[[training/post-training/grpo|GRPO]] 则在 reasoning RL 中尤其常见。它对同一 prompt 采样一组 responses，用组内 reward 相对值估计 advantage，减少 value model 依赖。对于数学、代码、可验证答案等任务，GRPO 可以配合 verifier reward 强化可验证推理能力。

### 5. 从采样到数据再利用：Rejection Sampling

[[training/post-training/rejection-sampling|Rejection Sampling]] 不是单纯 inference trick，而是后训练数据生产的重要机制。模型可以对同一 prompt 采样多个候选，再由 reward model、verifier、规则或人工筛选出高质量样本，用于 SFT、DPO、RL warmup 或蒸馏。

### 6. 从大模型到可部署模型：Knowledge Distillation

[[training/post-training/knowledge-distillation|Knowledge Distillation]] 将 teacher model 的能力迁移到 student model。后训练中的蒸馏可以学习 teacher logits、完整 answers、reasoning traces、preference labels 或 on-policy feedback。它既服务于小模型部署，也服务于把昂贵 RL / verifier / long-CoT 产生的能力固化到更便宜的模型中。

## 方法图谱

| 方法 | 主要输入 | 训练信号 | 典型目标 | 主要风险 |
|---|---|---|---|---|
| [[training/post-training/sft|SFT]] | prompt-response / conversation | assistant token NLL | 指令跟随、格式对齐 | 过拟合风格、数据幻觉、模板错配 |
| [[training/post-training/instruction-tuning|Instruction Tuning]] | 多任务指令数据 | supervised loss | 泛化到新指令 | 任务混合偏斜、浅层格式模仿 |
| [[training/post-training/reward-model|Reward Model]] | chosen/rejected pairs | preference loss | 学习人类偏好 | reward hacking、标注偏差 |
| [[training/post-training/rlhf|RLHF]] | prompt、policy rollout、reward | RL objective + KL | 优化偏好回报 | 训练不稳、分布漂移、过优化 reward |
| [[training/post-training/dpo|DPO]] | preference pairs、reference model | pairwise logistic loss | 直接偏好优化 | beta 敏感、负样本质量依赖 |
| [[training/post-training/grpo|GRPO]] | group rollouts、reward/verifier | group-relative advantage | reasoning RL、可验证任务 | reward 稀疏、组内样本质量依赖 |
| [[training/post-training/rejection-sampling|Rejection Sampling]] | 多候选回答 | filter / ranking | 生成高质量训练数据 | 多样性下降、筛选器偏差 |
| [[training/post-training/knowledge-distillation|Knowledge Distillation]] | teacher outputs/logits/traces | KD / SFT / preference loss | 能力迁移、压缩部署 | teacher 错误继承、distribution mismatch |

## 经典论文路径

- [[sources/papers/2022-instructgpt|Training language models to follow instructions with human feedback]]：SFT + Reward Model + PPO 的经典 RLHF pipeline。
- [[sources/papers/2017-deep-rl-from-human-preferences|Deep Reinforcement Learning from Human Preferences]]：从人类偏好学习 reward 并进行 RL 的早期代表。
- [[sources/papers/2020-learning-to-summarize-from-human-feedback|Learning to summarize from human feedback]]：把 preference learning + RLHF 用于 summarization。
- [[sources/papers/2023-dpo|Direct Preference Optimization]]：直接从 pairwise preference 优化 policy。
- [[sources/papers/2017-ppo|Proximal Policy Optimization Algorithms]]：PPO 的原始算法论文。
- [[sources/papers/2021-flan|Finetuned Language Models Are Zero-Shot Learners]]：instruction tuning 的代表性多任务路线。
- [[sources/papers/2021-t0|Multitask Prompted Training Enables Zero-Shot Task Generalization]]：prompted multitask instruction tuning。
- [[sources/papers/2022-self-instruct|Self-Instruct]]：用模型自生成指令数据扩展 instruction tuning。
- [[sources/papers/2016-sequence-level-knowledge-distillation|Sequence-Level Knowledge Distillation]]：sequence-level KD 的经典起点。
- [[sources/papers/2023-distilling-step-by-step|Distilling Step-by-Step]]：通过 rationales / reasoning traces 辅助小模型学习。
- [[sources/papers/2024-deepseekmath|DeepSeekMath]]：GRPO 与数学推理 RL 的重要公开案例。

## Notes

- [[training/post-training/sft|SFT 监督微调]]
- [[training/post-training/instruction-tuning|Instruction Tuning]]
- [[training/post-training/chat-template|Chat Template]]
- [[training/post-training/reward-model|Reward Model]]
- [[training/post-training/rlhf|RLHF]]
- [[training/post-training/ppo|PPO]]
- [[training/post-training/dpo|DPO]]
- [[training/post-training/grpo|GRPO]]
- [[training/post-training/rejection-sampling|Rejection Sampling]]
- [[training/post-training/knowledge-distillation|Knowledge Distillation]]
- [[training/post-training/offline-kd|Offline KD]]
- [[training/post-training/on-policy-kd|On-policy KD]]
- [[training/post-training/logits-distillation|Logits Distillation]]
- [[training/post-training/sequence-level-distillation|Sequence-level Distillation]]
