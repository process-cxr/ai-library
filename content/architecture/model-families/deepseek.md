---
title: DeepSeek
created: 2026-02-08
published: 2026-02-08
modified: 2026-02-08
type: topic
status: growing
area: architecture
tags:
  - model-family
  - deepseek
  - moe
  - reasoning-model
aliases:
  - DeepSeek Family
  - DeepSeek Models
---

## 问题背景

DeepSeek 是近几年开放模型生态中非常重要的一条路线。它的重要性不只来自模型效果，也来自它把高性价比训练、MoE 架构、高效 attention、强化学习推理能力和开放技术报告结合在一起。

理解 DeepSeek 时，最好不要只把它看成“某个聊天模型”或“某个推理模型”。它更像一组互相连接的模型与训练方法：

- DeepSeek-V2 / V2.5：验证 Multi-Head Latent Attention 和 DeepSeekMoE 的高效路线。
- DeepSeek-V3：大规模 MoE base / chat model，强调 671B total parameters、37B activated parameters、MLA、DeepSeekMoE、auxiliary-loss-free load balancing 和 multi-token prediction。
- DeepSeek-R1-Zero：从 base model 出发，用强化学习直接激发 reasoning 行为。
- DeepSeek-R1：结合 cold-start data、RL、SFT 和蒸馏，形成更可用的推理模型。
- Distilled R1 models：把大模型推理能力迁移到较小 dense models 上，降低使用门槛。
- DeepSeek-V4：在 V3 的 DeepSeekMoE 和 MTP 路线之上，引入 Hybrid Attention、mHC、Muon optimizer、1M context 和更统一的 thinking / non-thinking 使用方式。

DeepSeek 对知识库的价值在于，它同时连接 [[architecture/sparse-and-efficient/moe|Mixture of Experts]]、高效推理、长上下文、[[training/post-training/rlhf|Reinforcement Learning]]、[[training/post-training/knowledge-distillation|Knowledge Distillation]]、reasoning model 和 agentic coding。

## 家族定位

DeepSeek-V3 是 DeepSeek 家族中的通用 MoE language model。根据技术报告，它有 671B total parameters，但每个 token 激活 37B parameters。它使用 MLA 提升推理效率，使用 DeepSeekMoE 降低训练成本，并在 14.8T tokens 上预训练，再经过 SFT 和 RL 阶段增强能力。

DeepSeek-R1 则是 reasoning model 路线。它关注的问题不是单纯扩大模型，而是如何通过强化学习激发模型的长链推理、自我反思、验证和策略调整能力。R1 系列的重点从“模型能不能生成答案”推进到“模型能不能在生成过程中进行更长、更可验证的推理”。

DeepSeek-V4 进一步尝试把高效基础模型、长上下文、推理模式和 agent 能力统一起来。它仍然是 MoE language model series，但新增 Hybrid Attention、Manifold-Constrained Hyper-Connections 和 Muon optimizer，并把 1M context 作为官方服务的标准能力。

因此，DeepSeek 家族可以粗略分成三条互相连接的主线：

| 主线 | 代表模型 | 核心问题 |
|---|---|---|
| 高效基础模型 | DeepSeek-V2, DeepSeek-V3 | 如何用 MoE、MLA 和工程优化训练高性价比大模型 |
| 推理模型 | DeepSeek-R1-Zero, DeepSeek-R1 | 如何用 RL 激发 reasoning 能力，并通过蒸馏降低部署成本 |
| 统一长上下文与 Agent 模型 | DeepSeek-V4-Flash, DeepSeek-V4-Pro | 如何把 MoE、长上下文、thinking modes 和 agent/tool-use 能力整合到同一代模型中 |

## DeepSeek-V3 的架构重点

DeepSeek-V3 的技术报告把模型架构概括为三个关键点：MLA、DeepSeekMoE 和 multi-token prediction。

### Multi-Head Latent Attention

[[architecture/attention/multi-head-latent-attention|Multi-Head Latent Attention]]，简称 MLA，是 DeepSeek 用来提升推理效率的 attention 设计。它的核心动机和 [[inference/kv-cache-and-memory/kv-cache|KV Cache]] 有关：自回归推理时，模型需要为历史 token 保存 K/V states；上下文越长、batch 越大，KV Cache 越容易成为显存和带宽瓶颈。

MLA 试图通过 latent representation 压缩 attention 中需要缓存和读取的信息，从而降低推理成本。和 [[architecture/attention/grouped-query-attention|Grouped-Query Attention]] 类似，它也是围绕 KV Cache 成本进行设计，但具体机制不同：GQA 通过多个 query heads 共享较少的 K/V heads 来降低缓存；MLA 则通过 latent compression 改变 K/V 表示方式。

在这篇家族页里，先把 MLA 理解为 DeepSeek 系列的高效 attention 路线即可；更完整的公式和实现细节适合单独写成 `Multi-Head Latent Attention` 笔记。

### DeepSeekMoE

DeepSeek-V3 的 FFN 部分采用 DeepSeekMoE。相比传统 MoE，DeepSeekMoE 强调 finer-grained experts，并区分 shared experts 和 routed experts。

- Shared experts：所有 token 都会使用，提供通用能力。
- Routed experts：由 router 针对 token 选择，提供稀疏容量。
- Finer-grained experts：把 expert 切得更细，使 routing 更灵活。

这个设计和 [[architecture/sparse-and-efficient/moe|Mixture of Experts]] 中的基本思想一致：模型拥有很大的 total parameters，但每个 token 只激活其中一部分参数。DeepSeek-V3 的公开数字是 671B total parameters / 37B activated parameters。理解这个数字时要注意：37B activated parameters 更接近每 token 计算路径，671B total parameters 仍然影响容量、存储和部署复杂度。

### Auxiliary-loss-free load balancing

MoE 的一个核心难点是负载均衡。如果 router 总把 token 发给少数 experts，会造成 expert overload、训练不稳定和容量浪费。传统方法常加入 auxiliary loss 来鼓励负载均衡，但这个 loss 如果过强，可能干扰模型按内容选择 expert。

DeepSeek-V3 技术报告强调 auxiliary-loss-free load balancing，目标是在减少性能损害的同时保持 expert load 平衡。这个点很关键，因为它说明 DeepSeek 的 MoE 路线不只是“堆更多 experts”，而是在解决 MoE 训练中的 routing collapse 和 load balance trade-off。

### Multi-Token Prediction

DeepSeek-V3 还使用 multi-token prediction training objective。标准语言模型通常预测下一个 token，而 MTP 让模型在训练中预测多个未来 token。它的目标是增强模型的训练信号，提高表示学习和生成能力。

需要谨慎理解：MTP 不是把自回归生成改成一次性生成多个 token。推理时模型仍然可以按自回归方式生成；MTP 更多是训练目标上的增强。

## DeepSeek-V4 的统一路线

DeepSeek-V4 是 V3 / R1 之后的重要更新。根据 DeepSeek 官方技术文档和 release note，V4 包含两个主要版本：

| 模型 | 总参数量 | 每 token 激活参数量 | 定位 |
|---|---:|---:|---|
| DeepSeek-V4-Pro | 1.6T | 49B | 更强能力版本，面向复杂推理、agentic coding 和高质量任务 |
| DeepSeek-V4-Flash | 约 285B | 13B | 更快、更经济的版本，适合成本敏感场景 |

DeepSeek-V4 仍然保留 DeepSeekMoE framework 和 Multi-Token Prediction strategy，这说明它不是抛弃 V3，而是在 V3 的 MoE + MTP 基础上继续演进。它的新增重点主要在 Hybrid Attention、mHC、Muon optimizer、1M context 和 thinking modes。

### Hybrid Attention

V4 引入 [[architecture/attention/hybrid-attention|Hybrid Attention]] Architecture，用于提升长上下文效率。官方 model card 将它描述为 Compressed Sparse Attention 和 Heavily Compressed Attention 的组合：

- Compressed Sparse Attention，简称 CSA：沿 sequence dimension 压缩 KV cache，并结合 DeepSeek Sparse Attention。
- Heavily Compressed Attention，简称 HCA：使用更强压缩，同时保留 dense attention。

这个设计的目标是降低 1M context 下的 compute 和 memory 成本。它和 V3 的 MLA 有连续性：二者都围绕长上下文和 KV Cache 成本展开；但 V4 的重点从 latent compression 进一步转向 hybrid attention，把 sparse attention 和更重的压缩结合起来。

### Manifold-Constrained Hyper-Connections

V4 还引入 Manifold-Constrained Hyper-Connections，简称 mHC。官方说明中，mHC 将 residual mapping 约束到 doubly stochastic matrices 的流形，也就是 Birkhoff polytope，以增强 signal propagation stability，同时保持模型表达能力。

在当前知识库里，mHC 可以先理解为 V4 的连接结构创新。它关注的不是“token 之间如何 attention”，也不是“token 路由到哪个 expert”，而是深层网络中 residual / hidden-state propagation 如何更稳定。由于公开资料对数学细节展开有限，暂时不应把 mHC 写成过度确定的完整理论。

### Muon Optimizer

DeepSeek-V4 使用 Muon optimizer，官方定位是更快收敛和更稳定训练。这里也需要保守处理：如果没有更完整的 technical report 或 optimizer 论文支撑，不应在 DeepSeek 家族页里展开具体更新公式，只需记录它是 V4 训练稳定性相关的关键组件。

### 1M context 与 thinking modes

DeepSeek-V4 支持 1M context，官方 release note 还强调 1M context 已成为所有官方 DeepSeek 服务的默认标准。这和 V4 的 Hybrid Attention 直接相关：长上下文不只是模型参数问题，也依赖 attention 结构、KV Cache 压缩、推理系统和服务端缓存策略。

V4 还提供不同 reasoning modes。Model card 中列出：

- Non-think：快速、直觉式回答。
- Think High：更审慎的逻辑分析。
- Think Max：最大容量的扩展推理。

API release note 中也使用 Thinking / Non-Thinking modes 的说法。这说明 DeepSeek 正在把 V3 的通用模型路线和 R1 的推理模型路线统一为同一个产品化接口：用户不一定再需要显式区分 `deepseek-chat` 和 `deepseek-reasoner`，而是通过 mode 控制推理深度。

### Agent 与 Tool Use

DeepSeek-V4 明确强调 agentic capabilities，尤其是 agentic coding、tool-use 和长上下文任务。官方 release note 提到 V4 已集成到 Claude Code、OpenClaw、OpenCode 等 agent 场景，并用于 DeepSeek 内部 agentic coding。

这说明 V4 的定位不只是“更强聊天模型”，而是面向更长任务链条：读取大量上下文、调用工具、修改代码、生成文档、完成多步任务。对知识库来说，这会连接到 [[application/agents/|Agents]]、[[application/tool-use/|Tool Use]] 和 [[inference/kv-cache-and-memory/kv-cache|KV Cache]] 等主题。

## DeepSeek-R1 的推理路线

DeepSeek-R1 的核心问题是：能否通过 reinforcement learning 激发模型的 reasoning capability，而不是完全依赖大量人工标注的 reasoning traces。

根据 DeepSeek-R1 论文摘要，RL 训练可以促使模型出现一些推理模式，例如：

- self-reflection：模型在推理中反思已有步骤。
- verification：模型检查中间结果或答案合理性。
- dynamic strategy adaptation：模型在推理中调整解题策略。

这让 DeepSeek-R1 成为 reasoning model 讨论中的重要节点。它表明模型推理能力不只是来自更多 SFT 数据，也可以通过可验证任务上的 RL 信号被激发。

## R1-Zero 与 R1

DeepSeek-R1-Zero 和 DeepSeek-R1 的区别很重要。

R1-Zero 更强调“从 base model 直接进行 RL”，用于观察 reasoning 行为是否能在没有人工标注推理轨迹的情况下涌现。这个方向有研究价值，但纯 RL 过程也可能带来可读性、语言混杂、输出格式不稳定等问题。

DeepSeek-R1 则更像工程上更可用的版本：它在推理能力之外，还需要考虑输出可读性、指令跟随、格式稳定和蒸馏到较小模型。也就是说，R1-Zero 更像验证“RL 能不能激发推理”，R1 更像把这种能力整理成可发布、可使用的模型。

## 蒸馏的意义

DeepSeek-R1 的另一个重要影响是推动了 reasoning distillation。大推理模型在数学、代码和复杂推理上能力强，但推理成本高、延迟长、部署门槛高。蒸馏的目标是把大模型产生的 reasoning traces 或高质量答案迁移到较小模型上。

这和 [[training/post-training/knowledge-distillation|Knowledge Distillation]] 直接相关：student model 不一定复现 teacher 的全部能力，但可以通过学习 teacher 的输出分布、推理轨迹或答案风格，获得更强的推理表现。

需要注意：蒸馏出来的 reasoning model 并不等同于 teacher model。它可能在某些 benchmark 上表现很好，但在 out-of-distribution 问题、长推理稳定性和自我纠错能力上仍然受限。

## DeepSeek 为什么重要

### 1. 把高效架构推到前台

DeepSeek-V3 不是单纯依赖 dense scaling，而是强调 MLA、DeepSeekMoE、FP8 training、MTP 和工程优化。DeepSeek-V4 又继续强调 Hybrid Attention、mHC、Muon optimizer 和 1M context。它让开放模型社区看到：模型能力不只来自更大的 dense 参数，也来自架构和系统效率。

### 2. 重新强化 RL 在 reasoning 中的地位

R1 系列让强化学习重新成为 reasoning model 的核心议题。它关注的是如何通过可验证奖励训练模型产生更长、更有结构的思考过程，而不仅仅是模仿人工标注答案。

### 3. 推动开放模型竞争

DeepSeek 的技术报告、模型权重和蒸馏模型推动了社区复现、分析和二次开发。它也促使很多知识点变得更重要：MoE serving、KV Cache 优化、RL training、reasoning trace、distillation、long-context evaluation 和 agent evaluation。

### 4. 把 reasoning 与 agent 接口统一

DeepSeek-V4 的 thinking / non-thinking modes 说明 reasoning capability 正在从“单独的 reasoner 模型”变成一个可控制的模型使用模式。对应用层来说，这比单纯发布一个更强模型更重要：系统可以根据任务成本、延迟和复杂度选择不同推理深度。

## 设计取舍

| 设计点 | 作用 | 取舍 |
|---|---|---|
| MLA | 降低 attention 推理中的 KV Cache 压力 | 机制更复杂，需要专门理解和实现 |
| DeepSeekMoE | 用 sparse experts 扩大容量并降低每 token 计算 | routing、负载均衡和部署复杂度上升 |
| Auxiliary-loss-free balancing | 减少负载均衡 loss 对性能的干扰 | 需要更精细的 routing 控制策略 |
| Multi-token prediction | 增强训练信号 | 不等于推理时一次生成多个 token |
| Hybrid Attention | 降低 1M context 下的 compute 和 memory 成本 | CSA/HCA 机制更复杂，需要结合推理系统理解 |
| mHC | 改善深层网络信号传播稳定性 | 公开资料有限，不宜过度推导数学细节 |
| Muon optimizer | 提升收敛速度和训练稳定性 | 需要结合更完整优化器资料理解 |
| RL for reasoning | 激发自我反思、验证和策略调整 | 训练稳定性、奖励设计和可读性更难 |
| Distillation | 把大推理模型能力迁移到小模型 | student 不一定保留 teacher 的全部泛化能力 |

## 常见误解

### 误解一：DeepSeek 只是一个 MoE 模型

不完整。MoE 是 DeepSeek-V3/V4 的重要架构，但 DeepSeek 家族还包括 MLA、Hybrid Attention、MTP、FP8 训练、mHC、Muon optimizer、post-training、R1/RL 推理路线和蒸馏模型。只看 MoE 会低估它的系统性。

### 误解二：R1 的推理能力完全来自人类写好的 CoT

不准确。R1 系列的关键贡献之一就是展示 RL 可以在可验证任务中激发 reasoning patterns。虽然最终可用模型会结合多阶段训练和数据整理，但不能把它简单理解成“模仿人工 CoT”。

### 误解三：MoE 的 37B activated parameters 等于模型只有 37B

不对。DeepSeek-V3 是 671B total / 37B activated。每 token 计算路径接近 activated parameters，但容量、存储和部署都仍然受到 total parameters 影响。

### 误解四：推理模型总是更适合所有任务

不一定。Reasoning model 适合数学、代码、复杂规划和需要多步验证的任务，但在简单问答、低延迟交互和格式固定任务上，长推理可能增加成本和延迟，甚至引入不必要的复杂性。

### 误解五：1M context 等于所有 1M token 任务都可靠

不严谨。DeepSeek-V4 支持 1M context，这是重要的长上下文能力，但真实任务效果仍然取决于检索位置、信息密度、推理系统、prompt 结构、缓存策略和评测方式。长上下文能力应该被验证，而不是只看最大长度数字。

## 参考资料

- DeepSeek-AI, [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437), arXiv:2412.19437.
- DeepSeek-AI, [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948), arXiv:2501.12948.
- DeepSeek-AI, [DeepSeek V4 Technical Documentation](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf), 2026.
- DeepSeek-AI, [DeepSeek V4 Preview Release](https://api-docs.deepseek.com/news/news260424), 2026.

## 相关概念

- [[architecture/sparse-and-efficient/moe|Mixture of Experts]] — DeepSeekMoE 的基础架构背景。
- [[architecture/attention/attention|Attention]] — MLA 属于 attention efficiency 的一条路线。
- [[architecture/attention/multi-head-latent-attention|Multi-Head Latent Attention]] — DeepSeek-V2/V3 的高效 attention 设计。
- [[architecture/attention/hybrid-attention|Hybrid Attention]] — DeepSeek-V4 面向 1M context 的 attention 设计。
- [[architecture/attention/grouped-query-attention|Grouped-Query Attention]] — 另一种降低 KV Cache 成本的 attention 设计，可与 MLA 对比。
- [[inference/kv-cache-and-memory/kv-cache|KV Cache]] — MLA 和 V4 Hybrid Attention 的重要动机之一。
- [[application/agents/|Agents]] — DeepSeek-V4 强调的 agentic coding 和多步任务场景。
- [[application/tool-use/|Tool Use]] — V4 API 和 agent 能力相关的应用层主题。
- [[training/post-training/rlhf|Reinforcement Learning]] — DeepSeek-R1 的核心训练方法。
- [[training/post-training/knowledge-distillation|Knowledge Distillation]] — R1 推理能力迁移到小模型的重要方法。
- [[training/post-training/sft|SFT]] — R1 可用性增强中的重要阶段。
