---
title: DeepSeek V4 Technical Documentation
created: 2026-05-28
published: 2026-05-28
modified: 2026-05-28
type: source
status: processed
area: sources
tags:
  - source
  - report
  - deepseek
  - moe
  - long-context
  - agent
aliases:
  - DeepSeek V4 Report
  - DeepSeek V4 Model Card
source_url: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf
source_type: technical-documentation
publisher: DeepSeek AI
---

## Source

- Title: [DeepSeek V4 Technical Documentation](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)
- Model card mirror: [DeepSeek V4 Model Card](https://fe-static.deepseek.com/chat/transparency/deepseek-V4-model-card-EN.pdf)
- Release note: [DeepSeek V4 Preview Release](https://api-docs.deepseek.com/news/news260424)
- Publisher: DeepSeek AI
- Publication date: 2026-04-27
- Release date: 2026-04-24

这份资料是 DeepSeek V4 的官方 technical documentation / model card。它不像传统论文那样完整展开模型结构、公式、训练 ablation 和复现实验，而更接近一份面向发布、合规、使用和技术概览的官方文档。

阅读它时要把握两层信息：第一层是可以直接记录的事实，例如模型版本、参数规模、上下文长度、reasoning modes、license 和分发方式；第二层是需要继续追踪的技术点，例如 Hybrid Attention、mHC、Muon optimizer 的完整机制和实验依据。

## 文档的核心主张

DeepSeek V4 的核心主张是：在 DeepSeek-V3 的 MoE、DeepSeekMoE 和 Multi-Token Prediction 路线基础上，进一步统一长上下文、推理模式、agent 能力和更高效的注意力结构。

它不是单纯的参数放大。文档明确说 V4 保留 V3 的 DeepSeekMoE framework 和 MTP strategy，同时引入新的架构创新：

- Hybrid Attention Architecture。
- Manifold-Constrained Hyper-Connections。
- Muon optimizer。

这说明 V4 的叙事重点不是“我们把模型做得更大”，而是“我们让更大、更长上下文、更复杂任务的模型变得更可用”。

## Model Variants

DeepSeek V4 主要包含两个版本：DeepSeek-V4-Pro 和 DeepSeek-V4-Flash。

| Model | Total Parameters | Activated Parameters | 定位 |
|---|---:|---:|---|
| DeepSeek-V4-Pro | 1.6T | 49B / token | 更强能力版本，面向复杂推理、agentic coding 和高质量任务 |
| DeepSeek-V4-Flash | 约 285B | 13B / token | 更快、更经济的版本，适合成本敏感场景 |

这里最重要的是 total parameters 和 activated parameters 的区分。DeepSeek-V4-Pro 的 1.6T total parameters 说明模型整体容量非常大；49B activated parameters 表示每个 token 前向时实际激活的参数量远小于总参数量。这是 MoE 模型的典型特征。

V4-Flash 则是更轻量的选择。官方 release note 中写作 284B total / 13B active，model card 中写作 285B parameters / 13B activated。这里可以理解为不同文档四舍五入口径略有差异，实际写作时用“约 285B total / 13B active”更稳妥。

## Architecture Overview

DeepSeek V4 仍然是 Mixture-of-Experts language model series。它延续 DeepSeekMoE 和 MTP，但新增了一组围绕长上下文效率和深层训练稳定性的机制。

从文档看，V4 的架构关键词可以组织成三类：

1. Sparse capacity：DeepSeekMoE 继续负责用稀疏专家扩大模型容量，同时控制每 token active compute。
2. Long-context attention：Hybrid Attention 负责降低 1M context 下的 compute 和 memory 压力。
3. Training / signal propagation stability：mHC 和 Muon optimizer 负责改善深层网络训练与收敛稳定性。

这种组织方式有助于理解 V4：它不是一个孤立的新结构，而是在 V3 已经验证过的 MoE 路线之上，把瓶颈进一步推向长上下文、稳定训练和 agent 使用场景。

## Hybrid Attention Architecture

Hybrid Attention 是 V4 最值得重点阅读的架构变化。官方 model card 描述它由两部分组成：Compressed Sparse Attention 和 Heavily Compressed Attention。

### Compressed Sparse Attention

Compressed Sparse Attention，简称 CSA。文档给出的描述是：CSA 沿 sequence dimension 压缩 KV cache，并应用 DeepSeek Sparse Attention。

这句话包含两个动作：

- compress KV cache along the sequence dimension：减少长上下文中需要保留或处理的序列维度信息。
- apply DeepSeek Sparse Attention：不是对所有 token 做完整 dense attention，而是引入稀疏选择，降低长序列 attention 的计算和内存压力。

CSA 的直觉是：在 1M context 下，完整 dense attention 的成本太高，不能把每个 token 都当成同等重要的信息源。因此需要压缩和稀疏化，让模型在长序列里更有效地选择信息。

### Heavily Compressed Attention

Heavily Compressed Attention，简称 HCA。文档说 HCA applies heavier compression with dense attention。

这意味着 HCA 和 CSA 的侧重点不同：CSA 更强调 sparse attention，HCA 更强调强压缩后仍保留 dense attention 的信息交互方式。可以把它理解为长上下文中的另一条路径：不是完全依赖稀疏选择，而是在更压缩的表示上保留更密集的交互。

### 为什么叫 Hybrid

Hybrid 的意义在于组合两种注意力处理方式：一边用 sparse attention 控制长序列成本，一边用 heavier compression + dense attention 保留更全局的压缩信息。它试图避免两个极端：

- 完整 dense attention：信息充分，但 1M context 下成本过高。
- 过度 sparse attention：成本降低，但可能丢失关键全局信息。

这也是 V4 相比 V3 值得关注的地方。V3 的重点是 MLA，通过 latent compression 降低 KV Cache 成本；V4 进一步把长上下文效率写成 Hybrid Attention，说明 DeepSeek 在长上下文模型上继续沿着“压缩 + 稀疏 + 系统效率”的方向推进。

## Manifold-Constrained Hyper-Connections

mHC，全称 Manifold-Constrained Hyper-Connections，是文档中另一个重要但不宜过度展开的点。官方描述是：mHC 将 residual mapping 约束到 doubly stochastic matrices 的 manifold，也就是 Birkhoff polytope，以增强 signal propagation stability，同时保持 expressivity。

这段话可以拆开理解：

- residual mapping：它关注的是层与层之间的信息传递，不是 attention 或 MoE routing。
- doubly stochastic matrices：行和列都满足归一化约束的一类矩阵。
- Birkhoff polytope：所有 doubly stochastic matrices 构成的凸多面体。
- signal propagation stability：深层网络中信号向前传播和梯度向后传播时不应过快衰减、爆炸或混乱。

所以 mHC 的目标不是让模型“记住更多知识”，而是让深层结构的信息流更稳定。它更接近 residual connection / architecture stability 的问题。

但目前这份文档没有完整给出 mHC 的数学推导、ablation 或实现细节。因此在阅读笔记里应该保守：可以记录它是 V4 的结构创新，但不应把它写成已经完全理解、完全验证的机制。

## Muon Optimizer

DeepSeek V4 使用 Muon optimizer。官方 model card 的表述是：Muon optimizer 用于更快收敛和更稳定训练。

这说明 V4 的改进不只在模型结构，也包括训练优化。对于超大 MoE 模型，optimizer 的稳定性、收敛速度和硬件效率都会直接影响训练成本。

不过，model card 没有展开 Muon 的更新公式、和 AdamW 的系统比较、适用范围或失败模式。因此当前更适合把它记录为“V4 使用的训练优化组件”，而不是在没有原始论文和 ablation 的情况下展开过多理论解释。

## 1M Context

DeepSeek V4 支持 1M context。官方 release note 甚至说 1M context now becomes the default across all official DeepSeek services。

这点非常重要，因为它说明 V4 的长上下文不是一个附加实验能力，而是产品和 API 层面的默认定位。它也解释了为什么 V4 要强调 Hybrid Attention：如果官方服务默认支持 1M context，就必须控制长序列下的计算和内存成本。

但 1M context 仍然不能被简单理解为“模型能完美理解 1M token”。长上下文能力至少涉及：

- 模型是否能保留远距离信息；
- 关键信息在上下文中的位置；
- prompt 是否有结构；
- attention 和 KV cache 如何压缩；
- 推理系统是否能稳定承载；
- benchmark 是否真的测试了跨长距离推理，而不是只测试检索。

因此，1M context 是强信号，但不是效果保证。

## Reasoning Modes

Model card 列出三种 reasoning modes：

- Non-think：快速、直觉式回答。
- Think High：审慎的逻辑分析。
- Think Max：最大容量的扩展推理。

API release note 中也写到 V4-Pro 和 V4-Flash 支持 Thinking / Non-Thinking modes，并且旧的 `deepseek-chat` 和 `deepseek-reasoner` 后续会退役，目前会分别路由到 V4-Flash 的 non-thinking / thinking。

这说明 DeepSeek 正在把“普通聊天模型”和“推理模型”的区分转化为同一模型家族下的使用模式。用户不一定需要选择两个完全不同的模型，而是通过 mode 控制推理深度、延迟和成本。

这对应用开发很重要：很多任务不需要 Think Max。简单问答、格式转换、短摘要可能 Non-think 就够了；数学证明、复杂代码修改、多步工具调用才需要更深推理。

## Agent and Tool Use

V4 release note 很强调 agent capabilities，尤其是 agentic coding。它提到 V4 已经集成到 Claude Code、OpenClaw、OpenCode 等 agent 场景，也用于 DeepSeek 内部 agentic coding。

这部分反映了模型使用场景的变化。V4 不只是要回答问题，还要能处理更长任务链：

- 读取大量上下文；
- 理解项目或文档结构；
- 调用工具；
- 写代码和修改文件；
- 多轮规划与纠错；
- 在长上下文中保持任务状态。

这和 1M context、thinking modes、tool calls、Anthropic/OpenAI-compatible API 都是相关的。也就是说，V4 的产品定位已经从“单轮聊天模型”明显转向“agent substrate”。

## Distribution and License

Model card 说明 DeepSeek V4 通过 open-source repositories 和 API 两种渠道分发。Hugging Face 上的模型权重和代码采用 MIT License。

这点对开放模型生态很关键：V4-Pro 和 V4-Flash 不只是 API 模型，也提供 open weights。区别在于本地部署成本很高，尤其 V4-Pro 的权重规模巨大，因此实际使用中 API、推理服务商、量化版本和分布式 serving 会很重要。

## 这份文档没有充分回答的问题

这份 technical documentation / model card 已经给出很多关键信息，但还不是完整技术论文。它没有充分展开：

- Hybrid Attention 的完整公式和实现细节；
- CSA / HCA 在不同上下文长度下的 ablation；
- mHC 与普通 residual connection、Hyper-Connections 的对比实验；
- Muon optimizer 的具体更新规则和大规模训练稳定性证据；
- V4 的完整训练数据组成、过滤规则和后训练数据；
- reasoning modes 在训练和推理系统中如何实现；
- agentic coding benchmark 的详细评测设置和可复现条件。

这些问题决定了后续能否从“知道 V4 有哪些组件”进一步走向“理解 V4 为什么这样设计”。

## 我的理解

DeepSeek V4 的核心不是单个新模块，而是一次系统整合：DeepSeekMoE 继续提供稀疏容量，Hybrid Attention 服务 1M 长上下文，mHC 和 Muon optimizer 服务深层训练稳定性，thinking modes 把 R1 的推理能力产品化，agent 优化则把模型推向更复杂的真实任务。

如果说 DeepSeek-V3 的关键词是“高性价比 MoE 基础模型”，DeepSeek-R1 的关键词是“用 RL 激发推理能力”，那么 DeepSeek-V4 的关键词更接近“统一模型接口下的长上下文 reasoning agent”。

不过，因为当前资料仍偏 model card / release note，很多机制只能先建立概念框架，不能写成最终定论。后续如果有 arXiv 技术报告、源码说明或更细的 ablation，需要重新补读。

## Related Notes

- [[architecture/model-families/deepseek|DeepSeek]]
- [[architecture/sparse-and-efficient/moe|Mixture of Experts]]
- [[architecture/attention/attention|Attention]]
- [[inference/kv-cache-and-memory/kv-cache|KV Cache]]
- [[application/agents/|Agents]]
- [[application/tool-use/|Tool Use]]
