---
title: Meta Llama 4 Multimodal Intelligence
created: 2026-05-28
published: 2026-05-28
modified: 2026-05-28
type: source
status: processed
area: sources
tags:
  - source
  - blog
  - llama
  - multimodal
  - moe
aliases:
  - Llama 4 Blog
  - The Llama 4 Herd
source_url: https://ai.meta.com/blog/llama-4-multimodal-intelligence
source_type: official-blog
publisher: Meta AI
---

## Source

- Title: [The Llama 4 herd: The beginning of a new era of natively multimodal AI innovation](https://ai.meta.com/blog/llama-4-multimodal-intelligence)
- Publisher: Meta AI
- Type: official blog / release note
- Topic: Llama 4 Scout, Llama 4 Maverick, and Llama 4 Behemoth

这是一篇 Meta 发布 Llama 4 系列时的官方博客。它的主要价值不是提供完整技术报告，而是给出 Llama 4 的产品定位、模型规模、关键架构变化和官方叙事：Llama 4 是 LLaMA 家族第一次把 open-weight、native multimodality、Mixture-of-Experts 和超长上下文明确放在同一代模型里的发布。

阅读这篇文章时要注意它的文体：它是 release blog，不是 paper。因此它会强调模型亮点、benchmark 对比、生态可用性和部署入口，但不会完整交代训练数据、模型超参数、视觉编码器、routing loss、后训练细节或评测复现条件。

## 文章的核心主张

这篇博客想传达的核心主张可以概括为三点：

1. Llama 4 是 LLaMA 家族的新阶段：从 text-only dense LLM 走向 natively multimodal MoE。
2. Scout 和 Maverick 分别服务不同部署/能力侧重点：Scout 强调可部署和超长上下文，Maverick 强调更强的通用多模态能力。
3. Behemoth 虽未发布，但作为 teacher model 支撑了 Scout / Maverick 的质量提升。

这意味着 Llama 4 不应该只被理解为 Llama 3.1 的参数放大版。它更像一次路线切换：模型家族开始把 sparse compute、多模态输入、long context 和 distillation teacher-student 关系组合起来。

## Llama 4 Scout

Scout 是这篇文章中最强调“效率”和“长上下文”的模型。Meta 给出的关键信息是：

- 17B active parameters。
- 16 experts。
- open-weight natively multimodal model。
- 支持 10M context window。
- 可以在单张 NVIDIA H100 GPU 上运行，文章中提到 Int4 quantization 场景。

Scout 的定位很清楚：它不是追求最大总参数量，而是试图在可部署成本和长上下文能力之间取得平衡。10M context 是文章中最醒目的卖点，Meta 举的应用场景包括多文档总结、理解大量用户活动记录、对大型代码库进行推理等。

但 10M context 需要谨慎理解。它表示模型和系统支持这样长度的上下文窗口，不等于任意 10M tokens 输入都能被模型可靠、均匀、无损地利用。长上下文效果还会受到信息位置、prompt 结构、训练分布、attention 机制、KV Cache 管理和评测方式影响。

## Llama 4 Maverick

Maverick 是更偏通用能力和多模态质量的模型。Meta 给出的关键信息是：

- 17B active parameters。
- 128 experts。
- 400B total parameters。
- MoE 层包含 128 routed experts 和 shared expert。
- 每个 token 会进入 shared expert，并被路由到一个 routed expert。
- 可以在单个 NVIDIA H100 DGX host 上运行，或通过分布式推理提升效率。

Maverick 最值得关注的是 active parameters 和 total parameters 的分离。它的 17B active parameters 描述的是每个 token 前向时实际参与计算的参数规模；400B total parameters 则描述模型整体容量和权重规模。

这个区别是 MoE 模型理解中的关键。如果只看 active parameters，会低估模型的总容量和部署存储压力；如果只看 total parameters，又会高估每个 token 的计算成本。Maverick 的官方描述很好地展示了 MoE 的核心 trade-off：用更大的总参数容量换取更低的每 token 激活计算量。

## Llama 4 Behemoth

Behemoth 在这篇博客里不是一个已经开放下载的模型，而是一个仍在训练中的 teacher model。Meta 给出的关键信息是：

- 288B active parameters。
- 16 experts。
- nearly 2T total parameters。
- multimodal MoE。
- 用于教授 / 蒸馏较小的 Llama 4 模型。

Behemoth 的作用不是简单展示“Meta 有一个更大的模型”，而是揭示 Llama 4 系列背后的 teacher-student 组织方式。文章提到 Maverick 从 Behemoth 中进行 codistillation，并使用动态权重的 distillation loss 来结合 soft targets 和 hard targets。

这说明 Scout / Maverick 的能力并不只来自自身架构和训练数据，也来自更大 teacher model 的能力迁移。对理解现代模型家族很重要：发布出来的模型往往只是最终可部署形态，背后可能还有更大的 teacher、数据筛选系统、评测系统和蒸馏流程。

## Native Multimodality

文章反复强调 Llama 4 是 natively multimodal。这个表述的重点是：Llama 4 不是在传统文本 LLM 外面简单外挂一个视觉模块，而是把文本和图像能力作为模型设计的一部分来发布。

不过，这篇博客没有完整展开 native multimodality 的训练细节。例如：

- 图像 token 如何进入模型；
- 视觉编码器或视觉前端如何设计；
- 图文数据比例如何安排；
- 多模态预训练和后训练如何衔接；
- 不同模态下的上下文长度如何计算。

因此，阅读时可以确认“Llama 4 是原生多模态路线”，但不应仅凭这篇博客推导具体实现细节。

## MoE 的意义

Llama 4 是 LLaMA 家族第一次明确采用 MoE 的一代。MoE 的直觉是让模型拥有很大的 total parameters，但每个 token 只激活其中一部分参数。这样可以提升模型容量，同时控制每 token 计算成本。

Maverick 的例子尤其清楚：400B total parameters 代表整体容量，17B active parameters 代表每 token 主要计算路径。Scout 则用 17B active parameters 和 16 experts 强调较低部署门槛。

但 MoE 也会带来额外问题：

- experts 仍然需要存储；
- routing 会影响训练稳定性和推理效率；
- expert load balancing 会影响吞吐；
- 分布式部署需要处理通信和 expert placement；
- active / total parameter 的对比容易被误读。

这篇博客没有深入讲这些系统问题，但从官方参数描述中已经能看出 Llama 4 的架构重心发生了变化。

## 这篇博客没有回答的问题

作为 release blog，它留下了很多值得追踪的问题：

- 是否会有完整 Llama 4 technical report？
- Scout / Maverick 的具体训练数据和过滤流程是什么？
- native multimodality 的模型结构细节是什么？
- MoE router、load balancing、expert parallelism 的训练细节是什么？
- Scout 的 10M context 在不同任务上的有效利用程度如何？
- Behemoth 后续是否会开放，或只长期作为 teacher model？

这些问题决定了后续能否把 Llama 4 从“发布信息”理解到“可复现的技术路线”。在没有完整报告前，写作应尽量区分官方确认事实和合理推测。

## 我的理解

Llama 4 的关键不在于某一个 benchmark，而在于 Meta 把 LLaMA 家族的技术路线从 dense text LLM 推向了更复杂的模型系统：MoE 负责扩大容量和控制计算，多模态负责输入范围扩展，long context 负责更长任务场景，Behemoth distillation 负责把更大 teacher 的能力压缩到可部署模型中。

这也意味着“LLaMA 架构”这个词需要更新。对于 LLaMA 1 到 Llama 3.1，说它是现代 dense decoder-only Transformer 路线基本合理；但到 Llama 4，这个说法已经不够了。更准确的理解是：LLaMA 是一个持续演进的开放权重模型家族，早期以 dense decoder-only 为主，Llama 4 开始进入 MoE、多模态和长上下文阶段。

## Related Notes

- [[architecture/model-families/llama|LLaMA]]
- [[architecture/sparse-and-efficient/moe|Mixture of Experts]]
- [[architecture/transformer/decoder-only-transformer|Decoder-only Transformer]]
- [[inference/kv-cache-and-memory/kv-cache|KV Cache]]
- [[training/post-training/knowledge-distillation|Knowledge Distillation]]
