---
title: LLaMA
created: 2026-02-07
published: 2026-02-07
modified: 2026-02-07
type: topic
status: growing
area: architecture
tags:
  - architecture
  - model-family
  - llama
  - meta
  - open-weight
aliases:
  - Llama
  - LLaMA Family
  - Meta Llama
---

## 问题背景

LLaMA / Llama 是 Meta 发布的一系列开放权重大语言模型。它的重要性不只在于模型效果，也在于它把 decoder-only Transformer、RMSNorm、SwiGLU、RoPE、GQA 等设计组合成了一条影响很大的开放模型路线。到 Llama 4，家族路线发生了明显扩展：从主要面向文本的 dense decoder-only LLM，进一步走向 Mixture-of-Experts、原生多模态和超长上下文。

在理解 LLaMA 时，不应该只把它看作“一个模型名字”，而应该把它看成一个持续演进的模型家族：

- LLaMA 1：强调用公开数据训练高效 foundation model。
- Llama 2：在更大数据、更长上下文和 chat/alignment 上推进，并在较大模型中引入 GQA。
- Llama 3：继续使用优化后的 decoder-only Transformer，引入更大 tokenizer vocabulary，并在 8B/70B 上使用 GQA。
- Llama 3.1：扩展到 8B/70B/405B，支持更长上下文和多语言能力，核心仍是 dense decoder-only Transformer 路线。
- Llama 4：引入开放权重的原生多模态 MoE 模型 Scout 和 Maverick，并预览作为 teacher model 的 Behemoth。

这条路线对后续开放模型生态影响很大：很多模型直接基于 LLaMA 架构、权重格式、tokenizer 生态或训练经验发展。Llama 4 之后，讨论 LLaMA 家族时还必须区分 dense LLM 路线、MoE 路线和 multimodal 路线，不能再把早期版本的结构直接泛化到所有 Llama 模型。

## 家族定位

LLaMA 系列属于 autoregressive decoder-only language model。它的基本训练目标是 next-token prediction：

$$
p(x_t \mid x_{<t})
$$

也就是说，模型根据前面的 token 预测下一个 token。这个目标和 [[training/pretraining/pretraining|Pretraining]]、[[training/pretraining/objective|Training Objective]]、[[fundamentals/information-theory/cross-entropy|Cross Entropy]] 直接相关。

从架构上看，LLaMA 1 到 Llama 3.1 主要是相对标准但经过现代化改造的 [[architecture/transformer/decoder-only-transformer|Decoder-only Transformer]]，通过 dense Transformer block 堆叠形成语言建模能力。Llama 4 仍然是 autoregressive 模型，但 Scout 和 Maverick 已经转向 sparse [[architecture/sparse-and-efficient/moe|Mixture-of-Experts]]，并加入原生多模态能力。因此，“LLaMA 是 dense decoder-only Transformer”只适合描述 LLaMA 1 到 Llama 3.1 的主线，不能覆盖整个最新家族。

## 核心架构

一个典型 LLaMA block 可以概括为：

```text
input hidden states
  -> RMSNorm
  -> Causal Self-Attention with RoPE
  -> Residual Add
  -> RMSNorm
  -> SwiGLU FFN
  -> Residual Add
output hidden states
```

这和现代 LLM 常见的 Pre-Norm decoder-only Transformer 一致。核心组件包括：

- [[architecture/attention/attention|Causal Self-Attention]]：自回归上下文建模。
- [[architecture/positional-encoding/rope|RoPE]]：向 Q/K 注入位置信息。
- RMSNorm：用 root mean square 做归一化，常用于 pre-normalization。
- SwiGLU：门控 FFN 激活结构。
- Residual connection：让深层 block 更容易训练。
- GQA：在部分版本中降低推理时 KV Cache 压力。

## Decoder-only 结构

LLaMA 使用 decoder-only Transformer，也就是每个位置只能关注自己和之前的 token。它通过 causal mask 保证模型不能看到未来 token。

这种结构适合统一多种任务：

```text
prompt -> continuation
instruction -> answer
context + question -> response
code prefix -> next code tokens
```

因此，LLaMA 的基础模型可以通过 prompt、SFT、RLHF/DPO 等方式变成 chat model、code model、instruction model 或 domain model。

Decoder-only 的优势是简单、可扩展、训练和推理形式统一；代价是所有任务都要被组织成自回归文本生成问题。

## RMSNorm

LLaMA 系列使用 RMSNorm 作为 normalization 方法，并采用 pre-normalization 结构。RMSNorm 和 LayerNorm 都用于稳定 hidden states，但 RMSNorm 不减去均值，只使用 root mean square 进行缩放。

给定向量 $x \in \mathbb{R}^d$，RMSNorm 可以写成：

$$
\mathrm{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2 + \epsilon}} \odot g
$$

其中：

- $d$ 是 hidden dimension。
- $\epsilon$ 是防止除零的小常数。
- $g$ 是可学习缩放参数。
- $\odot$ 表示逐元素乘法。

RMSNorm 的直觉是控制向量整体尺度，减少激活尺度漂移。相比 LayerNorm，它去掉均值中心化步骤，计算更简单。严谨地说，RMSNorm 不一定在所有场景都优于 LayerNorm，但在 LLaMA 等大模型实践中，它是稳定且高效的常见选择。

## SwiGLU FFN

LLaMA 的 FFN 使用 SwiGLU，而不是原始 Transformer 中的 ReLU FFN。SwiGLU 是一种门控前馈网络结构，可以写成近似形式：

$$
\mathrm{SwiGLU}(x) = \mathrm{SiLU}(xW_{gate}) \odot (xW_{up})
$$

然后再通过 down projection 回到 $d_{model}$：

$$
\mathrm{FFN}(x) = \left(\mathrm{SiLU}(xW_{gate}) \odot (xW_{up})\right)W_{down}
$$

其中：

- $W_{gate}$ 产生门控分支。
- $W_{up}$ 产生被门控的内容分支。
- $W_{down}$ 把中间维度投回 hidden size。

直观地说，SwiGLU 让 FFN 不只是“升维-激活-降维”，而是引入一个可学习的门控机制，决定哪些特征应该通过。它通常比普通 ReLU/GELU FFN 更强，但也有额外参数和计算代价。

## RoPE

LLaMA 使用 [[architecture/positional-encoding/rope|RoPE]] 作为位置编码。RoPE 不是把 position embedding 加到输入 token embedding 上，而是在 attention 的 query/key 上做位置相关旋转。

在 attention score 中：

$$
(R_m q_m)^T(R_n k_n) = q_m^T R_{n-m}k_n
$$

这个形式让 query-key 匹配自然包含相对位置信息。

LLaMA 采用 RoPE 的意义在于：它让位置信息和 attention 机制结合得更紧密，也为后续长上下文扩展提供了更方便的调整空间。不过，需要谨慎理解：RoPE 不是“无限上下文”的保证。上下文长度还受到训练长度、RoPE scaling、数据分布、KV Cache 和推理系统约束。

## GQA 与推理效率

LLaMA 早期版本主要使用 MHA；Llama 2 在较大模型中引入 GQA；Llama 3 的 8B 和 70B 也使用 GQA；Llama 3.1 系列也继续使用 GQA。

[[architecture/attention/grouped-query-attention|Grouped-Query Attention]] 的核心思想是：多个 query heads 共享一组 key/value heads。它介于 MHA 和 MQA 之间：

- MHA：每个 query head 有自己的 K/V，表达能力强，但 KV Cache 大。
- MQA：所有 query heads 共享一组 K/V，KV Cache 小，但表达能力可能受影响。
- GQA：多个 query heads 分组共享 K/V，在效果和推理成本之间折中。

对 LLaMA 这类自回归模型，GQA 的意义主要体现在推理阶段。生成时需要保存历史 token 的 K/V，形成 [[inference/kv-cache-and-memory/kv-cache|KV Cache]]。减少 K/V heads 可以降低显存占用和 memory bandwidth 压力，让大模型 serving 更可行。

## LLaMA 1 到 Llama 4 的演进

下面只记录公开资料中较稳定、适合知识库长期维护的高层演进，不把每个版本的所有 benchmark 当作架构知识。

| 版本 | 公开时间 | 主要规模 | 架构关键词 | 重要变化 |
|---|---:|---|---|---|
| LLaMA 1 | 2023 | 7B / 13B / 33B / 65B | decoder-only, RMSNorm, SwiGLU, RoPE | 强调公开数据和高效 foundation model |
| Llama 2 | 2023 | 7B / 13B / 34B / 70B | LLaMA 1 架构基础，较大模型使用 GQA | 上下文从 2K 扩到 4K，发布 chat 版本 |
| Llama 3 | 2024 | 8B / 70B | decoder-only, GQA, RoPE, RMSNorm, SwiGLU | 128K tokenizer vocabulary，8K context，更多预训练 token |
| Llama 3.1 | 2024 | 8B / 70B / 405B | dense decoder-only, GQA, long context | 128K context，多语言支持，引入 405B |
| Llama 4 Scout | 2025 | 17B active / 109B total, 16 experts | autoregressive MoE, native multimodal | 支持 10M context，面向单 H100 GPU + Int4 量化部署 |
| Llama 4 Maverick | 2025 | 17B active / 400B total, 128 experts | autoregressive MoE, native multimodal | 使用 128 routed experts + shared expert，面向更强通用和多模态能力 |
| Llama 4 Behemoth | 2025 preview | 288B active, nearly 2T total, 16 experts | multimodal MoE teacher model | 尚未正式发布，用于蒸馏 Scout / Maverick |

需要注意：不同版本的具体超参数、tokenizer、context length、训练数据和 license 都不同。写作时应避免把某一个版本的配置泛化成整个 LLaMA 家族的永久属性。尤其是 Llama 4 已经不是简单的 dense text-only LLM 延续，而是把 MoE、原生多模态和更激进的长上下文放进同一代模型中。

## Llama 4：MoE、原生多模态与超长上下文

Llama 4 是 LLaMA 家族中一次比较明显的架构转向。根据 Meta 官方发布页，Llama 4 Scout 和 Llama 4 Maverick 是开放权重的原生多模态 MoE 模型；Llama 4 Behemoth 则是仍在训练中的 teacher model，用于提升较小模型的质量。

### Sparse MoE

Llama 4 Maverick 的公开信息比较能说明 MoE 的设计思路：它有 17B active parameters 和 400B total parameters，MoE 层包含 128 个 routed experts 和 1 个 shared expert。每个 token 会进入 shared expert，同时被路由到 128 个 routed experts 中的一个。

这个结构的含义是：

- 所有参数仍然需要以某种形式存储在内存中。
- 单个 token 前向计算时只激活部分参数，因此 active parameters 远小于 total parameters。
- MoE 降低的是每个 token 的计算成本，不等于模型“只有 17B 参数”。
- serving 系统仍然要处理 expert routing、专家并行、显存放置和负载均衡等问题。

Meta 表示 Llama 4 Maverick 使用 alternating dense and MoE layers 来提升推理效率，并可以在单个 NVIDIA H100 DGX host 上运行，或通过分布式推理获得更高效率。Llama 4 Scout 则是 17B active / 109B total、16 experts，并强调可以在单 H100 GPU 上通过 Int4 quantization 部署。

### Native Multimodality

Llama 4 也把多模态能力放进了模型家族主线。Meta 称 Scout 和 Maverick 是 natively multimodal models，这意味着它们不是单纯在文本 LLM 外面附加一个视觉模块来完成演示，而是面向文本和图像输入进行联合建模。对知识库来说，最重要的是记录这个方向性变化：Llama 4 之后，LLaMA 家族不再只是 text-only LLM 家族，也进入了 vision-language model 的讨论范围。

不过，写作时需要谨慎区分：官方 blog 说明了 Llama 4 的原生多模态定位和 benchmark 结果，但并没有完整公开所有训练数据、模态混合比例、视觉编码器细节、过滤规则和训练超参数。不能把“原生多模态”进一步脑补成某个具体实现细节，除非有官方技术报告或模型卡支持。

### Long Context

Llama 4 Scout 的一个核心卖点是 10M context。Meta 官方说法是它把 Llama 3 的 128K supported context length 显著提升到 10 million tokens，用于多文档总结、长用户活动记录理解和大代码库推理等场景。

这里要特别注意两个边界：

- supported context length 不等于所有任务在该长度下都保持同等质量。
- 超长上下文能力不只来自位置编码，也依赖训练数据、后训练、推理系统、KV Cache 管理和注意力实现。

因此，Llama 4 Scout 的 10M context 是一个重要架构/系统信号，但不能简单理解为“把任意 10M token 塞进去都能可靠推理”。

### Behemoth 与蒸馏

Llama 4 Behemoth 是 Meta 预览的 teacher model，官方称其为 multimodal MoE，具有 288B active parameters、16 experts 和接近 2T total parameters。Meta 表示 Behemoth 仍在训练中，暂未发布，但已经用于对 Llama 4 Maverick 等较小模型进行 codistillation。

这说明 Llama 4 的模型家族不只是“多个尺寸直接训练出来”，而是包含 teacher-student / distillation 关系：更大的 teacher model 通过蒸馏损失函数向较小、可部署模型转移能力。对训练主题来说，这可以继续连接到 [[training/post-training/knowledge-distillation|Knowledge Distillation]]；对推理主题来说，它说明开放发布的 Scout / Maverick 是在能力与部署成本之间权衡后的结果。

## Tokenizer 与词表

LLaMA 1/2 和 Llama 3 的 tokenizer 路线有所变化。Llama 3 使用更大的 128K vocabulary，并强调更高 tokenization efficiency。更大的词表可能让同一文本被切成更少 token，从而影响上下文利用效率和训练/推理成本。

但 tokenizer 的影响要谨慎理解：

- 更大词表不一定总是更好。
- tokenization efficiency 会因语言、领域、代码和文本格式而变化。
- 词表变化会影响 embedding/output head 参数量。
- 不同 tokenizer 下的 perplexity 不宜直接比较。

这部分可以和 [[training/pretraining/tokenizer|Tokenizer]]、[[fundamentals/information-theory/perplexity|Perplexity]] 关联起来理解。

## LLaMA 为什么重要

LLaMA 系列的重要性主要体现在三个层面。

### 1. 架构范式清晰

LLaMA 并没有依赖非常复杂的模型结构，而是在 decoder-only Transformer 上采用一组经过验证的现代组件：RMSNorm、SwiGLU、RoPE、GQA。它让开源社区有了一个相对清晰、可复现、可扩展的 dense LLM 结构参考。

### 2. 生态影响大

大量开放模型、微调方法、部署框架和教学实现都围绕 LLaMA 展开。例如 Hugging Face Transformers、llama.cpp、vLLM、各种 LoRA/SFT/DPO 实验都经常以 LLaMA 系列或兼容架构为对象。Llama 4 进一步把这种影响扩展到开放权重多模态模型和 MoE serving 生态。

### 3. 连接训练与推理

LLaMA 的架构选择直接影响推理系统：RoPE 影响 position id 和长上下文，GQA 影响 KV Cache，SwiGLU 和 hidden size 影响 FFN 计算量，tokenizer 影响 token 数和上下文利用。这使它成为理解“模型结构如何影响系统部署”的好案例。

## 设计取舍

| 设计点 | 作用 | 取舍 |
|---|---|---|
| Decoder-only | 统一生成式训练和推理 | 所有任务都组织成自回归生成 |
| RMSNorm | 稳定尺度，减少 normalization 开销 | 不做均值中心化，性质不同于 LayerNorm |
| SwiGLU | 提升 FFN 表达能力 | 参数和计算成本更高 |
| RoPE | 让 attention 感知相对位置 | 长上下文仍需 scaling 和训练支持 |
| GQA | 降低 KV Cache 和推理成本 | K/V 表达能力相对 MHA 有折中 |
| MoE | 用更多 total parameters 提供容量，同时降低每 token active compute | 权重仍需存储，serving 需要处理 routing、expert placement 和负载均衡 |
| 原生多模态 | 让模型直接处理文本与图像输入 | 训练配方、模态比例和具体实现细节未必完全公开 |
| 大词表 tokenizer | 提升部分文本的 tokenization efficiency | 增加 embedding/output head 规模，跨 tokenizer 指标不可直接比较 |

## 常见误解

### 误解一：LLaMA 是一种全新 Transformer 架构

LLaMA 更准确地说是现代 decoder-only Transformer 的一个高影响力实现路线。它的重要性在于组件选择、训练规模、开放生态和工程可复用性，而不是完全发明了一种新架构。

### 误解二：所有 LLaMA 版本架构完全一样

不严谨。LLaMA 1、Llama 2、Llama 3、Llama 3.1 在 context length、GQA 使用范围、tokenizer、规模和训练数据上都有变化。到 Llama 4，Scout 和 Maverick 已经采用 MoE，并加入原生多模态能力。因此只能说早期版本共享 dense decoder-only Transformer 主线，不能说整个 LLaMA 家族永远是同一种架构。

### 误解三：GQA 只是无损压缩

GQA 是推理效率和表达能力之间的折中。它通常能显著降低 KV Cache 成本，但不能简单说完全没有代价。实际效果依赖模型规模、训练设置和任务。

### 误解四：RoPE 让 LLaMA 天然支持任意长度

RoPE 有利于位置建模和一定外推，但长上下文能力还依赖训练、scaling、数据和推理系统。Llama 3.1 的 128K context 不是单靠 RoPE 得到的。

### 误解五：开放权重等于完全开放训练细节

LLaMA 系列开放权重和部分技术材料，但并不意味着所有训练数据、过滤规则、超参数和对齐细节都完全可复现。以 Llama 4 为例，Meta 官方 blog 给出了 Scout、Maverick、Behemoth 的关键规模、MoE、多模态、上下文和蒸馏信息，但截至这篇笔记写作时，并没有等价于完整 technical report 的全部训练配方。写知识笔记时要区分公开事实、合理推断和未知细节。

### 误解六：MoE 模型的参数量只看 active parameters

不准确。active parameters 描述每个 token 前向时实际参与计算的参数规模，total parameters 描述模型整体容量和权重规模。Llama 4 Maverick 是 17B active / 400B total；Scout 是 17B active / 109B total。推理成本会受 active parameters 影响，但部署仍然要考虑 total parameters 的存储、通信和 expert routing。

## 与训练的关系

LLaMA base model 通过大规模 next-token prediction 形成基础能力。Llama 2、Llama 3 等版本还提供 instruction-tuned / chat variants，这些通常通过 [[training/post-training/sft|SFT]]、RLHF 或其他 post-training 方法增强指令跟随和安全性。

Llama 4 还凸显了 distillation 在模型家族构建中的作用。Meta 表示 Llama 4 Behemoth 作为 teacher model，用于对 Maverick 等模型进行 codistillation，并带来端到端评测质量提升。这说明模型能力不只来自单个模型的架构和训练数据，也可能来自更大 teacher model 的知识迁移。

架构本身决定了模型如何处理 token 序列，但模型能力很大程度来自数据规模、数据质量、训练 compute、tokenizer、训练目标、蒸馏和后训练策略。因此不能把 LLaMA 的效果只归因于 RMSNorm、SwiGLU、RoPE 或 MoE；它是架构、数据、训练和系统共同作用的结果。

## 与推理的关系

LLaMA 是部署生态中非常重要的模型家族。它的推理成本主要受以下因素影响：

- 参数规模：决定权重显存和矩阵乘法成本。
- 上下文长度：决定 prefill 成本和 KV Cache 大小。
- GQA：决定 K/V heads 数量，从而影响 KV Cache。
- MoE：决定每个 token 激活哪些 experts，也引入 expert routing 和并行调度问题。
- tokenizer：影响同一文本需要多少 token。
- RoPE / position id：影响长上下文和 prefix cache 的正确性。
- quantization：影响权重显存、吞吐和可能的精度损失。

因此，理解 LLaMA 不只是读模型卡参数，还要理解它背后的 [[inference/kv-cache-and-memory/kv-cache|KV Cache]]、[[inference/serving-systems/vllm|vLLM]]、quantization 和 batching 等推理系统问题。

## 参考资料

- Meta AI, [The Llama 4 herd: The beginning of a new era of natively multimodal AI innovation](https://ai.meta.com/blog/llama-4-multimodal-intelligence), 2025.

## 相关概念

- [[architecture/transformer/transformer|Transformer]] — LLaMA 的基础架构。
- [[architecture/transformer/decoder-only-transformer|Decoder-only Transformer]] — LLaMA 的核心模型形态。
- [[architecture/attention/attention|Attention]] — LLaMA block 的上下文交互机制。
- [[architecture/attention/grouped-query-attention|Grouped-Query Attention]] — Llama 2/3/3.1 中提升推理效率的重要机制。
- [[architecture/sparse-and-efficient/moe|Mixture-of-Experts]] — Llama 4 Scout / Maverick 的核心架构变化。
- [[architecture/positional-encoding/rope|RoPE]] — LLaMA 系列常用的位置编码机制。
- [[architecture/transformer/feed-forward|Feed Forward Network]] — SwiGLU FFN 的结构背景。
- [[training/pretraining/tokenizer|Tokenizer]] — 理解 Llama 3 词表变化和 token efficiency。
- [[training/post-training/sft|SFT]] — LLaMA chat/instruct 模型后训练的重要步骤。
- [[training/post-training/knowledge-distillation|Knowledge Distillation]] — Llama 4 Behemoth 向较小模型迁移能力的训练机制。
- [[inference/kv-cache-and-memory/kv-cache|KV Cache]] — LLaMA 自回归推理中的核心缓存。
- [[inference/serving-systems/vllm|vLLM]] — 常见 LLM serving 系统，可用于部署 LLaMA 类模型。
