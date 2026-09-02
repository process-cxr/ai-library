---
title: Mixture of Experts
created: 2026-02-14
published: 2026-02-14
modified: 2026-08-31
type: topic
status: mature
area: architecture
tags:
  - architecture
  - scaling
  - moe
  - sparse-model
aliases:
  - MoE
  - Mixture-of-Experts
---

## 核心问题

Mixture of Experts，通常简称 MoE，是一种用“稀疏激活”扩大模型容量的架构。它想解决的问题是：如果 dense model 每一层、每个 token 都要经过全部参数，那么模型变大时训练和推理成本都会快速上升；有没有办法让模型拥有很大的总参数量，但每个 token 只使用其中一小部分参数？

MoE 的基本回答是：把某些层，通常是 Transformer block 里的 FFN，拆成多个 expert；每个 token 经过 router 选择少数几个 expert 参与计算。这样模型可以有很多 total parameters，但每个 token 的 active parameters 远小于总参数量。

直观地说，dense FFN 像一间所有人都要经过的“大办公室”；MoE FFN 像一个专家团队，每个 token 会被分配给最相关的几个专家处理。

## 基本结构

在标准 [[architecture/transformer/transformer|Transformer]] 中，一个 block 通常包含 attention 和 FFN：

```text
hidden states
  -> self-attention
  -> FFN
```

MoE 通常替换的是 FFN 部分：

```text
hidden states
  -> self-attention
  -> router
  -> selected experts
  -> weighted combine
```

一个简化的 MoE 层可以写成：

$$
y = \sum_{i \in \mathrm{TopK}(r(x))} g_i(x) E_i(x)
$$

其中：

- $x$ 是当前 token 的 hidden state。
- $E_i$ 是第 $i$ 个 expert，通常是一个 FFN。
- $r(x)$ 是 router，给每个 expert 打分。
- $\mathrm{TopK}$ 表示只选择得分最高的 $K$ 个 expert。
- $g_i(x)$ 是 router 给选中 expert 的权重。
- $y$ 是多个 expert 输出加权后的结果。

如果 $K=1$，每个 token 只进入一个 routed expert；如果 $K=2$，每个 token 进入两个 routed experts。不同模型会选择不同的 routing 策略。

## Active Parameters 与 Total Parameters

理解 MoE 时最容易混淆的是 active parameters 和 total parameters。

- `total parameters`：模型里所有 expert 和其他模块的总参数量。
- `active parameters`：一个 token 前向计算时实际参与计算的参数量。

例如 [[architecture/model-families/llama|Llama 4]] Maverick 被描述为 17B active parameters / 400B total parameters。这并不表示模型“只有 17B 参数”，也不表示部署时完全不需要考虑 400B 参数。它表示单个 token 的主要计算路径只激活其中一部分参数，但系统仍然要存储、调度和加载整个 expert 集合。

这个区别非常重要：

- 从计算量看，active parameters 更接近每 token 的前向成本。
- 从显存和部署看，total parameters、expert placement、并行策略仍然很关键。
- 从模型容量看，total parameters 代表模型可以容纳更多知识和模式，但能否有效利用取决于 routing 和训练。

## Router 做了什么

Router 是 MoE 的调度器。它根据 token 的 hidden state 计算每个 expert 的分数，然后选择 top-k experts。一个常见流程是：

```text
token hidden state
  -> router projection
  -> expert scores
  -> top-k selection
  -> expert computation
  -> weighted sum
```

Router 学到的不是人类可读的任务标签，而是一个高维空间里的分配规则。它可能把某些数学 token、代码 token、特定语言 token 或语义模式路由到不同 expert，但这种 specialization 不一定直接对应人类命名的领域。

Router 的质量决定了 MoE 是否真的有效：

- 如果 router 总把 token 发给少数 expert，会出现 expert overload。
- 如果很多 expert 长期收不到 token，会造成容量浪费。
- 如果 routing 太不稳定，训练和推理都会变复杂。
- 如果负载均衡约束太强，可能损害模型按内容选择 expert 的自由度。

## 负载均衡问题

MoE 的核心难点不是“多放几个 FFN”这么简单，而是如何让 experts 被合理使用。

假设模型有 64 个 experts，但 router 总是偏爱其中 5 个，那么会产生两个问题：

1. 热门 experts 计算压力过大，影响吞吐和延迟。
2. 冷门 experts 学不到足够内容，参数容量被浪费。

早期 MoE 通常使用 auxiliary load balancing loss，让 router 不要过度集中到少数 experts。但这个 loss 本身也有取舍：如果负载均衡约束太强，router 可能为了平均而平均，反而损害模型性能。

[[architecture/model-families/deepseek|DeepSeek]] 系列中的 DeepSeek-V3 强调 auxiliary-loss-free load balancing，目标就是减少负载均衡损失对模型效果的干扰，同时仍然避免 expert load 失衡。这说明 MoE 的关键问题已经从“能不能稀疏激活”进一步发展到“如何稳定、高效、低损失地路由”。

## Capacity 与 Token Dropping

MoE 训练中还要处理 expert capacity。每个 expert 在一个 batch 内能接收的 token 数通常有上限。如果 router 把太多 token 发给同一个 expert，系统需要决定如何处理溢出 token：

- 丢弃部分 expert computation；
- 转发到备选 expert；
- 增大 capacity factor；
- 使用更强的负载均衡或 routing 约束。

Capacity factor 越大，越不容易丢 token，但计算和通信 buffer 也更大；capacity 太小，可能损害训练信号和模型质量。这个问题说明 MoE 的稀疏性不只是数学结构，也是 batch-level 系统调度问题。

## Shared Experts 与 Routed Experts

一些现代 MoE 架构会区分 shared experts 和 routed experts。

- Shared expert：每个 token 都会经过，提供通用能力。
- Routed expert：由 router 选择，提供更稀疏、更专门的容量。

这种设计的直觉是：并不是所有知识都应该被稀疏路由。有些基础变换、常见语言模式或通用能力，可能让所有 token 共享更稳定；而更细分的模式可以交给 routed experts。

Llama 4 Maverick 的官方描述就提到，每个 token 会进入 shared expert，同时被路由到 128 个 routed experts 中的一个。DeepSeekMoE 也使用了 shared experts 和 routed experts 的设计。

## MoE 的优势

MoE 的优势主要来自稀疏计算。

### 1. 扩大模型容量

在同等每 token 计算预算下，MoE 可以拥有更多 total parameters。更多参数给模型提供了更大的表达容量，尤其适合在大规模数据上学习多语言、多领域、多任务模式。

### 2. 降低每 token 计算成本

如果每个 token 只激活少数 experts，那么每 token FLOPs 可以远小于同等 total parameters 的 dense model。这让“更大容量”不必完全等价于“更高计算成本”。

### 3. 适合专家并行

MoE 的 expert 可以分布在不同设备上，通过 expert parallelism 扩展训练和推理。但这同时也引入通信、调度和负载均衡问题。

## 训练与系统复杂度

MoE 的训练通常需要组合多种并行方式：

- data parallel：复制或切分 batch；
- tensor parallel：切分矩阵计算；
- expert parallel：把不同 experts 放到不同设备；
- pipeline parallel：按层切分模型。

其中 expert parallel 会引入 token dispatch / combine 通信：token 先被路由到对应 expert 所在设备，expert 计算后再把结果发回原位置。若路由分布不均，某些设备会成为 straggler，拖慢整个 step。

因此 MoE 的吞吐不仅取决于 active parameters，也取决于：

- router 分布是否均衡；
- expert placement 是否合理；
- all-to-all 通信是否高效；
- batch size 是否足以填满 experts；
- serving 时请求是否能形成稳定 batch。

## Switch Transformer：Top-1 Routing 的基线

[[sources/papers/2021-switch-transformer|Switch Transformer]] 将传统 MoE 的 top-$k$ routing 简化为 top-1：每个 token 只发送给最高 probability 的一个 expert，并用该 gate value 缩放 expert output。它保留了 conditional computation 的容量优势，同时减少了每 token 的 expert computation、capacity buffer 和 dispatch/communication 开销。

对一个包含 $T$ 个 tokens、$N$ 个 experts 的 routing group，Switch 的 expert capacity 为：

$$
\text{capacity}=\frac{T}{N}\times\text{capacity factor}
$$

超过 capacity 的 token 不经过该 expert FFN，而通过 residual connection 进入下一层。它使用：

$$
\mathcal{L}_{balance}=\alpha N\sum_{i=1}^{N}f_iP_i
$$

其中 $f_i$ 是实际发送到 expert $i$ 的 token 比例，$P_i$ 是 router 对 expert $i$ 分配的平均 probability mass。这个损失同时约束 hard assignment 与 soft routing probability，但不等价于严格的 device-level 或 network-level balance。

Switch 的关键稳定性经验包括：router 内部使用 FP32、其余路径保留 BF16；将 initialization scale 从 1.0 降到 0.1；fine-tuning 时把 expert FFN dropout 提高到 0.4，而非 expert layers 保持 0.1。这些方法说明，router 数值稳定、expert capacity 和 downstream regularization 是一组相互关联的系统问题。

在约束相同 FLOPs per token 的 T5/C4 实验中，增加 experts 从 2 到 256 能改善 pre-training sample efficiency；但 Switch 的论文结果基于 encoder-decoder 的 span-corruption objective、TPU 和 Mesh TensorFlow，不能把其速度倍率直接外推到 decoder-only NTP 或 GPU/NCCL 环境。

## GShard：MoE 与 Automatic Sharding 的早期系统范式

[[sources/papers/2020-gshard|GShard]] 展示了一个完整的 MoE 扩展路径：用 sparsely-gated MoE 扩大 total capacity，再用 XLA SPMD partitioner 将 expert weights 分片到不同设备。普通 Transformer layers 可以保持 replicated layout，MoE experts 则按 expert dimension shard；token 通过 router 选择 experts 后，在 group/token layout 与 expert layout 之间进行 dispatch 和 combine。

GShard 的 top-2 gating 包含 expert capacity、local group dispatching、auxiliary load-balancing loss 和 second-expert random routing。超过 capacity 的 token 会跳过当前 expert computation，通过 residual connection 继续前向。这个设计说明，MoE 的 routing 不只是一个 top-k 函数，还同时决定 buffer shape、token overflow、负载均衡和通信模式。

在系统层面，GShard 用 `replicate`、`split`、`shard` annotations 描述 tensor layout，再由 compiler 自动传播 sharding、插入 resharding 和生成 collective communication。MoE dispatch 的关键 layout 转换使用 `AllToAll`；contracting dimension 上的 partial result 使用 `AllReduce`；需要恢复完整 tensor 时使用 `AllGather`；halo exchange 和局部 device reorder 使用 `CollectivePermute`。

这套设计与现代 expert parallel 的关系可以概括为：

```text
router top-k
  -> capacity / local grouping
  -> token dispatch
  -> AllToAll to expert partitions
  -> local expert FFN
  -> AllToAll combine
```

GShard 在 100-language translation 上的实验中，将 expert 数从 128 扩展到 2048，并观察到高资源语言更受益于增加 experts，低资源语言则更依赖共享网络带来的 positive transfer。它因此提醒我们：expert 数、shared capacity、routing balance 和任务间 transfer 必须联合分析。

## MoE 的代价

MoE 不是免费午餐。它把 dense model 的一部分计算问题，转化成了路由和系统问题。

### 1. 部署复杂度更高

Dense model 的每层计算比较规则，矩阵乘法容易优化。MoE 的 token 会被动态分配到不同 experts，batch 内 token 的 expert 分布可能很不均匀，导致 kernel、通信和并行调度更复杂。

### 2. 显存仍然受 total parameters 影响

即使每个 token 只激活一部分参数，模型权重仍然要被存储。对于 400B total parameters 的 MoE，如果没有量化、分片或按需加载策略，部署成本仍然很高。

### 3. 训练稳定性更难

Router、experts 和主干网络要共同训练。训练过程中可能出现 expert collapse、负载不均、梯度不稳定或某些 experts 学不到东西。

### 4. 指标解释更容易混乱

MoE 模型经常同时报告 active parameters 和 total parameters。如果只拿 active parameters 和 dense model 的总参数比较，可能低估 MoE 的模型容量；如果只拿 total parameters 比较，又可能高估每 token 计算成本。

### 5. 微调与部署更复杂

MoE 的参数分布在多个 experts 中。微调时，如果只更新部分参数，可能改变 router 与 expert 的协同；如果全量微调，显存和通信成本又很高。部署时，量化、专家放置、batching 和 KV Cache 管理都需要结合 MoE 的动态路由特点设计。

## 与 Attention 的关系

MoE 通常不直接替代 [[architecture/attention/attention|Attention]]，而是替代或增强 FFN。Attention 负责 token 之间的信息交互，FFN / experts 负责对每个 token 的表示进行非线性变换。

因此，一个 MoE Transformer block 仍然需要 attention 来混合上下文信息。MoE 主要改变的是“每个 token 在通过 FFN 变换时使用哪些参数”。这也是为什么很多模型会同时讨论高效 attention 和 MoE：前者影响上下文交互和 KV Cache，后者影响模型容量和 FFN 计算。

## 典型模型

### Mixtral

Mixtral 是开源 MoE 路线中非常有代表性的模型之一。它让社区更直观地看到 sparse MoE 可以在开放权重生态中工作，并推动了 MoE serving、量化和微调工具的发展。

### DeepSeek-V2：从 Expert Balance 到 Device Balance

[[sources/papers/2024-deepseek-v2|DeepSeek-V2]] 将 DeepSeekMoE 扩展到 236B total / 21B activated parameters。除第一层外，每个 FFN 都由 2 个 shared experts 和 160 个 routed experts 组成，每个 token 激活 6 个 routed experts。Fine-grained expert segmentation 提高组合粒度，shared expert isolation 则让通用变换不必在多个 routed experts 中重复学习。

Fine-grained experts 也会扩大 token 的设备访问范围。V2 的 device-limited routing 先选择包含高 affinity experts 的 $M$ 个 devices，再只在这些 devices 内进行 expert top-k；正式训练采用 $M=3$。论文观察到 $M\geq3$ 时，效果与 unrestricted routing 大致相当。这类 routing constraint 不减少激活 expert 数，而是把物理通信拓扑纳入选择空间。

V2 进一步把 load balance 分为三个层次：

- expert-level balance：避免 routing collapse，使 experts 获得训练信号；
- device-level balance：均衡不同设备上的计算量，减少 straggler；
- communication balance：均衡各设备接收的 tokens，避免网络热点。

三者分别约束模型使用、device compute 和 network traffic，不能由单一 expert-count 指标替代。由于 auxiliary losses 不能保证严格均衡，V2 在 training 中以 capacity factor 1.0 丢弃每个 device 上 affinity 最低的溢出 assignments，同时保护约 10% 的 sequences 永不丢 token；evaluation 不进行 dropping。

V2 与 V3 的差异应明确保留：V2 使用三类 auxiliary balance loss 和 training-time token dropping；V3 转向 auxiliary-loss-free batch-wise bias correction，只保留很小的 sequence-wise loss，并取消 training / inference token dropping。后者是后续系统演进，不应反向写成 V2 的原始做法。

### DeepSeekMoE / DeepSeek-V3

DeepSeek-V3 使用 DeepSeekMoE，并结合 Multi-Head Latent Attention、auxiliary-loss-free load balancing 和 multi-token prediction。它的重要性在于把 MoE 放进一个高性价比训练和推理路线中，而不是只把 MoE 当成扩大参数量的技巧。

DeepSeek-V3 的具体配置是 1 个 shared expert、256 个 routed experts，每个 token 激活 8 个 routed experts，并限制一个 token 最多被发送到 4 个节点。模型共有 671B total parameters，每个 token 激活约 37B parameters。这个数字组合同时体现了 MoE 的两个维度：total parameters 提供容量，active parameters 更接近每 token 的主要计算路径；但训练和部署仍需承担全部 experts 的存储、放置与通信成本。

### Auxiliary-loss-free Routing

DeepSeek-V3 没有把全局负载均衡主要交给 auxiliary loss，而是为每个 routed expert 维护一个 bias $b_i$。router 用 $s_{i,t}+b_i$ 决定 top-k expert，真正参与 expert output 加权的 gating value 仍来自原始 affinity score $s_{i,t}$：

$$
g'_{i,t}=
\begin{cases}
s_{i,t},&
s_{i,t}+b_i\in\mathrm{TopK}(\{s_{j,t}+b_j\},K_r)\\
0,&\text{otherwise}
\end{cases}
$$

每个 training step 结束后，根据 batch-level expert load 更新 bias：overloaded expert 的 bias 减少一个更新步长，underloaded expert 的 bias 增加一个更新步长。bias 只用于 routing，不通过主 loss 反向传播，因此可以在保持负载平衡的同时减少对内容相关 routing score 的直接干扰。

这并不表示模型完全不使用 balance loss。DeepSeek-V3 仍保留极小的 sequence-wise balance loss，主要用来避免单条 sequence 内出现极端失衡。这个分工很重要：

- batch-wise bias update 负责全局设备和 expert load；
- sequence-wise auxiliary loss 负责局部极端情况；
- 原始 affinity score 负责内容相关的 expert specialization。

作者的 ablation 还表明，batch-wise auxiliary loss 可以取得接近 auxiliary-loss-free 的 validation loss，而 sequence-wise balance 的约束更强，可能限制不同 domain 使用不同 experts 的自由度。

### Node-Limited Routing 与 Token Dropping

DeepSeek-V3 的 node-limited routing 先按照每个节点上候选 experts 的高分聚合结果选择目标节点，使每个 token 最多跨越 4 个节点。这样可以减少 InfiniBand traffic，并配合节点内 NVLink forwarding 实现更高的通信利用率。

得益于负载均衡策略，DeepSeek-V3 在训练和推理中都不进行 token dropping。需要注意，这不是 MoE 的普遍性质，而是当前 routing、batch size、expert parallelism 和部署策略共同达到的结果。其他模型若仍有明显 expert overload，仍需要 capacity factor、备用 expert 或 token dropping 等机制。

### Llama 4 Scout / Maverick

Llama 4 是 LLaMA 家族首次公开采用 MoE 的一代。Scout 是 17B active / 109B total、16 experts；Maverick 是 17B active / 400B total、128 experts。它们说明 MoE 已经进入主流开放权重模型家族。

## 常见误解

### 误解一：MoE 模型只需要看 active parameters

不对。active parameters 影响每 token 计算，但 total parameters 影响容量、权重存储和部署复杂度。MoE 模型应该同时报告并同时理解这两个数字。

### 误解二：MoE 等于多个模型投票

不准确。MoE 不是简单 ensemble。Experts 是同一个模型内部的子模块，由 router 在 token 级别选择，通常端到端共同训练。

### 误解三：每个 expert 都对应一个人类可解释领域

不一定。Experts 可能产生某种 specialization，但它们未必对应“数学专家”“代码专家”“中文专家”这种清晰标签。把 expert 过度拟人化会误导理解。

### 误解四：MoE 一定比 dense model 更好

不一定。MoE 提供的是容量和计算之间的新 trade-off。它需要足够大的数据、稳定的 routing、合适的负载均衡和成熟的训练/推理系统。小规模或简单场景下，dense model 可能更简单、更稳定。

### 误解五：active parameters 等于实际部署成本

不对。active parameters 更接近每 token 计算路径，但部署仍要存储 total parameters，并处理 expert dispatch、通信和负载均衡。

## 相关概念

- [[architecture/transformer/transformer|Transformer]] — MoE 通常作为 Transformer FFN 的稀疏替代。
- [[architecture/transformer/feed-forward|Feed Forward Network]] — MoE experts 常常就是多个 FFN。
- [[architecture/attention/attention|Attention]] — 与 MoE 分工不同，负责 token 间交互。
- [[training/distributed-training/megatron|Megatron]] — 多维并行训练和 MoE 系统相关。
- [[architecture/model-families/deepseek|DeepSeek]] — DeepSeekMoE 的重要模型家族案例。
- [[architecture/model-families/llama|LLaMA]] — Llama 4 Scout / Maverick 引入 MoE。
- [[inference/serving-systems/vllm|vLLM]] — MoE serving 常需要高效推理系统支持。

## 经典论文与资料

- [[sources/papers/2017-outrageously-large-neural-networks|Outrageously Large Neural Networks]]
- [[sources/papers/2020-gshard|GShard]]
- [[sources/papers/2021-switch-transformer|Switch Transformer]]
- [[sources/papers/2024-deepseekmoe|DeepSeekMoE]]
- [[sources/papers/2024-deepseek-v2|DeepSeek-V2]]
- [[sources/papers/2024-deepseek-v3|DeepSeek-V3]]
