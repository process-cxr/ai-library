---
title: Multi-Head Latent Attention
created: 2026-01-31
published: 2026-01-31
modified: 2026-05-31
type: topic
status: mature
area: architecture
tags:
  - attention
  - mla
  - kv-cache
  - deepseek
aliases:
  - MLA
  - Multi Head Latent Attention
---

## 核心问题

Multi-Head Latent Attention，简称 MLA，是 DeepSeek-V2 / DeepSeek-V3 中使用的一种高效 attention 结构。它要解决的核心问题是：自回归推理时，标准 Multi-Head Attention 需要为每个历史 token 缓存大量 key/value states，导致 [[inference/kv-cache-and-memory/kv-cache|KV Cache]] 随上下文长度和 batch size 快速增长。

在长上下文推理中，KV Cache 往往比模型权重更容易成为瓶颈。因为生成每个新 token 时，模型都需要读取历史 token 的 K/V；如果每层、每个 head、每个历史 token 都保存完整 K/V，显存和 memory bandwidth 压力会非常大。

MLA 的核心思想是：不要直接缓存每个 head 的完整 K/V，而是先把 K/V 联合压缩成一个低维 latent vector；推理时缓存这个 latent representation，再通过投影恢复 attention 所需的 K/V 表示。

一句话概括：MLA 用 latent compression 换取更小的 KV Cache，同时尽量保留接近 MHA 的表达能力。

## 标准 Attention 的 KV Cache 问题

在标准 [[architecture/attention/multi-head-attention|Multi-Head Attention]] 中，每个 token 会产生 query、key、value：

$$
q_t = h_t W_Q, \quad k_t = h_t W_K, \quad v_t = h_t W_V
$$

自回归生成时，第 $t$ 个 token 之后，后续 token 还需要访问它的 $k_t$ 和 $v_t$。因此系统会把历史 token 的 K/V 缓存下来。

如果有：

- $L$ 层 Transformer；
- $n_h$ 个 attention heads；
- 每个 head 的维度是 $d_h$；
- 上下文长度是 $T$；

那么 KV Cache 大致随 $L \times T \times n_h \times d_h$ 增长。对于 32K、128K 甚至 1M context，这个成本会非常夸张。

这也是为什么现代 LLM 会使用 MQA、GQA、MLA、sliding window attention、paged KV cache 等方法降低推理成本。

## MLA 的基本想法

MLA 不是简单让多个 query heads 共享 K/V heads。它的做法更像低秩压缩：先把当前 token 的 hidden state 压缩到一个低维 latent space，再从这个 latent representation 生成 keys 和 values。

DeepSeek-V3 技术报告把 MLA 的关键点称为 low-rank joint compression for attention keys and values。

直观流程是：

```text
hidden state h_t
  -> down projection
  -> compressed latent vector c_t^{KV}
  -> up projection
  -> keys and values used by attention
```

推理时真正需要缓存的不是完整 K/V，而是压缩后的 $c_t^{KV}$ 以及和 RoPE 相关的一小部分 key 表示。这样可以显著减少 KV Cache。

## 关键变量

设：

- $h_t \in \mathbb{R}^{d}$：第 $t$ 个 token 在某一层的 hidden state。
- $n_h$：attention heads 数量。
- $d_h$：每个 head 的维度。
- $d_c$：KV compression dimension，并且 $d_c \ll n_h d_h$。
- $c_t^{KV} \in \mathbb{R}^{d_c}$：压缩后的 latent vector。

标准 MHA 会直接为每个 token 保存接近 $n_h d_h$ 规模的 key 和 value。MLA 则希望只保存更小的 $d_c$ 维 latent vector，再从中恢复需要的 K/V。

这就是它节省 KV Cache 的根本原因：缓存维度从完整 multi-head K/V 表示变成了更低维的 latent representation。

## K/V 联合压缩

MLA 的第一步是把 hidden state 压缩为 K/V 共用的 latent vector：

$$
c_t^{KV} = W_D^{KV} h_t
$$

其中：

- $W_D^{KV}$ 是 down-projection matrix。
- $c_t^{KV}$ 是低维压缩表示。

然后用 up-projection 从 $c_t^{KV}$ 中恢复 attention 所需的 key/value 表示：

$$
k_t^C = W_U^K c_t^{KV}
$$

$$
v_t^C = W_U^V c_t^{KV}
$$

这里的上标 $C$ 可以理解为 content part。它们来自压缩后的 latent vector，而不是直接从 $h_t$ 独立生成完整 K/V。

这种做法和低秩分解有相似直觉：原本一个大矩阵投影被拆成“先降维、再升维”的结构。模型不再为每个 token 缓存完整 K/V，而是缓存更紧凑的中间表示。

## RoPE 为什么要特殊处理

如果只是压缩 K/V，会遇到一个位置编码问题。DeepSeek 系列使用 [[architecture/positional-encoding/rope|RoPE]]，而 RoPE 通常作用在 query/key 上，用旋转方式注入位置信息。

位置相关的信息不能被随便压进同一个 latent vector 后再恢复，否则可能影响 attention score 中的相对位置信息。因此 MLA 会把 key 拆成两部分：

- content part：来自 latent compression。
- RoPE part：专门承载位置信息。

DeepSeek-V3 技术报告中提到，生成时需要缓存的是 $c_t^{KV}$ 和 $k_t^R$。其中 $k_t^R$ 是和 RoPE 相关的 key 部分。

可以把它理解为：MLA 压缩主要内容信息，但保留一条专门处理位置信息的路径，以避免 RoPE 信息在压缩中损失过多。

## Query 也可以压缩

DeepSeek-V3 技术报告还提到对 query 也进行 low-rank compression，以减少训练时的 activation memory。

大致流程是：

```text
h_t
  -> query down projection
  -> compressed query latent vector
  -> query up projection
  -> per-head query representations
```

这部分主要影响训练内存和计算组织。和 KV Cache 最直接相关的是 K/V compression，因为推理时历史 token 的 K/V 要长期保存，而 query 通常只针对当前 token 计算，不需要像历史 K/V 那样缓存。

## MLA、MQA、GQA 的区别

MLA、MQA、GQA 都在解决 attention 推理成本问题，但方法不同。

| 方法 | 核心做法 | 主要节省点 | 代价 |
|---|---|---|---|
| MQA | 所有 query heads 共享一组 K/V | KV Cache 最小 | K/V 表达能力可能受限 |
| GQA | 多个 query heads 分组共享 K/V | 在 MHA 与 MQA 之间折中 | 分组数量需要权衡 |
| MLA | K/V 联合压缩成 latent vector，再恢复使用 | 用低维 latent cache 降低 KV Cache | 结构更复杂，实现和训练更难 |

MQA/GQA 的直觉是“减少 K/V heads 数量”。MLA 的直觉是“保留多头表达，但把缓存形式压缩到 latent space”。

这也是为什么 MLA 不能简单等同于 GQA。GQA 是 head sharing，MLA 是 latent compression。

## 为什么 MLA 适合 DeepSeek

DeepSeek-V3 是 671B total / 37B activated parameters 的 MoE 模型，目标之一是高性价比训练和推理。MoE 主要降低每 token FFN 计算成本，但 attention 的 KV Cache 仍然会随上下文增长。

因此，DeepSeek 需要同时处理两个瓶颈：

- FFN / model capacity：用 DeepSeekMoE 解决。
- Attention / KV Cache：用 MLA 解决。

这两个设计是互补的。MoE 让模型容量更大但每 token 激活参数较少；MLA 让长上下文推理时的 K/V 缓存更小。只理解 MoE 而不理解 MLA，会低估 DeepSeek 架构中 attention efficiency 的重要性。

## 优势

### 1. 降低 KV Cache

MLA 最直接的优势是减少生成时需要缓存的内容。DeepSeek-V3 技术报告明确说，MLA 只需要缓存 $c_t^{KV}$ 和 $k_t^R$，从而显著降低 KV Cache。

### 2. 保留较强表达能力

相比直接用 MQA 把所有 heads 的 K/V 合并，MLA 通过 latent representation 和 up-projection 尝试保留更丰富的 K/V 表达能力。报告称它在降低 KV Cache 的同时保持了接近标准 MHA 的表现。

### 3. 更适合长上下文

上下文越长，KV Cache 节省越重要。MLA 的收益会随着 context length 和 batch size 增大而更明显。

## 代价与难点

### 1. 结构更复杂

MLA 引入 down-projection、up-projection、RoPE decoupling 等设计，实现复杂度高于标准 MHA/GQA。

### 2. 训练和推理框架需要适配

如果推理框架默认假设缓存的是完整 K/V，那么 MLA 需要专门支持 latent cache 的存储、恢复和 attention 计算。

### 3. 压缩维度需要权衡

$d_c$ 太小，可能损失表达能力；$d_c$ 太大，KV Cache 节省有限。压缩维度是效果和效率之间的关键超参数。

### 4. RoPE 处理不能随意简化

RoPE 与相对位置建模有关。如果位置相关 key 被压缩处理不当，可能影响长上下文位置关系。因此 MLA 对 RoPE key 做了专门路径。

## 常见误解

### 误解一：MLA 就是 GQA 的另一种名字

不对。GQA 通过多个 query heads 共享 K/V heads 来减少缓存；MLA 通过低维 latent vector 压缩 K/V。二者目标相似，但机制不同。

### 误解二：MLA 只影响训练，不影响推理

不对。MLA 的核心动机之一就是降低推理时的 KV Cache。query compression 更偏训练 activation memory，但 K/V compression 直接影响自回归生成。

### 误解三：压缩 KV 一定无损

不严谨。MLA 设计目标是在显著降低 KV Cache 的同时保持接近 MHA 的性能，但压缩本身就是 trade-off。具体效果依赖压缩维度、训练规模、模型结构和任务。

### 误解四：MLA 解决了所有长上下文问题

不对。MLA 降低 KV Cache 成本，但长上下文还涉及训练长度、位置编码、数据分布、attention pattern、检索能力、推理系统和评测方式。

## Related Notes

- [[architecture/attention/attention|Attention]]
- [[architecture/attention/multi-head-attention|Multi-Head Attention]]
- [[architecture/attention/grouped-query-attention|Grouped-Query Attention]]
- [[architecture/attention/multi-query-attention|Multi-Query Attention]]
- [[architecture/positional-encoding/rope|RoPE]]
- [[architecture/model-families/deepseek|DeepSeek]]
- [[inference/kv-cache-and-memory/kv-cache|KV Cache]]

## 经典论文与资料

- [[sources/papers/2024-deepseek-v2|DeepSeek-V2]]
- [[sources/papers/2024-deepseek-v3|DeepSeek-V3]]
