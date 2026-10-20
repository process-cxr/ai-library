---
title: Sliding Window Attention
created: 2026-01-31
published: 2026-01-31
modified: 2026-10-20
type: topic
status: mature
area: architecture
tags:
  - attention
  - long-context
---

Sliding Window Attention 是一种局部 attention 机制。它限制每个 token 只关注附近窗口内的 token，以降低长序列 attention 的计算和中间存储。用于因果生成时，它还可以减少所需的历史键值缓存（KV Cache）；用于双向编码器时，窗口通常覆盖当前位置的两侧。

## 因果窗口与双向窗口

### 因果窗口

在标准 causal attention 中，第 $t$ 个 token 可以关注所有历史 token：

$$
x_1,\dots,x_t
$$

Sliding window attention 只允许它关注最近 $w$ 个 token：

$$
x_{\max(1,t-w+1)},\dots,x_t
$$

其中 $w$ 是 window size。

Mask 形式可以理解为：

$$
A_{tj}=0 \quad \text{if } j>t \text{ or } j<t-w+1
$$

这样每个 token 的可见范围从 $O(T)$ 降到 $O(w)$。

### 双向窗口

长文档编码器可以同时读取左右邻居。例如把单侧半径定义为 $r$，有：

$$
\mathcal W_t=\{j:1\le j\le T,\ |j-t|\le r\}.
$$

窗口至多覆盖 $2r+1$ 个位置。Longformer 将左右两侧合计宽度记为 $w$，通常描述为左右各 $w/2$；它与“左侧共 $w$ 个历史位置”的因果窗口符号不同。比较实现时应核对窗口参数究竟表示单侧半径、两侧总宽度还是 block 数。

[[sources/papers/2020-longformer|Longformer 原文解析]]展示了两种使用方式：字符级语言建模使用历史一侧的窗口，RoBERTa 迁移后的编码器使用双向窗口。因此滑动窗口描述的是访问范围，不能仅凭这个名称判断模型是否满足因果约束。

## 复杂度收益

标准 full attention 对序列长度 $T$ 的 score matrix 是：

$$
T \times T
$$

复杂度约为：

$$
O(T^2d)
$$

Sliding window attention 中，每个 token 只关注 $w$ 个位置，复杂度变为：

$$
O(Twd)
$$

当 $w \ll T$ 时，长序列成本显著降低。训练时可以减少 attention activation；推理时也可以只保留或读取窗口内的 K/V。

## 信息传播

Sliding window 限制的是单层直接可见范围，但多层堆叠可以扩大有效感受野。假设每层窗口大小为 $w$，经过多层后，信息可以逐层向前传播。

不过这种传播不是免费的：

- 远距离信息需要经过多层传递；
- 精确检索远处 token 可能变难；
- 层数有限时，窗口外信息可能无法充分影响当前 token；
- 对需要全局证据的任务，纯局部窗口可能不足。

因此 sliding window 常与 global attention、稀疏 attention、压缩记忆、retrieval 或局部/全局混合策略结合。

## 扩张窗口：用访问密度换覆盖跨度

扩张窗口在被读取的位置之间留出间隔，在近似相同的配对数下覆盖更远的距离。它扩大的是跨度，不能保留跨度内每个 token 的细粒度直接访问。通常需要搭配非扩张 heads 或层来保留近邻信息。

Longformer 在字符级语言建模中让低层使用较局部的窗口，高层增大窗口并在两个 heads 上使用 dilation；其 RoBERTa 迁移配置则使用非扩张窗口，因为加入 dilation 损害了已有权重的表现。具体消融及适用范围见 [[sources/papers/2020-longformer#实验、消融与归因|Longformer 的实验解析]]。

## 从局部窗口到全局中转

局部窗口适合邻近依赖，但问题 token 或分类标记常需要聚合整篇输入。Longformer 为少量位置配置双向全局连接：这些位置读取所有 token，普通位置也能读取它们。若全局位置为 $g$ 个，attention 主成本约为 $O(T(w+g)d)$；在 $w,g$ 固定时随序列长度线性增长。独立的局部/全局投影和模型适配过程见 [[sources/papers/2020-longformer#Attention 机制|Longformer 的机制解析]]。

BigBird 则进一步分析窗口、全局位置和随机连接组成的稀疏图，通过 block 组织这些访问以适配硬件。其部分 ETC 问答配置没有随机连接，不能将所有收益都归因给随机边；变体和理论条件详见 [[sources/papers/2020-big-bird|BigBird 原文解析]]。更一般的邻域定义与块级计算见 [[architecture/attention/sparse-attention|Sparse and Block-Sparse Attention]]。

## Mask 必须配合计算实现

若先计算完整 attention matrix，再把窗口外位置屏蔽，仍然付出了全量 score 计算和存储成本。实际加速需要只计算窗口对应的带状区域，或者以重叠 chunk、block-sparse kernel 组织计算。

Longformer 的实现比较说明：逐对角线循环虽然少算无效位置，却可能很慢；重叠 chunks 多算部分区域，但能调用高效稠密矩阵乘法。算法配对数与实际 kernel 时间需要分别验证，细节见 [[sources/papers/2020-longformer#稀疏模式如何在硬件上执行|Longformer 的实现解析]]。

## 与 Full Attention 的对比

| 机制                     | 可见范围          | 成本         | 长距离能力             |
| ------------------------ | ----------------- | ------------ | ---------------------- |
| Full causal attention    | 全部历史 token    | 高，$O(T^2)$ | 直接访问远处 token     |
| Sliding window attention | 最近 $w$ 个 token | 低，$O(Tw)$  | 依赖层间传播或额外机制 |

Sliding window 的优势是成本可控，代价是全局信息访问能力变弱。

## 与 KV Cache 的关系

在 decoder-only 推理中，如果模型的 attention pattern 只需要最近窗口，系统可以只保留或只读取窗口内 K/V，从而降低长上下文 decode 的显存和带宽压力。

但具体实现要看模型是否所有层都使用窗口注意力，以及是否有周期性 full attention / global attention 层。如果部分层仍需要全局 K/V，系统就不能简单丢弃全部窗口外缓存。

## 适用场景

Sliding window attention 适合：

- 局部依赖占主导的语言或代码模式；
- 长文本中主要需要邻近上下文的生成；
- 需要控制 attention 成本的长序列训练；
- 与全局/稀疏/压缩机制组合的 hybrid architecture。

它不适合单独承担所有长程依赖，尤其是需要跨文档检索、远距离多跳推理或精确引用远处证据的任务。

## 常见误解

- **误解：sliding window 就等于长上下文能力。** 它降低成本，但纯局部窗口会限制直接访问远处信息。
- **误解：窗口外信息完全不可用。** 多层堆叠可以传播窗口外信息，但传播效率和精度有限。
- **误解：可以随便缩小窗口。** 窗口太小会损害局部语法、代码结构和长依赖建模。
- **误解：KV Cache 可以一定只保留窗口。** 这取决于模型所有层的 attention pattern 和实现。

## 相关概念

- [[architecture/attention/attention|Attention]]
- [[architecture/attention/self-attention|Self-Attention]]
- [[architecture/attention/sparse-attention|Sparse and Block-Sparse Attention]]
- [[architecture/attention/hybrid-attention|Hybrid Attention]]
- [[training/mid-training/long-context-training|Long-context Training]]
- [[inference/kv-cache-and-memory/kv-cache|KV Cache]]
- [[inference/attention-acceleration/flash-attention|FlashAttention]]

## 经典论文与资料

- [[sources/papers/2020-longformer|Longformer]]
- [[sources/papers/2020-big-bird|Big Bird]]
