---
title: Sliding Window Attention
created: 2026-01-31
published: 2026-01-31
modified: 2026-05-31
type: topic
status: mature
area: architecture
tags:
  - attention
  - long-context
---

Sliding Window Attention 是一种局部 attention 机制。它限制每个 token 只关注附近窗口内的 token，而不是完整历史上下文。它的目标是降低长序列训练和推理中的 attention 计算、显存和 KV Cache 压力。

## 基本形式

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

## 与 Full Attention 的对比

| 机制 | 可见范围 | 成本 | 长距离能力 |
|---|---|---|---|
| Full causal attention | 全部历史 token | 高，$O(T^2)$ | 直接访问远处 token |
| Sliding window attention | 最近 $w$ 个 token | 低，$O(Tw)$ | 依赖层间传播或额外机制 |

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
- [[architecture/attention/hybrid-attention|Hybrid Attention]]
- [[training/mid-training/long-context-training|Long-context Training]]
- [[inference/kv-cache-and-memory/kv-cache|KV Cache]]
- [[inference/attention-acceleration/flash-attention|FlashAttention]]

## 经典论文与资料

- [[sources/papers/2020-longformer|Longformer]]
- [[sources/papers/2020-big-bird|Big Bird]]
