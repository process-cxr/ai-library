---
title: Sparse and Block-Sparse Attention
created: 2026-10-20
published: 2026-10-20
modified: 2026-10-20
type: topic
status: growing
area: architecture
tags:
  - attention
  - sparse-attention
  - long-context
aliases:
  - Sparse Attention
  - Block Sparse Attention
  - Block-Sparse Attention
---

## 从全连接到可见邻域

Sparse attention 改变的是每个 query 能读取哪些 key/value（键和值，简称 K/V）位置。标准 full attention 让当前位置与所有可见位置配对；稀疏 attention 只计算一个较小的邻域。核心问题是：保留哪些连接，才能既降低成本，又让必要的信息在有限层数内到达当前 token？

用 $\mathcal N_i$ 表示 query $q_i$ 的可见 key 集合，单个 head 的输出为：

$$
o_i=\sum_{j\in\mathcal N_i}
\frac{\exp(q_i^Tk_j/\sqrt d)}
{\sum_{t\in\mathcal N_i}\exp(q_i^Tk_t/\sqrt d)}v_j,
$$

其中 $d$ 是 query/key 的 head dimension。softmax 只在可见邻域内归一化；未连接的位置不参加本层的匹配和加权。它保留被访问位置的逐 token 表示，并不自动把历史压缩为更少的 K/V entries。

这一层面的稀疏指输入位置之间的连接，与 [[architecture/sparse-and-efficient/moe|Mixture of Experts]] 对前馈专家的稀疏激活是两个不同设计。

## 局部、全局与随机连接分别解决什么

[[architecture/attention/sliding-window-attention|滑动窗口]] 是最直观的稀疏模式：只保留主对角线附近的连接，优先处理词序相邻的信息。它的限制也直接：单层不能读取窗口之外的证据，远程信息需要经过多个层次传递。

Longformer 用少量全局位置补充这条路径。分类标记或问题 token 可以读取整段输入，普通 token 也能读取这些位置，从而通过信息中转缩短跨区间路径。其双向窗口、任务特定全局位置和独立投影的设计，以及从 RoBERTa 迁移的实验，可见 [[sources/papers/2020-longformer|Longformer 原文解析]]。全局位置更新后的信息通常要到下一层才会被其他位置读取，两跳连通不代表一层内完成两次聚合。

BigBird 进一步从图结构出发分析连接：局部边保留邻近依赖，全局节点提供中转，随机边让部分配置在远处建立额外通路。随机连接由结构采样得到，不是根据当前 query 的语义做 top-k 检索。[[sources/papers/2020-big-bird|BigBird 原文解析]]同时记录了这套设计的表达能力证明和限制：部分 ETC 实验没有随机边，而通用逼近性质也不意味着固定深度模型可以无损替代 full attention。

| 模式             | 单层直接可见范围              | 远程信息的主要路径                   |
| ---------------- | ----------------------------- | ------------------------------------ |
| 局部窗口         | 当前附近的位置                | 多层逐步传播                         |
| 局部加全局       | 邻居与少量全局位置            | 经过全局状态中转                     |
| 局部加随机加全局 | 再加入采样的远程邻居          | 额外跨区间路径与全局中转             |
| 内容相关选择     | 由 query 和历史表示决定的子集 | 选择器找到的相关位置；需计入选择开销 |

前三类可由预定义结构组织。内容相关选择还要解决选择器的计算成本和漏选问题，与固定随机边不能直接互换。

## 线性复杂度的前提

若每个普通 query 访问至多 $w$ 个局部、$r$ 个随机和 $g$ 个全局位置，同时这 $g$ 个全局 query 读取整段序列，那么单层 attention 主成本约为：

$$
O\bigl(n(w+r+g)d\bigr).
$$

这里 $n$ 是总序列长度，并把全局 query 的 $O(ngd)$ 成本计入。只有当 $w,r,g$ 和表示维度相对 $n$ 固定时，才得到关于 $n$ 的线性关系。增大窗口、增加全局位置、加入选择器，都会改变实际成本。

这个公式描述 attention 的主要配对运算，不代表整个 Transformer 没有投影、前馈层、通信和其他开销，也不保证处理更多上下文后任务精度不变。

## 为什么要以 block 为单位

在数学上，可以为每个 query 单独列出零散 key；但在图形处理器（Graphics Processing Unit，GPU）上，不规则 gather、细碎矩阵乘法和线程间同步可能抵消省下的运算。Block-sparse attention 因而将连续的 $b$ 个 token 组织为一个 block，再按 block 建立连接。

若 $n/b$ 个 query blocks 各访问至多 $k_b$ 个 key blocks，则配对数量的量级为：

$$
\frac nb\,k_b\,b^2=nk_bb.
$$

每个连接块内部仍计算 $b\times b$ 个 token 配对。这让内存访问和矩阵乘法更规整，代价是可能连带计算一些并不重要的 token 配对。

例如 $n=4096,b=64$ 时共有 64 个 blocks。如果一个普通 query block 连接 3 个局部块、3 个随机块和 2 个全局块，它最多访问 8 个 key blocks，而不是全部 64 个；重叠连接和边界会改变实际数量，全局 query blocks 仍需读取整个序列。BigBird 如何通过 reshape、roll 和随机块 gather 实现这一计算，详见 [[sources/papers/2020-big-bird#Block-sparse 实现|BigBird 的块稀疏实现解析]]。

**把一块 token 打包计算不等于把它们压成一个 KV entry。** 前者保留逐 token 的内容，改变连接和计算布局；后者改变历史表示的粒度，属于 [[architecture/attention/hybrid-attention|压缩与混合 attention]] 的问题。

## 稀疏 mask 与实际稀疏执行

如果先计算完整 $QK^T$，再把某些位置置为负无穷，逻辑上限制了可见范围，却没有省下此前的全量矩阵乘法；完整矩阵若已分配，二次显存开销也仍然存在。获得稀疏收益需要 kernel 真正跳过无效位置或 blocks。

Longformer 的窗口实现就展示了这种取舍：逐对角线循环内存效率较好但很慢，重叠 chunk 的稠密乘法会多算一些元素却更实用，自定义 kernel 则进一步适配带状与扩张窗口。机制细节和实现代价在 [[sources/papers/2020-longformer#稀疏模式如何在硬件上执行|Longformer 原文解析]]中展开。

[[inference/attention-acceleration/flash-attention|FlashAttention]] 处理的是执行中的显存读写与片上复用，可以与稀疏图组合。原始论文中的 block-sparse 扩展跳过没有连接的块，在已有连接上仍使用分块 softmax。它相对于指定稀疏 mask 计算 attention，并不等同于未屏蔽的 full attention；相关 IO 分析见 [[sources/papers/2022-flashattention|FlashAttention 原文解析]]。

## 信息传播与表达能力的边界

稀疏图上的“可达”只表示存在传播路径，不保证有限维表示能保留沿途所有细节。全局节点数量很少时，许多远程信息通过少数向量汇聚，任务仍可能受到表示容量和层数限制。

BigBird 的理论分析说明，具有合适全局结构的网络族可以保留通用逼近能力；同一论文也证明，在特定复杂度假设和网络规模约束下，有些 full attention 容易处理的任务需要稀疏网络增加很多层。它没有证明任意稀疏 mask、固定宽度和固定深度都与 dense 模型等价。

## 双向编码器与因果解码器

Longformer 和 BigBird 的主要文本实验包含双向编码器。把它们用于自回归生成时，所有局部、随机、全局连接都必须满足因果约束，任何作为中转的状态也不能已经读到未来信息。

纯因果窗口可以只保留所需的近期 K/V；如果随机边、全局层或动态选择仍可能读取很久以前的位置，窗口外的 K/V 就不能一概删除。因此“每次读取更少位置”与“长期存储更少历史”必须分别分析。

## 相关知识链接

- [[architecture/attention/attention|Attention 机制]]
- [[architecture/attention/sliding-window-attention|Sliding Window Attention]]
- [[architecture/attention/hybrid-attention|Hybrid Attention]]
- [[inference/attention-acceleration/attention-kernels|Attention Kernels]]
- [[inference/kv-cache-and-memory/kv-cache|KV Cache]]
