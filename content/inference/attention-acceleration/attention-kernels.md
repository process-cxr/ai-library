---
title: Attention Kernels
created: 2026-04-05
published: 2026-04-05
modified: 2026-10-20
type: topic
status: growing
area: inference
tags:
  - attention
  - kernel
  - gpu
---

Attention kernel 将注意力的计算规则映射到 GPU（Graphics Processing Unit，图形处理器）的矩阵单元、线程、片上存储与显存。它优化的不只是矩阵乘法，还包括数据搬运、归约、候选选择和中间结果管理。

## 先区分数学目标与执行方式

理解一个 attention 优化时，需要先明确它与什么结果保持等价：

- **Dense attention 的精确实现**：保持完整 scaled dot-product attention 的数学目标，例如 [[inference/attention-acceleration/flash-attention|FlashAttention]]，主要通过分块、online softmax 和片上复用减少中间矩阵读写。
- **稀疏 attention 的结构设计**：改变每个 query 实际访问的位置，例如 [[architecture/attention/sliding-window-attention|Sliding Window Attention]]。与 dense attention 的计算结果通常不同。
- **固定稀疏规则的精确实现**：模型已经定义评分与 Top-K 路由，kernel 在保持该规则的前提下减少执行成本。这里的精确性是相对既定 scorer，而非相对 dense attention。

同一个模型可以同时使用结构稀疏化和 kernel 优化。前者决定允许哪些交互，后者决定这些交互及其路由怎样高效执行。

## 三类主要成本

### 算术与线程利用率

矩阵单元吞吐高，并不代表所有操作都能同样快地完成。softmax、比较、排序和不规则索引等操作可能主要由普通线程处理。块大小、线程分工、寄存器压力以及可并行的 query 数，会共同影响利用率。

### 存储访问

HBM（High Bandwidth Memory，高带宽显存）与片上存储的容量、带宽不同。把完整 score matrix 写到显存再读回，可能比在片上分块累积更贵。tiling、加载复用、异步流水和 kernel fusion 都在减少等待与往返搬运，但需要与具体数据布局匹配。

### 路由与中间结果

稀疏 attention 可能只对 $K$ 个位置执行最终聚合，但为了找到这 $K$ 个位置，indexer 仍可能扫描全部历史 keys。完整评分、Top-K 选择和候选写回都属于真实成本，不能只计算最终稀疏聚合的运算量。

## 用合法上界避免不必要的完整评分

对某个 query，设候选 key $i$ 的原始分数为 $s_i$。先完整评分至少 $K$ 个互不重复的可见 keys，其第 $K$ 大分数 $\tau$ 是全局 Top-K 门槛的下界。

如果另一个候选具有合法上界 $U_i\ge s_i$，并且：

$$
U_i<\tau,
$$

那么它不可能进入最终 Top-K。执行系统可以先计算便宜的 bound，只对幸存候选补算完整分数，再用原始分数完成精确选择。

[[sources/papers/2026-dosa-dots-sparse-attention|DoSA]] 将这一原则用于带正、负权重的多头 ReLU indexer：正头贡献需要上界，负头扣分需要下界；二者合成总分上界。采样与几何启发式可以影响 bound 的松紧，但不应成为“不会漏掉赢家”的唯一依据。

这类剪枝必须满足几个条件：

1. bound 比完整评分便宜，而且足够多候选可以被排除；否则先筛再补算反而增加工作。
2. 幸存候选重新使用原始 scorer，不能拿上界作为最终分数。
3. 门槛相等的候选继续保留；要求相同 token 标识时，还需一致的并列排序规则。
4. 浮点实现必须对实际参考分数保持保守上界；实数不等式不能自动保证低精度 kernel 不误删。
5. 候选溢出时必须精确合并、扩容或完整回退，不能把 buffer 容量当作候选数量的数学上界。

高剪枝率也不等于只读取少量 keys。如果 screening 仍扫描所有 keys，它减少的是后续完整评分，不一定改变 key 扫描的渐近复杂度，更不直接缩小 [[inference/kv-cache-and-memory/kv-cache|KV Cache]]。

## Prefill 与 Decode 需要分别设计

Prefill 同时处理大量 queries，适合把工作组织成规则矩阵块，在多个 query 之间共享 key 加载。先筛选再稀疏补算时，要控制候选不规则性带来的额外加载与调度成本。

Decode 每步 query 较少，历史 cache 读取和选择更容易成为瓶颈。前一生成步选中的位置可以帮助构造当前步的候选门槛，但如果要保证精确，必须按当前 query 重算其分数，不能直接复用旧分数。

因此，同一优化在 prefill 与 decode 上可以使用不同路径，二者的加速比也不能直接互相外推。

## 性能评测需要保留哪些口径

- 测量范围是已准备好的 queries/keys 上的 kernel，还是包含准备与 cache 访问的完整 indexer。
- 端到端测的是 TTFT（Time to First Token，首 token 延迟）、生成阶段延迟，还是并发吞吐。
- 是否清空 prefix cache，batch、上下文长度、query chunk 和并行度是否相同。
- 是否计入筛选准备、候选压紧、工作区、容量溢出与异常回退。
- 数值一致性以什么算术、mask 和 tie-breaking 规则作为参照。

kernel 局部速度比不能直接当作整模型速度比；高剪枝率也不能直接当作整体计算量或延迟下降比例。

## 相关知识与论文

- [[inference/attention-acceleration/flash-attention|FlashAttention]]
- [[inference/attention-acceleration/flash-decoding|FlashDecoding]]
- [[inference/performance/ttft-tpot|TTFT / TPOT]]
- [[sources/papers/2026-dosa-dots-sparse-attention|DoSA: Dots Sparse Attention]]
