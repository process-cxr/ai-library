---
title: "DoSA: Dots Sparse Attention"
created: 2026-10-20
published: 2026-10-20
modified: 2026-10-20
type: source
status: processed
source_type: paper
area: sources
tags:
  - source
  - paper
  - sparse-attention
  - attention-indexer
  - inference
  - gpu
  - long-context
aliases:
  - DoSA
  - Dots Sparse Attention
source_url: https://arxiv.org/abs/2610.13990
paper_date: "2026-10"
paper_order: "13990"
---

# DoSA: Dots Sparse Attention

## 基本信息

- 标题：[DoSA: Dots Sparse Attention](https://arxiv.org/abs/2610.13990)
- 作者：Fanxu Meng、Taosong Fang
- 机构/来源：Dots Studio，Xiaohongshu Inc.（小红书）
- 发表：arXiv:2610.13990v1
- 日期：2026-10
- 代码/数据：[flash-indexer](https://github.com/llm-sys/flash-indexer)
- 相关主题：[[inference/attention-acceleration/attention-kernels|Attention Kernels]]、[[architecture/attention/attention|Attention]]、[[architecture/attention/hybrid-attention|Hybrid Attention]]、[[inference/kv-cache-and-memory/kv-cache|KV Cache]]、[[inference/performance/ttft-tpot|TTFT / TPOT]]

## 核心问题与结论

**稀疏 attention 只读取少量位置，并不意味着找到这些位置也只需要少量计算。** DeepSeek Sparse Attention（DSA，DeepSeek 稀疏注意力）使用多头 indexer 为历史 token 打分，再选出分数最高的 $K$ 个位置，即 Top-K。如果 indexer 仍然让每个 query 的全部索引头与所有历史 key 做点积，稀疏 attention 内部就保留了一次密集扫描。

DoSA（Dots Sparse Attention）优化的正是这一步。它不重新训练模型，也不改 indexer 的原始评分规则，而是为每个候选 token 构造一个**不会低估其真实分数的上界**。只要上界已经低于可靠的入选门槛，这个 token 就不可能进入最终 Top-K，无须计算它的完整多头分数。

论文的主要结论分成三个层次：

1. **算法层面**：借助 seed threshold、正权重头的投影上界和负权重头的聚合下界，可以安全排除一批候选，并在满足数值与存储条件时恢复原始 scorer 的精确 Top-K。
2. **系统层面**：筛选必须配合候选压紧、共享 key 加载和稀疏写回，才能把少算的点积转化为实际延迟收益。
3. **实验层面**：完整 indexer 相对 SGLang 最高加速 $2.74\times$；约 1M token 的单请求端到端实验中，GLM-5.2 和 DeepSeek-V3.2 的首 token 延迟分别获得 $1.86\times$ 和 $1.94\times$ 加速（§4.2、§4.3）。

这里的“精确”以**模型原有的稀疏 indexer**为参照，不是与完整 dense attention 等价，也不是保证生成文本逐 token 完全一致。

## 为什么 Indexer 会成为瓶颈

DSA 的执行路径可以简化为：

```text
hidden states
  -> prepare indexer queries / keys
  -> score visible keys with all indexer heads
  -> select Top-K token IDs
  -> run attention over selected positions
```

最后一步虽然只访问少量位置，前面的评分与选择仍可能很贵。论文对约 1M token 输入的 profiling 显示：

| 模型 | Indexer 占 prefill GPU 时间 | Indexer 占理论矩阵运算量 |
|---|---:|---:|
| GLM-5.2 | 73.0% | 47.4% |
| DeepSeek-V3.2 | 90.8% | 83.2% |

GPU 指 Graphics Processing Unit（图形处理器）；这里的时间占比统计 query/key 准备、key cache 访问、评分和选择，分母是完整模型的 prefill GPU 时间，不包含中央处理器侧的分词与请求传输。矩阵运算量按 FLOPs（Floating-Point Operations，浮点运算次数）统计，不包含 Top-K 比较和逐元素操作。因此，这两列并不是同一个指标（Figure 1、Appendix D.1）。

已有融合 kernel 可以减少中间分数写回和 Top-K 开销，但若每个 key 仍要计算全部索引头，就没有消除评分本身的算术成本。DoSA 在融合之外增加一层 certified pruning，即有正确性依据的提前排除。

用上界与 Top-K 门槛跳过无效评分的思想，在经典文档检索中已有基础。本文的新工作集中在带符号、带 ReLU 的多头 indexer：怎样低成本构造足够紧的逐 key 上界，并将其映射到 GPU，而不是首次提出“上界低于门槛就可排除”这一原则（§5）。

## 方法：怎样少算分数而不丢掉赢家

### 1. 原始 Scorer 是带符号的多头 ReLU 求和

对固定 query，设索引头数为 $H$，第 $h$ 个 query 向量为 $q_h\in\mathbb R^D$，权重为 $w_h\in\mathbb R$，历史 token $i$ 的 key 为 $k_i\in\mathbb R^D$。定义 ReLU（Rectified Linear Unit，修正线性单元）为 $[x]_+=\max(x,0)$，原始分数为：

$$
s_i=\sum_{h=1}^{H}w_h[q_h^\top k_i]_+,
\qquad i\in\Omega.
$$

$\Omega$ 只包含 causal mask 等规则允许访问的位置。这里的 $q_h$、$k_i$ 属于 indexer，不应与后续 attention 聚合使用的所有 query/key 表示混为一谈。

**ReLU 输出非负，但 $w_h$ 可以为负。** 因而有些头提高总分，有些头降低总分。这决定了哪些省略是安全的：

- 直接去掉正权重头，会低估分数，可能误删真正的 Top-K。
- 去掉负权重头，会抬高分数，可以形成安全但往往很松的上界。

DoSA 的难点不是近似一个“看起来相关”的分数，而是用低于完整评分的成本，构造足够紧的合法上界。

### 2. Seed 提供真实 Top-K 门槛的下界

先取至少 $K$ 个**互不重复且可见**的 seed keys，构成 $S\subseteq\Omega$。对它们用原始 scorer 完整打分，把第 $K$ 大的分数记为：

$$
\tau=\tau_K(S).
$$

由于完整集合包含 seed，完整集合的第 $K$ 大分数不可能更低：

$$
\tau_K(S)\le\tau_K(\Omega).
$$

若某个 key 的真实分数满足 $s_i\le U_i$，且 $U_i<\tau$，那么它必然落在完整 Top-K 门槛以下，可以安全排除。seed 不需要覆盖最终赢家；即使 seed 选得不好，也只会使门槛偏低、筛选变弱，不会破坏这个推理。

一个独立的示意例子：取 $K=2$，seed 的分数为 8 和 6，因此 $\tau=6$。

| 新候选 | 评分上界 | 后续处理 |
|---|---:|---|
| A | 5 | 真实分数不可能达到 6，直接排除 |
| B | 9 | 上界不能证明落选；完整评分若只有 4，再排除 |
| C | 12 | 完整评分若为 10，保留参与最终选择 |

最后在 seed 和保留候选中重新选 Top-2，得到 10 和 8。上界只负责判断“是否还可能入选”，**不能拿来替代最终排序分数**。

### 3. 保留少量头，直接计算难以界定的贡献

设 $R$ 是直接计算的 retained heads 集合，大小为 $r<H$；剩余的正、负权重头分别记为 $P$、$N$。

论文按 query 构造 screening plan：

- 如果正权重头数不超过 $r$，先保留全部正权重头，再从负权重头中补足预算。
- 如果正权重头数超过 $r$，按 $w_h\lVert q_h\rVert_2$ 选最强的 $r$ 个正权重头，其余正头需要额外上界。
- 补选负权重头时，优先级为 $|w_h|\lVert q_h\rVert_2\min(p_h,1-p_h)$，其中 $p_h$ 是该头在采样 seed keys 上点积为正的频率。

最后一项倾向于直接计算“贡献幅度大、激活状态又难预测”的负头。具体选头规则影响速度与上界松紧；正确性证明并不要求该启发式一定找到最优集合（§2.2、Appendix B.2）。

### 4. 正权重头：复用相近头的点积，再补残差上界

对未直接计算的正头 $h\in P$，从 retained heads 中选一个 anchor $j_h$，构造非负投影：

$$
\beta_h=
\frac{[q_h^\top q_{j_h}]_+}{\lVert q_{j_h}\rVert_2^2},
\qquad
d_h=q_h-\beta_hq_{j_h}.
$$

anchor 按 $[q_h^\top q_j]_+/\lVert q_j\rVert_2$ 最大化选择；零范数向量按零系数处理。对任意 key：

$$
[q_h^\top k_i]_+
\le
\beta_h[q_{j_h}^\top k_i]_+
+\lVert d_h\rVert_2\lVert k_i\rVert_2.
$$

第一项复用已经计算的 anchor 点积；第二项用 Cauchy-Schwarz 不等式覆盖未知残差。这个 bound 不要求两个头完全相同，但方向越相近，残差越小，上界通常越紧。

把多个正头折叠进对应 anchor 后，定义：

$$
\widetilde w_j=w_j+
\sum_{h\in P:\,j_h=j}w_h\beta_h,
\qquad
E_P=\sum_{h\in P}w_h\lVert d_h\rVert_2.
$$

这些 folded weights 只属于当前 query 的筛选计划，不会改写模型参数。若所有正头都已保留，则 $P=\varnothing$、$E_P=0$，也不需要这一步投影（§2.3）。

附录还利用残差之间的几何关系收紧 $E_P$。设 $u_h=w_hd_h$，可用：

$$
E_P^{\mathrm{joint}}=
\frac12\left(
\left\lVert\sum_{h\in P}u_h\right\rVert_2
+\sqrt{\sum_{h,\ell\in P}|u_h^\top u_\ell|}
\right)
\le E_P.
$$

这避免了逐头残差范数相加的部分保守性，但 query 侧的准备成本会随遗漏正头数呈二次增长。它是上界构造的扩展，不代表所有 query 都执行这套计算；实验中需要正头投影的 query-layer 行只占 0.42%（Appendix A.1、D.2）。

### 5. 负权重头：对“应扣除多少分”给出下界

负权重头的真实扣分量是：

$$
\sum_{h\in N}|w_h|[q_h^\top k_i]_+.
$$

如果能低成本算出一个不超过真实扣分的量，从正项上界中减去它，仍是总分的合法上界。

对任意子集 $A\subseteq N$，先在 query 侧聚合向量：

$$
a_A=\sum_{h\in A}|w_h|q_h.
$$

利用 ReLU 的次可加性，有：

$$
[a_A^\top k_i]_+
\le
\sum_{h\in N}|w_h|[q_h^\top k_i]_+.
$$

这样多个负头的扣分贡献可由一个聚合方向界定。但不能简单认为“聚合越多负头，上界越紧”：若两个负头的加权点积为 $+10$ 和 $-9$，逐头 ReLU 后真实扣分为 10，先求和再 ReLU 却只得到 1。未激活头的负点积会在聚合前抵消有效信号。

DoSA 用 seed 中的激活频率构造 $A=\{h\in N:p_h\ge 1/2\}$，减少这类抵消。如果新 key 的活跃负头集合恰好等于 $A$，该聚合是精确的；否则仍合法，只是可能更松。**采样激活模式帮助提升筛选效率，不承担正确性保证**（§2.4、Appendix B.2）。

### 6. 最终 Certificate 与两次过滤

将上述部分合并，得到论文式 (9)：

$$
\boxed{
U_i=
\sum_{j\in R}\widetilde w_j[q_j^\top k_i]_+
+E_P\lVert k_i\rVert_2
-[a_A^\top k_i]_+,
\qquad s_i\le U_i.
}
$$

完整流程是：

```text
score distinct visible seed keys with the original scorer
  -> obtain threshold tau
  -> prepare retained heads and bound coefficients per query
  -> scan non-seed keys and compute upper bounds U_i
  -> reject only when U_i < tau
  -> score surviving keys with all original heads
  -> keep only original scores s_i >= tau
  -> exact Top-K over seed records and surviving records
```

第一遍过滤省掉完整评分；第二遍过滤减少写回与最终选择的输入规模。两个过滤都保留等号，避免丢失门槛处的并列候选。论文实现的 prefill 扫描使用固定 seed threshold，并不是每处理一个 key 都更新全局门槛（§3、Algorithm 1）。

## 为什么有些 Query 能剪掉很多，有些不能

设完整集合的真实第 $K$ 大分数为 $\tau^*$，定义：

$$
\Delta_i=\tau^*-s_i,
\qquad
b=\tau^*-\tau\ge0,
\qquad
\delta_i=U_i-s_i\ge0.
$$

$\Delta_i$ 是 key 与真实入选门槛的距离，$b$ 是 seed 门槛偏低的幅度，$\delta_i$ 是上界的松弛量。于是：

$$
U_i<\tau
\quad\Longleftrightarrow\quad
\Delta_i>\delta_i+b.
$$

因此，剪枝效果由三者共同决定：seed 是否给出足够高的门槛、certificate 是否足够紧、候选分数是否与 Top-K 边界充分分离。若大量 token 的分数挤在门槛附近，即使上界合法，也可能无法排除它们（Appendix B.3）。

筛选还要付出自己的计算成本。设初始化完整评分 $N_0$ 个 keys，筛选 $N_s$ 个 keys，其中 $M$ 个幸存；筛选需要 $r$ 个 retained rows 和 $d_a$ 个 auxiliary rows，则逻辑点积数量近似为：

$$
D_{\mathrm{base}}=H(N_0+N_s),
$$

$$
D_{\mathrm{DoSA}}=HN_0+(r+d_a)N_s+HM.
$$

令 $m=M/N_s$，筛选能节省点积的必要比较为：

$$
m<1-\frac{r+d_a}{H}.
$$

多保留一个头不一定更快：它虽然可能收紧上界，却增加了所有 key 的筛选成本。上述模型还未计入准备、内存访问、硬件 padding、候选管理和 recovery，不能直接当作延迟预测式。

## GPU 实现：从合法剪枝到实际加速

### 规则扫描与稀疏补算分离

直接对每个 query 独立处理不规则候选，会让矩阵单元利用率变差，并产生大量零散读取。DoSA 把执行分成规则 screening 和稀疏 refinement（Figure 2、Figure 3）：

1. 把 retained-head 预算相近的 queries 组织成固定宽度的块，使用 Tensor Core 矩阵单元计算筛选点积。
2. SIMT（Single Instruction, Multiple Threads，单指令多线程）线程从 TMEM（Tensor Memory，张量存储）取得累加结果，求上界、过滤并压紧候选标识。
3. 对幸存候选重新加载 keys，使用全部原始头完整评分。
4. 融合原始 ReLU、权重归约和门槛过滤，只写出仍有资格参与最终 Top-K 的记录。

query 侧的筛选向量和系数可以跨 key tiles 复用；异步加载和矩阵计算重叠，用流水减少等待。实现使用 FP8（8-bit Floating Point，8 位浮点数）辅助行，并额外处理其数值误差。

### 理论上的聚合方向与实际 FP8 行不是同一个数量

数学式中的负头 aggregate 是一个向量，但实现使用两个缩放的 FP8 auxiliary rows 近似表示它，再将表示误差计入保守上界。因此，基本实现的筛选成本是 $r+2$ 个逻辑点积，而不是简单的 $r+1$。

当所有正头都已直接保留时，还可以用 $[x]_+=(x+|x|)/2$ 改写上界。定义 $v=a_A-\frac12\sum_{j\in R}w_jq_j$，有：

$$
U_{\mathrm{abs}}(k)=
\frac12\sum_{j\in R}w_j|q_j^\top k|-v^\top k
=S_R(k)-a_A^\top k
\ge S_R(k)-[a_A^\top k]_+
\ge s(k).
$$

$S_R(k)$ 是 retained heads 的原始贡献。该变体额外增加 $[-a_A^\top k]_+$ 的松弛，但保持合法上界。32-head 实现的 $r=14,16$ 路径采用这个形式；$r=20$ 路径仍用带正头投影的基本形式。两类路径都使用 $r+2$ 个逻辑点积，幸存者都回到原始 $H$-head scorer（Appendix C.1）。

### 共享 Key，减少重复搬运

在四个 query 组成的组内，如果某个 key 至少对两个 query 幸存，而且位于组内共同可见的范围，它会进入 shared refinement stream，其余进入各 query 自己的 stream。

共享流让一次 key 加载服务多个 query，代价是可能多做一些 query-key 计算。这是在搬运成本和算术成本之间进行交换，不是所有被共享评分的位置都最终入选。

### 稀疏写回与紧凑选择

完整评分后，仅 $s_i\ge\tau$ 的记录进入 HBM（High Bandwidth Memory，高带宽显存）。记录包含紧凑的单调 score code 和精确 key 标识。

最终选择先根据 score code 缩小候选，再对边界 code 内的候选使用原始分数消歧。压缩 code 不能独立承担最后的精确排序；边界处必须恢复原始分数的选择语义（§3）。

### Decode 使用不同策略

Decode 每步 query 少，不容易像 prefill 那样跨 query 复用 key。论文因此保留全量 FP8 评分，只利用近期被选中的互不重复 keys 作为 witnesses：用**当前 query**重算这些 keys 的分数，建立合法门槛，再减少选择与写回工作。

这不是直接沿用上一 token 的分数，也不是在 decode 同样跳过约 97% 的完整多头评分。prefill 和 decode 的收益来源不同。

## 精确性保证的实际条件

### 实数上界不自动等于浮点上界

Algorithm 1 与 Theorem 1 的精确选择需要：seed 和候选使用同一原始 scorer、每个被删 key 都确实落在合法门槛以下、候选集合不丢数据。若要连并列项的 token 标识也完全相同，还需要与 baseline 一致的全序 tie-breaking 规则。

在 FP8 实现中，还必须区分理想实数分数 $s_i$ 和原始 kernel 实际计算的 $s_i^{\mathrm{ref}}$。仅证明 $s_i\le U_i$ 不够，还要有：

$$
s_i^{\mathrm{ref}}\le U_i^{\mathrm{num}}.
$$

Appendix C 把数值保护写成：

$$
U_i^{\mathrm{num}}=
\operatorname{Eval}_{\uparrow}
\left[
\alpha_i\left(\widehat U_i+\rho_i C+\varepsilon_{\mathrm{raw}}\right)
+\varepsilon_{\mathrm{out}}
\right].
$$

其中 $k_i=\alpha_i\bar k_i$，$\widehat U_i$ 是 screening 在 key payload 上算出的近似表达式；$\rho_i$ 必须保守覆盖 key 的 RMS（Root Mean Square，均方根）范数，即 $\rho_i\ge\lVert\bar k_i\rVert_2/\sqrt D$；$C$ 与两个 $\varepsilon$ 覆盖向量表示、权重形成和浮点计算误差。$\operatorname{Eval}_{\uparrow}$ 表示结果向上包围，而不是普通的最近舍入。

只有 $U_i^{\mathrm{num}}<\tau$ 才能安全删除。这里的数值保护是 correctness contract 的一部分，不是可以随意取消的“保险系数”。

### 缓冲区满了不能直接截断

最终只输出 $K$ 个位置，不代表中间幸存候选最多也只有 $K$ 个。若所有 key 都与门槛同分，所有 key 都需要保留到正确处理。

候选缓冲区不足时，必须分块执行精确 Top-K 合并、扩容，或回退到完整域计算。若 scale、范数摘要或 bound 出现无效值，也不能继续基于不可靠的上界排除候选。论文要求 query-level 错误触发 full-domain recovery，重新覆盖**所有可见 keys**，包括错误发现前已经被删除的位置；仅在残留 buffer 中重选并不能恢复精确性（Appendix C.3）。

### 精确 Top-K 的边界

- 有共同 tie order 时，可恢复与完整评分相同的 Top-K 标识。
- 没有共同 tie order 时，只保证返回原分数定义下合法的 Top-K 集合；并列边界可能选中不同位置。
- 因而不能仅凭该定理声称最终生成文本逐 token 一致，更不能据此证明超出模型训练范围的长上下文任务准确率。

## 实验设置与结果

### 比较条件

实验使用 GLM-5.2 与 DeepSeek-V3.2，分别有 32 和 64 个 indexer heads，固定 Top-$K=2048$。微基准在单张 NVIDIA GPU 上运行 GLM-5.2；端到端推理使用 TP8，即 Tensor Parallelism（张量并行）的并行度为 8。两模型的 prefill chunk 分别为 32K 和 16K tokens（§4.1）。

所读版本只写明 NVIDIA GPU，没有在实验配置中给出具体型号和显存容量，因此这些数值不能当作指定卡型上的可复现性能承诺。

输入来自 InfiniteBench 长上下文评测集中的英文、中文书籍与代码，额外使用 Project Gutenberg 书籍和 SQLite 源码延长上下文。这里的 1K 是 1024 tokens，约 1M 实际为 1,048,569 tokens（Appendix D.1）。

两个 baseline 是 SGLang v0.5.18 和 LiteTopK。前者先完整多头评分再做 Top-K；后者融合完整评分、候选过滤与选择，但仍完整计算所有头。完整 indexer 微基准包含 query/key preparation；单独的 layer-wise retrieval 测量则不包含，两类速度比不能混为同一指标。

### Prefill：60 个配置中均最快

上下文取 128K、256K、512K、768K、约 1M，query chunk 取 4K、8K、16K、32K，并覆盖三个输入域，共 60 个配置。每个配置重放最后一个 query chunk，统计全部 21 个产生索引的层（§4.2、Figure 4）。

| 统计范围 | 相对 SGLang 的完整 Indexer 加速 | 相对 LiteTopK 的完整 Indexer 加速 |
|---|---:|---:|
| 全部 60 个配置 | $1.12$-$2.74\times$ | $1.17$-$1.69\times$ |
| 约 1M 上下文的配置 | $2.47$-$2.74\times$ | $1.47$-$1.69\times$ |

所有 512K 及以上配置相对 SGLang 都超过 $2\times$。对英文、16K query chunk，速度比从 128K 时的 $1.19\times$ 增至约 1M 时的 $2.58\times$。

相比之下，约 1M 下把 query chunk 从 4K 放大到 32K，速度比只从 $2.47\times$ 变到 $2.69\times$。收益更明显地随 key 序列增长，而不是随 query batch 同比例增长，符合“每个 query 的准备工作在更多 keys 上复用”的机制解释。

### 剪枝率不是整体计算量降幅

剪枝率定义为 $1-N_{\mathrm{keep}}/N_{\mathrm{screen}}$，计数单位是 query-key pairs。三个输入域的平均剪枝率为 97.2%-97.5%；每个域的 21 层中有 20 层超过 95%，最低的 layer 1 仍为 89%-93%（Figure 6）。

这说明大多数配对不再进入 full-head refinement，但它们仍经历了 screening，仍需要扫描 key。因此，不能把 97% 剪枝率写成“消除了 97% 的 attention FLOPs”“只读 3% 的 cache”或“整体推理加速 30 多倍”。

### 端到端：约 1M 时接近减半 TTFT

TTFT（Time to First Token，首 token 延迟）按单独请求测量，每次请求之间清空 prefix cache。约 1M 上下文的结果为（§4.3、Figure 8）：

| 模型 | 相对 SGLang 的 TTFT 加速 | 相对 LiteTopK 的 TTFT 加速 |
|---|---:|---:|
| GLM-5.2 | $1.86\times$ | $1.26\times$ |
| DeepSeek-V3.2 | $1.94\times$ | $1.50\times$ |

Indexer 加速不能直接等同整模型加速。若 indexer 原占比为 $f$、自身加速为 $s$，Amdahl 定律给出理想整体速度比：

$$
S=\frac{1}{1-f+f/s}.
$$

剩余 attention、前馈网络以及请求链路仍然消耗时间。论文在超出 DeepSeek 模型声明上下文的长度上进行测量，只用于观察计算扩展行为，不构成长上下文任务质量证明（Appendix D.1）。

### Decode：提升较温和，且只报告 Indexer

decode 输入包括文档分析与 SQLite 代码审阅；文档请求使用 HELMET（How to Evaluate Long-Context Language Models Effectively and Thoroughly，长上下文语言模型综合评测）、InfiniteBench 和 LongBench v2 等评测套件构造，总体填充约 1M-token 的 KV（Key-Value，键值）Cache。

单请求长度为 256K、512K 和约 1M 时，decode indexer 相对 SGLang 的加速分别为 $1.06\times$、$1.11\times$、$1.26\times$（Figure 5）。这些是 indexer 结果，不是整模型 decode throughput 或每个输出 token 的延迟提升。

### 消融：负头聚合与稀疏写回都重要

第一组消融使用 LOFT（Long-Context Frontiers，长上下文任务评测基准）中的一个 SciFact 科学论断证据检索样本，设置 16K queries、896K keys，跨 21 个索引层平均 retrieval latency（§4.4、Figure 7）：

| 变体 | 开启的机制 | 相对 Base 的延迟下降 |
|---|---|---:|
| Base | 不筛选、不稀疏写回 | 基准 |
| Full | 只做稀疏写回，仍完整评分 | 23.4% |
| Dense | 只做 screening，仍使用 dense 中间写回 | 37.4% |
| DoSA | screening 与稀疏写回同时开启 | 47.3%，约 $1.90\times$ 加速 |

收益不是简单相加；计算与搬运成本相互影响。论文的 `Full`、`Dense` 是消融变体名，不能按字面理解成两种不同模型架构。

第二组消融更直接检验 certificate 的来源（Appendix D.2、Figure 9）：

- 去掉负头 aggregate 后，候选幸存率从 2.65% 升至 95.06%，indexer 时间达到 DoSA 的 $9.51\times$；该计时包括候选池满时的 recovery。
- 聚合所有遗漏负头，而不是按 activation 选择 support，幸存率升至 4.06%，延迟增加 12.0%。
- 只按 strength 选择 retained negative heads、移除正头投影的总体影响较小；正头投影只在 0.42% 的被测 query-layer 行中需要。

这组证据说明：在当前 workload 中，**有效估计负头扣分量是紧上界的关键**。不能因方法中正投影的数学推导较复杂，就把它当成实验收益的主要来源。

## 证据边界与未解决问题

### 仍然扫描所有 Keys

DoSA 减少 full-head scoring，不消除逐 key screening。固定 query 的计算仍随可见 key 数线性增长；对完整 prefill，仍存在二次规模的 query-key 扫描，不是一般意义上的线性 attention。

短上下文、较松的 bound、大量边界附近候选，以及较高的 certificate 准备成本，都可能压缩甚至抵消收益。论文将不读取每个 key 的 block-level bounds 列为后续方向，而非已经完成的能力（§6）。

### 候选工作区存在明显显存代价

Appendix D.4 披露，layer-wise retrieval 的 252 个单元中，247 个沿用原实验，5 个在候选存储容量修复后对三种方法匹配重测；修复后**没有重跑完整矩阵**。

在约 1M-key 容量下，修复后的配置为每个 query 预留 1024 个 candidate-ID pages，并预留 114,688 条输出记录，相对原分配额外增加 **9.94 GiB 工作区**。GiB（Gibibyte，吉比字节）是 $2^{30}$ bytes。这不是整个模型的显存占用，也不是自动可忽略的暂存开销。

在这一混合测量矩阵内，DoSA 相对 SGLang 在 251/252 个单元更快，相对 LiteTopK 在 250/252 个单元更快；各自最高达到 $3.74\times$、$2.08\times$。这些只覆盖准备好 queries/keys 后的 retrieval，不包含 query/key preparation，不能代替前述完整 indexer 的速度比。

容量修复与重测范围应与速度数字一起保留。它不直接否定其余实验，但完整修复后重跑、峰值显存与多请求并发下的表现，仍需要进一步证据。

### 性能结果不等于能力评测

论文建立的是计算与延迟收益，没有展示任务准确率提升，也没有证明超训练长度的可靠推理能力。数学定理可以支撑特定选择契约下的等价性，但不能替代完整后端上的数值一致性测试、异常恢复测试和任务级回归。

此外，单请求 TTFT、固定 chunk 的离线微基准、有限输入域，尚未覆盖高并发 serving 的吞吐、尾延迟、显存竞争，以及不同卡型和不同 indexer 几何分布。

## 这篇论文的重要性

DoSA 将“稀疏注意力选什么”与“为了选出这些位置必须算多少”区分开来：即使路由函数已经固定，执行系统仍可以利用 score bounds 跳过那些不可能改变选择结果的完整评分。

它也把三个通常分开讨论的问题连在一起：**上界决定哪些计算可以省，kernel 组织决定省下的计算是否有实际收益，数值与候选存储契约决定这些收益是否仍保持精确选择。** 理解这三层，比单独记住最高速度比更能解释方法的适用条件。

## 相关知识链接

- [[inference/attention-acceleration/attention-kernels|Attention Kernels]]
- [[inference/attention-acceleration/flash-attention|FlashAttention]]
- [[architecture/attention/attention|Attention]]
- [[architecture/attention/hybrid-attention|Hybrid Attention]]
- [[inference/kv-cache-and-memory/kv-cache|KV Cache]]
- [[inference/quantization/fp8|FP8]]
- [[inference/performance/ttft-tpot|TTFT / TPOT]]
