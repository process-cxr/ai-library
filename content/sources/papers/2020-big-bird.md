---
title: "Big Bird"
created: 2026-06-01
published: 2026-06-01
modified: 2026-10-20
type: source
status: processed
source_type: paper
area: sources
tags:
  - source
  - paper
  - sparse-attention
  - long-context
source_url: https://arxiv.org/abs/2007.14062
paper_date: "2020-07"
paper_order: "14062"
aliases:
  - BigBird
  - "Big Bird: Transformers for Longer Sequences"
---

## 基本信息

- 标题：[Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062)
- 作者：Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, Amr Ahmed
- 机构/来源：Google Research
- 发表：NeurIPS 2020；arXiv:2007.14062v2（2021-01-08 修订）
- 日期：2020-07
- 代码/数据：[论文所列官方代码入口](http://goo.gle/bigbird-transformer)
- 相关主题：[[architecture/attention/sparse-attention|Sparse and Block-Sparse Attention]]、[[architecture/attention/sliding-window-attention|Sliding Window Attention]]、[[architecture/attention/attention|Attention]]、[[training/mid-training/long-context-training|Long-context Training]]

## 研究问题

BigBird 将 full self-attention 看作一个有向完全图：每个 query 对每个 key 计算内积，产生二次规模的连接。它问的是：能否只保留线性数量的连接，同时保留局部依赖、远距离交互，以及 Transformer 的某些表达能力？

论文同时提供结构设计、理论结果和任务验证。这三层证据回答的问题不同：理论说明某类稀疏网络存在何种表达能力，实验说明特定模型与训练设置有效，硬件实现说明哪些稀疏连接可以转化为高效运算。

## Attention 图：局部、随机与全局连接

### 在可见邻域上计算 attention

令 $\mathcal N_i$ 为第 $i$ 个 query 的 key 邻域，$q_i,k_j,v_j$ 分别为查询、键和值向量，$d$ 为 query/key 的维度。忽略多头输出投影和 residual，单个 head 的信息聚合可写成：

$$
o_i=\sum_{j\in\mathcal N_i}
\frac{\exp(q_i^Tk_j/\sqrt d)}
{\sum_{t\in\mathcal N_i}\exp(q_i^Tk_t/\sqrt d)}v_j.
$$

BigBird 保留这种在可见集合上归一化的计算，但把 $\mathcal N_i$ 从所有位置改为多个集合的并集。它没有把被选中的多个历史 token 合并成压缩 KV entry；未连接的位置在本层没有被读取（§2、Figure 1）。

| 连接               | 作用                                     | 代价与边界                                   |
| ------------------ | ---------------------------------------- | -------------------------------------------- |
| Local window       | 保留词序附近的强依赖和局部聚类           | 单层不能直接覆盖远处证据                     |
| Random connections | 提供跨区间的额外连接，改善非局部传播路径 | 随机邻居并不保证与当前 query 语义相关        |
| Global positions   | 少量位置读取整段输入，其他位置也读取它们 | 全局状态是有限维的中转表示；数量不能无界增长 |

对于非全局位置，可以抽象为 $\mathcal N_i=\mathcal W_i\cup\mathcal R_i\cup\mathcal G$；全局 query 则访问所有位置。三类 key 合并成同一个可见集合后做 softmax，而不是把三次独立归一化的输出直接相加。

若局部、随机、全局规模分别为 $w,r,g$，单层 attention 主成本约为 $O(n(w+r+g)d)$。当 $w,r,g$ 与 head dimension $d$ 固定时，它关于序列长度 $n$ 线性。全局 query 自身读取所有位置的成本也包含在 $O(ngd)$ 中。

### ITC 与 ETC 不是同一个模型配置

BigBird-ITC（Internal Transformer Construction，内部全局位置构造）把已有输入位置设为 global；BigBird-ETC（Extended Transformer Construction，扩展全局位置构造）加入额外 global token，为整段、段落、句子、问题或候选答案建立信息槽位。

必须区分 §2 的一般“三类连接”设计和附录里的实际实验：Appendix E 的 Tables 8、12、13 明确把若干 **ETC 配置的随机连接数设为 0**，而 ITC 通常保留随机块。大型问答测试模型使用 ETC。因此不能把所有 BigBird 成绩都描述为随机连接带来的收益。

Appendix E.3 还指出，ETC 与 [[sources/papers/2020-longformer|Longformer]] 的区别包括 global-local 相对位置编码，以及用于全局表示训练的 Contrastive Predictive Coding（CPC，对比预测编码）式目标；两者不只是“是否多几个随机邻居”的差别。

## Block-sparse 实现

随机单个 token 的 gather 会导致不规则访存，即使非零元素少，GPU（Graphics Processing Unit，图形处理器）或 TPU（Tensor Processing Unit，张量处理器）也不一定更快。Appendix D 把连接模式提升到 block 粒度：每个 query block 访问若干相邻块、随机块和全局块。

设 block size 为 $b$，每个普通 query block 连接 $w_b$ 个局部块、$r_b$ 个随机块、$g_b$ 个全局块。共有约 $n/b$ 个 query blocks，每对连接块仍执行 $b\times b$ 的稠密 attention，配对数量约为：

$$
\frac nb(w_b+r_b+g_b)b^2
=nb(w_b+r_b+g_b).
$$

因此 block 内仍保留逐 token 的 K/V 与 attention 权重。**blockification 是批量组织连接，不是把整个 block 压成一个 token。** Appendix D 的工程实现利用 reshape、roll 和少量随机块 gather，把稀疏图对应的运算组织成紧凑的稠密张量乘法。

## 理论结果及其条件

### 通用逼近：保留某类函数表达能力

§3.2 的 Theorem 1 与 Appendix A 证明，包含特定 star graph 的稀疏 attention 网络，在允许相应位置表示、附加全局节点和足够网络深度时，能够在紧致定义域上以 $L^p$ 意义逼近连续序列函数。证明的关键是利用全局节点和多层 sparse shift 构造 contextual mapping，再通过前馈网络映射到输出。

该结论并不保证固定层数、固定宽度、有限训练预算下与 dense attention 一样好，也不要求所有实际配置都必须有随机连接。它说明满足条件的网络族具备表达能力，不能代替特定模型的性能实验。

### 图灵完备：理想化精度与额外步骤

§3.3 / Appendix B 采用任意精度数值等理想化条件，构造稀疏 encoder-decoder 模拟图灵机。稀疏版本需要额外中间步骤聚合历史信息，一个 decoding step 不再一一对应图灵机的一步。这个存在性结果不意味着实际有限精度模型可以执行任意长的正确计算。

### 稀疏化存在深度代价

§3.4 / Appendix C 给出反例：寻找每个单位向量的最远向量，在论文的构造中 full attention 可用一层处理；对每层仅有近线性连接、且 head 数和宽度受到规定限制的网络，在 Orthogonal Vectors Conjecture（OVC，正交向量猜想）成立时，可能需要接近线性数量的层数。

这是一项有复杂度假设的条件下界，说明稀疏图保留通用表达能力，并不等于保留 dense attention 对每个任务的浅层计算效率。

## 模型训练与实验配置

### 继续预训练

Appendix E.1 从公开 RoBERTa checkpoint 开始，使用 Books、CC-News、Stories、Wikipedia 做 Masked Language Modeling（MLM，掩码语言建模），掩码比例为 15%，训练长度为 4096；较长文档切段，较短文档拼接。

Table 8 的 base 模型都为 12 层、12 heads、hidden size 768，batch size 256，Adam 学习率 $10^{-4}$，warmup 10000 步后线性衰减。Table 9 的四类语料分别约为 1.0B、7.4B、7.7B、3.1B token，语料 token 总量不等于唯一 token 数或实际训练曝光量。

| Base 预训练配置（Table 8） | ITC | ETC |
| -------------------------- | --: | --: |
| Block size                 |  64 |  84 |
| Global token 数            | 128 | 256 |
| 局部窗口覆盖的 token 数    | 192 | 252 |
| 随机连接覆盖的 token 数    | 192 |   0 |

这里的 192 个随机 token 来自 3 个 64-token blocks，不能误读为 192 个随机块。large 模型增至 24 层、16 heads、hidden size 1024，并采用更大 batch；具体参数须按任务区分。

### 问答与分类

§4 的 base 开发集与 large 测试集分别见 Tables 2、3。下表仅列 large ETC 的测试结果，其中 F1 是精确率与召回率的调和平均，WikiHop 使用 accuracy：

| 测试任务与指标（Table 3）                        |        BigBird-ETC |
| ------------------------------------------------ | -----------------: |
| HotpotQA，answer / supporting-fact / joint F1    | 81.2 / 89.1 / 73.6 |
| Natural Questions，long-answer / short-answer F1 |        77.8 / 57.9 |
| TriviaQA，Full / Verified F1                     |        84.5 / 92.4 |
| WikiHop，accuracy                                |               82.3 |

这些是任务系统结果，包含任务 head、输入结构与微调策略。例如 Appendix E.2 对超过 4096 的 Natural Questions 文档仍使用 stride 2048 的滑窗，不是整篇无界读取；ETC 还使用任务定义的全局位置和关系标签。

文档分类见 Appendix E.4 / Table 15：arXiv micro-F1 从 RoBERTa 的 87.42 提升至 92.31，IMDb 从 $95.0\pm0.2$ 到 $95.2\pm0.2$。长文档上的收益更明显；短上下文或其他任务并非全面领先。

### 摘要：稀疏编码器与完整解码器

摘要实验仅在 encoder 使用 BigBird 稀疏 attention，decoder 保留完整 self-attention 和 encoder-decoder attention。base 版本从继续预训练后的 RoBERTa 初始化，large 版本从面向摘要的 Pegasus 初始化，并不能与 encoder-only 问答模型混用。

Table 4 的 BigBird-Pegasus 在 arXiv / PubMed / BigPatent 上的 ROUGE-1/2/L 分别为 46.63/19.02/41.77、46.32/20.65/42.33、60.64/42.46/50.01。ROUGE（Recall-Oriented Understudy for Gisting Evaluation）衡量摘要与参考摘要的文本重叠度，这些得分不直接度量事实正确性。

输入长度存在需保留的公开口径差异：§4 概述称使用 4096 序列，Longformer 的后续对比也将 BigBird 摘要标为 4K；但本版本 Appendix E.5 / Table 17 给 arXiv 等长摘要数据集列出的最大 encoder 长度是 3072、decoder 长度是 256。精确复现应核对所选 checkpoint 与代码配置，不能把全文统一改写为同一个长度。

### 基因组任务

§5 / Appendix F 将 DNA 序列做 Byte-Pair Encoding（BPE，字节对编码）分词，再预训练和微调。Table 5 的 MLM BPC（Bits Per Character，每字符平均负对数概率的比特数）从 512-token BERT 的 1.23 降至 4096-token BigBird 的 1.12。

Table 6 的启动子分类 F1 为 99.9；Appendix F.2 同时说明，负样本的生成与 MLM 预训练有相近特征，因此高分不能直接解释为真实世界生物学任务的通用效果。Table 7 的染色质预测也不是所有指标都更好，例如 DNase I sensitivity 的 AUC（Area Under the Curve，此处指受试者工作特征曲线下面积）略低于 DeepSEA。Appendix F.3 使用两个独立模型的集成评测，并保留按染色体划分的训练测试设置。

## 证据与局限

- §2 / Table 1 比较了 random、window、二者组合和 BERT，但它不是“所有 BigBird 变体都必须有随机连接”的证明；ETC 的实际配置与理论图需分别核对。
- BigBird 与 Longformer 的成绩差异还涉及预训练预算、语料、全局位置、位置编码、辅助目标和微调过程，不能单独归因给随机连接。
- block 数固定才有关于 $n$ 的线性成本；稀疏结构本身不保证 GPU 延迟按相同比例缩短。
- 通用逼近、图灵完备与条件下界分别描述不同性质，均不等于有限模型的无损替换保证。
- 文本任务主要验证双向编码器或稀疏编码器加完整解码器，不能直接等同于因果语言模型的长程检索与 KV Cache 策略。

## 关键结论

BigBird 的贡献是把 attention 设计推进到**连接图、表达能力与块稀疏实现的共同分析**：局部连接保留近邻结构，全局节点提供信息中转，随机连接为部分配置增加远程路径；实际收益还取决于训练方案和硬件执行。它解释了为什么线性数量的连接仍可能支撑丰富的函数表达，也明确展示了稀疏图可能付出的深度代价。

## 相关知识链接

- [[architecture/attention/sparse-attention|Sparse and Block-Sparse Attention]]
- [[architecture/attention/sliding-window-attention|Sliding Window Attention]]
- [[sources/papers/2020-longformer|Longformer]]
- [[inference/attention-acceleration/flash-attention|FlashAttention]]
