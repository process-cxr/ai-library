---
title: "Longformer"
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
  - long-context
  - sparse-attention
source_url: https://arxiv.org/abs/2004.05150
paper_date: "2020-04"
paper_order: "05150"
aliases:
  - "Longformer: The Long-Document Transformer"
---

## 基本信息

- 标题：[Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150)
- 作者：Iz Beltagy, Matthew E. Peters, Arman Cohan（共同一作）
- 机构/来源：Allen Institute for Artificial Intelligence
- 发表：arXiv:2004.05150v2（2020-12-02 修订）
- 日期：2020-04
- 代码/数据：[allenai/longformer](https://github.com/allenai/longformer)
- 相关主题：[[architecture/attention/sliding-window-attention|Sliding Window Attention]]、[[architecture/attention/sparse-attention|Sparse and Block-Sparse Attention]]、[[training/mid-training/long-context-training|Long-context Training]]、[[inference/attention-acceleration/attention-kernels|Attention Kernels]]

## 研究问题与方案范围

标准 Transformer 的全连接 self-attention 对长度为 $n$ 的序列计算约 $n^2$ 个 query-key 配对。BERT（Bidirectional Encoder Representations from Transformers，双向 Transformer 编码表示）类模型因此常把输入限制在 512 token；截断、分段后拼接或先检索再阅读虽然可行，却可能损失跨段信息并引入额外任务结构。

Longformer 将 attention 的连接图改为局部窗口与少量全局位置的组合，在窗口和全局位置数量固定时使连接数随序列长度线性增长。其关注点不仅是长序列语言建模，还包括把已有双向预训练模型迁移到长文档问答、分类和指代消解。

本版本包含三类不同产物，不能混为同一个训练配置：

| 产物                                                  | 初始化与任务                                  | Attention 配置                                                                                 |
| ----------------------------------------------------- | --------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| 字符级自回归模型                                      | 从头训练，预测下一个字符                      | 因果窗口；高层少量 heads 使用 dilation                                                         |
| Longformer-base / large 编码器                        | 从 RoBERTa 继续掩码预训练，随后做下游任务微调 | 预训练为非扩张双向窗口；下游按任务配置全局位置                                                 |
| Longformer-Encoder-Decoder（LED，长文档编码器解码器） | 从 BART 初始化，做摘要生成                    | 编码器使用局部加全局 attention；解码器保留完整因果 self-attention 和 encoder-decoder attention |

## Attention 机制

### 局部窗口：从直接全连接到层间传播

在双向模式下，每个位置关注左右各约 $w/2$ 个 token。这里 $w$ 是两侧合计的窗口宽度，边界和是否计入当前位置只影响常数。对每个 head，attention 算术量约为 $O(nwd)$，其中 $d$ 是 head dimension。

局部窗口限制单层能直接读取的位置，多层仍能逐步传递远处信息。论文以 $\ell$ 层、固定窗口估计感受野跨度约为 $\ell w$。这表示存在信息传播路径，不保证任意远程事实都能无损保留，也不等于每层直接访问整篇文档（§3.1，Figure 2）。因果模式则把窗口限制在历史一侧，详见 [[architecture/attention/sliding-window-attention|滑动窗口注意力]]。

### Dilated sliding window：扩大跨度，保留局部 heads

扩张窗口在被访问的位置之间留出间隔，在配对数量不变时扩大可覆盖的距离。论文在字符语言模型中让低层主要建模局部依赖，高层使用更大的窗口，并仅在少量 heads 上加入 dilation，兼顾近邻信息与远程信息。

这一配置不直接用于 RoBERTa 迁移实验：§5 脚注明确指出，向迁移模型加入 dilation 会损害效果，作者推测与已有权重不兼容。因此“Longformer 支持 dilation”与“所有 Longformer 配置都使用 dilation”是不同结论。

### 全局位置：少量双向信息中转点

纯窗口难以让问句、分类标记等位置直接聚合整段输入。Longformer 因而为少量预选位置配置全局连接：全局位置读取整个序列，普通位置也读取这些全局位置。

设全局位置集合为 $\mathcal G$，局部窗口为 $\mathcal W_i$，第 $i$ 个 query 的可见集合可写为：

$$
\mathcal N_i=
\begin{cases}
\{1,\ldots,n\}, & i\in\mathcal G,\\
\mathcal W_i\cup\mathcal G, & i\notin\mathcal G.
\end{cases}
$$

非零配对数约为 $O(nw+ng)$，其中 $g=|\mathcal G|$。线性结论要求 $w$ 和 $g$ 不随 $n$ 增长。全局 token 的角色通常由任务规定：分类使用 `[CLS]`，TriviaQA 使用问题 token，WikiHop 还包括候选答案；指代消解实验未使用全局 attention（§3.1、§6）。

若某一层的全局状态刚汇总了远程信息，普通位置通常要到下一层才能读取该更新后的全局状态。图上的两跳连通不等于同一层内完成两次依次聚合。

### 全局与局部使用不同投影

论文为 query、key、value（查询、键、值，记为 Q、K、V）区分局部投影 $Q_s,K_s,V_s$ 与全局投影 $Q_g,K_g,V_g$，后者从前者复制初始化。它们允许局部上下文建模和全局聚合采用不同参数。WikiHop 消融表明，这个设计影响实际效果，不能把 Longformer 简化为只改一个二值 mask（§3.1、Table 10）。

## 从 RoBERTa 迁移到长文档

编码器保留 RoBERTa 的主体结构，将 attention 换成宽度 512 的局部窗口，把可学习绝对位置 embedding 从 512 扩展到 4096。新增位置以重复复制原位置 embedding 的方式初始化，再继续做 Masked Language Modeling（MLM，掩码语言建模）。重复复制仅提供初始化，不代表模型在适配前已经可靠理解更长位置。

§5 的继续预训练配置是序列长度 4096、batch size 64、65K 次更新、最大学习率 $3\times10^{-5}$、500 步 warmup，随后使用三次多项式衰减。Appendix C / Table 13 的语料包括 Books、英文 Wikipedia、筛选的 Realnews 和 Stories，报告的语料 token 数分别约为 0.5B、2.1B、1.8B、2.1B；这些是语料规模，不是去重后规模或实际训练曝光量。

Table 5 以 Bits Per Character（BPC，每字符平均负对数概率的比特数，越低越好）展示适配过程：

| Longformer-base 状态             | MLM BPC |
| -------------------------------- | ------: |
| 随机初始化新增位置               |  10.299 |
| 复制原位置 embedding             |   1.957 |
| 再继续预训练 2K 步               |   1.753 |
| 再继续预训练至 65K 步            |   1.705 |
| 仅训练新增位置、冻结原权重的方案 |   1.850 |

这支持“初始化和主体权重适配都很重要”，但与只接受 512 token 的 RoBERTa 直接比较时，上下文和训练阶段也同时变化。

## 稀疏模式如何在硬件上执行

只在完整 $n\times n$ score 矩阵上加 mask，并不会减少此前已完成的矩阵乘法。论文实现的是在图形处理器（Graphics Processing Unit，GPU）上仅计算所需带状区域的 kernel，具体比较了三种方式（§3.2、Appendix A）：

- **loop**：逐条对角线计算，显存效率较好但速度太慢，主要用于测试。
- **chunks**：让重叠 Q/K chunks 做稠密矩阵乘法，再屏蔽不需要的区域。它仅支持非扩张窗口，计算了部分额外元素，attention 中间存储约为理想实现的两倍，但更适合预训练与微调。
- **custom CUDA**：通过 TVM 编译生成带状乘法 kernel，支持 dilation 和因果模式，主要用于字符语言建模。

这是 [[architecture/attention/sparse-attention|块稀疏实现]] 的重要动机：GPU 上减少配对数量与实际加速之间，还隔着内存布局、矩阵乘法效率和 kernel 调度。

## 实验、消融与归因

### 字符级语言建模

§4 的 text8/enwik8 小模型为约 41M 参数，分别达到测试 BPC 1.10 和 1.00（Table 2）；102M 大模型在 enwik8 上为 0.99（Table 3）。大模型没有超过所有更大的对比模型，不能把小模型规模内的领先解释为全面领先。

训练分五阶段逐步增加窗口和序列长度；正文报告从 2048 训练至最后阶段的 23040 字符，评测采用长度 32256 的重叠序列，仅计最后 512 字符的结果。这里的字符数不能与编码器的 4096 subword token 混用。Table 4 的局部消融在 text8 小模型第一阶段训练 150K 步后比较：窗口随层数增大优于固定或减小窗口，两个 heads 加 dilation 有小幅收益；作者注明早期排序不必等于完整训练后的排序。

### 双向编码器下游任务

Table 7 比较 base 模型的开发集结果。F1 是精确率与召回率的调和平均；不同任务的指标和聚合方式不同，不能跨列求统一收益：

| 任务与指标            | RoBERTa-base | Longformer-base |
| --------------------- | -----------: | --------------: |
| WikiHop，accuracy     |         72.4 |            75.0 |
| TriviaQA，F1          |         74.3 |            75.2 |
| HotpotQA，joint F1    |         63.5 |            64.4 |
| OntoNotes，average F1 |         78.4 |            78.6 |
| IMDB，accuracy        |         95.3 |            95.7 |
| Hyperpartisan，F1     |         87.4 |            94.8 |

WikiHop 和 Hyperpartisan 的收益较明显；短文较多或证据主要为局部信息的任务收益较小。Hyperpartisan 只有 645 篇文档，采用随机 80/10/10 划分并报告五个种子的均值，不能视为大规模通用验证。

Table 10 的 WikiHop-base 消融统一以五轮微调的 73.8 accuracy 为参照：

| 改动                                    | Accuracy | 相对参照 |
| --------------------------------------- | -------: | -------: |
| 训练至 15 epochs                        |     75.0 |     +1.2 |
| 序列缩短至 2048                         |     73.1 |     -0.7 |
| 不做额外 MLM 预训练                     |     73.2 |     -0.6 |
| 不使用独立全局投影                      |     72.2 |     -1.6 |
| 同时去掉独立全局投影和 global attention |     65.5 |     -8.3 |

最后一行是**联合移除两个设计**的结果，不能把 -8.3 全部当作仅移除 global attention 的孤立贡献。Table 7 的 75.0 又使用了更长微调，不能直接与消融的五轮基线混合。

### LED 摘要生成

§7 中 LED 从 BART 初始化，编码器窗口为 1024，并将首个 `<s>` 设为全局位置；位置 embedding 复制扩展到 16384。解码器仍读取所有编码位置并对历史输出做因果 attention。

Table 11 的 arXiv 摘要结果以 ROUGE（Recall-Oriented Understudy for Gisting Evaluation，摘要与参考文本重叠度）衡量：LED-large 在 4096 输入时 ROUGE-1/2/L 为 44.40/17.94/39.76，在 16384 输入时为 46.63/19.62/41.83。该模型从 BART 初始化后直接做任务训练，没有额外的 LED 预训练；与从 Pegasus 初始化的 BigBird 摘要模型相比，输入长度和预训练来源均不同。

## 结论与适用边界

Longformer 的稳定结论是：**局部窗口处理邻近依赖，少量全局位置缩短远程信息路径，并且已有双向预训练权重可以通过继续训练适配这一稀疏结构。** 它为后续长文档编码器与局部/全局混合 attention 提供了可复用设计。

- 线性 attention 成本依赖窗口和全局数量固定；将大量位置都设为 global 会削弱节省效果。
- 输入超过模型长度仍需截断或分段，论文部分任务也保留了多段聚合；HotpotQA 使用两阶段段落筛选与答案预测。
- 编码器实验主要是双向阅读，不能直接视为现代 decoder-only 模型的 KV Cache 管理方案。
- LED 的 encoder self-attention 可随输入线性扩展，但完整生成系统仍包含解码器 self-attention 和长输入 cross-attention。
- 与 BigBird 的跨论文成绩不是纯 attention 模式消融；本版本脚注 9 也指出两者继续预训练计算预算存在差异。

## 相关知识链接

- [[architecture/attention/sliding-window-attention|Sliding Window Attention]]
- [[architecture/attention/sparse-attention|Sparse and Block-Sparse Attention]]
- [[sources/papers/2020-big-bird|Big Bird]]
- [[inference/attention-acceleration/flash-attention|FlashAttention]]
