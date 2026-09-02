---
title: "Scaling Laws for Neural Language Models"
created: 2026-05-31
published: 2026-08-31
modified: 2026-08-31
type: source
status: processed
source_type: paper
area: sources
tags:
  - source
  - paper
  - scaling-law
  - compute
  - pretraining
source_url: https://arxiv.org/abs/2001.08361
paper_date: "2020-01"
paper_order: "08361"
---

# Scaling Laws for Neural Language Models

## 基本信息

- 标题：[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- 版本：arXiv:2001.08361v1，2020-01-23
- 作者：Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei
- 研究对象：decoder-only Transformer language model
- 训练数据：WebText2
- 核心指标：token-level cross-entropy loss
- 相关 topic：[[training/scaling/scaling-law|Scaling Law]]，[[training/scaling/model-data-compute|Model Data and Compute]]，[[training/scaling/compute-optimal|Compute Optimal]]

这篇论文研究的不是某一个模型结构或训练技巧，而是一个更基础的训练规划问题：**当模型参数量、训练数据和训练计算量不断增加时，language modeling loss 会按照什么规律变化？在固定 compute 下，模型和数据应该如何分配？**

论文通过大量不同规模的 Transformer 实验，发现 validation / test loss 在很宽的范围内近似服从 power law。模型规模、数据规模和 compute 都是主要的 scaling 轴；在合理范围内，depth、width 和 attention heads 等 model shape 对 loss 的影响相对小。论文还把这些经验规律延伸到 overfitting、training time、batch size、跨数据分布 transfer 和 compute-efficient allocation，试图把 scaling law 从“画出一条趋势线”推进为可用于训练决策的预测框架。

这篇论文最重要的历史作用，是确立了早期大模型 scaling 研究的基本语言：**先用小规模实验拟合 loss 随 $N$、$D$ 和 $C$ 的变化，再据此外推更大训练的收益与预算分配。**它也是理解后来 Chinchilla 重新估计 model-data allocation 的必要基线。

## 研究问题

### 模型规模、数据规模与 compute 如何共同决定 loss

对 autoregressive language model 而言，训练目标是根据前面的 token 预测下一个 token。论文关注的主要性能指标是平均 token-level cross-entropy loss：

$$
L=-\mathbb{E}_{x_t\sim p^*}\log p_\theta(x_t\mid x_{<t})
$$

论文希望回答：

- 增加 non-embedding parameters $N$，loss 如何变化？
- 增加 dataset tokens $D$，loss 如何变化？
- 增加 training compute $C$，loss 如何变化？
- 这三者是否可以用统一的经验函数描述？
- 当 $N$ 和 $D$ 同时增加时，什么时候会出现 data-limited 或 model-limited？

### 固定 compute 时，应该选择更大的模型还是更多的数据

训练计算可以粗略表示为：

$$
C\approx 6NBS
$$

其中 $B$ 是以 token 计的 batch size，$S$ 是 parameter update steps。固定 $C$ 后，扩大模型会减少每一步能够处理的 token 数，增加数据则可能需要选择更小的模型。论文试图在这条约束下寻找 loss 最低的 model size、batch size、training steps 和 dataset size。

这使问题从“模型越大是否越好”变成了一个资源分配问题：**参数、数据、串行训练步数和并行 batch 之间的边际收益如何平衡？**

### 如何区分规模收益与训练配置收益

如果不同实验同时改变模型形状、tokenizer、learning-rate schedule、batch size 或 optimizer，那么 loss 差异不能直接归因于参数量或 compute。因此，论文还考察：

- model shape 在固定参数量下是否重要；
- training horizon 和 learning-rate schedule 是否会改变比较结果；
- batch size 是否应该随 loss 动态变化；
- 在训练分布之外，loss 是否仍然随模型规模改善。

## 核心主张

论文的主要结论可以压缩为以下几组经验规律。

1. **Loss 随模型规模、数据规模和 compute 近似按 power law 改善。** 这些趋势跨越多个数量级，在论文实验范围内没有明显拐点。
2. **Model scale 比 model shape 更重要。** 在固定 non-embedding parameter count 的合理范围内，改变 depth、width、attention heads 和 feed-forward ratio，loss 只发生较小变化；极浅或极端 depth-to-width ratio 的模型会偏离这一趋势。
3. **Overfitting 主要由 model size 与 dataset size 的组合决定。** 论文拟合的 overfitting penalty 主要取决于 $N^{\alpha_N/\alpha_D}/D\approx N^{0.74}/D$。模型扩大 8 倍时，数据量增加约 5 倍，才大致能保持相近的 overfitting 水平。
4. **大模型具有更高的 sample efficiency。** 在达到相同 loss 时，大模型需要更少的 optimization steps 和更少的 data examples。
5. **Critical batch size 主要随当前 loss 变化，而不是直接随模型参数量变化。** 当 loss 降低、gradient noise scale 增大时，适合的 batch size 也应增大。
6. **Compute-efficient training 倾向于更大的模型、较少的串行 steps 和较大的 batch。** Kaplan-style 拟合得到 $N_{opt}\propto C^{0.73}$、$B\propto C^{0.24}$、$S\propto C^{0.03}$，即更多 compute 主要投入模型规模，而不是显著延长训练时间。
7. **跨数据分布的 transfer loss 与训练分布 loss 高度相关。** 在论文测试的数据分布上，分布外 loss 大致等于训练分布 loss 加上相对稳定的 offset。

这些结论都属于特定模型族、数据集、tokenizer 和 optimization regime 下的经验拟合，不应被当作脱离条件的物理定律。尤其是 compute-optimal 分配，后来被 Chinchilla 在不同训练设计下重新估计。

## 记号与计算口径

论文使用以下记号：

| 记号 | 含义 | 论文中的口径 |
|---|---|---|
| $L$ | cross-entropy loss | 以 nats 计，通常对 context 中 token 平均 |
| $N$ | model size | non-embedding parameters，不计 vocabulary 和 positional embeddings |
| $D$ | dataset size | tokenizer 后的 training tokens |
| $C$ | training compute | non-embedding training compute，近似为 $6NBS$ |
| $B$ | batch size | 每个 parameter update 处理的 token 数 |
| $S$ | training steps | parameter update 次数 |
| $B_{crit}$ | critical batch size | 在 compute efficiency 与 step efficiency 之间折中的 batch |
| $C_{min}$ | minimum compute | 用足够小的 batch 达到目标 loss 所需的最小 compute |
| $S_{min}$ | minimum steps | 用足够大的 batch 达到目标 loss 所需的最少串行 steps |

论文中的 `PF-day` 指 $10^{15}$ FLOPs/s 持续运行一天，即：

$$
1\ \mathrm{PF\text{-}day}=10^{15}\times 24\times 3600=8.64\times 10^{19}\ \mathrm{FLOPs}
$$

需要注意的是，$6ND$ 只是 dense Transformer 的高层估算。论文在附录中拆解了 embedding、attention、feed-forward、logits 和 backward pass 的计算，发现主实验范围内该近似足够好；但长 context、MoE、通信、optimizer update 和 activation memory 都可能使真实训练成本偏离这一估算。

## 方法与实验设计

### 数据集与 tokenizer

论文使用 WebText2，这是对 WebText 的扩展版本。数据来自截至 2018 年 10 月、被 Reddit 用户以至少 3 karma 评价过的 outbound links。

- 文档数：约 20.3M；
- 原始文本：约 96GB；
- `wc` 统计的词数：约 $1.62\times10^{10}$；
- BPE tokenizer 后 token 数：约 $2.29\times10^{10}$；
- 保留测试集：约 $6.6\times10^8$ tokens；
- vocabulary size：50,257；
- 常规 context length：1024 tokens。

此外，论文还在 Books Corpus、Common Crawl、English Wikipedia 和 Internet Books 等分布上测试 transfer。所有这些数量都依赖具体 tokenizer，不能直接与其他论文的 raw text size 或 token count 混用。

### Transformer 参数化

模型使用 decoder-only Transformer，并用以下超参数描述结构：

- $n_{layer}$：layer 数；
- $d_{model}$：residual stream 维度；
- $d_{ff}$：feed-forward intermediate dimension；
- $d_{attn}$：attention output dimension；
- $n_{heads}$：attention heads 数；
- $n_{ctx}$：context length。

non-embedding parameter count 近似为：

$$
N\approx 2d_{model}n_{layer}(2d_{attn}+d_{ff})
$$

标准设置取 $d_{attn}=d_{ff}/4=d_{model}$，因此：

$$
N\approx 12n_{layer}d_{model}^2
$$

论文特意将 embedding parameters 排除在 $N$ 之外。原因是 vocabulary embedding 的规模通常随 vocabulary size 和 $d_{model}$ 变化，但它对主体 Transformer capacity 的贡献方式不同；排除后，不同 depth 的模型更容易落在同一条 scaling trend 上。

### Compute 估算

一个 token 的 Transformer forward compute 可近似写成：

$$
C_{forward}\approx 2N+2n_{layer}n_{ctx}d_{attn}
$$

当 $d_{model}$ 相对于 context length 足够大时，context-dependent 的 attention 项在主实验中相对较小。forward、backward 和 gradient 计算合起来约为 forward 的三倍，因此论文采用：

$$
C\approx 6NBS
$$

这个口径用于比较模型规模、batch 和训练步数，不等价于实际 wall-clock time。真实训练还要加入 pipeline / tensor parallel communication、checkpoint、data loading、evaluation 和系统利用率等成本。

### 训练设置

除非另有说明，论文采用：

- Adam optimizer；
- 250,000 training steps；
- batch size 为 512 条、每条 1024 tokens 的 sequence；
- 大于 1B parameters 的模型因 memory constraints 使用 Adafactor；
- 3,000-step linear warmup；
- warmup 后 cosine decay 到 0；
- 主要指标为 WebText2 test distribution 上的平均 cross-entropy loss。

论文还改变 model size、dataset size、shape、context length 和 batch size，分别研究各变量对 loss 的影响。不同实验并非全部采用同一个训练 regime，因此比较时需要区分主实验设置和专门的消融设置。

## Scaling Law 结果

### Model size scaling

在数据足够大、模型训练接近 convergence、没有明显 data bottleneck 时，loss 随 non-embedding parameters $N$ 按 power law 变化：

$$
L(N)=\left(\frac{N_c}{N}\right)^{\alpha_N}
$$

经验拟合约为：

$$
\alpha_N\approx 0.076,
\qquad
N_c\approx 8.8\times10^{13}
$$

这里 $N_c$ 是与 tokenization 和 loss normalization 相关的尺度常数，不具有普适的物理意义。指数的直观含义是：在该实验范围内，参数量翻倍，loss 大约乘以 $2^{-0.076}\approx0.95$。收益是递减的，但在实验上仍然平滑。

论文实验的 non-embedding parameter range 约为 768 到 1.5B。这个范围远小于今天的大模型规模，因此不能把拟合曲线机械外推到任意数量级。

### Dataset size scaling

在模型足够大、训练因数据规模受限并采用 early stopping 时，loss 随 dataset tokens $D$ 变化为：

$$
L(D)=\left(\frac{D_c}{D}\right)^{\alpha_D}
$$

拟合约为：

$$
\alpha_D\approx0.095,
\qquad
D_c\approx5.4\times10^{13}\ \text{tokens}
$$

这个关系描述的是 data-limited regime。若模型本身太小，继续增加数据会撞上 model capacity bottleneck；若模型足够大但训练步数或 schedule 不合适，也不能把 loss 改善完全归因于新增数据。

### Compute scaling

当 dataset 足够大、model size 和 batch size 适合当前 compute budget 时，论文定义 optimally allocated compute $C_{min}$，并拟合：

$$
L(C_{min})=\left(\frac{C^c_{min}}{C_{min}}\right)^{\alpha^{min}_C}
$$

其中：

$$
\alpha^{min}_C\approx0.050,
\qquad
C^c_{min}\approx3.1\times10^8\ \text{PF-days}
$$

论文也展示了固定 batch size 下的经验 $L(C)$，但认为为了比较真正的 compute efficiency，应先校正 batch size 带来的 step / sample trade-off，再使用 $C_{min}$。

### Model shape 的影响

在保持 non-embedding parameter count 相近的条件下，论文分别改变 depth、width、attention heads 和 feed-forward ratio。主要结果是：

- 在合理范围内，多种 shape 的 test loss 很接近；
- depth-to-width aspect ratio 可以变化约 40 倍，但 loss 只增加几个百分点；
- 极浅模型、少于两层的模型或极端 depth-to-width ratio 会明显偏离；
- 排除 embedding parameters 后，不同 depth 的模型更接近同一条 $L(N)$ scaling curve。

这并不意味着 architecture 不重要。它只说明，在该模型族和实验范围内，总的 non-embedding scale 是更强的第一阶解释变量；当模型进入极端 shape、不同 context regime、不同稀疏结构或不同任务目标时，这一结论未必成立。

### Context length 与 token position

论文按 context 中的 token position 分析 loss，发现 Transformer 能够在整个 1024-token context 中持续利用较后位置的信息。作为对照，LSTM 在较早位置表现接近 Transformer，但随着 context position 增加，其 loss 改善趋于停滞；Transformer 则继续改善。

这一结果支持两个判断：

1. context length 不是只改变输入格式，它会改变模型可以利用的条件信息范围；
2. 对长上下文任务，平均 loss 可能掩盖不同位置的学习差异，应该额外观察 position-wise loss 和长距离依赖能力。

论文还比较了 recurrent Transformer，发现参数复用可以带来一定 parameter efficiency，但会增加每个参数的 compute。由此可见，model size、depth、context length 和 per-token compute 之间不能只看一个数字。

### Transfer scaling

模型只在 WebText2 上训练，却在 Books、Wikipedia、Common Crawl 和 Internet Books 等其他数据分布上测试。结果显示：

- 分布外 test loss 也随模型规模平滑下降；
- 不同分布上的 loss 与 WebText2 validation loss 高度相关；
- 主要差异近似表现为相对稳定的 additive offset；
- transfer performance 更依赖模型在训练分布上的 loss，而不是是否已经训练到 convergence 或训练了多久；
- 论文没有观察到明显的 depth-specific transfer advantage。

这可以理解为：在这些实验分布之间，模型规模带来的基础语言建模能力提升能够迁移，但不同分布仍然保留不可忽略的 domain gap。它不意味着一个 in-distribution loss 很好的模型会自动具备所有领域能力，更不能替代 domain-specific validation。

## Model-Data Trade-off 与 Overfitting

### 联合 loss 拟合

仅分别拟合 $L(N)$ 和 $L(D)$ 还不能回答模型和数据同时变化时的表现。论文提出联合形式：

$$
L(N,D)=
\left[
\left(\frac{N_c}{N}\right)^{\alpha_N/\alpha_D}
 +\frac{D_c}{D}
\right]^{\alpha_D}
$$

该形式满足三个边界条件：

- 当 $D\to\infty$ 时，退化为模型规模 scaling law；
- 当 $N\to\infty$ 时，退化为数据规模 scaling law；
- 在大 $D$ 区域，对 $1/D$ 具有整数次幂展开的形式。

对联合数据拟合得到：

$$
\alpha_N=0.076,
\qquad
\alpha_D=0.103,
\qquad
N_c=6.4\times10^{13},
\qquad
D_c=1.8\times10^{13}
$$

这些数值与分别拟合 $L(N)$、$L(D)$ 所得结果略有差异，这是因为联合拟合同时覆盖了完整的 model-data surface。

### Overfitting ratio

论文定义相对于无限数据极限的 overfitting penalty：

$$
\delta L(N,D)=\frac{L(N,D)}{L(N,\infty)}-1
$$

由联合 scaling law 可得，它主要取决于：

$$
\frac{N^{\alpha_N/\alpha_D}}{D}
\approx\frac{N^{0.74}}{D}
$$

这意味着数据量不需要与参数量线性增长，才能避免明显 overfitting；在该实验范围内，数据需求相对模型规模是 sublinear 的。

论文估计不同 random seed 带来的 loss variation 约为 0.02。若希望 overfitting penalty 不超过这一量级，经验条件约为：

$$
D\gtrsim(5\times10^3)N^{0.74}
$$

按这一估计，参数量小于 $10^9$ 的模型可以在 22B-token WebText2 上以很小的 overfitting 训练，而更大的模型会出现轻微 overfitting。这个结论没有优化 dropout 等 regularization，因此只能作为该实验设置下的经验边界。

### 如何理解这条规律

这组结果揭示的是两个不同问题：

- **capacity scaling**：增加模型参数可以降低模型表示不足造成的误差；
- **data scaling**：增加训练 token 可以降低数据不足和有限样本带来的误差。

如果只扩大 $N$ 而不扩大 $D$，模型会进入 data-limited regime；如果只扩大 $D$ 而不扩大 $N$，模型会进入 model-limited regime。总 token 数也不是有效数据量的充分代理：tokenizer、重复、数据质量和 domain composition 都会改变实际收益。

## Training Time、Batch Size 与 Critical Batch Size

### Training curve

在 infinite-data limit 下，论文用模型规模 $N$ 和 batch-adjusted training steps $S_{min}$ 拟合 learning curve：

$$
L(N,S_{min})=
\left(\frac{N_c}{N}\right)^{\alpha_N}
+\left(\frac{S_c}{S_{min}}\right)^{\alpha_S}
$$

拟合值约为：

$$
\alpha_S\approx0.76,
\qquad
S_c\approx2.1\times10^3
$$

其中第一项表示有限模型规模带来的 loss floor，第二项表示训练 steps 不足带来的 optimization / training-time error。论文认为，在 warmup 之后的稳定训练区间，这个简单 power-law 对 learning curves 有相当好的描述；但在非常早期的 transient period，拟合会变差。

论文还提出，可以用早期 training curve 粗略外推更长训练后的 loss。这类外推必须建立在训练已经进入稳定 power-law 区间的前提上，不能把 warmup 或初始不稳定阶段直接拿去拟合。

### Critical batch size

论文沿用 gradient noise scale 相关的 batch-size 理论，把达到某个目标 loss 所需的 steps $S$ 与处理样本数 $E=BS$ 联系起来：

$$
\left(\frac{S}{S_{min}}-1\right)
\left(\frac{E}{E_{min}}-1\right)=1
$$

据此定义：

$$
B_{crit}(L)=\frac{E_{min}}{S_{min}}
$$

在 $B\lesssim B_{crit}$ 时，增大 batch 可以减少串行 steps，但不会显著损害 compute efficiency；当 $B\gg B_{crit}$ 后，继续增大 batch 的收益变小。取 $B\approx B_{crit}$，大约需要 $2S_{min}$ steps 和 $2E_{min}$ examples，是 step efficiency 与 compute efficiency 的折中点。

论文拟合的 critical batch size 为：

$$
B_{crit}(L)\approx\frac{B^*}{L^{1/\alpha_B}},
\qquad
B^*\approx2\times10^8\ \text{tokens},
\qquad
\alpha_B\approx0.21
$$

实验上，$B_{crit}$ 对当前 loss 的依赖明显，而对 model size 没有直接依赖。论文观察到，在最大的实验模型接近 convergence 时，合适的 batch size 大约在 1-2M tokens 量级；同时，loss 下降会使 gradient noise scale 增大，因此训练后期适合使用更大的 batch。

### Early stopping 与 data-limited training

把 finite-data learning curve 与 infinite-data learning curve 对齐，可以得到 data-limited regime 下 early stopping step 的下界：

$$
S_{stop}(N,D)\gtrsim
S_c\left[L(N,D)-L(N,\infty)\right]^{1/\alpha_S}
$$

它不是精确的 stopping rule，而是说明：数据越有限，模型到达最佳 test loss 的时间点越受 overfitting dynamics 影响；直接复用无限数据下的训练步数可能并不合适。

## Compute-efficient Allocation

### Kaplan-style 结论

论文把前面的 learning curve、critical batch size 和 compute 估算结合起来，研究固定 $C_{min}$ 时的最优分配。经验结果为：

$$
N_{opt}\propto C_{min}^{0.73}
$$

由此进一步得到：

$$
B\propto C_{min}^{0.24},
\qquad
S\propto C_{min}^{0.03},
\qquad
D=B\times S\propto C_{min}^{0.27}
$$

这组 exponent 表达了一个很强的资源分配倾向：在该论文的拟合 regime 中，compute 增加时，大部分预算应该用于扩大模型；batch size 随之增加，而串行 optimizer steps 几乎不增加。论文将这种方案描述为“large models trained on a relatively modest amount of data and stopped before convergence”。

### 为什么会得到“大模型、短训练”

原因不是大模型在每个方面都更优，而是论文估计大模型具有更高的 sample efficiency：

- 大模型每个 optimization step 的 compute 更贵；
- 但达到同一 loss 所需的 steps 更少；
- 通过增大 batch，可以把更多数据处理转成并行度，而不显著增加串行时间；
- 在固定 compute 下，模型容量收益在论文实验范围内被认为比延长训练的收益更强。

论文的理论推导基于：

$$
\alpha_C^{min}
=\frac{1}{1/\alpha_S+1/\alpha_B+1/\alpha_N}
\approx0.052
$$

进而预测：

$$
N(C_{min})\propto C_{min}^{\alpha_C^{min}/\alpha_N}
\approx C_{min}^{0.71}
$$

与经验拟合的 0.73 接近。

### Efficient 与接近 convergence 的训练

论文进一步比较了 compute-efficient training 与通常“训练到看起来接近 convergence”的方案。在其模型中，compute-efficient 方案达到的 loss 大约比 converged loss 高 10%；如果把这个差异压到 2%，则需要：

- 约 2.7 倍的参数量才能达到相同 loss；
- 约 7.7 倍的 parameter updates；
- 约 2.9 倍的 compute，或反过来说 compute-efficient 方案约少用 65% compute。

这个比较描述的是论文拟合曲线中的理想化 trade-off。实际训练是否接受更高的 pretraining loss，要看数据复用、推理成本、后续适配计划和目标能力，而不能仅由 average loss 决定。

### 最优模型尺寸不是尖锐的点

论文估计，在达到同一目标 loss 时，使用约 $0.6\times$ 到 $2.2\times$ 最优参数量的模型，compute 增加约 20% 以内。也就是说，compute-optimal frontier 附近存在一定宽容区：

- 更小的模型可能有更低的 inference cost；
- 更大的模型可能需要略多 compute，但串行 steps 更少，适合有足够并行资源的训练系统；
- 真实选型还应纳入 serving cost、显存、并行效率和后续训练空间。

### Scaling law 自身的矛盾与边界

论文注意到，compute-efficient allocation 得到的 data growth 速度较慢：

$$
D(C_{min})\propto C_{min}^{0.26\sim0.27}
$$

但为了控制 overfitting，根据 $D\propto N^{0.74}$ 和 $N\propto C_{min}^{0.73}$，数据似乎需要按约 $C_{min}^{0.54}$ 增长。两者最终会发生冲突：即使不重复数据，compute-efficient regime 也会逐渐遇到 data bottleneck。

论文估计这个交叉点大约在：

$$
C^*\sim10^4\ \text{PF-days},
\qquad
N^*\sim10^{12},
\qquad
D^*\sim10^{12}\ \text{tokens},
\qquad
L^*\sim1.7\ \text{nats/token}
$$

但作者明确认为这些数值不确定性很大，可能相差一个数量级；更合理的解读是：scaling law 必须在远超实验范围之前发生修正，而不是把交叉点当成自然语言能力的精确上限。

## 证据如何支持这些结论

### 多变量、跨数量级实验

论文不是只比较少数几个最终模型，而是分别改变：

- model size：约 768 到 1.5B non-embedding parameters；
- dataset size：约 22M 到 23B tokens；
- model shape：depth、width、heads 和 feed-forward dimension；
- context length；
- batch size；
- training steps 和 compute budget。

这种实验设计让作者能够把单变量 scaling、联合 model-data scaling、learning curve 和 compute allocation 连接起来。

### 拟合值并非跨设置通用

论文中不同章节的 exponent 和 scale constant 会有轻微差异。例如，单独拟合 $L(D)$ 与联合拟合 $L(N,D)$ 得到的 $\alpha_D$ 分别约为 0.095 和 0.103；training compute 的 naive curve 与 batch-adjusted $C_{min}$ curve 也不同。这不是矛盾，而是说明：

- 拟合对象不同；
- 是否校正 batch size 不同；
- 是否同时覆盖 model-data surface 不同；
- 数值常数依赖 tokenization、数据集和 loss 定义。

实际使用 scaling law 时，应保留拟合口径，不要只摘一个 exponent。

### Learning-rate schedule 是比较条件的一部分

论文在附录中指出，training horizon 与 cosine schedule 的 cycle length 需要匹配。若 schedule 的 cycle length 比实际训练目标长超过约 25%，性能会明显下降；因此不能把为较长训练设计的 schedule 中途 loss，直接当成一个独立短训练 run 的最终 loss。

这点对 scaling 实验尤其重要：如果不同模型的训练时长和 schedule 没有同步调整，模型间差异可能混入 optimization schedule 的影响。Scaling law pilot runs 至少应固定 tokenizer、数据配方、optimizer、batch 口径、warmup、decay schedule 和 checkpoint 评测方式。

## 与 Chinchilla 的关系

### 两组结论的直接差异

这篇论文给出的 Kaplan-style compute allocation 是：

$$
N_{opt}\propto C^{0.73},
\qquad
D_{opt}\propto C^{0.27}
$$

后来 Hoffmann 等人的 Chinchilla 研究在不同的实验设计下得到更均衡的分配，约为：

$$
N_{opt}\propto C^{0.46\sim0.50},
\qquad
D_{opt}\propto C^{0.50\sim0.54}
$$

常见的简化经验是 dense LM 约使用 20 training tokens per parameter。

### 为什么不能简单理解成谁对谁错

两篇论文并不是只在同一个实验上给出相反答案：

- Kaplan 主要使用 WebText2 和相对较小规模的模型，结合 learning curve 与 critical batch size 推导 compute-efficient allocation；
- Chinchilla 使用更大范围、更多不同 training horizon 的 Transformer 实验，并强调 model size、training tokens 和 learning-rate schedule 应一起重新估计；
- 数据配方、模型族、优化器、batch 策略、训练时长和拟合方式均不完全相同；
- 早期的“大模型、短训练”结论可能在当时的实验范围内有效，但不能自动推广到后来的更大训练规模。

因此，当前更稳妥的知识框架是：把 Kaplan 作为早期 scaling law 与 compute allocation 基线，把 Chinchilla 作为对 model-data frontier 的后续修正。真正的训练项目仍应通过 pilot runs 在自己的 tokenizer、data mix、模型族和目标 loss / capability 上重新估计。

## 对当前训练设计的启发

### Scaling experiment 应先固定实验语境

论文给当前训练规划的第一条启发，不是直接套用某个 exponent，而是把 scaling law 视为**实验语境内的经验模型**。若要为新的模型训练做外推，至少需要固定或显式记录：

- tokenizer 与 vocabulary；
- model parameter count 的统计口径；
- data mix、数据质量和重复次数；
- context length 与 packing 方式；
- optimizer、precision、global batch tokens；
- learning-rate warmup 和 decay schedule；
- validation split、domain validation loss 和 checkpoint 评估时点。

否则，拟合曲线可能把数据质量、schedule 或实现效率差异错误地解释成 scale effect。

### 不能用 average NTP loss 直接代替能力评测

论文证明了 loss scaling 很稳定，但作者自己也指出，平滑的 aggregate loss 变化可能隐藏 qualitative capability change。迁移到复杂训练任务时，应把以下指标分开看：

- 总 validation loss；
- code、math、academic、中文或其他目标 domain loss；
- 长上下文的 position-wise loss；
- task-level accuracy / success rate；
- 数据污染、重复和安全相关指标。

对于 Agent 或工具增强任务，平均 NTP loss 可以作为训练稳定性和分布拟合的信号，但不能单独证明模型学会了规划、工具选择、状态跟踪或错误恢复。

### 长轨迹数据需要区分 nominal tokens 与 effective supervision

如果训练语料包含很长的 environment observation、工具输出和重复上下文，token 数 $D$ 的增加不必然等价于同量的新能力信息。论文中 $D$ 的 scaling 结论建立在相对一致的 language modeling 数据分布之上；对于长轨迹，应至少额外统计：

- nominal tokens；
- assistant reasoning / action tokens；
- observation 与 tool-result tokens；
- unique information、重复率和数据质量；
- 不同 workflow / domain 的有效覆盖。

这不是对 NTP objective 的否定，而是对 effective data scaling 的细化：同样的 token budget，信息密度、时序结构和目标能力分布不同，带来的收益可能不同。

### Pilot runs 应同时看 loss curve 与能力 curve

一个可操作的实验方式是：在目标模型和全量训练之前，构造多组较小规模 pilot，分别改变 $N$、$D$、training horizon 或数据 mixture，拟合：

```text
training / validation loss
  -> domain validation loss
  -> target capability quick eval
  -> longer-horizon or post-training validation
```

如果总 loss 的 scaling 正常但目标能力不随之改善，说明该能力可能不是平均 token prediction 的充分统计量；如果某个 domain loss 恶化，则应优先检查 data mix 和数据配比，而不是继续扩大总 compute。

## 局限与疑问

### 缺少统一理论解释

论文明确承认，目前没有足够成熟的理论解释所有 scaling relation，尤其是 model size 和 compute scaling。power law 是对实验数据的高质量拟合，但还不是从数据分布、优化动力学和模型表示能力中严格推导出的定律。

### 外推范围有限

论文的主要模型规模、数据规模和 context length 都明显小于后续大模型训练。模型结构、tokenizer、数据质量、训练 precision 和优化器发生变化后，指数和常数都可能改变。论文自己也通过 compute-efficient data growth 与 overfitting growth 的冲突说明，当前 power law 必然需要在更大规模上修正。

### 目标是 language modeling loss，不是所有能力

论文主要优化和评估 cross-entropy loss。loss 与下游任务通常相关，但并不保证：

- 所有稀有能力都随平均 loss 同步提升；
- discrete benchmark 的阈值现象都是真实机制转变；
- transfer 到新的任务、交互协议或环境时仍保持相同 offset；
- post-training、tool use 和 agent success 可以用同一条曲线预测。

### 数据和正则化控制不充分

overfitting 实验没有对 dropout 等 regularization 做联合优化。大模型部分使用 Adafactor，而其他模型主要使用 Adam；不同 optimizer 可能引入额外差异。作者也指出，数据污染、训练集与测试集 overlap 以及 natural language entropy 的真实下限都限制了外推。

### Compute 估算不是系统成本

$6ND$ 主要描述 dense Transformer 的 arithmetic FLOPs。它没有直接反映：

- tensor / pipeline parallel communication；
- activation memory 和 checkpointing；
- data loading 与 I/O；
- hardware utilization；
- optimizer state 和 checkpoint storage；
- serving 与 inference cost。

因此 compute-optimal 不等于 wall-clock-optimal、memory-optimal 或 deployment-optimal。

## 关键结论

这篇论文可以沉淀为四个层次的认识：

1. **经验规律层**：在一致实验语境下，language model loss 随 $N$、$D$ 和 $C$ 呈现平滑的 power-law scaling。
2. **诊断层**：model-limited、data-limited、training-time-limited 和 batch-inefficient 是不同瓶颈，不能只用总参数量或总 token 数判断。
3. **规划层**：固定 compute 时，必须同时考虑 model size、data、batch 和串行 steps；Kaplan-style 早期拟合倾向大模型短训练，但这一分配后来被 Chinchilla 重新估计。
4. **迁移层**：loss scaling 可以作为训练外推和快速诊断工具，但对于新数据分布、长轨迹、Agent workflow 和离散能力，必须补充 domain-specific 与 task-level evaluation。

## 相关知识链接

- [[training/scaling/scaling-law|Scaling Law]]
- [[training/scaling/model-data-compute|Model Data and Compute]]
- [[training/scaling/compute-optimal|Compute Optimal]]
- [[training/scaling/training-budget|Training Budget]]
- [[training/pretraining/compute-optimal|Pretraining Compute Optimal]]
- [[training/pretraining/data-mix|Data Mix]]
- [[fundamentals/information-theory/cross-entropy|Cross Entropy]]
- [[fundamentals/information-theory/perplexity|Perplexity]]
- [[sources/papers/2022-training-compute-optimal-large-language-models|Training Compute-Optimal Large Language Models]]
- [[sources/papers/2023-a-pretrainers-guide-to-training-data|A Pretrainer's Guide to Training Data]]
