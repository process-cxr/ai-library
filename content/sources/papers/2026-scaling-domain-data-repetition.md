---
title: "Scaling Domain Data Repetition in LLM Pretraining"
created: 2026-09-08
published: 2026-09-08
modified: 2026-09-08
type: source
status: processed
source_type: paper
area: sources
tags:
  - source
  - paper
  - pretraining
  - data-mix
  - data-engineering
  - scaling
  - data-repetition
aliases:
  - Scaling Domain Data Repetition
source_url: https://arxiv.org/abs/2608.14071
paper_date: "2026-08"
paper_order: "14071"
---

# Scaling Domain Data Repetition in LLM Pretraining

## 基本信息

- 标题：[Scaling Domain Data Repetition in LLM Pretraining](https://arxiv.org/abs/2608.14071)
- 版本：arXiv:2608.14071v1，2026-08-14
- 作者：Jingwei Li、Xinran Gu、Rui Dai、Xintong Hao、Chengyin Xu、Yan Wu、Shuran Zheng、Jingzhao Zhang
- 研究背景：高校与产业研究团队合作完成的 LLM pretraining data scaling 研究
- 研究对象：固定 `tokens-per-parameter (TPP)` 条件下，高质量 domain data 的重复训练
- 相关 topic：[[training/pretraining/data-mix|Data Mix]]，[[training/data-engineering/deduplication|Deduplication]]，[[training/scaling/scaling-law|Scaling Law]]，[[training/pretraining/pretraining|Pretraining]]

这篇论文研究一个在大模型训练中非常实际、但容易被简单化的问题：**当某个高质量领域的数据不够多时，应该重复使用已有数据多少次？**

如果只看“重复会导致 overfitting”，结论会过于粗糙。高质量代码、数学、百科或医学数据往往比普通网页数据更难扩展；当模型规模和训练 token budget 同时增长时，固定规模的高质量数据会在 mixture 中被逐渐稀释。理解这篇论文，首先需要把 `unique domain data`、总 `token presentations` 和 `repetition count` 分开：前者决定内容覆盖，后两者决定模型对已有内容的训练深度与重复程度。论文围绕这几个变量，分析 domain、model size、unique-domain allocation fraction、总训练预算和 learning-rate schedule 如何共同改变 repetition 的收益与代价。

论文最重要的价值不是给出一个可以直接复制的 repetition 倍数，而是把 repetition 从一个数据清洗副作用，提升为一个需要单独调参、单独评估的 data-mix 变量。

## 研究问题

### 同一个模型里，数据不足和重复训练分别意味着什么

先固定模型规模 $N$，把一个 domain 的训练过程拆成三个量。这里的 `unique domain data` 有一个特定含义：它是本次 run 在 repetition 之前，从目标 domain 中选出的、彼此不重复的 high-quality token subset。它不是原始 domain corpus 经过清洗或 dedup 后的保留比例，也不是 repetition 之后该 domain 在训练流中的最终比例。

数据进入这组实验后的关系可以写成：

```text
raw domain corpus
  -> cleaning / deduplication
  -> select a fixed unique subset U
  -> repeat the subset e times
  -> domain presentations H = eU
```

论文进一步用 `unique-domain allocation fraction` $\alpha$ 表示这批 unique subset 相对于本次训练总 token budget 的比例：

$$
U=\text{unique domain token budget selected before repetition}
$$

$$
\alpha=\frac{U}{D}
$$

$$
E=\text{同一内容的平均 exposure / repetition}
$$

$$
D_{\text{seen}}=U\times E=\text{模型实际看到的 domain token presentations}
$$

如果重复次数为 $e$，则论文设置中的 $E=e$，最终 domain presentations 为 $H=eU$，占总训练预算的比例为：

$$
\rho=\frac{H}{D}=e\alpha
$$

因此需要区分三个比例：

- `dedup retention rate`：清洗 / 去重后保留的数据量，占原始 domain corpus 的比例；
- `unique-domain allocation fraction` $\alpha$：本次训练选取的 unique domain tokens，占总训练 token budget 的比例；
- `final domain fraction` $\rho$：这些 tokens 经过 repetition 后，占最终训练 token stream 的比例。

论文研究的是 $\alpha$ 和 $e$ 的训练配方关系，不把 raw corpus 到 deduplicated corpus 的保留率作为实验变量。

这三个量对应不同的问题：

- **Unique data 不足**：$U$ 太小，模型能够接触到的知识、表达和长尾模式有限。重复这些内容不能创造新的 coverage；
- **总 exposure 不足**：$U$ 可能足够大，但 $D_{\text{seen}}$ 太小，模型还没有把已有数据中的 signal 学充分；
- **Repetition 过高**：$U$ 固定而 $E$ 持续增加，前期可以降低 signal acquisition error，后期则可能开始拟合样本特定模式、模板和噪声，产生 memorization 或 validation degradation。

因此，训练数据不足和重复过拟合不是同一种现象。前者主要限制模型**学到多大范围**，后者主要决定模型**对已经见过的内容拟合多深**。同一个模型即使处于 data-limited 状态，也仍然可能因为反复训练有限数据而过拟合：它对整个 domain 的覆盖仍然不足，但对已经出现的样本已经拟合过度。

在固定总 domain token budget 的情况下，增加 repetition 通常意味着减少 unique data；此时观察到的结果是 coverage 与 exposure 的折中，而不是 repetition 的单独效应。要把二者分开，需要分别做两类比较：固定 $U$、增加 $E$，观察重复容忍度；固定 $D_{\text{seen}}$、改变 $U$ 与 $E$，比较更多 unique data 和更多 repeated exposure 哪个更有价值。

### 跨模型规模时，训练预算如何改变 repetition 的最优值

论文进一步考察不同模型规模下的 repetition。关键不在于模型规模单独变大，而在于模型变大时训练数据预算如何变化。已有不少工作在不同模型规模之间固定训练数据量 $D$，然后观察重复数据造成的 overfitting；本文则将其与固定 `TPP` 的设置进行对照：

$$
D_N = TPP \cdot N
$$

其中 $N$ 是模型规模，$D_N$ 是总训练 token 数，`TPP` 是固定的 tokens-per-parameter。此时模型变大，训练 token budget 也同步增加。论文要回答的是：在固定总数据量和固定 `TPP` 两种 scaling path 下，最优 repetition count 是否会呈现不同趋势。

### 最优 repetition 由什么决定

作者分别考察：

- 高质量 domain 的类型；
- model size；
- unique high-quality data fraction；
- 训练总 token budget 与 TPP；
- learning-rate schedule；
- 是否用 unique token 替换 repeated token；
- in-domain 与 out-of-domain validation loss。

这组变量的区分很重要。论文中的 `unique-domain allocation fraction` $\alpha$ 描述本次 run 选取了多少不同的 domain 内容，`repetition count` $e$ 描述这批固定内容被重新看到多少次，而最终 domain token presentations 由 $e\alpha D$ 决定。把这几个量混在一起，无法判断收益来自更多 domain exposure，还是来自更多内容多样性。

## 核心主张

### 1. Repetition 的最优值强烈依赖 domain

在论文测试的 code、math、Wikipedia 和 medical 四个 domain 中，最优 repetition count 明显不同。正文与附录给出的整体趋势是：

- Math 通常在约 `5-6` 次附近达到较低的 in-domain validation loss；
- Code 通常在约 `4-5` 次附近达到较低 loss；
- Wikipedia 通常在约 `3-4` 次附近达到较低 loss；
- Medical 通常在约 `3-4` 次附近达到较低 loss。

这些数字只属于论文的训练规模、数据构造、优化 recipe 和 validation set，不能被当成通用配方。可迁移的结论是：**不同 domain 应拥有独立的 repetition sweep，而不是共享一个全局 epoch 数。**

### 2. Fixed `TPP` 时，跨模型规模的最优 repetition 轻微增加

这是一个**跨模型规模的曲线比较**，不是在同一个模型上单独增加 repetition。作者固定 domain、unique-domain allocation fraction $\alpha$、`TPP` 和训练 recipe，然后改变模型规模 $N$。由于：

$$
D_N=TPP\cdot N,\qquad U_{N,\alpha}=\alpha D_N
$$

模型规模从 $N$ 增加到 $kN$ 时，总训练 token budget $D_N$ 和 unique domain data $U_{N,\alpha}$ 都扩大到原来的 $k$ 倍。作者再为每个模型规模分别扫描 $e\in\{1,\ldots,7\}$，比较各自的 final validation loss-repetition 曲线。实验发现，随着 $N$ 增加，曲线最低点对应的最优 repetition count 会向右轻微移动。

因此，论文在 fixed-TPP 主实验中得到的准确结论是：

> 当模型规模、总训练数据和 unique domain data 按上述关系一起扩大时，更大模型对应的最优 repetition count 略高。

它并没有证明“对同一份固定的 unique dataset，大模型天然可以重复更多次”。论文用 fixed $D$ 的对照说明这一点：如果增加 $N$，但 $D$ 和 unique observations 都不增加，那么 `TPP = D/N` 会下降；在论文的实验与理论条件下，loss-repetition 曲线的最低点反而会向较小 repetition 移动，最优值趋近于 1。

所以，两个 scaling regime 比较的是不同的数据扩展方式：

- **Fixed $D$**：只扩大模型，数据预算与 unique coverage 不变，最优 repetition 随规模增大而下降；
- **Fixed `TPP`**：模型、总数据和实验中的 unique domain data 同时扩大，最优 repetition 随规模增大而轻微上升。

这也是论文提出 proxy transfer 的依据：在相同 `TPP` 下，小模型上尚未造成过拟合的 repetition count，可以作为更大模型的保守起点。这个结论依赖 $U_{N,\alpha}=\alpha D_N$ 的实验设置；如果现实中的 unique domain corpus 已经封顶、无法随目标模型扩展，就不能直接套用。

### 3. 跨 repetition sweep，最优 repetition 与可达到的最低 validation loss 强负相关

这里的一个统计点不是单次训练 run，也不是笼统的一个 domain，而是一组固定 $(d,N,\alpha)$ 的 repetition sweep。对每一组 sweep，作者执行以下步骤：

1. 固定 domain $d$、model size $N$ 和 unique-domain allocation fraction $\alpha$；
2. 分别使用 $e\in\{1,2,\ldots,7\}$ 训练多个模型；
3. 记录每个 run 在完整 token budget 结束时的 final validation loss，得到 $L_{d,N,\alpha}(e)$；
4. 用二次曲线拟合这些离散结果，从中得到最优 repetition $\widehat e^*_{d,N,\alpha}$，以及该拟合曲线可达到的最低 loss $\widehat L^*_{d,N,\alpha}$。

完成所有 domain、model size 和 unique-domain allocation fraction 的 sweep 后，作者把每组得到的 $(\widehat e^*,\widehat L^*)$ 作为一个点，计算这些点之间的 Pearson correlation，结果约为 `-0.944`。它表示：

> 在论文比较的多组 $(domain, model size, unique-domain allocation fraction)$ 设置中，可达到的最低 validation loss 越低，该组 loss-repetition 曲线的最低点通常出现在更大的 repetition count。

这不是说同一组实验里 repetition 越大，validation loss 就越低。同一组 sweep 的曲线通常先下降、到达 $\widehat e^*$，随后因继续重复而上升。`-0.944` 描述的是多组曲线之间“最低点位置”和“最低点高度”的关联，而不是一条曲线内部的单调关系。

由于不同 domain 的 validation-loss 水平差异明显，论文将每组 sweep 的最低 loss 作为 domain-specific characteristics 的连续 proxy，并用理论模型中的 noise level 解释这种关联：更容易学习、有效噪声更低的 domain，可能允许 knowledge acquisition 持续更久，因而能承受更多 repetition。但 empirical minimum loss 并不是噪声方差的直接测量，这一相关性也不能单独证明因果关系。

作者用同样的 sweep-level 数据比较其他因素：

- $\widehat e^*$ 与 model size 的 Pearson correlation 约为 `0.400`，表现为较弱的正相关；
- $\widehat e^*$ 与 unique-domain allocation fraction 的 Pearson correlation 约为 `0.018`，在论文测试范围内接近零。

因此，proxy model 的用途不是根据一次 validation loss 直接推算 repetition，而是为每个 domain 完整扫描 loss-repetition 曲线。在与目标模型保持相同 `TPP` 的条件下，先得到小模型的 $\widehat e^*$ 和曲线形状，再把它们作为目标模型 repetition 区间的保守参考。

### 4. Unique data 比 repeated data 更有价值，但差异取决于 domain

前面的实验固定 unique-domain allocation fraction，增加 repetition count。论文还做了互补实验：固定训练中目标 domain 的总 token fraction，用更多 repetition 替换更多 unique data，直接比较“少量 unique data 重复多次”和“更多 unique data 各看一次”。

结果显示：

- Math 对 repetition 相对稳健。在总 domain token fraction 固定时，重复次数从 1 增加到 4，validation loss 的上升较小；
- Wikipedia 在 repetition 超过 2 次后退化明显；
- Code 和 Medical 的趋势更接近 Wikipedia，unique data 被 repeated data 替换后会出现更明显的 loss degradation。

所以，“重复 token 可以近似替代新 token”只在部分 domain 和有限 repetition 范围内成立。它更适合作为数据不足时的预算折中，而不是把重复当成新数据的等价物。

### 5. 在控制 domain fraction 和 web fraction 后，重复主要影响 in-domain 能力

如果固定 unique domain data，然后增加 repetition，训练中 domain token 总量上升、web token 总量下降。此时 OOD 变化无法归因于 repetition 本身。论文因此固定总 high-quality domain fraction，同时改变 unique data 与 repetition count，从而保持 web data fraction 不变。

在这一控制下，ArXiv 和 News 等 OOD pretraining validation loss 随 repetition 的变化总体较小，没有稳定的单调趋势。相对清晰的变化主要出现在被重复的 domain 内部。

这说明重复策略的影响需要分成两层看：

- in-domain validation：判断目标 domain 是否被充分学习、是否出现 memorization；
- OOD validation：判断重复是否挤压了通用数据，或破坏跨域迁移。

只看总 validation loss，很可能错过目标 domain 的过拟合；只看 in-domain gain，又可能忽略通用能力代价。

### 6. Learning-rate schedule 会改变 repetition 的耐受度

论文比较了 WSD schedule 在不同时间开始 decay 的配置，以及 constant learning rate。结果是：

- 更早开始 decay 时，validation loss 在更少 repetition 后开始上升；
- 推迟 decay 时，模型可以承受更多 repetition；
- constant learning rate 在论文设置下允许最大的 repetition count。

一个可能解释是，低 learning rate 阶段更容易精细拟合重复样本中的 sample-specific pattern；较长时间保持较高 learning rate，会保留更强的 optimization noise 和 implicit regularization，延迟对重复数据的过拟合。

这个结果不表示 constant learning rate 在所有训练中都更好，而是说明 repetition policy 不能脱离 learning-rate schedule 单独迁移。数据配方与优化配方是耦合的。

## 实验设置

### 固定 TPP 的训练预算

论文对不同模型规模 $N$ 设置：

$$
D_N = TPP \cdot N
$$

每个模型规模的总训练 token budget 固定，但不同模型规模的总预算按比例增长。`TPP` 在实验中设为大于 100 的常数。

对每个目标 domain $d$，作者先从 domain 中固定选取一批 unique high-quality tokens，再用它们构造训练 mixture。论文将这批 unique tokens 占总训练 token budget 的比例记为 $\alpha$，也就是 unique-domain allocation fraction，取值为：

$$
\alpha \in \left\{\frac{1}{40},\frac{1}{20},\frac{1}{10}\right\}
$$

repetition count 记为 $e$，取值为 `1` 到 `7`。高质量 domain 的实际 token presentations 为：

$$
H_{N,\alpha,e}=e\alpha D_N
$$

剩余预算由 non-repeated web data 填充：

$$
W_{N,\alpha,e}=(1-e\alpha)D_N
$$

因此，$e\alpha$ 决定最终训练流中该高质量 domain 的 token fraction，而 $\alpha$ 与 $e$ 分别控制 unique content 数量和每个内容的重复次数。这里的 $\alpha$ 是训练预算中的 unique subset allocation，不是 dedup retention rate；每个实验只重复一个 high-quality domain，不同时重复多个 domain。

这里有两组容易混淆、但回答不同问题的实验：

- **固定 $\alpha$、改变 $e$**：unique domain data 的比例保持不变，增加 $e$ 会增加该 domain 的 token presentations，同时挤出一部分 web data。这个实验主要观察在固定 unique coverage 下，更多 exposure 何时从继续学习 signal 转为 noise-fitting，但它并不是只改变 repetition 而完全不改变 mixture；
- **固定 $\rho=e\alpha$、改变 $e$**：总 high-quality domain token fraction 保持不变，增加 $e$ 时减少 $\alpha$。这个实验保持 domain 与 web 的总比例不变，直接比较“更多 unique content 各看一次”和“更少 unique content 重复多次”之间的取舍。

因此，第一组实验回答“同一批 unique data 还能从更多 exposure 中获得多少收益”，第二组实验回答“在固定 domain token budget 下，repetition 能否替代 unique coverage”。只有把这两组结果分开，才能同时理解 repetition tolerance 和 unique data 的不可替代性。

### 评测指标

作者对每个 run 同时评估：

- target domain 的 held-out validation loss，即 `IID` loss；
- 通用 pretraining validation corpus 上的 `OOD` loss；
- 在固定 domain fraction 的补充实验中，使用 ArXiv 和 News 作为 OOD validation。

这里的两个缩写需要区分：

- **IID** 是 `independently and identically distributed` 的缩写，通常译为“独立同分布”。在本文中，`IID loss` 更具体地指目标 domain 的 **in-distribution held-out validation loss**：验证数据与被重复训练的 domain 来自相同或相近的数据分布，但与训练样本保持独立、没有直接重叠。它主要用来判断目标 domain 是否还在被有效学习，以及 repetition 是否已经导致 domain 内过拟合；
- **OOD** 是 `out-of-distribution` 的缩写，指分布外评测。在本文中，OOD validation 使用通用 pretraining corpus，补充实验中还使用 ArXiv 和 News。它主要用来观察 repetition 是否影响目标 domain 之外的泛化能力，以及是否因调整 domain 配比而挤压了通用数据。

严格来说，语言模型中的 token 和文档并不满足字面意义上的完全独立同分布；这里的 `IID` 是相对于目标 domain 的常规评测称呼，核心含义是“同分布的独立留出集”，而不是对 token 生成过程的额外假设。

论文将最终 validation loss 对 repetition count 做二次拟合：

$$
\widehat{L}_{d,N,\alpha}(e)=a_{d,N,\alpha}e^2+b_{d,N,\alpha}e+c_{d,N,\alpha}
$$

再以：

$$
\widehat{e}^{*}_{d,N,\alpha}=-\frac{b_{d,N,\alpha}}{2a_{d,N,\alpha}}
$$

估计连续的最优 repetition count。这样做的目的不是声称真实曲线必然是二次函数，而是减少只在 `1-7` 离散点上取最小值带来的粗糙性，便于比较 domain、model size 和 unique-domain allocation fraction 的相关关系。

## 方法与理论解释

### Signal acquisition 与 noise fitting 的竞争

论文构造了一个 one-hot linear regression 模型，把每个坐标理解成一个需要学习的 knowledge unit。模型容量限制为前 $N$ 个坐标，训练数据量为 $D$，同一个 knowledge unit 在训练集合中出现的次数是随机变量 $M_k$。

在该模型中，population risk 可以分解为三部分：

$$
2R_{D,N}(r)=
\underbrace{\sum_{k>N}k^{-\beta}}_{\text{unrepresented knowledge}}
+
\underbrace{\sum_{k=1}^{N}k^{-\beta}\mathbb{E}\left(1-\frac{\eta M_k}{D}\right)^{2r}}_{\text{knowledge-acquisition error}}
+
\underbrace{\sigma^2\sum_{k=1}^{N}p_k\mathbb{E}\left[\mathbf{1}_{M_k>0}\left(1-\left(1-\frac{\eta M_k}{D}\right)^r\right)^2\frac{1}{M_k}\right]}_{\text{noise-fitting error}}
$$

三个项分别表示：

1. 模型容量之外、当前模型无法表示的 knowledge；
2. 已经能够表示、但在当前有限 exposure 和优化步数下还没有学充分的 signal；
3. 模型对有限样本噪声和 sample-specific pattern 的拟合。

这三个项不能机械地分别等同于“数据不足”“训练不够”和“重复过多”。其中第一项是模型容量造成的 unrepresented knowledge；unique data 太少会减少可观察的 knowledge coverage，并通过有限样本分布影响后两项；repetition 或优化步数增加，通常会降低第二项，但可能提高第三项。

增加 repetition 或优化步数会继续降低 knowledge-acquisition error，但也会提高 noise-fitting error。最优 repetition 出现在 signal 的边际收益开始低于 noise-fitting 的边际代价时。

### 为什么较低 validation loss 的 domain 可以承受更多重复

在理论模型中，较低的噪声水平 $\sigma^2$ 不会改变 signal acquisition 项，却会降低 noise-fitting 项。因此模型可以进行更多轮优化，才会让 fitting noise 的代价超过继续学习 signal 的收益。论文的 Theorem 4.2 给出：噪声越低，最早达到最优 stopping time 不会更早；当噪声趋近于零时，最优 stopping time 近似按 $\log(1/\sigma^2)$ 增加。

这为实验中的负相关提供了机制解释：validation loss 较低的 domain 往往具有更低的有效噪声或更容易学习的结构，因此更能从 repetition 中获益。但这里的 domain loss 不是噪声方差的直接估计，只能作为经验 proxy。

### 固定数据预算与固定 TPP 的差异

在固定 $D$ 的条件下，增加 $N$ 会把更多、更稀有的 knowledge unit 纳入模型容量，但不会增加它们的 observation count，`TPP` 也随之下降。此时模型一方面缺少覆盖整个知识空间的数据，另一方面在继续优化有限样本时更容易进入 noise-fitting。理论上，当模型规模超过由 $D$ 和噪声水平决定的 crossover scale 后，新增容量更容易让 noise-fitting 占主导，最优 stopping time 会不增反降。

在固定 $D/N$ 的条件下，$D$ 与 $N$ 同步增长。论文的 Theorem 4.4 给出一个渐近结果：在源分布和信号条件满足假设时，最优 stopping time 的量级随 $D^{\alpha/\beta}$ 增长。它不直接给出真实 LLM 的 repetition count，但说明为什么固定数据量下的跨规模结论不能直接套用到 fixed-TPP 的训练。

## 实验结果的理解

### Repetition count 与 domain loss 的关系

论文把四个 domain 和多个模型规模、unique-domain allocation fraction 的实验点放在一起比较。最优 repetition 的主要变化来源依次可以概括为：

1. domain 本身的可学习性和噪声水平；
2. model size 在固定 TPP 下带来的温和变化；
3. unique-domain allocation fraction 在测试范围内的较小影响。

这里的“unique-domain allocation fraction 影响较小”不能解读为 unique data 不重要。它只表示：当 $\alpha$ 在 `1/40` 到 `1/10` 的范围内变化时，达到最优 repetition 的位置相对稳定；绝对 validation loss 仍然会随 unique data budget 改变。换言之，更多 unique data 可以整体降低 loss，但不一定改变曲线最低点所在的 repetition 区间。

### 重复与新数据的不可替代性

固定总 domain fraction 的实验更接近工程中的一个数据预算问题：如果只能消耗固定数量的 domain token，是使用更多不同文档，还是反复使用小 subset。

Math 的结果说明，结构规整、信号密度较高且重复后仍能提供有效监督的 domain，可能在有限范围内更适合 repetition。Wikipedia、Code 和 Medical 的退化则说明，知识覆盖、表达多样性、代码上下文和事实长尾可能更依赖 unique content。

这并不意味着 Math 数据可以无限重复，也不意味着 Code 数据不应该上采样。实际策略仍应通过 domain validation loss、任务能力和 contamination audit 共同确定。

### OOD 稳定不等于没有代价

在总 high-quality domain fraction 与 web fraction 都固定的控制实验中，OOD loss 的变化较小。这说明 repeated domain token 的主要影响可以局部化到该 domain，而不是必然破坏所有通用能力。

但这不代表重复一定“没有通用代价”：

- 论文使用的是特定 OOD validation corpus，不覆盖所有能力；
- 固定 domain fraction 已经排除了 web token 被挤出的混淆因素；
- downstream reasoning、code execution、long-context 和 memorization 风险不一定由平均 OOD loss 反映。

因此，OOD 稳定应理解为一个受控条件下的结果，而不是普遍安全保证。

## 对训练配方的启发

### 把 repetition 记录成显式 data-mix 变量

训练记录不应只保存某个 domain 的最终采样比例，还应同时保存：

- unique document / token 数量；
- repetition count 或 expected exposure count；
- domain 在训练 token stream 中的最终占比；
- 不同文档的 exposure 分布，而不只是平均值；
- 重复发生在 document、sample、sequence 还是 token 层面；
- 重复前后的 dedup、quality filtering 和 contamination 结果。

对于一个 domain，至少要区分：

$$
\text{domain presentations}
=
\text{unique domain tokens}
\times
\text{repetition count}
$$

只有把这两个量分开，后续才知道能力提升来自更多 unique coverage，还是来自更高 exposure。

### 用 proxy model 做 domain-wise sweep

论文支持一种成本可控的调参流程：

```text
固定 tokenizer、模型族、TPP 和优化 recipe
  -> 为每个 domain 选择代表性的 unique-domain allocation fraction
  -> 在 proxy model 上 sweep repetition count
  -> 记录 IID loss、OOD loss 和下游能力
  -> 拟合 loss-repetition 曲线并确定候选区间
  -> 在更大模型上做小范围确认
```

关键是 proxy model 与 target model 使用相同的 TPP，而不是只保持相同的训练步数或相同的 domain token 数。否则 proxy 上的 repetition 结论可能属于另一个 scaling regime。

### 训练中应设置 repetition-sensitive validation

每个被上采样或重复的 domain 都应有独立 held-out validation。建议同时画出：

- domain IID loss 随 consumed tokens 的曲线；
- domain OOD / alternate-source loss；
- general web loss；
- 目标下游任务或 capability probe；
- 训练 loss 与 validation loss 的 gap；
- 文档 exposure count 与 memorization / contamination 指标。

如果 domain loss 继续下降但 OOD 和任务能力开始退化，说明 repetition 可能已经超过有效区间。如果总 loss 下降但目标 domain loss 不动，则更可能是该 domain 的采样比例或数据质量不足，而不是训练预算不足。

对固定模型规模的实验，建议先使用一个小型 factorial design 分开这几种情况：固定 $U$ 扫描 $E$，观察重复曲线；固定 $D_{\text{seen}}$ 同时改变 $U$ 与 $E$，观察 unique coverage 与 repeated exposure 的替代关系；再在固定 $E$ 下增加 $U$，观察更多 unique data 是否继续带来收益。这样得到的结论才不会把“数据覆盖不足”和“重复暴露过多”误判成同一个问题。

### Repetition 与 learning-rate schedule 要一起调

在数据稀缺的训练阶段，如果使用较早 decay 的 learning-rate schedule，重复数据可能更早进入精细拟合。候选方案可以包括：

- 延后目标 domain 的 decay 阶段；
- 在 data annealing 与 repetition-heavy 阶段保持更高的有效 learning rate；
- 通过 domain-specific sampler 控制重复 exposure，而不是直接复制数据文件；
- 对高风险 domain 保留 unique data holdout，避免 validation 被重复样本污染。

这些做法需要和整体优化稳定性、已有模型能力保持共同评估，不能把延后 decay 当成无条件的 repetition 修复手段。

## 与 Deduplication 的关系

这篇论文中的 repetition 与 [[training/data-engineering/deduplication|Deduplication]] 不是同一个概念。

- Deduplication 关注原始语料中不应存在的重复：镜像网页、重复抓取、fork、模板、benchmark overlap 和无信息复制；
- Repetition 关注训练配方中有意安排的 exposure：为了维持高价值 domain 的采样比例，受控地重新使用已清洗的数据。

同一份数据可以先经过严格的 source-level dedup，再在训练时被有意上采样。反过来，如果训练阶段的 repetition 没有被记录，后续只能看到“数据集大小”，无法知道模型究竟见过多少次同一内容。

工程上更稳妥的记录方式是保留两层统计：

1. **Content uniqueness**：原始 corpus 中有多少 unique document、paragraph 和 token cluster；
2. **Training exposure**：训练 token stream 中每类内容被呈现了多少次。

这样才能同时回答“数据是否重复得太多”和“目标 domain 是否被采样得足够多”。

## 局限与疑问

### 多 domain 同时重复的交互尚未研究

每次实验只重复一个 high-quality domain。真实 pretraining 往往同时对 code、math、multilingual、long-context 和 instruction-like data 做上采样。多个 domain 同时重复时，会产生：

- 总 mixture fraction 重新分配；
- 不同 domain 之间的 transfer 或 interference；
- 多个高 exposure domain 共同挤压 web data；
- repetition-induced memorization 的叠加。

论文在结尾将此列为后续方向，因此不能直接用单 domain sweep 推导完整 data mix 的全局最优解。

### Domain validation loss 不是完整的 domain value

低 validation loss 与更高 repetition tolerance 存在强相关，但这不等于低 loss domain 对最终模型最重要。某些高价值能力可能稀有、难学、validation loss 较高，却仍然值得投入更多数据或采用专门训练目标。

因此，domain loss 适合作为 repetition 的一个调参信号，不应替代下游能力、知识覆盖和安全评估。

### 领域划分和数据质量决定结论的外推范围

论文只测试四个 high-quality domain，且每个 domain 的数据选择、质量、语言、tokenizer 和 validation set 都有特定实现。不同数据管线可能改变 domain loss、重复敏感性和最优 repetition。

尤其是 code、medical 和 math 内部差异很大：代码语言、项目类型、数学难度、医学文体和来源可靠性都会影响有效 exposure。将每个大类视为同质 domain，可能掩盖更细粒度的数据差异。

### 下游能力与记忆风险仍需更多评估

论文主要使用 pretraining validation loss 和 OOD loss，尚未系统报告 code execution、数学推理、事实记忆、训练数据 memorization 或 benchmark contamination 的完整变化。重复更多可能降低 average loss，却增加具体样本记忆；也可能提升 domain benchmark，却损害生成多样性。

### 理论模型与真实 Transformer 之间存在距离

one-hot linear regression 提供了清晰的 signal acquisition / noise fitting 分解，但真实 Transformer 具有表示学习、token correlation、optimizer dynamics、attention、mixture interaction 和非线性 transfer。理论结果适合作为机制解释，不应当直接当作真实 LLM 的定量预测公式。

## 研究判断

这篇论文对训练数据设计最值得保留的判断是：**高质量数据稀缺时，重复不是简单的“数据污染”，也不是新数据的等价替代，而是一个受 domain learnability、模型规模、总训练预算和优化 schedule 共同约束的训练变量。**

对实际 pretraining / mid-training 配方而言，更合理的决策顺序是：

1. 先确认 domain 的 unique data 是否已经完成清洗、跨源去重和 contamination audit；
2. 再测量 domain validation loss、数据质量和目标能力，判断它是否值得被上采样；
3. 在固定 TPP 的 proxy model 上按 domain 做 repetition sweep；
4. 同时观察 IID、OOD、下游能力和 memorization-sensitive 指标；
5. 在目标模型上使用保守的 repetition 区间，并根据训练过程中的 loss gap 和能力评测调整；
6. 将 repetition count、最终 exposure 和 schedule 作为训练配方的一部分长期记录。

论文没有证明“某个固定 repetition count 对所有模型和数据都最优”。它真正建立的是一种更严谨的实验框架：**先区分数据数量、内容多样性和 exposure，再在与目标 scaling regime 一致的条件下估计每个 domain 的重复容忍度。**

## 相关知识链接

- [[training/pretraining/data-mix|Data Mix]]
- [[training/data-engineering/deduplication|Deduplication]]
- [[training/data-engineering/data-engineering|Data Engineering]]
- [[training/scaling/scaling-law|Scaling Law]]
- [[training/scaling/model-data-compute|Model Data and Compute]]
- [[training/pretraining/compute-optimal|Pretraining Compute Optimal]]
- [[training/pretraining/pretraining|Pretraining]]
- [[sources/papers/2020-scaling-laws-for-neural-language-models|Scaling Laws for Neural Language Models]]
- [[sources/papers/2023-deduplicating-training-data-makes-language-models-better|Deduplicating Training Data Makes Language Models Better]]
