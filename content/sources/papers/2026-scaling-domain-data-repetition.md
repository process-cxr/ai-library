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

如果只看“重复会导致 overfitting”，结论会过于粗糙。高质量代码、数学、百科或医学数据往往比普通网页数据更难扩展；当模型规模和训练 token budget 同时增长时，固定规模的高质量数据会在 mixture 中被逐渐稀释。适度重复可以维持目标 domain 的训练占比，过度重复则会增加 memorization 和 noise fitting。论文试图刻画这两种效应如何随 domain、model size、unique data fraction、总训练预算和 learning-rate schedule 变化。

论文最重要的价值不是给出一个可以直接复制的 repetition 倍数，而是把 repetition 从一个数据清洗副作用，提升为一个需要单独调参、单独评估的 data-mix 变量。

## 研究问题

### 在高质量 domain data 稀缺时，重复是否仍然值得

设训练总 token budget 随模型规模增长，而某个目标 domain 的 unique data 规模增长较慢。如果不重复该 domain，它在训练 mixture 中的占比会下降，模型可能无法充分学习其中的知识和结构。重复可以提高该 domain 的 token presentations，但这些 presentations 并不等价于同量的新数据。

论文关注的不是“重复数据后训练 token 是否增加”，而是：在固定总训练 token budget 下，用重复的 domain token 替代 web token 或 unique token，最终 validation loss 和泛化能力会怎样变化。

### 传统的跨规模结论是否适用于固定 TPP

已有不少工作在不同模型规模之间固定训练数据量 $D$，然后观察重复数据造成的 overfitting。这个设置下，模型变大而数据不变，重复样本很快成为相对不足的监督信号，因此更大的模型可能更早出现过拟合。

本文采用另一种更贴近实际 LLM scaling 的设置：

$$
D_N = TPP \cdot N
$$

其中 $N$ 是模型规模，$D_N$ 是总训练 token 数，`TPP` 是固定的 tokens-per-parameter。此时模型变大，训练 token budget 也同步增加。论文要回答：在这一设置下，最优 repetition count 是否仍然随模型变大而下降。

### 最优 repetition 由什么决定

作者分别考察：

- 高质量 domain 的类型；
- model size；
- unique high-quality data fraction；
- 训练总 token budget 与 TPP；
- learning-rate schedule；
- 是否用 unique token 替换 repeated token；
- in-domain 与 out-of-domain validation loss。

这组变量的区分很重要。`unique data fraction` 描述有多少不同的 domain 内容，`repetition count` 描述同一批内容被重新看到多少次，而最终 domain token presentations 还取决于两者的乘积。把这几个量混在一起，无法判断收益来自更多 domain exposure，还是来自更多内容多样性。

## 核心主张

### 1. Repetition 的最优值强烈依赖 domain

在论文测试的 code、math、Wikipedia 和 medical 四个 domain 中，最优 repetition count 明显不同。正文与附录给出的整体趋势是：

- Math 通常在约 `5-6` 次附近达到较低的 in-domain validation loss；
- Code 通常在约 `4-5` 次附近达到较低 loss；
- Wikipedia 通常在约 `3-4` 次附近达到较低 loss；
- Medical 通常在约 `3-4` 次附近达到较低 loss。

这些数字只属于论文的训练规模、数据构造、优化 recipe 和 validation set，不能被当成通用配方。可迁移的结论是：**不同 domain 应拥有独立的 repetition sweep，而不是共享一个全局 epoch 数。**

### 2. 在固定 TPP 下，模型变大时最优 repetition 轻微增加

这一结论与“固定数据量时大模型更容易过拟合”并不矛盾。固定总数据量 $D$ 时，增加模型规模不会增加样本观察次数，因此模型更容易拟合重复样本中的噪声；固定 TPP 时，$D$ 随 $N$ 增加，模型获得更多 token exposure，knowledge acquisition 的收益可以持续更久。

论文用理论分析解释了这两个 regime 的差异：

- 固定 $D$：模型容量增加，但 observations 不增加，noise-fitting 更早占主导；
- 固定 $D/N$：model size 与 data budget 一起增长，信号学习收益可以抵消一部分更大的拟合能力，因此最优 repetition count 反而轻微上升。

这意味着在目标模型上直接沿用小模型的 repetition count 通常是保守的，但必须保持 proxy 与 target 的 TPP、数据定义和训练 recipe 可比。

### 3. 最优 repetition 与 domain final validation loss 强负相关

作者对每个配置中 repetition count 与最终 validation loss 的关系拟合二次曲线，并从曲线中估计连续的最优 repetition count。结果显示，最优 repetition 与该 domain 的最低 validation loss 之间的 Pearson correlation 约为 `-0.944`。

直观上，validation loss 较低的 domain 更容易被模型稳定吸收，重复 exposure 带来的 knowledge acquisition 收益可以持续更久；validation loss 较高的 domain 更早进入对样本特定模式和噪声的拟合阶段。这里的 final validation loss 更像一个 domain learnability / noise level 的可观测 proxy，而不是只表示该 domain 的数据量。

相比之下：

- 最优 repetition 与 model size 的 correlation 约为 `0.400`，是较弱的正相关；
- 与 unique data fraction 的 correlation 约为 `0.018`，在论文测试范围内几乎没有明显关系。

因此，一个实用的 proxy model 方案是：先在与目标模型保持相同 TPP 的小模型上，为每个 domain 做 repetition sweep，再用各 domain 的 validation loss 和曲线形状决定目标模型的候选 repetition 区间。

### 4. Unique data 比 repeated data 更有价值，但差异取决于 domain

前面的实验固定 unique data fraction，增加 repetition count。论文还做了互补实验：固定训练中目标 domain 的总 token fraction，用更多 repetition 替换更多 unique data，直接比较“少量 unique data 重复多次”和“更多 unique data 各看一次”。

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

对每个目标 domain $d$，作者从固定的 unique subset 中构造训练 mixture。unique high-quality token fraction 记为 $\alpha$，取值为：

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

因此，$e\alpha$ 决定最终训练流中该高质量 domain 的 token fraction，而 $\alpha$ 与 $e$ 分别控制 unique content 数量和每个内容的重复次数。每个实验只重复一个 high-quality domain，不同时重复多个 domain。

### 评测指标

作者对每个 run 同时评估：

- target domain 的 held-out validation loss，即 in-distribution / IID loss；
- 通用 pretraining validation corpus 上的 OOD loss；
- 在固定 domain fraction 的补充实验中，使用 ArXiv 和 News 作为 OOD validation。

论文将最终 validation loss 对 repetition count 做二次拟合：

$$
\widehat{L}_{d,N,\alpha}(e)=a_{d,N,\alpha}e^2+b_{d,N,\alpha}e+c_{d,N,\alpha}
$$

再以：

$$
\widehat{e}^{*}_{d,N,\alpha}=-\frac{b_{d,N,\alpha}}{2a_{d,N,\alpha}}
$$

估计连续的最优 repetition count。这样做的目的不是声称真实曲线必然是二次函数，而是减少只在 `1-7` 离散点上取最小值带来的粗糙性，便于比较 domain、model size 和 unique data fraction 的相关关系。

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

1. 模型容量之外、训练中无法表示的 knowledge；
2. 已经能表示但还没有被充分学到的 signal；
3. 模型对有限样本噪声和 sample-specific pattern 的拟合。

增加 repetition 或优化步数会继续降低 knowledge-acquisition error，但也会提高 noise-fitting error。最优 repetition 出现在 signal 的边际收益开始低于 noise-fitting 的边际代价时。

### 为什么较低 validation loss 的 domain 可以承受更多重复

在理论模型中，较低的噪声水平 $\sigma^2$ 不会改变 signal acquisition 项，却会降低 noise-fitting 项。因此模型可以进行更多轮优化，才会让 fitting noise 的代价超过继续学习 signal 的收益。论文的 Theorem 4.2 给出：噪声越低，最早达到最优 stopping time 不会更早；当噪声趋近于零时，最优 stopping time 近似按 $\log(1/\sigma^2)$ 增加。

这为实验中的负相关提供了机制解释：validation loss 较低的 domain 往往具有更低的有效噪声或更容易学习的结构，因此更能从 repetition 中获益。但这里的 domain loss 不是噪声方差的直接估计，只能作为经验 proxy。

### 固定数据预算与固定 TPP 的差异

在固定 $D$ 的条件下，增加 $N$ 会把更多更稀有的 knowledge unit 纳入模型容量，但不会增加它们的 observation count。理论上，当模型规模超过由 $D$ 和噪声水平决定的 crossover scale 后，新增容量更容易进入 noise-fitting，最优 stopping time 会不增反降。

在固定 $D/N$ 的条件下，$D$ 与 $N$ 同步增长。论文的 Theorem 4.4 给出一个渐近结果：在源分布和信号条件满足假设时，最优 stopping time 的量级随 $D^{\alpha/\beta}$ 增长。它不直接给出真实 LLM 的 repetition count，但说明为什么“固定数据量下的跨规模结论”不能直接套用到 compute-optimal 或 TPP-scaled 的训练。

## 实验结果的理解

### Repetition count 与 domain loss 的关系

论文把四个 domain 和多个模型规模、unique data fraction 的实验点放在一起比较。最优 repetition 的主要变化来源依次可以概括为：

1. domain 本身的可学习性和噪声水平；
2. model size 在固定 TPP 下带来的温和变化；
3. unique data fraction 在测试范围内的较小影响。

这里的“unique data fraction 影响较小”不能解读为 unique data 不重要。它只表示：当 $\alpha$ 在 `1/40` 到 `1/10` 的范围内变化时，达到最优 repetition 的位置相对稳定；绝对 validation loss 仍然会随 unique data fraction 改变。换言之，更多 unique data 可以整体降低 loss，但不一定改变曲线最低点所在的 repetition 区间。

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
  -> 为每个 domain 选择代表性的 unique data fraction
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
