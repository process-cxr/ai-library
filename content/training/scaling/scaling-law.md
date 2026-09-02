---
title: Scaling Law
created: 2026-03-28
published: 2026-03-28
modified: 2026-05-31
type: topic
status: mature
area: training
tags:
  - training
  - scaling
---

Scaling Law 指模型性能随模型规模、数据规模、训练计算量等变量变化而呈现出的经验规律。在大语言模型训练中，它通常用来回答三个问题：

1. 如果继续增加参数量、训练 token 或计算量，loss 大约会怎样下降？
2. 在固定训练预算下，应该把资源分配给更大的模型，还是更多的数据？
3. 当前训练结果是否符合预期，还是暴露了数据、优化或实现问题？

Scaling law 不是物理定律，而是对特定模型族、数据分布、训练目标和优化 recipe 的经验拟合。它的价值在于让大规模训练从“拍脑袋试规模”变成“用小规模实验外推并管理风险”。

## 基本形式

语言模型 scaling law 最常见的观测量是 validation cross-entropy loss。对于 next-token prediction，loss 可以写成：

$$
\mathcal{L}(\theta)
= -\mathbb{E}_{x_t \sim p^\*}\log p_\theta(x_t \mid x_{<t})
$$

它与 [[fundamentals/information-theory/cross-entropy|Cross Entropy]] 和 [[fundamentals/information-theory/perplexity|Perplexity]] 直接相关。loss 越低，表示模型对真实 token 分布的平均预测越好。

在经验拟合中，loss 常被近似写成幂律下降形式：

$$
L(N, D) \approx L_\infty + A N^{-\alpha} + B D^{-\beta}
$$

其中：

- $N$ 是非 embedding 或总参数量，具体口径必须在实验中保持一致；
- $D$ 是训练 token 数；
- $L_\infty$ 是在当前数据分布和任务定义下不可消除的 irreducible loss；
- $A,B,\alpha,\beta$ 是通过实验拟合得到的常数；
- 这个形式省略了优化不足、架构差异、数据质量差异和长上下文注意力成本等因素。

如果直接以训练计算量 $C$ 作为自变量，也常看到类似：

$$
L(C) \approx L_\infty + E C^{-\gamma}
$$

这类式子适合做高层预算外推，但不能单独告诉我们 $C$ 应该分给 $N$ 还是 $D$。真正做训练决策时，需要回到 [[training/scaling/model-data-compute|Model Data and Compute]] 和 [[training/scaling/compute-optimal|Compute Optimal]]。

## 为什么是幂律

在 log-log 坐标下，很多大规模神经网络实验会出现近似直线趋势，这意味着性能改善不是线性收益，而是递减收益：规模越大，继续降低同样幅度的 loss 需要越来越多资源。

直观上，模型训练同时受到三类瓶颈限制：

- **Model-limited**：模型容量不足，无法表示足够复杂的分布；
- **Data-limited**：训练 token 不足，模型没有见到足够多样的模式；
- **Optimization-limited**：给定模型和数据，本应达到的 loss 因学习率、batch、稳定性或实现问题没有达到。

Scaling law 的拟合只在 optimization 足够稳定时才有意义。如果小规模实验本身有严重 loss spike、数据污染、batch 不合理或学习率 schedule 异常，外推会把训练问题误认为规模规律。

## Kaplan Scaling Law

Kaplan 等人的 Scaling Laws for Neural Language Models 系统研究了语言模型 loss 与参数量、数据量、计算量之间的关系，给出了早期大模型扩展的重要经验结论：

- loss 随模型参数、数据和计算量呈现平滑的 power-law 改善；
- 在一定范围内，模型规模对 loss 的影响非常强；
- 对固定 compute 的最优分配倾向于更快增大模型规模，而训练 token 增长相对较慢；
- 大模型即使没有完全训练到数据最优，也可能在给定 compute 下取得较好 loss。

这个结论解释了 GPT-3 时期“更大模型 + 相对有限 token”路线的合理性。但它依赖当时的模型族、数据设置、训练范围和拟合方式。后来的 Chinchilla 工作重新估计了 compute-optimal 分配，认为许多大模型在 token 数上显著 undertrained。

Kaplan 论文的关键拟合值可以作为早期 scaling baseline：

$$
L(N)\approx\left(\frac{N_c}{N}\right)^{\alpha_N},
\qquad \alpha_N\approx0.076
$$

$$
L(D)\approx\left(\frac{D_c}{D}\right)^{\alpha_D},
\qquad \alpha_D\approx0.095
$$

在其 WebText2、BPE tokenizer 和模型族下，联合 model-data 拟合得到 $\alpha_N\approx0.076$、$\alpha_D\approx0.103$。这些 exponent 的价值在于描述边际收益，而不是提供脱离实验上下文的常数；$N_c$、$D_c$ 等 scale constant 会随 tokenizer、loss normalization 和数据分布改变。

论文同时给出两个容易被忽略的结论：

- model shape 在合理范围内弱于总 non-embedding scale，但极浅模型和极端 depth-to-width ratio 会偏离主趋势；
- 对 Books、Wikipedia、Common Crawl 等其他文本分布，loss 仍随模型规模平滑改善，且与 WebText2 validation loss 大致保持 domain-specific offset。

因此，scaling pilot 不能只拟合一个总 loss。应在固定 tokenizer、data mix、optimizer、batch 和 learning-rate schedule 的前提下，同时记录 domain validation loss，检查跨分布 transfer 是否仍然沿着相似趋势变化。

### Overfitting、Training Horizon 与 Batch

Kaplan 论文将 model size 与 dataset size 的联合关系写成：

$$
L(N,D)=
\left[
\left(\frac{N_c}{N}\right)^{\alpha_N/\alpha_D}
 +\frac{D_c}{D}
\right]^{\alpha_D}
$$

在该拟合下，overfitting penalty 主要由 $N^{\alpha_N/\alpha_D}/D\approx N^{0.74}/D$ 决定。它给出了一个重要的诊断视角：增加参数和增加数据不是互相独立的两条收益曲线，模型扩大后需要更多数据才能保持相近的 data-limited 风险。

训练时长也有独立的 scaling：

$$
L(N,S_{min})=
\left(\frac{N_c}{N}\right)^{\alpha_N}
+\left(\frac{S_c}{S_{min}}\right)^{\alpha_S},
\qquad \alpha_S\approx0.76
$$

论文用 critical batch size 把串行 steps 与处理 token 数联系起来，拟合得到：

$$
B_{crit}(L)\approx\frac{B^*}{L^{1/\alpha_B}},
\qquad B^*\approx2\times10^8\ \text{tokens},
\qquad \alpha_B\approx0.21
$$

$B_{crit}$ 主要由当前 loss 决定，而不是直接由参数量决定。训练越接近低 loss，gradient noise scale 越大，适合的 batch 往往也越大。这个结论对 batch scaling 和分布式数据并行的规划有参考价值，但不应脱离具体优化器、序列长度和硬件系统直接套用。

## Chinchilla Revision

Chinchilla 的核心修正是：在固定训练 compute 下，应该更均衡地扩大模型参数量和训练 token 数。它的经验结论常被简化为：

$$
D \approx 20N
$$

也就是 dense decoder-only LM 在特定实验设置下，compute-optimal 训练大约需要每个参数 20 个训练 token。这个比例不是普适常数，而是一个重要的经验锚点：如果一个 70B dense LM 只训练了几百 billion tokens，它很可能是 data-undertrained；如果一个 1B 模型被训练到数十 trillion tokens，则可能已经接近模型容量瓶颈。

Chinchilla-style scaling 的意义不只是“多喂数据”。它改变了训练预算的默认思路：

- 参数量越大不一定越划算，尤其当训练 token 不足时；
- 训练更多高质量 token 可以让较小模型达到或超过更大但 undertrained 的模型；
- 推理成本也会受益，因为较小模型每生成一个 token 的计算和显存更低；
- 数据质量、重复训练和多语言覆盖会决定“更多 token”是否真的有效。

## Loss、Perplexity 与能力

Scaling law 通常拟合 validation loss，但用户关心的是模型能力。两者相关，却不等价。

Cross-entropy loss 是一个平均 token 预测指标。它对常见模式非常敏感，但可能低估稀有能力、推理能力、工具使用能力或安全行为。Perplexity 是 loss 的指数变换：

$$
\mathrm{PPL} = \exp(L)
$$

如果 loss 使用 natural log，PPL 表示模型平均在每个位置上还“困惑于”多少等效选择。PPL 适合比较同 tokenizer、同数据分布、同评测集上的 language modeling 能力；不适合直接跨 tokenizer、跨数据集、跨模型族比较。

因此，scaling 实验通常要同时看：

- validation loss / perplexity；
- domain-specific loss，例如 code、math、中文、学术文本；
- downstream benchmark；
- contamination 检查；
- post-training 后的能力转移。

## Emergent Abilities 的谨慎解释

大模型研究中常说某些能力会在规模达到阈值后“涌现”。从 scaling law 角度看，需要区分两种现象：

- **真实的机制性转变**：模型内部表示、搜索或组合能力可能在规模后出现新结构；
- **度量造成的阈值感**：如果 benchmark 是离散正确率，底层连续能力的平滑提升也可能表现为突然跨过及格线。

因此，emergent abilities 不应被简单理解为“规模到了就自动出现”。更严谨的表述是：模型规模、数据覆盖、训练目标和评测指标共同决定某些能力何时在可观测指标上变得显著。

## 用 Scaling Law 做训练决策

一个实用流程是：

1. 在多个小规模模型、多个 token budget 上训练 pilot runs。
2. 确保所有 run 使用一致的数据、tokenizer、架构比例、优化 recipe 和评测集。
3. 拟合 loss 对 $N$、$D$ 或 $C$ 的趋势。
4. 检查 residual：哪些点明显偏离拟合线，是否来自数据或训练不稳定。
5. 用拟合结果估算目标规模的 loss、训练时间和风险。
6. 把 compute-optimal 结果和显存、并行、推理成本约束一起审查。

Scaling law 最适合回答“在相似条件下继续扩大会怎样”。当训练数据、模型结构、目标函数或 tokenizer 发生大变化时，原来的曲线只能作为参考，不能当作保证。

## 常见误解

- **误解：scaling law 能预测所有能力。**
  更准确地说，它主要预测特定评测分布上的平均 loss。能力评测需要单独设计。

- **误解：参数越多越 compute optimal。**
  在固定 compute 下，参数和 token 存在分配问题。过大的模型如果训练 token 不足，会浪费一部分容量。

- **误解：Chinchilla 的 20 tokens/parameter 是定律。**
  它是特定 dense LM 设置下的经验规则，会随数据质量、模型族、训练目标、重复 epoch 和评测目标改变。

- **误解：只要 loss 符合 scaling law，模型就可靠。**
  loss 正常不代表没有数据污染、安全问题、偏见、事实错误或 post-training 难题。

## 相关概念

- [[training/scaling/model-data-compute|Model Data and Compute]]
- [[training/scaling/compute-optimal|Compute Optimal]]
- [[training/scaling/training-budget|Training Budget]]
- [[training/pretraining/compute-optimal|Pretraining Compute Optimal]]
- [[training/pretraining/data-mix|Data Mix]]
- [[fundamentals/information-theory/cross-entropy|Cross Entropy]]
- [[fundamentals/information-theory/perplexity|Perplexity]]
- [[sources/papers/2020-scaling-laws-for-neural-language-models|Scaling Laws for Neural Language Models]]
- [[sources/papers/2022-training-compute-optimal-large-language-models|Training Compute-Optimal Large Language Models]]
