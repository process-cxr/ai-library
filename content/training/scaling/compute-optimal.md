---
title: Compute Optimal
created: 2026-03-28
published: 2026-03-28
modified: 2026-05-31
type: topic
status: mature
area: training
tags:
  - scaling
  - compute-optimal
---

Compute Optimal 指在固定训练计算预算下，选择最合适的模型参数量 $N$ 和训练 token 数 $D$，使目标 validation loss 尽可能低。它不是问“最大能训练多大的模型”，而是问“同样算力怎样分配最划算”。

这个问题是 Scaling Law 在训练决策中的核心落点：如果 $C \approx 6ND$，那么固定 $C$ 时，增大 $N$ 通常意味着减少 $D$，增加 $D$ 通常意味着选择更小的 $N$。Compute optimal 就是在这条约束曲线上找最优点。

## 形式化问题

对于 dense decoder-only Transformer，训练 compute 可以粗略写成：

$$
C \approx 6ND
$$

假设 validation loss 可以近似表示为：

$$
L(N,D) = L_\infty + A N^{-\alpha} + B D^{-\beta}
$$

compute-optimal 问题就是：

$$
\min_{N,D} L(N,D)
\quad \text{s.t.} \quad
6ND \le C_0
$$

其中 $C_0$ 是训练计算预算。直观地说，最优点附近不应该出现“把少量 compute 从参数转给数据就能显著降低 loss”的情况，也不应该出现“把少量 compute 从数据转给参数就能显著降低 loss”的情况。参数和数据的边际收益应大致平衡。

这个形式只是解释框架。真实训练还要考虑：

- 数据质量和可用 token 上限；
- long context attention 的额外成本；
- MoE 的 active parameters 与 total parameters 区分；
- optimizer、batch size 和学习率 schedule；
- 分布式训练通信效率；
- 目标能力不完全等价于平均 validation loss。

## Kaplan 与 Chinchilla 的差异

早期 Kaplan-style scaling 给出的 compute-optimal 分配更偏向增大模型规模。其经验拟合为：

$$
N_{opt}\propto C^{0.73},
\qquad
D_{opt}\propto C^{0.27}
$$

同时，batch size 约按 $C^{0.24}$ 增长，而 batch-adjusted serial steps 仅按 $C^{0.03}$ 增长。其经验影响是：在给定 compute 下，训练非常大的模型，即使 token 数相对不多，也可能取得不错 loss。

这一结果建立在 Kaplan 论文对 WebText2、Transformer learning curve 和 critical batch size 的联合拟合上。它不能简单理解为“参数永远比数据重要”，而应理解为特定实验 regime 下的 compute-efficient frontier。论文还指出，这种分配会让数据增长速度逐渐落后于控制 overfitting 所需的数据增长速度，因此 scaling law 在更大规模上必须修正。

Chinchilla-style scaling 重新评估了这一点，认为许多大模型是 data-undertrained。它给出的重要经验判断是：固定 compute 下，模型大小和训练 token 数应更均衡地增长。常见简化规则是：

$$
D \approx 20N
$$

| 观点 | 资源分配倾向 | 典型风险 |
|---|---|---|
| Kaplan-style | 更快扩大模型参数，token 增长较慢 | 大模型 undertrained，训练和推理成本高 |
| Chinchilla-style | 参数量与 token 数更均衡增长 | 需要更多高质量数据，数据重复和质量成为瓶颈 |

这不是说 Kaplan “错误”而 Chinchilla “永远正确”。更准确地说，不同实验范围、数据质量、模型族和拟合方法会给出不同分配。Chinchilla 的长期影响在于提醒训练者：参数不是唯一规模，数据 token 也是同等重要的扩展轴。

## Chinchilla 的实验依据

Chinchilla 通过 400 多个不同规模和训练时长的 Transformer language model 估计 compute-optimal frontier。论文使用三种方法交叉验证参数量 $N$ 与训练 token 数 $D$ 随 compute $C$ 的增长关系：从训练曲线中取每个 compute 位置的最低 loss、构造 IsoFLOP profiles 寻找 loss valley，以及拟合

$$
L(N,D)=E+\frac{A}{N^\alpha}+\frac{B}{D^\beta}
$$

后三种方法得到的 exponents 都接近均衡扩展：$N_{opt}\propto C^{0.46\sim0.50}$，$D_{opt}\propto C^{0.50\sim0.54}$。在 Gopher 的训练预算附近，论文据此训练了 70B 参数、约 1.4T tokens 的 Chinchilla；它与 280B、约 300B tokens 的 Gopher 使用相近的 training FLOPs，但在 MMLU、BIG-bench、阅读理解、常识和问答等任务上整体更好。

这个实验支持的是“固定 compute 下需要重新平衡模型与数据”，而不是一个脱离上下文的 20 tokens/parameter 定律。Chinchilla 与 Gopher 在 optimizer、tokenizer、precision 和 batch 等方面并非完全相同，且大规模直接对照数量有限，因此实际项目仍需要 pilot runs 和目标能力评测。

## Undertraining 与 Overtraining

Compute optimal 的实践语言通常是 undertrained / overtrained：

- **Undertrained model**：模型参数量相对于训练 token 数过大。它可能很有容量，但没有被足够数据训练到合理 loss。
- **Overtrained small model**：模型参数量相对于训练 token 数过小。继续增加 token 仍可能有收益，但收益被模型容量限制，loss 下降变慢。

这两个词不是绝对评价，而是相对于某个 compute budget 和目标分布的评价。一个小模型在 Chinchilla 意义上可能“overtrained”，但如果产品要求低推理成本、高吞吐和端侧部署，它仍可能是正确选择。一个大模型在 compute-optimal 意义上 undertrained，但如果后续会继续训练或需要更高容量，它也可能有研究价值。

## 为什么 Compute Optimal 重要

Compute optimal 改变的是训练项目的默认问题。

错误的问题是：

> 我们最多能把模型做多大？

更好的问题是：

> 在可用数据、可用硬件、目标能力和部署约束下，哪个 $N,D$ 组合带来的总收益最大？

这个问题会影响：

- 预训练参数规模；
- 总 token budget；
- tokenizer 和 data mix；
- 是否做 continued pretraining；
- 是否训练小而充分的模型，还是大而可继续扩展的模型；
- 推理成本和 serving 架构；
- 后训练阶段能否稳定提升。

## Kaplan-style 的训练时间视角

Kaplan 论文将 loss 拆成 model-size error 与 training-time error：

$$
L(N,S_{min})=
\left(\frac{N_c}{N}\right)^{\alpha_N}
+\left(\frac{S_c}{S_{min}}\right)^{\alpha_S}
$$

其中 $\alpha_S\approx0.76$。结合 critical batch size 后，论文认为 compute-efficient training 不必把模型训练到非常接近 convergence：在其拟合中，约高于 converged loss 10% 的状态已经接近 compute-efficient frontier。与之相比，把差异压到约 2% 需要更多 serial updates 和更高 compute。

这个结论更适合作为训练预算中的 trade-off 参考，而不是统一 stopping rule。模型是否应该继续训练，要同时考虑数据是否会重复、目标能力是否仍在增长、推理成本、后续适配和部署约束。

## Critical Batch Size

固定 compute 时，batch size 不能只按“越大越快”理解。Kaplan 论文沿用 gradient noise scale 的经验关系：在 $B$ 不超过 $B_{crit}$ 的区域，增大 batch 可以减少串行 steps，同时保持相近的 compute efficiency；当 $B\gg B_{crit}$ 时，继续增大 batch 的收益明显递减。

$$
B_{crit}(L)\approx\frac{B^*}{L^{1/\alpha_B}}
$$

其 WebText2 拟合约为 $B^*\approx2\times10^8$ tokens、$\alpha_B\approx0.21$。$B_{crit}$ 主要随当前 loss 变化，不能仅由 model size 预先决定。使用分布式 data parallel 时，global batch、gradient accumulation、每卡 micro-batch 和当前 loss 应一起记录；否则不同 run 的 update steps 不能直接比较。

## 实践决策流程

一个可执行的 compute-optimal 规划流程：

1. 明确目标：通用 base model、domain model、code model、math model、embedding model 或 assistant base。
2. 估算可用高质量 token，而不是只统计原始文本大小。
3. 根据预算 $C_0$ 和 $C \approx 6ND$ 生成候选 $(N,D)$。
4. 用 Chinchilla-style ratio 作为初始锚点，而不是机械约束。
5. 检查每个候选的显存、并行、训练时间和 checkpoint 成本。
6. 做小规模 pilot runs，拟合 loss 趋势和 domain validation loss。
7. 选择不仅总 loss 低，而且能力分布、数据风险和部署成本更合理的方案。

如果没有足够高质量数据，不能简单地把 $D$ 设到 20N。重复 epoch、低质量网页和合成数据都会改变有效 token 的价值。如果推理成本是关键约束，较小模型训练更多 token 可能比大模型更合适。如果目标是探索更高能力上限，则可能接受短期 undertraining，为后续 continued pretraining 留空间。

## 与 Inference Cost 的关系

固定训练 compute 下，Chinchilla-style 模型通常比 Kaplan-style 模型更小、训练 token 更多。这带来一个重要后果：推理时每生成一个 token 的成本大致随 active parameter count 增长，因此较小但训练充分的模型可能同时拥有较好的 loss 和较低 serving 成本。

但推理成本不是唯一目标。大模型可能在以下方面仍有优势：

- 更高容量和更强 in-context learning；
- 更高的上限，适合后续 post-training；
- 更强的复杂任务表示能力；
- 更好的 tool use、长程推理或多领域迁移潜力。

因此 compute optimal 应与 deployment optimal 分开分析。前者优化训练 compute 下的 loss，后者还要优化延迟、吞吐、显存、量化、KV cache、并发和用户体验。

## 边界与失败模式

- **数据质量被忽略**：把低质量 token 当作等价高质量 token，会高估增加 $D$ 的收益。
- **重复 epoch 被误算**：重复数据可以继续训练，但边际收益和过拟合风险不同于新数据。
- **只看总 loss**：总 loss 改善可能掩盖 code、math、多语言或安全域退化。
- **外推过远**：小模型拟合到的曲线可能无法跨越架构、batch、optimizer 或数据变化。
- **忽视长上下文成本**：当 $S$ 很大时，attention 成本和 activation memory 会改变可行规模。
- **把经验比例当定律**：20 tokens/parameter 是有用锚点，不是所有训练任务的最优解。

## 相关概念

- [[training/scaling/scaling-law|Scaling Law]]
- [[training/scaling/model-data-compute|Model Data and Compute]]
- [[training/scaling/training-budget|Training Budget]]
- [[training/pretraining/compute-optimal|Pretraining Compute Optimal]]
- [[training/pretraining/data-mix|Data Mix]]
- [[training/optimization/training-memory-estimation|Training Memory Estimation]]
- [[sources/papers/2020-scaling-laws-for-neural-language-models|Scaling Laws for Neural Language Models]]
- [[sources/papers/2022-training-compute-optimal-large-language-models|Training Compute-Optimal Large Language Models]]
