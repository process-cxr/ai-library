---
title: Compute Optimal
created: 2026-02-22
published: 2026-02-22
modified: 2026-05-31
type: topic
status: mature
area: training
tags:
  - pretraining
  - scaling
---

Pretraining Compute Optimal 是把 [[training/scaling/compute-optimal|Compute Optimal]] 原理落到预训练项目中的决策问题。Scaling 层面的 compute optimal 关注固定 FLOPs 下 $N$ 与 $D$ 的理论分配；预训练层面的 compute optimal 还要同时考虑数据可得性、tokenizer、data mix、上下文长度、训练稳定性、评测目标和部署成本。

一句话说：它不是只算一个“最佳参数量”，而是设计一个可训练、可评估、能力分布合理、后续可继续演化的 pretraining plan。

## 与 Scaling Compute Optimal 的关系

理论层面的粗估常从 dense Transformer 公式开始：

$$
C_{\mathrm{train}} \approx 6ND
$$

其中 $N$ 是参数量，$D$ 是训练 token 数。Chinchilla-style 经验常把 $D \approx 20N$ 当作 dense LM 的初始锚点。这个锚点适合帮助识别 undertrained 大模型，但预训练项目不能机械套用它。

Chinchilla 的原始证据来自一组 dense autoregressive Transformer 实验：论文在约 70M 到超过 16B 参数、约 5B 到超过 400B/500B training tokens 的范围内构造多组训练曲线，再外推到 70B Chinchilla 与 280B Gopher 的同 compute 对照。三种估计方法都得到接近 $N_{opt}\propto C^{0.5}$、$D_{opt}\propto C^{0.5}$ 的结果。这里的 $D$ 是特定数据分布和训练 regime 下的 token 数，不能直接等同于任意领域或 agent trajectory 的名义 token 数。

在真实预训练中，至少还有五个额外约束：

- **有效数据量**：高质量、去重、无污染、目标相关的 token 是否足够；
- **训练稳定性**：batch size、学习率、优化器、loss spike、梯度裁剪是否支持目标规模；
- **系统可行性**：显存、并行策略、I/O、checkpoint、wall-clock time 是否可接受；
- **目标能力**：通用语言、code、math、多语言、长上下文和领域知识之间如何取舍；
- **部署约束**：最终模型的推理成本、延迟、吞吐、量化和 serving 预算。

因此，pretraining compute optimal 是一个受约束的工程-科学联合问题。

## 规划变量

预训练计划通常需要同时确定：

| 变量 | 决策问题 | 影响 |
|---|---|---|
| 参数量 $N$ | 模型容量和推理成本多大 | loss 上限、显存、训练/推理 FLOPs |
| token 数 $D$ | 总共训练多少 token | 数据覆盖、训练时间、over/undertraining |
| context length $S$ | 每条序列的上下文长度 | 长程能力、attention 成本、activation memory |
| data mix | 各数据域采样比例 | 能力分布、偏差、安全性、domain loss |
| tokenizer | 词表和分词算法 | 压缩率、多语言/code/math 表示、PPL 可比性 |
| batch tokens | 每步处理多少 token | 优化稳定性、吞吐、generalization |
| learning rate schedule | warmup/decay/最终 LR | 收敛速度、loss spike、训练末期收益 |
| checkpoint/eval cadence | 多久保存和评测 | 可恢复性、成本、异常发现速度 |

这些变量互相耦合。例如，增大 context length 会提高长程能力潜力，但会增加 activation memory 和 attention FLOPs；改变 tokenizer 会改变 $D$ 的统计口径；提高 code/math 比例会改变 domain validation loss 和最终能力分布。

## 设计流程

一个相对稳健的 pretraining compute-optimal 流程如下：

1. **明确模型目标**：通用 base、中文 base、code model、math model、领域模型、低延迟小模型或后续 assistant base。
2. **估算有效 token**：从原始 corpus 中扣除低质、重复、污染、格式错误和不可用数据，得到可训练 token 范围。
3. **生成候选规模**：用 $C \approx 6ND$ 和 tokens-per-parameter ratio 给出多个 $(N,D)$ 方案。
4. **检查系统约束**：用 [[training/optimization/training-memory-estimation|Training Memory Estimation]] 验证显存、并行、sequence length 和 checkpoint 是否可执行。
5. **做 pilot runs**：训练小规模模型，观察 total validation loss、domain loss、loss spike 和数据质量问题。
6. **拟合并修正 scaling**：用 pilot 结果更新对目标规模 loss 和风险的估计。
7. **确定主训练方案**：选择一个不是单点最优、而是对数据、系统和目标能力都稳健的配置。
8. **保留中途决策点**：在若干 token milestone 检查是否继续训练、调整 data mix、延长训练或提前停止。

这个流程的关键是不要把预训练看成一次不可回头的单次运行。大规模训练应当有 early diagnosis、mid-training decision 和 recovery plan。

## 何时偏离 Chinchilla-style 配比

Chinchilla-style 的 $D \approx 20N$ 是有用起点，但以下情况可能需要偏离：

- **高质量数据不足**：如果有效 token 不足，强行重复低质数据可能不如选择更小模型或更窄目标。
- **推理成本极敏感**：如果部署要求低延迟和低成本，可以选择较小模型并训练更多 token。
- **能力上限优先**：如果目标是探索更强推理、长上下文或后续 post-training 上限，可能接受较大模型短期 undertraining。
- **领域数据稀缺**：领域模型可能需要先通用预训练，再做 continued pretraining，而不是从头用少量领域数据训练大模型。
- **MoE 架构**：MoE 的 total parameters 和 active parameters 不同，不能直接用 dense LM 的 $N$ 口径。
- **长上下文训练**：超长 context 会改变 FLOPs 和 memory 构成，$6ND$ 粗估会低估 attention 和 activation 成本。

偏离不是问题，问题是没有说明偏离的目标和代价。

## Data Mix 是 Compute Optimal 的一部分

同样的 $D$ 可以产生完全不同的模型。如果 token 大多来自普通网页，模型可能通用性较好但代码和数学弱；如果 code/math 占比很高，模型可能在推理和程序能力上更强，但自然语言广度或多语言能力可能受影响。

因此，预训练中的 compute optimal 应该使用“有效 token”而不是“名义 token”。有效 token 取决于：

- 去重和近重复处理；
- 文档质量过滤；
- domain sampling weight；
- 语言覆盖；
- 数据污染控制；
- 合成数据比例；
- 是否包含推理轨迹、代码执行结果或学术文本。

这部分由 [[training/pretraining/data-mix|Data Mix]] 和 [[training/data-engineering/data-engineering|Data Engineering]] 承担。

## Continued Pretraining 的位置

如果从头预训练的 compute 或数据不足，可以把 compute optimal 拆成两个阶段：

1. 先训练一个通用 base model，保证语言、知识和通用表示；
2. 再用 [[training/mid-training/continued-pretraining|Continued Pretraining]] 注入领域、语言、代码、数学或长上下文能力。

这种做法的优点是降低从零训练的风险，并允许在已有模型上更快验证 data mix。但它也有边界：continued pretraining 不能无限弥补 base model 容量不足、tokenizer 不适配或早期数据缺陷；如果领域分布差异很大，还可能发生遗忘或通用能力退化。

## 评测与监控

预训练 compute-optimal 方案至少需要监控：

- total validation loss；
- 按 domain 拆分的 validation loss；
- held-out benchmark；
- 训练数据 contamination；
- loss spike 和梯度异常；
- 不同 checkpoint 的 downstream transfer；
- tokenizer 后的 token 分布变化；
- 训练后期 learning rate decay 是否仍产生有效收益。

只看训练 loss 很危险。训练 loss 下降可能来自重复数据、污染或过拟合；总 validation loss 改善也可能掩盖某些关键 domain 的退化。

## 常见失败模式

- **先定参数量再找数据**：容易得到大而 undertrained 的模型。
- **只按原始文本大小估 token**：忽略 tokenizer、去重、过滤和有效信息密度。
- **忽略 tokenizer 对 $D$ 的影响**：同一语料用不同 tokenizer 会得到不同 token 数，PPL 也不可直接比较。
- **把 data mix 当数据工程细节**：混合比例会决定能力分布，是训练目标的一部分。
- **pilot run 太少**：无法区分 scaling 趋势、数据问题和优化问题。
- **没有中途停机标准**：训练异常时只能凭感觉继续烧 compute。

## Data-centric Pilot Scale

[[sources/papers/2024-datacomp-lm|DataComp-LM]] 给出了一个与 compute-optimal 规划互补的经验：数据策略可以先在较小模型上进行筛选。论文设计了 400M-1x、1B-1x、3B-1x、7B-1x 和 7B-2x 五个 scale，发现 400M、1B、3B 的 dataset ranking 与 7B-1x 的 Pearson correlation 分别为 `0.838`、`0.956` 和 `0.982`。

这并不等于小模型能够准确预测目标模型的绝对性能，而是说明在固定 tokenizer、训练 recipe、token budget 定义和 evaluation suite 后，小规模 run 可以作为 data curation 的排序信号。实际决策仍应保留少量 target-scale confirmation，并检查长上下文、领域能力、遗忘和训练稳定性等 proxy 难以覆盖的指标。

## 相关概念

- [[training/scaling/compute-optimal|Compute Optimal]]
- [[training/scaling/model-data-compute|Model Data and Compute]]
- [[training/scaling/scaling-law|Scaling Law]]
- [[training/scaling/training-budget|Training Budget]]
- [[training/pretraining/data-mix|Data Mix]]
- [[training/pretraining/tokenizer|Tokenizer]]
- [[training/pretraining/objective|Training Objective]]
- [[training/mid-training/continued-pretraining|Continued Pretraining]]
