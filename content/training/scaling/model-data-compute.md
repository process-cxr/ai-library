---
title: Model Data and Compute
created: 2026-03-28
published: 2026-03-28
modified: 2026-05-31
type: topic
status: mature
area: training
tags:
  - scaling
  - compute
---

Model Data and Compute 描述大模型预训练中三个核心变量的关系：

- $N$：模型参数量；
- $D$：训练 token 数；
- $C$：训练计算量，通常用 FLOPs 表示。

这三个量决定了训练规模的第一性预算。[[training/scaling/scaling-law|Scaling Law]] 关心它们如何影响 loss，[[training/scaling/compute-optimal|Compute Optimal]] 关心固定 $C$ 时如何分配 $N$ 和 $D$，[[training/scaling/training-budget|Training Budget]] 则把它们转成 GPU、时间、显存和成本。

## 基本记号

| 记号 | 含义 | 常见单位 |
|---|---|---|
| $N$ | 模型参数量，dense LM 中通常近似等于每个 token 激活的参数量 | B / M parameters |
| $D$ | 训练 token 总数，通常统计 tokenizer 后的 token | tokens |
| $C$ | 训练总计算量 | FLOPs |
| $B$ | global batch size，以 token 计更适合比较 | tokens/step |
| $T$ | optimizer update steps | steps |
| $S$ | sequence length / context length | tokens |

训练步数与 token 数的关系是：

$$
T \approx \frac{D}{B}
$$

这里的 $B$ 是 global batch tokens，而不是样本条数。例如 batch 中有 1024 条序列、每条 4096 tokens，则 $B \approx 4.19M$ tokens/step。实际训练还会受到 packing、padding、drop remainder 和 curriculum 的影响。

## Dense Transformer 的 FLOPs 近似

对于标准 dense decoder-only Transformer，预训练总 FLOPs 常用近似：

$$
C_{\mathrm{train}} \approx 6ND
$$

这个公式的含义是：每个 token 的一次 forward 大约需要 $2N$ FLOPs，forward + backward + gradient 计算大约是 forward 的 3 倍，因此约为 $6N$ FLOPs/token。

它依赖几个假设：

- 模型是 dense Transformer，每个 token 激活大部分参数；
- sequence length 不极端长，attention 的 $S^2$ 项没有压倒 FFN 和 projection 成本；
- 忽略 embedding、normalization、loss、optimizer update、通信和数据加载开销；
- 参数量口径和 FLOPs 口径在不同实验中一致。

当上下文很长时，自注意力的二次复杂度会变得不可忽略。更细地看，attention score 和 attention-value 计算包含与 $S^2$ 相关的项，因此长上下文训练不能只靠 $6ND$ 判断成本。对 MoE 模型也不能直接用总参数量 $N_{\mathrm{total}}$；更合理的是用每个 token 激活的参数量 $N_{\mathrm{active}}$ 估算训练计算，同时单独记录总参数带来的存储和通信成本。

## 参数量与 token 数的配比

定义 tokens-per-parameter ratio：

$$
\rho = \frac{D}{N}
$$

$\rho$ 是判断训练配比的简单信号。Chinchilla-style dense LM 常用经验锚点是 $\rho \approx 20$，即训练 token 数约为参数量的 20 倍。但这个数字不是定律，它取决于数据质量、模型架构、训练目标、重复 epoch 和评测目标。

不同 $\rho$ 对应不同风险：

| 配比状态 | 典型表现 | 风险 |
|---|---|---|
| $D$ 太少，$N$ 太大 | 模型容量大，但 validation loss 仍可被更多 token 明显改善 | undertrained，训练 compute 没有转化为足够泛化 |
| $D$ 与 $N$ 大致匹配 | 模型容量和数据量都在发挥作用 | 通常更接近 compute-optimal |
| $D$ 很多，$N$ 很小 | 小模型被大量训练，loss 下降逐渐受容量限制 | model-limited，继续加 token 收益递减 |

对于产品或研究任务，最优 $\rho$ 还会被推理成本改变。较小模型训练更多 token 可能在固定训练 compute 下有更低 loss，同时部署时每 token 成本更低；但如果任务需要更高容量、更强 in-context learning 或更复杂表示，过小模型也会形成能力上限。

## Compute、Wall-clock 与 MFU

FLOPs 是算法计算量，不等于真实训练时间。真实 wall-clock time 取决于硬件峰值、GPU 数量和利用率：

$$
\mathrm{time}
\approx
\frac{C_{\mathrm{train}}}{G \cdot P_{\mathrm{peak}} \cdot \eta}
$$

其中：

- $G$ 是 GPU 数；
- $P_{\mathrm{peak}}$ 是单 GPU 理论峰值 FLOPs/s；
- $\eta$ 是 model FLOPs utilization, 即 MFU；
- 通信、checkpoint、评测、数据加载和失败重跑会进一步增加总耗时。

例子：一个 $N=7B$ 的 dense LM 训练 $D=140B$ tokens，按 $6ND$ 估算：

$$
C \approx 6 \times 7 \times 10^9 \times 140 \times 10^9
= 5.88 \times 10^{21} \text{ FLOPs}
$$

这只是训练计算的粗估。真实项目还需要把显存、并行效率、数据管线、checkpoint I/O 和评测开销纳入 [[training/scaling/training-budget|Training Budget]]。

## Compute 与 Memory 是不同约束

模型参数量和 token 数主要决定训练计算量，但显存约束的组成不同。训练显存包括：

- parameters；
- gradients；
- optimizer states；
- activations；
- temporary buffers；
- distributed communication buffers；
- CUDA / framework overhead。

因此两个训练计划可能 FLOPs 接近，但显存压力完全不同。例如，增加 sequence length 会显著增加 activation memory；增加参数量会增加参数、梯度和 optimizer state；使用 AdamW 会比 SGD 保存更多 optimizer states；ZeRO/FSDP 会改变每卡模型状态存储。详细公式见 [[training/optimization/training-memory-estimation|Training Memory Estimation]]。

## 数据质量改变有效 $D$

$D$ 不是越大越好，因为 token 有质量差异。低质量、重复、模板化或污染数据虽然增加 nominal token count，却可能无法等价增加有效训练信息。

可以区分：

- **nominal tokens**：tokenizer 后统计的总 token 数；
- **effective tokens**：真正提供新信息、泛化价值和目标能力的 token；
- **contaminated tokens**：泄漏评测或带来虚假 benchmark 提升的 token；
- **harmful tokens**：引入噪声、偏见、安全风险或格式污染的 token。

这就是 [[training/pretraining/data-mix|Data Mix]] 和 [[training/data-engineering/quality-filtering|Quality Filtering]] 会直接影响 scaling 的原因。高质量数学、代码或学术文本可能在目标能力上比同量普通网页 token 更有价值，但也可能因为分布窄而牺牲通用性。

## 训练配比的诊断信号

判断训练计划是否合理，可以看以下信号：

- validation loss 是否按预期随 compute 平滑下降；
- 增加模型参数时，loss 改善是否大于增加 token 的改善；
- 增加 token 时，模型是否仍持续学习而没有明显进入容量瓶颈；
- 不同 domain validation loss 是否同步改善；
- 训练后期 learning rate decay 阶段是否仍有有效 loss 下降；
- downstream benchmark 是否与 language modeling loss 一致；
- 小规模 pilot run 的外推 residual 是否可解释。

如果总 loss 下降正常但 math/code domain loss 停滞，问题可能不是总 compute 不够，而是 data mix 或 tokenizer 不合适。如果训练 loss 下降但 validation loss 不降，可能是数据重复、过拟合或分布偏差。如果所有曲线都有尖峰，应先检查优化稳定性，而不是继续做 scaling 外推。

## 相关概念

- [[training/scaling/scaling-law|Scaling Law]]
- [[training/scaling/compute-optimal|Compute Optimal]]
- [[training/scaling/training-budget|Training Budget]]
- [[training/pretraining/compute-optimal|Pretraining Compute Optimal]]
- [[training/pretraining/data-mix|Data Mix]]
- [[training/optimization/training-memory-estimation|Training Memory Estimation]]
- [[training/distributed-training/fsdp|FSDP]]
- [[training/distributed-training/zero|ZeRO]]
