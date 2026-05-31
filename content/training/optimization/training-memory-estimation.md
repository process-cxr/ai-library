---
title: Training Memory Estimation
created: 2026-05-31
published: 2026-05-31
modified: 2026-05-31
type: topic
status: mature
area: training
tags:
  - training
  - training-optimization
  - memory
  - estimation
---

Training Memory Estimation 指在训练开始前估算模型会占用多少显存，并判断给定 GPU 数量、精度、batch size、sequence length 和并行策略是否能支撑训练。它不是一个单一公式，而是一组分项预算：parameters、gradients、optimizer states、activations、communication buffers、temporary workspace 和 framework overhead 共同决定训练显存峰值。

这类估算的价值在于：训练 OOM 通常不是“模型参数太大”这么简单。一个 7B 模型的 bf16 参数本身约 14GB，但 full fine-tuning with AdamW 还需要梯度、fp32 master weights、Adam 一阶/二阶矩、activation 和通信缓冲。真正决定能否训练的是这些状态在单卡上是否复制、切分、重算或 offload。

## 显存组成

训练显存可以分为两类：**模型状态显存**和**运行时显存**。

| 类别 | 组成 | 是否与 batch / sequence 相关 | 主要控制手段 |
|---|---|---|---|
| Parameters | 模型权重 | 否，主要与参数量和 dtype 相关 | mixed precision、ZeRO-3/FSDP、tensor parallel |
| Gradients | 反向传播梯度 | 否，主要与可训练参数量和 dtype 相关 | ZeRO-2/3、gradient accumulation、freeze |
| Optimizer states | Adam m/v、master weights 等 | 否，主要与可训练参数量和 optimizer 相关 | ZeRO-1/2/3、optimizer 选择、offload |
| Activations | 前向中间激活，用于 backward | 是，随 batch、sequence、hidden、layers 增长 | activation checkpointing、sequence parallel、micro-batch |
| Temporary buffers | attention workspace、通信缓冲、loss buffer | 部分相关 | kernel、并行策略、framework 配置 |
| Fragmentation / overhead | allocator 碎片、padding、框架缓存 | 部分相关 | memory profiling、固定 shape、allocator 配置 |

粗略估算时，先算模型状态，再估算 activation 和 buffer。精确峰值必须通过 profiler 验证。

一个实用原则是：**模型状态给出下界，activation 和 buffer 决定是否真的能跑起来。** 如果下界已经超过单卡显存，必须改变训练方式或并行策略；如果下界看似足够，仍需要为 activation、communication peak 和 allocator overhead 留出余量。

## Bytes per Parameter

设模型参数量为 $N$，某个 tensor dtype 的字节数为 $b$。常见 dtype：

| dtype | bytes / element | 典型用途 |
|---|---:|---|
| fp32 | 4 | optimizer state、master weights、部分稳定计算 |
| bf16 | 2 | 大模型训练主力参数/梯度 dtype |
| fp16 | 2 | 混合精度训练，常配合 loss scaling |
| fp8 | 1 | 部分训练/推理加速场景，依赖硬件和 recipe |
| int8 | 1 | 主要用于推理或量化训练中特定状态 |
| int4 | 0.5 | QLoRA / 推理量化中的权重存储 |

参数显存近似为：

$$
M_{\text{param}} = N \cdot b_{\text{param}}
$$

如果 $N=7B$，bf16 参数为：

$$
7 \times 10^9 \times 2 \text{ bytes} \approx 14 \text{ GB}
$$

这里的 GB 是十进制近似。实际显存显示常用 GiB，且还会受 padding、alignment、flattening 和 framework buffer 影响。

## Full Fine-tuning 的模型状态

对 full fine-tuning with AdamW，一个常见混合精度配置是：

- bf16/fp16 model weights：$2N$ bytes；
- bf16/fp16 gradients：$2N$ bytes；
- fp32 master weights：$4N$ bytes；
- Adam first moment $m$：$4N$ bytes；
- Adam second moment $v$：$4N$ bytes。

因此模型状态近似为：

$$
M_{\text{states}}
= N(2 + 2 + 4 + 4 + 4)
= 16N \text{ bytes}
$$

也就是说，**bf16 full fine-tuning with AdamW 的模型状态大约是 16 bytes / parameter**。如果某些框架不保留单独 fp32 master weights，或者 optimizer state dtype 被压缩，这个常数会变化。

常见估算表：

| 训练配置 | 参数 | 梯度 | Optimizer states | 近似合计 |
|---|---:|---:|---:|---:|
| fp32 + AdamW | 4N | 4N | 8N | 16N |
| bf16/fp16 + AdamW + master weights | 2N | 2N | 12N | 16N |
| bf16/fp16 + AdamW no master copy | 2N | 2N | 8N | 12N |
| bf16/fp16 + SGD momentum | 2N | 2N | 4N | 8N |
| bf16/fp16 frozen model | 2N | 0 | 0 | 2N |

这解释了为什么训练显存远高于推理显存：推理主要存参数和 KV Cache，而训练还必须存梯度、优化器状态和 activations。

## 单卡 DDP 估算

在普通 data parallel / DDP 中，每张 GPU 都持有完整模型副本、完整梯度和完整 optimizer states。因此不含 activation 的单卡模型状态约为：

$$
M_{\text{DDP}}
\approx N(b_p + b_g + b_o)
$$

其中：

- $b_p$ 是每个参数的训练权重字节数；
- $b_g$ 是每个参数的梯度字节数；
- $b_o$ 是每个参数的 optimizer state 字节数。

以 bf16 + AdamW + fp32 master weights 为例，$b_p=2$，$b_g=2$，$b_o=12$：

$$
M_{\text{DDP}} \approx 16N
$$

| 参数量 | 模型状态估算 | 不含 activation 的直观结论 |
|---:|---:|---|
| 7B | $7B \times 16 \approx 112$ GB | 单卡 80GB 无法 full fine-tune |
| 13B | $13B \times 16 \approx 208$ GB | 必须切分或参数高效微调 |
| 70B | $70B \times 16 \approx 1120$ GB | 必须多维并行和切分 |

这些数字还没算 activation，因此只是下界。

## ZeRO / FSDP 切分公式

[[training/distributed-training/zero|ZeRO]] 和 [[training/distributed-training/fsdp|FSDP]] 的核心是减少 data parallel 维度上的状态冗余。设 data parallel world size 为 $D$。

### ZeRO-1

ZeRO-1 切分 optimizer states，但每卡仍复制完整参数和梯度：

$$
M_{\text{ZeRO-1}}
\approx N(b_p + b_g + \frac{b_o}{D})
$$

### ZeRO-2

ZeRO-2 切分 optimizer states 和 gradients，但每卡仍复制完整参数：

$$
M_{\text{ZeRO-2}}
\approx N(b_p + \frac{b_g}{D} + \frac{b_o}{D})
$$

### ZeRO-3 / FSDP full shard

ZeRO-3 / FSDP full shard 切分 parameters、gradients 和 optimizer states：

$$
M_{\text{ZeRO-3}}
\approx N(\frac{b_p + b_g + b_o}{D})
$$

这个公式是长期驻留状态的近似下界。实际训练中，FSDP / ZeRO-3 会在 layer forward/backward 时 all-gather 当前层完整参数，因此会出现 transient peak：

$$
M_{\text{peak}}
\approx M_{\text{sharded states}}
+ M_{\text{current full layer}}
+ M_{\text{activations}}
+ M_{\text{buffers}}
$$

因此，ZeRO-3 并不意味着单卡显存严格等于总状态除以 GPU 数。wrap granularity、prefetch、reshard 策略和通信 buffer 都会影响峰值。

## Activation Memory

Activation 显存通常是训练中最难估的部分，因为它与模型结构、batch size、sequence length、checkpointing 策略和 attention kernel 有关。对 Transformer block 中需要为 backward 保留的 hidden states，一个常用粗略形式是：

$$
M_{\text{act}}
\propto
B_{\mu} \cdot S \cdot L \cdot H \cdot c_{\text{act}}
$$

其中：

- $B_{\mu}$ 是 per-GPU micro-batch size；
- $S$ 是 sequence length；
- $L$ 是 Transformer layers；
- $H$ 是 hidden size；
- $c_{\text{act}}$ 是由 MLP、normalization、dropout、checkpointing、dtype 和具体实现决定的常数。

但这个线性形式不是完整上界。naive attention 如果显式 materialize attention scores / probabilities，还可能产生与序列长度平方相关的中间量：

$$
M_{\text{attn tmp}}
\propto B_{\mu} \cdot n_{\text{heads}} \cdot S^2
$$

FlashAttention、memory-efficient attention 和 recomputation 策略会显著改变这部分临时显存。因此长上下文训练的 activation 估算必须区分“需要保存的 hidden activations”和“attention kernel 临时 workspace”。

更直观地说：

- sequence length 翻倍，hidden activation 通常近似翻倍；若使用 naive attention，attention 临时量可能按 $S^2$ 增长；
- micro-batch size 翻倍，activation 近似翻倍；
- layer 数越多，需要为 backward 保留的中间结果越多；
- [[training/optimization/gradient-checkpointing|Gradient Checkpointing]] 可以少存 activation，但要在 backward 重算 forward；
- FlashAttention 类 kernel 可以减少 attention score/probability 矩阵的 materialization，但不等于消除所有 activation。

Activation memory 与 optimizer state 不同：ZeRO/FSDP 主要切分模型状态，不自动把 activation 除以 GPU 数。要降低 activation，通常要调整 micro-batch、sequence length、checkpointing、sequence parallel 或模型结构。

## Gradient Accumulation 与 Micro-batch

Global batch size 可写为：

$$
B_{\text{global}}
= B_{\mu} \cdot D \cdot G
$$

其中：

- $B_{\mu}$ 是每卡 micro-batch size；
- $D$ 是 data parallel world size；
- $G$ 是 gradient accumulation steps。

显存主要受 $B_{\mu}$ 影响，而不是直接受 $B_{\text{global}}$ 影响。增大 gradient accumulation 可以在不增加 activation 显存的情况下增大 effective batch size，但会增加每个 optimizer step 的时间。

## LoRA / QLoRA 估算

参数高效微调改变的是“可训练参数量”。设 base model 参数量为 $N$，LoRA 可训练参数量为 $N_L$，且 $N_L \ll N$。

LoRA 通常需要：

- base weights：冻结存储，通常 bf16/fp16；
- LoRA weights：可训练；
- LoRA gradients；
- LoRA optimizer states。

近似训练显存可写成：

$$
M_{\text{LoRA train}}
\approx N \cdot b_{\text{base}}
+ N_L(b_p + b_g + b_o)
+ M_{\text{act}}
$$

因为 optimizer states 只为 LoRA 参数维护，所以相比 full fine-tuning 大幅降低模型状态显存。但 activation 仍然与 forward/backward 经过的模型结构有关，不会因为 LoRA 参数少就消失。

QLoRA 进一步把 base weights 以 4-bit 存储，并在计算时反量化到较高精度：

$$
M_{\text{QLoRA base}}
\approx N \cdot 0.5 \text{ bytes}
+ \text{quantization metadata}
$$

实际 QLoRA 显存还包括 quantization scales、zero points、paged optimizer、LoRA states 和 activation。估算时不能只用 $0.5N$ 得出最终显存。

QLoRA 通常不会在显存中长期保留一份完整的高精度 base weights；更常见的是按层或按 kernel 在计算时反量化并产生临时 buffer。因此估算时应区分长期驻留的 4-bit weights 与运行时 dequantization workspace。

## Checkpoint 与磁盘存储

训练预算还要估算 checkpoint storage。一个完整 checkpoint 可能包含：

- model weights；
- optimizer states；
- lr scheduler state；
- random states；
- dataloader / trainer metadata；
- sharded checkpoint metadata。

如果保存 full training state，磁盘占用可能接近或超过训练时模型状态：

$$
M_{\text{checkpoint}}
\approx N(b_p + b_o) + \text{metadata}
$$

只保存 inference weights 则通常小得多。例如 bf16 7B 权重约 14GB，但 full optimizer checkpoint 可能超过 100GB。

Sharded training 下，checkpoint 还要考虑格式问题。ZeRO/FSDP 的 sharded checkpoint 适合原训练拓扑快速恢复，但跨框架迁移、合并成单文件权重、或改变 GPU 数量 resume 时可能需要额外转换。训练前应明确保存的是“可恢复训练状态”还是“可发布推理权重”，两者的容量、加载路径和可靠性要求不同。

## 快速估算流程

1. 确定训练方式：full fine-tuning、LoRA、QLoRA、continued pretraining 或 RLHF。
2. 确定参数量 $N$ 和可训练参数量 $N_{\text{trainable}}$。
3. 确定 dtype：parameter、gradient、master weights、optimizer states 分别多少 bytes。
4. 计算 DDP 下模型状态下界。
5. 根据 ZeRO/FSDP/TP/PP 策略切分模型状态。
6. 根据 micro-batch、sequence length、layers、hidden size 估算 activation。
7. 加上 communication buffers、temporary workspace 和 10%-30% overhead。
8. 用最小 batch 实测 profiler，再反推可用 batch / sequence。

## 示例：为什么 7B Full Fine-tuning 需要切分

假设训练 7B dense model，使用 bf16 parameters / gradients、AdamW fp32 states，并保留 fp32 master weights。模型状态近似：

$$
M_{\text{states}} = 7B \times 16 \text{ bytes} \approx 112\text{ GB}
$$

单张 80GB GPU 即使完全不考虑 activation，也放不下完整 DDP 模型状态。因此可行路线通常是：

- 使用 ZeRO/FSDP shard 模型状态；
- 或改用 LoRA / QLoRA，只训练少量 adapter 参数；
- 或减少 optimizer state，例如低精度 optimizer / offload；
- 同时用 checkpointing、减小 micro-batch 或缩短 sequence 控制 activation。

如果使用 8 张 GPU 的 ZeRO-3 / FSDP full shard，长期驻留模型状态下界约为：

$$
\frac{112\text{ GB}}{8}\approx 14\text{ GB / GPU}
$$

但这不是最终峰值。每卡还会有当前层 all-gather 参数、activation、temporary workspace、communication buffer 和 allocator overhead。因此实际配置仍需要 profiler 验证。

## 常见误区

### 只按参数权重估算训练显存

“7B bf16 只有 14GB”只适用于权重本身。full fine-tuning with AdamW 的模型状态可能是 112GB 量级，还不含 activation。

### 认为 ZeRO/FSDP 会切分所有显存

ZeRO/FSDP 主要切分 parameters、gradients 和 optimizer states。Activation、临时 buffer、通信峰值和 allocator overhead 仍然可能 OOM。

### 混淆 global batch 与 micro-batch

显存峰值更直接受 per-GPU micro-batch 影响。Gradient accumulation 可以增大 effective batch，但不能减少单个 micro-batch 的 activation。

### 忽略 sequence length

长上下文训练的显存压力常来自 activation 和 attention 相关中间量，而不只是模型参数。sequence length 从 4k 到 32k 时，原来的 batch size 通常不能直接沿用。

### 忽略 optimizer 和 checkpoint

显存能跑起来不代表磁盘能保存 checkpoint，也不代表 resume 成本可接受。大规模训练需要同时预算 GPU memory、CPU memory、NVMe 和网络带宽。

## 相关概念

- [[training/optimization/optimizer-state|Optimizer State]]
- [[training/optimization/mixed-precision|Mixed Precision Training]]
- [[training/optimization/gradient-checkpointing|Gradient Checkpointing]]
- [[training/distributed-training/zero|ZeRO]]
- [[training/distributed-training/fsdp|FSDP]]
- [[training/distributed-training/tensor-parallel|Tensor Parallel]]
- [[training/distributed-training/pipeline-parallel|Pipeline Parallel]]
- [[training/scaling/training-budget|Training Budget]]
- [[training/post-training/sft|SFT]]
