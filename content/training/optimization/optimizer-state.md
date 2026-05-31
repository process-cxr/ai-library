---
title: Optimizer State
created: 2026-03-21
published: 2026-03-21
modified: 2026-05-31
type: topic
status: mature
area: training
tags:
  - training-optimization
  - optimizer
  - memory
---

Optimizer State 指优化器为了更新参数而额外维护的 per-parameter 状态。它是训练显存中经常被低估的一部分：模型权重本身可能只占 2 bytes / parameter，但 AdamW 训练时还可能为每个参数保存 fp32 master weights、一阶矩 $m$ 和二阶矩 $v$，使优化器相关状态远大于参数权重本身。

这篇笔记关注 optimizer state 的内存结构；完整训练显存估算见 [[training/optimization/training-memory-estimation|Training Memory Estimation]]。

## AdamW 的状态

Adam / AdamW 对每个可训练参数维护两个移动平均：

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
$$

其中：

- $g_t$ 是当前梯度；
- $m_t$ 是 first moment estimate；
- $v_t$ 是 second moment estimate；
- $\beta_1,\beta_2$ 控制指数移动平均。

在大模型训练中，$m$ 和 $v$ 通常以 fp32 保存以提高数值稳定性。因此仅 Adam states 就需要：

$$
M_{\text{Adam states}} = N(4 + 4) = 8N \text{ bytes}
$$

如果还保留 fp32 master weights，则额外增加：

$$
M_{\text{master}} = 4N \text{ bytes}
$$

AdamW 与 Adam 的主要差异在于 weight decay 的处理方式。AdamW 使用 decoupled weight decay，使权重衰减不混入梯度矩估计。显存估算上二者类似：只要使用一阶矩、二阶矩和可选 master weights，per-parameter state 的数量级基本相同。

## 混合精度下的常见组成

bf16/fp16 full fine-tuning with AdamW 常见模型状态：

| 项 | dtype | bytes / parameter |
|---|---|---:|
| training weights | bf16 / fp16 | 2 |
| gradients | bf16 / fp16 | 2 |
| fp32 master weights | fp32 | 4 |
| Adam $m$ | fp32 | 4 |
| Adam $v$ | fp32 | 4 |
| 合计 | - | 16 |

所以 7B 参数模型的模型状态可粗略估算为：

$$
7B \times 16 \text{ bytes} \approx 112 \text{ GB}
$$

这还不含 activation 和临时 buffer。

## 不同 Optimizer 的状态差异

| Optimizer | 主要额外状态 | 粗略内存 |
|---|---|---|
| SGD | 无或 momentum | 0N / 4N |
| Adam / AdamW | $m$、$v$ | 8N |
| AdamW + master weights | master、$m$、$v$ | 12N |
| Adafactor | factored second moments | 通常低于 Adam |
| 8-bit optimizer | quantized states | 低于 fp32 Adam，但依赖实现 |

Optimizer 选择不只是收敛问题，也是内存预算问题。AdamW 稳定、常用，但状态显存高；低内存 optimizer 可能节省显存，但需要评估收敛质量和数值稳定性。

## 什么时候会有 Master Weights

Master weights 的存在取决于训练 recipe 和框架实现。早期 fp16 mixed precision 常保留 fp32 master copy，因为直接在 fp16 权重上做小幅更新容易丢失精度。bf16 训练由于 exponent range 更宽，有些实现可以不保留单独 master weights，但 optimizer state 仍常以 fp32 保存。

因此估算显存时不要只写“bf16 AdamW = 多少 GB”，而要确认：

- parameter copy 是否只有一份；
- optimizer 是否维护 fp32 master weights；
- gradients 是 bf16/fp16 还是 fp32；
- optimizer states 是否被压缩或 offload；
- FSDP / ZeRO 是否对这些状态进行 shard。

这也是不同框架、不同配置下同一模型训练显存差异很大的原因。

## ZeRO / FSDP 如何切分 Optimizer State

在普通 DDP 中，每张 GPU 都复制完整 optimizer state。若 data parallel world size 为 $D$，ZeRO / FSDP 可以按不同 stage 切分：

- [[training/distributed-training/zero|ZeRO-1]]：切 optimizer state；
- ZeRO-2：切 optimizer state 和 gradients；
- ZeRO-3 / [[training/distributed-training/fsdp|FSDP]] full shard：切 optimizer state、gradients 和 parameters。

只看 optimizer state：

$$
M_{\text{optimizer per GPU}}
\approx \frac{N b_o}{D}
$$

其中 $b_o$ 是每个参数的 optimizer state bytes。例如 AdamW + master weights 可取 $b_o=12$。

## Freeze / LoRA 场景

Optimizer state 只为可训练参数维护。冻结参数不需要梯度和 optimizer state。

这就是 [[training/post-training/sft|SFT]] 中 LoRA / QLoRA 显存低很多的关键原因：base model 很大，但可训练 LoRA 参数 $N_L$ 很小，因此 optimizer state 从 $12N$ 下降到 $12N_L$。

需要注意：activation 显存不会因为 optimizer state 变少而同比例消失，因为 forward/backward 仍经过模型主体。

## Checkpoint 中的 Optimizer State

Optimizer state 不只占 GPU 显存，也占 checkpoint 存储。完整训练 checkpoint 通常需要保存：

- 模型权重；
- optimizer 的 $m$、$v$ 和可能的 master weights；
- learning rate scheduler；
- random states；
- sharded state metadata。

如果只导出推理权重，checkpoint 可能接近 $2N$ bytes；如果保存完整 AdamW training state，则可能接近 $14N$ 到 $16N$ bytes 或更高，具体取决于 dtype 和分片格式。大规模训练中，optimizer state checkpoint 的写入、读取和合并常常是恢复时间的重要瓶颈。

## Offload 与低精度 Optimizer

Optimizer state 显存过高时，常见缓解方式有：

- ZeRO-1/2/3 或 FSDP sharding；
- CPU / NVMe offload；
- 8-bit optimizer 或 quantized optimizer state；
- Adafactor 等低状态优化器；
- LoRA / QLoRA 等减少可训练参数量的方法。

这些方法都不是无代价的。Offload 会引入 PCIe/NVMe 带宽瓶颈；低精度 optimizer 可能影响收敛或需要更谨慎的超参；减少可训练参数量可能限制可适配能力。选择时应同时看显存、吞吐、收敛质量和恢复复杂度。

## 常见误区

- **误区：参数权重就是训练显存主体。** 对 AdamW full fine-tuning，optimizer states 往往才是大头。
- **误区：bf16 训练所有状态都是 bf16。** 为稳定性，optimizer states 和 master weights 常常是 fp32。
- **误区：ZeRO 只优化通信。** ZeRO 最核心的动机之一就是减少 optimizer state、gradient 和 parameter 的冗余存储。
- **误区：LoRA 显存只由 LoRA 参数决定。** LoRA 降低 optimizer state，但 activation、base weights 和 temporary buffers 仍然存在。

## 相关概念

- [[training/optimization/training-memory-estimation|Training Memory Estimation]]
- [[training/optimization/mixed-precision|Mixed Precision Training]]
- [[training/distributed-training/zero|ZeRO]]
- [[training/distributed-training/fsdp|FSDP]]
- [[fundamentals/optimization/adam|Adam]]
- [[fundamentals/optimization/adamw-weight-decay|AdamW Weight Decay]]
