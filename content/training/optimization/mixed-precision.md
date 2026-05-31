---
title: Mixed Precision Training
created: 2026-03-15
published: 2026-03-15
modified: 2026-05-31
type: topic
status: mature
area: training
tags:
  - training
  - optimization
  - memory-optimization
---

Mixed Precision Training 指在训练中混合使用不同数值精度，以降低显存、提高吞吐，同时尽量保持收敛稳定性。大模型训练中常见做法是：模型权重、激活和梯度使用 fp16 或 bf16，部分累加、归一化、optimizer state 或 master weights 保持 fp32。

混合精度的核心不是“把所有东西都变小”，而是在不同张量上选择合适 dtype：低精度负责节省带宽和显存，高精度负责保护数值稳定性。

## 常见精度

| dtype | bytes | 特点 | 常见用途 |
|---|---:|---|---|
| fp32 | 4 | 动态范围和精度较高 | optimizer state、master weights、稳定计算 |
| fp16 | 2 | 精度低、动态范围窄，易 overflow/underflow | 早期 mixed precision、推理和部分训练 |
| bf16 | 2 | mantissa 少于 fp32，但 exponent 与 fp32 接近 | 现代 LLM 训练主力 |
| fp8 | 1 | 显存/带宽更低，但 recipe 和硬件要求高 | 部分大规模训练和推理优化 |

bf16 常比 fp16 更适合大模型训练，因为它保留了 fp32 的 exponent range，数值溢出风险更低，通常不需要像 fp16 那样依赖 loss scaling。

选择 dtype 时应区分三个问题：

- **存储精度**：tensor 在显存中长期以什么格式保存；
- **计算精度**：矩阵乘、attention、normalization 等 op 用什么格式计算；
- **累加精度**：reduction、dot product、optimizer update 的中间累加用什么格式。

很多硬件上的低精度矩阵乘会用 fp16 / bf16 输入，但在更高精度 accumulator 中累加，再输出低精度结果。因此“bf16 训练”并不意味着每一步计算都只有 bf16 精度。

## Loss Scaling

fp16 的数值范围较窄，小梯度可能 underflow 成 0。Loss scaling 的做法是先把 loss 放大：

$$
\tilde{\mathcal{L}} = s \mathcal{L}
$$

反向传播得到放大的梯度后，再在 optimizer update 前除以 $s$：

$$
g = \frac{\tilde{g}}{s}
$$

如果检测到 overflow，则降低 scale；如果一段时间稳定，则可以增大 scale。这就是 dynamic loss scaling。bf16 通常不需要 loss scaling，但仍可能需要关注归一化、softmax、attention score 等局部数值稳定性。

## Master Weights 与 Optimizer State

混合精度训练常保留 fp32 master weights：

- forward/backward 使用 fp16/bf16 weights；
- optimizer 在 fp32 master weights 上更新；
- 更新后再同步或 cast 到低精度训练权重。

这会增加显存。以 bf16 + AdamW + fp32 master weights 为例：

| 项 | bytes / parameter |
|---|---:|
| bf16 weight | 2 |
| bf16 gradient | 2 |
| fp32 master weight | 4 |
| Adam $m$ | 4 |
| Adam $v$ | 4 |
| 合计 | 16 |

因此混合精度节省了参数、梯度和 activation 的部分显存，但 optimizer state 可能仍是 fp32 大头。详见 [[training/optimization/optimizer-state|Optimizer State]]。

## AMP

AMP，Automatic Mixed Precision，是框架自动选择部分 op 使用低精度、部分 op 保持高精度的机制。它通常包括：

- autocast：根据 op 类型自动选择 dtype；
- grad scaler：fp16 下动态调整 loss scale；
- fp32 fallback：对数值敏感 op 保持高精度。

AMP 降低了手写 mixed precision 的复杂度，但不代表可以忽略数值问题。训练中仍需监控 loss spike、NaN/Inf、gradient norm 和收敛曲线。

常见需要保留或回退到高精度的部分包括：

- loss reduction 与 gradient norm 统计；
- softmax 前后的稳定化计算；
- LayerNorm / RMSNorm 的部分统计；
- optimizer update；
- logits 与 cross entropy 的某些实现路径；
- 大规模 all-reduce 后的梯度检查。

具体哪些 op 使用 fp32 取决于框架、硬件和 kernel。实践中应优先使用成熟 AMP / distributed training recipe，而不是手动对所有张量强制 cast。

## 与 FSDP / ZeRO 的配合

[[training/distributed-training/fsdp|FSDP]] 和 [[training/distributed-training/zero|ZeRO]] 会切分 parameters、gradients 和 optimizer states。Mixed precision 决定这些状态的 dtype，FSDP/ZeRO 决定这些状态是否在 data parallel 维度上复制。

二者解决的是不同问题：

- mixed precision：减少每个 tensor element 的 bytes；
- ZeRO/FSDP：减少每张 GPU 持有的 tensor elements 数量。

例如 bf16 full fine-tuning 可以把 parameter/gradient 从 fp32 的 4 bytes 降到 2 bytes；ZeRO-3 可以进一步把 parameter/gradient/optimizer state 分片到多个 GPU。

分布式训练中还要区分 parameter dtype、reduce dtype 和 buffer dtype。有些配置会用 bf16 参数和 bf16 梯度通信，有些会在 fp32 中做 gradient reduction 以保护稳定性。通信 dtype 越低，带宽压力越小；但如果梯度数值范围或累加误差变大，可能引入训练不稳定。

## FP8 的边界

FP8 进一步降低显存和带宽，但训练 recipe 更复杂，通常需要：

- 支持 FP8 tensor core 的硬件；
- per-tensor 或 per-channel scaling；
- amax history 或动态缩放；
- 对 attention、MLP、normalization、optimizer 等模块分别设定精度策略；
- 更严格的 loss spike 和 NaN 监控。

因此 FP8 更适合成熟训练栈中的性能优化，而不是初次训练或小规模实验的默认选择。对知识库中的通用估算，bf16/fp16 仍是更稳定的基准。

## 失败模式与边界

- **Overflow / Underflow**：fp16 下尤其常见，表现为 loss NaN、gradient Inf 或训练停滞。
- **Loss Spike**：低精度计算可能放大不稳定训练配置的问题。
- **不稳定 op**：softmax、normalization、large reduction 等可能需要 fp32 或特殊 kernel。
- **dtype mismatch**：手写模块或第三方 kernel dtype 不一致，可能造成隐式 cast 和性能损失。
- **误判显存节省**：即便参数是 bf16，optimizer states 仍可能是 fp32。

## 诊断信号

混合精度问题通常可以通过以下信号定位：

- loss 突然变为 NaN / Inf；
- gradient norm 突然爆炸或长期为 0；
- dynamic loss scale 频繁下降；
- bf16 与 fp32 小规模对照收敛曲线明显不同；
- 某个 kernel 或自定义 op 触发大量隐式 cast；
- 只在多卡通信后出现数值异常。

排查时应先用小 batch、短序列和较保守 dtype 复现，再逐步开启 fused kernel、FlashAttention、ZeRO/FSDP、FP8 或其他高性能配置。

## 相关概念

- [[training/optimization/training-memory-estimation|Training Memory Estimation]]
- [[training/optimization/optimizer-state|Optimizer State]]
- [[training/optimization/loss-spike|Loss Spike]]
- [[training/distributed-training/fsdp|FSDP]]
- [[training/distributed-training/zero|ZeRO]]
- [[fundamentals/linear-algebra/numerical-stability|Numerical Stability]]
