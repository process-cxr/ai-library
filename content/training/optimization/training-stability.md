---
title: Training Stability
created: 2026-03-21
published: 2026-03-21
modified: 2026-05-31
type: topic
status: mature
area: training
tags:
  - training-optimization
  - stability
---

Training Stability 指大模型训练在长时间、多节点、混合精度和大规模数据条件下保持可收敛、可恢复、可解释的能力。稳定训练不仅要求 loss 最终下降，还要求训练过程中没有不可控的 loss spike、NaN/Inf、梯度爆炸、数据异常、通信错误或 checkpoint 损坏。

稳定性是训练工程的核心，因为大模型训练成本高，单次不稳定可能浪费大量 GPU 时间，并让 scaling law 外推失去可信度。

## 稳定性的主要维度

| 维度 | 典型问题 | 观测信号 |
|---|---|---|
| Optimization | 学习率过大、warmup 不足、batch size 不合适 | loss spike、grad norm 爆炸 |
| Numerical | fp16 overflow、softmax 溢出、NaN/Inf | loss NaN、grad Inf、loss scale 下降 |
| Data | 异常 batch、乱码、重复、极端长度、污染数据 | 单 batch loss 异常、domain loss 异常 |
| System | 通信错误、checkpoint 写坏、worker 数据不一致 | rank divergence、hang、恢复失败 |
| Distributed | all-reduce 不一致、shard 错误、随机种子不一致 | 不同 rank loss/grad 不一致 |
| Monitoring | 指标不足，异常发现太晚 | spike 后无法定位原因 |

稳定性问题通常不是单因子。例如，数据异常可能触发大梯度，大梯度在 fp16 下 overflow，再被过高 learning rate 放大。

## Loss、Gradient 与 Update

训练更新可写为：

$$
\theta_{t+1} = \theta_t - \eta_t \cdot u_t
$$

其中 $\eta_t$ 是 learning rate，$u_t$ 是由 optimizer 根据梯度和状态产生的更新方向。稳定性取决于：

- loss surface 是否被 batch 正常采样；
- gradient norm 是否在可控范围；
- optimizer state 是否健康；
- learning rate 是否匹配 batch size 和训练阶段；
- dtype 是否能表达中间值；
- 更新后参数是否进入异常区域。

[[fundamentals/optimization/gradient-clipping|Gradient Clipping]] 可以限制梯度范数，但不能解决所有根因。它是安全阀，不是诊断替代品。

## Learning Rate 与 Warmup

学习率过大是 loss spike 和发散的常见原因。Warmup 的作用是在训练初期逐步增大学习率，让 optimizer state、activation scale 和模型表示逐步进入稳定区域。

常见策略：

- linear warmup；
- cosine decay；
- warmup + stable plateau + decay；
- 针对 CPT / SFT 使用更低 learning rate；
- 数据分布切换时重新 warmup 或降低 learning rate。

如果 warmup 太短，训练初期容易 spike；如果 decay 太快，模型可能欠训练；如果 CPT 使用接近预训练早期的高 learning rate，可能破坏已有表示。

## Mixed Precision 稳定性

[[training/optimization/mixed-precision|Mixed Precision Training]] 会改变数值风险：

- fp16 动态范围窄，容易 overflow/underflow；
- bf16 动态范围接近 fp32，通常更稳定；
- softmax、normalization、large reduction 等 op 仍可能需要 fp32 或稳定 kernel；
- loss scaling 可缓解 fp16 underflow，但不能修复所有数值问题。

需要监控：

- NaN / Inf count；
- dynamic loss scale；
- gradient norm；
- activation statistics；
- attention score range；
- optimizer state 是否出现 NaN。

## 数据异常

数据异常是训练稳定性的重要来源：

- 超长或异常短样本；
- 乱码、重复 token、异常 Unicode；
- 错误 packing / attention mask；
- loss mask 错误；
- 极端高 loss 文档；
- benchmark 或答案泄漏；
- 某个 shard 格式错误；
- 某一语言/领域突然过采样。

数据异常常表现为 isolated spike。稳定训练需要记录 batch metadata，使异常发生后能定位到具体 shard、document ID、language、domain 和 packing 规则。

## 监控指标

最低限度应监控：

- training loss；
- validation loss；
- per-domain validation loss；
- learning rate；
- gradient norm；
- update norm / parameter norm；
- NaN / Inf；
- loss scale；
- tokens/sec；
- GPU memory；
- data shard ID；
- batch sequence length distribution；
- checkpoint save/load status。

更精细的监控包括：

- attention entropy；
- activation norm；
- optimizer state norm；
- per-rank loss；
- skipped steps；
- gradient clipping ratio；
- data source sampling ratio；
- MFU 和通信时间。

## 恢复策略

遇到不稳定时，常见恢复顺序：

1. 确认是否出现 NaN/Inf，若有则回滚到上一个健康 checkpoint。
2. 检查最近数据 shard 和 batch metadata。
3. 降低 learning rate 或增加 warmup。
4. 启用或收紧 gradient clipping。
5. 检查 mixed precision op 和 loss scaling。
6. 排查 packing、mask 和 tokenizer 变化。
7. 从健康 checkpoint 重启，并跳过或隔离异常数据。
8. 若反复出现，做小规模复现实验定位。

不要在已经污染 optimizer state 的 checkpoint 上盲目继续训练。NaN 进入 optimizer state 后，即使后续 loss 看似恢复，也可能留下不可解释问题。

## 常见失败模式

- **只看 loss 不看梯度**：loss spike 出现时已经太晚，grad norm 更早暴露风险。
- **没有 batch provenance**：无法定位异常样本。
- **checkpoint 太稀疏**：恢复点过远，浪费大量 compute。
- **忽略 rank divergence**：分布式训练某些 rank 已异常但平均指标掩盖问题。
- **用 clipping 掩盖根因**：梯度被裁掉，但数据或学习率问题仍存在。
- **CPT 切换数据不降学习率**：已有表示被破坏。

## 相关概念

- [[training/optimization/loss-spike|Loss Spike]]
- [[training/optimization/mixed-precision|Mixed Precision Training]]
- [[training/optimization/gradient-checkpointing|Gradient Checkpointing]]
- [[training/optimization/optimizer-state|Optimizer State]]
- [[fundamentals/optimization/gradient-clipping|Gradient Clipping]]
- [[training/data-engineering/data-engineering|Data Engineering]]
- [[training/mid-training/continued-pretraining|Continued Pretraining]]
