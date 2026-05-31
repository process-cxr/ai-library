---
title: Loss Spike
created: 2026-03-21
published: 2026-03-21
modified: 2026-05-31
type: topic
status: mature
area: training
tags:
  - training-optimization
  - loss-spike
---

Loss Spike 指训练 loss 在一个或多个 step 内突然显著升高，随后可能恢复，也可能演化为发散、NaN 或模型能力退化。它是 [[training/optimization/training-stability|Training Stability]] 中最常见的异常信号之一。

Loss spike 不一定意味着训练失败。关键要判断 spike 是孤立 batch 噪声、可恢复不稳定，还是系统性发散的前兆。

## 类型

| 类型 | 表现 | 常见原因 |
|---|---|---|
| Isolated spike | 单个或少数 step loss 升高后恢复 | 异常 batch、长样本、噪声数据 |
| Repeated spikes | 周期性或频繁 spike | learning rate 偏高、数据 shard 问题 |
| Persistent elevation | loss 升高后不回落 | 模型状态被破坏、数据分布切换 |
| NaN/Inf spike | loss 或梯度变 NaN/Inf | 数值溢出、fp16 问题、unstable op |
| Rank-local spike | 部分 rank 异常 | 数据分片、通信或随机性问题 |

不同类型对应不同处理方式。孤立 spike 可能只需记录和继续观察；NaN/Inf 或 persistent spike 通常需要回滚 checkpoint。

## 常见原因

### Learning Rate

学习率过大时，单步更新可能越过稳定区域，表现为 loss sudden increase。Warmup 不足、数据切换时不降 LR、CPT/SFT 沿用过高 LR 都可能触发 spike。

诊断信号：

- spike 前 gradient norm 已升高；
- update norm / parameter norm 异常；
- 降低 LR 后 spike 消失；
- spike 常出现在 warmup 早期或数据阶段切换后。

### Gradient Explosion

梯度爆炸会导致更新过大。原因可能是极端 batch、长序列、初始化或 optimizer state 异常。

可用 [[fundamentals/optimization/gradient-clipping|Gradient Clipping]] 限制梯度范数：

$$
g \leftarrow c \frac{g}{\|g\|}
\quad \text{if } \|g\| > c
$$

但如果大量 step 都被 clipping，说明根因仍在。

### Mixed Precision

fp16 overflow、loss scale 失效、不稳定 softmax 或 dtype mismatch 都可能造成 spike。bf16 更稳定，但仍可能在 attention score、normalization、large reduction 中出现数值问题。

诊断信号：

- NaN/Inf count 上升；
- dynamic loss scale 频繁下降；
- 某些 kernel 或 op 后出现异常；
- 切换 bf16/fp32 fallback 后改善。

### 数据异常

数据异常常造成 isolated spike：

- 文档乱码；
- 极端长度；
- tokenizer 异常；
- packing mask 错误；
- loss mask 错误；
- 某个 shard 含大量低质或非目标语言数据；
- code/math 样本格式破坏。

如果 spike 可以定位到特定 shard 或 batch，应优先检查数据，而不是只调 optimizer。

### 分布式训练问题

分布式训练中，某些 rank 的数据、随机数、梯度或通信异常可能被平均指标掩盖。

需要检查：

- per-rank loss；
- per-rank grad norm；
- all-reduce / reduce-scatter 是否正常；
- FSDP/ZeRO shard 是否一致；
- checkpoint resume 后 optimizer state 是否完整。

## 检测

可以设置 spike detector：

$$
L_t > \mu_{t-k:t-1} + \alpha \sigma_{t-k:t-1}
$$

其中 $\mu$ 和 $\sigma$ 是最近窗口的 loss 均值和标准差，$\alpha$ 是阈值。也可以使用相对阈值：

$$
\frac{L_t}{\mathrm{EMA}(L)} > r
$$

但自动检测只能提示异常，不能替代诊断。不同阶段的正常 loss 波动不同，数据切换和学习率变化也会改变阈值。

## 处理策略

按严重程度处理：

1. **单次小 spike**：记录 batch metadata，继续观察。
2. **重复 spike**：降低 LR、检查数据 shard、查看 grad norm 和 clipping ratio。
3. **persistent spike**：回滚 checkpoint，缩小 LR，隔离近期数据。
4. **NaN/Inf**：停止训练，回滚到健康 checkpoint，检查 mixed precision 和 optimizer state。
5. **rank-local spike**：检查分布式数据加载、通信和 checkpoint resume。

如果 spike 后 validation loss 或 benchmark 明显退化，即使 training loss 后续恢复，也应谨慎使用该 checkpoint。

## 预防

- 使用足够 warmup；
- 监控 grad norm、update norm 和 NaN/Inf；
- 使用 bf16 或稳定 mixed precision recipe；
- 配置合理 gradient clipping；
- 记录 batch provenance；
- 对数据做 cleaning、dedup 和 quality filtering；
- 在数据阶段切换时降低 learning rate；
- 保持频繁 checkpoint；
- 在小规模 pilot run 中测试新数据和新 recipe。

## 与 Loss Curve 的关系

健康训练曲线并不要求每一步 loss 单调下降。mini-batch loss 本来有噪声，尤其在混合数据和多领域训练中。判断 spike 是否有害，需要看：

- spike 幅度；
- 持续时间；
- 是否伴随 grad norm/NaN；
- validation loss 是否受影响；
- 是否集中在某些数据源；
- checkpoint 后续能力是否下降。

## 相关概念

- [[training/optimization/training-stability|Training Stability]]
- [[training/optimization/mixed-precision|Mixed Precision Training]]
- [[training/optimization/optimizer-state|Optimizer State]]
- [[fundamentals/optimization/gradient-clipping|Gradient Clipping]]
- [[training/data-engineering/data-cleaning|Data Cleaning]]
- [[training/data-engineering/packing|Packing]]
