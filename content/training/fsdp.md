---
title: FSDP 训练知识点总结
tags:
  - distributed-training
  - pytorch
  - fsdp
---

> 帮助回顾和梳理 FSDP (Fully Sharded Data Parallel) 训练的核心概念和实战要点

## 1. FSDP 概述

FSDP (Fully Sharded Data Parallel) 是 PyTorch 提供的分布式训练策略，通过将模型参数、梯度和优化器状态**切分**到多个 GPU 上，大幅降低单个 GPU 的显存占用，使得在大模型训练中能够使用更少的 GPU 训练更大的模型。

**核心思想**：
- 数据并行：每个 GPU 处理不同的数据 batch
- 参数切分：模型参数分散存储在多个 GPU 上
- 按需聚合：计算时临时聚合需要的参数，用完立即释放

**适用场景**：

FSDP 本身只是显存切分策略，预训练、中训练、后训练都能用，但实际侧重不同：

| 训练阶段 | FSDP 的角色 | 说明 |
|---------|------------|------|
| **后训练**（SFT/RLHF/DPO） | 主力，单独使用即可 | 数据量小、集群规模几张到几十张卡，7B-70B 模型 FSDP 就够。veRL、TRL、LLaMA-Factory 默认后端都是 FSDP |
| **中训练**（Continual Pre-training） | 主力，单独使用即可 | 和后训练类似，规模适中 |
| **大规模预训练** | 3D 并行中的一环 | 70B+ 模型、几百上千张卡，需要 TP（张量并行）+ PP（流水线并行）+ FSDP（数据并行）配合，FSDP 只负责数据并行维度的切分 |

>  模型不太大、卡不太多 → FSDP 单独够用；超大规模预训练 → FSDP 是 3D 并行里数据并行那一维。

> 关于 3D 并行和 Megatron-LM 的详细介绍，参见 [[training/megatron|Megatron]]。

---

## 2. 开源生态

### 2.1 底层实现

| 库 | 维护方 | 说明 |
|----|--------|------|
| **PyTorch FSDP** | Meta | `torch.distributed.fsdp`，当前最主流的实现。有 FSDP1（`FullyShardedDataParallel`）和 FSDP2（`fully_shard`）两代 API |
| **DeepSpeed ZeRO** | Microsoft | ZERO 概念的原创者，Stage 1/2/3 对应不同切分级别，比 PyTorch FSDP 更早推出 |
| **FairScale** | Meta | FSDP 最初在此孵化，后 upstream 到 PyTorch 主库，现已基本弃用 |

### 2.2 上层训练框架

这些框架本身不实现切分逻辑，而是封装上述底层库，提供更友好的配置接口：

| 框架 | 封装的底层库 | 典型场景 |
|------|-------------|----------|
| **HuggingFace Accelerate** | PyTorch FSDP / DeepSpeed 均支持 | 通用训练，一套配置切换两种后端 |
| **PyTorch Lightning** | PyTorch FSDP | 通用训练，Strategy 插件机制 |
| **veRL** | PyTorch FSDP | LLM 强化学习训练（RLHF/GRPO 等） |
| **TRL** | Accelerate → PyTorch FSDP / DeepSpeed | HuggingFace 的 LLM 后训练框架 |
| **LLaMA-Factory** | Accelerate → PyTorch FSDP / DeepSpeed | LLM 微调一站式工具 |

> 选型建议：新项目推荐 **PyTorch FSDP2** 或 **DeepSpeed ZeRO Stage 3**，上层框架按团队习惯选。

---

## 3. FSDP 与 DDP 的区别

| 特性 | DDP (DistributedDataParallel) | FSDP |
|------|------------------------------|------|
| 参数存储 | 每个 GPU 完整复制一份参数 | 参数切分到多个 GPU |
| 显存占用 | 高（每卡存完整模型副本） | 低（约 1/K，K 为 GPU 数） |
| 梯度同步 | AllReduce | All-Gather + Reduce-Scatter |
| 优化器状态 | 每个 GPU 一份完整状态 | 优化器状态也被切分 |
| 通信开销 | 每步梯度同步 | 计算过程中多次 All-Gather |

**关键区别**：DDP 每个节点存完整的模型参数，而 FSDP 将参数分散存储，按需聚合。

---

## 4. ZERO 切分策略

FSDP 实现了 Microsoft DeepSpeed 的 ZERO (Zero Redundancy Optimizer) 策略，分为三个级别：

### 4.1 ZERO1 — 切分优化器状态

- **切分内容**：仅优化器状态（Adam 的 m 和 v）
- **参数存储**：每个 GPU 完整复制
- **梯度存储**：每个 GPU 完整复制
- **节省显存**：优化器部分（fp32 下 8 bytes/参数）

> PyTorch FSDP 没有原生 ZERO1 对应，DeepSpeed ZeRO Stage 1 提供此功能。

### 4.2 ZERO2 (SHARD_GRAD_OP)

- **切分内容**：优化器状态 + 梯度
- **参数存储**：每个 GPU 完整复制
- **节省显存**：优化器 + 梯度部分被切分

```python
fsdp_strategy = ShardingStrategy.SHARD_GRAD_OP  # 对应 ZERO2
```

### 4.3 ZERO3 (FULL_SHARD) — 默认策略

- **切分内容**：参数 + 梯度 + 优化器状态，全部切分
- **节省显存**：最大化，所有状态均被切分

```python
def get_sharding_strategy(device_mesh, zero3_enable=True):
    if zero3_enable:
        fsdp_strategy = ShardingStrategy.FULL_SHARD  # ZERO3
    else:
        fsdp_strategy = ShardingStrategy.SHARD_GRAD_OP  # ZERO2
```

---

## 5. 参数切分机制

### 5.1 存储：每层参数均匀切分到所有 GPU

FSDP 会将每个被包装模块的参数 **flatten 成一维张量**，然后**均匀切分到所有 GPU 上**。每个 GPU 持有每一层的一个 shard，而不是持有某几个完整的层。

```
Layer 1 参数 (flatten) = [p1, p2, p3, p4, p5, p6]
                          ├─ GPU 0 ─┤├─ GPU 1 ─┤├─ GPU 2 ─┤
                          [p1, p2]    [p3, p4]    [p5, p6]

Layer 2 参数 (flatten) = [q1, q2, q3, q4, q5, q6]
                          ├─ GPU 0 ─┤├─ GPU 1 ─┤├─ GPU 2 ─┤
                          [q1, q2]    [q3, q4]    [q5, q6]

→ 每个 GPU 持有所有层的一部分参数，而非某几个完整的层
```

### 5.2 计算：每个 GPU 仍然跑完所有层

参数虽然是分片存储的，但计算时每个 GPU 仍然执行**完整的前向/反向传播**（所有层都要算）。计算到某一层时，先通过 All-Gather 临时拼回该层的完整参数，算完再释放。

这和**流水线并行（Pipeline Parallelism）** 完全不同——流水线并行是不同 GPU 各负责不同的层，而 FSDP 是每个 GPU 都算所有层，只是参数分着存。

```
流水线并行：GPU 0 算 Layer 1-2，GPU 1 算 Layer 3-4（不同 GPU 算不同层）
FSDP：      GPU 0 算 Layer 1-2-3-4，GPU 1 也算 Layer 1-2-3-4（每个 GPU 算所有层）
```

> **一句话总结**：存储是切分的，计算是完整的。

---

## 6. All-Gather 与 Reshard

### 6.1 All-Gather：按需聚合

当某个层需要计算时，FSDP 会**临时**从所有 GPU 收集该层的完整参数：

```python
# 前向传播到 Layer N 时
all_gather(Layer N 参数片段 from 所有 GPU)  # 获得完整参数
执行 Layer N 计算
释放完整参数  # Reshard
```

**重要特性**：
- **不是一次性聚合整个模型**，而是按需聚合当前层
- 聚合后**立即释放**（reshard_after_forward=True）
- 每层计算都要进行一次 All-Gather

### 6.2 Reshard：立即释放

```python
# FSDP 配置
reshard_after_forward: true  # 前向传播后立即释放参数
```

为什么需要释放？
- 避免显存峰值飙升
- 为后续层的计算腾出显存
- 确保显存占用保持稳定

### 6.3 通信与计算重叠

为了减少通信开销，FSDP 支持通信与计算重叠：
- 在计算当前层时，预先 All-Gather 下一层的参数
- 减少等待时间，提高 GPU 利用率

---

## 7. 显存分析与估算

### 7.1 显存组成

训练时的显存占用主要由四部分组成：

| 组成部分 | 说明 | bf16 混合精度下的占比 (参考) |
|---------|------|----------------|
| **参数** | 模型权重 (bf16) | ~8% |
| **梯度** | 反向传播计算的梯度 (bf16) | ~8% |
| **优化器状态** | fp32 master weight + Adam m + v | ~50% |
| **激活值** | 前向传播的中间结果（含梯度检查点） | ~34% |

其中优化器状态是最大头：Adam 为每个参数维护两个额外状态（一阶矩 m 和二阶矩 v），加上 fp32 master weight，优化器总共需要 **12 bytes/参数**（master 4 + m 4 + v 4），这也正是 ZERO 优先切分优化器状态的原因。

> 关于 Adam 公式的详细推导，参见 [[fundamentals/gradient-descent|梯度下降]]。

### 7.2 各策略显存占用对比

以 7B 参数模型为例（单 GPU，混合精度 bf16 + fp32 master weights）：

| 策略 | 参数 | 梯度 | 优化器状态 (master+m+v) | 总计 (不含激活) |
|------|------|------|-----------|----------------|
| DDP (bf16) | 14GB | 14GB | 84GB | **112GB** |
| ZERO2 (4卡) | 14GB | 3.5GB | 21GB | **38.5GB** |
| ZERO3 (4卡) | 3.5GB | 3.5GB | 21GB | **28GB** |

> 计算方式：参数/梯度 = N × 2 bytes (bf16)。优化器包含 fp32 master weight + Adam m + v = N × (4+4+4) = 12 bytes/参数。
> ZERO2 切分梯度和优化器，ZERO3 额外切分参数。

**FSDP 不节省激活值显存**：FSDP 的显存节省主要来自参数、梯度和优化器状态，**不包括激活值**。激活值需要保留用于反向传播，每个 GPU 仍需计算完整的激活值，这也是为什么 FSDP 不能无限扩展显存节省。

### 7.3 单卡显存估算公式

对于 N 参数、L 层的模型，在 K 个 GPU 上用 FSDP ZERO3 训练：

**持久显存（训练全程占用）**：

```
参数 shard   = (N / K) × sizeof(dtype)
梯度 shard   = (N / K) × sizeof(dtype)
优化器 shard = (N / K) × 3 × sizeof(model_dtype)   # master weight + Adam m + v
```

**临时显存（峰值时额外占用）**：

```
All-Gather 临时参数 = (N / L) × sizeof(dtype)     # 仅当前层的完整参数
```

> **这是一个关键理解**：All-Gather 不是一次性把整个模型的参数都拉到每个 GPU 上。FSDP 是按 wrap 单元（通常是一个 Transformer Block）逐层 gather 的——gather 当前层，算完立即 reshard 释放，再 gather 下一层。所以临时显存开销只有 **一层的完整参数大小**，而不是整个模型。

**激活值显存**：

激活值和模型参数量没有简单的线性关系，它取决于 **batch size × 序列长度 × 隐藏层维度 × 层数**。注意这里说的是**训练**场景，每个样本的完整序列都要参与前向计算并保留中间结果用于反向传播，所以激活值和序列长度成正比。这和推理时的 decode 不同——decode 阶段每步只生成一个 token，用 KV Cache 避免重复计算，激活值开销很小。

粗略估算（不含梯度检查点）：

```
激活值 ≈ L × B × S × H × a × sizeof(dtype)

L = 层数, B = batch size, S = 序列长度, H = 隐藏层维度
a = 每层需要保存的激活张量数（通常 10-12 个，包括 attention scores、
    FFN 中间结果、LayerNorm 输入等）
```

**举个具体例子**（LLaMA-7B, bf16, B=32, S=4096）：

```
L=32, H=4096, a≈10

激活值 ≈ 32 × 32 × 4096 × 4096 × 10 × 2 bytes
       ≈ 32 × 32 × 4096 × 4096 × 20 bytes
       ≈ 343GB
```

这个数字非常夸张，实际中**一定会开梯度检查点**（activation checkpointing），只保留每层的输入，反向时重算中间激活，可以将激活值降低到原来的 **~1/a**：

```
开启梯度检查点后：
激活值 ≈ L × B × S × H × sizeof(dtype)
       ≈ 32 × 32 × 4096 × 4096 × 2 bytes
       ≈ 34GB
```

### 7.4 综合公式（bfloat16, 含梯度检查点）

```
单卡显存 ≈ 持久 shard + All-Gather 临时 + 激活值
         ≈ (5N/K) + (2N/L) + (2 × L × B × S × H) bytes
```

### 7.5 实际案例

以 LLaMA-7B（L=32, H=4096）、ZERO3 4 卡、bf16、梯度检查点为例：

| 组成部分 | 公式 | 估算值 |
|---------|------|--------|
| 参数 shard | N/K × 2 | 3.5 GB |
| 梯度 shard | N/K × 2 | 3.5 GB |
| 优化器 shard (master+m+v) | N/K × 12 | 21 GB |
| All-Gather 临时（1 层） | N/L × 2 | ~0.44 GB |
| 激活值 (B=32, S=4096, 梯度检查点) | L×B×S×H×2 | ~34 GB |
| **总计** | | **~62 GB** |

> 不开梯度检查点的话激活值会膨胀到 ~343GB，根本放不下。所以大模型训练梯度检查点几乎是必开的。

### 7.6 不同训练场景下的激活值分析

前面公式中的 S（序列长度）在实际训练中是指前向传播处理的完整 token 数。由于 Transformer 的自回归特性，前向传播必须从第一个 token 算到最后一个 token 才能得到每个位置的 logits，因此激活值和**整个输入序列的长度**相关，而非仅和某一部分相关。

以 SFT 和 GRPO 两个典型场景为例：

**SFT**：输入是 prompt + response 拼接的完整序列，前向传播处理全部 token：

```
S = len(prompt) + len(response)

激活值 ∝ B × S × H × L
```

> loss 虽然 mask 掉 prompt 只算 response 部分，但这只影响哪些 token 产生梯度，不影响前向传播的计算量和激活值大小。如果使用 packing（多个样本拼满 max_seq_len），则 S = max_seq_len。

**GRPO**：分两个阶段，显存特征不同：

```
Rollout（生成）阶段：
  推理模式，自回归逐 token 生成，不保留激活值
  显存开销主要是 KV Cache，和训练阶段不在一个量级

Training（策略更新）阶段：
  把完整的 prompt + 生成的 response 重新前向传播（需要梯度）
  S = len(prompt) + len(response)
  有效 batch = B × G（每个 prompt 生成 G 个 response）

  激活值 ∝ B × G × S × H × L
               ↑
         G 倍扩展是 GRPO 比 SFT 更吃显存的主因
```

> GRPO 的显存压力主要不是来自序列更长，而是来自 G 倍的 response 扩展（通常 G=4~16）。这也是为什么 GRPO 实践中往往需要更激进的梯度累积或更小的 batch size。

### 7.7 注意事项

- 激活值和 **B × S** 成正比，降 batch size 或缩短序列长度可以直接减少激活值显存
- 可结合 [[training/mixed-precision|混合精度]] 和梯度检查点进一步优化
- All-Gather 临时开销很小（单层参数），这也是 FSDP 按 Transformer Block 粒度 wrap 的原因之一

---

## 8. 混合精度训练

### 8.1 dtype 与 model_dtype

FSDP 配置中有两个精度设置：

```yaml
# FSDP 配置
strategy: fsdp
dtype: bfloat16          # 计算和存储精度
model_dtype: fp32        # 模型参数原始精度（主权重）
reshard_after_forward: true
param_offload: false
optimizer_offload: false
```

**dtype (bfloat16)**：
- 前向/反向传播的计算精度
- 临时存储的参数精度
- 梯度的存储精度
- 显存占用：2 bytes/参数

**model_dtype (fp32)**：
- 优化器中维护的"主权重"精度
- 更新参数时使用的精度
- 避免 bfloat16 的数值精度问题
- 显存占用：4 bytes/参数

### 8.2 精度切换流程

```
训练循环：
1. 从优化器取出 fp32 master weights
2. 转换为 bf16 用于计算
3. 前向传播 (bf16)
4. 反向传播，计算梯度 (bf16)
5. 将 bf16 梯度传给优化器
6. 优化器用 fp32 精度更新 master weights（Adam: w -= lr × m/(√v+ε)）
7. 回到第 1 步
```

---

## 9. PyTorch FSDP 实战

### 9.1 auto_wrap_policy

`auto_wrap_policy` 决定了哪些模块被包装成独立的 FSDP 单元，直接影响通信粒度和显存效率：

```python
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# 以 Transformer 层为单位包装，最常用的策略
auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={LlamaDecoderLayer},
)
```

- 粒度太粗（整个模型一个 FSDP 单元）→ All-Gather 一次性聚合太多参数，显存峰值高
- 粒度太细（每个 Linear 层都包装）→ 通信次数太多，开销大
- **最佳实践**：以 Transformer Block 为单位包装

### 9.2 完整的 FSDP 配置

```yaml
# verl/trainer/config/engine/fsdp.yaml
strategy: fsdp
dtype: bfloat16          # 计算精度
model_dtype: fp32        # 主权重精度
reshard_after_forward: true   # 前向后立即释放参数
param_offload: false     # 不卸载参数到 CPU
optimizer_offload: false # 不卸载优化器到 CPU

# 切分策略（在代码中设置）
zero3_enable: true       # 使用 ZERO3 (FULL_SHARD)
```

### 9.3 优化器配置

```yaml
# 学习率
lr: 1e-5                 # Adam 的 alpha 参数

# Adam 其他参数（通常使用默认值）
beta1: 0.9               # 一阶矩衰减
beta2: 0.999             # 二阶矩衰减
epsilon: 1e-8            # 数值稳定常数
weight_decay: 0.0        # 权重衰减
```

### 9.4 代码中的切分策略选择

```python
from torch.distributed.fsdp import ShardingStrategy

def get_sharding_strategy(device_mesh, zero3_enable=True):
    """根据配置选择 FSDP 切分策略"""
    if zero3_enable:
        # ZERO3: 完全切分（参数 + 梯度 + 优化器状态）
        fsdp_strategy = ShardingStrategy.FULL_SHARD
    else:
        # ZERO2: 切分梯度 + 优化器状态
        fsdp_strategy = ShardingStrategy.SHARD_GRAD_OP
    return fsdp_strategy
```

### 9.5 FSDP2：新一代 API

PyTorch 2.x 推出了 FSDP2（`torch.distributed.fsdp.fully_shard`），相比旧 API 的主要变化：

- **per-parameter sharding**：不再 flatten 所有参数为一维，每个参数独立切分，调试更友好
- **DTensor 原生支持**：基于 DTensor 抽象，与 Tensor Parallelism 等组合更自然
- **API 更简洁**：`fully_shard(module)` 替代原来的 `FSDP(module, ...)`

```python
from torch.distributed.fsdp import fully_shard

# FSDP2 用法
fully_shard(model.layers)  # 对每个 Transformer 层应用
fully_shard(model)          # 对整个模型应用
```

> 如果是新项目，建议直接使用 FSDP2。

---

## 10. DeepSpeed ZeRO 实战

### 10.1 DeepSpeed 配置结构

DeepSpeed 通过一个 JSON 配置文件控制所有分布式训练行为：

```json
{
  "bf16": { "enabled": true },
  "zero_optimization": {
    "stage": 3,
    "offload_param": { "device": "none" },
    "offload_optimizer": { "device": "none" },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6
  },
  "gradient_accumulation_steps": 4,
  "train_micro_batch_size_per_gpu": 2,
  "wall_clock_breakdown": false
}
```

### 10.2 关键配置项说明

| 配置项 | 说明 |
|--------|------|
| `stage` | ZeRO 级别：1（切优化器）、2（+梯度）、3（+参数） |
| `offload_param` | 参数卸载到 CPU/NVMe，显存不够时开启 |
| `offload_optimizer` | 优化器状态卸载到 CPU/NVMe |
| `overlap_comm` | 通信与计算重叠，建议开启 |
| `reduce_bucket_size` | 梯度通信桶大小，影响通信效率 |
| `stage3_prefetch_bucket_size` | Stage 3 预取桶大小，类似 FSDP 的 prefetch |
| `stage3_param_persistence_threshold` | 小于此值的参数不切分，减少通信开销 |

### 10.3 通过 Accelerate 使用 DeepSpeed

HuggingFace Accelerate 可以一套代码切换 FSDP 和 DeepSpeed：

```yaml
# accelerate_config.yaml
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
deepspeed_config:
  deepspeed_config_file: ds_config.json
  zero3_init_flag: true
```

```bash
accelerate launch --config_file accelerate_config.yaml train.py
```

### 10.4 PyTorch FSDP vs DeepSpeed ZeRO 选型

| 维度 | PyTorch FSDP | DeepSpeed ZeRO |
|------|-------------|----------------|
| 生态整合 | PyTorch 原生，与 torch.compile 等兼容性好 | 独立库，需额外安装 |
| CPU/NVMe Offload | 支持但功能较基础 | ZeRO-Offload / ZeRO-Infinity，更成熟 |
| 超大模型 | FSDP2 + DTensor 组合较灵活 | ZeRO Stage 3 + Offload 经验丰富 |
| 调试体验 | FSDP2 per-parameter sharding 更透明 | 黑盒感较强，排错依赖日志 |
| 社区趋势 | Meta 主推，与 PyTorch 路线图一致 | 微软持续维护，生态广泛 |

> 两者功能上高度重叠，选哪个主要看团队技术栈和上层框架的支持情况。
---
