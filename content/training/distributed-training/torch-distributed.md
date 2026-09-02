---
title: torch.distributed
created: 2026-07-06
published: 2026-07-06
modified: 2026-07-06
type: topic
status: growing
area: training
tags:
  - distributed-training
  - pytorch
  - communication
---

`torch.distributed` 是 PyTorch 的分布式通信与分布式训练基础模块。DDP、FSDP、Tensor Parallel、Pipeline Parallel、Megatron-style 训练以及许多 RL 训练框架都会直接或间接依赖它。理解 `torch.distributed` 的关键，不是先记住某个训练框架的封装，而是理解多进程、多 GPU 之间如何组织通信。

在大模型训练中，`torch.distributed` 主要承担三类职责：

- 初始化 distributed process group；
- 在 ranks 之间执行 point-to-point 或 collective communication；
- 为 DDP/FSDP/TP/PP 等上层并行策略提供底层通信原语。

## 基本概念

分布式训练通常以多进程形式运行。每个进程绑定一个或多个 GPU，进程之间通过通信 backend 交换 tensor。

常见术语包括：

- **rank**：当前进程在某个 process group 中的编号；
- **world size**：process group 中进程总数；
- **local rank**：当前节点内部的进程编号，通常用于绑定本机 GPU；
- **global rank**：跨所有节点的全局进程编号；
- **process group**：一组参与通信的 ranks；
- **backend**：实际执行通信的后端，例如 NCCL、Gloo、MPI；
- **collective operation**：一组 ranks 共同参与的通信操作；
- **point-to-point operation**：两个 ranks 之间的发送与接收。

最小化理解如下：

```text
rank:
  一个训练进程

process group:
  一组需要互相通信的 ranks

backend:
  负责把 tensor 真正传过去的通信实现
```

## 初始化 Process Group

典型 PyTorch 分布式训练会先调用：

```python
import torch.distributed as dist

dist.init_process_group(
    backend="nccl",
    init_method="env://",
)
```

常见环境变量包括：

- `MASTER_ADDR`：主节点地址；
- `MASTER_PORT`：主节点端口；
- `WORLD_SIZE`：全局进程数；
- `RANK`：当前进程全局 rank；
- `LOCAL_RANK`：当前节点内 rank。

单机 8 卡训练通常有 8 个进程，`WORLD_SIZE=8`，每个进程一个 `RANK`。多机训练则把所有节点上的进程组成一个更大的 global process group。

训练脚本中常见绑定方式：

```python
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
```

如果 rank、device、process group 初始化错误，后续 collective operation 可能挂起、超时或产生错误结果。

## Backend

`torch.distributed` 支持不同 backend。大模型 GPU 训练中最常见的是 NCCL。

常见 backend：

- **NCCL**：NVIDIA GPU 上的高性能 collective communication backend，大模型训练主力；
- **Gloo**：CPU 和部分 GPU 场景可用，常用于调试或非 NVIDIA 环境；
- **MPI**：依赖系统 MPI 环境，更多见于 HPC 生态。

对 LLM 训练而言，NCCL 通常负责 GPU 间的 AllReduce、AllGather、ReduceScatter、Broadcast、AllToAll 等操作。用户在 PyTorch 里调用 `dist.all_reduce(...)`，底层通常由 NCCL 在 NVLink、PCIe、InfiniBand 等硬件链路上执行。

## 通信分类

`torch.distributed` 中的通信可分为两大类：

```text
Distributed training communication
  -> point-to-point communication
       - send
       - recv
       - isend
       - irecv
  -> collective communication
       - all_reduce
       - all_gather
       - reduce_scatter
       - broadcast
       - scatter
       - gather
       - all_to_all
```

Point-to-point communication 面向两个 ranks 之间的直接通信。Collective communication 面向整个 process group 或其中一组 ranks 的协同通信。

大模型训练中，collective 更常见，因为梯度同步、参数聚合、张量并行和 expert dispatch 往往都需要多个 ranks 同时参与。

## Point-to-point Communication

Point-to-point communication 包括 `send` 和 `recv`。它要求发送方和接收方配对出现。

```python
if rank == 0:
    dist.send(tensor, dst=1)
elif rank == 1:
    dist.recv(tensor, src=0)
```

异步版本包括：

```python
work = dist.isend(tensor, dst=1)
work.wait()
```

P2P 通信适合表达相邻 stage 之间的数据传递，例如 [[training/distributed-training/pipeline-parallel|Pipeline Parallel]] 中相邻 pipeline stages 传递 activation 和 activation gradient。它也常用于更复杂调度中的局部通信。

P2P 的风险是配对关系必须完全正确。若 rank 0 在等 rank 1 接收，而 rank 1 没有对应 `recv`，程序可能挂起。

## Collective Communication

Collective communication 是一组 ranks 同时参与的通信操作。所有参与 ranks 通常必须以相同顺序调用同一个 collective，并且 tensor shape、dtype、device 等需要匹配。

错误示例：

```text
rank 0 调用了 all_reduce
rank 1 调用了 all_gather
```

这类错误通常不会得到有意义的训练结果，常见表现是 hang、timeout 或 NCCL error。

## AllReduce

`all_reduce` 的语义是：所有 ranks 各自提供一个 tensor，先做 reduce 操作，再把 reduce 结果发回所有 ranks。

以 sum 为例：

```text
Before:
  rank 0: [1, 2]
  rank 1: [3, 4]
  rank 2: [5, 6]

AllReduce(sum):
  [1+3+5, 2+4+6] = [9, 12]

After:
  rank 0: [9, 12]
  rank 1: [9, 12]
  rank 2: [9, 12]
```

PyTorch 调用：

```python
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
```

在 [[training/distributed-training/data-parallel|Data Parallel]] / DDP 中，AllReduce 常用于同步梯度。每个 rank 对不同数据 shard 计算本地梯度，再通过 AllReduce 得到全局平均梯度。实际实现中通常先 sum，再除以 data parallel world size。

核心特征：

```text
输入：每个 rank 一个完整 tensor
输出：每个 rank 都得到完整 reduce 结果
典型用途：DDP gradient synchronization
```

## AllGather

`all_gather` 的语义是：每个 rank 提供一个 shard，所有 ranks 收集所有 shards，并在每个 rank 上得到完整结果。

```text
Before:
  rank 0: [A]
  rank 1: [B]
  rank 2: [C]

After:
  rank 0: [A, B, C]
  rank 1: [A, B, C]
  rank 2: [A, B, C]
```

PyTorch 中常见调用形式：

```python
gathered = [torch.empty_like(local) for _ in range(world_size)]
dist.all_gather(gathered, local)
```

在 [[training/distributed-training/fsdp|FSDP]] / ZeRO-3 中，AllGather 常用于临时聚合当前层参数。平时每张 GPU 只保存参数 shard；计算某层时，需要完整参数，于是 all-gather 当前层参数，计算后再 reshard。

核心特征：

```text
输入：每个 rank 一个 shard
输出：每个 rank 都得到所有 shards
典型用途：FSDP/ZeRO-3 parameter all-gather
```

## ReduceScatter

`reduce_scatter` 的语义是：先对所有 ranks 的输入做 reduce，再把 reduce 后的结果切分给不同 ranks。

可以理解为：

```text
ReduceScatter = Reduce + Scatter
```

示例：

```text
Before:
  rank 0: [1, 2, 3, 4]
  rank 1: [10, 20, 30, 40]

Reduce(sum):
  [11, 22, 33, 44]

Scatter:
  rank 0: [11, 22]
  rank 1: [33, 44]
```

PyTorch 调用形式随版本和 tensor 组织略有差异，核心语义不变：

```python
dist.reduce_scatter(output, input_list, op=dist.ReduceOp.SUM)
```

在 FSDP / ZeRO 中，ReduceScatter 常用于梯度归约并分片保存。这样每个 rank 不必保存完整全局梯度，而是只保存 reduce 后属于自己的 gradient shard。

核心特征：

```text
输入：每个 rank 提供完整或分块输入
输出：每个 rank 得到 reduce 结果的一片
典型用途：FSDP/ZeRO gradient reduce-scatter
```

## Broadcast

`broadcast` 的语义是：一个源 rank 持有 tensor，并把它发送给 process group 中所有 ranks。

```text
Before:
  rank 0: [A]
  rank 1: [?]
  rank 2: [?]

Broadcast(src=0):
  rank 0: [A]
  rank 1: [A]
  rank 2: [A]
```

PyTorch 调用：

```python
dist.broadcast(tensor, src=0)
```

Broadcast 常用于：

- 初始化时同步参数；
- 同步配置或标量状态；
- 从某个 rank 分发随机种子或小型 metadata；
- 某些 pipeline / model parallel 初始化逻辑。

核心特征：

```text
输入：src rank 有有效 tensor
输出：所有 ranks 得到 src tensor
```

## Scatter

`scatter` 的语义是：源 rank 持有多个 shards，并将不同 shard 分发给不同 ranks。

```text
Before:
  rank 0: [A, B, C]
  rank 1: [?]
  rank 2: [?]

Scatter(src=0):
  rank 0: [A]
  rank 1: [B]
  rank 2: [C]
```

PyTorch 调用：

```python
if rank == 0:
    scatter_list = [a, b, c]
else:
    scatter_list = None

dist.scatter(output, scatter_list=scatter_list, src=0)
```

Scatter 比 AllGather / AllReduce 在大模型训练主路径中少见，但它是理解参数分发、数据分片和 collective family 的基础操作。

## Gather

`gather` 的语义与 scatter 相反：各 rank 提供一个 tensor，目标 rank 收集所有 tensor。

```text
Before:
  rank 0: [A]
  rank 1: [B]
  rank 2: [C]

Gather(dst=0):
  rank 0: [A, B, C]
  rank 1: [B]
  rank 2: [C]
```

PyTorch 调用：

```python
if rank == 0:
    gather_list = [torch.empty_like(local) for _ in range(world_size)]
else:
    gather_list = None

dist.gather(local, gather_list=gather_list, dst=0)
```

Gather 常用于汇总小规模统计、debug 信息或评测输出。训练主路径中若把大 tensor 全部 gather 到单个 rank，容易形成内存和通信瓶颈。

核心区别：

```text
Gather:
  只有 dst rank 得到完整结果

AllGather:
  所有 ranks 都得到完整结果
```

## AllToAll

`all_to_all` 的语义是：每个 rank 都把自己的输入切成多份，分别发送给所有 ranks，同时也从所有 ranks 接收对应分片。

示意：

```text
Before:
  rank 0: [A0, A1, A2]
  rank 1: [B0, B1, B2]
  rank 2: [C0, C1, C2]

After:
  rank 0: [A0, B0, C0]
  rank 1: [A1, B1, C1]
  rank 2: [A2, B2, C2]
```

PyTorch 调用：

```python
dist.all_to_all(output_list, input_list)
```

AllToAll 在 MoE expert parallel、sequence/context layout 转换、某些 tensor redistribution 场景中很重要。例如 MoE 中每个 rank 上的 tokens 需要根据 router 分配给不同 expert ranks，这类 token dispatch / combine 往往会用到 AllToAll 或类似通信模式。

核心特征：

```text
输入：每个 rank 按目标 rank 切分数据
输出：每个 rank 收到来自所有 ranks 的目标分片
典型用途：MoE expert dispatch、layout redistribution
```

## 通信原语与训练策略的关系

不同并行策略可以理解为在不同张量和不同 process group 上组合通信原语。

| 训练策略 | 主要通信对象 | 常见通信原语 |
|---|---|---|
| DDP / Data Parallel | gradients | AllReduce |
| FSDP / ZeRO-3 | parameters, gradients, optimizer states | AllGather, ReduceScatter |
| Tensor Parallel | layer activation / partial output | AllReduce, AllGather, ReduceScatter |
| Pipeline Parallel | activation, activation gradient | Send / Recv |
| Sequence Parallel | activation layout | ReduceScatter, AllGather |
| Context Parallel | K/V blocks, attention outputs | AllGather, AllToAll, P2P or custom collectives |
| MoE Expert Parallel | token dispatch / expert output | AllToAll |

因此，读 Megatron、FSDP 或 verl 源码时，看到 `all_reduce`、`all_gather`、`reduce_scatter` 并不是孤立 API，而是在实现某个并行策略的数学语义和状态布局。

在 [[training/distributed-training/megatron|Megatron]] 中，`torch.distributed` 处在最底层通信原语层。Megatron 的 `parallel_state` 负责创建 TP/PP/DP/CP/EP process groups；tensor parallel layer、pipeline schedule 和 distributed optimizer 再基于这些 groups 调用 collective 或 point-to-point communication。换言之，Megatron 的核心不是替代 `torch.distributed`，而是在其上组织大模型并行训练语义。

## Process Group 与多维并行

大模型训练很少只有一个全局 process group。多维并行通常会创建多个 process groups：

- data parallel group；
- tensor parallel group；
- pipeline parallel group；
- context parallel group；
- expert parallel group；
- global group。

例如 world size 为 64，配置为：

```text
TP = 4
PP = 2
DP = 8
```

则：

- 一个 TP group 内有 4 个 ranks，共同计算同一层的张量分片；
- 一个 PP group 内有 2 个 pipeline stages；
- 一个 DP group 内有 8 份模型并行副本，用于不同数据 shard 的梯度同步或状态切分。

同一个 `dist.all_reduce(tensor, group=...)`，传入不同 process group，语义完全不同。对源码阅读而言，必须先确认通信发生在哪个 group 上。

## 同步、异步与 Work Handle

部分 `torch.distributed` API 支持 `async_op=True`，返回 work handle：

```python
work = dist.all_reduce(tensor, async_op=True)

# 做一些可并行的计算

work.wait()
```

异步通信的目标是 overlap communication with computation。DDP 的 bucketed gradient AllReduce、FSDP 的 parameter prefetch、Megatron 中的通信重叠，都与这一思想有关。

但异步通信也会增加调试难度：

- 使用通信结果前必须 `wait()`；
- tensor 在通信完成前不应被错误修改；
- 不同 ranks 的调用顺序仍需一致；
- 错误可能延迟暴露。

## 常见错误模式

- **collective 调用顺序不一致**：不同 ranks 以不同顺序调用 collective，容易 hang。
- **process group 用错**：把应在 TP group 内做的通信放到 global group，导致结果错误或 shape 不匹配。
- **tensor shape / dtype / device 不一致**：NCCL collective 需要参与 ranks 的 tensor layout 匹配。
- **rank 与 GPU 绑定错误**：多个进程绑定同一 GPU，或 local rank 设置错误。
- **没有设置 timeout / debug 环境**：通信 hang 后难以定位。
- **异步通信未 wait**：使用了尚未完成通信的 tensor。
- **通信量超过收益**：并行度过高，collective latency 和 bandwidth 成为瓶颈。
- **把 collective 当作普通函数调用**：collective 是分布式协议的一部分，必须从所有参与 ranks 共同、同序调用。

## 阅读源码时的定位方法

阅读 Megatron、FSDP、verl 或其他训练框架时，遇到通信 API 可按以下顺序分析：

1. 当前 rank 属于哪些 process groups；
2. 这个 collective 在哪个 group 上调用；
3. 通信前 tensor 的 shape、dtype、device 和 shard layout；
4. 通信后的 tensor shape 和语义；
5. 该通信对应哪种并行策略；
6. 是否与计算重叠；
7. checkpoint 或 dataloader 是否需要记录对应 parallel metadata。

例如：

```text
看到 all_reduce:
  可能是 DDP 梯度同步
  也可能是 TP 中 row-parallel output 合并

看到 all_gather:
  可能是 FSDP 当前层参数聚合
  也可能是 TP/CP 中 activation 或 K/V 收集

看到 reduce_scatter:
  可能是 FSDP 梯度分片
  也可能是 sequence parallel layout 转换
```

判断依据不是 API 名字本身，而是 process group、tensor shape 和上下文模块。

## 与 NCCL / MPI 的关系

`torch.distributed` 是 PyTorch 层的 API。NCCL、Gloo、MPI 是底层 backend 或通信库。

在 NVIDIA GPU 大模型训练中，常见路径是：

```text
training framework
  -> torch.distributed API
  -> NCCL backend
  -> NVLink / PCIe / InfiniBand
```

因此，通信性能不只取决于 PyTorch 代码，还取决于：

- GPU 拓扑；
- 节点内 NVLink / NVSwitch；
- 节点间 InfiniBand / RoCE；
- NCCL 算法选择；
- bucket size；
- tensor size；
- process placement；
- communication overlap。

## GShard 与 Collective Communication

[[sources/papers/2020-gshard|GShard]] 提供了一个理解自动分片的早期范例。它基于 XLA SPMD partitioner，根据 tensor 的 sharding annotations 自动把 full-size operator 转成 partition-sized operator，并插入 `AllReduce`、`AllGather`、`AllToAll` 和 `CollectivePermute` 等通信。

GShard 中的通信语义与当前 `torch.distributed` 的 collective 抽象是相通的，但实现栈不同：GShard 主要面向 TPU/XLA，由 compiler 在 HLO 层生成通信；GPU 训练通常由 PyTorch 调用 NCCL backend 执行。阅读现代训练代码时，可以用同样的问题分析两者：通信前 tensor 按哪一维分片，通信后 layout 如何变化，为什么这里需要 reshard 而不是 local compute，以及设备拓扑是否支持这个通信模式。

在 MoE 中，GShard 的 group-to-expert layout 转换使用 `AllToAll` 完成 token dispatch/combine；矩阵沿 contracting dimension 分片时使用 `AllReduce` 合并 partial results；需要把 sharded tensor 恢复为 replicated tensor 时使用 `AllGather`。这也是现代 expert parallel、tensor parallel 和 sequence/context layout 转换中反复出现的基本思路。

## 相关概念

- [[training/distributed-training/data-parallel|Data Parallel]]
- [[training/distributed-training/fsdp|FSDP]]
- [[training/distributed-training/zero|ZeRO]]
- [[training/distributed-training/tensor-parallel|Tensor Parallel]]
- [[training/distributed-training/pipeline-parallel|Pipeline Parallel]]
- [[training/distributed-training/sequence-parallel|Sequence Parallel]]
- [[training/distributed-training/context-parallel|Context Parallel]]
- [[training/distributed-training/megatron|Megatron 与 3D 并行]]
- [[training/data-engineering/distributed-dataloader|Distributed Dataloader]]
- [[training/optimization/checkpoint-sharding|Checkpoint Sharding]]
