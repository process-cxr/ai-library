---
title: 06 Single Controller and Ray
created: 2026-09-02
published: 2026-09-02
modified: 2026-11-03
type: project
status: growing
tags:
  - projects
  - source-reading
  - verl
  - ray
  - distributed-systems
---

05 已经把 P0 的回答、reward、advantage 和 token probability 连成了一个可求导的目标。但 controller 中没有八个 GPU 上的 forward/backward 循环，它只调用了一次：

```python
output = self.actor_rollout_wg.update_actor(batch)
```

这里的 `batch` 还是 512 个 trajectory key 的 metadata。为什么这一行能让两台机器的八张 GPU 各自取出数据、累计四个 micro-batch 的梯度，并协作完成四次参数更新？这一章沿着这次调用展开。`P0_0_0` 始终是我们跟踪的那条回答，硬件与数据配置继续使用 [[projects/verl-source-reading/01-sync-grpo-run-overview|01 的两机八卡案例]]。

## 调用前，512 条数据和八个进程分别在哪里

controller 知道本轮使用哪些 trajectory、每次更新的全局大小是多少。TransferQueue 保存这些 trajectory 的实际字段。GPU worker 持有模型和 optimizer，并且已经加入 torch.distributed process group。

所以传给 `update_actor` 的对象可以概括为：

```text
KVBatchMeta
  partition_id = "train"
  keys = 512 个 trajectory key，含 P0_0_0、P0_1_0、P0_2_0、P0_3_0
  extra_info:
    mini_batch_size = 128
    global_batch_size = 128
    epochs = 1
    temperature = 1.0
    ...
```

`len(batch)=512` 与 `mini_batch_size=128` 并不矛盾。前者表示此次 RPC 交给 worker 的全部样本，后者告诉 worker 应当每次拿多少全局样本执行一次更新。key 在此次调用前已按 workload 重排；它们的行位置可以变化，key 和对应 fields 的身份关系保持不变。

本例的资源配置是：

```python
resource_pool_spec = {
    "global_pool": [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
}
# n_gpus_per_node=4，nnodes=2 → {"global_pool": [4,4]}
```

[`_init_resource_pool_mgr`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L737) 把 `Role.ActorRolloutRef` 映射到这个 pool。本例启用 reference KL、没有 LoRA，因此选用包含 actor、rollout 和 reference 的角色；没有 critic 或 teacher worker。

`[4,4]` 表示两个节点各有四个 worker slot，合计 world size=8。下面将排序后的两个节点简称 Node 0、Node 1。GPU slot 是资源池分配位置，不是预先保证的宿主机 CUDA 设备编号。

| 节点   | 本机 GPU slot | global rank | 本机 rank | actor DP rank | 本次收到的 key 数 |
| ------ | ------------: | ----------: | --------: | ------------: | ----------------: |
| Node 0 |             0 |           0 |         0 |             0 |                64 |
| Node 0 |             1 |           1 |         1 |             1 |                64 |
| Node 0 |             2 |           2 |         2 |             2 |                64 |
| Node 0 |             3 |           3 |         3 |             3 |                64 |
| Node 1 |             0 |           4 |         0 |             4 |                64 |
| Node 1 |             1 |           5 |         1 |             5 |                64 |
| Node 1 |             2 |           6 |         2 |             6 |                64 |
| Node 1 |             3 |           7 |         3 |             7 |                64 |

actor 采用 FSDP、`fsdp_size=-1`、Ulysses SP=1。FSDP engine 的 [`get_data_parallel_size`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine/fsdp/transformer_impl.py#L640) 返回 `world_size / SP = 8`，因此这里的 global rank 恰好等于 DP rank。每个 rank 处理不同样本，同时八个 rank 共同维护被分片的模型状态；DP=8 不表示八份完整参数必须一直常驻显存。

## Ray 先把进程放到这些 GPU slot 上

模型计算之前，[`RayResourcePool.get_placement_groups`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/single_controller/ray/base.py#L131) 为资源池中的每个节点位置构造 placement group，每组含四个 bundle。GPU bundle 含一份 GPU 资源和供 colocated worker 使用的 CPU 资源。

[`RayWorkerGroup._init_with_resource_pool`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/single_controller/ray/base.py#L538) 随后按 placement group、本机 rank 创建远程 actor。以表中的 global rank 5 为例，创建时会设置：

```text
WORLD_SIZE=8
RANK=5
RAY_LOCAL_WORLD_SIZE=4
MASTER_ADDR=<本次分布式任务的rendezvous地址>
MASTER_PORT=<本次分布式任务的rendezvous端口>
WG_BACKEND=ray
```

这里的 `MASTER_ADDR/PORT` 用于训练进程 rendezvous，与后面的 vLLM HTTP 服务地址不同。worker 启动后，`TrainingWorker.__init__` 调用 `initialize_global_process_group_ray`，再由 engine 创建模型相关的 mesh 和通信组。

这两步各解决一个问题：Ray 让八个进程得到执行位置和句柄；torch.distributed 让这八个进程能够共同执行 collective。只创建八个 Ray actor，并不会自动得到 FSDP，也不会自动把梯度同步起来。

本例两个节点共同参与同一个 actor FSDP world，因此模型状态通信会跨节点。网络性能影响这一阶段的执行时间，但不会改变 `512/8=64` 的样本分配数量。

## `actor_rollout_wg.update_actor` 为什么是一个可调用方法

进入 worker 之前，先看 controller 上的这个方法来自哪里。worker 类中有如下真实方法：

```python
@register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
def update_actor(self, data: TensorDict) -> TensorDict:
    output = self.actor.train_mini_batch(data=data)
    return output.cpu() if output is not None else None
```

这里省去了 profiling 和 routing-replay 装饰器，主体见 [`ActorRolloutRefWorker.update_actor`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine_workers.py#L707)。`register` 保存 dispatch、execute、blocking 属性，并包装 TQ 数据桥接逻辑；它本身不创建远程进程。

WorkerGroup 初始化时，[`_bind_worker_method`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/single_controller/base/worker_group.py#L185) 扫描类方法，读取这些属性，生成 controller 侧的同名代理。代理的核心只有几行：

```python
args, kwargs = dispatch_fn(self, *args, **kwargs)
output = execute_fn(method_name, *args, **kwargs)
if blocking:
    output = ray.get(output)
output = collect_fn(self, output)
```

这段来自 [`func_generator`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/single_controller/ray/base.py#L49)，省略了 padding cleanup。对本次调用，四行分别意味着：将 512 个 key 切成八份；远程调用八个 worker；等待执行完成；按 collect 规则合并返回值。

因此 controller 上的 `update_actor` 不是某一张卡的方法，也不是它自己执行 Python 函数中的训练循环。它是一组远程执行的代理，只是保持了与 worker 相同的训练语义名称。

## dispatch 先问 actor mesh，再切 key

`mesh_name="actor"` 指定要查询 actor 的 DP mapping。worker 初始化 actor engine 后，把 engine 返回的 DP rank 和是否产出有效输出注册到这个 mesh 名称下：

```text
FSDP engine 的 DP rank / output rank
  → TrainingWorker 的 train dispatch 信息
  → ActorRolloutRefWorker 的 actor dispatch 信息
  → WorkerGroup 查询并缓存
```

[`dispatch_lazy_compute_data_proto`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/single_controller/base/decorator.py#L266) 的核心是：

```python
dp_rank_mapping = worker_group._dispatch_info[mesh_name]
dp_size = max(dp_rank_mapping) + 1
return dispatch_nd_compute_dataproto(
    dp_rank_mapping, dp_size, worker_group, *args, **kwargs
)
```

本例查询结果为：

```text
global rank:    0 1 2 3 4 5 6 7
actor DP rank:  0 1 2 3 4 5 6 7
is_collect:    T T T T T T T T
```

于是 `BatchData(...).chunk(chunks=8)` 对 `KVBatchMeta` 分块；每块包含 64 个 key，并保留 `mini_batch_size=128` 等公共信息。这里还没有从 TQ 读取全部训练 tensor，也没有切成 128 条的全局 mini-batch。

设重排后的 key 列表为 `K`，分发关系是：

```text
rank 0 ← K[  0: 64]
rank 1 ← K[ 64:128]
rank 2 ← K[128:192]
rank 3 ← K[192:256]
rank 4 ← K[256:320]
rank 5 ← K[320:384]
rank 6 ← K[384:448]
rank 7 ← K[448:512]
```

沿用 04 和 05 的同一份教学排列，P0 的四条回答对应下面的位置，所有下标从 0 开始：

| trajectory key | 重排后的全批下标 | 收到它的 actor DP rank | rank 内本地下标 |
| -------------- | ---------------: | ---------------------: | --------------: |
| `P0_0_0`       |               16 |                      0 |              16 |
| `P0_1_0`       |              144 |                      2 |              16 |
| `P0_2_0`       |              272 |                      4 |              16 |
| `P0_3_0`       |              400 |                      6 |              16 |

例如 `144 - 2×64 = 16`，所以 `P0_1_0` 是 rank 2 的第 17 条。这个排列用于贯穿各章，不是实际运行记录，也不是按 session 编号分配 GPU 的规则。`_balance_batch` 根据全批序列工作量构造等条数的 partition，实际布局取决于全部 512 条数据。

四条回答分开并不会破坏 GRPO。05 已在 controller 收齐全批字段，按 `uid=P0` 算好 advantage，并分别写入四个 trajectory key。现在每个 rank 只需取自己的 key 及其 advantage，无须重新收齐同题回答来计算 loss。

## worker 收到 key 后，才取出真实 tensor

Ray 把 64 个 key 送到 rank 0 后，首先进入 `register` 包装的 [`tqbridge`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/utils/transferqueue_utils.py#L347)。桥接层发现参数是 `KVBatchMeta`，查询对应存储 metadata，再调用 TQ client 读取实际数据。

[`_async_meta_to_realdata`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/utils/transferqueue_utils.py#L159) 中决定输入内容的两步是：

```python
tensordict = await tq_client.async_get_data(meta)
# 将 meta.extra_info 中的运行参数附到 TensorDict 中。
```

因此 rank 0 方法体真正收到的是包含 64 条数据的 TensorDict，里面既有 batch 字段，也有非 batch 的配置值：

```text
样本字段，各有64行：
  prompts / responses / input_ids / position_ids
  response_mask / loss_mask
  old_log_probs / ref_log_prob / advantages / ...

整批共享信息：
  mini_batch_size=128
  global_batch_size=128
  epochs=1
  temperature=1.0
```

`input_ids` 包含完整 prompt+response；old probability 和 advantage 已是 response 对齐的 nested 字段。此时仍不能统一写成一个 `[64,L]` 矩阵，因为各字段长度不同，序列也长短不一。稍后每个 micro-batch 才移动到计算设备，并按模型需要准备 packed 输入。

桥接层解决的是“key 对应的数据怎样到达计算函数”。真正的 loss 不需要知道 TQ 连接、partition 或 Ray ObjectRef，只需要看到字段齐全的 TensorDict。

## `TrainingWorker` 是进程内对象，不是第二轮远程派发

现在 `ActorRolloutRefWorker.update_actor` 执行：

```python
output = self.actor.train_mini_batch(data=data)
```

`self.actor` 是当前 Ray actor 进程内创建的 `TrainingWorker`，再往下是 FSDP engine。这里是普通 Python 对象调用，不会再创建八个 Ray actor，也不会把本地 64 条重新当成全局数据分发一次。

调用层次如下，每个竖向链条只发生在一张卡所对应的训练进程内：

```text
controller：actor_rollout_wg.update_actor(512 keys)
                 │ dispatch，8个remote calls
                 ▼
rank 0 Ray actor                    rank 1 … rank 7 Ray actors
  tqbridge：64 keys → TensorDict      各自取64条
  ActorRolloutRefWorker.update_actor
    self.actor: TrainingWorker
      train_mini_batch
        train_batch
          FSDP engine
```

内层 `TrainingWorker.train_mini_batch` 也标了 `blocking=False`，因为这个类还可直接作为 WorkerGroup 使用。但本例是进程内调用，`blocking` 属性只在生成远程代理时生效，不会让普通 Python 函数自动变成 future。外层 `update_actor` 没有覆盖 `blocking`，使用默认 `True`，controller 会等这次完整调用结束。

这与 trainer 的 sync/async 模式是两个维度。一个远程方法可以非阻塞返回 future，而整个训练仍按“先生成、后优化、再同步权重”运行。不能看到 `blocking=False` 就推断 rollout 与 actor update 已经重叠。

## rank 0 怎样把 64 条变成四次更新

[`TrainingWorker.train_mini_batch`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine_workers.py#L241) 首先区分本地 batch 数量和全局 mini-batch 数量：

```python
batch_size_per_dp = data.shape[0]       # 本例为64
mini_batch_size = tu.pop(data, key="mini_batch_size", default=None)
mini_batch_size_per_gpu = mini_batch_size // self.engine.get_data_parallel_size()
```

所以 `mini_batch_size_per_gpu=128/8=16`。iterator 用本地 64 条构造四个大小为 16 的 mini-batch。本例 `actor.shuffle=false`，因此本地下标 `0…15` 进入第一个 mini-batch，`16…31` 进入第二个，依此类推。若另行开启 shuffle，iterator 使用加上 DP rank 的 seed，mini-batch 就不再是这些连续片段；一个 epoch 仍使用本地 64 条各一次。

每个本地 mini-batch 进入 engine 后，[`prepare_micro_batches`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine/utils.py#L92) 在静态分批模式下按每卡 4 条切开：

```text
rank 0 的64条
  mini 0：16条 → micro 0/1/2/3，各4条 → optimizer step 0
  mini 1：16条 → micro 0/1/2/3，各4条 → optimizer step 1
  mini 2：16条 → micro 0/1/2/3，各4条 → optimizer step 2
  mini 3：16条 → micro 0/1/2/3，各4条 → optimizer step 3

rank 1…7：同时执行相同层级的循环，各自使用不同样本
```

所有 rank 在同一更新序号下合计处理 `8×16=128` 条。每个 rank 总共执行 16 次训练 micro forward/backward，但只做四次 optimizer step。

现在落到 rank 0 第二个 mini-batch 的第一个 micro-batch：全批下标 `16…19` 是 05 中的 `P0_0_0、P17_2_0、P42_1_0、P85_3_0`，完整序列长度为 `[9,11,10,8]`。FSDP model engine 将其转成 `[1,38]` packed 输入，用第一个 mini-batch 更新后的 `θ1` 得到 current log-prob，抽出 response 形成 `[4,5]` loss 矩阵；old log-prob 仍固定在本轮开始的 `θ0`。Ray 此时不再参与 token 维度的切片和矩阵计算；它已经把任务交给正确的进程。

## micro-batch 之间何时同步，何时更新参数

[`BaseEngine.train_batch`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine/base.py#L113) 清楚地给出了 optimizer 的边界：

```python
self.optimizer_zero_grad()
outputs = self.forward_backward_batch(data, loss_function, forward_only=False)
grad_norm = self.optimizer_step()
```

`zero_grad` 在本地 mini-batch 开始时执行；四个 micro-batch 的 backward 都发生在中间一行；`optimizer_step` 在它们完成后执行一次。micro-batch 的主要作用是控制一次 forward/backward 的激活规模，同时通过梯度累计保留较大的全局更新 batch。

本例 FSDP 在完整八卡 world 上分片模型状态。forward 时需要取得相应层的参数，backward 时需要让各 rank 对不同样本算出的梯度合并，并归约到参数分片的所有者。相关 tensor collective 由 torch.distributed/FSDP 执行，不经过 controller 的 Python `ray.get` 收集梯度。

当前基线的 [`engine/fsdp.yaml`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/config/engine/fsdp.yaml#L61) 将 `use_no_sync_for_gradient_accumulation` 设为 `false`。它覆盖 Python 配置类中同名字段的默认值，因此 [`_gradient_sync_context`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine/fsdp/transformer_impl.py#L687) 在每个 micro-batch 都保留正常的梯度同步。梯度已经可以跨 rank 归约，但参数仍然等四次 backward 全部完成后才更新：

```text
zero_grad
  → micro 0 forward/backward，正常同步梯度并累计
  → micro 1 forward/backward，正常同步梯度并累计
  → micro 2 forward/backward，正常同步梯度并累计
  → micro 3 forward/backward，正常同步梯度并累计
  → 梯度裁剪 → 一次 optimizer step
```

若显式把 `actor_rollout_ref.actor.fsdp_config.use_no_sync_for_gradient_accumulation` 改为 `true`，FSDP1 才在前三个 micro-batch 使用 `module.no_sync()`，最后一个再同步累计梯度；FSDP2 有对应的 `set_requires_gradient_sync` 控制。这个变体减少梯度通信次数，但 FSDP1 的未同步完整梯度可能增加显存峰值。它改变通信安排，不改变一个 mini-batch 只更新一次参数的边界。

因此“micro-batch 变小”不意味着显存占用的每个组成部分都同比缩小：激活通常减少，参数、optimizer state、累计梯度和通信 buffer 仍受其他机制控制。

05 中的全局 loss 归一化在这里发挥作用：所有 micro-batch 共用本次全局 mini-batch 的有效 token 分母，并乘 DP size 抵消分布式梯度平均。八张卡最终共同实现一次全局 128 条样本的更新，而不是八个独立模型各自更新。

## 返回 controller 的是 metrics，参数留在 worker

四次更新完成后，`TrainingWorker.train_mini_batch` 汇总各次训练的 metrics，并返回一个不带样本 batch 维度的 TensorDict。外层 `update_actor` 再把返回值放到 CPU。

`tqbridge` 根据返回对象判断是否需要写回 TQ：有 batch 维度、且长度与输入 metadata 一致的结果，才按原 key 写回样本 fields。此次返回的是聚合 metrics，因此不会把“更新后的 actor 参数”写成 512 条 trajectory 的字段。

```text
留在 worker：actor 参数 θ4、optimizer state
留在 TQ：response、reward、old/ref log-prob、advantages 等本轮样本字段
回到 controller：loss、KL、grad_norm、学习率和性能等 metrics
```

collect 层会根据 `is_collect` 选择有效输出。FSDP SP=1 时八个 rank 都是有效输出 rank；若某种模型并行配置让多个 rank 共同处理相同 DP shard，就需要避免重复收集同一份样本结果。

controller 的 [`_update_actor`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L1734) 最后整理返回 metrics。参数并没有通过这个返回值交给 rollout；后面的 checkpoint/weight-sync 机制会单独把新 actor 权重送到生成引擎。

## 用同一条 key 比较 `compute_log_prob` 的返回

回看更新之前计算 old probability 的调用，可以更清楚地看出桥接层的作用。两条调用的前半段相同：

```text
512 keys → 按actor DP切成8份 → worker各取64条真实数据
```

但计算和输出不同：

| 调用               | worker 内部执行                                           | 结果怎样离开 worker                               |
| ------------------ | --------------------------------------------------------- | ------------------------------------------------- |
| `compute_log_prob` | `self.actor.infer_batch`，每次 4 条做无梯度 forward       | 返回有 64 行的 model output，桥接层按 key 写回 TQ |
| `update_actor`     | `self.actor.train_mini_batch`，本地 64→16→4，完成四次更新 | 返回聚合 metrics，模型状态保留在 worker           |

对 `P0_0_0`，old forward 先产生长度 9 的全序列 log-prob，桥接层按同一个 key 写回。controller 随后取出它，按 response 长度 3 从预测位置 `[5:8]` 提取 `old_log_probs`，再写回。这个“计算结果按原 key 回到 TQ”的过程，使下一阶段不必依赖上一个 worker 的内存地址或张量行号。

因此数据流不是“controller 一直抱着全部 GPU tensor 传来传去”。controller 持有 key 和阶段顺序，在需要全局算法处理时读取必要字段；worker 围绕 key 取出计算输入、写回样本结果。参数流则由模型 engine 和专门的权重同步机制维护。

## 同样八张 GPU，rollout 为什么只有四个副本

前面 rank 表中的 DP=8 属于 actor FSDP。rollout 使用另一套划分：`vLLM TP=2、DP=1、PP=1`，因此每个生成副本需要两张 GPU，八张卡承载四个副本。

一种与本例 colocated 配置相符的设备分组示意是：

```text
                 actor阶段                 rollout阶段
Node 0 slots 0,1：DP ranks 0,1      ←→      replica 0，TP ranks 0,1
Node 0 slots 2,3：DP ranks 2,3      ←→      replica 1，TP ranks 0,1
Node 1 slots 0,1：DP ranks 4,5      ←→      replica 2，TP ranks 0,1
Node 1 slots 2,3：DP ranks 6,7      ←→      replica 3，TP ranks 0,1
```

这里描述的是设备使用关系，不表示训练 worker 和 vLLM server 是同一个进程。同步模式会在生成、训练阶段之间协调 sleep、wake-up、模型状态和显存使用。

生成阶段的 TP=2 表示两张卡共同执行一个 server replica 的模型计算；训练阶段的 actor DP=8 表示八个 rank 各消费不同样本。不能将 `8/rollout.TP=4` 当成 actor DP，再错误地推导每卡 mini-batch=32。

P0 的 `rollout.n=4` 也没有规定四个 session 必须分别进入四个不同 server。那四个 `4` 分别可能表示节点内 GPU 数、每题采样次数、生成副本数或每个本地 mini-batch 的 micro-batch 数，含义由配置路径决定。

## 多个角色怎样共享进程

本例初始化时实际调用的是 [`create_colocated_worker_cls`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/single_controller/ray/base.py#L984)。它构造远程 `WorkerDict`，在进程内实例化角色对象，并给方法加上角色前缀。随后 `spawn(prefix_set=...)` 创建对应的 WorkerGroup 视图。

这意味着多个 controller 侧角色视图可以共享底层 Ray handles。对本案例，`ref_policy_wg` 与 `actor_rollout_wg` 指向同一角色 group，但分别调用 `compute_ref_log_prob` 和 `compute_log_prob`。内部 `self.ref` 与 `self.actor` 是不同 TrainingWorker/model 对象：reference 保持固定，actor 有训练 optimizer 并持续更新。

共享 Ray actor、共享 GPU allocation、共享模型参数是三件不同的事。本例具有前两者，不具有第三者。若启用 LoRA，reference 可以通过关闭 adapter 从 actor 路径得到；那是另一条配置分支。

仓库另外提供 `create_colocated_worker_cls_fused` 和 `_fuw_execute` 的路由机制，但本例的 V1 `_setup` 调用的是上述 `WorkerDict` 路径。理解当前执行链时，不能因为另一个类也支持 colocate 就把它插入这次调用。

## 换一种 dispatch 时，哪些部分会变

主线已经从 metadata 走到参数更新，可以用它比较其他 dispatch。下面的模式描述的是调用如何分发，不是训练算法。

| 分发方式                       | 输入的含义                                    | 与本例的差别                                                           |
| ------------------------------ | --------------------------------------------- | ---------------------------------------------------------------------- |
| `ONE_TO_ALL`                   | 一份参数广播给每个 worker                     | 适合初始化、load/offload 等操作，不按样本拆分                          |
| `ALL_TO_ALL`                   | 调用者已按 worker 准备输入                    | dispatch 原样通过，执行层按传入结构发给 worker                         |
| `DP_COMPUTE`                   | 每个参数已经是 world size 长度的列表          | 要求调用者先完成按 rank 的组织                                         |
| `DP_COMPUTE_PROTO`             | 可切分的 DataProto，直接按 world size 分块    | 支持相应 DataProto auto-padding；不等于所有 V1 metadata 都自动走此路径 |
| 按 mesh 的 dynamic DP dispatch | 先查询 global rank→DP rank，再按 DP size 分块 | 本例使用它，当前 mapping 恰好为 0…7                                    |

若训练 backend 改为一种 DP=2、每个 DP replica 有四个模型并行 rank 的配置，mapping 可能变为：

```text
global rank：0 1 2 3 4 5 6 7
DP rank：    0 0 0 0 1 1 1 1
```

同样 512 个 key，此时只能切成两份各 256 条；前四个 rank 共同接收第一份，后四个共同接收第二份。再由 backend 决定张量、序列和模型层怎样分工，collect 选择真正拥有输出的 rank。这个例子说明为什么 WorkerGroup 要向 engine 查询 DP mapping，而不能永远用 world size 除 batch。

`execute_mode` 则决定全部 worker 执行还是只在 rank zero 执行；`blocking` 决定代理在调用处等待还是交回 future。分发、执行对象、等待方式分别有自己的职责，一旦把它们映射回本例的“512 keys、八个进程、四次更新”，这些选项就不再只是抽象的枚举名。

下一节 [[projects/verl-source-reading/07-model-engine-and-backend|07 Model Engine and Backend]] 可以继续沿 rank 0 的那四条样本，深入 FSDP 参数分片、模型 forward 和 backend 接口；[[projects/verl-source-reading/08-hybrid-engine-and-weight-sync|08 Hybrid Engine and Weight Sync]] 则接住留在 worker 内的 `θ4`，解释它怎样成为下一轮生成所用的权重。
