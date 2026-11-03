---
title: 08 Hybrid Engine and Weight Sync
created: 2026-09-02
published: 2026-09-02
modified: 2026-11-03
type: project
status: growing
tags:
  - projects
  - source-reading
  - verl
  - hybrid-engine
  - weight-sync
---

07 中，P0 的回答已经参与了模型前向、backward 和 optimizer update。本轮 512 条 trajectory 共推动 actor 更新四次，参数从 $\theta_0$ 变成 $\theta_4$。但下一批问题由 vLLM 生成：它怎样获得这次学习的结果？

本章沿用两节点八卡的案例：actor 使用 FSDP1、`fsdp_size=-1`、SP=1、DP=8；vLLM 使用 TP=2、DP=1、PP=1，共四个生成副本，与训练共用八张 GPU。B=128、G=4、M=32、E=1、每卡 micro-batch=4，全部保持不变。主线是 V1 sync trainer 的 `naive` 权重同步，没有 LoRA、MTP、量化或 prefill/decode 分离。源码固定于 [5b79827](https://github.com/process-cxr/verl-upstream/tree/5b79827b04cee6e4b7b5ff737e047f1d72d50433)；下文解释静态调用及教学示例，尚未进行多机显存或通信实测。

## 四次 optimizer update 之后，谁拿到了新参数

先把本轮训练中的几种状态放在一起。$\theta_i$ 的下标只表示本例中的第几次 optimizer update，不是源码的 `global_steps`。

| 时刻                     | 当前 actor | rollout 使用的参数         | P0 的已有训练字段                                  |
| ------------------------ | ---------- | -------------------------- | -------------------------------------------------- |
| 首次生成前               | $\theta_0$ | 初始化同步后的 $\theta_0$  | 尚无回答                                           |
| 512 条 trajectory 收齐   | $\theta_0$ | $\theta_0$，随后进入 sleep | 回答、reward；接着补齐 old/ref log-prob、advantage |
| 第一次 mini-batch 更新后 | $\theta_1$ | rollout 正在 sleep         | 本轮已计算的字段保持固定                           |
| 第四次 mini-batch 更新后 | $\theta_4$ | 仍未安装 $\theta_4$        | 同上                                               |
| 本轮权重同步完成         | $\theta_4$ | $\theta_4$                 | 旧回答仍是由 $\theta_0$ 生成的样本                 |

actor 内部完成一次 `optimizer.step()`，只修改训练模型拥有的参数。vLLM 不会因为也放在这八张 GPU 上就自动看到这些修改。它使用自己的模型对象、TP 参数布局和推理缓存，必须显式接收新权重。

同样，权重同步不会重算 P0 的答案、reward、advantage 或 `old_log_probs`。这些字段描述的是上一轮采样及其训练目标。下一轮获得的新回答来自 $\theta_4$；即使偶然再次抽到同一道数学题，也是新的采样过程。

reference 的参数也不随这四次更新变化。本例 actor 和 reference 起点都是 Qwen3-4B，但 reference 作为 KL 对照保持固定；`update_weights` 导出的源头是 `self.actor.engine`，目的地是 rollout，没有把 reference 纳入更新。

## 同步发生在整批训练结束，而不是每个 micro-batch 后

[`PPOTrainerSync`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_sync.py#L31) 的三个 hook 几乎就定义了同步训练的节奏：

```python
def on_init_end(self):
    self.checkpoint_manager.update_weights(self.global_steps)

def on_sample_end(self):
    self.checkpoint_manager.sleep_replicas()

def on_step_end(self):
    # 省略计时上下文
    self.checkpoint_manager.update_weights(self.global_steps)
```

第一次调用发生在模型初始化和训练 checkpoint 加载之后；第二次让生成侧在本轮样本收齐后让出显存；第三次把本轮学习得到的新参数交还给生成侧。

沿着 [`fit`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L450) 往下看，trainer 等待 `step()` 完成，才调用 `on_step_end()`。因此本例中，八个 actor rank 各自做完四轮本地 mini-batch 更新后，才进行一次 rollout 权重同步。每个 micro-batch 的梯度通信、mini-batch 末尾的 optimizer update、这里的 rollout 权重同步，是三个不同层次的操作。

本轮的时间线可以写成：

```text
vLLM 用 θ0 生成 P0 等 128 道题的 512 条 trajectory
  → 收齐样本，等待生成请求排空，rollout sleep
  → old/ref log-prob、advantage
  → 4次mini-batch更新：θ0 → θ1 → θ2 → θ3 → θ4
  → 导出θ4，写入四个vLLM副本
  → 清理旧缓存，恢复KV cache空间
  → 开始下一批问题的生成
```

这解释了为什么同步模式中，P0 的四个 session 不会在四次 actor 更新之间穿插生成。生成阶段与训练阶段在这一主路径上分开，partial rollout 也不属于本章基线。

## 为什么名叫 Hybrid，仍然需要跨进程传权重

[`RolloutReplica.init_hybrid`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/rollout/replica.py#L131) 从训练 WorkerGroup 中选择一段 worker 句柄作为本副本对应的 GPU 位置。每个副本占两卡，所以八个训练 rank 对应：

| 节点   | 训练 global rank | 对应 vLLM 副本 | 副本内 TP rank |
| ------ | ---------------- | -------------- | -------------- |
| Node 0 | 0、1             | replica 0      | 0、1           |
| Node 0 | 2、3             | replica 1      | 0、1           |
| Node 1 | 4、5             | replica 2      | 0、1           |
| Node 1 | 6、7             | replica 3      | 0、1           |

这里沿用 06 的节点和 rank 排列。`ServerAdapter` 用 [`rank // rollout_world_size` 和取余](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/rollout/vllm_rollout/vllm_rollout.py#L81) 得到 replica rank 和 rollout rank；本例 `rollout_world_size=2`。rank 5 因而对应 replica 2、TP rank 1。

训练进程内有 FSDP 模型，也有 `ServerAdapter`。后者负责向 vLLM 发出恢复显存、装载权重等请求，并不是执行 token generation 的模型。vLLM server 按相同 GPU 位置启动，传给 vLLM 的 [`distributed_executor_backend="mp"`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/rollout/vllm_rollout/vllm_async_server.py#L323) 会使用其推理 worker 子进程。

只看 Node 1 上的训练 rank 5：

```text
同一张物理GPU
├─ 训练Ray actor进程
│   ├─ FSDP engine：持有本rank训练分片
│   └─ ServerAdapter：导出权重并发送
└─ vLLM worker进程
    └─ replica 2 / TP rank 1：持有推理参数切片
```

两个进程共享 GPU 位置，但 Python 对象、模型参数 storage、通信组并不自动共享。部分上层 docstring 把 hybrid 简称为“same process”；实际读到 server adapter 和 vLLM `mp` worker 这一层时，必须区分本地适配器与真正的推理模型。CUDA IPC 正是用来跨过这个同卡进程边界的。

训练分片也不能直接等同于推理分片。actor 的八个 FSDP rank 为训练分摊状态，vLLM 的每两个 TP rank 为一次前向划分算子；两种分片的目的和布局均不同。

## manager 在这条路径上选择了哪个分支

初始化时，trainer 把训练 WorkerGroup 和四个 rollout replica 交给 `CheckpointEngineManager`。这里有一个容易忽略的实际赋值：

```python
checkpoint_engine_config = omega_conf_to_dataclass(
    self.config.actor_rollout_ref.rollout.checkpoint_engine
)
checkpoint_engine_config.backend = "naive"
```

它位于 [`PPOTrainer` 的初始化路径](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L357)。因此这里不仅是“YAML 恰好默认 naive”，manager 还明确选定了 hybrid 本地同步分支。

[`CheckpointEngineManager.update_weights`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/checkpoint_engine/base.py#L505) 进入该分支后执行：

```python
if self.backend == "naive":
    ray.get(self.actor_wg.update_weights(
        global_steps=global_steps, mode=self.backend
    ))
    return {}
```

`update_weights` 注册为 `ONE_TO_ALL`：八个 actor worker 全部参与，controller 等待这一组远程调用完成。worker 收到显式的 `mode="naive"` 后，以它决定分支，不会再用 `mode="auto"` 的配置回退。

因此当前调用并不经过后面的 `build_process_group → send_weights → receive_weights` 远端传输流程。训练侧 FSDP 恢复完整逻辑参数时仍可能跨节点通信；“naive”并不表示整个同步过程无网络通信，只是没有另建 actor 到独立 rollout 集群的 checkpoint transport group。

## 先恢复推理参数空间，再从 FSDP 导出

进入 [`ActorRolloutRefWorker.update_weights`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine_workers.py#L767) 后，主干如下，省略 LoRA 分支及显存日志：

```python
await self.rollout.resume(tags=["weights"])

per_tensor_param, peft_config = self.actor.engine.get_per_tensor_param(
    layered_summon=self.layered_summon, base_sync_done=True
)
await self.rollout.update_weights(
    per_tensor_param, peft_config=peft_config,
    base_sync_done=True, global_steps=global_steps,
)

# 只有启用actor参数offload时才移动训练模型到CPU。
if self.actor.engine.is_param_offload_enabled:
    self.actor.engine.to("cpu", model=True, optimizer=False, grad=False)

await self.rollout.resume(tags=["kv_cache"])
```

两次 `resume` 在原函数中都受 `free_cache_engine` 控制，本例沿用其默认 `true`。先只恢复 weights，为新参数找到接收空间；等装载完成和临时内存清理后，再恢复 KV cache。若先把完整推理缓存恢复，参数导出和传输的峰值显存会与缓存占用叠加。

这段代码搬运的不是 P0 的 `input_ids` 或 `[512,R]` advantage，而是模型的命名参数。P0 的学习信号已经通过梯度体现在 $\theta_4$ 中，接下来流动的是这些更新后的数值。

## 从训练分片恢复命名参数，而不是发送 optimizer

[`get_per_tensor_param`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine/fsdp/transformer_impl.py#L973) 对普通全参数模型先取 `self.module.state_dict()`，经过 `convert_weight_keys` 还原模型 checkpoint 命名，再构造迭代器。每次迭代的核心为：

```python
(
    name,
    param.to(device, non_blocking=True).full_tensor()
    if isinstance(param, DTensor) else param,
)
```

多卡 FSDP1 在构造模型时设置了 [`SHARDED_STATE_DICT`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine/fsdp/transformer_impl.py#L482)，并使用 device mesh。导出时面对的是有原参数名称和逻辑形状的状态条目；对 DTensor 条目，`full_tensor()` 汇集分布式分片，产生当前进程可用的完整逻辑 tensor。

比如某个 projection 的权重不再是“rank 5 手上的一段 flat parameter”，而是：

```text
("model.layers.0.self_attn.q_proj.weight", W)
W.shape = 该层完整q_proj权重的逻辑形状
```

这是交给推理模型 loader 的接口。八个训练 rank 都参与相应导出；本地接收端再按照自己的 TP rank 安装所需部分。不能只调用一个训练 rank 的导出函数，却指望跨八卡的 collective 自行完成。

导出的迭代器允许消费者逐项取参数，但不意味着所有 backend 都能“从头到尾只占一个参数的临时空间”。本例 FSDP1 会先执行 `state_dict()`；其内部 unshard、state 条目和通信 buffer 的峰值还要结合 PyTorch 实现判断。FSDP2 的 DTensor staging 路径与这里也有差别。

`state_dict()` 来自模型模块，没有把 Adam 的一阶矩、二阶矩或学习率调度状态加入这次传输。vLLM 只需用模型参数生成 token，不执行 optimizer update。reference 模型也不在这个导出入口中。

## 用一个小矩阵区分 FSDP 分片与 TP 分片

下面用 $W\in\mathbb{R}^{6\times4}$ 说明布局转换，它不是 Qwen3-4B 的实际层大小。暂时假设训练侧一个 flat buffer 只包含这个矩阵的 24 个元素，八个 FSDP rank 平分，每个拿 3 个：

```text
W = [w00 w01 w02 w03
     w10 w11 w12 w13
     w20 w21 w22 w23
     w30 w31 w32 w33
     w40 w41 w42 w43
     w50 w51 w52 w53]

训练flat buffer的局部片段：
rank 0: [w00 w01 w02]
rank 1: [w03 w10 w11]
rank 2: [w12 w13 w20]
...
rank 7: [w51 w52 w53]
```

rank 1 的三个元素甚至跨越了矩阵的行边界，无法直接当成一个 TP linear 的完整子矩阵。真实 FSDP flat buffer 往往还组合多个参数，并可能包含对齐 padding；此处只是用一个矩阵说明为什么不能按 rank 直接照搬。

导出恢复名字和完整形状之后，假设推理模型的这个 linear 按输出通道划分，TP=2：

```text
命名tensor W [6,4]
   ├─ vLLM TP rank 0 ← W[0:3,:]，形状[3,4]
   └─ vLLM TP rank 1 ← W[3:6,:]，形状[3,4]
```

四个 vLLM replica 都完成这一装载，各副本的两个 TP rank 合起来表示同一个 $W$。这样 replica 0 和 replica 2 虽然服务不同请求，使用的模型版本仍然相同。

“按输出通道切行”只适用于这个示意 linear。实际模型的 row-parallel linear、QKV 合并权重、词表切分和复制参数有各自的规则，不能把所有参数统一写成 `W.chunk(2, dim=0)`。verl 把命名 tensor 交给 vLLM 的模型 loader，具体 TP 切分和融合由所用 vLLM 模型实现决定。

## 命名 tensor 怎样穿过同卡进程边界

[`ServerAdapter.update_weights`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/rollout/vllm_rollout/vllm_rollout.py#L209) 先通过 server 的 collective RPC，让该副本的 vLLM workers 进入 `update_weights_from_ipc` 接收流程；然后每个训练 rank 的 sender 发送本地导出的参数。

本例每个副本的 TP rank 0 对应的 adapter 负责发起 server RPC，其余 rank 仍执行本地 sender。发送端和接收端用 job、replica、local rank 对齐 IPC endpoint。所以 rank 5 导出的数据交给同卡的 replica 2 / TP rank 1，而不是送到所有生成副本共同使用的一个 Python 对象里。

[`BucketedWeightSender`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py#L103) 按字节打包多个命名 tensor。对上面的 $W$，假设导出 dtype 为 FP32：

```text
W [6,4] × 每个元素4 byte = 96 byte

元信息：name、shape=(6,4)、dtype=float32、offset
数据：  byte buffer[offset:offset+96]
接收：  这一段byte → view(float32) → view(6,4)
```

本路径的 bucket 大小来自 [`checkpoint_engine.update_weights_bucket_megabytes`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/config/rollout/rollout.yaml#L276)，YAML 默认 2048，代码用 `<< 20` 换算字节。它影响传输缓冲区大小，与训练 mini-batch=32、micro-batch=4 没有数量换算关系。

普通 tensor 复制到共享的 GPU byte buffer；ZMQ 发送 buffer handle 和每项元信息，接收进程借 CUDA IPC 重建可访问的 tensor view。接收端完成当前 bucket 的模型装载并同步设备后回 ACK，发送端才复用这块 buffer。超过 bucket 大小的单个 tensor，在 CUDA IPC 路径可以直接传 handle，并不是一律切成任意小片。

因此 CUDA IPC 节省的是跨进程传递 tensor 的额外中转，并不等于整个更新过程零复制：sender 填充 bucket、loader 把数据安装到模型参数都可能复制。代码还有不支持 IPC 时的 CPU shared-memory 回退，那条路径的传输成本不同。

sender 刻意没有强制把所有权重转换为 BF16。训练参数、特定 buffer 与推理参数的 dtype 可能不同，接收 loader 按目标模型处理。不能只根据 rollout 使用 BF16，就断言 wire 上每个元素一定占 2 byte。

## 接收之后，谁真正改写 vLLM 模型

vLLM 启动参数指定了 verl 的 `vLLMColocateWorkerExtension`。其 [`update_weights_from_ipc`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/rollout/vllm_rollout/utils.py#L241) 创建 receiver，并对收到的每个 bucket 调用 `_update_weights`。

本例没有 LoRA 或量化，因而走普通参数分支。函数先区分参数和 buffer；参数最终进入：

```python
model.load_weights(param_updates)
```

这行位于 [`_update_weights`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/rollout/vllm_rollout/utils.py#L379)。上面的 `[6,4] → [3,4]` 切分示例描述的正是模型 loader 应承担的工作，传输层本身只理解 name、shape、dtype 等信息。

到这里，rank 5 的训练分片已经经历“恢复命名参数 → IPC 传递 → vLLM 模型装载”，成为 replica 2 的推理参数的一部分。等所有参数和四个副本都完成这条流程，下一批请求才能使用完整的 $\theta_4$。

本仓库可以直接确认调用到了 vLLM 的 `load_weights`；Qwen 各层在某个安装版本中的精确 slicing、fused layout 和 dtype cast，要继续进入该版本的 vLLM 模型源码核对。这是实现边界，不能把一个教学矩阵当成已经验证的 Qwen 参数布局。

## 为什么更新参数之后还必须处理 KV cache

P0 的 prompt 在 $\theta_0$ 下经过各层投影，产生了相应的 key/value。即使下一轮出现相同 token 前缀，换成 $\theta_4$ 后这些中间结果也未必相同。继续复用旧 cache，会把新参数和旧隐藏状态混在一次前向中。

普通同步路径因此在 `ServerAdapter.update_weights` 等待接收端完成后，调用 `clear_kv_cache`。server 的 [`clear_kv_cache`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/rollout/vllm_rollout/vllm_async_server.py#L865) 清理 prefix cache，并传 `reset_connector=True`，还清理多模态和 encoder cache。本例是纯文本，没有外部 KV connector，但共用接口仍执行这套清理。

sleep/wake 处理显存资源，cache reset 处理缓存内容的有效性，两者有关联却不是同一操作。本例 [`_sleep_hybrid`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/rollout/vllm_rollout/vllm_async_server.py#L1248) 选择 level 2，让 rollout 参数与 KV cache 的设备空间让位；恢复时先为 weights 分配空间、装入新值，再恢复 KV cache 空间。

vLLM replica 的 [`sleep`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/rollout/vllm_rollout/vllm_async_server.py#L1377) 先等待请求 drain，再调用 server sleep。本例因而不是把 P0 的生成突然中断在某个 token，随后用新参数续写；那是后续异步及 partial-rollout 分支要讨论的行为。

CUDA graph 是执行计划和设备资源管理的一部分，不能与“某段 prompt 的 KV 值”混为一谈。这里没有在每次更新时重新创建整台 vLLM server，也没有显式重新捕获全部 graph；graph 与 sleep 分配器的兼容性由 vLLM 实现管理，不能仅凭一次 `clear_kv_cache` 推导其所有显存都已释放。

## sleep 不等于八张卡上所有训练状态都消失

本例 actor 的 [`param_offload=false`、`optimizer_offload=false`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/config/engine/fsdp.yaml#L10)。所以 rollout sleep 后，actor 的分片状态仍由训练进程保有；rollout wake 后，也不能默认 actor 参数和 Adam 状态已经全部移到 CPU。

| 状态                         | 本例中由谁保有          | 在线权重同步时怎样处理                                  |
| ---------------------------- | ----------------------- | ------------------------------------------------------- |
| actor 参数分片               | 八个训练 rank           | 导出命名参数供 rollout 装载，默认不做阶段性参数 offload |
| actor optimizer 状态         | 训练 engine/optimizer   | 保留用于后续更新，不发送给 vLLM                         |
| reference 参数               | 独立的 reference engine | 保持固定，按其 forward-only/offload 路径管理            |
| rollout 参数                 | 四个副本中的 TP workers | sleep 让位后，wake 并装入 actor 新权重                  |
| rollout KV/prefix cache      | vLLM runtime            | 旧内容失效，恢复空间供新请求使用                        |
| IPC bucket 与导出临时 tensor | 当前发送/接收过程       | 随传输消费及清理释放，不是长期训练状态                  |

若另外启用 actor `param_offload`，`update_weights` 中的条件分支才会把训练模型移到 CPU，并且该调用明确写着 `optimizer=False, grad=False`。optimizer offload 是单独的配置和生命周期，不能从模型 offload 自动推导。

这套时序减少了 rollout 与训练高峰的重叠，但不承诺任意八张 GPU 都能容纳 Qwen3-4B 的本例配置。实际峰值还包括常驻 shards、导出临时状态、IPC bucket、vLLM 参数和 cache。优化之前需要知道哪个阶段、哪类内存占用最高，不能仅用“总参数量除以八”估算。

## `global_steps=1` 为什么对应这里的 $\theta_4$

[`_load_checkpoint`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L793) 先将 `global_steps` 设为 0。本例 `resume_mode=disable`，因此初始化同步带版本标签 0。进入 `fit` 后，trainer 在第一次训练前把 `global_steps` 加到 1。

本例的两种计数由此对应：

```text
初始化：actor θ0 → rollout θ0，set_global_steps(0)
step 1：用版本0生成；actor更新4次得到θ4
step 1末：actor θ4 → rollout θ4，set_global_steps(1)
step 2：用版本1生成；actor再更新4次得到θ8
step 2末：actor θ8 → rollout θ8，set_global_steps(2)
```

`global_steps` 标记 trainer 的训练轮次，不是每次 `optimizer.step()` 递增的计数。改变 B/M、`ppo_epochs` 或其他更新组织方式后，一次同步之间的 optimizer 次数会改变，不能把 $\theta_{4k}$ 当成普遍版本规则。

在 [`ServerAdapter.update_weights` 的末尾](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/rollout/vllm_rollout/vllm_rollout.py#L243)，adapter 等待接收完成、清理 cache，再把标签传给 server 的 `set_global_steps`。生成结果通过 `extra_fields["global_steps"]` 带回这个版本，之后成为 trajectory 的 `min_global_steps/max_global_steps`。

因此 P0 的首批 trajectory 可以同时具有“所属训练 step 为 1”和“生成权重版本为 0”。前者表示问题在哪一轮从 dataloader 提交，后者表示生成时用了哪次同步的参数；二者相差一并不自动意味着异步滞后。

## 首次同步也解释了 `load_format=dummy` 的用途

02 的 rollout 配置沿用 `load_format: dummy`。vLLM 据模型结构创建推理 runtime，但初始占位权重不能用于有意义的采样。actor 从 Qwen3-4B 加载训练参数，若启用恢复还会先恢复 checkpoint，然后 `on_init_end()` 把这些参数同步给 vLLM，首轮生成才使用 $\theta_0$。

这一前提不适用于任意另建的推理服务。例如 [[projects/verl-source-reading/12-on-policy-distillation-and-mopd|MOPD 中的固定教师]] 没有“从学生 actor 接收教师预训练权重”这条路径，教师需要正确加载自己的模型权重，不能照搬学生 rollout 的 dummy 初始化。

## 在线同步和落盘 checkpoint 不是同一件事

`CheckpointEngineManager` 的名字容易让人以为每轮先保存文件、再让 vLLM 读取。当前链路实际上从模型内存导出参数，通过 IPC 装载，没有要求中间写一个 Hugging Face checkpoint。

训练恢复使用的落盘 checkpoint 由另一条 [`_save_checkpoint`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L889) 调用路径触发，保存内容由训练 checkpoint 配置决定，可以包含模型、optimizer 等恢复状态。本例 `save_freq=-1` 关闭定期保存，`on_step_end()` 的在线权重同步仍每轮执行。

两者回答不同问题：在线同步让下一次生成使用刚训练出的模型；磁盘 checkpoint 让训练在进程退出后有机会恢复。同步成功不代表已经持久保存，保存成功也不代替向当前运行的 rollout 安装新权重。

同步等待同样不等于数据库事务。接收端按 bucket 更新参数；若中途某个 worker 出错，可能已经有部分 tensor 或副本完成装载。当前 naive 分支没有显示实现全副本原子提交、失败自动回滚或整套参数校验。正常路径通过等待完成保证时序；异常时应停止推进并按失败原因恢复，不能只看一个版本标签就断言所有数值一致。

## 改成独立 rollout 集群后，哪些地方会变化

本章先闭合了“同八卡、sync、naive”这一条路径。切换到 [[projects/verl-source-reading/10-async-trainer-and-off-policy|异步或训练推理分离]] 后，manager 的非 naive 分支才引入独立 transport：暂停/中断生成、管理 cache 空间、建立 actor 与 rollout 的通信拓扑、并行发送接收、finalize，最后恢复生成。具体入口见 [`CheckpointEngineManager.update_weights` 的后半段](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/checkpoint_engine/base.py#L517)。

其中 NCCL、NIXL 等 backend 解决跨进程组传输；本快照还有 [`delta_sharded`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/checkpoint_engine/delta_checkpoint_engine.py#L569)：首次发送完整 seed，保存比较基准，后续生成带位置和值的变化数据，由接收端应用到现有权重。它有额外的快照、编码和 loader 语义，并非本例每轮天然只发送“梯度差值”。

现在可以沿 P0 走完整个闭环：$\theta_0$ 生成回答，reward 和 advantage 推动 actor 变为 $\theta_4$，FSDP 导出命名参数，四个 vLLM 副本按自己的 TP 布局装载，清理与旧参数相关的缓存，下一批问题才真正受益于刚完成的学习。[[projects/verl-source-reading/09-ppo-and-grpo-codepath|09]] 回到这条闭环中的算法计算，比较 PPO 与 GRPO 对训练字段和角色的不同需求。
