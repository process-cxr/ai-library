---
title: 07 Model Engine and Backend
created: 2026-09-02
published: 2026-09-02
modified: 2026-11-03
type: project
status: growing
tags:
  - projects
  - source-reading
  - verl
  - fsdp
  - megatron
  - model-engine
---

06 已经把 512 个 trajectory key 分发到八个 actor DP rank。rank 0 取得其中 64 条，分成四个本地 mini-batch，每个 16 条。现在进入第二个 mini-batch：P0 的第一条回答 `P0_0_0` 位于它的第一行，准备与其他三条回答一起做 forward。

这一章继续沿着这四条数据深入：模型为什么能接收 `[1,38]` 的 packed 输入，又不会让四个样本互相看见；参数只有一部分常驻本卡时，完整的 Transformer 层怎样计算；四次 backward 怎样形成一次更新。主线仍是 [[projects/verl-source-reading/01-sync-grpo-run-overview|01 的两机八卡案例]]，采用 FSDP1、SP=1、静态 micro-batch=4、remove-padding，未启用 fused LM-head、LoRA 或 teacher。

源码固定在提交 `5b79827b04cee6e4b7b5ff737e047f1d72d50433`。文中的短序列、rank 排列和参数版本用于推演同一次训练，不是集群运行记录；PyTorch、Transformers 和 FlashAttention 内部的通信与 kernel 尚未通过 GPU trace 验证。

## 从 worker 交下来的 16 条数据开始

在 rank 0，第二个本地 mini-batch 对应本地下标 `16…31`。此时 actor 已完成第一轮 optimizer update，当前参数为 `θ1`；这些数据保存的 `old_log_probs` 仍来自 `θ0`。`TrainingWorker.train_batch` 把模型、静态分批和 remove-padding 选项附到数据中，再调用 engine：

```python
output = self.engine.train_batch(data, loss_function=self.loss_fn)
```

这里 `data` 有 16 行，`loss_fn` 是已经绑定 actor 配置的 PPO loss。engine 不重新生成回答，也不计算 GRPO 分组：它接收完整输入和固定 advantage，让当前模型产生概率并更新参数。

[`BaseEngine.train_batch`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine/base.py#L113) 把更新分成三个连续操作：

```python
self.optimizer_zero_grad()
outputs = self.forward_backward_batch(data, loss_function, forward_only=False)
grad_norm = self.optimizer_step()
```

`zero_grad()` 只在这个 mini-batch 开始时调用一次。中间的方法包含四次 micro-batch 的 forward/backward；最后一行才将 `θ1` 更新为 `θ2`。因此本次四个 micro-batch 都用 `θ1` 计算 current probability，不能把它们写成四个连续变化的 policy 版本。

## `strategy=fsdp` 怎样找到具体模型

这个 engine 已在 worker 初始化时创建。[`TrainingWorker.__init__`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine_workers.py#L146) 将模型类型和配置策略交给 registry：

```python
self.engine = EngineRegistry.new(
    model_type=self.config.model_type,
    backend=self.engine_config.strategy,
    model_config=self.model_config,
    engine_config=self.engine_config,
    optimizer_config=self.optimizer_config,
    checkpoint_config=self.checkpoint_config,
)
```

actor 的 `model_type="language_model"`，`strategy="fsdp"`。CUDA 路径匹配到注册的 [`FSDPEngineWithLMHead`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine/fsdp/transformer_impl.py#L1139)。它继承 FSDP 的初始化、通信和 optimizer 操作，并补上语言模型输入、词表概率与 loss 的处理。

registry 还会检查设备和 vendor，允许某种设备使用专门实现。`BaseEngine` 规定 `train_batch`、`infer_batch`、保存 checkpoint、导出参数等接口，但这些接口不保证不同 backend 使用相同通信过程。

reference 也可以使用带 LM head 的 engine，只是配置为 `forward_only`，不创建训练 optimizer。critic 则选择 `value_model`，模型末端输出标量 value；本例没有启用 critic，所以后续始终追踪词表 logits。

### 先构造 Hugging Face 模型，再包上 FSDP

[`_build_model_optimizer`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine/fsdp/transformer_impl.py#L583) 中，顺序是：

```text
HFModelConfig：读取 Qwen3-4B 的模型配置和 tokenizer
  → _build_module：from_pretrained + attention/forward 适配
  → _build_fsdp_module：按 wrap policy 包装、同步和分片参数
  → _build_optimizer：对包装后的参数建立 AdamW
  → _build_lr_scheduler：建立学习率调度器
```

[`_build_module`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine/fsdp/transformer_impl.py#L246) 根据 HF config 选择 AutoModel 类，再调用 `from_pretrained`。本例 `model_dtype=fp32`，决定模型构造时的参数精度。初始化还会根据权重共享情况选择 CPU 或 meta tensor 上下文，避免要求每个 rank 在初始化期间都独立保有完整的真实参数。

接下来调用 `apply_monkey_patch`，安装 remove-padding 所需的 attention 适配。开启 gradient checkpointing 时，还会执行 `gradient_checkpointing_enable(..., use_reentrant=False)`。后者减少需要一直保留的中间激活，以 backward 中的重计算换取显存；它不会改变这次训练包含哪四条回答。

[`get_fsdp_wrap_policy`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/utils/fsdp_utils.py#L80) 在本例默认 `min_num_params=0` 时读取模型的 `_no_split_modules`，寻找 Transformer layer 类决定包装边界。FSDP 因而可以按层级获取和释放参数。实际包装类名来自所安装 Transformers 的模型定义。

## DP=8 时，为什么每张卡没有独立的一整份模型状态

`fsdp_size=-1` 使 [`create_device_mesh`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine/fsdp/utils.py#L35) 创建大小为 8 的一维 `fsdp` mesh。`reshard_after_forward=true` 对应 `FULL_SHARD`。八个 rank 分别处理不同样本，同时在同一个 group 中分片模型状态。

将一个 FSDP 包装单元的扁平参数暂记为 `W`，忽略分片对齐的 padding，它在非计算期间可示意为：

```text
rank 0 持有 W 的第0片     rank 1 持有 W 的第1片
rank 2 持有 W 的第2片     rank 3 持有 W 的第3片
rank 4 持有 W 的第4片     rank 5 持有 W 的第5片
rank 6 持有 W 的第6片     rank 7 持有 W 的第7片
```

rank 0 处理 P0 时，不能只用 `W` 的第 0 片计算这一层。FSDP 在该单元计算前通过 all-gather 恢复所需完整参数；rank 0 用这些参数处理自己的 micro-batch，rank 1 用相同逻辑参数处理另一组样本。开启 FULL_SHARD 后，参数在相应 forward 之后重新分片，backward 所需参数再由 FSDP 获取。

因此 FSDP 的参数分片不等于把 Qwen3 的第 0 层固定放在 rank 0、第 1 层固定放在 rank 1。每个数据 rank 仍执行完整模型的层序列；按层切分模型执行是后文 PP 的做法。

backward 得到各 rank 对自己样本的梯度贡献。FSDP 将对应梯度归约，并把归约后的分片交给其所有者；optimizer 建立在分片参数上，维护相应 AdamW 状态。模型参数、梯度和 optimizer state 得以分摊，激活则主要由各 rank 的本地样本产生。

这里的“约八分之一”只适合粗略描述可分片状态的常驻规模。层级参数 all-gather、激活、通信 buffer、词表 logits、allocator 和分片 padding 都会增加峰值显存，不能用模型总参数量除以 8 直接预测训练是否 OOM。

## 把 16 条拆成四次可执行的 forward

真正计算前，FSDP engine 先对本地 16 条数据的 `loss_mask` 求和，再在 DP group 中 all-reduce，得到本次全局 128 条 trajectory 的有效 loss token 数 `N`。随后把 `N` 和 `dp_size=8` 附给各 micro-batch，供 05 的全局 loss 归一化使用。

[`prepare_micro_batches`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine/utils.py#L122) 的静态分支按 `micro_batch_size_per_gpu=4` 切分这 16 条，不按 token 长度重新分组。本例第一组仍是：

| 行  | trajectory key | prompt token 数 | response token 数 | 合计 |
| --- | -------------- | --------------: | ----------------: | ---: |
| 0   | `P0_0_0`       |               6 |                 3 |    9 |
| 1   | `P17_2_0`      |               6 |                 5 |   11 |
| 2   | `P42_1_0`      |               6 |                 4 |   10 |
| 3   | `P85_3_0`      |               6 |                 2 |    8 |

四条完整序列共有 38 个 token，只有 14 个 response token 参与本例 loss。`N` 则覆盖所有八个 rank 的整个 mini-batch，不能用 38 或 14 替代。其他 124 条 trajectory 的长度没有逐条指定，因此这里保留 `N`，不伪造一个全局 token 总数。

[`forward_step`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine/fsdp/transformer_impl.py#L1545) 将当前 micro-batch 移到计算设备，再调用 `prepare_model_inputs`。jagged 输入原本是：

```text
values：  [第0条9个token | 第1条11个token | 第2条10个token | 第3条8个token]
offsets： [0, 9, 20, 30, 38]
```

remove-padding 分支直接执行：

```python
input_ids_rmpad = input_ids.values().unsqueeze(0)
position_ids_rmpad = position_ids.values().unsqueeze(0)
```

得到 `input_ids: [1,38]` 和 `position_ids: [1,38]`。若按四行补到最长序列 11，会得到 `[4,11]`，其中 6 个位置只是 padding；当前路径避免这些位置参与模型计算。本例 SP=1、`pad_to_length=false`，没有额外切分或 bucket 补齐。

## 四条序列拼在一行，attention 怎样保持隔离

engine 传给模型的确是：

```python
model_inputs = {
    "input_ids": input_ids_rmpad,
    "attention_mask": None,
    "position_ids": position_ids_rmpad,
}
```

但这不是一段长度 38 的普通连续文本。每条样本的 `position_ids` 都从 0 开始：

```text
packed 下标： 0 …… 8 | 9 ………… 19 | 20 ……… 29 | 30 …… 37
position：   0 …… 8 | 0 ………… 10 |  0 ………  9 |  0 ……  7
样本：        P0      P17           P42          P85
```

[`HFModelConfig`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/config/model.py#L184) 默认选择 `flash_attention_2`。[`apply_monkey_patch`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/models/transformers/monkey_patch.py#L541) 在 remove-padding 启用时，将模型模块或 Transformers FlashAttention integration 的 `_flash_attention_forward` 替换成 verl 的 wrapper。wrapper 最后仍调用 Transformers 的 `_flash_attention_forward`，并把重置后的 `position_ids` 传下去。

本例 SP=1，wrapper 中的 Ulysses all-to-all 分支不执行。此时起作用的是 FlashAttention 对 packed 序列边界的识别。兼容的 Transformers 变长路径根据 position 的重置点得到各段起点，加上终点后，可将本例表示为：

```text
cu_seqlens_q = cu_seqlens_k = [0, 9, 20, 30, 38]
各段长度 = [9, 11, 10, 8]
max_seqlen_q = max_seqlen_k = 11
```

这些边界告诉 varlen attention：query/key 的 `[0:9]` 是第一条，`[9:20]` 是第二条，依此类推。每段内部再执行 causal attention。允许关系相当于下面的分块矩阵，其中 `C_n` 表示一个 `n×n` 的 causal 下三角块，`×` 表示整块禁止访问：

```text
            keys
            P0     P17     P42     P85
query P0    C_9     ×       ×       ×
      P17    ×     C_11     ×       ×
      P42    ×      ×      C_10     ×
      P85    ×      ×       ×      C_8
```

这张矩阵解释的是可见性，源码不需要先物化一个 `[38,38]` 的 dense mask。P17 的第一个 token 位于 packed 下标 9，却不能注意下标 `0…8` 的 P0；P0 的回答仍可注意它自己前面的六个 prompt token。

重置 position 本身不会让任意 attention 自动隔离样本。它必须被支持 packed 输入的实现解释成序列边界；仅把 RoPE 位置归零、再交给普通全局 causal mask，后面的样本仍可能看到前面的样本。本例依赖的正是上面的模型配置、patch 和变长 attention 调用组合。

固定 verl 源码可以确认 [wrapper 的调用与参数传递](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/models/transformers/monkey_patch.py#L135)；实际边界提取和 GPU kernel 在外部 Transformers/FlashAttention 中，必须与安装版本匹配。当前没有运行 packed-versus-unpacked 数值对照，不能把接口语义当作本机 kernel 已经验证正确的结果。

此外，engine 会检查模型 forward 是否显式声明 `cu_seqlens`，仅在支持时直接传入 offsets。不能仅因为 engine 计算了 `packed_cu_seqlens`，就断言所有 Qwen 模型都直接接收这一参数。本例普通 Qwen3 的主解释依赖 position-aware FlashAttention 路径，不套用 Qwen3.5 的专门 forward。

## packed 输入经过 Transformer，怎样回到回答概率

设 hidden size 为 `H`、词表大小为 `V`，当前 FSDP/SP=1 路径的主要逻辑形状为：

```text
input_ids           [1,38]
  → token embedding [1,38,H]
  → decoder layers  [1,38,H]
  → LM head logits  [1,38,V]
  → squeeze         [38,V]
  → 选取下一token的log-prob [38]
  → 按原offsets还原4条nested序列
  → 提取response对应预测位置，补齐成 [4,5]
```

FSDP 状态分片不会把本例隐藏维 `H` 自动切成 `H/8`。每个 rank 都为自己的四条数据执行这些逻辑形状；FSDP 管理模型状态的存储与通信。后文改用 TP 时，层内张量的局部形状才会随 tensor-parallel 划分改变。

模型调用设置 `use_cache=False`，因为这里一次输入已知的 prompt+response，计算所需条件概率；没有走 vLLM 那种逐 token 解码、维护生成 KV cache 的过程。

普通非 fused 路径的 [`prepare_model_outputs`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine/fsdp/transformer_impl.py#L1377) 取得 logits，并按 temperature 缩放。本例 temperature=1.0，缩放是恒等操作。随后 `logprobs_from_logits` 提取实际 sampled token 的概率，输出 38 个标量，不把完整词表 logits 交给 controller。

label 来自 `torch.roll(input_ids_rmpad, shifts=-1)`。对 `P0_0_0`，六个 prompt token 后接三个回答 token，所以 logit 位置 `5、6、7` 分别预测 `r0、r1、r2`。packed 位置 8 的 label 是下一样本首 token，但该位置被 response 切片排除：

```text
P0：  values[ 5: 8] → 3个回答概率
P17： values[14:19] → 5个回答概率
P42： values[25:29] → 4个回答概率
P85： values[35:37] → 2个回答概率
```

最后有 14 个有效 response log-prob，临时补成 `[4,5]`，与 old/ref probability、advantage、mask 逐位置运算。两个边界不能混淆：attention 边界阻止上下文串到另一条样本；response 切片排除跨样本 label 和其他不参与 PPO 的预测位置。后者不能修复前者发生的上下文污染。

[`ppo_loss`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/utils/losses.py#L59) 对这些概率执行 05 的 clipping、mask、token-mean 和 reference KL，得到可微标量。此时 P0 的学习信号已连接到 LM head、decoder、embedding 及其 FSDP 参数，接下来才有 `backward()` 可执行的计算图。

## fp32 参数、bf16 forward 与 loss scaling 分别是什么

02 的配置没有逐项覆盖 mixed precision。[`_build_fsdp_module`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine/fsdp/transformer_impl.py#L386) 在 `mixed_precision=None` 时实际选择：

| 设置                          | 本例值 | 影响的位置                        |
| ----------------------------- | ------ | --------------------------------- |
| `model_dtype`                 | fp32   | 模型初始参数与 FSDP 持久参数精度  |
| `MixedPrecision.param_dtype`  | bf16   | forward/backward 使用的低精度参数 |
| `MixedPrecision.reduce_dtype` | fp32   | 梯度归约精度                      |
| `MixedPrecision.buffer_dtype` | fp32   | 模型 buffer 精度                  |
| `torch.autocast` dtype        | bf16   | 按 PyTorch 算子规则选择计算精度   |
| `ShardedGradScaler`           | 不创建 | 本例直接对 loss 调用 backward     |

`forward_step` 的 autocast dtype 使用前面解析出的 `param_dtype`。因此“模型按 fp32 加载”和“forward 使用 bf16”同时成立。autocast 也不表示所有中间张量一律为 bf16；归一化、概率运算等仍由具体算子的规则与实现决定。

FSDP YAML 还有 `dtype: bfloat16`，但本路径创建 `MixedPrecision` 的直接依据是 `mixed_precision` 字段及上述 fallback。解释具体运行时，应沿实际消费者核对，不能只找到名称像 dtype 的配置项就结束。

只有把实际 `param_dtype` 配成 fp16，当前 engine 才创建 `ShardedGradScaler`，在 backward 前放大 loss，并在 gradient clipping 前反缩放。它用于减轻低精度数值下溢，与 PPO 的全局 token 分母不同。本例 bf16 不走 scaler 分支，但仍必须正确执行 `8/N` 的全局目标归一化。

## 四次 backward 如何累计，以及何时通信

[`forward_backward_batch`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine/fsdp/transformer_impl.py#L719) 为四个 micro-batch 依次执行 `forward_step`，再调用 `loss.backward()`。每次 loss 都已按同一全局 mini-batch 的有效 token 数 `N` 归一化：

$$
L_{r,j}=\frac{8}{N}\sum_{t\in\mathcal{T}_{r,j}}\ell_t.
$$

`r` 是八个 DP rank 之一，`j` 是本地四个 micro-batch 之一，`𝒯` 只包含 loss mask 有效位置，`ℓ` 包含 policy 项和本例 `0.001` 倍 reference KL 项。四次反传累加、DP 梯度平均后，对应：

$$
\frac{1}{8}\sum_{r=0}^{7}\sum_{j=0}^{3}\nabla L_{r,j}
=\nabla\left(\frac{1}{N}\sum_{t\in\mathcal{T}_{\mathrm{global}}}\ell_t\right).
$$

右侧覆盖本次全局 128 条 trajectory 的有效 token。因此不能再习惯性地将每个 micro loss 除以 4，那会重复缩小已经按全局 token 总数归一化的梯度。

本例的同步时机由实际组合后的 YAML 决定：[`engine/fsdp.yaml`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/config/engine/fsdp.yaml#L59) 明确设置 `use_no_sync_for_gradient_accumulation: false`。虽然 Python dataclass 和 `getattr` fallback 可取 `true`，它们没有覆盖这份已加载 YAML。因此 [`_gradient_sync_context`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine/fsdp/transformer_impl.py#L686) 在四个 micro-batch 中都不进入 `no_sync()`：

```text
zero_grad
micro 0：forward(θ1) → backward → 正常FSDP梯度同步，保留累计梯度
micro 1：forward(θ1) → backward → 正常FSDP梯度同步，继续累计
micro 2：forward(θ1) → backward → 正常FSDP梯度同步，继续累计
micro 3：forward(θ1) → backward → 正常FSDP梯度同步，继续累计
clip_grad_norm → optimizer.step → θ2
```

一次“同步”表示该 backward 正常触发 FSDP 所需的梯度 collectives，通常涉及多个包装单元和通信操作，不表示整张模型只发出一个 NCCL 调用。gradient accumulation 也不要求关闭同步：不清空累计梯度、延后 optimizer step，已经能实现累计。

若显式把选项改成 `true`，FSDP1 才在前三个 micro-batch 进入 `module.no_sync()`，最后一次同步累计梯度。这可以减少梯度通信，但此前要保留未分片的累计梯度，可能增加峰值显存。参数 all-gather 与梯度归约是不同操作，`no_sync()` 不表示这些计算完全没有跨卡通信。

本例训练阶段还会在 backward 后删除返回结果中的完整 model outputs，避免把多次 micro-batch 的全序列结果一直收集到 mini-batch 结束。这不等于能在 backward 前任意释放计算图仍需使用的张量。

## 一次 optimizer step 实际做了什么

四次 backward 完成后，engine 进入 [`optimizer_step`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine/fsdp/transformer_impl.py#L783)。本例没有 scaler，直接调用 FSDP 模型的 `clip_grad_norm_`，使用 optimizer 配置中的阈值 `1.0`。

这里必须计算分布式模型的梯度范数。仅在 rank 0 对其本地梯度分片求 norm，会漏掉其他 rank 持有的参数梯度；FSDP 的方法处理这种分片情形。返回 `grad_norm` 是裁剪前的范数，便于观察本次更新强度。

若范数有限，AdamW 用累计梯度更新本地参数分片及其状态；八个 rank 的更新合起来形成 `θ2`。若范数为 inf/NaN，当前 bf16 分支清空梯度并跳过 optimizer step。前文 `θ0→θ1→…→θ4` 因而隐含各次更新都成功的正常条件，不能拿 attempted step 数证明参数一定变化。

P0 所在 mini-batch 完成后，还有两次 mini-batch 更新。四次都结束，actor 为 `θ4`。学习率调度器却不是每个 micro 或每个 mini 都前进一步：[`train_mini_batch`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine_workers.py#L304) 只给最后一次迭代设置 `update_lr_scheduler=True`，`train_batch` 据此调用 scheduler。本例一次 `update_actor` 包含四次 optimizer step、一次 scheduler step。

输出给 controller 的仍是 loss、KL、grad norm 等 metrics。参数和 AdamW 状态留在 worker；怎样把它们重组并将生成需要的权重送给 vLLM，是 [[projects/verl-source-reading/08-hybrid-engine-and-weight-sync|08 Hybrid Engine and Weight Sync]] 的起点。

## 换成 Megatron，同一批数据会怎样执行

现在仍使用相同 512 个 key、全局 mini-batch=128、micro-batch=4 和八张 GPU，只把训练 backend 改为一个静态 Megatron 对照：TP=2、PP=2、CP=1、DP=2，无 virtual pipeline、无 dynamic CP，采用常规 dense 模型路径。这是另一套配置的源码推演，不是前面训练中途直接切换 backend。

这里 TP=2 属于训练 engine；vLLM rollout TP=2 属于另一个运行时，不能拿后者推算训练 DP。[`_init_device_mesh`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine/megatron/transformer_impl.py#L271) 使用训练配置建立 Megatron process groups，engine 再从 `mpu` 查询 DP size。

| 数量或操作                        | FSDP 主线                     | Megatron 对照                        |
| --------------------------------- | ----------------------------- | ------------------------------------ |
| 数据并行大小                      | 8                             | `8/(TP2×PP2×CP1)=2`                  |
| 每个 DP replica 收到的数据        | 64 条                         | 256 条                               |
| 每个 DP replica 的本地 mini-batch | 16 条                         | 64 条                                |
| 每个本地 mini 的 micro-batch 数   | `16/4=4`                      | `64/4=16`                            |
| 每轮 optimizer 更新次数           | 4                             | 4                                    |
| 处理同一 micro-batch 的 GPU       | SP=1 时一个数据 rank 执行各层 | 两个 PP stage，各有两个 TP rank 协作 |

一个 Megatron DP replica 的四个模型并行 rank 共同处理相同样本，不能把它们消费的行数重复相加。`2个DP replica×64条=128条` 才是一次全局更新的样本数。

若沿用 06 的完整 key 顺序重新按 DP 分发，两个 DP replica 分别得到 `K[0:256]`、`K[256:512]`。P0 四条回答的位置变成：

| key      | DP replica | replica 内下标 | 本地 mini 下标 |
| -------- | ---------: | -------------: | -------------: |
| `P0_0_0` |          0 |             16 |              0 |
| `P0_1_0` |          0 |            144 |              2 |
| `P0_2_0` |          1 |             16 |              0 |
| `P0_3_0` |          1 |            144 |              2 |

它们不再同时落到 FSDP 案例的第二个 mini-batch。保持全批数据和 batch 参数相同，只能保证数量关系，不能据此宣称两个 backend 每次更新用了完全相同的样本、得到相同 `θ1`。严格比较数值，还需固定各次全局 mini-batch 成员、随机性和精度等条件。

### 模型层由 PP 分工，层内运算由 TP 分工

Megatron engine 通过 bridge 将 HF 模型配置和权重转换成 MCore 模块，见 [`_build_tf_config`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine/megatron/transformer_impl.py#L283) 与 [`_build_megatron_module`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine/megatron/transformer_impl.py#L465)。它没有直接复用前面的 HF `FSDP(module)` 对象。

PP=2 将模型层分给两个 pipeline stage。P0 的 hidden states 经过前一个 stage 后，以激活形式送到下一个 stage；backward 时相应梯度反向传回。TP=2 在每个 stage 内分担线性层等运算，执行 tensor-parallel collectives。CP=1 表示本对照不沿 context 维分割序列。

因此“参数本地不足一整份”对应不同原因。FSDP 围绕层计算临时聚合参数；Megatron 的 PP rank 只执行自己拥有的层，TP rank 持有并计算层的一部分。LM head 也可沿词表分片，源码使用 [`vocab_parallel_log_probs_from_logits`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine/megatron/transformer_impl.py#L1226) 计算概率，不能把每个 TP rank 的局部词表当作完整词表独立做 softmax。

### backward 的调度和缩放也由 backend 负责

FSDP 中看到的是 Python 依次执行四次 forward/backward。Megatron 的 [`forward_backward_batch`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine/megatron/transformer_impl.py#L928) 从 Megatron Core 取得 pipeline schedule，将 16 个 micro-batch、模型 stage 和 forward callback 交给它安排。具体 warmup、steady-state、cooldown 以及通信重叠属于该外部 schedule 的实现。

传给 schedule 的 `seq_length=1`、`micro_batch_size=1` 不表示本例突然每次只训练一条、一个 token。源码旁注说明真实通信形状由 p2p communication 获取；数据 iterator 仍包含每组四条真实序列。

loss 归一化还有一处适配：[`postprocess_micro_batch_func`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine/megatron/transformer_impl.py#L1424) 先调用同类 PPO loss，再执行：

```python
scaled_loss = loss * data["num_micro_batch"]
```

在本对照的常规 schedule 归一化分支中，乘数为 16，用来抵消 pipeline schedule 内部按 micro-batch 数的除法。PPO loss 自身仍按全局有效 token 数和 DP=2 归一化。不能把这处乘数抄回 FSDP，也不能据此认定 Megatron 把学习率放大 16 倍。若启用 `calculate_per_token_loss`，后面还有按 token 计数处理的专门分支，需连同 schedule 重新推导。

Megatron 的梯度同步由 MCore 模块包装、训练 hooks 和 optimizer 协作完成；本对照保留 `use_distributed_optimizer=true`，optimizer 状态可沿 DP 进一步分片。FSDP 的 `use_no_sync_for_gradient_accumulation` 不控制这套过程。二者共享的仍是“完成一个全局 mini-batch 后更新一次参数”的边界。

## 保存训练状态与导出生成权重为什么分开

两种 engine 都提供保存 checkpoint 和导出参数的接口，但文件与张量布局不同。

FSDP 初始化 [`FSDPCheckpointManager`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine/fsdp/transformer_impl.py#L205)，处理模型分片、optimizer、scheduler 与恢复所需状态；Megatron 初始化 [`MegatronCheckpointManager`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine/megatron/transformer_impl.py#L627)，还需理解 TP/PP 与 distributed optimizer 布局。一个 backend 的训练 checkpoint 不能被当作另一个 backend 的相同目录直接读取。

rollout 接收权重需要生成模型能理解的参数名和张量。Megatron 的 [`get_per_tensor_param`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/engine/megatron/transformer_impl.py#L1024) 通过 bridge 导出 HF 权重；FSDP 也提供相应导出接口。这个过程通常不把 AdamW 动量、梯度或 scheduler 交给 vLLM，因为它们不参与生成。

本例关闭定期磁盘 checkpoint，但每轮更新后的 rollout 权重同步仍要发生。至此，P0 的回答已经通过 loss 和梯度改变 worker 中的 actor。08 接着解释如何让四个 vLLM replica 使用新的 `θ4`，并在共用八张 GPU 时协调训练状态与生成显存。

FSDP2、Ulysses SP>1、dynamic batching 等扩展也应从这条主线比较：FSDP2 更换参数分片与同步 API；SP 让同一条序列的计算跨卡；dynamic batching 改变 micro-batch 的成员和形状。它们不会使原来的 `512→64→16→4` 与 `[1,38]` 无条件保持成立。通用机制可结合 [[training/distributed-training/fsdp|FSDP]] 和 [[training/distributed-training/megatron|Megatron 与 3D 并行]] 阅读。
