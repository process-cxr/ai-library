---
title: 12 On-Policy Distillation and MOPD
created: 2026-10-09
published: 2026-10-09
modified: 2026-11-03
type: project
status: growing
tags:
  - projects
  - source-reading
  - verl
  - on-policy-distillation
  - mopd
  - multi-teacher
---

P0 的四条回答经过规则评分后得到了 `[1,0,1,0]`。在 [[projects/verl-source-reading/05-advantage-and-policy-loss|05 Advantage and Policy Loss]] 中，这四个分数通过 GRPO 变成每条回答的相对优势。同一条回答中的有效 token 共享一个组内标准化后的分数，规则没有进一步说明：某个中间步骤已经偏离正确解法，或者另一个 token 虽然出现在错误答案中，却仍是合理的选择。

本章沿着这四条回答增加一种监督来源：让 teacher 阅读 student 已经生成的序列，在每个前缀处提供 token 概率。这样既能继续沿用 student 访问过的状态，也能获得比最终对错更细的训练信号。多个数据来源分别选择不同 teacher，就是这里的 Multi-Teacher On-Policy Distillation（MOPD）。

以下调用和字段均对应固定源码 [`5b79827b04cee6e4b7b5ff737e047f1d72d50433`](https://github.com/process-cxr/verl-upstream/tree/5b79827b04cee6e4b7b5ff737e047f1d72d50433)。数字用于手算和追踪，没有执行完整的多 GPU 蒸馏训练。

## P0 的回答保留下来，监督信号增加一条来源

沿用 [[projects/verl-source-reading/01-sync-grpo-run-overview|01 Sync GRPO Run Overview]] 的问题：

> 盒中有 3 袋铅笔，每袋 4 支，一共多少支？请把最终答案写在 #### 后面。

它的 `data_source` 为 `openai/gsm8k`，标准答案为 `12`。四条 student response 分别为：

| Trajectory | Student response          | 规则奖励 |
| ---------- | ------------------------- | -------: |
| `P0_0_0`   | `3 × 4 = 12。#### 12`     |        1 |
| `P0_1_0`   | `3 + 4 = 7。#### 7`       |        0 |
| `P0_2_0`   | `4 + 4 + 4 = 12。#### 12` |        1 |
| `P0_3_0`   | `4 × 4 = 16。#### 16`     |        0 |

这里的 `P0` 是实际 UUID 的简写。后面的图仍将 prompt 长度缩为 6、四条 response 长度缩为 `[3,5,4,2]`；这些是符号序列的长度，不是上面中文字符串的 tokenizer 结果。

以第二条错误回答为例，teacher 接收到的是这条回答实际访问过的前缀。它可以对“已经写出 `3 +` 以后，下一个 token 为 `4`”的概率评分。teacher 没有先把错误前缀改成 `3 ×`，也没有用自己的完整解答替换这条 trajectory。OPD 中的 on-policy 首先体现在这些状态由 student rollout 产生；在同一批样本上进行多次更新时，仍需要区分采样时的 student 和正在更新的 student。

本章的 student 继续使用 2 节点 × 4 GPU、FSDP DP=8、SP=1、`B=128`、`G=4`、`M=32`、一个 PPO epoch、每个 micro-batch 4 条 trajectory。因而一轮仍有 512 条 trajectory，分成四次全局 mini-batch 更新。teacher 不改变这些数量关系。

为了先保留第五章已经讲清楚的任务学习，本章主要使用 `loss_mode=k1`、`use_policy_gradient=true`、`use_task_rewards=true`，蒸馏系数为 1。这样原有规则奖励和 reference KL loss 仍保留，再加入 teacher 信号。仓库 MOPD 示例默认关闭 task rewards；后文会解释这一配置差别怎样改变最终目标。

## Teacher 从哪里来，占用哪一组 GPU

前六章中的 8 张卡属于 student actor / rollout 资源池。teacher 使用单独的资源池：[`PPOTrainer`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L340) 为它构造 `MultiTeacherModelManager`，再把客户端交给 AgentLoop。teacher 是推理服务，不参与 student 的梯度同步，也没有 student optimizer。

为解释多教师分配，可以在基线之外再增加一个 4 GPU 节点，形成下面的资源示意。这里仅说明分配算术，没有确认具体型号 GPU 的显存是否足以容纳 32B 模型。

```text
原有 student pool：Node 0、Node 1，各 4 GPU
  actor FSDP DP=8；rollout TP=2，共 4 个 rollout replica

额外 teacher pool：Node 2，4 GPU
  GPU 0–1：GSM8K teacher，1 replica，TP=2
  GPU 2–3：Geo3K teacher，1 replica，TP=2

全部资源：8 student GPU + 4 teacher GPU = 12 GPU
```

teacher 的一个 replica 占用 `TP × inference DP × PP` 张卡。若两位教师的 inference DP、PP 都为 1，则 teacher pool 的容量为：

$$
1\times(2\times1\times1)+1\times(2\times1\times1)=4.
$$

[`DistillationConfig.__post_init__`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/config/distillation.py#L268) 要求所有 teacher 的需求之和**等于** `distillation.nnodes × distillation.n_gpus_per_node`。随后 manager 按教师、再按 replica 切分资源。跨节点时还会检查 replica 子池的节点跨度，不能只保证总卡数相等。

这也解释了 teacher replica 和 teacher 数量的区别：增加同一 teacher 的 replica 是增加同一种模型的服务容量；增加 teacher 条目则引入另一套模型和路由目的地。

## `openai/gsm8k` 怎样选到正确的 teacher

下面的 YAML 是叠加在第二章 student 基线和默认 `distillation.yaml` 上的配置片段，不是独立配置或完整启动命令。两个模型名沿用仓库示例；本文实际追踪的 P0 是纯文本样本，只会访问 GSM8K teacher。片段显式设置本章使用的 `k1`、policy-gradient 蒸馏和 task rewards；默认 YAML 的 `loss_mode=k3`、`use_policy_gradient=false` 不会自动变成这套组合。

```yaml
distillation:
  enabled: true
  teacher_key: data_source
  nnodes: 1
  n_gpus_per_node: 4
  distillation_loss:
    loss_mode: k1
    use_policy_gradient: true
    use_task_rewards: true
    distillation_loss_coef: 1.0
  teacher_models:
    gsm8k:
      key: openai/gsm8k
      model_path: Qwen/Qwen3-32B
      num_replicas: 1
      inference:
        name: vllm
        load_format: auto
        tensor_model_parallel_size: 2
        data_parallel_size: 1
        pipeline_model_parallel_size: 1
        prompt_length: ${actor_rollout_ref.rollout.prompt_length}
        response_length: ${actor_rollout_ref.rollout.response_length}
        max_model_len: 2049
    geo3k:
      key: hiyouga/geometry3k
      model_path: Qwen/Qwen3-VL-32B-Instruct
      num_replicas: 1
      inference:
        name: vllm
        load_format: auto
        tensor_model_parallel_size: 2
        data_parallel_size: 1
        pipeline_model_parallel_size: 1
        prompt_length: ${actor_rollout_ref.rollout.prompt_length}
        response_length: ${actor_rollout_ref.rollout.response_length}
        max_model_len: 2049
```

这段配置有一个必须沿源码理解的前提。[默认 YAML](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/config/distillation/distillation.yaml#L83) 已经提供 `teacher_models.teacher_model`，Hydra 合并后暂时保留它，同时新增 `gsm8k`、`geo3k` 两项。[`_resolve_teacher_models`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/config/distillation.py#L289) 先断言默认项存在，再在发现命名教师时删除它。因此不能把上面的片段单独交给 `DistillationConfig`，也不能手动提前删掉默认项。第一位实际教师必须使用 `gsm8k` 这样的新名字；若把它的配置留在 `teacher_model` 下，再添加第二位教师，第一位会被一并删除。

删除默认项也不会把它的字段复制给新教师。新条目通过 [`omega_conf_to_dataclass`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/utils/config.py#L61) 与 Python dataclass 默认值合并。对于未填写的 `inference.load_format`，[`RolloutConfig`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/config/rollout.py#L249) 给出的是 `dummy`，默认单教师 YAML 中的 `auto` 不会继承过来。

这直接决定 teacher 的概率是否来自指定 checkpoint。设置 `model_path` 只给出了模型位置；[vLLM server](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/rollout/vllm_rollout/vllm_async_server.py#L323) 仍按 `load_format` 选择加载方式。`dummy` 初始化不会加载预训练权重，而 [teacher manager](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/experimental/teacher_loop/teacher_model.py#L62) 启动推理服务后，没有像 student rollout 那样再从训练 actor 同步权重。因此示例给每位教师显式设置 `load_format: auto`。

长度也需要显式传递。两个插值沿用 student 的 prompt、response 上限，在本例中都解析为 1024；否则新教师会使用 `RolloutConfig` 中的 `512 + 512`。[`validate_and_prepare_for_distillation`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/config/distillation.py#L172) 随后将 teacher 的输入上限改为 `1024 + 1024 = 2048`、输出上限改为 1，因为已有 student response 也会作为 teacher 的输入。这里的 `max_model_len: 2049` 为这一次额外生成留出位置；若 student 长度配置增大，这个总上限也要相应增大。后文会用 P0 的具体 token 解释这一步。

完成这些配置后，再看路由。配置里的 `gsm8k` 是 YAML 条目名，样本中的 `openai/gsm8k` 是路由值，两者不必同名。解析器以每个条目的 `key` 重新建立运行时字典，因此实际过程是：

```text
YAML teacher_models.gsm8k.key = "openai/gsm8k"
  → 运行时 teacher_models["openai/gsm8k"]

P0.data_source = "openai/gsm8k"
  → 对应的 teacher client
  → 该 teacher 的某个 inference replica
```

`_compute_teacher_logprobs` 从样本参数中读取 `teacher_key` 指定的字段，并把可能的 NumPy scalar 转成 Python 值。[`_resolve_teacher_key`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/experimental/teacher_loop/teacher_manager.py#L100) 在多个 teacher 时要求精确匹配，否则报错；仅有一个 teacher 时则直接选它，不依赖样本的路由值。

所以 P0 的四条回答会选择同一种 teacher。这里没有把四条回答分配给四位老师，也没有逐 token 选择不同老师。若批内另一些样本来自 `hiyouga/geometry3k`，那些样本才会选 Geo3K teacher。

## 第二条回答结束后，teacher 实际收到什么

V1 的 [`_agent_loop_postprocess`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/agent_loop_tq.py#L151) 先完成规则评分，再对 final output 调用 teacher，最后把字段写回 TransferQueue。因此 teacher 请求位于这一条 session 完成后的处理过程中，无须等全局 512 条回答一起完成。

对于 `P0_1_0`，符号缩尺后的请求可以写成：

```text
prompt_ids：   [p0 p1 p2 p3 p4 p5]             长度 6
response_ids： [b0 b1 b2 b3 b4]                长度 5

sequence_ids： [p0 p1 p2 p3 p4 p5 b0 b1 b2 b3 b4]
                                                  长度 11
```

[`compute_teacher_logprobs_single`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/experimental/teacher_loop/teacher_manager.py#L116) 将这 11 个已有 token 作为 `prompt_ids` 交给 teacher server。请求参数包含 `max_tokens=1`、`temperature=1.0`、`detokenize=false`，以及 `prompt_logprobs`。接口会额外生成一个 token，但训练使用的是已有输入序列的 prompt log-prob，额外生成的 token 不会替换 student response。

使用 sampled-token estimator 时，`prompt_logprobs=0`，每个位置取 student 已选 token 的一个概率；使用 top-k loss 时，参数为 K，每个位置取 teacher 的 K 个候选 token 及其概率。teacher 的上下文上限因此要覆盖 `max_prompt_length + max_response_length + 1`。主线的两个上限都是 1024，对 teacher 就至少需要 2049 个位置。

这里还有一个容易错一位的细节。vLLM 最初返回的第 i 项表示第 i 个输入 token 在此前上下文下的概率，而模型第 i 个 hidden state 预测的是第 i+1 个 token。verl 的 [`extract_prompt_logprobs`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/rollout/vllm_rollout/utils.py#L527) 去掉第一项，并在最后补一行 dummy，使 teacher 结果与 student 的预测位置对齐：

```text
存储行      0       1       …       5       6       …       9       10
当前输入    p0      p1      …       p5      b0      …       b3      b4
预测目标    p1      p2      …       b0      b1      …       b4      dummy
监督区域                           └───── response ───────┘
```

这时 `teacher_ids` 和 `teacher_logprobs` 的形状都是 `[11,1]` 或 `[11,K]`，覆盖完整序列的预测位置。manager 中的断言也要求第一维等于 `len(sequence_ids)`；不能因局部注释写了 response length，就把它理解成 `[5,K]`。

## 两个新字段怎样跟着原来的 trajectory 进入训练

teacher 返回后，`AgentLoopOutput.extra_fields` 增加 `teacher_ids`、`teacher_logprobs`。[`AgentLoopOutput.as_dict`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/experimental/agent_loop/agent_loop.py#L150) 把它们提到可写入数据存储的字段中。V1 将这些结果和 P0 原有的 `input_ids`、`response_mask`、`rm_scores` 等一起写在同一个 trajectory key 下。

因此 student worker 读出 `P0_1_0` 时，不需要再次请求 teacher，也不通过 batch 中的位置猜测对应关系。teacher 信号属于这条具体的 student trajectory，采样之后保持固定；当前 student 的概率会在后续更新中重新计算。

为了观察布局变化，暂时把 P0 的四条回答放在一个演算批中。实际负载均衡和 DP 分发可能把它们分开，并不保证四条都位于同一个 micro-batch。

```text
完整序列长度：6 + [3,5,4,2] = [9,11,10,8]

teacher_logprobs 的逐行张量：
  [9,K]、[11,K]、[10,K]、[8,K]

组成 nested tensor：
  values  [38,K]
  offsets [0,9,20,30,38]

FSDP、SP=1 的 top-k 计算：
  teacher values → [1,38,K]
  student logits → [1,38,V]
  token loss     → [1,38]

截取 response 预测位置并补齐：
  [4,5]，再使用 response_mask 聚合
```

这里 V 是 student vocabulary size。完整序列的 token 数为 38，四条 response 的有效 token 总数为 14；这两个数对应不同阶段，不能互换。`no_padding_2_padding` 会按每条序列的长度，从 `prompt_length - 1` 开始取预测 response 所需的位置，排除末尾 dummy，然后才形成 response 对齐的矩阵。

teacher 返回的 token ID 会直接用于 student vocabulary 的索引。因而 teacher 与 student 需要兼容的 token ID 语义；“同属语言模型”并不足以保证兼容。当前链路没有自动把任意两种 tokenizer 的概率分布映射到同一词表。

## `k1` 怎样在一个 token 上变成更新信号

先沿本章的主要配置看 sampled-token 路径。假设某个 student 已生成 token 在当前 student 下的概率为 0.2，在 teacher 下为 0.5。这里的概率是手算示意值。

[`compute_distillation_loss_reverse_kl_estimator`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/distillation/losses.py#L373) 取出 response 对齐的 student、teacher log-prob；`k1` 对该位置给出：

$$
d_t=\log q_\theta(a_t\mid s_t)-\log p_T(a_t\mid s_t)
=\log(0.2/0.5)\approx-0.916.
$$

其中 $s_t$ 是 student 生成的前缀，$a_t$ 是已经采样出的 token。单个采样点的这个差值允许为负，不能把每个点都视为非负的完整 KL divergence。

`use_policy_gradient=true` 时，代码将 `-distillation_losses.detach()` 当作 advantage。本例就是 `+0.916`，随后送入与第五章相似的 PPO loss：当前 student 比 teacher 更低估这个已采样 token，正 advantage 会推动提高其概率，同时受到 PPO ratio 和 clipping 的影响。

这里的 teacher advantage 是**逐 token** 的，不是把 P0 的四个规则分数重新做一次 GRPO 标准化。规则产生的组内 advantage 与 teacher 产生的 token advantage 可以同时存在，分别计算后在最终 loss 中相加。

`detach()` 也很关键：policy-gradient 路径把本次算出的差值作为权重，通过 student log-prob 求梯度。如果直接对 `log q - log p_T` 反向传播，teacher 项是常量，梯度不再反映 teacher 的概率大小。因此配置类明确拒绝 `loss_mode=k1` 与 `use_policy_gradient=false` 的组合。

teacher 的概率在本轮保持固定，不过 `k1` 中的当前 student 概率仍会变化，所以由它构造的 distillation advantage 可以在不同更新中改变。这与第五章提前算好、随后固定的 GRPO advantage 不同。

同一 estimator 路径还注册了 `k2`、`k3` / `low_var_kl` 等形式，具体数值由 `kl_penalty` 决定。`loss_max_clamp` 可以在聚合或构造 PG 权重前限制 token loss 的幅度；因此实际更新中还要区分原始概率差值与截断后的值。当前配置检查只接受蒸馏侧的 `policy_loss_mode=vanilla`，不能直接把主 actor 其他 policy-loss 变体的名字填入这里。

## Top-k 路径怎样使用未采样 token 的信息

若切换为 `loss_mode=forward_kl_topk`、`use_policy_gradient=false`，teacher 会提供一组候选 token，student 需要比较这些候选的分布。实际 K 可以设为 64；为了手算，下面仅演示 K=3 的一个位置：

| Teacher 候选 token ID | Teacher 概率 | Student 在同一 token ID 上的概率 |
| --------------------- | -----------: | -------------------------------: |
| 17                    |         0.50 |                             0.20 |
| 42                    |         0.20 |                             0.25 |
| 9                     |         0.10 |                             0.05 |

[`compute_forward_kl_topk`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/distillation/fsdp/losses.py#L77) 先对 student logits 做 vocabulary 维的归一化，再用 **teacher IDs** gather 出 `[1,N,K]` 的 student log-prob。因此比较的是相同 token，不能分别取 teacher、student 的 top-k 后按第几名直接相减。

在不触发 log-prob clamp 的示例中，这个位置的截断求和为：

$$
\sum_{v\in\mathrm{TopK}(p_T)}p_T(v)\log\frac{p_T(v)}{q_\theta(v)}
=0.5\log2.5+0.2\log0.8+0.1\log2\approx0.483.
$$

teacher 的 top-k 概率和为 0.8，student 在这些 token 上的概率和为 0.5。实现没有把它们重新归一化成两个和为 1 的 K 维分布，因此这是截断后的 KL 求和，不等于完整 vocabulary 的 KL，并且在其他数值下可能为负。最终 loss 层会将 top-k 结果截到非负，再按 response mask 聚合；`teacher_mass`、`student_mass` 则用于观察这部分候选覆盖了多少概率质量。

标准 FSDP 分支会形成 vocabulary 大小的 log-softmax 中间结果。`use_chunked_topk=true` 的分支改用分块 `logsumexp + gather`，避免再物化同尺寸的 log-softmax 缓冲区，但 student logits 本身仍然存在。这是内存与额外计算开销之间的取舍。当前快照还记录 teacher / student top-k overlap，用于观察两者候选集合的重合情况。

K 也影响服务配置。vLLM 的 `max_logprobs` 必须至少为 K，teacher 配置校验会补齐未设置的上限，或在显式上限过小时报错。student FSDP 侧还会拒绝同时启用 `pad_to_length` 和 top-k distillation，因为额外静态补齐会破坏 student logits 与原有 teacher token 行的对齐；本章保持这一额外补齐关闭。

top-k loss 在 logits processor 中提前计算，变成每个位置一个数，再进入最终的 mask 聚合。若把 `use_policy_gradient` 重新打开，这些数又会被转成 sampled-token 的 PG 权重；代码会提示非采样 token 的直接分布监督没有被充分使用。要理解这两条路径，应看梯度最终作用于哪些 logits，而不能只看它们都使用了 top-k teacher 输出。

## 最终 student loss 还保留哪些原有部分

[`distillation_ppo_loss`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/distillation/losses.py#L165) 有两个调用阶段：提供 `student_logits` 时计算 top-k 辅助输出；最终调用时组合 policy loss 和 distillation loss。组合关系是：

$$
L=\begin{cases}
L_{\mathrm{ppo\_loss}}+c_{\mathrm{distill}}L_{\mathrm{distill}},&\texttt{use\_task\_rewards=true},\\
L_{\mathrm{distill}},&\texttt{use\_task\_rewards=false}.
\end{cases}
$$

这里的 `L_ppo_loss` 是整个 `ppo_loss` 返回值，包括已启用的 reference KL、entropy 等项。本章沿用主线 `use_kl_loss=true`、`kl_loss_coef=0.001`、`kl_loss_type=low_var_kl`，所以在保留 task rewards 时，它们与 distillation 项并存。

若改为仓库示例常用的 `use_task_rewards=false`，代码会将整个 policy loss 置零，蒸馏系数也固定按 1 处理。不能以为这里只删除规则 reward 项、原来的 reference KL 仍会自动加在最终 loss 上。PG 蒸馏分支仍会调用 `ppo_loss` 后丢弃其结果，因此也不能仅由“最终不使用 task loss”推断所有相关中间字段都不再需要。

reference 与 teacher 此时各有明确用途。reference 为“离初始化策略偏移了多少”提供基准，teacher 则对 student 前缀提供希望学习的分布。它们可以来自不同 checkpoint，也可以采用不同运行后端；相同的是都不跟随当前 student 的每次 optimizer update 学习。

student 更新仍经过原有四次 mini-batch，再把权重同步到 rollout。teacher 输出不需要同步成 student 权重；下一轮由新 student 生成新的前缀，再重新请求 teacher。

## 当前快照支持到哪里

仓库有明确的 [FSDP MOPD 示例](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/examples/on_policy_distillation_trainer/run_qwen3_8b_mopd_fsdp.sh) 和 [VeOmni MOPD 示例](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/examples/on_policy_distillation_trainer/run_qwen3_8b_mopd_veomni.sh)，把 GSM8K 文本教师与 Geo3K 视觉语言教师接入同一 student；还有 [V1 separate-async OPD 端到端脚本](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/tests/special_e2e/run_v1_separate_async_opd.sh)。这些代码说明当前快照具有多教师路由和对应训练实现，但不是本地已经复现其性能、显存和训练效果的证据。

Megatron 有 distillation loss 实现及单教师 OPD 示例；当前快照没有专门命名为 MOPD Megatron 的标准示例，不能仅凭共用 loss 接口就宣称所有多教师后端组合都已验证。

本文选择单轮、每个 session 一个 output，还避开了一个具体边界：V1 后处理当前只对 `final_output` 请求 teacher，源码留有支持 output list 的 TODO。对一个 session 产生多个 `AgentLoopOutput` 的模式，不能假定所有 output 都已获得完整 teacher 字段。validation 路径也不会触发 teacher 请求。

继续研究算法时，可对照 [[training/post-training/on-policy-kd|On-Policy Distillation]] 与 [[sources/papers/2026-mopd-multi-teacher-on-policy-distillation-for-capability-integration|MOPD 论文笔记]]；验证这条工程链则进入 [[projects/verl-source-reading/labs/05-multi-teacher-opd|Lab 05 Multi-Teacher OPD]]，观察 P0 的路由、完整序列长度、response 切片以及最终 loss 中实际保留的项。
