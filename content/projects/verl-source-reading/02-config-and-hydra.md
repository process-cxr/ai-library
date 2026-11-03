---
title: 02 Launch and First Batch
created: 2026-09-02
published: 2026-09-02
modified: 2026-11-03
type: project
status: growing
tags:
  - projects
  - source-reading
  - verl
  - hydra
  - configuration
---

01 用一轮训练说明了 128 个 prompt 如何生成 512 条 trajectory。这一篇回到数据产生之前：从命令进入 Hydra，创建 trainer 和 dataloader，再取出包含 P0 的第一批问题。阅读终点是 `generate_sequences(batch)` 收到 128 行原始消息；模型生成放在 03。

机器、算法和样本沿用 [[projects/verl-source-reading/01-sync-grpo-run-overview#贯穿全文的案例|01 的固定案例]]：两节点八卡、FSDP SP=1、vLLM TP=2，B=128、G=4、M=32、E=1。下面将这些数字逐个落实到配置读取和数据代码中。

## 从一条明确的命令开始

上游 [Qwen3-4B GRPO 脚本](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/examples/grpo_trainer/run_qwen3_4b_fsdp.sh) 默认使用 dynamic batch。为了先学习固定行数的分批方式，本例显式关闭它，并同时指定 actor、rollout log-prob、reference log-prob 的 micro-batch。

先看直接决定数据量和执行规模的参数。表中 `actor.*`、`rollout.*` 是 `actor_rollout_ref` 下配置路径的简写：

| 参数                                               | 本例值  | 沿 P0 看它的作用                              |
| -------------------------------------------------- | ------- | --------------------------------------------- |
| `model_engine`                                     | `dp`    | 选择 FSDP actor/reference 配置组              |
| `trainer.v1.trainer_mode`                          | `sync`  | 本轮生成完成后训练，再同步权重                |
| `trainer.nnodes × trainer.n_gpus_per_node`         | `2 × 4` | 共有 8 张训练 GPU                             |
| `data.train_batch_size`                            | `128`   | 先取 128 道题                                 |
| `rollout.n`                                        | `4`     | P0 启动四个独立 session                       |
| `actor.ppo_mini_batch_size`                        | `32`    | 每次全局更新消费 `32 × 4 = 128` 条 trajectory |
| `actor.ppo_epochs`                                 | `1`     | 本批 trajectory 只遍历一次                    |
| `actor.ppo_micro_batch_size_per_gpu`               | `4`     | 每个 rank 一次 forward/backward 处理 4 条     |
| `actor.fsdp_config.ulysses_sequence_parallel_size` | `1`     | 本例 actor DP 为 `8 / 1 = 8`                  |
| `rollout.tensor_model_parallel_size`               | `2`     | 两张 GPU 共同运行一个生成副本                 |

现在已经能预测本轮的 `128 → 512 → 每 rank 64 → mini-batch 16 → micro-batch 4`。下面按代码解释这些值如何进入对象；需要查看所有覆盖项时，再展开完整命令。KL、clipping 等算法参数会在 05 使用，此处不必先逐项记忆。

<details>
<summary>展开完整教学命令与运行前提</summary>

以下命令表示训练配置，假设已在固定源码环境中安装依赖、准备好数据，并让两个各有四张 GPU 的节点加入同一个 Ray 集群。`/data/gsm8k/*.parquet` 是示例数据路径，需要替换成运行环境可读取的文件；仅设置 `trainer.nnodes=2` 不会自动创建或连接第二台机器。

```bash
python -m verl.trainer.main_ppo \
  model_engine=dp \
  trainer.use_v1=true trainer.v1.trainer_mode=sync trainer.resume_mode=disable \
  trainer.nnodes=2 trainer.n_gpus_per_node=4 \
  +ray_kwargs.ray_init.address=auto \
  data.train_files=/data/gsm8k/train.parquet \
  data.val_files=/data/gsm8k/test.parquet \
  data.train_batch_size=128 data.gen_batch_size=null \
  data.max_prompt_length=1024 data.max_response_length=1024 \
  data.filter_overlong_prompts=true data.truncation=error \
  algorithm.adv_estimator=grpo algorithm.norm_adv_by_std_in_grpo=true \
  algorithm.use_kl_in_reward=false algorithm.rollout_correction=null \
  algorithm.filter_groups.enable=false \
  trainer.v1.sampler.sync_refill_failed_groups=false \
  critic.enable=false distillation.enabled=false \
  actor_rollout_ref.model.path=Qwen/Qwen3-4B \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
  actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.actor.fsdp_config.use_no_sync_for_gradient_accumulation=false \
  actor_rollout_ref.actor.pad_to_length=false \
  actor_rollout_ref.actor.use_dynamic_bsz=false \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.ppo_epochs=1 actor_rollout_ref.actor.shuffle=false \
  actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.clip_ratio=0.2 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 actor_rollout_ref.actor.clip_ratio_high=0.2 \
  actor_rollout_ref.actor.clip_ratio_c=3.0 \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.agent.num_workers=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.data_parallel_size=1 \
  actor_rollout_ref.rollout.pipeline_model_parallel_size=1 \
  actor_rollout_ref.rollout.disaggregation.enabled=false \
  actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.calculate_log_probs=true \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=false \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=false \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  reward.reward_model.enable=false reward.reward_manager.name=naive \
  trainer.val_before_train=false trainer.test_freq=-1 trainer.save_freq=-1 \
  trainer.logger='[console]'
```

这是基于源码构造的教学命令，未执行集群训练。为了只查看最终配置，可以在同一命令末尾追加 `--cfg job --resolve`；Hydra 会输出组合结果，不进入训练入口。配置打印成功只证明配置组合完成，不代表模型、GPU、Ray 和 vLLM 已经跑通。

本例从指定模型开始，不恢复输出目录中的旧训练 checkpoint；保留正常的数据集迭代，关闭训练前验证、定期验证和保存，方便先看清训练主干。这些旁路之后可单独恢复。日志、checkpoint 与实际多机启动的操作记录放在 [[projects/verl-source-reading/labs/01-trace-config-and-batch|Lab 01]]。

</details>

## Hydra 如何找到 actor 的配置

[main_ppo.py](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/main_ppo.py#L166) 的入口写着：

```python
@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    ...
```

因此 Hydra 从 trainer 的 `config/ppo_trainer.yaml` 开始组合配置。下面摘出与 actor 有关的 defaults，省略其他配置组：

```yaml
defaults:
  - model_engine: dp
  - actor@actor_rollout_ref.actor: ${model_engine}_actor
  - data@data: legacy_data
  - ref@actor_rollout_ref.ref: ${model_engine}_ref
  - rollout@actor_rollout_ref.rollout: rollout
  - model@actor_rollout_ref.model: hf_model
  # 其他配置组省略
  - _self_
```

`model_engine=dp` 使 actor 选择 `dp_actor.yaml`。`actor@actor_rollout_ref.actor` 同时规定存放位置：从 actor 配置目录加载内容，放入最终配置树的 `actor_rollout_ref.actor` 节点。

接着 [dp_actor.yaml](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/config/actor/dp_actor.yaml) 又组合公共 actor、FSDP engine 和 optimizer 配置。公共配置给出 mini-batch 默认值 256，CLI 将它覆盖成 32。把这一个值的来源连起来就是：

```text
actor/actor.yaml: ppo_mini_batch_size=256
  → dp_actor 继承公共 actor 配置
  → 挂载到 actor_rollout_ref.actor
  → CLI 覆盖为 32
  → config.actor_rollout_ref.actor.ppo_mini_batch_size == 32
```

`_self_` 决定当前 YAML 自身字段与导入配置的合并位置；它位于末尾时，当前文件的同名字段覆盖此前导入值。命令行覆盖在组合后生效。理解这一步，才能判断“我看见 YAML 写 256，为何实际是 32”。

## 一个字段引用另一个字段时

本例命令设置了 `data.max_response_length=1024`，但没有直接设置 `rollout.response_length`。后者来自 [rollout.yaml](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/config/rollout/rollout.yaml#L35)：

```yaml
response_length: ${oc.select:data.max_response_length,512}
```

它读取 `data.max_response_length`，不存在时才使用 fallback 512。本例得到 1024。同样，actor 的 `rollout_n` 引用 `rollout.n`，所以 actor 能知道每题有 4 个 session。

OmegaConf 支持按访问惰性求值；不能理解成 resolve 之前读取任何插值都只会拿到原始字符串。`TaskRunnerV1.run` 中的 `OmegaConf.resolve(config)` 是把整棵树的引用就地解析为值，便于后续使用和传递。打印 `OmegaConf.to_container(config, resolve=True)` 也能看到解析结果，但它本身不负责创建模型。

对当前案例，可以在 resolved config 中核对：

| 路径                                                                 | 应得到的值 | 后续消费者                               |
| -------------------------------------------------------------------- | ---------- | ---------------------------------------- |
| `data.train_batch_size`                                              | 128        | `_next_train_batch`、ReplayBuffer sample |
| `actor_rollout_ref.rollout.n`                                        | 4          | `AgentLoopWorkerTQ._run_prompt`          |
| `actor_rollout_ref.actor.ppo_mini_batch_size`                        | 32         | trainer `_update_actor`                  |
| `actor_rollout_ref.rollout.response_length`                          | 1024       | AgentLoop 和生成请求的长度限制           |
| `actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size` | 1          | FSDP engine 的并行布局                   |

## 验证配置时已经会创建部分 dataclass

[main](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/main_ppo.py#L177) 把角色需求传给 `validate_config`：

```python
validate_config(
    config=config,
    use_reference_policy=need_reference_policy(config),
    use_critic=need_critic(config),
)
```

本例 `actor.use_kl_loss=true`，因此需要 reference；`critic.enable=false`，因此不需要 critic。GRPO 的组内 baseline 与 reference KL 是两种不同用途。

[`validate_config`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/utils/config.py#L74) 会调用 `omega_conf_to_dataclass(config.actor_rollout_ref.actor)`，再调用对象的 `validate`。所以 dataclass 转换不必等到 worker 内才首次发生。配置里的 `_target_: verl.workers.config.FSDPActorConfig` 指明转换的目标类；若调用方显式传入 dataclass 类型，工具函数则采用合并 structured config 后 `OmegaConf.to_object` 的路径。

为什么命令同时指定三处 micro-batch？关闭 dynamic batch 后，actor training、old log-prob 和 reference log-prob 都需要静态分批参数。`validate_config` 会拒绝缺失的 log-prob micro-batch，也会拒绝同时填写旧的全局字段和新的 per-GPU 字段。只写 actor 的 4，不能据此假设 reference 自动拿到了同样的 4。

本例的几层整除关系分别是：

```text
全批 trajectory:       128 × 4 = 512
全局 actor mini-batch:  32 × 4 = 128
actor DP:              8 / SP1 = 8
本地 mini-batch:        128 / 8 = 16
本地 micro-batch:       4
每个 mini-batch 内:     16 / 4 = 4 次 forward/backward
```

这些约束并非全部在同一个验证函数中检查。入口、actor config、TrainingWorker 和 engine 各自检查所负责的部分；提前算清它们能理解后面各处 assertion 为什么存在。

## TaskRunner 怎样让这些设置进入运行对象

配置检查后，`run_ppo` 使用 `ray_kwargs.ray_init` 初始化 Ray，再创建远程 `TaskRunnerV1`。`address=auto` 让本例连接已存在的集群，GPU 资源池稍后按 `[4,4]` 请求资源。

[`TaskRunnerV1.run`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/main_ppo.py#L133) 的关键操作是：

```python
trainer_cls = get_trainer_cls(config.trainer.v1.trainer_mode)
config.transfer_queue.enable = True
OmegaConf.resolve(config)
tq.init(config.transfer_queue)
self.trainer = trainer_cls(config=config)
self.trainer.init()
self.init_agent_loop_manager()
self.trainer.fit(self.agent_loop_manager)
```

这里省略打印和清理的 `try/finally`。它选择 `PPOTrainerSync`、启动 TQ，再创建 trainer。随后 `trainer.init` 准备数据、模型和服务；AgentLoopManager 获得 rollout client 与 reward handles，才有条件接收数据。

`trainer_mode=sync` 决定训练与生成的时间顺序；`model_engine=dp` 决定 actor/reference 等模型配置组。`rollout.mode=async` 这个默认值描述生成服务 API，不等于选择 async trainer；本例仍然是同步训练。

## 数据集中的 P0 还不是 token 张量

[`RLHFDataset`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/utils/dataset/rl_dataset.py#L169) 根据后缀读取 parquet、json 或 jsonl，拼接数据集。启用超长过滤时，预处理会检查 prompt 的格式化长度。每行实际取出时，`__getitem__` 主要保留原始消息：

```python
row_dict: dict = self.dataframe[item]
row_dict["raw_prompt"] = self._build_messages(row_dict, key=self.prompt_key)
row_dict["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)
```

这是 [RLHFDataset 的取样方法](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/utils/dataset/rl_dataset.py#L386)的节选。P0 的 `raw_prompt` 包含铅笔问题，`reward_model.ground_truth` 保留字符串 `"12"`，`index` 从 `extra_info` 提取，工具 kwargs 在纯文本单轮案例中为空。

`dummy_tensor` 只是兼容数据协议的占位，不是模型输入。真正进入生成的 chat-template tokenization 在 AgentLoop 中执行，03 会从这里继续。

## 为什么一次 DataLoader fetch 恰好是 128 行

[`_init_dataloader`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L657) 使用：

```python
gen_batch_size = self.config.data.get("gen_batch_size", None) or self.config.data.train_batch_size
```

本例 `gen_batch_size=null`，因此 fetch 大小等于 128。没有开启 DAPO group filtering、失败组 refill 或 async，它也不会被初始化代码强制改为 1。`StatefulDataLoader` 使用 `drop_last=True`，不足完整 fetch 的尾批不会照常作为 128 行返回。

[`collate_fn`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/utils/dataset/rl_dataset.py#L41) 将 tensor stack，将其他字段组成一维 object array。此时可以把 batch 看成：

| 字段           | collate 后的表示        | P0 对应的一行              |
| -------------- | ----------------------- | -------------------------- |
| `dummy_tensor` | uint8 tensor，`[128,1]` | `[0]`                      |
| `raw_prompt`   | object array，`[128]`   | 一组 user messages         |
| `data_source`  | object array，`[128]`   | `openai/gsm8k`             |
| `reward_model` | object array，`[128]`   | `ground_truth="12"`        |
| `index`        | object array，`[128]`   | 数据行 metadata 中的 index |

这个阶段没有 `responses`，也没有 `[512,R]` 的训练矩阵。`rollout.n=4` 尚未应用。

## 给本批数据建立身份，再交给 AgentLoop

[`_fetch_one_gen_batch`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L1373) 在取出 batch 后创建 uid：

```python
batch_dict["uid"] = np.array(
    [str(uuid.uuid4()) for _ in range(len(batch_dict["raw_prompt"]))],
    dtype=object,
)
return tu.get_tensordict(batch_dict)
```

P0 是其中一个 UUID 的简写。`get_tensordict` 让数值张量和非张量字段可以统一按 batch 维切分；消息、字符串和字典仍然是非张量数据。接着 `_next_train_batch` 写入本批 `global_steps`。在本例中只 fetch 一次；显式设置较小的 `gen_batch_size` 时，该函数可以合并多个 fetch，并要求请求数量是它的整数倍。

最后，[`_submit_batch_to_rollout`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L1403) 为这 128 个 uid 建立 tags：

```python
tags = [
    {"is_prompt": True, "status": "pending", "global_steps": self.global_steps}
    for _ in range(len(batch))
]
```

在 sync 分支，它调用 `tq.kv_batch_put` 时只传入 keys、partition 和 tags，完整 prompt batch 则交给 `self.agent_loop_manager.generate_sequences(batch)`。两处存放内容不同：

```text
TQ train partition:
  key=P0
  tag={is_prompt: true, status: pending, global_steps: 本 step}

AgentLoop 输入 batch 的 P0 行:
  raw_prompt、data_source、reward_model、extra_info、uid、index ...
```

P0 此时已经可被 controller 查询状态，但回答仍未产生。[[projects/verl-source-reading/03-data-protocol-and-batch-lifecycle|03 From Prompts to Trajectories]] 从这个输入继续：拆出一行，创建四个 session，把消息转换成生成请求。

## 回查一个配置值时从哪里找

阅读更深的 worker 代码时，可以沿“运行时读取处 → 配置类/插值 → defaults → CLI”反向追踪同一个值。例如 mini-batch 在 trainer 处乘 G，micro-batch 在 engine 处决定分块；看到两个 batch size 时先定位它们的消费者。

仓库中的 `_generated_ppo_trainer.yaml` 是组合配置的参考产物，不是本入口新增的配置源。修改真实行为应回到 defaults 对应 YAML 或 CLI override。运行中 trainer 还会写入 optimizer schedule 等派生值，因此一份启动前的配置打印也不是对象生命周期中所有后续变化的记录。
