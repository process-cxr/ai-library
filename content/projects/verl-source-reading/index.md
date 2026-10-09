---
title: verl Source Reading
created: 2026-09-02
published: 2026-09-02
modified: 2026-10-09
type: project
status: growing
tags:
  - projects
  - source-reading
  - verl
  - post-training
  - reinforcement-learning
---

verl（Volcano Engine Reinforcement Learning for LLMs）是一套面向大模型 post-training 的分布式 RL 训练框架。它并不只实现 PPO、GRPO 等算法公式，而是把 prompt data、rollout、环境交互、reward、advantage estimation、actor / critic update、分布式执行和参数同步组织成一个可以持续运行的系统。

本项目以固定 revision 下的 V1 trainer 为阅读边界，先追踪 `sync GRPO + FSDP + vLLM + rule-based reward` 这条基线，再比较 PPO、Megatron、Agentic RL 和 async trainer。页面中的“支持”“默认”和“版本”都只在这个阅读边界内成立；其他 backend 或新版本行为需要重新核对源码。

这组源码阅读关注的核心不是“某个 API 怎样调用”，而是一次 policy update 如何跨越多种 runtime：

```text
prompt 从哪里进入系统？
  -> 谁负责生成 response 或 agent trajectory？
  -> reward 在哪里计算并写回？
  -> old / reference / current policy 如何区分？
  -> advantage 如何变成 actor / critic 的训练字段？
  -> batch 如何映射到 FSDP 或 Megatron workers？
  -> 更新后的参数如何回到 rollout engine？
```

## verl 的主要应用场景

verl 主要用于大语言模型的强化学习后训练：让模型生成回答或与工具、环境交互，根据任务结果获得奖励，再用 PPO、GRPO 等算法更新策略。典型场景包括使用规则或奖励模型评价生成结果、根据奖励优化回答，以及通过多轮工具交互训练 Agent。

在这些场景中，训练所用的回答或交互轨迹需要在训练过程中采样生成。verl 将生成、奖励计算和参数更新连接起来，使更新后的模型继续生成下一批训练样本，形成持续迭代的训练闭环。

verl 也提供监督微调（Supervised Fine-Tuning，SFT）功能，使用给定的输入和目标输出训练模型，见 [[projects/verl-source-reading/verl-sft-training-path|verl SFT Training Path]]。本项目以强化学习训练闭环为学习主线，SFT 作为辅助对照。

```text
                    ┌──────── rollout / environment ──────────┐
prompt dataset ───> policy generation ───> trajectory         │
                         │                    │               │
                         │                    v               │
                         │               reward / verifier    │
                         │                    │               │
                         └──── new weights <──┼───────────────┘
                                              v
                                  advantage / policy loss
                                              │
                                              v
                                      actor update
```

verl 的主要职责，就是管理这条闭环中的数据依赖、模型角色、分布式资源和版本关系。算法决定“优化什么”，training backend 决定“梯度如何计算”，rollout backend 决定“trajectory 如何生成”，controller 则把它们连接成一轮可执行的训练过程。

## 系统地图

从源码职责看，可以先把 verl 分成五个职责区。它们不是五个彼此隔离的进程层，而是阅读时用来划分 ownership 和接口边界的视角。

### 入口与配置

[`examples/grpo_trainer/`](https://github.com/process-cxr/verl-upstream/tree/5b79827b04cee6e4b7b5ff737e047f1d72d50433/examples/grpo_trainer) 提供可运行配置，[`verl/trainer/main_ppo.py`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/main_ppo.py) 负责 Hydra config、运行前校验、Ray 初始化和 TaskRunner 选择。阅读这一层需要建立“shell override、config default 和 runtime object”之间的对应关系。

### Trainer 与算法编排

[`verl/trainer/ppo/v1/`](https://github.com/process-cxr/verl-upstream/tree/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1) 负责一轮训练的控制流：提交 prompt、等待 trajectory、安排 reward 计算、构造 log-prob 和 advantage、触发 actor / critic update、记录 metrics，并在 step 结束后清理或保留状态。具体 reward 执行仍由 RewardLoop 或配置指定的 reward 路径完成。

这一层表达的是 controller-level 的训练语义。`train_batch_size`、`rollout.n`、prompt group、old policy 和 reference policy 首先在这里被组织起来；进入 backend 后，它们还要被映射为具体的本地 batch、process group 和执行 schedule。

### Trajectory、Reward 与状态存储

AgentLoop 把 single-turn generation 和 multi-turn tool interaction 统一为 trajectory；RewardLoop 对接 rule-based verifier、custom reward 或 reward model；TransferQueue 保存 prompt group 状态和逐步补齐的训练字段；ReplayBuffer 决定哪些 trajectory 可以进入当前 update。

这部分是 verl 从普通 model training framework 走向 agentic RL system 的关键：response 不再只是连续生成文本，还可能包含多轮 action、tool observation 和失败状态。在 async 或 off-policy 路径中，trajectory 还会携带用于判断生成时序和 policy staleness 的 step / version metadata。

启用 OPD 时，trajectory 还会附带 teacher log-prob 字段；MOPD 进一步使用样本中的 routing key 选择不同 teacher。teacher 只提供 inference-time supervision，student actor 仍负责最终参数更新，详见 [[projects/verl-source-reading/12-on-policy-distillation-and-mopd|12 On-Policy Distillation and MOPD]]。

### Training backend

本项目先以 FSDP 和 Megatron 作为主要 training backend 对照。不同 backend 消费相同的 controller-level 字段，但 process-group topology、batch dispatch、forward-backward schedule、gradient communication、optimizer state 和 checkpoint layout 并不相同；其他 backend 不属于当前基线的默认结论。

```text
global actor batch
  ├─ FSDP:    FSDP shard mesh + optional Ulysses SP
  └─ Megatron: TP / PP / CP / DP / EP + pipeline schedule
```

因此 `DP size` 不是一个脱离 backend 的固定数字，也不能由 rollout TP 推导。

### Rollout backend 与权重循环

vLLM、SGLang、TensorRT-LLM 或 HF rollout 负责生成；LLMServerManager 管理 rollout replicas；CheckpointEngineManager 负责 actor training weights 与 rollout weights 之间的切换和同步。sync、colocate async 与 separate async 的主要区别，也集中体现在 rollout 和 training 是否重叠、trajectory 会跨越多少 policy versions，以及何时同步新权重。

## 一次 V1 Training Run

在本项目固定基线中，V1 trainer 的一次 sync GRPO update 可以先压缩为：

```text
run script
  -> main_ppo.main
  -> TaskRunnerV1
  -> PPOTrainerSync.init
  -> AgentLoopManager
  -> PPOTrainer.fit
       -> prepare_step
       -> rollout.n sessions per prompt
       -> TransferQueue / ReplayBuffer
       -> reward
       -> old / reference log-prob
       -> GRPO advantage
       -> actor update
       -> rollout weight sync
       -> metrics and cleanup
```

这条链路是后续专题页共享的坐标。每篇笔记会向下展开其中一个局部，但最终都需要回答它与整轮训练的关系。

## 阅读地图

第一次进入本项目，先读 [[projects/verl-source-reading/00-learning-guide|00 verl Learning Guide]]。它给出先修知识、证据等级、源码阅读 ledger、最小实践顺序和自测问题；具体专题的依赖顺序与源码入口见 [[projects/verl-source-reading/roadmap|verl Source Reading Roadmap]]。

### 主运行链与数据协议

- [[projects/verl-source-reading/01-sync-grpo-run-overview|01 Sync GRPO Run Overview]]：从启动脚本追踪到 actor update 和 rollout weight sync。
- [[projects/verl-source-reading/02-config-and-hydra|02 Config and Hydra]]：解释 Hydra composition、CLI overrides 与 runtime config。
- [[projects/verl-source-reading/03-data-protocol-and-batch-lifecycle|03 Data Protocol and Batch Lifecycle]]：追踪 `DataProto`、`TensorDict`、`KVBatchMeta`、TransferQueue 和训练字段生命周期。
- [[projects/verl-source-reading/verl-sft-training-path|verl SFT Training Path]]：补充监督训练的数据、mask、loss、engine 和 checkpoint 对照链路。

### Rollout、Reward 与优化算法

- [[projects/verl-source-reading/04-rollout-reward-and-agent-loop|04 Rollout, Reward and AgentLoop]]：分析 single-turn、multi-turn、tool observation 与 reward 路径。
- [[projects/verl-source-reading/05-advantage-and-policy-loss|05 Advantage and Policy Loss]]：把 reward、advantage、KL 和 policy objective 映射到 tensor 字段。
- [[projects/verl-source-reading/09-ppo-and-grpo-codepath|09 PPO and GRPO Codepath]]：比较 PPO 与 GRPO 在相同 trainer 中的真实分支。

### 分布式控制与执行后端

- [[projects/verl-source-reading/06-single-controller-and-ray|06 Single Controller and Ray]]：解释 controller method 如何调度到 Ray workers。
- [[projects/verl-source-reading/07-model-engine-and-backend|07 Model Engine and Backend]]：比较公共 model-engine contract 与 FSDP / Megatron 执行语义。
- [[projects/verl-source-reading/08-hybrid-engine-and-weight-sync|08 Hybrid Engine and Weight Sync]]：分析训练态、生成态、显存切换和参数版本同步。

### Agentic RL 与异步训练

- [[projects/verl-source-reading/11-agentic-rl-extension|11 Agentic RL Extension]]：研究多轮 tool-use trajectory 的状态、mask、termination 和 credit assignment。
- [[projects/verl-source-reading/10-async-trainer-and-off-policy|10 Async Trainer and Off-policy]]：比较三种 V1 trainer mode、policy staleness 和 off-policy control。
- [[projects/verl-source-reading/labs/|verl Source Reading Labs]]：用 resolved config、字段 trace 和最小实验检验静态源码判断。

### On-Policy Distillation

- [[projects/verl-source-reading/12-on-policy-distillation-and-mopd|12 On-Policy Distillation and MOPD]]：分析 OPD 的 teacher log-prob 路径、MOPD 的样本级 teacher routing、teacher resource pool 和 distillation loss。
- [[projects/verl-source-reading/labs/05-multi-teacher-opd|Lab 05: Multi-Teacher OPD]]：验证多教师路由、字段回写和资源池约束。

完整学习顺序见 [[projects/verl-source-reading/roadmap|verl Source Reading Roadmap]]。

## 阅读基线

- Repository：[process-cxr/verl-upstream](https://github.com/process-cxr/verl-upstream)
- Revision：[`5b79827b04cee6e4b7b5ff737e047f1d72d50433`](https://github.com/process-cxr/verl-upstream/tree/5b79827b04cee6e4b7b5ff737e047f1d72d50433)
- Branch at reading start：`main`
- Reading start：2026-09-02
- Primary trainer：[`main_ppo.py -> TaskRunnerV1 -> ppo/v1`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/main_ppo.py#L102)

源码链接固定到完整 commit，避免 upstream 演化导致函数、配置和行号与笔记失配。本项目入口选择 V1 trainer，并在该 revision 的范围内比较 `sync`、`colocate_async` 和 `separate_async`；旧 `main_ppo_v0.py` 只用于理解历史教程和旧架构，不作为正文主线。

本项目将 `sync GRPO + FSDP + vLLM + rule-based reward` 的静态调用链与运行事实分开记录。GPU shape、runtime timing、显存行为和 backend performance 只有在可复现的 lab 中执行后，才能写成 Tested behavior。

## 与通用知识的连接

源码项目页保存 verl 特有的调用链、数据结构、调度和 backend contract。可迁移的算法与系统机制继续沉淀在通用主题页：

- [[training/post-training/ppo|PPO]]
- [[training/post-training/grpo|GRPO]]
- [[training/post-training/reward-model|Reward Model]]
- [[training/distributed-training/fsdp|FSDP]]
- [[training/distributed-training/megatron|Megatron 与 3D 并行]]
- [[training/distributed-training/torch-distributed|torch.distributed]]
- [[inference/serving-systems/vllm|vLLM]]
