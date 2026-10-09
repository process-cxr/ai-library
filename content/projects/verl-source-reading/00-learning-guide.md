---
title: 00 verl Learning Guide
created: 2026-10-08
published: 2026-10-08
modified: 2026-10-09
type: project
status: growing
tags:
  - projects
  - source-reading
  - verl
  - learning-guide
  - post-training
---

这份指南说明如何进入 verl 源码阅读项目。它负责先修知识、证据等级、阅读方法、实践门槛和自测问题；具体专题的学习顺序、源码入口和产物见 [[projects/verl-source-reading/roadmap|verl Source Reading Roadmap]]。

项目的三页导航各有一个问题：[[projects/verl-source-reading/|项目首页]]回答“系统由什么组成”，roadmap 回答“按什么依赖顺序学习”，本页回答“如何判断自己真的理解了”。

## 先建立三个坐标系

阅读 verl 时，算法、数据和分布式运行时会同时出现。先分别建立三个坐标系，再研究它们的交点。

| 坐标系 | 核心对象 | 首先回答的问题 |
| --- | --- | --- |
| 算法语义 | policy、reference、critic、reward、advantage、KL | 当前优化的目标是什么，哪些模型参与但不更新？ |
| 数据语义 | prompt group、session、trajectory、token mask、batch | 一个样本如何生成、扩展、打分并进入 loss？ |
| 系统语义 | controller、worker、training engine、rollout engine | 计算在哪个进程和设备发生，参数何时同步？ |

这三个坐标不能互相替代。`rollout.n` 描述同一 prompt 的独立采样数，不是 data-parallel size；rollout tensor parallelism 描述生成侧布局，也不决定 actor training 的 DP；在需要 reference policy 的配置中，reference policy 参与 KL 或 reference log-prob 计算，但通常不是被 optimizer 更新的 actor。

## 先修知识与最低标准

不需要先掌握整个 RLHF 栈，但开始源码主线前，至少应能解释以下关系：

- [[training/post-training/ppo|PPO]]：behavior policy、current policy、ratio、clipping、advantage 和 value target；
- [[training/post-training/grpo|GRPO]]：同一 prompt 的 response group、group-relative baseline，以及为什么可以不训练 critic；
- [[training/post-training/reward-model|Reward Model]]：rule-based reward、learned reward model 和 verifier 的边界；
- [[training/distributed-training/fsdp|FSDP]]：参数、梯度和 optimizer state 的 sharding，以及它与普通 DDP 的差别；
- [[training/distributed-training/torch-distributed|torch.distributed]]：rank、world size、process group 和 collective communication；
- [[inference/serving-systems/vllm|vLLM]]：continuous batching、KV cache 和 rollout serving 为什么不同于训练 forward。

最低标准不是能推导所有公式，而是看到 `old_log_probs`、`ref_log_prob`、`advantages`、`response_mask` 和 `update_weights` 时，知道它们分别属于哪一个坐标系以及可能的 producer / consumer。

## 阶段一：闭合一次 Sync GRPO Update

第一阶段固定 `sync GRPO + FSDP + vLLM + rule-based reward`，暂时不切换算法、backend 或异步模式。

阅读顺序：

```text
example shell script
  -> Hydra resolved config
  -> main_ppo / TaskRunnerV1
  -> prompt batch and uid
  -> AgentLoop rollout.n sessions
  -> reward and log-prob fields
  -> GRPO advantage
  -> actor update
  -> rollout weight sync
```

对应笔记：

1. [[projects/verl-source-reading/01-sync-grpo-run-overview|01 Sync GRPO Run Overview]]：先获得完整时序，不在第一次阅读时下钻所有实现。
2. [[projects/verl-source-reading/02-config-and-hydra|02 Config and Hydra]]：把 shell override 映射到 resolved config 和 runtime consumer。
3. [[projects/verl-source-reading/03-data-protocol-and-batch-lifecycle|03 Data Protocol and Batch Lifecycle]]：追踪 uid、trajectory key、字段 producer / consumer 和 cleanup。
4. [[projects/verl-source-reading/04-rollout-reward-and-agent-loop|04 Rollout, Reward and AgentLoop]]：确认生成、tool interaction 和 reward 的边界。
5. [[projects/verl-source-reading/05-advantage-and-policy-loss|05 Advantage and Policy Loss]]：把 GRPO 与 PPO 公式映射到实际 tensor、mask 和 reduction。

完成本阶段后，应能不看图复述一次 update，并回答：

- 一个 prompt 为什么会产生 `rollout.n` 个 sessions；
- reward、old log-prob、reference log-prob 和 advantage 分别在哪里生成；
- 哪些 token 进入 policy loss，padding 和 observation 为什么必须被 mask；
- actor 更新后，下一批 rollout 何时开始使用新权重。

## 阶段二：拆开控制面与执行面

第二阶段保持算法和数据不变，只替换观察角度：controller 负责组织训练语义，Ray 负责远程调度，FSDP 或 Megatron 负责设备内外的模型计算。

```text
trainer request
  -> WorkerGroup dispatch
  -> Ray actor invocation
  -> TrainingWorker contract
  -> FSDP / Megatron engine
  -> local micro-batches and collectives
  -> collected metrics or tensors
```

依次阅读：

1. [[projects/verl-source-reading/06-single-controller-and-ray|06 Single Controller and Ray]]：区分 Ray actor、logical role 和 distributed rank。
2. [[projects/verl-source-reading/07-model-engine-and-backend|07 Model Engine and Backend]]：用同一个 actor update 对比 FSDP 与 Megatron。
3. [[projects/verl-source-reading/08-hybrid-engine-and-weight-sync|08 Hybrid Engine and Weight Sync]]：追踪 training layout 到 rollout layout 的参数转换。

这一阶段最重要的习惯，是每遇到一个 batch size 或 parallel size 都写出它的作用域。至少区分 dataset batch、prompt group、trajectory batch、global actor batch、PPO mini-batch、micro-batch、actor DP、rollout TP 和 world size。

## 阶段三：只增加一个新变量

同步闭环和执行层稳定后，再进入 PPO、Agentic RL、OPD/MOPD 与异步训练。每次只增加一个主要变量，避免把算法差异、轨迹差异和调度差异混在一起；完整的十轮专题顺序仍以 roadmap 为准。

| 扩展 | 保持不变 | 新增变量 | 对应笔记 |
| --- | --- | --- | --- |
| GRPO -> PPO | rollout、actor backend | critic、value、GAE、value loss | [[projects/verl-source-reading/09-ppo-and-grpo-codepath|09 PPO and GRPO Codepath]] |
| single-turn -> tool agent | trainer、policy update | turn state、observation、termination、trajectory mask | [[projects/verl-source-reading/11-agentic-rl-extension|11 Agentic RL Extension]] |
| OPD -> MOPD | student rollout、teacher log-prob、distillation loss | routing key、多个 teacher、teacher resource pool | [[projects/verl-source-reading/12-on-policy-distillation-and-mopd|12 On-Policy Distillation and MOPD]] |
| sync -> async | reward、advantage 基本语义 | policy version、staleness、drop/refill、correction | [[projects/verl-source-reading/10-async-trainer-and-off-policy|10 Async Trainer and Off-policy]] |

SFT 是辅助对照，不是 RL 主线的前置阶段。[[projects/verl-source-reading/verl-sft-training-path|verl SFT Training Path]] 用来观察相同 dataset、worker 和 engine 在没有 rollout、reward、advantage 时如何工作。

## 源码阅读方法

### 同时维护四类 ledger

阅读一个函数时，不只记录调用关系，同时更新四类表。它们对应 source-code-reading skill 中的 control、configuration、data、tensor、topology 和 state/version 追踪要求：

```text
control/config: caller -> callee -> dispatch -> worker
                default -> override -> resolved value -> runtime object
field/data:     producer -> field -> representation -> consumer -> cleanup
tensor/topology shape -> dtype -> device -> mask -> layout -> process group
state/version:  owner -> creation -> step/version -> mutation -> sync -> recovery
```

仅有 call graph 不足以解释训练正确性。很多问题来自字段仍存在但语义已经过期，例如 trajectory 对应旧 policy version，或者 observation token 被错误纳入 loss。

### 从不变量定位问题

比逐行记代码更有效的是先写不变量：

- 同一个 GRPO group 必须能追溯到同一个 prompt；
- policy loss 只聚合有效的 policy action token；
- sync update 使用的 trajectory 应能追溯到生成时的 step / version metadata；如果当前实现只有 `global_steps` 而没有独立 policy-version 字段，应明确记录这一限制；
- global batch 经过 dispatch 后，样本身份和 mask 语义不能改变；
- weight sync 前后的 rollout 权重状态应能通过 checkpoint、step 或其他可观察 metadata 区分；如果源码没有显式 version counter，不能把它写成已验证的运行事实。

当 trace 违反不变量时，再沿 producer、dispatch、consumer 和 cleanup 回溯。

### 区分源码事实与运行事实

静态源码可以确认注册关系、字段写入位置、默认分支和理论 shape；它不能替代真实环境中的 dtype、device、显存峰值、通信耗时、Ray placement 和 backend performance。正文中的结论应按三类证据标记：**Source fact** 直接来自固定 revision，**Tested behavior** 来自可复现运行或 trace，**Engineering interpretation** 是基于前两者的解释。运行结果应回填到 [[projects/verl-source-reading/labs/|verl Source Reading Labs]]。

## 最小实践顺序

实践不应直接从完整多机训练开始。推荐按证据成本递增：

1. 用 CPU unit test 验证 advantage、mask、loss reduction 和 custom reward 的纯函数行为；
2. 输出 resolved Hydra config，确认关键 override 的最终值和 runtime consumer；
3. 用最小 GPU run 记录一个 prompt group 的 uid、sessions、fields、shape、device 和 step / version metadata；
4. 在控制变量下切换 GRPO/PPO、FSDP/Megatron 或 sync/async，每次只比较一个维度；
5. 最后运行 multi-turn tool trajectory，审计 action/observation mask、termination 和 reward。

MOPD 的最小实践应放在 multi-turn agent 之前或并行进行：先用两个数据域验证 sample-level teacher routing，再把 teacher fields 与 student loss 对齐。对应实验见 [[projects/verl-source-reading/labs/05-multi-teacher-opd|Lab 05: Multi-Teacher OPD]]。

具体记录模板见 [[projects/verl-source-reading/labs/01-trace-config-and-batch|Lab 01: Trace Config and Batch]]。实验输出至少应包含 revision、环境、resolved config、命令、关键 trace、失败条件和结论；没有实际执行的预期结果不能标为 tested behavior。

## 学习完成检查

完成主线后，尝试在不查笔记的情况下解释以下场景：

1. 在 `train_batch_size` 等配置保持不变时，`rollout.n` 从 4 改为 8，会改变哪些 trajectory batch 和计算，哪些 training / rollout parallel group 不应仅因这一项而改变？
2. `use_kl_loss` 与 `use_kl_in_reward` 分别把 KL 放到哪条路径？
3. 为什么 rollout TP=2 不代表 actor DP 减半？
4. PPO 相比 GRPO 新增了哪些模型角色、字段和 update？
5. 在当前 AgentLoop 的 mask contract 下，tool observation 为什么可以进入下一轮上下文，却不应产生 policy gradient？如果自定义 AgentLoop 改变了 mask，应该重新验证什么？
6. async trainer 中 old policy、rollout policy 和 current actor 为什么可能不是同一版本？
7. loss 突然异常时，如何判断问题来自 reward、mask、advantage、dispatch、backend 还是 weight sync？
8. MOPD 中 `teacher_key`、teacher 配置里的 `key` 和样本的 `data_source` 分别承担什么作用？

如果回答只能停留在算法定义，而不能指出对应字段、调用层和可观察证据，说明还没有完成从“理解 RL”到“理解 verl 系统”的转换。
