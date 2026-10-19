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

这份指南帮助你进入 verl 的源码主线。它先说明需要哪些基础，再用一次训练运行介绍阅读方法，最后给出几个可以检验理解的问题。具体专题的顺序和代码入口见 [[projects/verl-source-reading/roadmap|verl Source Reading Roadmap]]。

项目的三页入口各有一个作用：[[projects/verl-source-reading/|项目首页]]介绍系统由什么组成，roadmap 安排阅读顺序，本页帮助你建立阅读视角并检查理解。

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

## 先看懂一次 Sync GRPO Update

先固定 `sync GRPO + FSDP + vLLM + rule-based reward` 这条路径，不要一开始同时切换算法、backend 和 trainer mode。

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

按下面的顺序阅读：

1. [[projects/verl-source-reading/01-sync-grpo-run-overview|01 Sync GRPO Run Overview]]：先获得完整时序，不在第一次阅读时下钻所有实现。
2. [[projects/verl-source-reading/02-config-and-hydra|02 Config and Hydra]]：把 shell override 映射到 resolved config 和 runtime consumer。
3. [[projects/verl-source-reading/03-data-protocol-and-batch-lifecycle|03 Data Protocol and Batch Lifecycle]]：追踪 uid、trajectory key、字段 producer / consumer 和 cleanup。
4. [[projects/verl-source-reading/04-rollout-reward-and-agent-loop|04 Rollout, Reward and AgentLoop]]：确认生成、tool interaction 和 reward 的边界。
5. [[projects/verl-source-reading/05-advantage-and-policy-loss|05 Advantage and Policy Loss]]：把 GRPO 与 PPO 公式映射到实际 tensor、mask 和 reduction。

读完这条主线后，应该能够不看图复述一次 update，并回答：

- 一个 prompt 为什么会产生 `rollout.n` 个 sessions；
- reward、old log-prob、reference log-prob 和 advantage 分别在哪里生成；
- 哪些 token 进入 policy loss，padding 和 observation 为什么必须被 mask；
- actor 更新后，下一批 rollout 何时开始使用新权重。

## 再看控制器如何调用训练后端

接下来保持算法和数据不变，只追踪一次 actor update 是怎样被执行的：controller 组织训练步骤，Ray 把调用送到远程 worker，FSDP 或 Megatron 在各自的进程组中完成模型计算。

```text
trainer request
  -> WorkerGroup dispatch
  -> Ray actor invocation
  -> TrainingWorker contract
  -> FSDP / Megatron engine
  -> local micro-batches and collectives
  -> collected metrics or tensors
```

对应的文章是：

1. [[projects/verl-source-reading/06-single-controller-and-ray|06 Single Controller and Ray]]：区分 Ray actor、logical role 和 distributed rank。
2. [[projects/verl-source-reading/07-model-engine-and-backend|07 Model Engine and Backend]]：用同一个 actor update 对比 FSDP 与 Megatron。
3. [[projects/verl-source-reading/08-hybrid-engine-and-weight-sync|08 Hybrid Engine and Weight Sync]]：追踪 training layout 到 rollout layout 的参数转换。

读到这里，需要开始区分不同参数的作用域：dataset batch、prompt group、trajectory batch、global actor batch、PPO mini-batch、micro-batch、actor DP、rollout TP 和 world size 并不是同一个层次的概念。

## 最后再看扩展路径

理解同步闭环和执行层之后，再进入 PPO、Agentic RL、OPD/MOPD 与异步训练。每次只引入一个主要变化，这样才能看清算法、轨迹和调度分别改变了什么；完整顺序以 roadmap 为准。

| 扩展 | 保持不变 | 新增变量 | 对应笔记 |
| --- | --- | --- | --- |
| GRPO -> PPO | rollout、actor backend | critic、value、GAE、value loss | [[projects/verl-source-reading/09-ppo-and-grpo-codepath|09 PPO and GRPO Codepath]] |
| single-turn -> tool agent | trainer、policy update | turn state、observation、termination、trajectory mask | [[projects/verl-source-reading/11-agentic-rl-extension|11 Agentic RL Extension]] |
| OPD -> MOPD | student rollout、teacher log-prob、distillation loss | routing key、多个 teacher、teacher resource pool | [[projects/verl-source-reading/12-on-policy-distillation-and-mopd|12 On-Policy Distillation and MOPD]] |
| sync -> async | reward、advantage 基本语义 | policy version、staleness、drop/refill、correction | [[projects/verl-source-reading/10-async-trainer-and-off-policy|10 Async Trainer and Off-policy]] |

SFT 是辅助对照，不是 RL 主线的前置阶段。[[projects/verl-source-reading/verl-sft-training-path|verl SFT Training Path]] 用来观察相同 dataset、worker 和 engine 在没有 rollout、reward、advantage 时如何工作。

## 怎样读这套源码

### 不要从目录开始

verl 的目录很大，从目录逐个文件读，容易记住很多名字，却不知道一次训练是怎样跑起来的。更有效的方式是先选一条可以闭合的运行路径，再沿着同一个问题向下追踪：一个 prompt 从哪里进来，谁生成 response，reward 写到哪里，最后哪一个 worker 更新参数。

读到一个函数时，可以顺手问下面几件事：

```text
这个调用由谁发起，又在哪个 worker 中执行？
配置文件里的值，最后变成了哪个运行时对象？
这个字段在哪里产生，经过什么表示转换，又在哪里被消费？
这个 tensor 的 shape、dtype、device、mask 和并行布局怎样变化？
这份状态由谁创建，什么时候修改，什么时候与另一个 runtime 同步？
```

这样读出来的不是一张孤立的调用图，而是一条从配置到数据、从数据到计算、再从计算回到参数同步的完整链路。很多训练问题并不是字段不存在，而是字段的语义已经过期，例如 trajectory 属于旧 policy version，或者 observation token 被错误地放进了 policy loss。

### 抓住几条不会轻易改变的关系

阅读时可以先记住几条关系，再用源码确认它们：

- 同一个 GRPO group 必须能追溯到同一个 prompt；
- policy loss 只聚合有效的 policy action token；
- sync update 使用的 trajectory 应能追溯到生成时的 step / version metadata；如果当前实现只有 `global_steps` 而没有独立 policy-version 字段，应明确记录这一限制；
- global batch 经过 dispatch 后，样本身份和 mask 语义不能改变；
- weight sync 前后的 rollout 权重状态应能通过 checkpoint、step 或其他可观察 metadata 区分；如果源码没有显式 version counter，不能把它写成已验证的运行事实。

如果某处违反这些关系，再回头检查它的 producer、dispatch、consumer 和 cleanup。

### 分清代码写明的内容和实际跑出来的内容

源码能够告诉我们调用关系、字段写入位置、默认分支和理论 shape；真实运行才会告诉我们 dtype、device、显存峰值、通信耗时、Ray placement 和 backend performance。阅读笔记中需要把这两类内容分开：代码明确写出的结论，实际运行观察到的结果，以及根据前两者做出的工程解释，不要混成一句话。运行观察会记录在 [[projects/verl-source-reading/labs/|verl Source Reading Labs]] 中。

## 如果要实际跑起来

不要一开始就启动完整多机训练。可以按下面的顺序逐步增加复杂度：

1. 先用 CPU 测试熟悉 advantage、mask、loss reduction 和 custom reward；
2. 打印 resolved Hydra config，确认命令行参数最后变成了什么；
3. 再用最小 GPU run 看一个 prompt group 如何生成 sessions、字段和 tensor；
4. 最后才比较 GRPO/PPO、FSDP/Megatron、sync/async 和 multi-turn tool trajectory。

MOPD 可以在 multi-turn agent 之前学习：先用两个数据域看清 sample-level teacher routing，再把 teacher fields 和 student loss 对上。对应实验见 [[projects/verl-source-reading/labs/05-multi-teacher-opd|Lab 05: Multi-Teacher OPD]]。

第一个追踪实验见 [[projects/verl-source-reading/labs/01-trace-config-and-batch|Lab 01: Trace Config and Batch]]。它的重点是看清字段和控制流，而不是先追求 benchmark 分数。

## 读完之后试着回答

读完主线后，可以先不看笔记，自己解释下面这些问题：

1. 在 `train_batch_size` 等配置保持不变时，`rollout.n` 从 4 改为 8，会改变哪些 trajectory batch 和计算，哪些 training / rollout parallel group 不应仅因这一项而改变？
2. `use_kl_loss` 与 `use_kl_in_reward` 分别把 KL 放到哪条路径？
3. 为什么 rollout TP=2 不代表 actor DP 减半？
4. PPO 相比 GRPO 新增了哪些模型角色、字段和 update？
5. 在当前 AgentLoop 的 mask contract 下，tool observation 为什么可以进入下一轮上下文，却不应产生 policy gradient？如果自定义 AgentLoop 改变了 mask，应该重新验证什么？
6. async trainer 中 old policy、rollout policy 和 current actor 为什么可能不是同一版本？
7. loss 突然异常时，如何判断问题来自 reward、mask、advantage、dispatch、backend 还是 weight sync？
8. MOPD 中 `teacher_key`、teacher 配置里的 `key` 和样本的 `data_source` 分别承担什么作用？

如果只能复述算法定义，却说不清对应字段、调用层和可以观察到的现象，就还没有真正把 RL 算法和 verl 的实现联系起来。
