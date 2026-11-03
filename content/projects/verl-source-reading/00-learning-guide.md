---
title: 00 verl Learning Guide
created: 2026-10-08
published: 2026-10-08
modified: 2026-11-03
type: project
status: growing
aliases:
  - projects/verl-source-reading/roadmap
tags:
  - projects
  - source-reading
  - verl
  - learning-guide
  - post-training
---

这套笔记从一轮同步 GRPO 训练开始。数据集提供 128 道题，每道题生成 4 条回答，经过评分和优势计算，再分到两台机器的 8 张 GPU 更新 actor。更新后的权重回到生成服务，开始下一轮。

[[projects/verl-source-reading/01-sync-grpo-run-overview|01 Sync GRPO Run Overview]] 给出固定配置基线和贯穿全文的数学题 P0，02 将配置落实到启动命令。后文持续追踪它的四条回答：先看消息和 token 怎样生成，再看 reward、log-prob 和 advantage 怎样附着到同一条 trajectory，最后看 worker 怎样计算梯度。源码版本和项目介绍见 [[projects/verl-source-reading/|项目首页]]。

## 先沿一次训练读完 01–08

| 章节                                                                                                   | 从哪里接手                          | 在哪里结束                                              |
| ------------------------------------------------------------------------------------------------------ | ----------------------------------- | ------------------------------------------------------- |
| [[projects/verl-source-reading/01-sync-grpo-run-overview\|01 Sync GRPO Run Overview]]                  | 固定训练配置与 P0                   | 一轮采样、四次参数更新、权重同步的全貌                  |
| [[projects/verl-source-reading/02-config-and-hydra\|02 Launch and First Batch]]                        | 一条启动命令                        | 128 行 prompt 获得 uid、登记状态并提交给 AgentLoop      |
| [[projects/verl-source-reading/03-data-protocol-and-batch-lifecycle\|03 From Prompts to Trajectories]] | AgentLoop 收到原始消息              | 四个 session 各自生成 AgentLoopOutput                   |
| [[projects/verl-source-reading/04-rollout-reward-and-agent-loop\|04 Reward and Training Batch]]        | 待评分的 AgentLoopOutput            | reward 写回，收齐 trajectory keys，完成训练前的样本重排 |
| [[projects/verl-source-reading/05-advantage-and-policy-loss\|05 Advantage and Policy Loss]]            | 带 reward 的同一批 trajectory       | old/ref log-prob、advantage、policy loss 与一次参数更新 |
| [[projects/verl-source-reading/06-single-controller-and-ray\|06 Single Controller and Ray]]            | controller 发起 compute/update 调用 | keys 分发、worker 取数、FSDP 计算和结果回写             |
| [[projects/verl-source-reading/07-model-engine-and-backend\|07 Model Engine and Backend]]              | rank 0 的本地 mini-batch            | packed attention、参数分片、梯度归约和 optimizer        |
| [[projects/verl-source-reading/08-hybrid-engine-and-weight-sync\|08 Hybrid Engine and Weight Sync]]    | 更新后留在训练 worker 的 θ4         | 参数传入 vLLM、清理旧缓存、下一轮使用新权重             |

01 先走完整轮；02–05 按执行阶段展开。06 回到其中的远程调用，07 继续进入模型执行，08 把更新后的参数交回生成服务。第一次阅读时，可以在每篇旁边打开它引用的源码函数，把文章中的输入输出与代码中的变量对上。

## 在整批、一组和一个 token 之间切换

读 batch 分配时看整批：128 个 prompt 展开为 512 条 trajectory，进入 8 个训练 rank。读 GRPO 时只看 P0 的四条回答；读 log-prob 或 mask 时，再看其中一条回答的某个 token。

文章会标明这些不同尺度。为手算展示的 P0 四行矩阵，不表示真实训练一定将它们放进同一个 micro-batch；短 token 序列也不是示例中文文本的实际分词结果。TQ 存储的变长结构、算法临时使用的 padded 矩阵和模型使用的 packed 输入，会在相应的转换代码旁分别说明。

读到一个函数时，可以先问：P0 现在有哪些字段，保存在谁那里？执行这段代码后多了什么、改变了什么？下一段为什么需要这个结果？如果只记得类名而说不出数据变化，就回到最近的输入输出示例。

## 主线闭合后再比较扩展

第一次阅读 07、08 时，先跟完 FSDP 与 vLLM 的固定路径；后端对照、no-sync 优化和其他权重传输方式可以在第二遍比较。这样可以先看到本例从数据到参数、再到下一轮生成的完整循环。

算法和轨迹可以沿另一个方向展开：

- [[projects/verl-source-reading/09-ppo-and-grpo-codepath|09 PPO and GRPO Codepath]]：在同一 trainer 中加入 critic，比较 GAE 和组内 advantage。
- [[projects/verl-source-reading/11-agentic-rl-extension|11 Agentic RL Extension]]：把单轮回答换成工具交互，观察消息、token mask、终止条件和 reward 的变化。
- [[projects/verl-source-reading/10-async-trainer-and-off-policy|10 Async Trainer and Off-policy]]：允许生成与训练重叠，追踪 trajectory 的参数版本和采样策略。
- [[projects/verl-source-reading/12-on-policy-distillation-and-mopd|12 On-Policy Distillation and MOPD]]：保留学生生成的数据，让额外的教师服务提供逐 token 监督。
- [[projects/verl-source-reading/verl-sft-training-path|SFT Training Path]]：用同一道题的标准答案直接训练，比较 SFT 与 RL 的数据来源、batch 和 loss。

教师资源池和 SFT batch 规则会在对应章节明确说明。它们是在基线之上改变条件，不能直接把 GRPO 的“乘 rollout.n”规则套用过去。

## 遇到具体计算时补哪些基础

| 阅读位置                  | 对应基础                                                                                     |
| ------------------------- | -------------------------------------------------------------------------------------------- |
| 组内奖励、advantage       | [[training/post-training/grpo\|GRPO]] 的相对评分和 baseline                                  |
| log-prob、ratio、clipping | [[training/post-training/ppo\|PPO]] 的策略梯度目标                                           |
| nested、padding、mask     | 张量索引、broadcast、序列长度、下一 token 预测                                               |
| rank 与通信               | [[training/distributed-training/torch-distributed\|torch.distributed]] 的进程组与 collective |
| 参数分片                  | [[training/distributed-training/fsdp\|FSDP]] 的参数、梯度、optimizer state                   |
| 生成服务                  | [[inference/serving-systems/vllm\|vLLM]] 的请求调度和 KV cache                               |

不必先读完所有基础文章。比如 05 出现 `[4] → [4,1] → [4,5]` 时，配合具体矩阵理解 broadcast；06 出现八个 rank 的分发时，再补进程组和通信，会更容易把概念放回用途。

## 怎样确认自己已经理解

读完 01–08 后，试着不看正文解释 P0：它在哪里获得 uid，四个 session 怎样创建，0 分回答的有效长度从哪里来，组内分数怎样覆盖 token，以及它的字段怎样进入某个 rank 的 micro-batch。再追问这次 backward 怎样改变训练参数、这些参数如何变成 vLLM 能使用的权重，下一轮为何不会继续沿用旧 KV cache。

还应该能区分三个时钟：dataset 的训练 epoch、同一批轨迹的 PPO epoch、每个 mini-batch 的 optimizer update。它们改变的是不同的状态，不能仅凭名字中的“epoch”或“step”视作同一件事。

需要实际观察时，进入 [[projects/verl-source-reading/labs/01-trace-config-and-batch|Lab 01]] 查看配置和字段追踪方法。正文中的源码解释与手算示例帮助形成预期，labs 记录实际设备、调用和张量结果，两者可以相互核对。
