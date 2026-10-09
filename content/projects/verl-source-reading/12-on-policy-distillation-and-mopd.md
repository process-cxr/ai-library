---
title: 12 On-Policy Distillation and MOPD
created: 2026-10-09
published: 2026-10-09
modified: 2026-10-09
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

本页研究当前 revision 中的 On-Policy Distillation（OPD）和 Multi-Teacher On-Policy Distillation（MOPD）。它们把一个额外的 teacher inference runtime 接入 student 的 rollout 和 actor update：student 先在自己的状态分布上生成 response，teacher 再对同一条 student-induced state sequence 提供 token-level log-probability，最后由 student 计算 distillation loss。

本页对应的固定源码是 [`5b79827b04cee6e4b7b5ff737e047f1d72d50433`](https://github.com/process-cxr/verl-upstream/tree/5b79827b04cee6e4b7b5ff737e047f1d72d50433)。当前快照有明确的 MOPD 配置、teacher manager、FSDP / VeOmni 示例和 separate-async 多教师端到端脚本；本地尚未执行完整 GPU 训练，因此运行性能和模型效果仍属于待验证内容。

## 它在 verl 主线中的位置

普通 SFT 使用预先给定的目标输出；这些 target 可以来自人工标注、规则构造或 teacher 生成。标准 KD 通常也沿着固定的 teacher 轨迹提供监督。OPD 先让 student 自己 rollout，再让 teacher 评价 student 实际访问到的 token 状态，因此训练状态与 student 推理状态保持一致，同时获得比稀疏 outcome reward 更密集的 token-level supervision。

```text
student prompt
  -> student rollout / agent trajectory
  -> prompt + response token states
  -> teacher log-probability inference
  -> teacher_ids + teacher_logprobs
  -> distillation_ppo_loss
  -> student actor update
```

MOPD 在 OPD 上增加的是样本级 teacher routing：混合数据集中的每条样本带一个 routing field，例如 `data_source`；该值选择一个领域教师。它把多个专门教师的能力蒸馏到一个 student 中，但当前实现不是多个教师对同一条样本做 ensemble，也不是 token 级动态混合。

## 运行时角色与边界

| 角色 | 作用 | 是否更新参数 |
| --- | --- | --- |
| Student actor / rollout | 生成 response 或 agent trajectory，并完成最终 actor update | 是 |
| Teacher inference server | 对 student 已生成的 `prompt + response` 计算 log-probs 或 top-k log-probs | 否；作为推理服务运行 |
| AgentLoop | 在每个 rollout 完成后触发对应 teacher 请求，并把结果写入 output | 不负责参数更新 |
| MultiTeacherModelManager | 为每个 teacher 创建 inference replicas，切分 teacher resource pool | 不更新模型参数 |
| AsyncTeacherLLMServerManager | 根据样本 routing key 选择 teacher client，并发起 log-prob 请求 | 不更新模型参数 |
| `distillation_ppo_loss` | 将 teacher signal 与可选 task reward 组合成 student loss | 作用于 student 梯度 |

在 V1 trainer 中，teacher resource pool 与 student actor / rollout pool 分开配置。`PPOTrainer` 初始化 `MultiTeacherModelManager`，再把各 teacher 的 client 传入 AgentLoop；teacher 的 log-prob 计算发生在 rollout 后处理阶段，而不是等一个完整 batch 统一生成 teacher response。

源码入口：

- [`PPOTrainer` 初始化 teacher manager](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/ppo/v1/trainer_base.py#L340)
- [`MultiTeacherModelManager`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/experimental/teacher_loop/teacher_model.py#L164)
- [`AgentLoopWorker._compute_teacher_logprobs`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/experimental/agent_loop/agent_loop.py#L1001)
- [`AsyncTeacherLLMServerManager.compute_teacher_logprobs_single`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/experimental/teacher_loop/teacher_manager.py#L116)

## MOPD 的 teacher routing

默认 routing 配置是：

```yaml
distillation:
  enabled: true
  teacher_key: data_source
  teacher_models:
    gsm8k:
      key: openai/gsm8k
      model_path: Qwen/Qwen3-32B
    geo3k:
      key: hiyouga/geometry3k
      model_path: Qwen/Qwen3-VL-32B-Instruct
```

运行时的关键关系是：

```text
sample[data_source]
  -> routing_key
  -> teacher_models[routing_key]
  -> teacher client / replica
  -> teacher_logprobs
```

`AsyncTeacherLLMServerManager._resolve_teacher_key` 对单教师和多教师分别处理：单教师时忽略样本 routing key；多教师时 routing key 必须存在，并且必须精确匹配某个 teacher 的 `key`，否则直接报错。配置多个命名教师时，默认的 `teacher_model` 条目会被解析逻辑移除，因此第一个教师也应使用显式名称，例如 `teacher_model1`。

源码入口：

- [`distillation.yaml`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/config/distillation/distillation.yaml#L64)
- [`DistillationConfig._resolve_teacher_models`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/config/distillation.py#L289)
- [`teacher_key` routing](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/experimental/teacher_loop/teacher_manager.py#L100)

## 一次 teacher log-prob 请求

teacher 不重新生成一条 teacher response，而是对 student 已经走过的 token prefix 做 prompt log-prob computation：

```text
student prompt_ids + response_ids
  -> teacher inference server
  -> teacher_ids
  -> teacher_logprobs
  -> AgentLoopOutput.extra_fields
  -> DataProto / TensorDict
```

当 loss mode 需要 top-k 时，teacher 返回每个位置的 top-k log-probs；否则返回 sampled token 的单点 log-prob。teacher 请求会携带多模态输入和 processor 参数，因此当前 MOPD 示例可以让文本样本和视觉语言样本路由到不同教师。

teacher resource pool 的容量必须满足：

```text
distillation.n_gpus_per_node × distillation.nnodes
  = Σ teacher.num_replicas × teacher.per_replica_world_size
```

其中单个 teacher replica 的并行规模由 inference-time 的 tensor、data 和 pipeline parallel size 决定。`MultiTeacherModelManager` 按每个 teacher 的 world size 切分总资源池，并为每个 teacher 建立独立的 server handles 和 load-balancer handle。

## Distillation loss 与 task reward

当前实现注册了两类主要 distillation loss：

| 类型 | 典型配置 | 训练含义 |
| --- | --- | --- |
| GKD OPD | `loss_mode=forward_kl_topk`，`use_policy_gradient=false` | 用 teacher top-k 分布近似 forward KL，直接反向传播 distillation loss |
| PG OPD | `loss_mode=k1` 等，`use_policy_gradient=true` | 将单样本 reverse-KL estimator 转为 advantage，走 PPO-style policy-gradient loss |

最终 student loss 由配置决定：

```text
use_task_rewards = true:
  policy_loss + distillation_loss_coef × distillation_loss

use_task_rewards = false:
  只保留 distillation_loss
```

当前快照中 `use_policy_gradient=true` 时只支持 `policy_loss_mode=vanilla`。`forward_kl_topk` 搭配 policy-gradient 会丢失非 sampled token 的分布信息，因此文档明确将这两个选项视为不匹配的组合。

源码入口：

- [`distillation_ppo_loss`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/distillation/losses.py#L165)
- [`compute_forward_kl_topk` FSDP 实现](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/distillation/fsdp/losses.py#L77)
- [`compute_forward_kl_topk` Megatron 实现](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/distillation/megatron/losses.py#L77)

## 当前快照的适配范围

当前仓库直接提供：

- [`run_qwen3_8b_mopd_fsdp.sh`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/examples/on_policy_distillation_trainer/run_qwen3_8b_mopd_fsdp.sh)：GSM8K 文本教师 + Geo3K 视觉语言教师，vLLM teacher inference，FSDP student training；
- [`run_qwen3_8b_mopd_veomni.sh`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/examples/on_policy_distillation_trainer/run_qwen3_8b_mopd_veomni.sh)：同一多教师 routing，VeOmni student engine；
- [`run_v1_separate_async_opd.sh`](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/tests/special_e2e/run_v1_separate_async_opd.sh)：包含多教师 OPD 的 V1 separate-async 端到端脚本。

仓库还提供单教师 Megatron OPD 示例，以及 distillation loss 的 Megatron 实现；但当前快照没有一个专门命名为 MOPD Megatron 的 canonical script，因此多教师与 Megatron 的组合应继续以 resolved config 和实际运行作为验证依据。

## 阅读时必须保留的不变量

- routing field 的值必须和 teacher 配置中的 `key` 完全一致；
- 每个样本在当前实现中选择一个 teacher，不能把样本级 routing 解释为 token 级 mixture；
- teacher resource pool 总容量必须等于所有 teacher replica 的 GPU 需求之和；
- teacher 只提供 inference-time log-prob，student 才执行反向传播和 optimizer update；
- teacher 必须能处理 student 的 tokenizer / vocabulary 和对应的多模态输入；
- teacher log-prob 的 shape、top-k 设置、`max_model_len` 和 student prompt/response 长度必须相互兼容；
- teacher log-prob 只在训练 rollout 后处理路径中计算；当前 `AgentLoopWorker` 的 validation 路径不会触发 teacher 请求；
- MOPD 的支持是源码和配置层面的事实，完整吞吐、显存、稳定性和效果仍需要 Lab 05 验证。

## 相关知识与实验

- [[projects/verl-source-reading/05-advantage-and-policy-loss|05 Advantage and Policy Loss]]：理解 policy loss、mask 和 token-level signal；
- [[projects/verl-source-reading/06-single-controller-and-ray|06 Single Controller and Ray]]：理解 teacher resource pool 和 client dispatch；
- [[projects/verl-source-reading/07-model-engine-and-backend|07 Model Engine and Backend]]：理解 FSDP / Megatron 如何消费 distillation fields；
- [[training/post-training/on-policy-kd|On-Policy Distillation]]：理解 OPD 的通用目标、状态分布和 loss 变体；
- [[sources/papers/2026-mopd-multi-teacher-on-policy-distillation-for-capability-integration|MOPD: Multi-Teacher On-Policy Distillation for Capability Integration]]：查看 MOPD 的方法背景与论文证据；
- [[projects/verl-source-reading/labs/05-multi-teacher-opd|Lab 05: Multi-Teacher OPD]]：验证 routing、teacher fields 和资源池约束；
- [verl OPD documentation](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/docs/algo/opd.md)：查看完整 loss 和配置说明。
