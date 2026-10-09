---
title: Lab 05 - Multi-Teacher OPD
created: 2026-10-09
published: 2026-10-09
modified: 2026-10-09
type: project
status: seed
tags:
  - projects
  - verl
  - experiment
  - on-policy-distillation
  - mopd
---

## Objective

验证当前 revision 中 MOPD 的最小闭环：混合数据集中的 `data_source` 如何路由到不同 teacher，teacher log-prob 如何写回 rollout output，以及 student 如何在同一批数据上计算 distillation loss。

本 lab 是运行设计，尚未标记为 Tested behavior。它不以最终 benchmark 为目标，而是先确认 teacher routing、字段 shape、资源池切分和错误边界。

## Fixed Scope

- Repository：[process-cxr/verl-upstream](https://github.com/process-cxr/verl-upstream)
- Revision：`5b79827b04cee6e4b7b5ff737e047f1d72d50433`
- Entry：[run_qwen3_8b_mopd_fsdp.sh](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/examples/on_policy_distillation_trainer/run_qwen3_8b_mopd_fsdp.sh)
- Student：Qwen3-VL-8B-Instruct（或同 tokenizer / vocabulary 的小型替代模型）
- Teachers：一个文本 teacher 和一个视觉语言 teacher
- Student engine：FSDP；teacher inference：vLLM
- Routing field：`data_source`

## Trace Stages

```text
resolved config
  -> teacher_models and teacher resource pool
  -> mixed dataset rows with data_source
  -> student rollout
  -> per-sample teacher routing
  -> teacher_ids / teacher_logprobs
  -> distillation_ppo_loss
  -> student metrics and cleanup
```

## Required Observations

记录每个样本的：

```text
data_source
resolved teacher key
teacher model path
teacher replica / server handle
prompt length and response length
teacher_ids shape
teacher_logprobs shape
loss_mode and use_policy_gradient
distillation loss
teacher pool world size
```

需要特别核对：

- `openai/gsm8k` 是否只路由到文本 teacher；
- `hiyouga/geometry3k` 是否只路由到视觉语言 teacher；
- 未配置的 routing key 是否明确报错；
- teacher pool 总容量是否等于各 teacher replica footprint 之和；
- `use_task_rewards=False` 时 student loss 是否只保留 distillation loss；
- validation 路径是否不会误触发 teacher log-prob 请求。

## Negative Cases

1. 删除一个 teacher 的 `key`，确认配置验证失败；
2. 在数据中加入未知 `data_source`，确认 routing 失败而不是静默使用错误 teacher；
3. 保留默认 `teacher_model` 并同时增加命名 teacher，确认配置解析会触发文档所述的 pop 行为；
4. 故意让 teacher pool GPU 数与 replica footprint 不一致，确认初始化阶段拒绝运行；
5. 将 teacher 的 `max_model_len` 设得小于 student prompt + response + 1，确认长度校验失败。

## Acceptance Criteria

- 两个数据域都能得到对应 teacher 的 log-prob；
- `teacher_ids` 和 `teacher_logprobs` 能进入 student training batch；
- routing、resource pool 和长度约束的失败模式可复现；
- 输出中能够区分 Source fact、Tested behavior 和 Engineering interpretation；
- 结果回填 [[projects/verl-source-reading/12-on-policy-distillation-and-mopd|12 On-Policy Distillation and MOPD]]。

## Source Anchors

- [MOPD FSDP script](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/examples/on_policy_distillation_trainer/run_qwen3_8b_mopd_fsdp.sh)
- [teacher routing](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/experimental/teacher_loop/teacher_manager.py#L100)
- [teacher output fields](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/experimental/agent_loop/agent_loop.py#L1017)
- [distillation loss](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/distillation/losses.py#L165)

