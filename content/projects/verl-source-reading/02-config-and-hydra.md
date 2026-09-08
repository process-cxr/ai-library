---
title: 02 Config and Hydra
created: 2026-09-02
published: 2026-09-02
modified: 2026-09-02
type: project
status: seed
tags:
  - projects
  - source-reading
  - verl
  - hydra
  - configuration
---

本页研究 verl 如何用 Hydra config group、OmegaConf interpolation 和 shell overrides 构造一条训练任务。目标是能从任意 example script 反推出最终 runtime configuration，而不是只抄参数说明。

## 本页问题

1. `ppo_trainer.yaml` 如何组合 actor、critic、reference、rollout、reward 和 engine configs？
2. example script 中的 override 覆盖了哪一层默认值？
3. `_target_`、`${oc.select:...}` 与 dataclass conversion 分别在何时生效？
4. GRPO/PPO、FSDP/Megatron、vLLM/SGLang 的选择由哪些最小参数决定？
5. config validation 如何阻止不一致的 batch size、reference policy 和 critic 配置？

## Source Anchors

- [ppo_trainer.yaml](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/config/ppo_trainer.yaml)
- [actor config](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/config/actor/actor.yaml)
- [rollout config](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/config/rollout/rollout.yaml)
- [algorithm dataclasses](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/trainer/config/algorithm.py)
- [worker configs](https://github.com/process-cxr/verl-upstream/tree/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/workers/config)
- [config validation](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/verl/utils/config.py)
- [print_cfg.py](https://github.com/process-cxr/verl-upstream/blob/5b79827b04cee6e4b7b5ff737e047f1d72d50433/scripts/print_cfg.py)

## Configuration Ledger

完整笔记需要建立以下映射：

```text
shell override
  -> resolved OmegaConf path
  -> dataclass field
  -> runtime consumer
  -> observable training behavior
```

重点参数包括 `adv_estimator`、`rollout.n`、`train_batch_size`、`ppo_mini_batch_size`、`use_kl_loss`、`use_kl_in_reward`、actor strategy、rollout backend、trainer mode 和 off-policy threshold。

## Completion Criteria

- 保存一份最小 GRPO resolved config；
- 标出 algorithm、resource placement、batching 和 performance knobs；
- 解释 generated config 与 source config 的关系；
- 记录至少三类常见配置错误及其 validation path。

