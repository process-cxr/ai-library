---
title: "Training"
created: 2026-02-21
published: 2026-02-21
modified: 2026-05-31
---

大模型训练全流程，从预训练、中训练到后训练，并包含数据工程、训练优化、分布式训练和 scaling 等横切模块。

## Main Stages

- [[training/pretraining/|预训练]] — 从大规模通用语料中学习基础语言建模能力。
- [[training/mid-training/|中训练]] — continued pretraining、领域适配、长上下文和能力注入。
- [[training/post-training/|后训练]] — SFT、RLHF、DPO/GRPO、拒绝采样和蒸馏。
- [[training/pretraining/reinforcement-pretraining|Reinforcement Pretraining]] — 把 RL-style objective 前移到预训练或 continued pretraining 阶段。

## Cross-cutting Modules

- [[training/data-engineering/|训练数据工程]] — 数据清洗、去重、过滤、packing、distributed dataloader 和合成数据。
- [[training/optimization/|训练优化工程]] — 显存估算、混合精度、checkpointing、optimizer state、checkpoint sharding 和稳定性。
- [[training/distributed-training/|分布式训练]] — DP、TP、PP、SP、CP、ZeRO、FSDP、Megatron。
- [[training/scaling/|Scaling]] — scaling law、模型/数据/算力配比和训练预算。
