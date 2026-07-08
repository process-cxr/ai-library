# training

`training/` 负责整理大模型训练全流程。当前采用二级平铺结构：主线按训练阶段组织，横切模块按工程与规模问题组织。

## 当前模块职责

- 解释模型从 base model 到 assistant 的训练阶段：预训练、中训练、后训练。
- 记录训练数据工程、训练优化、分布式训练和 scaling 等横切主题。
- 连接基础数学、模型架构、推理部署和应用评测之间的因果关系。

## 直接子项

- `index.md` — Training 网页入口与模块索引。
- `pretraining/` — 预训练阶段，包括训练目标、数据混合、tokenizer 和 compute optimal。
- `mid-training/` — 中训练阶段，包括 continued pretraining、领域适配、长上下文、annealing 和能力注入。
- `post-training/` — 后训练阶段，包括 SFT、RLHF、DPO/GRPO、拒绝采样和 KD。
- `data-engineering/` — 训练数据清洗、去重、过滤、packing、distributed dataloader 和合成数据。
- `optimization/` — 训练工程优化，如混合精度、activation checkpointing、optimizer state、checkpoint sharding 和稳定性。
- `distributed-training/` — DP、TP、PP、SP、CP、ZeRO、FSDP、Megatron 等分布式训练方法。
- `scaling/` — Scaling Law、模型/数据/算力配比和训练预算。

## 文件契约

- 二级目录的 `index.md` 负责该训练模块入口和直接子项索引。
- 具体方法笔记可以先以 TODO 占位，后续再按知识写作规范扩展。
- 阶段类目录按训练生命周期组织，横切目录按工程问题组织。
- 后训练蒸馏相关内容平铺在 `post-training/` 下，不再额外拆三级目录。

## 边界

- 数学、概率、信息论和优化原理放入 `../fundamentals/`。
- 模型结构本身放入 `../architecture/`。
- 线上推理、加速和压缩放入 `../inference/`。
- 应用层评测和产品工作流放入 `../application/`。

## Direct-Call 信息

- 目录入口链接：`[[training/|Training]]`。
- 阶段入口示例：`[[training/pretraining/|预训练]]`、`[[training/post-training/|后训练]]`。
- 后训练 KD 示例：`[[training/post-training/offline-kd|Offline KD]]`、`[[training/post-training/on-policy-kd|On-policy KD]]`。
