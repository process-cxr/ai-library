# architecture

`architecture/` 负责整理大模型架构与核心模块，回答模型“由什么组成、为什么这样设计、不同架构如何取舍”。当前采用二级目录结构：基础机制、模型家族和扩展架构分开维护。

## 当前模块职责

- 整理 Transformer、Attention、位置编码等基础架构机制。
- 记录 GPT、LLaMA、Qwen、DeepSeek、Mistral、Gemma 等模型家族作为架构组合案例。
- 承接 `../fundamentals/` 中的数学与神经网络基础，并连接训练、推理和应用能力。

## 直接子项

- `index.md` — Architecture 网页入口与模块索引。
- `transformer/` — Transformer 基础结构，包括 decoder-only、encoder-decoder、FFN、归一化和残差。
- `attention/` — Attention 机制及现代变体，包括 MHA、MQA、GQA、滑动窗口注意力。
- `positional-encoding/` — 位置编码，包括绝对位置、正弦位置、RoPE、ALiBi、YaRN。
- `model-families/` — 主流模型家族案例，包括 GPT、LLaMA、Qwen、DeepSeek、Mistral、Gemma、Claude。
- `sparse-and-efficient/` — 稀疏与高效架构，包括 MoE、Mamba、SSM、Linear Attention。
- `multimodal/` — 多模态架构，包括 VLM、CLIP、LLaVA、Qwen-VL、projector。

## 文件契约

- 二级目录的 `index.md` 负责该架构模块入口和直接子项索引。
- 具体架构笔记可以先以 TODO 占位，后续再按知识写作规范扩展。
- 模型家族笔记应区分公开事实、论文/技术报告信息和推测内容。
- 基础机制笔记应优先说明结构、输入输出、复杂度、适用边界和与主流模型的关系。

## 边界

- 数学、概率、信息论和神经网络基础放入 `../fundamentals/`。
- 训练策略、后训练和分布式训练放入 `../training/`。
- KV Cache、量化、服务框架和推理加速放入 `../inference/`。
- RAG、Agent 和产品化应用放入 `../application/`。

## Direct-Call 信息

- 目录入口链接：`[[architecture/|Architecture]]`。
- 基础结构入口：`[[architecture/transformer/|Transformer]]`、`[[architecture/attention/|Attention]]`。
- 模型家族入口：`[[architecture/model-families/|Model Families]]`。
