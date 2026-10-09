---
title: Multimodal
created: 2026-02-15
published: 2026-02-15
modified: 2026-10-08
type: topic
status: growing
area: architecture
tags:
  - multimodal
  - representation
  - architecture
---

Multimodal 研究文本、图像、音频、视频等信息如何被联合表示、对齐、理解和生成。本页以视觉与语言为主要阅读入口，提供连接文本语言模型与世界模型的知识地图。三者的全局定位见 [[architecture/|Architecture]]。

Multimodal Model 的范围大于 Multimodal LLM：图文双塔表示模型可以用于检索，并不必须包含生成式语言模型；多模态大语言模型则把非文本信息与语言建模结合。Vision-Language Model（视觉语言模型，VLM）也有不同用法，需要结合具体结构判断它指双塔表示模型还是生成式视觉语言模型。

常见的生成式路线采用模态编码器、projector / connector（投影或连接模块）和语言骨干：编码器提取特征，连接模块适配表示接口，fusion（融合）机制让不同模态交互。这是一种重要结构，统一 token 建模、cross-attention（交叉注意力）和原生多模态架构还可以采用不同接口。输出端也可能继续连接图像、音频、视频或动作模块，因此不能从“有多模态输入”直接推断“有多模态输出”。

## 它解决什么问题

多模态系统需要把不同来源、不同采样率、不同空间结构的数据放入一个可以交互的表示空间。重点问题包括：

- 图像、音频、视频和文本如何分别编码成 token 或 continuous feature；
- 不同模态的时间、空间和语义粒度如何对齐；
- projector、cross-attention 或 shared embedding 如何完成接口转换；
- 视觉或音频信息如何参与语言生成、检索、推理和工具调用；
- 表示是否保留 grounding（输出与输入区域或时刻的对应）、时间顺序和动作相关信息。

## 多模态能力的三个独立维度

描述一个模型或系统时，至少要把输入、输出和任务类型分开写：

| 维度              | 要记录什么                     | 例子                                                  |
| ----------------- | ------------------------------ | ----------------------------------------------------- |
| Input modalities  | 模型能够读取哪些模态           | text、image、audio、video、sensor                     |
| Output modalities | 模型能够直接生成或控制哪些模态 | text、image、audio、video、action                     |
| Interaction type  | 模态之间如何被使用             | understanding、generation、transformation、any-to-any |

因此，Multimodal LLM 可能只是“多模态输入、文本输出”：

```text
image / audio / video / text -> Multimodal LLM -> text
```

也可能支持生成或转换：

```text
text / image / audio -> unified or composed system -> image / audio / video / text
```

后者不一定由一个 LLM 独立完成。很多产品级系统由 Multimodal LLM 负责理解与编排，再调用 image decoder、diffusion model、speech synthesizer 或外部工具完成最终输出。写模型笔记时，建议显式列出 `Input modalities`、`Output modalities` 和 `Interaction type`，并注明是 unified model 还是 composed system。

## 常见类型

| 类型                     | 输入                       | 输出                      | 例子或任务                                   |
| ------------------------ | -------------------------- | ------------------------- | -------------------------------------------- |
| Multimodal Understanding | 一种或多种非文本与文本模态 | 主要是文本                | image QA、OCR、video QA、audio understanding |
| Multimodal Generation    | 文本或多模态条件           | 图像、音频、视频或文本    | text-to-image、image editing、text-to-speech |
| Any-to-Any / Omni        | 多种模态                   | 多种模态                  | 实时语音与视觉联合交互                       |
| VLA                      | 视觉、语言和环境状态       | action 或 control command | 机器人和 GUI Agent                           |

“VLM”或“Multimodal LLM”只说明模型覆盖的模态或语言骨干，不能单独说明它支持哪些输出。图文双塔模型可以是 Multimodal Model，但不一定是生成式 LLM；VLA 的动作输出也首先说明它承担 policy 功能，不自动说明它拥有 World Model。

## Reading Path

```text
原始图像 / 音频 / 视频 / 文本
  -> 各模态的表示与 token 化
  -> 跨模态对齐与融合
  -> 理解、检索、推理或生成

环境交互分支：观测历史 -> 状态估计 -> 动力学预测 / 动作策略
```

阅读顺序按问题展开：

1. **表示**：[[fundamentals/representation-learning/|Representation Learning]] 解释原始输入如何变成特征；[[architecture/multimodal/clip|CLIP]] 是图文对齐的阅读入口。
2. **接口**：[[architecture/multimodal/vision-language-model|Vision-Language Model]] 与 [[architecture/multimodal/multimodal-projector|Multimodal Projector]] 连接视觉表示和语言骨干。
3. **模型组合**：[[architecture/multimodal/llava|LLaVA]] 与 [[architecture/multimodal/qwen-vl|Qwen-VL]] 用于比较编码、融合和输入组织方式。
4. **训练与部署**：[[training/multimodal/|Multimodal Training]] 关注目标和数据；[[inference/multimodal/|Multimodal Inference]] 关注 token 数量、上下文、缓存和延迟。
5. **任务反馈**：[[application/multimodal/|Multimodal Applications]] 连接图像问答、文字识别、视频理解和环境交互评测。

音频、视频和交错图文可沿同一框架展开，但需要额外处理时间同步、长序列和跨模态输出。多模态表示是视觉世界模型的一条输入路径，掌握所有模态并不是学习世界模型的前置条件。对世界模型而言，输出未来 observation、reward 或 action consequence 比输出模态数量更重要。

## 与 World Model 的边界

| 问题        | Multimodal                                                | World Model                                              |
| ----------- | --------------------------------------------------------- | -------------------------------------------------------- |
| 主要对象    | 不同模态的 observation 和语义                             | state、action、transition 和未来结果                     |
| 时间要求    | 可以只处理单个静态 observation                            | 通常需要建模时间演化                                     |
| action 要求 | 可选，工具调用或动作输出不代表内部预测                    | 预测型模型可以无显式 action；控制型模型需刻画行动后果    |
| 主要评测    | recognition、grounding、generation、cross-modal reasoning | prediction、planning、control、long-horizon task success |
| 典型失败    | 看错、对不齐、幻觉、时空 grounding 错误                   | model error、compounding error、uncertainty、错误规划    |

以机器人推杯子为例：识别杯子、桌沿和语言指令，属于多模态感知与理解；预测不同推力下杯子的位置和跌落风险，属于环境动力学建模；比较这些结果并决定推力，属于规划或策略选择。这些能力可以共享参数，也可以由不同模块实现。

Vision-Language-Action（视觉—语言—动作，VLA）模型直接输出动作时，首先承担 policy（策略）功能；是否还具有 World Model，需要检查它是否建模未来环境结果。视频生成模型则需要区分画面质量、时间一致性和动作可控性，不能仅凭画面质量推断控制能力。一个系统可以拥有多模态输出，却仍然只是生成或转换系统；是否是 World Model 要看它是否建模环境演化和行动后果。

## 知识入口

- [[fundamentals/representation-learning/|Representation Learning]] — 从 raw signal 到 feature、embedding 和 latent state 的共同基础。
- [[architecture/multimodal/vision-language-model|Vision-Language Model]]
- [[architecture/multimodal/clip|CLIP]]
- [[architecture/multimodal/llava|LLaVA]]
- [[architecture/multimodal/qwen-vl|Qwen-VL]]
- [[architecture/multimodal/multimodal-projector|Multimodal Projector]]
- [[training/multimodal/|Multimodal Training]] — 数据、对齐目标、联合训练和 instruction tuning。
- [[inference/multimodal/|Multimodal Inference]] — multimodal context、token compression、cache 和 serving。
- [[application/multimodal/|Multimodal Applications]] — 图像问答、文字识别、grounding、视频理解和工具交互。
- [[architecture/world-models/|World Models]] — 从 observation representation 进一步进入 state、dynamics 和环境预测。

## 回顾入口

- **快速理解**：[[architecture/multimodal/vision-language-model|Vision-Language Model]] → [[architecture/multimodal/multimodal-projector|Multimodal Projector]]。
- **理解模型如何训练和部署**：[[training/multimodal/|Multimodal Training]] → [[inference/multimodal/|Multimodal Inference]]。
- **进入 World Model**：[[fundamentals/dynamical-systems/|Dynamical Systems]] → [[architecture/world-models/|World Models]]。

具体模型的比较围绕“输入输出模态 → 表示与对齐 → 融合方式 → 训练目标 → 推理行为 → 评测边界”展开。由 observation 进入 state、dynamics 和决策闭环的完整路线见 [[architecture/world-models/|World Models]]。
