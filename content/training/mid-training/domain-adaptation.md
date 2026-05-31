---
title: Domain Adaptation
created: 2026-02-28
published: 2026-02-28
modified: 2026-05-31
type: topic
status: mature
area: training
tags:
  - mid-training
  - domain-adaptation
---

Domain Adaptation 是让已有模型更适应特定领域分布的训练过程。它通常通过 [[training/mid-training/continued-pretraining|Continued Pretraining]]、领域 SFT、检索增强或参数高效微调完成；在 mid-training 语境下，domain adaptation 主要指继续使用预训练式目标，在领域语料上更新 base model 或 intermediate checkpoint。

它回答的问题是：当通用 base model 已经具备语言和世界知识后，如何用有限领域数据提高法律、医疗、金融、代码、数学、科研、企业文档等场景的能力，同时避免通用能力退化。

## 领域适配的对象

领域可以按数据和能力划分：

| 领域 | 典型数据 | 目标能力 |
|---|---|---|
| Code | Git repositories、API docs、issues、tests | 程序理解、补全、debug、工具调用 |
| Math | 题库、教材、证明、解题轨迹 | 符号推理、证明、计算、验证 |
| Medical | 指南、论文、病例教材、药品说明 | 专业知识、术语理解、风险意识 |
| Legal | 法条、判例、合同、法律解释 | 条文检索、案例分析、严谨表达 |
| Finance | 财报、公告、研报、时间序列说明 | 财务语言、指标解释、风险分析 |
| Enterprise | 内部文档、工单、代码库、流程规范 | 私域知识、组织术语、工作流适配 |

不同领域对数据质量和风险要求不同。医疗、法律、金融属于高风险领域，不能只看 language modeling loss，还需要事实性、时效性、安全边界和责任边界评估。

## 训练方式

常见路线：

1. **Domain CPT**：在领域文档上继续 next-token prediction。
2. **Mixed CPT**：领域数据与通用高质量数据混合，降低遗忘。
3. **Domain SFT**：使用领域 prompt-response 数据训练任务行为。
4. **RAG-first adaptation**：不改变模型参数，而用检索系统注入领域知识。
5. **PEFT adaptation**：使用 LoRA / adapters 做低成本领域适配。

选择取决于目标：

- 需要改变 base representation：优先 CPT。
- 需要学会领域任务格式：优先 SFT。
- 知识更新频繁或合规敏感：优先 RAG。
- 数据少、成本低、可回滚：优先 PEFT。

这些方法可以组合，但应分清每一阶段解决什么问题。

## Data Mix 与 Replay

领域适配最重要的变量是领域数据和通用数据的比例。设领域数据采样比例为 $\lambda$：

$$
p_{\mathrm{train}}(x)
= \lambda p_{\mathrm{domain}}(x)
+ (1-\lambda)p_{\mathrm{general}}(x)
$$

$\lambda$ 越大，领域注入越强，但遗忘风险越高；$\lambda$ 越小，通用能力保留更好，但领域提升可能不足。实际训练常用阶段式策略：先较高比例领域数据注入，再用高质量混合数据退火。

Replay data 是降低遗忘的关键。它可以来自原预训练高质量子集，也可以是覆盖通用语言、代码、数学、多语言和安全场景的保留集。

## 灾难性遗忘

Catastrophic forgetting 在领域适配中表现为：

- 通用 benchmark 下降；
- 多语言能力下降；
- 原有代码/数学能力下降；
- 指令跟随或安全行为变差；
- 生成风格过度领域化；
- 对非领域问题过度套用领域术语。

缓解方式：

- 降低 learning rate；
- 增加 replay data；
- 缩短训练 token；
- 分阶段调整领域比例；
- 使用 adapter/LoRA 降低对全参数的破坏；
- 对 checkpoint 做多目标选择，而非只选领域 loss 最低点。

## 评测设计

领域适配评测至少包括：

- 领域 validation loss；
- 领域 benchmark 或专家题集；
- 通用 benchmark；
- 安全和拒答边界；
- 时效性和事实性；
- contamination 检查；
- 与 RAG baseline 对比；
- 后续 SFT/RLHF 后的能力保留。

对高风险领域，不应只依赖自动 benchmark。需要人工专家审查、引用可追溯性和不确定性表达评估。

## 常见失败模式

- **领域语料过窄**：模型学到领域文风，但没有真正提升任务能力。
- **过拟合题库**：benchmark 上升但泛化差。
- **忽略知识时效性**：模型吸收过时医疗/法律/金融信息。
- **用 CPT 解决交互问题**：领域知识提升不等于会回答用户问题。
- **不设通用保留集**：训练后才发现通用能力退化。
- **合规边界不清**：高风险领域生成建议超出可接受范围。

## 相关概念

- [[training/mid-training/continued-pretraining|Continued Pretraining]]
- [[training/mid-training/capability-injection|Capability Injection]]
- [[training/pretraining/data-mix|Data Mix]]
- [[training/data-engineering/quality-filtering|Quality Filtering]]
- [[training/post-training/sft|SFT]]
- [[application/rag/rag|RAG]]
