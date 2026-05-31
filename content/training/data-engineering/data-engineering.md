---
title: Data Engineering
created: 2026-03-14
published: 2026-03-14
modified: 2026-05-31
type: topic
status: mature
area: training
tags:
  - training
  - data
---

Data Engineering 是大模型训练中把原始数据转化为可训练 token stream 的工程与研究过程。它不是简单的 ETL，而是在定义训练分布、控制噪声、减少污染、提升有效 token 密度，并为 [[training/pretraining/data-mix|Data Mix]]、[[training/pretraining/tokenizer|Tokenizer]] 和 [[training/scaling/compute-optimal|Compute Optimal]] 提供可靠输入。

在 next-token prediction 中，模型学习的是训练语料的分布。如果数据工程质量差，模型会把乱码、重复、低质模板、评测泄漏、偏见和错误格式也一并学习进去。因此，数据工程决定的不只是训练效率，也决定模型能力边界和评测可信度。

## 核心目标

训练数据工程通常同时追求五个目标：

1. **Coverage**：覆盖目标语言、领域、任务格式和知识范围。
2. **Quality**：提高单位 token 的信息密度和可学习性。
3. **Diversity**：避免过度重复、模板化或单一来源支配训练分布。
4. **Safety and compliance**：控制毒性、隐私、版权、PII 和敏感内容风险。
5. **Evaluation integrity**：降低 benchmark contamination 和数据泄漏。

这些目标经常冲突。例如，强过滤可以提高平均文本质量，但可能损失长尾知识和低资源语言；大规模 web data 覆盖广，但噪声、重复和污染严重；合成数据可控性强，但容易引入 teacher bias 和模板化。

## 典型 Pipeline

一个 LLM pretraining data pipeline 通常包括：

1. **Source acquisition**：Common Crawl、代码仓库、书籍、论文、论坛、百科、领域文档、多语言数据、合成数据。
2. **Parsing and extraction**：从 HTML、PDF、Markdown、代码仓库、压缩包或数据库中抽取文本。
3. **Normalization**：统一编码、Unicode normalization、空白符、换行、标点、控制字符和文档边界。
4. **Language and domain identification**：识别语言、领域、格式和数据来源。
5. **Cleaning**：去除 boilerplate、导航栏、广告、乱码、异常短文本、重复模板和明显垃圾内容。
6. **Deduplication**：exact dedup、near dedup、document-level / paragraph-level / benchmark-level 去重。
7. **Quality filtering**：规则、分类器、perplexity filtering、model-based scoring、domain-specific filters。
8. **Safety and privacy filtering**：PII、密钥、恶意代码、毒性内容、版权和敏感数据控制。
9. **Mixture construction**：按语言、领域、质量层级和目标能力定义采样比例。
10. **Tokenization and packing**：用固定 tokenizer 编码，并把样本打包成训练序列。
11. **Sharding and metadata**：分片存储 token stream，记录 provenance、过滤版本、mix 权重和文档元数据。
12. **Validation and audit**：构建 held-out validation sets、contamination 检查和数据统计报告。

这个流程应该是可复现的。训练数据一旦进入大规模实验，任何过滤规则、去重阈值、tokenizer 版本或 mix 权重变化都可能影响 loss curve 和最终能力。

## 数据工程与 Scaling 的关系

[[training/scaling/model-data-compute|Model Data and Compute]] 中的 $D$ 通常指 tokenizer 后的训练 token 数。但数据工程决定的是有效 token：

$$
D_{\mathrm{effective}} \le D_{\mathrm{nominal}}
$$

其中 $D_{\mathrm{nominal}}$ 是名义 token 数，$D_{\mathrm{effective}}$ 是真正提供新信息、泛化价值和目标能力的 token。重复、模板化、乱码、污染和低质文本会消耗 compute，却降低有效 token 密度。

因此，compute-optimal 不只是选择 $N$ 和 $D$，也要选择数据 pipeline。一个更小但高质量的数据集可能比更大的低质数据集更有效；但过度过滤也可能牺牲覆盖和多样性。数据工程的目标不是让数据“越干净越好”，而是在目标能力、覆盖、成本和风险之间取得可解释的平衡。

## Provenance 与版本管理

训练数据需要像代码一样版本化。至少应记录：

- 原始数据来源和抓取时间；
- license / usage policy；
- 清洗规则版本；
- 去重算法和阈值；
- 质量过滤模型和阈值；
- tokenizer 版本；
- data mix 权重；
- train / validation / contamination blocklist；
- shard ID 与文档 ID 映射；
- 数据统计报告。

没有 provenance，就很难解释某个 checkpoint 的能力变化，也无法在发现污染或合规问题后定位受影响数据。

## 数据评估指标

常见数据层指标包括：

| 指标 | 作用 |
|---|---|
| token count | 估算训练规模和 compute |
| document count | 观察来源与粒度 |
| language distribution | 检查多语言覆盖 |
| domain distribution | 检查能力分布 |
| dedup rate | 判断重复和近重复程度 |
| quality score distribution | 观察过滤是否过强或过弱 |
| toxicity / PII rate | 安全与合规风险 |
| tokenizer fertility | 检查不同语言/领域的切分效率 |
| benchmark overlap | 评估污染风险 |
| validation loss by domain | 验证数据 mix 是否支持目标能力 |

这些指标不能替代训练结果，但可以帮助解释训练结果。如果模型在某领域表现异常，通常应先回看该领域数据数量、质量、去重、tokenization 和 contamination。

## 常见失败模式

- **只追求 token 数**：名义数据越大，不代表有效训练信号越多。
- **过滤不可复现**：没有记录规则和阈值，后续无法解释实验差异。
- **过度依赖单一来源**：模型会继承来源的风格、偏见和知识时效性。
- **去重不足**：增加记忆、过拟合和 benchmark leakage。
- **去重过强**：误删常见表达、低资源语言或合法重复结构。
- **质量过滤偏向主流语言**：英语或高资源语言更容易被判为高质，低资源语言被系统性削弱。
- **忽略 tokenizer 影响**：同一文本经不同 tokenizer 后的 token count 和训练成本不同。
- **缺少 held-out audit**：无法区分真实泛化和数据泄漏。

## 相关概念

- [[training/data-engineering/data-cleaning|Data Cleaning]]
- [[training/data-engineering/deduplication|Deduplication]]
- [[training/data-engineering/quality-filtering|Quality Filtering]]
- [[training/data-engineering/packing|Packing]]
- [[training/data-engineering/synthetic-data|Synthetic Data]]
- [[training/pretraining/data-mix|Data Mix]]
- [[training/pretraining/tokenizer|Tokenizer]]
- [[training/scaling/model-data-compute|Model Data and Compute]]
- [[sources/papers/2021-the-pile|The Pile]]
- [[sources/papers/2023-refinedweb|RefinedWeb]]
- [[sources/papers/2024-datacomp-lm|DataComp-LM]]
