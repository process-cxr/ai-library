---
title: Quality Filtering
created: 2026-03-14
published: 2026-03-14
modified: 2026-05-31
type: topic
status: mature
area: training
tags:
  - data-engineering
  - quality-filtering
---

Quality Filtering 是根据文本质量、信息密度、目标相关性和风险水平选择训练数据的过程。它位于 [[training/data-engineering/data-cleaning|Data Cleaning]] 和 [[training/pretraining/data-mix|Data Mix]] 之间：清洗让文本可读，质量过滤决定哪些文本值得消耗训练 compute。

质量过滤不是单一算法，而是一组评分、筛选和采样策略。它的目标不是让数据变得单调“干净”，而是在覆盖、多样性、信息密度和安全风险之间取得可解释的平衡。

## 质量的含义

“高质量数据”在不同训练目标下含义不同：

| 目标 | 高质量信号 | 可能误判 |
|---|---|---|
| 通用预训练 | 信息密度、自然语言流畅、来源多样、结构完整 | 低质网页伪装成流畅文本 |
| Code model | 可解析代码、真实项目、注释、测试、API 使用 | generated/vendor/minified 文件 |
| Math model | 正确题目、解答、证明、公式格式稳定 | OCR 错误、答案泄漏、低质合成题 |
| 多语言模型 | 母语质量、覆盖方言和领域、多脚本支持 | 机器翻译文本被误判为高质 |
| 领域模型 | 权威来源、专业术语、时效性 | 领域文档过窄导致遗忘 |

因此，quality filtering 必须与目标能力绑定。没有通用过滤器可以在所有语言、领域和任务上都最优。

## 规则过滤

规则过滤便宜、可解释、可复现，常用于第一层粗筛：

- 文档长度阈值；
- 平均词长或 token/character 异常；
- 重复 n-gram 比例；
- 标点、数字、URL、HTML tag 比例；
- 非目标语言字符比例；
- 乱码、控制字符、不可见字符比例；
- profanity / toxicity keyword；
- 代码文件扩展名、minified ratio、generated file pattern；
- PII / secret pattern。

规则过滤适合清除明显坏数据，但不能判断语义正确性、事实质量或教学价值。规则越多，越容易引入隐性偏差。

## 分类器过滤

分类器过滤训练一个模型判断文档是否高质量。标签来源可以是：

- 人工标注；
- curated corpus 作为正样本，普通 web data 作为负样本；
- heuristic rules 生成弱标签；
- high-quality domain data 作为正样本；
- 模型偏好评分。

分类器输出可以用于 hard filter，也可以用于 sampling weight：

$$
q_i \propto p_i \cdot w(s_i)
$$

其中 $p_i$ 是原始数据源比例，$s_i$ 是质量分数，$w(s_i)$ 是由质量分数映射出的采样权重。相比硬删除，按权重采样更柔和，可以保留长尾多样性。

分类器过滤的主要风险是 bias amplification：如果正样本主要来自英语百科、新闻或学术文本，分类器可能把口语、低资源语言、论坛、代码注释和非主流表达系统性判低。

## Perplexity Filtering

Perplexity filtering 使用语言模型给文本打分。直觉是：质量较高、语言自然的文本应当有较低 perplexity；乱码、模板、机器翻译或异常文本 perplexity 较高。

对文档 $x_1,\dots,x_T$，平均负 log-likelihood 为：

$$
L(x) = -\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t \mid x_{<t})
$$

Perplexity 为：

$$
\mathrm{PPL}(x)=\exp(L(x))
$$

但 perplexity filtering 有明显边界：

- 太低的 perplexity 可能表示文本模板化或重复；
- 高 perplexity 可能是高价值专业文本、低资源语言、代码或数学；
- 打分模型的 tokenizer 和训练分布会影响结果；
- 对多语言和领域数据直接使用同一阈值会有偏差；
- PPL 只能衡量模型熟悉程度，不等于事实正确或教学价值。

因此，更稳妥的做法是按语言、领域、长度分桶设置阈值，并结合人工抽样审查。

## Model-based Scoring

更强的模型可以作为 judge 或 scorer，用于评估：

- 文档是否有信息量；
- 是否像高质量教程、论文、代码、问答；
- 是否包含错误、广告、垃圾或低质生成痕迹；
- 是否适合作为某类能力训练数据；
- 是否存在安全、隐私或合规风险。

Model-based scoring 的优势是语义能力强，能处理规则和 PPL 难以捕捉的问题。风险是成本高、可复现性差、judge bias、prompt 敏感，以及可能把 scorer 的偏好写入训练数据。

对于大规模预训练，常见做法不是用模型逐条深度判断全部数据，而是对候选子集、边界样本或高价值 domain 使用模型评分，再结合轻量过滤器扩展。

## Domain-specific Filtering

不同领域需要专门过滤：

- **Code**：删除 minified/generated/vendor 文件，保留 license metadata，检查可解析性、测试文件和项目结构。
- **Math**：保留公式结构，过滤 OCR 噪声，检查题目-答案边界和 benchmark overlap。
- **Academic**：处理 PDF parsing、参考文献、公式、表格和章节结构。
- **Medical/Legal/Financial**：重视来源权威性、时效性、免责声明和高风险错误。
- **Multilingual**：按语言分别建质量指标，避免英语中心过滤器压制低资源语言。

领域过滤通常比通用过滤更能提升目标能力，但过强会牺牲通用性。领域数据最好进入 [[training/pretraining/data-mix|Data Mix]] 的显式采样权重，而不是混在通用数据里不可追踪。

## 阈值选择与 Ablation

过滤阈值应通过实验确定，而不是只凭直觉。常见方法：

1. 构建多个过滤强度版本；
2. 保持 token budget、tokenizer、训练 recipe 一致；
3. 训练 small proxy models；
4. 比较 total validation loss、domain validation loss、benchmark 和 toxicity；
5. 检查各语言/领域保留率；
6. 选择在目标能力和覆盖之间最稳健的阈值。

过滤强度过低会保留噪声；过高会造成数据单一、覆盖不足和偏差放大。很多情况下，质量分数更适合转成 sampling weight，而不是绝对保留/删除。

## 常见失败模式

- **把 PPL 低等同于高质量**：模板化和重复文本也可能 PPL 很低。
- **分类器正样本偏置**：过滤器学习了某种文体，而不是真正质量。
- **跨语言共用阈值**：低资源语言和非拉丁文字被系统性删掉。
- **过度过滤 web data**：损失长尾知识、真实噪声和用户表达。
- **忽略数据污染**：高质量题解如果泄漏 benchmark，仍然不适合训练。
- **只看过滤后平均质量**：没有检查 domain coverage 和 diversity。

## 相关概念

- [[training/data-engineering/data-engineering|Data Engineering]]
- [[training/data-engineering/data-cleaning|Data Cleaning]]
- [[training/data-engineering/deduplication|Deduplication]]
- [[training/pretraining/data-mix|Data Mix]]
- [[training/pretraining/tokenizer|Tokenizer]]
- [[fundamentals/information-theory/perplexity|Perplexity]]
- [[sources/papers/2023-a-pretrainers-guide-to-training-data|A Pretrainer's Guide to Training Data]]
- [[sources/papers/2024-datacomp-lm|DataComp-LM]]
