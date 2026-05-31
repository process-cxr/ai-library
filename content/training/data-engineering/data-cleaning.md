---
title: Data Cleaning
created: 2026-03-14
published: 2026-03-14
modified: 2026-05-31
type: topic
status: mature
area: training
tags:
  - data-engineering
  - cleaning
---

Data Cleaning 是把原始文本抽取结果转化为可训练文档的过程。它处理的是数据进入 [[training/data-engineering/deduplication|Deduplication]]、[[training/data-engineering/quality-filtering|Quality Filtering]] 和 [[training/pretraining/data-mix|Data Mix]] 之前最基础的可读性、格式和结构问题。

清洗的目标不是让文本变得“漂亮”，而是保留可学习信息，同时去除明显不会提高模型能力、甚至会污染训练分布的噪声。

## 输入噪声来源

LLM 训练数据常见来源包括 web pages、PDF、代码仓库、论坛、书籍、论文、百科和合成数据。不同来源有不同噪声：

| 来源 | 常见噪声 |
|---|---|
| HTML / Web | 导航栏、广告、cookie banner、脚注、版权声明、推荐列表、SEO keyword stuffing |
| PDF | OCR 错误、断行、页眉页脚、页码、参考文献碎片、公式损坏 |
| Code repositories | vendor dependency、minified file、自动生成代码、二进制转储、license 重复 |
| Forums / Q&A | 签名、投票信息、引用嵌套、低质回复、过时答案 |
| Books | 扫描错误、目录、页码、版权页、章节边界错误 |
| Multilingual web | 语言混杂、机器翻译噪声、编码错误、脚本混用 |

清洗策略必须与来源匹配。用同一套规则处理所有数据，很容易误删高价值结构或保留来源特有噪声。

## Parsing 与 Text Extraction

清洗首先依赖正确抽取文本。对 HTML，通常需要区分 main content 与 boilerplate；对 PDF，需要恢复阅读顺序；对代码，需要保留文件路径、缩进和换行；对对话或论坛，需要保留发言边界。

常见抽取问题：

- HTML DOM 中正文和导航/广告混杂；
- PDF 双栏顺序被打乱；
- 表格被线性化后语义丢失；
- 数学公式被 OCR 成乱码；
- Markdown / code block 边界被破坏；
- 文档标题、作者、时间等 metadata 丢失。

如果抽取阶段破坏结构，后续 tokenizer 和模型会学习错误格式。对于代码、数学、长文档和 instruction-like 数据，结构保留尤其重要。

## 规范化

Normalization 用来减少无意义变体，但不能破坏语义。常见操作包括：

- Unicode normalization；
- 统一换行符；
- 清除控制字符；
- 规范连续空白；
- 修复编码错误；
- 处理 HTML entity；
- 标准化 document boundary；
- 保留或规范 Markdown/code block 标记。

需要谨慎的操作：

- 大小写转换：可能破坏专有名词和代码；
- 标点替换：可能影响数学、代码和多语言文本；
- 删除特殊符号：可能删除公式、emoji、变量名或非拉丁文字；
- 过度合并换行：可能破坏诗歌、列表、代码和表格。

规范化应以 tokenizer 和训练目标为边界。对自然语言合理的清洗规则，对代码和数学可能是破坏性规则。

## Language Identification

语言识别用于构建多语言分布、过滤异常文档和选择后续规则。常见做法包括：

- fastText / CLD 类轻量分类器；
- 字符 n-gram 特征；
- script detection；
- 文档级和段落级双层识别；
- 对混合语言文档保留比例信息。

语言识别的难点在于：

- 短文本容易误判；
- 代码、数学、URL、表格会干扰分类；
- 低资源语言训练数据少，分类器偏差大；
- 多语言文档不适合硬分到单一语言；
- 机器翻译文本可能被判为目标语言但质量较低。

因此，语言识别结果应作为 metadata 和 sampling 信号，而不总是硬过滤条件。

## 基础过滤规则

Data cleaning 常用规则包括：

- 删除空文档、过短文档和异常长文档；
- 删除乱码比例高的文档；
- 删除重复字符、重复 n-gram、模板化列表；
- 删除 URL、HTML tag、脚本代码比例异常的文本；
- 删除非目标语言或语言置信度过低的文本；
- 删除 PII、密钥、token、邮箱、手机号等敏感片段；
- 对代码数据删除 minified、generated、binary-like 文件；
- 对 PDF 数据删除页眉页脚、页码和参考文献噪声。

规则过滤可解释、便宜、容易复现，但容易过度简化。它适合清除明显异常，不适合判断深层语义质量。

## Safety 与 Compliance 清洗

训练数据清洗还需要处理：

- personally identifiable information, PII；
- API keys、private keys、passwords；
- copyrighted material 的使用边界；
- malicious code；
- toxic / hateful / sexual / violent content；
- medical/legal/financial 高风险建议；
- 儿童安全和其他敏感类别。

这些内容不一定都应简单删除。某些安全相关文本对模型识别、拒答和安全分类有价值，但需要可控采样、标注和隔离。对于 base pretraining，常见做法是降低高风险内容比例、移除明显违法或隐私数据，并为安全数据保留 metadata，以便后续审计。

## 清洗评估

清洗效果不能只看删除比例。更有用的检查包括：

- 随机抽样人工审查；
- 各来源保留率；
- 各语言保留率；
- 文档长度分布；
- 字符类别分布；
- tokenizer 后 token/character 分布；
- HTML / URL / boilerplate 残留率；
- PII / secret 检出率；
- domain validation loss；
- 下游任务与 contamination 检查。

如果清洗后某些语言或领域保留率异常低，需要检查规则是否偏置。如果清洗前后 token 数差异很大，也要确认删除的是噪声而不是结构化高价值内容。

## 常见失败模式

- **抽取失败被误认为低质量**：PDF 或网页解析坏了，后续过滤只是在处理错误输入。
- **一刀切规则破坏代码/数学**：删除特殊符号、空白和换行会损害结构。
- **语言识别误删低资源语言**：分类器偏差会强化数据不平等。
- **清洗过度**：模型失去真实用户噪声、长尾表达和格式多样性。
- **清洗不足**：boilerplate、广告和乱码被模型学习。
- **缺少 provenance**：清洗后无法追踪某段数据来自哪里、被哪些规则处理过。

## 相关概念

- [[training/data-engineering/data-engineering|Data Engineering]]
- [[training/data-engineering/deduplication|Deduplication]]
- [[training/data-engineering/quality-filtering|Quality Filtering]]
- [[training/pretraining/data-mix|Data Mix]]
- [[training/pretraining/tokenizer|Tokenizer]]
- [[sources/papers/2021-documenting-large-webtext-corpora|Documenting Large Webtext Corpora]]
