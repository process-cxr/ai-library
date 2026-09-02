---
title: Deduplication
created: 2026-03-14
published: 2026-03-14
modified: 2026-05-31
type: topic
status: mature
area: training
tags:
  - data-engineering
  - deduplication
---

Deduplication 是识别并移除训练数据中重复或近重复内容的过程。它直接影响有效 token 数、训练效率、模型记忆、benchmark contamination 和最终泛化能力。

重复数据不只是浪费 compute。模型反复看到同一文档或高度相似文本时，更容易记忆具体片段，并在评测中因泄漏获得虚高结果。但去重过强也可能删除合法的常见表达、模板、代码模式和低资源语言数据。

## 重复类型

训练数据中的重复可以分为：

| 类型 | 例子 | 风险 |
|---|---|---|
| Exact duplicate | 完全相同文档、镜像网页、重复抓取 | 浪费 token，增强记忆 |
| Near duplicate | 只改标题/时间/广告的网页 | 增加模板记忆和分布偏差 |
| Boilerplate duplicate | 页脚、导航栏、版权声明反复出现 | 模型学习无意义模板 |
| Paragraph duplicate | 文档不同但段落相同 | 局部重复仍会影响记忆 |
| Code duplicate | fork、vendor dependency、复制粘贴文件 | 代码模型评测泄漏和许可证风险 |
| Benchmark overlap | 训练集包含评测题目、答案或变体 | 评测结果不可信 |

不同重复需要不同粒度处理。只做文档级 exact dedup，无法解决 paragraph-level boilerplate 和 benchmark contamination。

## Exact Deduplication

Exact dedup 通常使用 hash：

1. 对文本进行规范化，例如统一换行、去除首尾空白；
2. 计算文档 hash，例如 SHA-1 / SHA-256；
3. 保留每个 hash 的一个实例；
4. 记录重复来源和删除数量。

Exact dedup 的优点是简单、确定、可复现，适合大规模流水线。缺点是对轻微变化不敏感：多一个广告、不同抓取时间、不同 HTML 残留都会让 hash 改变。

Exact dedup 也可用于段落、句子、代码文件或 benchmark item。粒度越细，越能清理局部重复，但越容易误删合法重复表达。

## Near Deduplication

Near dedup 处理高度相似但不完全相同的文本。常见方法是 shingling + MinHash + LSH：

1. 将文档切成 $k$-shingles，例如 token n-grams 或 character n-grams；
2. 把文档表示为 shingle set；
3. 用 Jaccard similarity 衡量两个文档集合相似度：

$$
J(A,B)=\frac{|A \cap B|}{|A \cup B|}
$$

4. 用 MinHash 近似 Jaccard similarity；
5. 用 Locality-Sensitive Hashing, LSH，快速找到候选相似文档；
6. 对超过阈值的候选进行去重或聚类。

MinHash 的价值在于可扩展性。直接比较所有文档对是 $O(n^2)$，不可行；MinHash/LSH 可以在海量文档中高效发现近重复候选。

## 文档级、段落级与数据集级去重

去重粒度会改变结果：

- **Document-level dedup**：保留文档完整性，适合网页、书籍、论文；无法清除局部 boilerplate。
- **Paragraph-level dedup**：能清理重复段落和模板；可能破坏文档结构和常见表达。
- **Line-level dedup**：常用于代码和日志；对自然语言可能过强。
- **Dataset-level dedup**：在多个数据源之间去重，避免同一内容因来源不同被重复采样。
- **Train-eval dedup**：用评测集、benchmark、题库和 held-out data 对训练集做 overlap blocking。

对 LLM 训练，通常需要组合使用：先做 source 内部 exact/near dedup，再做跨 source dedup，最后针对评测集做 contamination audit。

## Benchmark Contamination

Benchmark contamination 是去重中最敏感的场景。训练数据如果包含评测题、答案、解析、近似改写或源网页，模型可能看似能力更强，实际只是记忆。

污染检测通常包括：

- exact string match；
- normalized match；
- n-gram overlap；
- embedding similarity；
- MinHash near match；
- 题目、选项、答案、解析分别匹配；
- 对代码 benchmark 检查函数签名、测试用例和解答；
- 对数学 benchmark 检查题目变体和答案形式。

完全消除污染很难，尤其对于公开 benchmark 和 web-scale corpora。更现实的做法是保留 contamination report，并在高风险 benchmark 上谨慎解释结果。

## 对训练动态的影响

适当去重通常会：

- 提高有效 token 多样性；
- 降低训练样本记忆；
- 降低 validation leakage；
- 让 loss 曲线更可信；
- 减少重复模板对生成风格的影响。

但过度去重可能：

- 删除高频但合法的语言模式；
- 降低代码 API、法律条文、数学定义等重复结构的学习机会；
- 对低资源语言造成额外数据损失；
- 破坏文档内部连贯性；
- 让模型对真实重复场景不适应。

因此，去重阈值应按数据源和目标能力区分。代码、网页、书籍、论文、多语言数据不应机械使用同一阈值。

## 常见实践建议

- 保留去重前后的 token count、document count 和来源分布。
- 对不同 source 分别设置 near dedup 阈值。
- 对 benchmark 相关数据使用更严格 overlap blocking。
- 对代码数据特别处理 fork、vendor、generated 和 minified 文件。
- 对低资源语言审查去重损失，避免少量数据被大量误删。
- 保存 duplicate cluster metadata，便于后续审计。
- 不只报告最终 token 数，也报告 dedup rate。

## 常见失败模式

- **只做 exact dedup**：无法清除网页镜像、模板变体和轻微改写。
- **near dedup 阈值过低**：误删主题相近但内容不同的文档。
- **只在 source 内去重**：跨数据源重复仍然保留。
- **忽略 benchmark 变体**：题目改写和答案解析仍可能污染训练集。
- **没有保留 cluster 信息**：无法解释哪些数据被删除以及为什么。
- **把去重等同于质量过滤**：去重解决重复，不直接判断语义质量。

## DataComp-LM 的工程对照

[[sources/papers/2024-datacomp-lm|DataComp-LM]] 在统一训练协议下比较 MinHash、suffix array 和 modified Bloom filter。两类 pipeline 在 7B-2x scale 的 downstream performance 差异约在 `0.2 CORE` 以内，但 Bloom filter 同时处理 document-level 与 paragraph-level near duplicate，更容易扩展到超过 10TB 的数据，因此被用于 DCLM-Baseline。

论文还显示，去重超参会影响不同任务：在 7B-2x 实验中，较短的 `min_ngram_size=5` 可以保持较高 CORE，但会明显损害 MMLU；使用 `min_ngram_size=13` 的配置取得更均衡的结果。去重策略不能只按 removal rate 或剩余 token 数排序，还要观察短文本、任务覆盖、benchmark overlap 和数据来源分布。

## 相关概念

- [[training/data-engineering/data-engineering|Data Engineering]]
- [[training/data-engineering/data-cleaning|Data Cleaning]]
- [[training/data-engineering/quality-filtering|Quality Filtering]]
- [[training/pretraining/data-mix|Data Mix]]
- [[application/evaluation/evaluation|Evaluation]]
- [[sources/papers/2021-deduplicating-training-data-makes-language-models-better|Deduplicating Training Data Makes Language Models Better]]
