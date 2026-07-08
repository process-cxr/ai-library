---
title: Data Engineering
created: 2026-03-14
published: 2026-03-14
modified: 2026-07-06
---

训练数据工程负责数据收集、清洗、去重、过滤、混合、packing 和合成数据构造，是训练效果的基础约束。

建议阅读顺序：

1. [[training/data-engineering/data-engineering|Data Engineering]]：理解从原始数据到 token stream 的完整 pipeline。
2. [[training/data-engineering/data-cleaning|Data Cleaning]]：处理解析、规范化、语言识别和基础噪声。
3. [[training/data-engineering/deduplication|Deduplication]]：控制重复、记忆和评测污染。
4. [[training/data-engineering/quality-filtering|Quality Filtering]]：用规则、分类器、PPL 和模型评分选择有效数据。
5. [[training/data-engineering/packing|Packing]]：把 tokenized samples 组织成高吞吐训练序列。
6. [[training/data-engineering/distributed-dataloader|Distributed Dataloader]]：理解多卡训练中的数据分片、shuffle、packing、resume 和吞吐瓶颈。
7. [[training/data-engineering/synthetic-data|Synthetic Data]]：理解 teacher generation、Self-Instruct、verifier 和合成数据风险。

## Notes

- [[training/data-engineering/data-engineering|训练数据工程]]
- [[training/data-engineering/data-cleaning|数据清洗]]
- [[training/data-engineering/deduplication|数据去重]]
- [[training/data-engineering/quality-filtering|质量过滤]]
- [[training/data-engineering/packing|Packing]]
- [[training/data-engineering/distributed-dataloader|Distributed Dataloader]]
- [[training/data-engineering/synthetic-data|合成数据]]

## Related Source TODOs

- [[sources/papers/2019-t5|T5]]
- [[sources/papers/2021-documenting-large-webtext-corpora|Documenting Large Webtext Corpora]]
- [[sources/papers/2021-deduplicating-training-data-makes-language-models-better|Deduplicating Training Data Makes Language Models Better]]
- [[sources/papers/2023-roots|ROOTS]]
- [[sources/papers/2024-dolma|Dolma]]
- [[sources/papers/2024-fineweb|FineWeb]]
