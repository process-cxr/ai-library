---
title: Data Mix
created: 2026-02-22
published: 2026-02-22
modified: 2026-08-31
type: topic
status: mature
area: training
tags:
  - pretraining
  - data
---

Data Mix 指预训练中不同数据来源、语言、领域、质量层级和格式类型的采样比例。它不是简单的数据清单，而是在定义模型要学习什么、优先学习什么、以及在哪些能力上承担风险。

在 next-token prediction 中，模型优化的是训练分布上的平均 log-likelihood。因此，data mix 本质上改变了训练分布 $p_{\mathrm{train}}(x)$。如果一个能力在数据中出现得少、质量低或格式不稳定，模型很难在预训练阶段稳定获得它。

## 数据类型与能力影响

| 数据类型 | 主要贡献 | 典型风险 |
|---|---|---|
| Web text | 常识、百科、长尾知识、自然语言广度 | 噪声、重复、SEO spam、偏见、安全风险 |
| Books | 长篇叙事、篇章结构、语言质量 | 版权、题材偏差、更新慢 |
| Code | 程序结构、符号推理、工具 API、格式约束 | 许可证、重复仓库、生成不安全代码 |
| Math | 形式化表达、证明、逐步推理、符号操作 | 规模小、格式不统一、题目泄漏 |
| Academic papers | 专业知识、严谨表达、公式和引用结构 | PDF/OCR 噪声、领域偏差、版权 |
| Q&A / Forum | 问题解决、对话式解释、实践经验 | 低质回答、过时信息、风格污染 |
| Multilingual data | 多语言能力、跨语言迁移、低资源覆盖 | tokenizer fertility 高、质量差异大 |
| Synthetic data | 定向补足稀缺能力、可控格式 | 模型自我污染、模式单一、错误放大 |
| Instruction-like text | 指令跟随雏形、问答格式、解释风格 | 与 post-training 边界模糊，可能引入 assistant 风格偏差 |

一个通用 base model 通常需要广覆盖；一个 code/math model 则会主动提高 code、math、竞赛题、证明、执行轨迹或技术文档比例。关键是要把比例与目标能力绑定，而不是照搬另一个模型的数据配方。

## Mix Ratio 不是 Corpus Ratio

原始语料大小不等于训练采样比例。常见做法是先把数据分成多个 corpus 或 domain，然后定义训练时采样概率 $q_i$。如果第 $i$ 个数据源有 token 占比 $p_i$，可以使用温度采样：

$$
q_i = \frac{p_i^\alpha}{\sum_j p_j^\alpha}
$$

其中：

- $\alpha = 1$ 表示按原始 token 占比采样；
- $0 < \alpha < 1$ 会压平分布，提高小数据源采样比例；
- $\alpha > 1$ 会更偏向大数据源；
- 实际系统常对某些高价值或高风险数据源额外设置 cap、floor 或人工权重。

例如，代码数据在原始 corpus 中可能只占少数，但训练时可以被上采样，以增强程序能力。低资源语言也常被上采样，否则它们会被英语网页数据淹没。上采样的代价是重复 epoch 增加，过拟合和模式重复风险上升。

## Quality 比 Quantity 更关键

Data mix 需要和 [[training/data-engineering/quality-filtering|Quality Filtering]]、[[training/data-engineering/deduplication|Deduplication]] 联动。低质量 token 会消耗 compute，却不一定提供有效信息。

高质量数据通常具有：

- 文档结构完整；
- 语言自然、信息密度高；
- 来源可信或可追溯；
- 格式稳定；
- 重复率低；
- 不包含明显广告、模板、乱码和机器翻译噪声；
- 与目标能力相关。

质量过滤也有风险。过强过滤可能让模型失去长尾知识、口语表达、多语言变化和真实用户噪声适应能力。对 safety 或 toxicity 的过滤也需要谨慎：完全删除某类内容可能降低模型识别和拒答能力，而保留过多又可能增加生成风险。

## Deduplication 与 Contamination

去重影响两个层面：

- **训练效率**：重复数据会让模型反复学习相同模式，降低有效 token；
- **评测可信度**：benchmark 泄漏会让评测结果虚高。

去重至少包括 exact dedup 和 near dedup。对代码数据，还要处理 fork、vendor dependency、自动生成文件和 minified 文件。对数学和 benchmark 相关数据，需要更严格的 contamination 检查，因为题目、答案、解析或变体都可能泄漏。

需要注意，去重不是越狠越好。某些高频模板、API 用法和语言常用表达确实应该被模型多次看到。真正的问题是无信息重复和评测泄漏。

## Curriculum 与 Annealing

Data mix 可以在训练过程中变化，而不是全程固定。常见策略包括：

- 早期使用更广泛的数据，帮助模型建立通用语言和世界知识；
- 中期提高高质量、代码、数学或目标领域比例；
- 后期使用更干净、更高质量或更接近目标分布的数据做 annealing；
- 在 long context 扩展阶段提高长文档、书籍、论文和多轮结构数据比例。

这种 curriculum 的直觉是：早期需要覆盖和稳定优化，后期更强调能力塑形和分布对齐。但动态 data mix 会让 scaling law 更难拟合，因为训练分布随时间变化。记录每个阶段的数据比例和 token milestone 非常重要。

## 多语言与 Tokenizer 耦合

多语言 data mix 不能只按文档数或字节数估计。不同语言在同一 tokenizer 下的 tokenization fertility 差异很大。某些语言会被切成更多 token，导致：

- 训练 compute 被更多 token 消耗；
- 单位文本的信息密度比较困难；
- PPL 和 loss 跨语言不可直接解释；
- 低资源语言即使文档数不少，也可能因质量和 tokenizer 不适配而表现差。

因此，多语言预训练要同时检查：

- 每种语言的文档质量；
- token 数、字符数和字节数；
- fertility，即单位字符/词对应的 token 数；
- domain 覆盖；
- held-out multilingual validation loss；
- 跨语言 benchmark 和翻译污染。

相关内容见 [[training/pretraining/tokenizer|Tokenizer]]。

## Instruction-like Data 的边界

预训练语料中经常包含问答、教程、论坛、issue、README、StackExchange、notebook、代码注释等 instruction-like 数据。这些数据会让 base model 提前接触“问题-回答”“步骤解释”“代码修复”等模式，对后续 SFT 有帮助。

但 pretraining 不等于 post-training。过多 assistant-style 或 synthetic instruction 数据可能让 base model：

- 过早形成固定对话风格；
- 学到低质量解释模板；
- 模仿错误答案；
- 增加与后续 SFT/RLHF 的分布耦合；
- 在纯 language modeling 评测上出现不可解释偏移。

比较稳妥的做法是区分自然出现的 instruction-like text、人工高质量 instruction 数据和模型生成 synthetic instruction，并分别记录比例与来源。

## Data Mix 的评估

Data mix 需要用分域评测闭环，而不是只看总 validation loss。

建议至少维护：

- general web validation；
- books / long-form validation；
- code validation；
- math validation；
- academic / technical validation；
- multilingual validation；
- safety / toxicity probes；
- contamination-sensitive benchmark holdout；
- downstream task set。

如果总 loss 改善但 code loss 变差，说明 mix 可能牺牲了代码能力。如果多语言平均 loss 正常但低资源语言很差，说明采样或 tokenizer 仍有问题。如果 benchmark 提升异常大，优先检查 contamination。

## 经验依据：异构性与目标覆盖

[[sources/papers/2023-a-pretrainers-guide-to-training-data|A Pretrainer's Guide to Training Data]] 将 The Pile 的 22 个来源划分为 9 个 domain cluster，并逐一做删除实验。结果显示，删除 Common Crawl、Books 和 OpenWeb 对平均 QA 性能的损失最大；删除 Common Crawl 后数据量仍约为完整数据的 73%，但平均 QA 性能相对下降约 4.8 个百分点。

直接对应某个评测领域的数据并不总是最重要。删除 Academic 对平均 QA 的影响接近零，而删除 Common Crawl 对 Academic QA 的影响更大。论文据此说明，大型异构来源能够提供跨主题、跨文体和跨任务的覆盖，不能只按“与 benchmark 名称是否相同”判断数据价值。

另一方面，Books、OpenWeb 和 Common Crawl 也包含更多被判为 toxic 的内容。它们对泛化有益、同时增加生成风险，说明 data mix 的决策目标至少包括能力覆盖、domain transfer、风险控制和数据质量四个维度。

## Data Mix 的实验记录

数据 mixture 进入训练前，应至少留下以下可比较信息：

- 每个 source / domain 的名义 token 数和实际采样比例；
- 过滤、去重和重复 epoch 后的保留量；
- 各 domain 的 quality、toxicity、PII 和时间分布；
- general、domain-specific、long-context 和目标能力 validation loss；
- 每个 ablation 版本的训练预算、tokenizer、优化 recipe 和 downstream 结果。

这样才能区分“数据更多带来的收益”“来源异构性带来的收益”和“过滤策略改变分布带来的收益”。

## DeepSeek-V3 的数据配方经验

DeepSeek-V3 的 pre-training corpus 约为 14.8T tokens。相较前代，作者提高数学与 programming 数据比例，扩展多语言覆盖，并优化去冗余与 document packing。它还以 0.1 的比例加入 FIM 数据，使模型学习根据 prefix 与 suffix 恢复 middle。

这组设计提供了两个可复用的记录原则：

1. 数据总量之外，要记录 domain mixture、FIM / synthetic / natural data 比例、重复次数、packing 方式和 tokenizer 后 token 数。
2. 数据结构改变会改变模型的条件分布。document packing 如果强调文档完整性，和 SFT packing 中通过 sample masking 隔离不同样本，是两种不同的训练语义，不能混为普通的“拼接”。

DeepSeek-V3 的 tokenizer 也与 data mix 强耦合：它使用 128K byte-level BPE，并针对多语言 compression 引入 punctuation / line-break 组合 token。组合 token 可能造成 token boundary bias，因此训练时随机拆分一部分组合 token。对多语言或代码数据，tokenizer fertility、边界形式和 FIM 格式都应该作为 data pipeline 的一部分记录。

## 常见失败模式

- **把更多数据等同于更好数据**：名义 token 增加不一定带来有效 token 增加。
- **忽略重复 epoch**：小而高质量的数据被过度上采样后可能造成过拟合。
- **只用总 validation loss 决策**：会掩盖关键领域能力退化。
- **混入过多低质合成数据**：模型会学习模板化、错误或过窄分布。
- **过滤策略过强**：可能丢失长尾知识、真实噪声和低资源语言。
- **数据来源不可追溯**：后续无法解释能力变化、版权风险和污染问题。

## DeepSeekMath：数学语料与 Code/Math 配比

DeepSeekMath 从 Common Crawl 构造数学语料的过程说明，大规模 data mix 的核心不只是收集更多网页，而是建立可迭代的 quality-controlled recall pipeline。作者以 OpenWebMath 作为 positive seed，用 500K positive examples 和 500K Common Crawl negative examples 训练 fastText classifier；之后根据 domain 的召回比例发现新的数学来源，再人工标注数学 URL path，扩充 seed 并更新 classifier。四轮迭代后得到约 35.5M 数学网页和 120B math tokens，并对 GSM8K、MATH、CMATH、AGIEval 做 n-gram contamination filtering。

在相同 1.3B 模型、150B math-training tokens 的对照中，DeepSeekMath Corpus 的 GSM8K / MATH / CMATH 结果为 `23.8% / 13.6% / 41.5%`，高于 MathPile、OpenWebMath 和 Proof-Pile-2；中文 benchmark 的优势也更明显。这支持一个可复用的判断：domain data 的有效性同时取决于内容质量、覆盖范围、语言分布和重复程度，不能只按 corpus 名义 token 数排序。

论文还比较了 code 与 math 的训练顺序。对 1.3B 模型，`Code 400B -> Math 150B` 相比 `General 400B -> Math 150B` 提高了 GSM8K、MATH 以及 `GSM8K + Python`、`MATH + Python`；`Code + Math mixed` 则在保持 HumanEval / MBPP 方面更有优势，但牺牲了部分不使用工具的数学 reasoning。这个结果可作为 data mix ablation 的参考：目标能力提升、跨域迁移和 catastrophic forgetting 需要联合评估，不能只看单一领域 benchmark。

在论文测试的 arXiv-only 设置中，MathPile 和 ArXiv-RedPajama 没有带来稳定数学收益，部分结果出现退化。作者没有测试 arXiv 与其他数据混合、更大模型规模以及 theorem informalization，因此这里只能得出“当前配置下 arXiv-only 未显示明显收益”，不能推导出 arXiv 来源本身没有价值。

## DataComp-LM：把 Data Mix 变成受控实验

[[sources/papers/2024-datacomp-lm|DataComp-LM]] 将数据研究拆成 filtering track 和 mixing track：前者只从统一的 Common Crawl pool 选择数据，后者允许加入 Wikipedia、Books、StackExchange、arXiv、GitHub 等外部来源。这样可以把“从同一原始分布中筛选更好样本”和“不同来源如何混合”分开评估。

在 1B-1x scale，加入 RPJ extras 能改善 C4、RedPajama-CC 和 RefinedWeb，但对已经经过强 model-based filtering 的 DCLM-Baseline，CORE 从 `31.1` 降到 `29.9`，EXTENDED 从 `16.0` 降到 `15.0`。这说明外部高质量来源的边际收益取决于 base dataset 是否已经覆盖相同信息，混合不是默认正收益操作。

DataComp-LM 的另一个重要做法是使用 small proxy models 先比较数据策略，再用更大模型验证。400M、1B、3B 与 7B-1x 的 dataset ranking 相关性分别达到 `0.838`、`0.956` 和 `0.982`。这为大规模 data mix 的快速迭代提供了可操作路径，但 proxy 仍不能替代目标规模对长上下文、代码、数学和训练稳定性的最终验证。

## 相关概念

- [[training/pretraining/pretraining|Pretraining]]
- [[training/pretraining/compute-optimal|Pretraining Compute Optimal]]
- [[training/pretraining/tokenizer|Tokenizer]]
- [[training/data-engineering/data-engineering|Data Engineering]]
- [[training/data-engineering/quality-filtering|Quality Filtering]]
- [[training/data-engineering/deduplication|Deduplication]]
- [[training/data-engineering/packing|Packing]]
- [[training/scaling/model-data-compute|Model Data and Compute]]
- [[sources/papers/2021-the-pile|The Pile]]
- [[sources/papers/2023-a-pretrainers-guide-to-training-data|A Pretrainer's Guide to Training Data]]
- [[sources/papers/2023-refinedweb|RefinedWeb]]
- [[sources/papers/2024-datacomp-lm|DataComp-LM]]
