---
title: "DataComp-LM"
created: 2026-05-31
published: 2026-08-31
modified: 2026-08-31
type: source
status: processed
source_type: paper
area: sources
tags:
  - source
  - paper
  - pretraining-data
  - datacomp
  - data-curation
  - quality-filtering
source_url: https://arxiv.org/abs/2406.11794
paper_date: "2024-06"
paper_order: "11794"
---

# DataComp-LM: In search of the next generation of training sets for language models

## 基本信息

- 论文：[DataComp-LM: In search of the next generation of training sets for language models](https://arxiv.org/abs/2406.11794)
- 作者：Jeffrey Li、Alex Fang、Georgios Smyrnis 等
- 发布信息：arXiv:2406.11794，v4 发布于 2025-04-21
- 项目主页：[DataComp for Language Models](https://datacomp.ai/dclm)
- 开源仓库：[mlfoundations/dclm](https://github.com/mlfoundations/dclm)
- 研究对象：web-scale language model pre-training dataset curation
- 相关主题：[[training/pretraining/data-mix|Data Mix]]、[[training/data-engineering/quality-filtering|Quality Filtering]]、[[training/data-engineering/deduplication|Deduplication]]、[[training/data-engineering/data-engineering|Data Engineering]]、[[training/pretraining/compute-optimal|Compute Optimal]]

DataComp-LM 的核心贡献不是发布一个“更大的网页数据集”，而是建立一个可以控制变量的数据研究平台：参与者从统一的 Common Crawl pool 中进行 filtering 或 mixing，使用固定的模型架构、训练 token 数、超参数和评测套件训练模型，再比较数据处理策略带来的性能差异。

这种设计回应了预训练数据研究中最常见的归因问题：两个数据集的结果不同，究竟是数据本身更好，还是训练时使用了不同的模型、learning rate、token budget、训练代码或评测协议。如果这些变量没有被固定，单次模型结果很难说明哪一种数据处理方法真正有效。

论文最终给出 DCLM-Baseline，并在 7B 模型上进行 2.6T-token 级别的扩展实验。结果表明，合适的 text extraction、deduplication 和 model-based filtering 能够显著改善训练数据的有效性；但论文同时也说明，DCLM-Baseline 主要优化通用语言理解，代码和数学能力仍需要专门的数据补充。

## 研究问题

### 如何比较数据处理策略本身

已有预训练数据工作通常分别提出 heuristic filtering、quality filtering、deduplication、data mixing 或 synthetic data 方法，但不同工作之间的模型规模、训练长度和优化 recipe 往往不一致。因此很难回答：

- 哪一种过滤器在相同训练预算下更有效；
- small-scale proxy 是否能够预示 larger-scale 结果；
- 数据质量改进能否跨 architecture 和 hyperparameter 配置保持；
- 过滤、去重和混合是相互叠加，还是某些策略会互相抵消；
- 人工认为“高质量”的文本，是否真的更适合 language model pre-training。

### 如何让数据实验可被不同计算预算的研究者参与

完整的 7B 预训练成本很高。论文设计了从 400M 到 7B-2x 的五个 compute scales，并检验小规模模型上的 dataset ranking 能否迁移到更大规模。这个设计使数据研究可以先在 400M、1B 或 3B 模型上进行筛选，再决定是否投入 7B 训练。

### 高质量数据来自哪里

论文还比较了：

- 重新进行 HTML text extraction 与直接使用 Common Crawl WET files 的差异；
- MinHash、suffix array 和 Bloom filter 的 deduplication 差异；
- PageRank、semantic deduplication、BGE classifier、AskLLM、perplexity、top-k logits 和 fastText quality filtering；
- 过滤后的 Common Crawl 是否还需要加入 Wikipedia、Books、StackExchange、arXiv 和 GitHub 等额外来源；
- human judgment 与 downstream usefulness 是否一致。

## 核心主张

1. **固定训练与评测条件的数据 benchmark 能显著改善 data curation 研究的可解释性。** DCLM 让不同策略在相同模型、token budget 和 evaluation suite 下比较。
2. **小规模模型可以为较大规模的数据策略提供有效信号。** 400M、1B、3B 规模与 7B-1x 的 dataset ranking 具有较高相关性，Pearson correlation 分别为 `0.838`、`0.956` 和 `0.982`。
3. **Model-based filtering 是 DCLM-Baseline 获得高质量数据的关键步骤。** 在论文比较的过滤器中，使用 fastText classifier、以 OpenHermes-2.5 和高分 ELI5 内容作为 positive reference 的方法表现最好。
4. **数据抽取方式会直接改变训练数据质量。** `resiliparse` 和 `trafilatura` 相比 Common Crawl WET files 带来更高的 downstream performance；`resiliparse` 的速度约为 `trafilatura` 的 8 倍。
5. **Deduplication 的工程可扩展性与下游效果需要同时考虑。** DCLM 的 modified Bloom filter 在 7B-2x 规模上与 MinHash + suffix array 的效果接近，但更容易扩展到超过 10TB 的数据。
6. **额外混合高质量来源并不总是有益。** 对 C4、RedPajama-CC 和 RefinedWeb，加入 RPJ extras 可以提升表现；对已经经过强过滤的 DCLM-Baseline，额外混合反而降低平均得分。
7. **人工质量判断不等于预训练有效性。** AskLLM 与人工标签的相关性高于若干 fastText filter，但它作为大规模 data filter 的 downstream 表现更差，说明“人认为文档好”与“文档能改善 LM”不是同一个目标。
8. **DCLM-Baseline 在通用语言理解上具有较强的 compute-performance trade-off。** 7B 模型使用 2.6T tokens 后，CORE 为 `57.1`、MMLU 为 `63.7`、EXTENDED 为 `45.4`；但论文明确指出其 code 和 math 能力仍不如专门的数据配方。

## DCLM 的实验框架

```text
DCLM-Pool
  -> 选择 compute scale
  -> filtering track / mixing track
  -> 固定 OpenLM training recipe
  -> 训练 decoder-only LM
  -> 53-task evaluation
  -> 比较 dataset quality
```

### DCLM-Pool

DCLM-Pool 是 2023 年以前 Common Crawl 数据的未过滤 web-text corpus。作者不直接采用 Common Crawl 提供的预抽取文本，而是从 HTML 重新进行 text extraction：

- 约 200B documents；
- gzip 压缩后约 370TB；
- 使用 GPT-NeoX tokenizer 约得到 240T tokens；
- 保留 URL、crawl time、WARC metadata、language 等信息。

DCLM-Pool 本身不做最终 decontamination，而是提供工具让参与者检查训练数据与 53 个 evaluation tasks 的 overlap，并要求参赛结果披露 decontamination report。这样可以把“研究数据策略”和“评测污染审计”拆开记录。

### 五个 compute scales

每个 scale 使用 `D = 20N × multiplier` 的训练 token 数，其中 `1x` 对应 Chinchilla-style 的约 20 tokens per parameter：

| Scale | Model parameters | Train tokens | Train FLOPs | H100 hours | Filtering pool |
|---|---:|---:|---:|---:|---:|
| 400M-1x | 412M | 8.2B | `2.0e19` | 26 | 469B tokens |
| 1B-1x | 1.4B | 28.8B | `2.4e20` | 240 | 1.64T tokens |
| 3B-1x | 2.8B | 55.9B | `9.4e20` | 740 | 3.18T tokens |
| 7B-1x | 6.9B | 138B | `5.7e21` | 3,700 | 7.85T tokens |
| 7B-2x | 6.9B | 276B | `1.1e22` | 7,300 | 15.7T tokens |

这里的 pool size 随 scale 增长，是为了让更高 compute scale 的 filtering strategy 有足够候选数据，也避免把小规模实验中极端严格的过滤比例直接假设为可扩展的 frontier recipe。

### Filtering track 与 mixing track

Filtering track 要求只从指定的 DCLM-Pool 中筛选数据，不能加入外部数据；mixing track 可以将 DCLM-Pool 与 Wikipedia、StackExchange、Books、arXiv、GitHub 或其他可用来源混合。

这种分轨设计把两个不同问题分开：

- filtering track 研究“如何从相同原始 pool 中找出更好的数据”；
- mixing track 研究“不同数据来源之间应该如何分配训练比例”。

如果不区分这两种情况，外部数据质量和过滤算法的收益会混在一起，很难判断提升来自 filtering 还是来自额外加入的 corpus。

## 固定训练与评测协议

### Training recipe

DCLM 使用 OpenLM 构建统一训练 infrastructure，模型是 decoder-only、pre-normalization Transformer，架构受到 GPT-2 和 Llama 的启发。主要设置包括：

- LayerNorm 不带 bias parameter；
- query 和 key 使用 qk-LayerNorm；
- MLP 使用 SwiGLU；
- depth-scaled initialization；
- sequence length：2048；
- GPT-NeoX tokenizer，vocabulary size 约 50K；
- standard next-token prediction objective；
- 使用 z-loss 稳定 output logit magnitude；
- 多个 document packing 到同一个 sequence，并用 EOS 分隔。

packing 时允许 causal attention 跨越 document 边界。作者对跨 document attention masking 做过实验，早期结果显示其对 downstream performance 影响很小，因此主实验采用更简单的 packing 方式。

不同 scale 使用固定的模型结构和训练 hyperparameters。7B-1x / 7B-2x 的主配置为 32 layers、32 attention heads、`d_model=4096`、`d_head=128`、5,000 warmup steps、learning rate `2e-3`、weight decay `0.05` 和 z-loss `5e-6`。部分早期表格使用较低 learning rate 的旧配置，论文在比较时明确区分了这些运行。

### Evaluation suite

完整 evaluation suite 包含 53 个 base-model tasks，覆盖 question answering、language understanding、commonsense、math、reading comprehension、world knowledge、bias 和 toxicity 等类型。论文主要报告两个 centered accuracy：

- **CORE**：22 个低方差任务的平均 centered accuracy，适合小模型观察学习信号；
- **EXTENDED**：全部 53 个任务的 centered accuracy，用于更全面的能力概览。

另外报告 MMLU 5-shot accuracy。Centered accuracy 会把每个任务的 random guessing 映射为 0、perfect accuracy 映射为 1，再进行平均，避免不同任务的随机基线和类别数直接混合。

论文还比较了 LightEval 和 LLM Foundry 在 MMLU 上的差异。LightEval 对完整 answer passage 的 log-probability 进行比较，在小模型上更早超过 random baseline；但在较大模型上分数更容易挤在一起，区分度反而不如 LLM Foundry。因此，评测 framework 本身也是 dataset comparison 的控制变量。

## DCLM-Baseline 的构造

### 从 Common Crawl 到 Baseline

DCLM-Baseline 的主要流程可以概括为：

```text
Common Crawl HTML
  -> resiliparse extraction
  -> RefinedWeb-style heuristic filters
  -> Bloom filter deduplication
  -> fastText OH-2.5 + ELI5 quality filtering
  -> retain top 10%
  -> DCLM-Baseline
```

图 4 以原始 document 数量为基准展示过滤漏斗。不同步骤分别删除 language 不匹配、页面过短、word removal ratio 异常、repetition、其他 heuristic 问题和 deduplication 命中的内容，最后使用 model-based fastText filter 保留得分最高的 10%。最终发布的 DCLM-Baseline 约 3.8T tokens、约 3B documents。

### Existing dataset comparison

在 7B-1x scale 上，论文比较了 C4、Dolma-V1、RedPajama 和 RefinedWeb：

| Dataset | CORE | EXTENDED |
|---|---:|---:|
| C4 | 34.2 | 18.0 |
| Dolma-V1 | 35.0 | 18.4 |
| RedPajama | 35.3 | 18.2 |
| RefinedWeb | **36.9** | **19.8** |

RefinedWeb 只从 Common Crawl 过滤得到，没有像 RedPajama 和 Dolma-V1 那样显式混入 Wikipedia 等额外高质量来源，但在这个固定协议下表现最好。因此，论文后续以 RefinedWeb-style heuristic pipeline 作为 DCLM-Baseline 的起点。

## Text Extraction

### 为什么 HTML extraction 是训练数据变量

Common Crawl 的 WET files 已经包含预抽取文本，但抽取器会决定导航栏、广告、页脚、版权声明、重复标题和正文之间如何被保留。若大量 boilerplate 进入训练数据，不仅浪费 token，也会让 quality filter 学到网站模板而不是正文质量。

论文在相同 heuristic filters 下比较三种抽取方式，使用 1B-1x scale：

| Text extraction | CORE | EXTENDED |
|---|---:|---:|
| `resiliparse` | 24.1 | **13.4** |
| `trafilatura` | **24.5** | 12.5 |
| WET files | 20.7 | 12.2 |

从性能上看，`resiliparse` 和 `trafilatura` 都明显优于 WET files；从工程效率看，`resiliparse` 大约比 `trafilatura` 快 8 倍，因此作者在 DCLM-Pool 和后续实验中选择 `resiliparse`。

附录 profiling 显示，在约 900K 个页面的样本上，`resiliparse` 平均约 1,329 tokens、吞吐约 `4.55 MB/sec/core`；`trafilatura` 平均约 1,179 tokens、吞吐约 `0.56 MB/sec/core`；WET files 平均约 2,824 tokens。这里较长的 WET 输出并不代表信息更多，其中包含更多低价值页面元素。

这个实验说明，data pipeline 的“预处理细节”本身可能改变 downstream performance，不能只记录最终保留 token 数而不记录 extractor 和 extraction failure rate。

## Deduplication

### MinHash、Suffix Array 与 Bloom Filter

论文比较了多种 deduplication pipeline：

- MinHash：以 5-token n-gram 和约 `Jaccard=0.8` 的近重复为目标；
- suffix array：移除 corpus 中重复出现、长度至少 50 tokens 的 substring；
- modified Bloom filter：同时处理 document-level 和 paragraph-level near duplicate。

在 7B-2x scale 上，Bloom filter 和 MinHash + suffix array 的 downstream performance 差异在 `0.2 CORE` 以内，但 Bloom filter 更适合超过 10TB 的数据。DCLM-Baseline 因此使用 Bloom filter；其他对照实验使用 MinHash。

论文的一个重要工程结论是：不同去重方法定义的 duplicate 并不完全相同。MinHash 更接近 document 对 document 的比较，而 modified Bloom filter 是 document / paragraph 对整个已处理 corpus 的 membership query。去重率相同不代表保留的数据分布相同。

### 去重粒度与超参

Modified Bloom filter 使用 `min_ngram_size`、`max_ngram_size`、paragraph/document removal threshold、sharding 和 false positive rate 等变量。作者在大规模实验中选择 `min_ngram_size=13`；`min_ngram_size=5` 虽然在 CORE 上可以有竞争力，但会显著损害 MMLU。原因可能是过短 n-gram 把合法的短问题、选项和列表内容误判成重复。

在 7B-2x 对照中，`Bloom Filter, min=13, 10 shards` 的 MMLU / CORE 为 `44.3 / 45.3`，与 `MinHash+Suffix Array` 的 `44.4 / 45.5` 接近；`min=5` 的 CORE 仍为 `44.5`，但论文特别指出其对 MMLU 的影响不利。

这说明 deduplication 不应只用“删除了多少 token”衡量，还需要观察不同任务、短文本、低资源数据和 benchmark contamination 的变化。

## Model-based Quality Filtering

### 不同过滤器的比较

在 1B-1x scale 上，论文比较了多种过滤器：

| Filter | CORE | EXTENDED |
|---|---:|---:|
| RefinedWeb reproduction | 27.5 | 14.6 |
| Top 20% by PageRank | 26.1 | 12.9 |
| Semantic Deduplication | 27.1 | 13.8 |
| Classifier on BGE features | 27.2 | 14.0 |
| AskLLM | 28.6 | 14.3 |
| Perplexity filtering | 29.0 | 15.0 |
| Top-k average logits | 29.2 | 14.7 |
| fastText, OH-2.5 + ELI5 | **30.2** | **15.4** |

结果有几个层次：

- PageRank 只反映网页 host 的链接中心性，不等价于文本适合 pre-training；
- semantic deduplication 受 embedding model、聚类和删除策略影响，在该设置下没有带来收益；
- AskLLM 的语义判断能力更强，但成本高、输入通常被截断，且在 downstream filtering 上不如 fastText；
- perplexity 和 top-k logits 能反映模型熟悉度或局部可预测性，但并不直接等于信息价值；
- 简单 fastText classifier 经过恰当 reference data 设计后，反而获得了最好的结果。

### fastText reference data

每个 classifier 使用约 400K examples，其中 200K positive、200K negative。negative examples 从较早的 RefinedWeb reproduction 随机抽取；positive examples 则比较了 Wikipedia、OpenWebText2、GPT-3-style mix，以及 `OpenHermes-2.5 + ELI5`。

`OH-2.5 + ELI5` 的构造方式是：从 OpenHermes-2.5 取 100K examples；从 `r/ExplainLikeImFive` 中把 post 与最高 karma 的 answer 拼接起来，并要求 post score 不小于 0、best comment score 至少为 5、总 comment 数至少为 3。作者认为这种 positive reference 同时具有 instruction / question-answer 结构和广泛 topic coverage，能够让 classifier 更好地区分有用正文与低质量网页。

在 fastText feature ablation 中，unigrams + bigrams 比只使用 unigrams 更好。最终 DCLM-Baseline 使用 `OH-2.5 + ELI5` classifier，并保留 classifier score 排名前 10% 的 documents：

| Positive reference | Threshold | CORE | MMLU | EXTENDED |
|---|---:|---:|---:|---:|
| OH-2.5 + ELI5 | top 10% | **41.0** | **29.2** | 21.4 |
| Wikipedia | top 10% | 35.7 | 27.0 | 19.1 |
| OpenWebText2 | top 10% | 34.7 | 25.0 | 18.7 |
| GPT-3 Approx | top 10% | 37.5 | 24.4 | 20.0 |
| OH-2.5 + ELI5 | top 15% | 39.8 | 27.2 | **21.5** |
| OH-2.5 + ELI5 | top 20% | 38.7 | 24.2 | 20.3 |

较严格的 top 10% 在 CORE 和 MMLU 上更好，但 top 15% 的 EXTENDED 略高。这提醒我们，过滤阈值应该作为独立实验变量，根据目标能力和覆盖范围选择，而不是默认“越严格越好”。

### Human judgment 的边界

论文让 16 名英语 AI graduate students and professors 对 499 个已经过规则过滤和 deduplication 的随机文档进行标注，每个文档得到 3 个 annotations，平均 inter-annotator agreement 为 71%，其中 281 个样本三位标注者完全一致。

作者比较了人工标签与多种 quality filter 的相关性，以及由这些 filter 产生的数据训练出的模型表现。AskLLM 与人工 majority label 的 ROC-AUC 约为 82%，高于若干 fastText filter 的约 73%；但 AskLLM 生成的数据在 CORE 上约为 28.5，而多个 fastText filter 超过 31。对 StrategyQA 和 SQuAD 等具体任务，也没有观察到稳定相关性，相关回归的 `R^2 < 0.3`。

这并不意味着人工判断没有价值，而是说明标注任务的目标必须清楚：人类判断的是“文档读起来是否高质量、是否包含有用信息”，而 pre-training filter 需要预测“把它加入大规模训练集后，是否能改善模型在指定任务上的泛化”。两者相关但不等价。

## Dataset Mixing

论文使用 Llama / RedPajama 的 `67% Common Crawl + 33% RPJ extras` 比例，向不同 Common Crawl 子集加入 Wikipedia、Books、StackExchange、arXiv 和 GitHub 等来源。1B-1x 结果如下：

| Base Common Crawl dataset | Base CORE | Mixed CORE | Base EXTENDED | Mixed EXTENDED |
|---|---:|---:|---:|---:|
| C4 | 23.7 | **25.9** | 12.5 | **13.3** |
| RedPajama CC | 24.0 | **25.7** | 12.1 | **13.5** |
| RefinedWeb | 25.1 | **26.5** | 12.9 | **13.1** |
| DCLM-Baseline | **31.1** | 29.9 | **16.0** | 15.0 |

额外来源对质量较弱的 Common Crawl 子集有帮助，但对已经由强 model-based filtering 得到的 DCLM-Baseline 反而产生平均退化。这一结果不支持“高质量来源越多越好”，更准确的结论是：混合的边际收益取决于 base dataset 已经覆盖了什么，以及新增来源能否提供互补信息。

## Decontamination

论文重点检查 MMLU 和 HellaSwag 的污染。它标记同时包含 question text 与至少一个 answer option 的页面，并删除匹配的 question / option strings。为了提高 MMLU 的 recall，对长 passage-style question 只检测最后一句，因此可能产生 false positives。

在 7B-2x scale 上，DCLM-Baseline 的原始结果为 MMLU `51.8`、HellaSwag `77.9`；删除检测到的 overlap 后，结果变为 `52.7` 和 `78.4`，没有出现性能下降。全量数据的 MMLU-specific document flag rate 约为 `0.007%`，与 Dolma-V1.7 的 `0.001%` 和 FineWeb-Edu 的 `0.009%` 处于相近数量级。

作者也使用 10-gram token overlap 对全部评测任务做更宽泛的分析，定义超过 80% token 被污染为 dirty、低于 20% 为 clean。不同污染阈值会带来 false positive 与 false negative 的取舍，因此 contamination report 不能只给出一个绝对数字，还应说明匹配粒度、阈值和清理规则。

## Scaling up DCLM-Baseline

### 7B、trillion-token 训练

为了测试 DCLM-Baseline 的优势能否扩展到远超 competition scale 的训练，作者把 3.8T-token DCLM-Baseline 与 StarCoder、ProofPile2 组合为约 4.1T-token dataset，并训练 7B 模型 2.5T tokens。

大规模训练采用两阶段与 cooldown：

1. 先按统一 recipe 训练约 2T tokens；
2. 在重新加权的 cooldown distribution 上继续训练；
3. cooldown 中约 70% 是使用更严格 top 7% fastText threshold 的 DCLM-Baseline，约 30% 是 ProofPile / math data；
4. 分别训练 200B 和 270B tokens 的两个 cooldown checkpoint；
5. 使用 model soup，以 `0.2` 权重合并 200B checkpoint、以 `0.8` 权重合并 270B checkpoint；
6. 再进行约 100B tokens 的 long-context continual learning，将 context length 从 2048 扩展到 8192。

最终模型在 7B、约 2.6T tokens 的配置下达到：

| Model | CORE | MMLU | EXTENDED |
|---|---:|---:|---:|
| DCLM-Baseline + StarCoder + ProofPile2 | **57.1** | **63.7** | **45.4** |

与公开数据训练的 MAP-Neo 7B、OLMo 等模型相比，该模型具有较好的 compute-performance trade-off，并接近使用更多私有数据训练的 Llama 3 8B、Mistral 7B 和 Gemma 8B。这里的结果来自额外的 code/math data、2.6T token 预算、cooldown、model soup 和 long-context adaptation，不能全部归因于 DCLM-Baseline 的 filtering。

### Long-context adaptation

作者从 2048 context 的 DCLM 7B checkpoint 出发，使用约 120B tokens 做 long-context continual learning：

- warm up 到 learning rate `1e-4`，再 cosine anneal 到 `1e-5`；
- 使用 64 到 8192 的 variable sequence length curriculum；
- 使用 Grow-Linear curriculum，进行 4 个 cycles；
- 将 RoPE base frequency 从 `10,000` 调整为 `100,000`；
- 训练数据按不同 sequence lengths 分配，逐步提高长 sequence 比例。

普通 CORE / MMLU 表现基本保持，同时 multi-document QA 显著提升：DCLM-8k 在 1-doc、10-doc、20-doc、30-doc 设置下的结果为 `76.9 / 49.8 / 46.1 / 38.8`。这说明长上下文能力不仅依赖更大的 position limit，还依赖训练时对长序列分布和 curriculum 的持续暴露。

## Instruction tuning 结果

虽然 DCLM 的主 benchmark 面向 base model，但论文附录还验证了 DCLM-Baseline 是否能被有效 instruction-tune。使用 OpenHermes-2.5 对 7B 模型进行 10 epochs instruction tuning 后，DCLM-Baseline 的 AlpacaEval 2.0 length-controlled win-rate 为 `13.8%`；进一步把 UltraFeedback、Tulu-v2 SFT、CodeFeedback、OpenHermes-2.5、Nectar、NoRobots、WildChat、WebInstruct 和 StarCoder2-Self-OSS-Instruct 组合成约 4M instances、8B tokens 的 DCLM-IT 后，win-rate 提升到 `16.6%`。

DCLM-IT 结果说明，强 base model 仍需要高质量 instruction data 才能释放交互能力。它也支持本文的阶段边界：data curation 可以塑造 base model，但不能替代后续针对 interaction protocol 的训练。

## 结果如何解读

### 数据改进与 hyperparameter 改进可以叠加

论文做了 learning rate 和 weight decay sweep，并发现不同 dataset 的排名在多种 hyperparameter 设置下大体稳定。进一步的 7B-1x 对照显示，fastText filtering 和更好的 learning rate / weight decay 配置带来的收益可以叠加。

这不是说数据与 optimization 永远独立，而是说明在该实验范围内，dataset design 的收益没有被某一套 hyperparameter 完全解释。进行 data ablation 时仍然需要固定并记录 optimizer、learning rate schedule、batch size、tokenizer 和训练长度。

### 数据策略可以跨 architecture 迁移

作者在 Gemma-like architecture 和 Mamba architecture 上测试数据排名，并观察到它们与 OpenLM architecture 的 CORE 结果具有较高相关性。这个结果支持数据策略具有一定跨架构迁移性，但不能理解为不同架构会得到完全相同的绝对性能。

### “小模型代理”是筛选工具，不是最终结论

小模型的作用是降低候选数据策略比较的成本，而不是替代目标规模训练。即使 dataset ranking 具有较高相关性，绝对收益、long-context 行为、代码/数学能力和训练稳定性仍可能在更大规模上发生变化。因此，合理的流程是：small proxy 快速筛选，少量 target-scale runs 验证，再进行完整训练。

## 局限与疑问

### 规模范围仍有限

由于 compute constraints，论文最多训练到 7B，没有系统测试更大 parameter scale；不同数据策略在 frontier scale 是否保持相同排序仍是开放问题。

### 只做了部分单变量消融

论文无法遍历 extraction、heuristic filtering、deduplication、model-based filtering、mixing 和 tokenizer 的所有组合，大部分实验一次只改变一个或一组变量。因此，DCLM-Baseline 是一条有效 baseline pipeline，不是已被证明全局最优的 data recipe。

### Run-to-run variation 研究不足

预训练和 downstream evaluation 存在随机性，论文没有充分测量所有配置的 run-to-run variance。对于差距较小的过滤器、去重方法和混合比例，不应过度解读单次 run 的细微差异。

### 任务覆盖偏向通用语言理解

DCLM v1 的 53 tasks 主要评估 language understanding、reading comprehension、world knowledge 和 commonsense reasoning。作者明确指出，DCLM-Baseline 在 code 和 math 上不如专门数据集；这更可能是 benchmark 和数据目标的选择结果，而不是 DCLM 方法无法扩展到 code/math。

### tokenizer、语言与安全覆盖有限

大部分实验使用 GPT-NeoX tokenizer；其他 tokenizer 在 multilingual、math 或 code 上可能有不同结果。论文将 multilinguality、fairness、safety 和更专门的 code/math coverage 列为后续扩展方向。Common Crawl、OpenHermes、StarCoder 和 ProofPile2 的许可和数据来源也需要在实际使用前单独审查。

## 对当前训练数据工程的启发

### 把数据策略变成可比较的实验变量

DataComp-LM 最重要的工程价值，是给 data pipeline 引入类似 model ablation 的实验纪律：每个数据版本都要绑定 extraction、cleaning、deduplication、quality score、threshold、mix ratio、tokenizer、token budget、训练 recipe 和评测版本。否则“数据升级”很容易变成多个变量同时变化后的不可解释结果。

### 用 small proxy 支持快速迭代

论文对多尺度 ranking 的验证，为大规模数据研究提供了现实路径：先用 400M / 1B 规模模型比较候选过滤器和混合比例，再用 3B 或 7B 做少量确认。对于百 B 级别的 agent trajectory 数据，也可以采用类似层级：先在较小模型、较短 token budget 和局部能力集上筛掉明显无效的 data recipe，再投入完整 mid-training。

### 质量过滤需要和目标能力绑定

DCLM 的 fastText classifier 并没有寻找一个抽象的“人类普遍认可的高质量”，而是用特定 reference data 和 downstream tasks 定义可用性。迁移到 agent trajectory 时，quality label 也应对应任务完成、工具调用合法性、环境反馈利用、错误恢复和轨迹完整性，而不能只按 reasoning 文本是否流畅打分。

### 把长尾覆盖和风险一起纳入评测

更严格的 top 10% filtering 在部分指标上更好，但可能损失多样性、长尾知识、代码和数学分布。实际数据配方需要同时记录保留率、domain/language coverage、重复率、contamination、toxicity、PII 和目标能力，不能只看一个 aggregate score。

## 关键结论

1. DataComp-LM 把预训练数据研究转化为一个受控 benchmark，使 filtering 和 mixing 的效果可以在统一训练与评测协议下比较。
2. DCLM-Pool 提供规模化 Common Crawl 原始数据，DCLM-Baseline 通过更好的 extraction、deduplication 和 fastText model-based filtering 获得较高的数据效率。
3. small proxy model 在该实验范围内能够较好预示 larger-scale dataset ranking，是降低数据迭代成本的重要工具。
4. `resiliparse`、Bloom filter 和 fastText OH-2.5 + ELI5 是论文 baseline pipeline 中的关键工程选择，但它们的收益来自完整组合，不能割裂成单独的万能方法。
5. 额外混合高质量来源的收益取决于 base dataset 的质量和互补性；强过滤后的数据再混合，可能反而降低性能。
6. 人工标签、LLM judge、perplexity 和下游 usefulness 衡量的是不同的质量概念，不能把其中一个当作通用 ground truth。
7. DCLM-Baseline 的强结果主要覆盖通用语言理解；代码、数学、长上下文、multilinguality 和 agentic workflow 仍需专门的数据与评测设计。
8. 对大规模训练而言，数据版本、过滤策略、训练 recipe 和评测结果必须共同记录，才能建立可复现、可归因的数据迭代闭环。

## 相关知识链接

- [[training/pretraining/data-mix|Data Mix]]
- [[training/data-engineering/quality-filtering|Quality Filtering]]
- [[training/data-engineering/deduplication|Deduplication]]
- [[training/data-engineering/data-engineering|Data Engineering]]
- [[training/pretraining/compute-optimal|Compute Optimal]]
- [[sources/papers/2023-refinedweb|RefinedWeb]]
- [[sources/papers/2023-a-pretrainers-guide-to-training-data|A Pretrainer's Guide to Training Data]]
