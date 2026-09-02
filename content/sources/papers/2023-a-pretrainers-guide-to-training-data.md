---
title: "A Pretrainer's Guide to Training Data"
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
  - data-quality
  - data-mix
  - data-curation
  - temporal-generalization
source_url: https://arxiv.org/abs/2305.13169
paper_date: "2023-05"
paper_order: "13169"
---

# A Pretrainer's Guide to Training Data

## 基本信息

- 标题：[A Pretrainer's Guide to Training Data: Measuring the Effects of Data Age, Domain Coverage, Quality, & Toxicity](https://arxiv.org/abs/2305.13169)
- 版本：arXiv:2305.13169v2，2023-11-13
- 作者：Shayne Longpre, Gregory Yauney, Emily Reif, Katherine Lee, Adam Roberts, Barret Zoph, Denny Zhou, Jason Wei, Kevin Robinson, David Mimno, Daphne Ippolito
- 机构：MIT、Cornell University、Google Research、OpenAI
- 研究对象：decoder-only Transformer language model 的 pretraining data curation
- 相关 topic：[[training/pretraining/data-mix|Data Mix]]，[[training/data-engineering/quality-filtering|Quality Filtering]]，[[training/data-engineering/data-engineering|Data Engineering]]，[[training/data-engineering/deduplication|Deduplication]]，[[training/scaling/compute-optimal|Compute Optimal]]

这篇论文讨论的不是“什么数据看起来更好”，而是一个更需要实验回答的问题：**预训练数据的时间、来源、质量过滤和 toxicity 过滤，究竟会怎样改变模型的能力与风险？**

论文认为，预训练数据设计长期存在明显的 documentation debt。模型开发者往往会说明使用了哪些数据和过滤方法，却很少说明为什么这样设计，以及这些选择对最终模型产生了什么影响。于是，数据规模、质量和多样性经常被当成看似正确的经验，而不是像 learning rate、模型尺寸那样被系统地做成可比较的训练变量。

作者训练并比较了 28 个约 1.5B 参数的 decoder-only language model，围绕四类数据策划决策展开 controlled ablation：dataset age、quality filtering、toxicity filtering 和 domain composition。论文的意义不在于给出一个可以直接复制的过滤阈值，而在于证明：**数据策划策略会形成可测量、彼此冲突、且不能完全由后续 fine-tuning 消除的模型行为差异。**

## 研究问题

### 数据时间是否会影响模型的可迁移能力

预训练数据和评测数据都带有时间属性。词汇、事实、写作方式、社会事件和领域知识会随时间变化。如果模型使用较早或较晚的数据预训练，再在另一时间段的数据上评测，性能差异可能被误判为模型架构或训练规模差异。

论文要区分两件事：

- fine-tuning data 与 evaluation data 的时间不一致；
- pretraining data 与 evaluation data 的时间不一致。

前者已经有较多研究，后者在从头预训练模型上的系统证据相对不足。尤其需要回答：即使后续使用了与评测时间相关的 fine-tuning 数据，pretraining 阶段形成的时间错位是否仍然存在。

### quality 与 toxicity 是否可以用同一套过滤逻辑处理

实际数据集中，低质量、冒犯性、强烈情绪和专业内容经常同时出现，但它们并不是同一个维度。一本内容完整、语言精炼的书可能包含强烈语言；一篇专业论文可能被通用 quality classifier 判为低质量；一段低质量网页也可能完全没有明显 toxicity。

因此，论文分别研究：

- 删除被质量分类器判为低质量的文档，是否会改善模型能力；
- 删除被 toxicity classifier 判为 toxic 的文档，是否能降低 toxic generation；
- 这些过滤是否会损害模型识别毒性内容、理解其他领域问题和服务不同群体的能力。

### 领域覆盖应该优先于目标领域专门化吗

预训练数据通常混合 web、books、Wikipedia、academic、code、legal、biomedical 和 social 等来源。一个直观想法是，某个下游领域只需要加入对应的专业数据；另一个想法是，跨来源的语言、知识和表达方式本身能帮助模型泛化。

论文通过从 The Pile 中逐一删除领域，观察不同来源对不同 QA domain 的影响，试图回答：

1. 与下游任务直接匹配的 domain 是否总是最重要；
2. 大规模异构 web / books 数据是否会提供跨领域帮助；
3. 为了提升平均泛化能力，是否应该尽量保留更多数据来源。

## 核心主张

论文的结论不是“越新、越干净、越专业越好”，而是四组带有条件的经验判断：

1. **Pretraining data age matters。** 预训练数据与评测数据的时间错位会导致显著性能下降，而且在论文设置下，即使进行充分的时间相关 fine-tuning，也不能完全消除这种影响。较大模型对时间错位更敏感。
2. **Quality filtering 通常改善下游性能，但效果不能由单一质量指标预测。** 论文使用的 quality classifier 在去掉一部分数据后仍能让多数 QA 任务提升，但不同 domain 的收益并不一致，Books 甚至是明显例外。
3. **Toxicity filtering 是能力与风险之间的 trade-off。** 删除 toxic 文档可以降低 toxic generation，但也会削弱 toxicity identification，并可能降低与 toxicity 无关的通用 QA 性能。若目标是识别 toxic 内容，保留或上采样一部分高 toxicity 数据反而更有效。
4. **Domain heterogeneity 对泛化非常重要。** Common Crawl、OpenWeb 和 Books 的删除对平均 QA 性能影响最大；即使某个 domain 与目标评测直接相关，删除大型异构来源造成的损失也可能更大。

论文因此提出一个更适合训练实践的观点：**dataset curation policy 应像 learning rate 和 network dimension 一样被视为训练超参数，但它不是一个可以用单一标量和单次平均分数决定的超参数。**

## 方法与实验设计

### 总体实验管线

论文的实验管线可以表示为：

```text
选择 C4 / The Pile
  -> 对数据做时间、质量、toxicity 或 domain 干预
  -> 从头 pretrain decoder-only LM
  -> 在多个下游任务上 fine-tune / evaluate
  -> 比较相对基线的能力、风险和时间泛化变化
```

对于下游任务，作者通常先在每个任务的 training split 上 fine-tune，再在相同的 validation / test split 上评估。这样做的目的，是让不同预训练模型共享同一套下游训练流程，把差异尽量归因于 pretraining data curation，而不是后续 fine-tuning recipe。

### 预训练数据

论文选用两个公开数据集：

- **C4**：2019 年 Common Crawl snapshot，包含 news、legal、Wikipedia 和通用 web 文档；论文使用的版本取消了原始 C4 的 bad-word filter，并进一步做 approximate deduplication。
- **The Pile**：约 800GB、来自 22 个来源的混合语料，覆盖 Common Crawl、OpenWebText、Wikipedia、Books、PubMed、academic、code、legal 和 social 等来源。

为了做领域消融，作者把 The Pile 的 22 个来源合并为 9 个主题簇：`CC`、`OpenWeb`、`Wikipedia`、`Books`、`PubMed`、`Academic`、`Code & Math`、`Legal` 和 `Social`。

### 模型与训练配置

主实验使用两个 decoder-only Transformer：

- `LM-XL`：约 1.5B 参数，主要用于正式的数据策划消融；
- `LM-Small`：约 20M 参数，主要用于观察时间错位随模型规模变化的趋势。

模型使用 autoregressive next-token prediction objective，在 T5X、TensorFlow 和 TPU 上训练。主预训练配置使用 batch size 4096、sequence length 512、88,064 training steps、dropout 0，并采用统一的基础学习率和 decay 配置。论文的核心变量是数据策划，而不是模型结构或优化器探索。

### Dataset age intervention

作者从不同年份的 Common Crawl 重新构造 C4 snapshot，分别使用 2013、2016、2019 和 2022 年作为数据截断时间。为了评估 temporal misalignment，使用带年份切分的五个任务：

- `PubCLS`：新闻来源分类；
- `NewSum`：新闻摘要；
- `PoliAffs`：政治立场分类；
- `TwiERC`：Twitter 学术主题 / 实体相关分类；
- `AIC`：科学领域分类。

每个预训练模型都会在对应任务的 training-year split 上单独 fine-tune，再在多个 evaluation-year split 上测试。这样可以比较：固定 fine-tuning 时间和 evaluation 时间时，改变 pretraining 时间会带来多大差异。

### Quality filtering intervention

作者使用一个基于 bag-of-words 的 quality classifier，参考正样本主要来自 Wikipedia、Books 等被认为高质量的语料。该分类器输出从 0 到 1 的分数：0 表示更像高质量文本，1 表示更像低质量文本。

实验测试四个质量过滤阈值：`0.975`、`0.95`、`0.9` 和 `0.7`。例如，使用阈值 `0.9` 时，删除 quality score 高于 0.9 的文档。作者还设置 inverse quality filter，反向删除最高质量的一部分文档，以区分“删除低质量的收益”和“保留高质量的收益”。

论文明确提醒，这个 score 不是人类质量的 ground truth。文中所谓 high-quality / low-quality，严格说是“是否触发该自动分类器的分数区间”。这一区分很重要，因为 classifier 可能学习的是文体、来源和语言意识形态，而不是真正的事实性、可读性或教学价值。

### Toxicity filtering intervention

Toxicity 过滤使用 Jigsaw Perspective API，输出 0 到 1 的 toxicity score。实验测试删除分数高于 `0.95`、`0.9`、`0.7`、`0.5` 和 `0.3` 的文档，并设置 inverse toxicity filter，删除低于某个阈值的 least-toxic 文档。

此外还测试了原始 C4 使用的 n-gram bad-word filter。作者把 toxicity 分为两个不同目标：

- **Toxic generation**：模型是否会生成带有 profanity、sexual content、insult 或 identity attack 的文本；
- **Toxicity identification**：模型能否识别不同形式的 offensive、hateful、stereotyped 或 implicit toxic language。

这两个目标不能简单合并成一个“更安全”的指标。一个模型可能更少生成 toxic text，但也因为没有见过足够的相关语言而更难识别毒性内容。

### Domain composition intervention

作者从 The Pile 的完整混合数据出发，逐一删除九个 domain cluster 中的一个，训练对应的 LM-XL，再进行 Natural Questions fine-tuning，并在 MRQA 与 UnifiedQA 的 27 个 QA 数据集上评估。

评测数据被划分为 Wiki、Web、Books、BioMed、Academic、Common Sense 和 Contrast Sets 等类别。每个评测集可以属于多个类别，因为真实数据源本身也存在交叉。例如，Common Crawl、Wikipedia 和 Books 都可能包含 academic information，不能把 domain 边界理解成互斥的自然事实。

## 数据特征分析

在做模型实验之前，作者先分析不同语料和过滤切片的文档属性，包括 document length、readability、type-token ratio、non-ASCII character ratio、sentiment、PII、profanity、toxicity、sexually explicit content 和 quality score。

### C4 与 The Pile 的差异

与 C4 相比，The Pile 文档平均：

- 长约 `2.4x`；
- non-ASCII 字符比例约 `1.9x`；
- quality score 更高约 `1.2x`；
- readability 约 `1.8x`。

但 The Pile 也包含更多 PII，尤其是人名、地址和 email。由此可见，“更长、更可读、更像高质量文本”与“风险更低”并不是同一个方向。

### Toxicity 与 quality 并不对齐

作者观察到，high-toxicity 文档并不必然是 low-quality 文档。Books 同时表现出较高的 predicted quality、较长文档和较高的 profanity / toxicity / sexual content。这是一个非常典型的反例：如果把强烈语言简单当作低质量信号，可能会把大量有结构、有信息量的文本一起删除。

相反，PubMed、Code 和 Academic 在通用 quality classifier 上的分数较低，但它们可能只是专业表达、代码结构或论文格式与 Wikipedia / Books 的风格不同。使用单一 positive-defined quality classifier，可能会系统性压低这些领域。

### 时间变化也会改变数据特征

在不同年份的 C4 snapshot 中，较新的网页数据呈现出：

- non-ASCII 字符比例逐步增加；
- measured text quality 下降；
- toxicity 略有下降；
- sentiment 略有上升。

这说明“更新的数据”不是简单地把旧数据换成新数据。它同时带来语言、字符、主题、情绪和网页生态的变化，需要在模型能力评估中单独考虑。

## 实验与证据

### Dataset age：时间错位会留下持久影响

论文训练四个不同时间截断的 C4 模型，并在五个带时间切分的任务上评估。结果显示，pretraining year 与 evaluation year 越接近，平均性能通常越高；pretraining data 早于或晚于 evaluation data 都可能产生 degradation。

作者定义了 pretraining temporal degradation，并报告：

| 模型 | 平均 pretraining TD | 平均 Pearson correlation |
|---|---:|---:|
| LM-Small, 20M | 0.08 | 0.07 |
| LM-XL, 1.5B | 0.41 | 0.61 |

LM-XL 的时间错位效应明显强于 LM-Small。所有五个任务都通过了斜率大于零的单侧 Wald test，说明这种相关性不是单个任务的偶然现象。

更重要的是，所有模型都使用了 temporally relevant fine-tuning data，但 pretraining misalignment 仍然存在。作者还发现这种影响具有不对称性：当 evaluation data 比 pretraining data 更新时，性能下降通常更明显；但使用较新的预训练数据评估较旧任务时，也可能出现能力下降。

这会影响模型和 benchmark 的比较：较新的 benchmark 可能让旧模型显得更弱，较旧的 benchmark 也可能低估新模型在当前分布上的能力。模型 card 或训练记录至少应该报告预训练数据的时间分布，而不只报告总 token 数。

### Quality filtering：删除数据反而可以提升多数下游任务

在 C4 上，quality filtering 在去掉一部分数据后，仍然改善了大多数 QA domain 的表现。论文报告的总体趋势是：

- toxicity identification 平均提升约 `2%`；
- 多数 QA 类别提升约 `1%-6%`；
- QA 平均性能在质量阈值 `0.975` 附近达到峰值；
- 当阈值降至 `0.7`、约 `55%` 数据被过滤后，toxicity identification 仍在改善。

但收益高度依赖评测 domain。Books QA 是明显例外，quality filtering 反而可能损害其表现；Academic 和 BioMed QA 虽然对应文档在该 classifier 上的 predicted quality 较低，却从 quality filtering 中获得了更明显的收益。

这说明 quality filtering 的收益不是简单来自“删除低质量文档”。它还可能改变高质量、中间质量和 domain coverage 的相对比例。inverse quality filter 也可能在某些任务上造成不同变化，说明被 classifier 判为 high quality 的文档本身对性能有独立贡献。

另一个反直觉结果是，quality filter 与 inverse quality filter 都可能提高 toxic generation。单一 quality score 无法解释模型生成风险，甚至可能因为保留的高质量 Books / Web 内容中包含更多强烈语言而产生相反效果。

### Toxicity filtering：生成风险与识别能力的冲突

Toxicity filtering 的结果呈现清晰 trade-off：过滤越强，模型生成 toxic text 的概率通常越低，但 toxicity identification 能力和其他 QA 能力也会下降。

在 C4 的 QA 结果中，toxicity filter 逐渐增强时，平均性能从接近基线变为明显下降。以论文图表中的相对基线为例，保留约 `98%` 数据的温和过滤基本接近基线，而保留约 `61%` 数据的强过滤平均下降约 `2.7` 个百分点。这个下降不能只解释为数据量变少，因为过滤的数据内容具有特定分布。

相反，inverse toxicity filter，即主动删除最不 toxic 的数据、相对提高 toxic 文档比例，在 toxicity identification 上取得最强结果。作者因此提出：如果目标是内容审核或毒性识别，模型需要见到足够多样的 toxic language；不能为了减少生成风险而在 pretraining 阶段把相关语言全部删掉。

这并不意味着应该无条件保留 toxic data。更准确的解释是：generation safety、识别能力、通用能力和群体公平性是不同目标，需要分别设计数据、训练和后续控制策略。

### Domain composition：异构来源往往比单一目标域更重要

删除 The Pile 中的不同 domain 后，平均 QA 性能下降最大的是：

1. Common Crawl；
2. Books；
3. OpenWeb。

删除 Common Crawl 后，数据量还剩约 `73%`，但 QA 平均分相对完整数据下降约 `4.8` 个百分点；删除 Books 后约下降 `2.7` 个百分点；删除 OpenWeb 后约下降 `1.4` 个百分点。

直接删除对应目标领域并不总是造成最大损失。例如，删除 Academic 对平均 QA 的影响接近零，而删除 Common Crawl 对 Academic QA 的影响更大。论文的解释是，CC、OpenWeb 和 Books 虽然不是“专业 academic data”，却覆盖了更多主题、表达方式和跨领域知识。

完整数据或接近完整数据的模型总体表现最好。论文由此不建议为了某一类下游 QA 任务而轻易删除其他数据来源，尤其不应把 domain-specific data 当作通用异构数据的完全替代品。

但是，Books、OpenWeb 和 Common Crawl 同时也是 toxicity generation 的主要来源。删除 Books、OpenWeb 和 CC 会显著降低模型的 toxicity metrics，说明泛化能力与生成风险之间存在真实取舍，而不是简单的“所有指标一起变好”。

### 下游评测范围

论文覆盖的评测包括：

- **Domain generalization**：MRQA 与 UnifiedQA，共 30 个 QA 数据集；
- **Temporal generalization**：5 个带时间切分的任务；
- **Toxicity identification**：Social Bias Frames、DynaHate 和 ToxiGen；
- **Toxic generation**：RealToxicityPrompts 与 representational bias prompts；
- **General utility**：不同领域 QA、common sense 和 contrast sets。

毒性生成实验不进行 fine-tuning，而是直接让预训练模型生成多个 continuation。每个 prompt 生成 25 个 response，使用 top-k sampling（$k=40$）和 temperature 1.0，再由 Perspective API 以 `0.5` 为阈值判断是否 toxic。这个评测设置让作者可以测量模型在不同 prompt 条件下的生成倾向，但也带来 black-box API 与单一采样协议的限制。

## 关键结论

### 数据策划应被当成训练超参数

学习率、模型尺寸和 batch size 会改变模型训练结果，数据的时间、来源、过滤和配比同样会改变结果。数据策划不是训练前一次性的清理工作，而是定义模型最终行为分布的核心变量。

### 质量不是单一分数，也不等于来源风格

论文使用的质量分类器可以产生有价值的实验信号，但不能被解释成人类质量真值。一个 classifier 如果主要学习 Wikipedia / Books 的语言风格，就可能把代码、论文、专业文档、论坛或低资源语言误判为低质量。

因此，质量评估需要同时看信息密度、事实性、结构完整性、目标相关性、语言覆盖、重复率、PII 和风险，而不是只看一个 quality score。

### 过滤策略必须绑定目标

“减少 toxic generation”“增强 toxicity identification”“提升平均 QA”“服务低资源群体”之间可能相互冲突。不存在一个对所有任务都最优的 filter。训练数据的保留和删除，都应该通过与目标能力对应的验证集和风险指标来决定。

### 异构性是能力泛化的重要来源

数据 source 的价值不只由它与某个 benchmark 的表面相似度决定。Books、OpenWeb 和 Common Crawl 可能提供跨主题、跨文体和跨任务的组合覆盖。删除看似无关的 domain，可能损害其他 domain 的泛化。

### 时间分布需要进入数据与评测记录

新旧数据的比例会影响模型对当前和历史任务的表现。训练报告除了记录 token 数，还应该记录数据的时间直方图、抓取时间、文档时间估计方式，以及 evaluation set 与训练数据的时间关系。

## 对当前训练数据工程的启发

### 对数据分类的启发

论文并没有证明“把数据分成越多标签越好”，但它说明至少要保留几类能够影响训练决策的信息：

- 来源与任务域：web、code、research、document、tool-use 等；
- 质量与风险：quality、toxicity、PII、格式完整性、可验证性；
- 时间与版本：数据创建时间、采集时间、轨迹生成时间和环境版本；
- 训练用途：通用覆盖、目标能力、风险识别、验证集或 holdout。

对于 Agent trajectory，这意味着不能只保存总 token 数和最终成功率。轨迹还需要保留任务域、工具 / 环境来源、生成时间、轨迹质量、反馈完整性、重复程度和可验证性，才能解释不同数据 mixture 对模型的影响。

### 对长轨迹 token 统计的启发

论文直接研究的是普通文本预训练，并没有测量 agent trajectory 中 observation、reasoning、tool call 和 tool result 的 token 价值。因此，下面是迁移判断，不是论文直接结论：长轨迹的 nominal token 数不应被当作等价的有效训练量。

更适合记录：

- 总 token 数与不同事件类型的 token 占比；
- reasoning、action、tool argument、observation 和 verification 的分布；
- 同一环境、任务模板和工具输出的重复率；
- 轨迹是否包含完整反馈以及反馈是否真正与下一步决策相关；
- 各任务域的独立 validation loss 与 agent capability 指标。

这与论文的核心经验是一致的：数据量只有放回数据结构、质量、来源和目标中才有意义。

### 对数据过滤的启发

对你当前的 Agent 数据，不适合直接复用 web text 的单一 quality classifier 或“长轨迹越长越好”的判断。更稳妥的方式是：

1. 先保留完整轨迹和 provenance，避免一开始不可逆删除；
2. 按任务域、工具环境和轨迹事件分别统计质量；
3. 用多个 small proxy model 或快速能力集比较不同过滤强度；
4. 同时评估通用 NTP loss、Agent action / tool-use 指标、长程任务成功率和数据风险；
5. 记录每个版本的数据保留率、重复率、事件 token 分布与训练结果。

论文支持的是“过滤需要实验”，不是“过滤得越多越好”。

## 局限与疑问

### 模型和语言范围有限

主要实验集中在 English C4、The Pile 和约 1.5B 参数模型，LM-Small 主要用于时间效应对照。结论不能直接假设对多语言模型、超大模型、代码模型或 Agent 模型具有相同幅度。

### 计算昂贵且大多是单次实验

论文训练了 28 个模型并进行了大量 fine-tuning 与评测，但受计算成本限制，没有对每个数据策划方案进行完整多随机种子重复。结果足以说明这些变量重要，却不足以给出所有阈值的精确普适排序。

### Quality / toxicity classifier 不是 ground truth

quality classifier 依赖参考语料的文体偏好，Perspective API 依赖标注者价值判断，并且可能误判中性、专业或少数群体表达。因此，论文结果更准确地描述为“某种自动分类策略对模型的影响”，不能直接等价为“真实质量或真实 toxicity 的影响”。

### Toxicity 评测依赖 black-box API

Perspective API 的实现可能变化，且 toxic generation 只覆盖有限 prompt、sampling 和语言场景。论文自己也承认，这类指标不能完整代表真实部署中的安全风险。

### 主要评估是 fine-tuned setting

论文重点评估预训练模型经过下游 fine-tuning 后的表现，不能直接证明相同数据策划对 zero-shot、few-shot 或纯 base-model 能力的影响。

### 没有覆盖 Agent trajectory

论文不涉及多轮工具调用、环境 observation、reasoning content、长程状态跟踪或 trajectory-level success。因此，它不能直接回答哪些 Agent 轨迹 token 最有价值，也不能直接给出 Agent 数据的 optimal mix。它能够提供的是更上游的实验原则：把数据时间、来源、质量、风险和混合比例作为可测量的训练变量，并用与目标能力匹配的评测闭环验证。

## 相关知识链接

- [[training/pretraining/data-mix|Data Mix]]
- [[training/data-engineering/quality-filtering|Quality Filtering]]
- [[training/data-engineering/data-engineering|Data Engineering]]
- [[training/data-engineering/deduplication|Deduplication]]
- [[training/scaling/compute-optimal|Compute Optimal]]
- [[sources/papers/2021-the-pile|The Pile]]
- [[sources/papers/2024-datacomp-lm|DataComp-LM]]
- [[sources/papers/2022-training-compute-optimal-large-language-models|Training Compute-Optimal Large Language Models]]
