---
title: "Language Models are Few-Shot Learners"
created: 2026-06-01
published: 2026-08-31
modified: 2026-08-31
type: source
status: processed
source_type: paper
area: sources
tags:
  - source
  - paper
  - gpt
  - decoder-only
  - language-model
  - in-context-learning
  - scaling
  - benchmark-contamination
source_url: https://arxiv.org/abs/2005.14165
paper_date: "2020-05"
paper_order: "14165"
---

# Language Models are Few-Shot Learners

## 基本信息

- 标题：[Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- 作者：Tom B. Brown、Benjamin Mann、Nick Ryder、Melanie Subbiah 等
- 机构：OpenAI
- 版本：arXiv:2005.14165v4，2020-07-22
- 模型：GPT-3，175B-parameter autoregressive language model
- 训练硬件：Microsoft 提供的 high-bandwidth cluster，V100 GPUs
- 上下文窗口：2,048 tokens
- 相关主题：[[architecture/model-families/gpt|GPT]]，[[architecture/transformer/decoder-only-transformer|Decoder-only Transformer]]，[[application/prompting/few-shot|Few-shot Prompting]]，[[training/scaling/scaling-law|Scaling Law]]，[[training/pretraining/data-mix|Data Mix]]

这篇论文的核心问题不是“怎样为一个任务训练一个更好的模型”，而是：**一个只做 language-model pre-training、没有针对测试任务进行 gradient update 的模型，能否通过自然语言指令和少量示例，在 inference time 适应新任务？**

GPT-3 以 175B parameters 回答了这个问题。它没有使用 task-specific fine-tuning，而是把任务说明、demonstrations 和待解决输入组织成一段文本，让模型继续生成答案。论文将这种方式称为 `in-context learning`：模型不更新参数，适应过程发生在当前 forward pass 的 activations 与上下文条件中。

论文的历史意义不只在于发布了一个更大的 GPT。它把三个此前相对分离的问题连接起来：

```text
larger autoregressive LM
  -> smoother language-modeling scaling
  -> stronger use of contextual demonstrations
  -> one model / many task formats at inference time
```

但论文并没有声称规模可以解决所有问题。GPT-3 在 NLI、部分 reading comprehension、common-sense physics、长文本连贯性和事实可靠性方面仍存在明显缺陷；训练语料来自互联网，也带来 benchmark contamination、bias 和 misuse 风险。论文自身对这些边界的记录，是理解它的另一半。

## 研究问题

### 能否减少每个任务都需要 task-specific fine-tuning 的依赖

当时的主流范式是：先训练通用 language representation，再为每个下游任务准备数千到数十万条标注数据并更新模型参数。这种方式通常效果强，但存在三类问题：

- 每个新任务都需要单独收集和维护 supervised dataset；
- 狭窄的 fine-tuning distribution 可能放大 spurious correlations，削弱 out-of-distribution generalization；
- 同一个模型切换任务时，需要重新训练或保存多个 task-specific checkpoints。

GPT-3 试图把任务定义本身也放入文本中，以降低适配成本：

```text
task description + demonstrations + query
  -> language model continuation
  -> predicted label / answer / translation / completion
```

### 大模型是否更擅长从上下文中识别或学习任务

论文区分了三种不更新权重的设置：

- **Zero-shot**：只提供自然语言 instruction，不提供 demonstration；
- **One-shot**：提供一个任务示例；
- **Few-shot**：提供尽可能多的任务示例，通常由 context window 容纳 10 到 100 个 examples。

Few-shot 的提升可能来自两种不同机制：模型可能真正从 examples 中快速学习了新任务，也可能只是识别了预训练中已经见过的任务或格式。论文没有把两者强行区分，而是把 in-context learning 作为一个从 task recognition 到 task adaptation 的连续谱来研究。

### 规模化带来的只是更低 language-model loss，还是更强的通用能力

论文训练了从 125M 到 175B 的 8 个 GPT-3 model sizes，在统一的 autoregressive language modeling 设置下比较：

- cross-entropy / perplexity；
- zero-shot、one-shot、few-shot 下游能力；
- demonstrations 数量与模型规模之间的交互；
- 翻译、问答、推理、代码式符号处理和文本生成等不同任务。

它要检验的是：下游能力是否随模型规模平滑提升，尤其是 few-shot performance 是否比 zero-shot 更依赖规模。

## 核心主张

1. **Scaling autoregressive language models 可以显著提升 task-agnostic in-context learning。** GPT-3 在不进行 gradient update 或 task-specific fine-tuning 的条件下，在翻译、问答、cloze、简单算术、符号操作和文本生成等任务上获得了有竞争力的结果。
2. **Few-shot performance 通常随 model size 提升得比 zero-shot 更快。** 这说明大模型不仅语言建模能力更强，也更能利用上下文 demonstrations；但提升不等于已经证明模型在 inference time 从零学会了任务。
3. **同一个 autoregressive decoder-only model 可以通过文本格式统一多种任务。** Classification、QA、translation、completion 和 synthetic task 都被转化为 prompt plus continuation，并用 next-token likelihood 或 generation 评估。
4. **GPT-3 的能力提升与训练数据和模型规模共同相关。** 模型训练 300B tokens，数据来自经过筛选和 deduplication 的 Common Crawl、WebText2、Books1、Books2 与 Wikipedia；高质量数据被有意重复采样，数据量不能只看原始 corpus size。
5. **Scaling 的结果不是“所有任务都达到 SOTA”。** GPT-3 在 TriviaQA、CoQA、部分 translation 与 synthetic tasks 上很强，但在 ANLI、WiC、QuAC、RACE、部分 commonsense reasoning 和长文本一致性上仍明显落后。
6. **大规模 web pre-training 必须伴随 contamination analysis。** 论文使用 n-gram overlap 构造 clean subsets，并对可能被污染的 PIQA、Winograd、LAMBADA 等结果做标记或谨慎解释；但由于过滤 bug，部分 benchmark 无法彻底排除污染。
7. **纯 self-supervised next-token prediction 可能不是最终的通用智能训练目标。** 论文指出，等权预测每个 token 缺少 goal-directed action、世界交互、视觉/物理 grounding 与重要性建模，未来可能需要 human feedback、RL 或 additional modalities。

## Approach

### Fine-Tuning、Zero-Shot、One-Shot 与 Few-Shot

论文将任务适配方式放在一个连续谱上：

| 设置 | 参数更新 | 任务示例 | 适配发生在哪里 |
|---|---|---|---|
| Fine-tuning | 有 | 通常数千到数十万条 | model weights |
| Few-shot | 无 | 多个 demonstrations | 当前 context / activations |
| One-shot | 无 | 一个 demonstration | 当前 context / activations |
| Zero-shot | 无 | instruction 或 task description | 当前 prompt 与 pretrained prior |

GPT-3 论文的实验焦点是后三种设置。这里的 `learning` 是操作性术语，不应直接等同于发生了参数更新的 optimization learning。更谨慎的理解是：模型在 inference 时根据上下文完成任务识别、格式对齐和局部行为适配。

Few-shot prompt 的抽象形式为：

```text
example 1: input -> output
example 2: input -> output
...
query: input ->
```

模型的下一个 token 分布同时受到 task wording、demonstrations 顺序、分隔符、答案格式和 query 的影响。论文因此把 prompt construction 作为 evaluation protocol 的一部分，而不是把它看成模型之外的无关细节。

### Model Architecture

GPT-3 延续 GPT-2 的 decoder-only Transformer，并使用：

- modified initialization；
- pre-normalization；
- reversible tokenization；
- alternating dense attention 与 locally banded sparse attention pattern；
- 2,048-token context window。

论文训练 8 个主要模型：

| Model | Parameters | Layers | $d_{model}$ | Heads | Batch size | Learning rate |
|---|---:|---:|---:|---:|---:|---:|
| GPT-3 Small | 125M | 12 | 768 | 12 | 0.5M | $6.0\times10^{-4}$ |
| GPT-3 Medium | 350M | 24 | 1,024 | 16 | 0.5M | $3.0\times10^{-4}$ |
| GPT-3 Large | 760M | 24 | 1,536 | 16 | 0.5M | $2.5\times10^{-4}$ |
| GPT-3 XL | 1.3B | 24 | 2,048 | 24 | 1M | $2.0\times10^{-4}$ |
| GPT-3 2.7B | 2.7B | 32 | 2,560 | 32 | 1M | $1.6\times10^{-4}$ |
| GPT-3 6.7B | 6.7B | 32 | 4,096 | 32 | 2M | $1.2\times10^{-4}$ |
| GPT-3 13B | 13.0B | 40 | 5,140 | 40 | 2M | $1.0\times10^{-4}$ |
| GPT-3 / 175B | 175.0B | 96 | 12,288 | 96 | 3.2M | $0.6\times10^{-4}$ |

所有模型都训练约 300B tokens。论文中的参数量是 total trainable parameters；它没有采用 MoE，因此不像 Switch 或 DeepSeek-V2 那样需要区分 total 与 activated parameters。

GPT-3 采用 model parallelism：矩阵乘内部切分，并沿 network depth 跨 GPU 切分，以降低节点间数据传输。其核心 scaling 方向是 dense model scaling，而不是 sparse conditional computation。

### Training Objective

GPT-3 使用标准 autoregressive language modeling objective：

$$
\mathcal{L}_{NTP}(\theta)
=-\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t})
$$

训练时没有 task-specific labels，也没有为测试 benchmark 进行 gradient update。论文的关键假设是：如果语料足够广、模型足够大，许多任务和任务格式会以 latent task distribution 的形式出现在预训练数据中，模型可以在 inference 时借助上下文把这种能力调出来。

这一假设与今天的 agent 训练有直接区别：GPT-3 的 base model 学到的是文本中的 task pattern 和 continuation distribution，而不是显式的 tool protocol、environment transition 或任务成功 reward。它说明了“行为可以通过序列分布被预训练”，但不等于纯 NTP 已经充分解决长期规划和环境交互。

## Training Data

### 数据来源与最终 mixture

作者没有直接使用原始 Common Crawl，而是先做 quality filtering、fuzzy deduplication，再加入高质量 reference corpora：

| Dataset | Available tokens | Training mixture weight | 训练 300B tokens 时的 epoch |
|---|---:|---:|---:|
| Filtered Common Crawl | 410B | 60% | 0.44 |
| WebText2 | 19B | 22% | 2.9 |
| Books1 | 12B | 8% | 1.9 |
| Books2 | 55B | 8% | 0.43 |
| Wikipedia | 3B | 3% | 3.4 |

表中的 available tokens 与 training mixture weight 不成比例。Common Crawl 和 Books2 因质量或规模原因没有被完整遍历，而 WebText2、Books1、Wikipedia 被更频繁采样。这个 data mix 是一种质量优先的重采样：作者接受部分高质量数据的重复，以提升平均训练数据质量。

### Common Crawl 过滤

过滤器以原始 WebText、Wikipedia 和 web books 等 curated corpora 作为 positive examples，以未过滤 Common Crawl 作为 negative examples，训练 logistic regression classifier。特征使用 Spark tokenizer 和 HashingTF。对每个 Common Crawl document 得到 quality score 后，采用 Pareto-based sampling：高分文档更容易被保留，但仍保留一部分分布外样本。

论文报告，这种 re-weighting 让模型在多种 out-of-distribution generative text samples 上的 loss 更低。这里的含义是提高平均质量与泛化，而不是证明 classifier score 等同于 document truth 或教学价值。

### Fuzzy Deduplication

作者在各数据集内部以及数据集之间做 document-level fuzzy deduplication，并额外去除 Common Crawl 中与 WebText 高度重叠的内容。使用 Spark MinHashLSH 与 10 个 hashes，对高 overlap documents 做删除，最终数据规模平均减少约 10%。

Deduplication 的目标有两个：

- 减少无信息重复，避免模型过度重复学习相同文本；
- 让 held-out validation 与 benchmark contamination analysis 更可信。

这也说明“300B training tokens”不能只按照未去重的原始文本量理解。真正决定训练分布的是 tokenizer 后 tokens、source mixture、resampling policy、deduplication 与数据过滤共同形成的有效 sample distribution。

## Training Recipe and Compute

### 优化设置

论文附录给出的通用设置为：

- Adam，$\beta_1=0.9$、$\beta_2=0.95$、$\epsilon=10^{-8}$；
- global gradient norm clipping 为 1.0；
- weight decay 为 0.1；
- 前 375M tokens 做 linear learning-rate warmup；
- 之后在 260B tokens 上做 cosine decay 到初始 learning rate 的 10%；
- 260B tokens 后继续以原始 learning rate 的 10% 训练；
- batch size 在前 4B 到 12B tokens 内，从约 32K tokens 线性增长到完整 batch；
- 数据在 epoch boundary 前尽量 without replacement sampling，以减少重复带来的 overfitting；
- 训练始终使用完整 2,048-token sequences。

这些细节对 scaling law 的可比性很重要。GPT-3 的模型规模、batch size、learning rate、warmup、数据重复和训练 horizon 是绑定的，不能只抽取 175B 参数和 300B tokens 作为独立配方。

### Packing 与文档边界

长度不足 2,048 tokens 的多个 documents 会被 packing 到同一个 sequence 中，以提高计算利用率。不同 documents 之间不使用特殊 sequence-level attention mask，而是用 end-of-text token 分隔，让 language model 通过显式 boundary token 学习它们不属于同一篇文档。

这是一种计算效率与语义隔离之间的折中：它避免了额外的 document-specific mask，但要求 tokenizer、boundary token 和模型训练充分学会 document reset 语义。对长轨迹训练而言，这种 packing 设计不能直接照搬，必须先确认不同事件之间是否允许 attention 互相读取。

### 训练计算

GPT-3 175B 的总训练计算约为：

| Model | Parameters | Training tokens | Total compute |
|---|---:|---:|---:|
| GPT-3 Small | 125M | 300B | $2.25\times10^{20}$ FLOPs |
| GPT-3 2.7B | 2.65B | 300B | $4.77\times10^{21}$ FLOPs |
| GPT-3 13B | 12.85B | 300B | $2.31\times10^{22}$ FLOPs |
| GPT-3 175B | 174.6B | 300B | $3.14\times10^{23}$ FLOPs |

论文用约 3.64K PF-days 表示 GPT-3 175B 的 pre-training compute。该估算主要按 active parameters、training tokens 和 forward/backward multiplier 计算，并忽略 attention 的部分成本；它适合做跨模型的高层比较，不等价于 wall-clock、energy、通信和集群利用率的完整账本。

## Evaluation Protocol

### Prompt 与 demonstrations

对 evaluation set 中的每个 example，作者从该任务的 training set 随机抽取 $K$ 个 examples 作为 conditioning；$K$ 从 0 到 context window 能容纳的最大数量。通常可以放入 10 到 100 个 examples，但不同任务的 demonstration 长度不同。

当存在 development set 时，作者在 development set 上尝试若干 $K$，再把选定的值用于 test set。没有 supervised training set 的 LAMBADA 和 StoryCloze 从 development set 抽取 demonstrations，在 test set 上评估。

这套 protocol 有几个需要注意的变量：

- demonstration 的随机抽样会改变结果；
- example 顺序、分隔符和 task wording 会影响输出分布；
- zero-shot 往往使用与 one/few-shot 不同的 prompt format；
- 不同 benchmark 的输出形式和 scoring method 不一致。

因此，few-shot score 不是只由模型参数决定，也反映了 prompt construction、示例选择和 evaluator protocol。

### Scoring

对于 multiple-choice tasks，prompt 包含 demonstrations 后，模型对每个 completion 计算 likelihood，通常用 per-token likelihood 做长度归一化。在 ARC、OpenBookQA 和 RACE 上，作者还尝试除以 completion 在 generic answer context 下的 unconditional probability，以减弱答案格式或先验频率的影响。

对于 binary classification，作者把选项换成有语义的词，例如 `True` / `False`，再按照 multiple-choice likelihood 评估；对于 free-form completion，则使用 beam search，并按任务采用 F1、BLEU 或 exact match。

这套设计展示了 decoder-only LM 如何承载 classification，但也带来 calibration 与 label-token bias：如果一个选项的表面文本更常见，token likelihood 可能部分反映语言先验，而不只是任务判断。

## 主要实验结果

### Language Modeling 与 Completion

GPT-3 在 Penn Tree Bank 上的 zero-shot perplexity 为 20.5，论文称相对先前结果有明显提升。LAMBADA 测试模型利用段落上下文预测最后一个词：

| Setting | LAMBADA accuracy | LAMBADA perplexity | StoryCloze | HellaSwag |
|---|---:|---:|---:|---:|
| GPT-3 zero-shot | 76.2 | 3.00 | 83.2 | 78.9 |
| GPT-3 one-shot | 72.5 | 3.35 | 84.7 | 78.1 |
| GPT-3 few-shot | 86.4 | 1.92 | 87.7 | 79.3 |

LAMBADA 的 one-shot 反而低于 zero-shot，说明 demonstration 并非在任何格式下都产生正收益；few-shot 的显著提升依赖模型看到足够多的“填空式” examples，并理解所需的 exact completion format。

### Closed-Book Question Answering

模型不接收外部 retrieved documents，也不对 QA 数据集 fine-tune，只依赖参数中的知识与 prompt demonstrations：

| Setting | Natural Questions | WebQuestions | TriviaQA |
|---|---:|---:|---:|
| GPT-3 zero-shot | 14.6 | 14.4 | 64.3 |
| GPT-3 one-shot | 23.0 | 25.3 | 68.0 |
| GPT-3 few-shot | 29.9 | 41.5 | 71.2 |

TriviaQA few-shot 达到当时 closed-book setting 中有竞争力的结果；Natural Questions 提升明显但仍低于 fine-tuned T5，可能因为 NQ 更偏向 Wikipedia 上的细粒度知识与特定答案格式。这个差异说明 model scale 增加了参数知识，但不能替代 retrieval、任务适配或严格输出控制。

### Translation

GPT-3 使用以英语为主的自然语言混合语料，没有专门的 translation objective。论文在英语、法语、德语和罗马尼亚语之间评估：

| Setting | En -> Fr | Fr -> En | En -> De | De -> En | En -> Ro | Ro -> En |
|---|---:|---:|---:|---:|---:|---:|
| GPT-3 zero-shot | 25.2 | 21.2 | 24.6 | 27.2 | 14.1 | 19.9 |
| GPT-3 one-shot | 28.3 | 33.7 | 26.2 | 30.4 | 20.6 | 38.6 |
| GPT-3 few-shot | 32.6 | 39.2 | 29.7 | 40.6 | 21.0 | 39.5 |

Few-shot 超过当时部分 unsupervised NMT 工作，但方向不对称：翻译成英语明显强于从英语翻译出去。论文将其与训练语料主要为英文、tokenizer 主要为英文优化联系起来。该结果说明多任务能力可以从自然混合语料中出现，也说明 data mix 和 tokenizer 会深刻影响跨语言能力。

### Reasoning 与 Reading Comprehension

GPT-3 展现了“部分任务可迁移、部分任务仍然困难”的结果：

- PIQA few-shot：82.8%，但存在潜在 contamination 标记；
- ARC Challenge few-shot：51.5%，显著低于当时 fine-tuned SOTA；
- OpenBookQA few-shot 相比 zero-shot 有明显提升，但仍低于整体 SOTA；
- CoQA few-shot：85.0 F1，接近当时 human / fine-tuned performance；
- QuAC few-shot：44.3 F1，明显较弱；
- DROP few-shot：36.5 F1，显示离散推理与数值计算仍是瓶颈；
- RACE-m / RACE-h few-shot：58.1 / 46.8，远低于强 fine-tuned 系统。

共同现象是：模型擅长把 task format 组织成语言延续，并在一些自然分布任务中受益于规模；但需要比较句子、执行结构化对话行为、精确处理数值或长段落证据时，单纯扩大 autoregressive LM 并不能保证稳健。

### SuperGLUE 与 NLI

GPT-3 在 SuperGLUE 上使用每个 task 32 个 examples，整体 few-shot score 为 71.8。分任务结果差异很大：

| Task | GPT-3 few-shot | Fine-tuned SOTA |
|---|---:|---:|
| BoolQ | 76.4 | 91.0 |
| CB | 75.6 F1 | 96.9 |
| COPA | 92.0 | 93.9 |
| RTE | 69.0 | 94.8 |
| WiC | 49.4 | 76.1 |
| WSC | 80.1 | 93.8 |
| MultiRC accuracy | 30.5 | 62.3 |
| ReCoRD accuracy | 90.2 | 92.5 |

WiC 几乎处于随机水平，反映出比较一个词在两个句子中的语义用法对 in-context learning 很难。ANLI 也显示类似现象：小模型接近随机，175B model 在 Round 3 才开始明显高于 chance，但仍距离强 fine-tuned 系统较远。

这组结果提醒我们，不要把 aggregate benchmark average 当成“通用理解能力”的单一尺度。需要按任务结构拆分：生成格式、证据读取、句间比较、常识推理和世界知识可能遵循不同的 scaling curve。

### Synthetic 与 On-the-Fly Tasks

为了避免只测试可能出现在 web corpus 中的固定任务，作者设计了随机 arithmetic、word scrambling、SAT-style analogy、新词定义与语法纠错等任务。

GPT-3 175B 的 arithmetic few-shot 结果：

| Task | Zero-shot | One-shot | Few-shot |
|---|---:|---:|---:|
| 2-digit addition | 76.9 | 99.6 | 100.0 |
| 2-digit subtraction | 58.0 | 86.4 | 98.9 |
| 3-digit addition | 34.2 | 65.5 | 80.4 |
| 3-digit subtraction | 48.3 | 78.7 | 94.2 |
| 4-digit addition | 4.0 | 14.0 | 25.5 |
| 4-digit subtraction | 7.5 | 14.0 | 26.8 |
| 5-digit addition | 0.7 | 3.5 | 9.3 |
| 5-digit subtraction | 0.8 | 3.8 | 9.9 |
| 2-digit multiplication | 19.8 | 27.4 | 29.2 |
| 1-digit composite operation | 9.8 | 14.3 | 21.3 |

作者抽查了 3-digit arithmetic problems 在训练集中的字符串匹配，发现 addition 约 0.8%、subtraction 约 0.1% 有匹配，且错误中经常出现没有进位等计算型错误。这支持模型并非只靠完整题目记忆，但不能证明其内部使用了可靠的 symbolic algorithm。

对 word manipulation tasks，few-shot 结果为：cycle letters 37.9%、anagrams excluding first/last letter 15.1%、anagrams excluding first/last two letters 39.7%、random insertion 67.2%、reversed words 0.44%。这些结果显示模型能从 examples 中获得一部分新规则，但不同规则的难度差异很大。

### Synthetic News Generation

作者还让模型根据 news title 和 subtitle 生成约 200-word news articles，并邀请人类判断文章是 human-written 还是 model-generated。随着模型规模增加，人类区分模型文本和真实文本的能力下降，参与者判断大模型输出时也花费更多时间。

这项实验不是事实准确性评估，而是文本表面自然度与可辨识度评估。它说明 scale 提高了生成文本的 fluency 与 plausibility，同时也增加了 misinformation、spam 和 social engineering 等 misuse 风险。

## Scaling 观察

### Training loss 与 downstream performance

GPT-3 的训练曲线延续此前 language-model scaling 的 power-law 趋势。模型从 125M 扩展到 175B，cross-entropy validation loss 仍然大致平滑下降，只有轻微偏离。

下游能力通常也随着规模提升，但并非所有任务以同样速度提升：

- zero-shot 通常更平滑；
- one-shot 与 few-shot 在大模型上受益更多；
- demonstrations 数量增加通常有帮助，但不是单调且无条件有效；
- arithmetic、symbol manipulation 等 synthetic task 可能在 13B 到 175B 之间出现明显跃升；
- NLI、reading comprehension 等结构复杂任务仍可能在大模型上接近 chance 或远低于 SOTA。

因此，论文的“emergence”更适合表述为：离散 benchmark 上的能力曲线在规模扩大后变得可观测，背后可能仍是连续的 loss 与 representation improvement。不能仅凭某个 threshold 分数断言出现了全新的内部算法。

### In-context learning curve

GPT-3 的 few-shot 结果体现了 model size 与 context examples 的交互：

```text
larger model
  -> better use of task description
  -> steeper improvement as demonstrations increase
  -> larger gap between zero-shot and few-shot on many tasks
```

不过，更多 demonstrations 也会消耗 context window，并可能引入不一致示例、格式噪声、答案泄漏或示例顺序偏差。`K` 不是越大越好，而是 prompt budget、example quality 和 task structure 的联合变量。

## Benchmark Contamination

### 为什么 web-scale pre-training 会遇到 contamination

Common Crawl、Books 和 Wikipedia 可能包含 benchmark 的题目、背景材料、答案或近似改写。模型参数量越大，memorization 的可能性越高；如果 test example 出现在训练 corpus 中，benchmark score 可能高估真实泛化能力。

GPT-3 采取了两步处理：

1. 训练数据阶段尝试搜索 benchmark train/dev/test 与预训练数据的 13-gram overlap，删除 collision 附近文本；
2. 训练后对每个 benchmark 构造 clean subset，用从训练数据中未发现 overlap 的 examples 与全量 score 对比。

但训练过滤程序存在 bug，长文档如 books 的 overlap 未被完整处理；由于重新训练成本过高，作者无法用完全修复后的数据重训 GPT-3。

### 清洁子集结果如何解释

多数 benchmark 的 clean/all performance difference 很小，作者没有观察到 contamination 比例与 performance gain 之间稳定相关。但这不是严格的无污染证明：

- n-gram overlap 可能产生 false positives；
- clean subset 与 full dataset 可能不是同一分布；
- background passage 出现在训练集，不等同于题目答案被记忆；
- 评测中随机抽取 demonstrations 也会带来额外方差。

论文因此对 PIQA、Winograd 和 LAMBADA 等结果加注说明，并直接放弃几乎完全被训练集覆盖的部分 Wikipedia language-modeling benchmarks 与 Children’s Book Test。对现代训练系统而言，contamination detection 应在数据构造、checkpoint 评估和 benchmark 发布前持续进行，而不是只在最终模型上补一个分析。

## 局限与疑问

### In-context learning 是否真的在学习新任务

论文没有解决最根本的机制问题：模型可能是在 test-time 学会了 task，也可能是在预训练数据中识别了相似 task、格式或知识分布。synthetic word manipulation 更接近 de novo adaptation，translation 则很可能依赖预训练中已有的 multilingual pattern；不同任务可能位于这个 spectrum 的不同位置。

### Pure NTP 的目标边界

等权预测每个 token 并不区分关键事实、格式 token、噪声和目标相关 action。语言模型可能在文本 likelihood 上继续变好，却不一定更可靠地完成 goal-directed behavior。论文明确指出，未来可能需要：

- 以人类目标为基础的 objective；
- reinforcement learning；
- visual、physical 或其他 modalities 的 grounding；
- 能表达 token importance 或 task utility 的训练信号。

### 上下文窗口限制

GPT-3 的 2,048-token context 限制了 demonstrations 数量、长文档读取和复杂任务状态保持。长上下文不是简单把 `n_ctx` 调大：还需要处理 position representation、attention compute、KV memory、示例选择、长距离依赖与评测可靠性。

### 任务能力不均衡

GPT-3 在知识问答、翻译成英文、自然文本生成和某些格式学习任务上强，在句间比较、结构化对话行为、common-sense physics、精确数值推理和长程 coherence 上弱。模型规模无法替代任务结构、外部工具、检索或专门验证器。

### 数据污染与复现边界

训练数据 mixture、quality classifier、sampling details 和完整 contamination-cleaning pipeline 没有完全公开；模型训练和评估使用了内部 infrastructure。论文的 scaling 结论因此具有重要参考价值，但不能被视为无需重新验证的 universal law。

### 可靠性、公平性与安全

作者观察到 GPT-3 继承互联网语料中的 gender、race 和 religion biases，也可能生成可用于 misinformation、phishing、spam、fraud 和 social engineering 的文本。模型输出并非总是 calibrated、interpretable 或 factually grounded；生成文本更自然不等于生成内容更真实。

## 对训练系统和当前研究的启发

### 把上下文当成能力接口，但不要把 prompt gain 当成参数学习

GPT-3 证明了任务格式、demonstrations 和 query 可以共享一个 decoder-only interface。对 agent 或 workflow 数据来说，类似接口可以把：

```text
task / state / available tools / prior actions / observation
  -> next action or response
```

组织成统一序列。但需要单独记录哪些 token 是 environment context、哪些 token 是 assistant action、哪些 token 是 result；in-context adaptation 与参数更新的训练收益不能混为一谈。

### Data mix 的“质量优先”不等于只保留高分数据

GPT-3 的 mixture 不是按原始 corpus size 等比例采样，而是提高 WebText2、Books1、Wikipedia 等来源的出现频率，同时保留一部分 Common Crawl 的覆盖。迁移到大规模 agent trajectory 时，类似的设计原则是：记录原始规模、过滤后规模、采样权重、重复次数和有效 token 分布，避免用名义 token 数代替训练 exposure。

### Long sequence packing 必须验证边界语义

GPT-3 使用 end-of-text token 分隔 packed documents，而不是为每个 document 加独立 attention mask。对于 agent trajectory，这种方案只有在相邻片段确实允许共享上下文、模型能够可靠识别 boundary 时才成立。若 observation、tool result 或不同样本之间存在不可见性约束，应使用显式 mask、事件标记或其他结构化边界，而不能只靠分隔符。

### 用多维评测解释 scaling

GPT-3 的经验说明，平均 loss、zero-shot、few-shot、任务格式和数据污染共同决定结论。当前训练实验也应至少分别维护：

- token-level validation loss；
- domain / task-specific loss；
- zero-shot 与 few-shot 行为；
- 长上下文位置与信息检索；
- 工具、代码或环境交互任务成功率；
- contamination 与数据重复审计。

单一 aggregate score 无法解释模型到底学会了知识、格式、局部模式还是稳定的任务策略。

### Synthetic task 是检验迁移而非证明推理的工具

GPT-3 的随机 arithmetic 与 word manipulation 任务通过降低训练集直接记忆的可能性，测试模型是否能从 examples 中执行新规则。这种思路适合构造快速诊断集，但模型答对不等于内部形成可复用 symbolic procedure。对 agent 模型，还应增加 counterfactual state、unseen tool schema、invalid action recovery 和 long-horizon execution tests。

### 将 scaling law 用于预算规划时保留不确定性

GPT-3 将 model size 扩展到 175B 后仍观察到平滑趋势，但当时的 training regime、data mix、context length、dense architecture 和 compute-optimal 假设都有限。后续 Chinchilla 已重新估计 model-data allocation。新训练计划应把 scaling curve 当作相似配方下的预测工具，并通过小规模 pilot、domain validation 和下游任务验证，而不是只按参数倍率外推。

## 关键结论

1. GPT-3 的核心贡献是证明大规模 autoregressive LM 可以在不更新参数的情况下，通过 task description 与 demonstrations 完成多种任务的 in-context adaptation。
2. Few-shot performance 往往比 zero-shot 更依赖模型规模，但它可能混合了 task recognition、format induction、知识调用和真正的新任务适应。
3. GPT-3 使用 dense decoder-only Transformer、300B training tokens、2,048 context 和 quality-weighted data mixture；它不是仅靠 175B 参数得出结果。
4. Data filtering、fuzzy deduplication、sampling without replacement、packing 与 contamination analysis 是训练结果可信度的重要组成部分。
5. 模型规模带来的 loss 改善能迁移到许多下游任务，却不能消除任务结构差异；NLI、长文理解、数值推理、common-sense physics 和事实可靠性仍有明显短板。
6. GPT-3 的 synthetic task 与 in-context curves 为研究 test-time adaptation 提供了可复用的评测思路，但不能单凭行为结果推断内部推理机制。
7. 论文自身已经指出 pure self-supervised prediction 的边界：goal-directed action、world grounding、token importance、human feedback 和 RL 可能是继续提升通用 agentic capability 所需的补充。

## 相关知识链接

- [[architecture/model-families/gpt|GPT]]
- [[architecture/transformer/decoder-only-transformer|Decoder-only Transformer]]
- [[application/prompting/few-shot|Few-shot Prompting]]
- [[training/pretraining/objective|Training Objective]]
- [[training/pretraining/data-mix|Data Mix]]
- [[training/data-engineering/deduplication|Deduplication]]
- [[training/scaling/scaling-law|Scaling Law]]
- [[training/scaling/model-data-compute|Model, Data and Compute]]
- [[sources/papers/2020-scaling-laws-for-neural-language-models|Scaling Laws for Neural Language Models]]
- [[sources/papers/2022-training-compute-optimal-large-language-models|Training Compute-Optimal Large Language Models]]
- [[sources/papers/2022-instructgpt|Training language models to follow instructions with human feedback]]
