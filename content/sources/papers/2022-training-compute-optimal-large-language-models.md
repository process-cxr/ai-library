---
title: "Training Compute-Optimal Large Language Models"
created: 2026-05-31
published: 2026-08-28
modified: 2026-08-28
type: source
status: processed
source_type: paper
area: sources
tags:
  - source
  - paper
  - chinchilla
  - compute-optimal
  - scaling-law
  - pretraining
source_url: https://arxiv.org/abs/2203.15556
paper_date: "2022-03"
paper_order: "15556"
---

# Training Compute-Optimal Large Language Models

## 基本信息

- 标题：[Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
- 版本：arXiv:2203.15556v1，2022-03-29
- 作者：Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford 等
- 机构：DeepMind
- 研究对象：dense autoregressive Transformer language model
- 相关 topic：[[training/scaling/compute-optimal|Compute Optimal]]，[[training/scaling/scaling-law|Scaling Law]]，[[training/pretraining/compute-optimal|Pretraining Compute Optimal]]，[[training/pretraining/data-mix|Data Mix]]，[[training/scaling/training-budget|Training Budget]]

这篇论文研究一个非常具体但影响深远的训练决策问题：当训练计算预算已经确定时，应该把预算分配给更大的模型，还是分配给更多的训练 token？论文的重点不是提出新的 Transformer 结构，而是重新估计 model size 与 training data 之间的 compute-optimal frontier。

论文通过 400 多个不同规模和不同训练时长的语言模型实验，发现此前许多大语言模型确实存在 data undertraining：模型参数量增长很快，但训练 token 数长期停留在约 300B 的量级。论文据此训练了 70B 参数的 Chinchilla，在与 280B Gopher 大致相同的训练 compute 下使用约 1.4T tokens，并在多类下游任务上显著超过 Gopher 以及若干更大的模型。

如果只保留一个核心认识：**在固定训练 compute 下，增加参数量和增加训练 token 的收益需要重新平衡；一个规模更小但训练更充分的模型，可能优于一个规模更大但训练不足的模型，而且推理和后续适配成本更低。**

## 研究问题

### 大模型规模是否被过度优先扩大

论文指出，GPT-3、Gopher 和 MT-NLG 等 dense language model 大多使用约 270B-300B training tokens，但参数量已经从 175B 扩展到 530B。这样的训练路线主要扩大了 model capacity，却没有同步扩大 data budget。

这带来一个问题：在固定 compute 下，一个更大的模型虽然单步计算更贵，但可以用更少的 token 完成训练；一个更小的模型则可以看到更多 token。两者的最终 validation loss 和 downstream capability 哪一个更好，不能只凭参数量判断。

### 固定 compute 下如何分配模型与数据

论文把问题写成：给定模型参数量 $N$、训练 token 数 $D$ 和训练计算量 $C$，寻找使最终 loss 最小的组合：

$$
(N_{opt}(C), D_{opt}(C))
= \arg\min_{N,D}\ L(N,D)
\quad \text{s.t.}\quad \mathrm{FLOPs}(N,D)=C
$$

对于 dense Transformer，训练 compute 可以近似写为：

$$
C \approx 6ND
$$

这个式子是高层预算估计，不是对所有模型和训练系统都精确成立的硬公式。论文在附录中对 embedding、attention、FFN、logits 和 backward pass 分别计算 FLOPs，发现其详细计算与 $6ND$ 近似的差异较小。

### 为什么需要重新审视 Kaplan scaling law

Kaplan et al. 的早期 scaling law 认为，在增加计算预算时，模型规模应比训练 token 增长得更快。论文给出的典型估计是：计算预算增加 10 倍时，模型规模增加约 5.5 倍，而训练 token 只增加约 1.8 倍。

Chinchilla 的实验得到不同结论：在其研究范围内，随着训练 compute 增加，参数量和训练 token 数应大致按相同比例增长。论文认为，Kaplan 分析中固定 training token horizon 和固定 learning-rate schedule 的设置，可能高估了少 token 训练的效果，并低估了训练更长时间的收益。

## 核心主张

论文的核心主张可以概括为以下几点：

1. 现有许多 large language model 在给定训练 compute 下是 data-undertrained 的，参数规模相对于训练 token 数偏大。
2. 三种不同的 empirical estimation 方法都得到接近的结论：compute-optimal model size 与 training token 数随 compute 近似等比例增长。
3. 在模型参数量和训练 token 数都翻倍时，才能大致跟上 compute-optimal frontier；常见的简化经验是 dense LM 约需要 20 tokens per parameter，但这不是普适常数。
4. 用同等训练 compute 训练的 70B Chinchilla 使用约 1.4T tokens，整体显著优于 280B、约 300B tokens 的 Gopher。
5. 更充分的数据训练不仅降低 pretraining loss，也能改善 MMLU、BIG-bench、阅读理解、常识、问答和 TruthfulQA 等多类任务表现。
6. 数据扩展必须伴随数据质量、数据污染和隐私风险控制。更多低质量 token 并不等价于更多有效训练信息。

## 方法与机制

### 实验总体设计

论文训练了 400 多个 autoregressive Transformer language model，模型规模从约 70M 到超过 16B parameters，训练 token 数从约 5B 到超过 400B/500B。研究并不是只比较两个最终模型，而是通过大量不同的 $(N,D,C)$ 组合拟合 compute-optimal frontier。

实验主要在 MassiveText、C4 和 GitHub code 数据上进行。主实验使用 MassiveText，包含 web text、books、C4、news、GitHub 和 Wikipedia 等子集。主分析关注少于一个完整数据集 epoch 的训练区域；论文同时指出，多 epoch regime 需要单独研究。

训练 loss 使用平滑后的 language-modeling loss。论文把它视为在当前 data distribution 上对 test loss 的无偏估计，并在固定 compute constraint 下寻找 loss 最低的 model/data 组合。

### Approach 1：固定模型规模，改变训练 token 数

第一种方法固定一组模型规模，对每个模型训练多个不同的 token horizon。论文对每个参数规模使用四种不同的 cosine cycle length，并从训练曲线中提取不同 compute 位置上的最低 loss。

流程可以概括为：

```text
固定 model size N
  -> 训练多个不同 horizon 的模型
  -> 平滑并插值 training loss curve
  -> 得到 loss 与 FLOPs 的连续映射
  -> 对每个 compute budget 找到最低 loss 的 N 和 D
  -> 拟合 N_opt(C) 与 D_opt(C)
```

该方法直接利用完整训练曲线寻找 efficient frontier。拟合得到：

$$
N_{opt}(C) \propto C^{0.50},
\qquad
D_{opt}(C) \propto C^{0.50}
$$

### Approach 2：IsoFLOP profiles

第二种方法固定若干个 training FLOPs budget，在每个 budget 下训练不同参数规模的模型。改变模型规模时，相应调整训练 token 数，使每个模型最终消耗相同的 FLOPs。

如果把同一个 FLOPs budget 下的 loss 对 model size 作图，会出现一个 valley。谷底对应该 compute budget 下的 loss-optimal model size。论文对这些 valley 的位置进行拟合，得到：

$$
N_{opt}(C) \propto C^{0.49},
\qquad
D_{opt}(C) \propto C^{0.51}
$$

这一步提供了比 Approach 1 更直接的验证：固定 compute 时，模型太小和太大都会导致更高 loss，中间存在明确的最优规模。

### Approach 3：参数化 loss 拟合

第三种方法直接对所有实验点的最终 loss 建立参数化模型：

$$
\hat L(N,D)=E+\frac{A}{N^\alpha}+\frac{B}{D^\beta}
$$

其中：

- $E$ 表示理想生成过程在当前数据分布上的不可约 loss；
- $A/N^\alpha$ 表示有限模型容量导致的 function approximation error；
- $B/D^\beta$ 表示有限训练数据和有限优化步数导致的 stochastic approximation error。

论文使用 Huber loss 拟合 log loss，并使用 L-BFGS 与多组初始化降低局部最优的影响。拟合得到的 loss 形式为：

$$
L(N,D) \approx 1.69+\frac{406.4}{N^{0.34}}+\frac{410.7}{D^{0.28}}
$$

在 $C\approx6ND$ 约束下，推导出的 compute-optimal scaling 约为：

$$
N_{opt}(C)\propto C^{0.46},
\qquad
D_{opt}(C)\propto C^{0.54}
$$

虽然三个方法的拟合形式不同，但都支持“参数和数据需要近似等比例扩展”的结论。

### 三种方法的对照

| 方法 | 核心操作 | $N_{opt}$ exponent | $D_{opt}$ exponent |
|---|---|---:|---:|
| Minimum over training curves | 固定模型规模，比较不同训练 horizon | 0.50 | 0.50 |
| IsoFLOP profiles | 固定 FLOPs，寻找 loss valley | 0.49 | 0.51 |
| Parametric loss modeling | 拟合 $L(N,D)$ 后求 efficient frontier | 0.46 | 0.54 |
| Kaplan et al. | 早期 scaling law 估计 | 0.73 | 0.27 |

论文给出的 bootstrap 区间表明，前三种方法之间的结论相互接近，而与 Kaplan et al. 的资源分配比例明显不同。

### Learning-rate schedule 不是无关变量

论文特别指出，训练 horizon 与 cosine learning-rate schedule 必须匹配。如果目标是训练 $D$ 个 token，cosine cycle length 应该大致覆盖这段训练过程，并在训练末期完成约 10 倍的 learning-rate decay。

当 cycle length 比目标训练步数长超过约 25% 时，模型性能出现明显下降。也就是说，不能用一个为 130B token 设计的 schedule，直接把中途 loss 当作模型只训练 50B token 时的最终性能估计。

这也是论文批评 Kaplan 分析的重要原因之一：如果不同模型的 training horizon 和 schedule 没有同时调整，比较出来的 loss 可能混入 optimization schedule 的影响。

### MassiveText 数据配方

Chinchilla 的 1.4T token 训练使用 MassiveText 的调整后采样比例：

| 数据子集 | 磁盘规模 | 文档数 | Chinchilla 采样比例 | 1.4T token 下的大致 epoch |
|---|---:|---:|---:|---:|
| MassiveWeb | 1.9 TB | 604M | 45% | 1.24 |
| Books | 2.1 TB | 4M | 30% | 0.75 |
| C4 | 0.75 TB | 361M | 10% | 0.77 |
| News | 2.7 TB | 1.1B | 10% | 0.21 |
| GitHub | 3.1 TB | 142M | 4% | 0.13 |
| Wikipedia | 0.001 TB | 6M | 1% | 3.40 |

这个表说明，训练 token 数不能脱离 data mix 解释。相同的 $D$，如果来自不同的数据来源、不同的重复次数和不同的信息质量，带来的收益可能完全不同。

## Chinchilla 模型

### 与 Gopher 的同 compute 对照

根据 scaling 分析，Gopher 的训练 compute 最适合用一个约 40B-70B 参数、训练更多 token 的模型承接。论文选择了 70B 参数的 Chinchilla，并将其训练到约 1.4T tokens。

| 模型 | 参数量 | 训练 token | 训练 compute | 主要特点 |
|---|---:|---:|---:|---|
| Gopher | 280B | 约 300B | 约 $5.76\times10^{23}$ FLOPs | 大参数、短训练 |
| Chinchilla | 70B | 1.4T | 与 Gopher 近似相同 | 小参数、长训练 |

Chinchilla 的参数量约为 Gopher 的四分之一，但训练 token 数约为其四倍。它不仅 pretraining loss 更低，后续 inference 和 fine-tuning 的 memory footprint 也更小。

### 训练配置与额外差异

Chinchilla 与 Gopher 尽量保持相同的模型和训练设置，但并非只有参数量和 token 数不同：

- Chinchilla 使用 AdamW，Gopher 使用 Adam；
- Chinchilla 使用不做 NFKC normalization 的稍作修改的 SentencePiece tokenizer；
- Chinchilla 的 forward/backward 使用 bfloat16，同时在 sharded optimizer state 中保留 float32 权重副本；
- Chinchilla 使用 80 layers、64 attention heads、$d_{model}=8192$，最大 learning rate 为 $1\times10^{-4}$；
- Gopher 使用 80 layers、128 attention heads、$d_{model}=16384$，最大 learning rate 为 $4\times10^{-5}$；
- 两个模型的 batch size 都在训练中途翻倍，Chinchilla 约从 1.5M tokens 增至 3M tokens。

论文在附录中对 Adam / AdamW 和 high-precision optimizer state 做了小规模对照。AdamW 在不同 schedule 下都表现更好，高精度权重副本也有助于训练结果。这些改动说明 Chinchilla 的优势不能被严格归因于 model/data allocation 一个因素，后文的局限部分需要保留这一点。

## 实验与证据

### Language modeling

在 The Pile 的多个子集上，Chinchilla 的 bits-per-byte 都低于 Gopher，包括 GitHub、arXiv、Books、StackExchange、PubMed 和 Wikipedia 等。Wikitext-103 上，Chinchilla 的 perplexity 为 7.16，Gopher 为 7.75。

不过，论文也提醒 Chinchilla 使用了约四倍的数据，语言建模测试集的 train/test overlap 可能让结果受到 contamination 影响。因此，相比单纯的 language modeling loss，论文更重视 MMLU、BIG-bench、阅读理解、常识和 closed-book QA 等下游结果。

### MMLU

在 57 个任务的 5-shot MMLU 平均准确率上：

| 模型 | MMLU 5-shot |
|---|---:|
| GPT-3 | 43.9% |
| Gopher | 60.0% |
| Chinchilla | 67.6% |
| Human expert | 89.8% |

Chinchilla 相比 Gopher 提升 7.6 个百分点，在 57 个子任务中有 51 个更好、2 个持平、4 个更差。改善并不是所有能力均匀发生，例如 college mathematics、econometrics、moral scenarios 和 formal logic 上仍然低于 Gopher。

### BIG-bench 与阅读理解

Chinchilla 在论文选取的 62 个 BIG-bench 任务上的平均准确率为 65.1%，Gopher 为 54.4%，平均提升 10.7 个百分点。Chinchilla 只在 4 个任务上低于 Gopher。

| 任务 | Chinchilla | Gopher | GPT-3 / 其他对照 |
|---|---:|---:|---:|
| LAMBADA zero-shot | 77.4% | 74.5% | GPT-3 76.2% |
| RACE-m few-shot | 86.8% | 75.1% | GPT-3 58.1% |
| RACE-h few-shot | 82.3% | 71.6% | GPT-3 46.8% |

这些结果支持一个重要判断：更多训练 token 带来的收益不只是 validation loss 的变化，也能迁移到需要篇章理解、综合知识和多步判断的任务。

### Common sense 与 closed-book QA

在 HellaSwag、PIQA、Winogrande、SIQA 和 BoolQ 上，Chinchilla 都不低于 Gopher，并且整体优于 GPT-3。典型结果包括 HellaSwag 80.8% vs 79.2%，Winogrande 74.9% vs 70.1%，BoolQ 83.7% vs 79.3%。

在 closed-book question answering 上，Chinchilla 也优于 Gopher：Natural Questions 5-shot 为 31.5% vs 24.5%，TriviaQA unfiltered 5-shot 为 73.2% vs 63.6%。论文还报告 Chinchilla 在 TriviaQA unfiltered 上超过 GPT-3。

TruthfulQA 上，Chinchilla 的 0-shot、5-shot 和 10-shot 准确率分别为 43.6%、58.5% 和 66.7%，Gopher 的对应公开结果明显更低。这个结果说明更好地建模 pretraining data 本身，也可能改善部分事实性任务，但它并不等价于完成完整的 truthfulness alignment。

### 跨数据集一致性

论文在 C4 和 GitHub code 上复现 IsoFLOP 分析：

| 数据集 | $N_{opt}$ exponent | $D_{opt}$ exponent |
|---|---:|---:|
| MassiveText | 0.49 | 0.51 |
| C4 | 0.50 | 0.50 |
| GitHub code | 0.53 | 0.47 |
| Kaplan et al. | 0.73 | 0.27 |

不同数据集上的结果仍然接近均衡扩展，说明结论并非只由 MassiveText 的单一数据配方造成。但这个证据主要覆盖少于一个 epoch 的训练区域，不能直接推广到重复训练很多轮的 regime。

### Optimizer 与 precision 消融

论文附录比较了 Adam、AdamW 以及 optimizer state 是否保留高精度权重副本。结果显示，AdamW 训练的模型在 training loss、Wikitext perplexity 和 C4 loss 上都优于 Adam；高精度权重副本也能改善训练结果。

这组消融有两重意义：

1. 训练 recipe 中的 optimizer 和 numerical precision 会影响最终 loss，不能把所有差异都解释为参数量和数据量的差异。
2. 如果要复用 scaling law 做新训练实验，必须尽量固定 optimizer、precision、batch、learning-rate schedule 和 tokenizer，否则拟合到的可能是 recipe 差异。

### Bias 与 toxicity

Chinchilla 在 Winogender 上整体优于 Gopher，但不同性别和 stereotype 条件下的提升幅度不一致，说明更低的 language-modeling loss 不会自动消除 bias。

在 25,000 个无提示样本的 toxicity 分析中，Gopher 的 toxicity mean/median 为 0.081/0.064，Chinchilla 为 0.087/0.066，95th percentile 分别为 0.230 和 0.238。两者没有明显差异。论文据此指出，unconditional generation 的 toxicity 与 language modeling quality 之间并不存在简单的反向关系。

## 关键结论

### 1. 规模不是只有参数量一个维度

论文最重要的修正，是把“扩大模型”改写成“共同扩大 model capacity 与 training data”。当模型变大但 token 数不增加时，新增参数可能没有得到足够数据训练，最终模型不一定处于更好的 compute-optimal 点。

### 2. 20 tokens per parameter 是经验锚点，不是定律

Chinchilla 的 70B / 1.4T 配置约对应 20 tokens per parameter，后续工作经常把它简化成 $D\approx20N$。但这个比例来自特定的 dense Transformer、数据分布、训练 horizon、optimizer、schedule 和 loss 目标，不能机械迁移到：

- MoE 的 total parameters / active parameters 口径；
- 多 epoch 重复数据训练；
- long-context 训练；
- domain-specific 或 synthetic data；
- mid-training 和 agent trajectory 数据；
- 以 downstream capability 而非 pretraining loss 为唯一目标的训练。

更稳妥的做法是把 20 倍作为候选预算的初始锚点，再通过 pilot runs、domain loss 和 downstream evaluation 修正。

### 3. 训练长度必须提前设计

训练数据量、cosine cycle length、batch size 和 learning-rate schedule 共同决定一个 run 的最终状态。训练结束前的中间 loss 不能简单等价于“如果在这里提前停止的最终 loss”，尤其当 schedule 仍未进入正确的 decay 区间时。

### 4. 数据扩展把数据质量推到核心位置

当训练 token 从数百 billion 扩展到 trillion 级别后，真正的瓶颈会从“能否训练更大模型”转向“能否获取足够多、足够高质量、低污染并且风险可控的数据”。数据量、数据 mix、重复次数和数据质量必须作为同一个训练决策问题分析。

### 5. 更低的 pretraining loss 仍需通过多维评测解释

Chinchilla 的下游结果整体优于 Gopher，但仍有个别任务退化，bias 和 toxicity 也没有随着 loss 改善而自动消失。因此 compute-optimal 主要回答的是训练 loss 与资源分配问题，不能替代能力、安全和部署评测。

## 对当前训练设计的启发

### 对 mid-training 的边界

Chinchilla 研究的是从较早期的 language-model training 中选择 model size 和 training token 数。对于一个已经存在的 base model，如果模型参数量 $N$ 已经固定，问题就不再是直接寻找 Chinchilla 的 $(N,D)$ 最优点，而更接近：

```text
固定模型容量
  -> 选择新增训练 token 与数据分布
  -> 设计训练 horizon 和 schedule
  -> 观察能力收益、遗忘风险和训练稳定性
```

因此，Chinchilla 可以帮助建立预算意识，但不能直接给出 agent mid-training 的 token 配方。

### 对长轨迹数据的解释

对 agent trajectory 来说，名义 token 数并不等于有效训练信息。长轨迹中的 observation、工具输出和环境上下文可能对下一步决策有帮助，也可能包含大量重复或与当前能力目标弱相关的内容。沿用 Chinchilla 的思想，更合理的统计对象应当是经过质量、去重、任务覆盖和能力目标加权后的 effective tokens，而不是原始上下文 token 总数。

这不是论文直接验证的结论，而是将其“数据是 compute-optimal 的一条独立扩展轴”迁移到 agent 数据时得到的训练设计判断。对于长轨迹，应至少同时观察：

- total NTP loss 与按任务域拆分的 loss；
- reasoning、tool call、argument、observation 和 final answer 等不同片段的 token 占比；
- 相同任务或相同环境上下文的重复率；
- 长上下文中的关键状态是否被模型真正利用；
- agent 能力、基础语言能力和代码能力是否出现此消彼长。

### 对训练实验的具体启发

如果把这篇论文的方法用于当前训练计划，最有价值的不是直接套用 20 tokens per parameter，而是复用它的实验纪律：

1. 为不同数据量或训练 horizon 设计可比较的 pilot runs。
2. 让 learning-rate schedule 与目标训练长度匹配，避免用不适配的 schedule 解释中间 checkpoint。
3. 记录每个 run 的真实 token 数、数据 mix、重复次数、有效 token 估计和训练 FLOPs。
4. 除总 validation loss 外，维护 code、research、tool-use、reasoning、long-context 等 domain / skill validation。
5. 将训练 loss 的收益与快速 agent 评测、长期完整 agent 评测和基础能力保持情况同时分析。
6. 把 model size、data budget、数据质量和能力目标作为联合变量，而不是只比较某一个 checkpoint 的 loss。

## 局限与疑问

### 大规模直接验证仍然有限

论文的 scaling 分析覆盖了大量小型和中型实验，但在接近大模型规模的位置，真正可直接比较的训练 run 很少。Chinchilla 与 Gopher 的对照很有说服力，但仍然主要是两个大规模模型之间的验证，不是完整的大规模 IsoFLOP 网格。

### 外推跨度很大

论文需要从最高约 16B 参数的实验外推到 70B、175B、280B 甚至 1T 参数。作者观察到 FLOP-loss frontier 在高 compute 区间出现 curvature，参数最优曲线可能不是简单的全局 power law，甚至可能意味着大 compute 下更小的模型仍然更优。

### 训练 regime 受限

主分析主要覆盖少于一个 epoch 的 regime。多 epoch 训练会引入数据重复、过拟合、数据顺序和 sample reuse 等影响，不能直接使用同一套比例外推。

### Chinchilla 与 Gopher 并非完全等变量比较

两个模型之间存在 Adam / AdamW、tokenizer normalization、optimizer state precision、batch 和具体架构尺寸等差异。论文通过附录消融说明 AdamW 和高精度 optimizer state 有益，但这也意味着 70B/1.4T 相对 280B/300B 的全部优势不能严格归因于 model/data allocation。

### Loss 与 Agent 能力不是同一个目标

论文主要优化和拟合 next-token language-modeling loss，并使用若干 downstream task 做验证。对 tool selection、长程规划、状态跟踪、错误恢复等 agentic capability，平均 token loss 可能不是充分指标。Agent 训练还需要考虑 action validity、environment success、trajectory length、recovery rate 和成本等结果。

### 训练 FLOPs 不是完整工程成本

$6ND$ 近似主要描述模型计算量。实际训练还会受到通信、I/O、activation memory、checkpoint、并行效率、长上下文 attention 和硬件利用率影响。两个方案在理论 FLOPs 上相同，不代表 wall-clock time、能耗或工程风险相同。

### 数据质量与伦理风险

Chinchilla 使用大规模 web 数据，仍然存在 toxicity、bias 和潜在个人信息。增加数据规模会增加这些内容的绝对数量，因此 dataset introspection、污染检查、隐私与风险评估不能被 compute-optimal 分析替代。

## 相关知识链接

- [[training/scaling/compute-optimal|Compute Optimal]]
- [[training/scaling/scaling-law|Scaling Law]]
- [[training/scaling/model-data-compute|Model Data and Compute]]
- [[training/scaling/training-budget|Training Budget]]
- [[training/pretraining/compute-optimal|Pretraining Compute Optimal]]
- [[training/pretraining/data-mix|Data Mix]]
- [[training/pretraining/tokenizer|Tokenizer]]
- [[training/data-engineering/data-engineering|Data Engineering]]
- [[training/optimization/optimizer-state|Optimizer State]]
- [[sources/papers/2020-scaling-laws-for-neural-language-models|Scaling Laws for Neural Language Models]]
- [[sources/papers/2023-a-pretrainers-guide-to-training-data|A Pretrainer's Guide to Training Data]]
