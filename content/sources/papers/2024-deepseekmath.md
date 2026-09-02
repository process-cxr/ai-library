---
title: "DeepSeekMath"
created: 2026-05-29
published: 2026-08-31
modified: 2026-08-31
type: source
status: processed
source_type: paper
area: sources
tags:
  - source
  - paper
  - reasoning
  - grpo
  - rlvr
  - math
  - data
source_url: https://arxiv.org/abs/2402.03300
paper_date: "2024-02"
paper_order: "03300"
---

# DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models

## 基本信息

- 论文：[DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- 作者：Zhihong Shao、Peiyi Wang、Qihao Zhu、Runxin Xu 等
- 发布信息：arXiv:2402.03300，v3 发布于 2024-04-27
- 开源代码：[deepseek-ai/DeepSeek-Math](https://github.com/deepseek-ai/DeepSeek-Math)
- 研究对象：7B 规模的数学 reasoning language model
- 相关主题：[[training/pretraining/data-mix|Data Mix]]、[[training/mid-training/continued-pretraining|Continued Pretraining]]、[[training/post-training/grpo|GRPO]]、[[training/post-training/rejection-sampling|Rejection Sampling]]、[[training/post-training/knowledge-distillation|Knowledge Distillation]]

这篇论文同时讨论了 math pre-training 和 reasoning RL。它的贡献不应只被概括为“提出了 GRPO”，因为论文的结果来自一条连续的训练链路：先从 Common Crawl 中筛出规模化、跨语言的数学语料，再用 code 与 math 的训练配方塑造具有较强数学先验的 base model，之后通过数学 instruction tuning 建立可交互的解题行为，最后用可验证 reward 进行 GRPO。

从研究方法上看，论文最值得保留的是两个观察：一是**数据选择和数据混合方式可以在不扩大模型规模的情况下显著改变数学 reasoning 能力**；二是**在已有 reasoning 能力上进行 RL 时，训练效果取决于数据来源、reward function 与 gradient coefficient 的组合，而不只是算法名称**。

## 研究问题

### 如何从开放网页构造高质量数学语料

数学训练数据通常规模有限，且高质量内容分散在教材、问答网站、数学社区、论文和代码中。论文首先要回答的问题是：Common Crawl 中是否存在足够多、足够高质量的数学网页，以及能否用可扩展的筛选流程把它们找出来。

这不是简单的关键词过滤。数学网页既可能包含自然语言讲解、公式和证明，也可能包含题目解析、程序化计算和不同语言的教育内容。筛选器如果只记住少数数学网站，召回率会很低；如果只追求召回率，又会引入大量低质量页面。

### 数学训练应如何与 code 训练组合

论文还比较了 general、code 和 math 数据在不同训练阶段的组合方式，考察三个问题：

- code training 是否能迁移到数学 reasoning；
- code training 对 program-aided reasoning 的帮助是否大于对纯文本解题的帮助；
- code 与 math 分阶段训练、混合训练时，能力提升与 forgetting 之间如何权衡。

### 如何在不训练大规模 critic 的情况下进行 reasoning RL

PPO 通常需要一个与 policy 规模相近的 value model。对语言模型而言，reward 往往只在 response 末端出现，value model 却要估计长 response 中各 token 的未来收益，训练成本和 credit assignment 难度都比较高。

论文提出 GRPO，使用同一个 question 下多条 response 的 reward 计算 group-relative advantage，从而去掉额外的 value model。它进一步比较 outcome supervision、process supervision、offline/online sampling 和 iterative RL，试图拆解 reasoning RL 真正有效的因素。

## 核心主张

1. **Common Crawl 可以提供规模化数学训练数据。** 通过 seed corpus、分类器、domain discovery 和人工 URL path 标注的迭代流程，论文构造了约 120B math tokens、3,550 万个数学网页的 DeepSeekMath Corpus。
2. **数据质量与跨语言覆盖比已有 corpus 的名义规模更关键。** 在相同 1.3B 模型和 150B 训练 token 的对照中，DeepSeekMath Corpus 在英文和中文数学 benchmark 上整体优于 MathPile、OpenWebMath 和 Proof-Pile-2。
3. **从 code model 初始化，并在数学训练前接触大量 code，有助于数学 reasoning。** 这种迁移在 program-aided mathematical reasoning 上尤其明显；code 与 math 混合训练则有助于减轻两阶段训练中的 coding forgetting。
4. **在论文当前的实验设置中，arXiv-only math training 没有带来稳定收益。** 这一结论只适用于论文测试的任务、数据处理方式、模型规模和训练配方，不能推广为“arXiv 数据普遍无用”。
5. **GRPO 用 group score 估计 baseline，保留 PPO-style clipped policy objective 和 reference KL regularization，同时省去 value model。**
6. **Process-supervised GRPO 优于 outcome-supervised GRPO。** 更细粒度的 process reward 能够缓解最终结果 reward 被广播到整条 response 所造成的 credit assignment 粗糙问题。
7. **Online RFT 优于 offline RFT，GRPO 优于 Online RFT，Iterative GRPO 还能继续提升。** 论文将差异归因于当前 policy 的数据分布，以及 GRPO 对不同 reward magnitude 的差异化强化和惩罚。
8. **RL 的主要收益表现为输出分布更稳定。** 在论文的 `Pass@K` / `Maj@K` 分析中，RL 明显提升 `Maj@K`，但没有明显提升 `Pass@K`，说明它更像是在提高已有正确解被稳定采样和聚合出来的概率，而不一定同等幅度地抬高能力上限。

## 方法总览

```text
Common Crawl
  -> seed corpus + fastText classifier
  -> domain discovery + URL path annotation
  -> DeepSeekMath Corpus
  -> code / math / natural language mixed pre-training
  -> DeepSeekMath-Base 7B
  -> CoT / PoT / tool-integrated SFT
  -> DeepSeekMath-Instruct 7B
  -> verifier / reward model + GRPO
  -> DeepSeekMath-RL 7B
```

这条链路中，每个阶段承担的职责不同：

- 数据筛选决定数学内容的覆盖、质量和语言分布；
- pre-training 让模型形成数学、代码和结构化推理先验；
- SFT 将这些先验组织成可模仿的解题行为；
- RL 使用 correctness 或 process reward 调整 response distribution；
- evaluation 需要分别观察能力上限、单次成功率和多次采样下的稳定性。

## 数学数据构造

### Seed-to-corpus 迭代流程

论文以 OpenWebMath 作为初始 positive seed，从 Common Crawl 中抽取网页作为 negative examples，训练 fastText 文本分类器。具体设置为：

- positive examples：从 seed corpus 随机抽取 500K 条；
- negative examples：从 Common Crawl 抽取 500K 条；
- word n-gram 最大长度：3；
- vector dimension：256；
- training epochs：3；
- learning rate：0.1；
- Common Crawl 预处理：URL deduplication 和 near-deduplication。

去重后，Common Crawl 约剩 40B HTML pages。分类器先对网页进行 recall 和排序，再保留排名靠前的数学页面。作者没有把第一轮分类结果视为最终语料，而是进一步统计每个 base URL domain 中被召回页面的比例：如果某个 domain 中超过 10% 的页面被召回，就把它视为潜在数学 domain，并对其中具有数学内容的 URL path 进行人工标注。新增的正样本随后用于更新 classifier。

整个流程迭代四轮，最终得到约 35.5M 数学网页、约 120B math tokens。第四轮中约 98% 的数据已经在第三轮收集到，因此停止继续扩展。

```text
高质量 seed
   |
   v
fastText classifier -> recall / rank Common Crawl pages
   |
   v
统计 domain 召回比例 -> 找到数学相关 domain
   |
   v
人工标注数学 URL path -> 扩充 positive seed
   |
   +--------------------> 更新 classifier，继续迭代
```

这个流程的关键不在 fastText 本身，而在于把**内容分类**和**来源结构发现**结合起来。分类器提供可扩展的页面级召回，domain 与 URL path 分析帮助发现初始 seed 没有覆盖的数学来源，人工标注只承担高价值的边界修正。

### Contamination filtering

为避免数学 benchmark 泄漏，论文对 GSM8K、MATH、CMATH 和 AGIEval 等数据进行 n-gram contamination filtering。对于 benchmark 中至少包含 10-gram 的文本，如果网页中存在完全匹配的 10-gram，就删除对应训练内容；对于长度不足 10-gram 但至少包含 3-gram 的 benchmark 文本，则使用 exact matching 过滤。

这一步说明，大规模网页数学语料不仅需要质量筛选，也需要单独的 contamination 审计。网页中可能存在题目、答案、解析或转载版本，不能因为它来自开放网页就默认与评测集独立。

### 与已有数学语料的比较

论文使用 1.3B 模型，对不同数学 corpus 各训练 150B tokens，比较 few-shot CoT 表现：

| Corpus | Corpus size | GSM8K | MATH | OCW | SAT | MMLU-STEM | CMATH | Gaokao MathCloze | Gaokao MathQA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| No Math Training | - | 2.9 | 3.0 | 2.9 | 15.6 | 19.5 | 12.3 | 0.8 | 17.9 |
| MathPile | 8.9B | 2.7 | 3.3 | 2.2 | 12.5 | 15.7 | 1.2 | 0.0 | 2.8 |
| OpenWebMath | 13.6B | 11.5 | 8.9 | 3.7 | 31.3 | 29.6 | 16.8 | 0.0 | 14.2 |
| Proof-Pile-2 | 51.9B | 14.3 | 11.2 | 3.7 | 43.8 | 29.2 | 19.9 | 5.1 | 11.7 |
| DeepSeekMath Corpus | 120.2B | **23.8** | **13.6** | **4.8** | **56.3** | **33.1** | **41.5** | **5.9** | **23.6** |

表中结果支持两个判断：第一，训练 token 数和 corpus 名义规模并不能单独解释效果；第二，DeepSeekMath Corpus 的 multilingual coverage 对中文数学 benchmark 尤其重要。论文还观察到，较小 corpus 在重复训练后更快达到 plateau，而更大的 DeepSeekMath Corpus 具有更持久的性能提升曲线。

## DeepSeekMath-Base 7B

### 预训练配方

DeepSeekMath-Base 7B 从 DeepSeek-Coder-Base-v1.5 7B 初始化，继续训练 500B tokens。论文给出的数据分布为：

- 56% DeepSeekMath Corpus；
- 4% AlgebraicStack；
- 10% arXiv；
- 20% GitHub code；
- 10% 英文和中文 Common Crawl natural language。

训练设置沿用小模型对照实验的整体 recipe，但把最大 learning rate 设为 `4.2e-4`，batch size 设为 `10M tokens`。这里的结果不能简单归因于“数学数据越多越好”：模型初始化、code 数据、数学数据质量、混合比例和训练预算共同决定了最终状态。

### 数学能力

DeepSeekMath-Base 7B 使用 few-shot CoT，在不调用外部工具的设置下得到：

| Benchmark | Score |
|---|---:|
| GSM8K | 64.2% |
| MATH | 36.2% |
| OCW | 15.4% |
| SAT | 84.4% |
| MMLU-STEM | 56.5% |
| CMATH | 71.7% |
| Gaokao-MathCloze | 20.3% |
| Gaokao-MathQA | 35.3% |

在工具增强设置下，模型通过 program-of-thought 生成 Python 程序，并以程序执行结果作为答案：

| Benchmark | DeepSeekMath-Base 7B |
|---|---:|
| GSM8K + Python | 66.9% |
| MATH + Python | 31.4% |
| miniF2F-valid | 25.8% |
| miniF2F-test | 24.6% |

miniF2F 的 informal-to-formal proving 使用 Isabelle，并结合 proof sketch 与 Sledgehammer 补齐证明细节。它表明该模型不仅能生成自然语言解题过程，也能在形式化数学任务中提供有用的中间结构，但这不等于模型已经具备稳定的自动定理证明能力。

### 通用能力与 code 保持

论文还评估了 MMLU、BBH、HumanEval 和 MBPP：

| Benchmark | Score |
|---|---:|
| MMLU | 54.9% |
| BBH | 59.5% |
| HumanEval Pass@1 | 40.9% |
| MBPP Pass@1 | 52.6% |

数学训练后 MMLU 和 BBH 相对前置的 DeepSeek-Coder-Base-v1.5 有提升，同时通过加入 code tokens 保持了较强的 coding performance。这为后续 code-to-math transfer 实验提供了背景：数学训练不是一个只改善 math benchmark、必然损害其他能力的过程，但数据 mix 仍需要显式控制。

## Code 与 Math 的训练关系

论文使用 1.3B 模型比较 general、code、math 的两阶段和单阶段训练。核心结果如下：

| Training setting | GSM8K | MATH | CMATH | GSM8K + Python | MATH + Python |
|---|---:|---:|---:|---:|---:|
| General 400B -> Math 150B | 19.1 | 14.4 | 37.2 | 14.3 | 6.7 |
| Code 400B -> Math 150B | **21.9** | **15.3** | **39.7** | **17.4** | **9.4** |
| Math 150B | 20.5 | 13.1 | 37.6 | 11.4 | 6.5 |
| Code 400B + Math 150B mixed | 17.6 | 12.1 | 36.3 | **19.7** | **13.5** |

对照结果显示：

- code training 在 math training 之前能够提高不使用工具和使用工具的数学 reasoning；
- 对 program-aided reasoning 的提升更明显，说明代码训练可能为程序化计算和结构化操作提供了迁移先验；
- 先 code 后 math 的两阶段训练在纯数学 reasoning 上更强，但后续 math training 会带来 coding forgetting；
- code 与 math 混合训练能较好地保持 coding，并在工具增强数学任务上取得更高结果；
- 在 1.3B 小模型上，code 与 math 混合会牺牲部分不使用工具的数学 reasoning，说明有限模型容量下存在数据共存 trade-off。

通用和 coding benchmark 的单阶段对照也体现了这个 trade-off：

| Training setting | MMLU | BBH | HumanEval Pass@1 | MBPP Pass@1 |
|---|---:|---:|---:|---:|
| General 400B -> Math 150B | 33.1 | 32.7 | 12.8 | 13.2 |
| Code 400B -> Math 150B | **36.2** | **35.3** | 12.2 | 17.0 |
| Math 150B | 32.3 | 32.5 | 11.6 | 13.2 |
| Code 400B + Math 150B mixed | 33.5 | **35.6** | **29.3** | **39.4** |

这里不应把“先 code 后 math”或“code/math mixed”抽象成对所有模型都成立的固定配方。论文只在特定模型规模、token budget 和数据设置下证明了这些趋势，实际 mixture 仍需要通过目标能力、通用能力和 forgetting 的联合评测确定。

### ArXiv-only 实验

论文分别测试了 MathPile 和 ArXiv-RedPajama。对 1.3B 模型训练 150B tokens、对 7B code model 训练 40B tokens 后，arXiv-only 训练在 GSM8K、MATH、MMLU-STEM、CMATH 和 miniF2F 等任务上没有显示稳定收益，部分设置出现退化。

作者同时明确了结论边界：尚未研究 theorem informalization、arXiv 与其他数据混合时的效果，也没有验证更大模型规模下潜在收益是否会出现。因此，更准确的表述是：**在论文当前的 arXiv-only 实验中，数学论文文本没有像经过筛选的网页数学语料一样带来明显的下游收益。** 这可能与任务分布、训练文本的结构、数据重复和模型对形式化内容的利用方式有关，不能简化为来源类型的绝对判断。

## Mathematical Instruction Tuning

### SFT 数据

DeepSeekMath-Instruct 7B 在 DeepSeekMath-Base 7B 上进行数学 instruction tuning，共使用 776K examples，覆盖英文和中文数学题，并混合三类解题形式：

- Chain-of-Thought：自然语言逐步推理；
- Program-of-Thought：通过程序完成计算；
- Tool-integrated reasoning：将自然语言 reasoning 与工具执行结合。

英文数据覆盖 algebra、probability、number theory、calculus、geometry 等领域。中文数据覆盖 K-12 数学的 76 个子主题，并同时标注 CoT 与 tool-integrated solution。

训练时将样本随机拼接到最大 4K context length，训练 500 steps，batch size 为 256，constant learning rate 为 `5e-5`。这一步的作用不是单纯增加数学知识，而是把 base model 的数学先验组织成可交互、可执行和可评估的输出形式。

### SFT 结果

在不使用工具的 CoT 评测中：

| Benchmark | DeepSeekMath-Instruct 7B |
|---|---:|
| GSM8K | 82.9% |
| MATH | 46.8% |
| MGSM-zh | 73.2% |
| CMATH | 84.6% |

在工具增强设置中：

| Benchmark | DeepSeekMath-Instruct 7B |
|---|---:|
| GSM8K + Python | 83.7% |
| MATH + Python | 57.4% |
| MGSM-zh | 72.0% |
| CMATH | 84.3% |

这组结果建立了后续 RL 的起点。特别是，RL 并不是从一个不会解题的 base model 直接发现数学能力，而是在已经经过数据塑形和 SFT 的 policy 上重新分配已有 response probability。

## GRPO

### 从 PPO 到 GRPO

PPO 的语言模型实现通常包含 policy model、reference model、reward model 和 value model。value model 用于估计 state value，并通过 GAE 生成 advantage；reference model 则通过 KL regularization 限制 policy 过快偏离初始 SFT 分布。

论文指出，在长数学 response 中，reward 经常只在最终答案处提供，而 value model 却需要为中间 token 估计未来回报。这会同时带来显存成本和训练难度。GRPO 的处理方式是：对同一个 question 采样一组 outputs，用 group reward 的统计量作为 baseline。

```text
question q
  -> old policy samples o_1, o_2, ..., o_G
  -> verifier / reward model scores r_1, r_2, ..., r_G
  -> group-relative advantage A_i
  -> PPO-style clipped policy update
  -> reference KL regularization
```

GRPO 仍然是 policy optimization，并没有取消 rollout、reward、old policy logprob 或 reference policy。它去掉的是与 policy 规模相近的 learned value model，而不是取消 RL 的核心数据和优化过程。

### Group-relative advantage

对同一个 question 的 $G$ 个 responses，最简单的 baseline 是组内平均 reward：

$$
A_i = r_i - \frac{1}{G}\sum_{j=1}^{G} r_j
$$

论文实验中常使用组内标准化：

$$
\hat{A}_i = \frac{r_i - \operatorname{mean}(r)}{\operatorname{std}(r)}
$$

同一组 responses 共享 question，因此它们面对的 prompt 难度基本相同。减去 group mean 的目的，是移除部分由 prompt 难度造成的 return 偏移，让 policy 更关注同一个 question 下不同 response 的相对质量。这解释了 GRPO 的 variance reduction 来源：它主要减少 prompt-level baseline error，并不能消除 reward noise、sampling noise 或长 response 的 credit assignment 问题。

这里要区分两个方差来源：

- 不同 prompt 难度不同，导致 batch 内 raw reward 的均值和尺度不同；
- 同一个 prompt 下不同 response 质量不同、采样不同，导致 group 内 reward 有波动。

GRPO 利用第二种波动来估计第一种因素的 baseline。若一个 group 全部答错、全部答对或 responses 过于相似，group variance 很小，relative signal 仍然会弱。

### Objective 与 KL

对第 $i$ 个 response 的第 $t$ 个 token，PPO-style ratio 为：

$$
\rho_{i,t}(\theta)
=
\frac{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}
{\pi_{\theta_{old}}(o_{i,t}\mid q,o_{i,<t})}
$$

GRPO 保留 clipped surrogate objective：

$$
J_{GRPO}(\theta)
=
\mathbb{E}\left[
\min\left(
\rho_{i,t}(\theta)\hat{A}_{i,t},
\operatorname{clip}(\rho_{i,t}(\theta),1-\epsilon,1+\epsilon)\hat{A}_{i,t}
\right)
-\beta D_{KL}(\pi_\theta\|\pi_{ref})
\right]
$$

论文将 KL 直接加入 objective，而不是把每个 token 的 KL penalty 混入 reward 再参与 advantage 计算。它使用如下无偏估计形式：

$$
D_{KL}(\pi_\theta\|\pi_{ref})
=
\frac{\pi_{ref}}{\pi_\theta}
-\log\frac{\pi_{ref}}{\pi_\theta}-1
$$

工程上，`clip range` 控制单批数据的 policy update，`KL coefficient` 控制 policy 相对 reference 的漂移，二者承担不同的稳定化职责。

### Outcome Supervision

Outcome Supervision 只在 response 结束位置提供 reward。对每条 response，先在 group 内将最终 reward 标准化，再将同一个 normalized reward 广播到该 response 的所有有效 tokens：

$$
\hat{A}_{i,t}=\hat{r}_i
$$

这种方式实现简单，适合最终答案可可靠验证的任务，例如数学 exact answer、代码 unit test 或编译结果。但它无法区分一条长 reasoning 中哪些步骤贡献了正确结果，哪些步骤只是无效展开。

### Process Supervision

Process Supervision 在每个 reasoning step 结束处提供 process reward。论文设第 $j$ 个 reasoning step 的结束 token index 为 $index(j)$，先对 process rewards 做标准化，再计算从当前位置向后的 process reward 累积：

$$
\hat{A}_{i,t}
=
\sum_{index(j)\geq t}\hat{r}_{i,index(j)}
$$

因此，越早的 token 能看到更多后续 process rewards，越接近末尾的 token 只受到剩余步骤的影响。相比 outcome supervision，这种方式提供了更细粒度的 step-aware credit assignment。论文实验显示，`GRPO + Process Supervision` 优于 `GRPO + Outcome Supervision`，但 process reward model 的标注与泛化本身也会引入噪声，不能把更密的 reward 自动等同于更可靠的 reward。

### Iterative GRPO

论文观察到，policy 不断变化后，旧 reward model 可能无法充分区分新 policy 产生的 outputs。因此提出 iterative GRPO：

1. 当前 policy 生成新一轮 responses；
2. 用 verifier 或规则产生新 reward data，持续训练 reward model；
3. replay buffer 保留约 10% historical data，避免 reward model 只适应最新分布；
4. 将当前 policy 设为下一轮的 reference model；
5. 使用更新后的 reward model 继续 GRPO。

论文实验进行两轮迭代，第一轮带来的收益最明显。这个结果支持一个更一般的训练判断：当 policy 会显著改变 rollout 分布时，reward model、reference model 和训练数据都需要有明确的更新策略，不能长期固定在初始 SFT 分布上。

## RFT、Online RFT、DPO、PPO 与 GRPO

论文用一个统一视角分析这些方法。对 policy 参数的梯度，可以抽象为：

$$
\nabla_\theta J_A(\theta)
=
\mathbb{E}\left[
\frac{1}{|o|}\sum_t
GC_A(q,o,t,\pi_{rf})
\nabla_\theta\log\pi_\theta(o_t\mid q,o_{<t})
\right]
$$

每种方法都由三个部分决定：

- **Data source**：question 和 response 从哪里来；
- **Reward function**：如何判断 response 的质量；
- **Gradient coefficient**：reward 如何转成每个 token 的强化、惩罚或忽略系数。

| 方法 | Data source | Reward / selection | 更新特征 |
|---|---|---|---|
| SFT | 人工或 teacher 选择的 $(q,o)$ | 选择本身 | 所有 target tokens 通常同向学习 |
| RFT | 初始 SFT policy 采样 | rule 过滤正确 responses | 只强化正确 response，不更新错误 response |
| Online RFT | 实时 policy 采样 | rule 过滤正确 responses | 更贴近当前 policy，但仍主要统一强化正确样本 |
| DPO | 初始 SFT policy 采样的 chosen/rejected pair | pairwise preference | 通过偏好差异直接更新 |
| PPO | 实时 policy 采样 | reward model + value model | GAE 与 critic 估计 advantage |
| GRPO | 实时 policy 对同一 question 采样 group | reward model + group-relative advantage | 去掉 value model，按相对 reward 强化和惩罚 |

### Offline 与 Online

RFT 和 DPO 使用初始 SFT model 采样的数据，属于 offline 风格；Online RFT 和 GRPO 使用实时 policy 采样的数据，属于 online 风格。训练早期，初始 SFT policy 与实时 policy 接近，两者差异有限；随着 policy 改变，实时采样能够更准确地反映当前模型的错误模式和可探索区域。

论文在 1.3B 模型上观察到 Online RFT 在后期明显优于 RFT。这不是因为 online 这个标签本身有特殊魔力，而是因为训练数据与当前 policy 的分布保持同步，避免模型一直在已经过时的 samples 上训练。

### GRPO 与 Online RFT 的关键差异

如果只用规则判断正确/错误，Online RFT 主要做的是“保留正确 samples 并统一强化”。GRPO 则根据 reward magnitude 构造正负 advantage：高于组均值的 response 被强化，低于组均值的 response 被抑制。

因此，GRPO 的额外收益来自 gradient coefficient 的差异化，而不仅是换了一个采样策略。论文实验中 GRPO 优于 Online RFT，`GRPO + Process Supervision` 又优于 `GRPO + Outcome Supervision`。

## GRPO 训练设置与结果

DeepSeekMath-RL 7B 从 DeepSeekMath-Instruct 7B 初始化。RL questions 来自 SFT 数据中的 GSM8K 与 MATH CoT-format questions，约 144K 条；作者排除其他 SFT questions，用来观察只在这部分问题上做 RL 是否仍能改善其他 benchmark。

关键训练设置：

- policy learning rate：`1e-6`；
- reward model learning rate：`2e-5`；
- KL coefficient：`0.04`；
- group size：每个 question 采样 64 outputs；
- max response length：1024；
- training batch size：1024；
- 每次 exploration stage 后只做一次 policy update。

结果如下：

| Benchmark | Instruct 7B, CoT | RL 7B, CoT | Instruct 7B, tool | RL 7B, tool |
|---|---:|---:|---:|---:|
| GSM8K | 82.9% | **88.2%** | 83.7% | **86.7%** |
| MATH | 46.8% | **51.7%** | 57.4% | **58.8%** |
| MGSM-zh | 73.2% | **79.6%** | 72.0% | **78.4%** |
| CMATH | 84.6% | **88.8%** | 84.3% | **87.6%** |

论文强调，DeepSeekMath-RL 7B 的 RL 训练只使用 GSM8K 和 MATH 的 CoT-format questions，但在其他中文和工具增强 benchmark 上也有提升。这说明在数学任务内，RL 可能产生一定的迁移；但训练问题仍来自相对窄的分布，不能据此推出对通用 agent 或 long-horizon environment 的泛化结论。

## 为什么 RL 有效：Pass@K 与 Maj@K

论文使用两类指标分析 SFT 与 RL 模型在 GSM8K、MATH 上的多次采样结果：

- `Pass@K`：$K$ 个 candidates 中只要有一个正确，就算成功；
- `Maj@K`：$K$ 个 candidates 通过 majority voting 选择结果，再判断是否正确。

实验发现 RL 显著提升 `Maj@K`，但没有明显提升 `Pass@K`。这意味着 RL 主要改变了 response distribution：原来已经存在于 sampling distribution 中的正确路径变得更容易被采样，并且同一问题的 outputs 更一致。

这个结果需要谨慎解释：

- 如果只看 top-1 或 majority accuracy，RL 的收益很明显；
- 如果看 `Pass@K`，模型的 capability ceiling 没有同等幅度地提升；
- 因而“RL 学会了新的解题能力”和“RL 让已有能力更稳定地出现”是两个不同命题。

这也是论文对 reasoning RL 的重要保守结论。后续工作若要证明 fundamental capability 真正提升，需要同时报告 single-sample accuracy、`Pass@K` 曲线、`Maj@K` 曲线、response diversity 和 OOD task performance，而不能只报告某一个聚合指标。

## 局限与疑问

### 能力覆盖仍然偏向竞赛数学

论文在 geometry 和 theorem proving 等方向上的能力相对弱，且数学数据选择可能对某些主题更有利，例如 triangle、ellipse 等内容的覆盖与处理质量仍可能不足。DeepSeekMath 的结果不能代表所有数学任务。

### RL prompt 分布有限

RL 主要使用 GSM8K 和 MATH 的 SFT questions。虽然其他 benchmark 有提升，但训练阶段没有充分覆盖更复杂、更开放或真正 OOD 的问题分布。论文也提出，后续应探索 OOD prompts 和 tree-search 等更强的 sampling strategy。

### Reward model 可能成为瓶颈

GRPO 不再需要 value model，但仍依赖 reward model 或 verifier。reward model 对 OOD questions、长 reasoning 和 advanced decoding 的泛化不可靠时，RL 可能只是在稳定模型输出分布，而不是提升底层能力。process reward 更细，但 process annotation 也会带来误标和偏置；论文引用 PRM800K 约 20% incorrect annotations 的例子说明了这一风险。

### 结果与规模化 agent 的距离

论文验证的是数学问题的单轮 reasoning 和 program-aided solving，主要 reward 也容易通过 final answer 或 process correctness 判断。它没有直接验证多轮工具调用、动态环境反馈、长程任务规划、错误恢复和跨工具协作。因此，GRPO 的 group baseline 和 process supervision 能否直接迁移到通用 agent，需要另行检查 group sampling、trajectory reward 和 credit assignment 的假设。

### 训练链路的归因并不完全独立

最终 DeepSeekMath-Base、Instruct 和 RL 的结果同时受数据构造、初始化 checkpoint、code/math mix、SFT 数据和 RL recipe 影响。小模型数据对照能够支持局部结论，但不能把最终 7B 模型的全部提升归因于某一个模块。

## 对训练系统的可迁移结论

### 对数据管线

DeepSeekMath 的数据构造流程提供了一套可迁移的数据工程模式：高质量 seed 用于初始化分类器，模型召回扩大覆盖，来源级结构分析发现盲区，人工标注修正边界，再迭代更新分类器。对于 agent trajectory 数据，也可以把“高质量完整轨迹”作为 seed，按任务环境、工作流和反馈模式扩充更大规模样本，并持续做 contamination、重复和质量审计。

### 对 agentic mid-training

论文直接证明的是数学能力注入，不是通用 agent mid-training。但它提供了一个有价值的训练逻辑：大量 NTP 数据不仅能注入知识，也能注入可迁移的行为结构；数据配方应同时记录任务类型、工具使用、reasoning 形态、反馈结构和质量等级，而不是只记录 token 数。

对于长 agent trajectory，论文的结论更适合作为数据混合和能力迁移的参考，而不能直接照搬 GRPO 的数学 recipe。具体的 agent 训练还需要单独验证 action validity、tool selection、argument grounding、state tracking、recovery 和 termination 等能力。

### 对 RL 评测

论文对 `Pass@K` 和 `Maj@K` 的拆分非常适合作为 RL 评测的最低要求。对于 agent 任务，也应区分：

- 单次 rollout 的任务成功率；
- 多次 rollout 中至少一次成功的 `Pass@K`；
- majority 或 verifier 选择后的 `Maj@K` / best-of-N 成功率；
- 轨迹长度、工具调用次数、失败恢复和成本。

如果训练只提高聚合成功率，却没有提高 `Pass@K` 或 OOD single-run success，就不能简单地说模型获得了更强的基础 agent capability。

## 关键结论

1. 数学 reasoning 的提升首先依赖可规模化、质量可控、语言覆盖充分的数据，而不是单纯增加模型参数。
2. DeepSeekMath Corpus 的构造重点是迭代式 seed expansion：classifier 负责规模化召回，domain/path 分析负责发现遗漏，人工标注负责修正高价值边界。
3. Code training 对数学 reasoning 存在迁移，尤其体现在 program-aided reasoning；但 code/math 的两阶段与混合训练具有明显 trade-off。
4. arXiv-only training 在论文当前设置下未显示稳定数学收益，结论应限定在任务、模型和数据配方范围内。
5. GRPO 用同一 question 的多条 response 估计 group-relative baseline，省去 value model，但并未消除 reward noise、group collapse 和 token-level credit assignment 问题。
6. Process supervision 为 reasoning RL 提供更细粒度的 advantage，论文中优于 outcome supervision；其可靠性仍取决于 process reward model。
7. Online data、差异化 gradient coefficient 和 iterative reward model 更新，是论文中比“算法命名”更值得复用的训练设计因素。
8. RL 的收益需要同时用 `Pass@K` 和 `Maj@K` 解读：它可以先改善正确 response 的可采样性和分布稳定性，而不必然提高能力上限。

## 相关知识链接

- [[training/pretraining/data-mix|Data Mix]]
- [[training/mid-training/continued-pretraining|Continued Pretraining]]
- [[training/post-training/grpo|GRPO]]
- [[training/post-training/rejection-sampling|Rejection Sampling]]
- [[training/post-training/knowledge-distillation|Knowledge Distillation]]
- [[training/post-training/rlhf|RLHF]]
