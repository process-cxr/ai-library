---
title: "Training language models to follow instructions with human feedback"
created: 2026-05-31
published: 2026-08-27
modified: 2026-08-27
type: source
status: processed
source_type: paper
area: sources
tags:
  - source
  - paper
  - rlhf
  - instruction-tuning
  - reward-model
  - ppo
  - alignment
source_url: https://arxiv.org/abs/2203.02155
paper_date: "2022-03"
paper_order: "02155"
---

# Training language models to follow instructions with human feedback

## 基本信息

| 字段 | 内容 |
|---|---|
| 论文 | [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) |
| 版本 | arXiv:2203.02155v1，2022-03-04 |
| 作者 | Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe |
| 机构 | OpenAI |
| 首次公开 | 2022-03，arXiv v1 |
| 模型 | GPT-3 架构，1.3B / 6B / 175B |
| 相关 topic | [[training/post-training/rlhf|RLHF]]，[[training/post-training/sft|SFT]]，[[training/post-training/reward-model|Reward Model]]，[[training/post-training/ppo|PPO]]，[[training/post-training/instruction-tuning|Instruction Tuning]] |

这篇论文系统展示了如何把一个只接受 next-token pretraining 的 GPT-3 语言模型，训练成更符合用户意图的 instruction-following model。论文将目标拆成三个层次：模型需要更好地完成用户指令，也要减少 hallucination 和 toxic output，同时尽量不要损害原有语言建模和公共 NLP 能力。

论文提出的 InstructGPT pipeline 由三步组成：先收集 labeler demonstrations 训练 SFT policy，再收集多个模型回答的排序训练 reward model，最后用 PPO 在 reward model 上优化 policy。论文还提出 `PPO-ptx`，在 PPO 中混入 pretraining distribution 的 language-modeling gradient，以缓解 RLHF 带来的 public NLP regression。

如果只保留一个核心认识：**InstructGPT 的贡献不是单独使用 PPO，而是将 instruction 数据、preference data、reward model、online policy optimization 和能力保持机制组合成了一条可运行的 post-training pipeline。** 论文同时用 human preference、truthfulness、toxicity、bias、公共任务和定性案例说明：alignment 结果不能只用一个 reward 或一个 benchmark 衡量。

## 研究问题

### Next-token Objective 与 User Intent 不完全一致

大型语言模型通过预测互联网文本中的下一个 token 学习语言、知识和任务模式，但这个训练目标不直接等价于：

- 按照用户真正的 instruction 完成任务；
- 对不确定的事实保持 truthful；
- 不生成 toxic、biased 或 harmful 内容；
- 遵守用户给出的格式、长度和内容约束；
- 在多个可接受回答中选择更 helpful 的一个。

一个 base language model 可能拥有完成任务所需的知识，却在 instruction-following、回答格式、事实性或安全性上表现不稳定。单纯增加模型规模也不会自动解决这些行为问题。

论文将目标概括为让模型更 helpful、honest、harmless。但这三个词不是可以直接计算的单一标签，必须通过示范、偏好比较、自动评测和人类评估共同近似。

### Public NLP 数据与真实使用场景存在差异

传统 public NLP benchmark 主要覆盖分类、问答、阅读理解、翻译和摘要等容易用自动指标评测的任务，而真实 API 用户的 prompt 还包括开放式 generation、brainstorming、rewrite、chat 和多种混合约束。

论文的 API prompt 分布中，RM 数据的 use-case 大致为：

| Use case | 占比 |
|---|---:|
| Generation | 45.6% |
| Open QA | 12.4% |
| Brainstorming | 11.2% |
| Chat | 8.4% |
| Rewrite | 6.6% |
| Summarization | 4.2% |
| Classification | 3.5% |
| Other | 3.5% |
| Closed QA | 2.6% |
| Extract | 1.9% |

Generation 和 brainstorming 合计超过一半，而 classification 与 QA 只占相对较小部分。因此，单纯在大量 public NLP task 上 instruction tuning，并不一定能得到更适合真实用户的 assistant。

### Alignment 可能产生能力代价

PPO 会推动模型提高 reward model 分数，但如果 policy 过度偏离原始语言模型，可能损害语言建模、知识和传统 NLP 任务能力。论文将这种现象称为 alignment tax：为了让模型更符合偏好，牺牲了其他仍然有价值的能力。

论文希望回答：

1. 人类反馈能否让不同规模的 GPT-3 更好地遵循广泛的自然语言指令？
2. 1.3B 的 InstructGPT 是否可能优于 175B 的原始 GPT-3 在真实 prompt 上的回答？
3. 训练后的 model 是否更 truthful、less toxic，并保持公共 NLP 能力？
4. pretraining distribution 的混合更新能否减小 PPO 的 alignment tax？
5. 模型究竟对谁的偏好进行了 alignment？training labelers、researchers、API customers 和 broader users 并不等价。

## 核心主张

论文的主张可以分成能力、训练和评估三部分。

### 能力结果

1. labelers 在 API prompt 分布上显著偏好 InstructGPT，而不是 GPT-3；
2. 1.3B InstructGPT 的回答可以胜过 175B GPT-3，尽管参数量少超过 100 倍；
3. 175B InstructGPT 相对 175B GPT-3 的直接偏好胜率为 $85\pm3\%$，相对 few-shot prompted GPT-3 为 $71\pm4\%$；
4. InstructGPT 更常遵守 explicit constraints、更少完全误解 instruction，也更少在 closed-domain 任务中 hallucinate；
5. 在 TruthfulQA 和 closed-domain hallucination 评测上，InstructGPT 比 GPT-3 更 truthful；
6. 在明确要求 respectful 时，InstructGPT 的 toxic output 更少，但在 bias 数据集上没有稳定改善；
7. PPO 会造成部分 public NLP regression，PPO-ptx 可以显著缓解，但不能完全消除。

### 方法结果

1. 三阶段 pipeline 可以从 GPT-3 base model 构造可用 assistant policy；
2. RM 不只使用二元 pair，而是让 labeler 对同一 prompt 的 $K=4$ 到 $9$ 个回答排序，再将组合 pair 一起训练；
3. PPO 环境被建模为 prompt-response bandit，RM 对完整 response 给 scalar reward，同时加入逐 token 的 SFT-reference KL penalty；
4. 在 PPO 中混入 pretraining loss，可以在不显著牺牲 labeler preference 的情况下恢复部分原始能力。

### 评估结果的边界

论文并没有证明 InstructGPT 已经“解决 alignment”。作者明确指出模型仍会 hallucinate、产生 toxic 或 biased output、接受 harmful instruction、误解 false premise，并且在复杂多约束任务上不稳定。论文更准确的贡献是：展示了一条可扩展的、面向真实用户 prompt 的 human-feedback post-training 路线，并把其收益与代价都测量出来。

## 三阶段训练方法

### 总体流程

给定 pretrained GPT-3、目标 prompt distribution 和 trained human labelers，论文执行：

```text
Step 1: demonstrations
  labeler writes desired responses
  -> supervised fine-tuning
  -> SFT policy

Step 2: comparisons
  sample multiple outputs from models
  -> labeler ranks outputs
  -> reward model

Step 3: policy optimization
  SFT policy generates responses
  -> RM scores responses
  -> PPO optimizes policy
  -> PPO-ptx optionally mixes pretraining updates
```

Step 2 和 Step 3 可以循环：当前较好的 policy 产生新的比较候选，新的 comparisons 用来训练新 RM，再继续更新 policy。论文的 comparison data 主要来自 SFT policy，也有一部分来自 PPO policy。

### Step 1：Demonstration 与 SFT

labelers 根据 prompt 写出他们认为符合用户意图的 desired response。GPT-3 以 supervised learning 目标拟合这些示范，得到 SFT model。

论文使用 1.3B、6B 和 175B 三种 GPT-3 模型。SFT 训练 16 epochs，使用 cosine learning-rate decay 和 residual dropout 0.2。一个容易忽略的实验观察是：SFT model 在 validation loss 上大约 1 epoch 后开始 overfit，但继续训练更多 epochs 仍然能提高 RM score 和 human preference。因此，validation language-modeling loss 并不能单独决定 SFT checkpoint。

SFT 的作用是建立一个可用的行为起点：

- 能识别 prompt 的 instruction 结构；
- 能以 assistant 风格生成响应；
- 为后续 response comparison 提供质量较好的候选；
- 让 PPO 不必从原始 GPT-3 的任意 continuation 分布开始探索。

SFT 仍然是 imitation learning。它主要学习 labeler 给出的一个 response，并不直接表达多个合理回答之间的相对偏好。

### Step 2：Comparison 与 Reward Model

对同一个 prompt，系统从模型采样多个 responses，labeler 对它们排序。论文用 $K=4$ 到 $9$ 个 responses 加速 comparison collection，这可以产生：

$$
\binom{K}{2}
$$

个 pairwise comparisons。

如果简单把这些 pair 打散成独立样本，一个 completion 会在同一个 labeling task 中参与 $K-1$ 次 gradient update，容易导致 RM 过拟合。论文把同一个 prompt 的全部组合 pair 作为一个 batch element，一次 forward 计算这 $K$ 个 completion 的 reward，再构造所有 pair 的 loss。这样既减少重复 forward，也保留了同一轮 ranking 的相关性。

RM 从去掉 final unembedding layer 的 SFT model 初始化，输入 prompt 和 completion，输出 scalar reward $r_\theta(x,y)$。对 chosen completion $y_w$ 和 rejected completion $y_l$，使用 Bradley-Terry 风格的 loss：

$$
\mathcal{L}_{RM}(\theta)
=-\frac{1}{\binom{K}{2}}
\mathbb{E}_{(x,y_w,y_l)\sim D}
\left[
\log\sigma(r_\theta(x,y_w)-r_\theta(x,y_l))
\right].
$$

reward difference 表示 chosen 相对 rejected 的 log odds。RM 只需要学习排序，不需要让 reward 的绝对尺度具有自然含义。

由于 RM loss 对 reward 的整体平移不敏感，论文在 RL 前加 bias，使 labeler demonstrations 的平均 reward 为 0。这是 reward calibration，而不是改变 preference ranking。

论文只使用 6B RM，因为 175B RM 训练不稳定，也不适合直接充当 RL 中的 value function。这里体现出一个工程折中：reward model 不必与 policy 等大，关键是排序质量、训练稳定性和推理成本。

### Step 3：PPO Policy Optimization

论文把每个 prompt-response 过程建模为一个 bandit episode：环境随机提供 customer prompt，policy 生成 response，RM 对 response 给 scalar reward，然后 episode 结束。

policy 初始为 SFT model。PPO 更新时：

1. 用当前 policy 生成 response；
2. RM 对 response 打分；
3. 对每个 generated token 加入相对于 SFT model 的 KL penalty；
4. 用 PPO 更新 policy；
5. value function 从 RM 初始化。

这里有两个不同的 policy reference：

- PPO ratio 使用 rollout policy，控制这批样本上的单次 update；
- KL penalty 使用固定的 SFT policy，控制 PPO policy 相对 SFT 起点的总体漂移。

可以把 sequence reward 写成一个简化的形式：

$$
R(x,y)=r_\theta(x,y)
-\beta\log\frac{\pi_\phi^{RL}(y\mid x)}
{\pi^{SFT}(y\mid x)}.
$$

实际 token-level implementation 会把 log-prob difference 分配到生成 token，再结合 PPO 的 advantage 与 clipped ratio 优化。KL penalty 防止 policy 通过利用 RM 漏洞获得高分，例如不自然地变长、套用特定模板或产生 reward model 偏好的表面风格。

### PPO-ptx：混入 Pretraining Distribution

论文观察到，单纯 PPO 在若干 public NLP datasets 上出现 regression。为恢复原始语言模型能力，论文在 PPO objective 中加入 pretraining distribution 的 language-modeling term：

$$
\begin{aligned}
\mathrm{objective}(\phi)=
&\mathbb{E}_{(x,y)\sim D_\pi^{RL}}
\left[
r_\theta(x,y)
-\beta\log\frac{\pi_\phi^{RL}(y\mid x)}{\pi^{SFT}(y\mid x)}
\right]\\
&+\gamma\mathbb{E}_{x\sim D_{pretrain}}
\left[\log\pi_\phi^{RL}(x)\right].
\end{aligned}
$$

其中 $\beta$ 控制 KL reward coefficient，$\gamma$ 控制 pretraining gradient coefficient。普通 PPO 设置 $\gamma=0$；论文默认将带有 pretraining mix 的 PPO 模型称为 `PPO-ptx`，也是 InstructGPT 的主要模型。

PPO-ptx 的逻辑不是简单把 KL coefficient 调大：

```text
KL penalty:
  约束当前 policy 不要偏离 SFT policy

pretraining mix:
  直接恢复 pretraining distribution 上的 language-modeling ability
```

论文实验显示，适当的 pretraining mix 比单纯增加 KL coefficient 更能缓解 DROP、SQuAD 等数据集上的 regression，同时保留较高的 validation reward。它也有成本：需要额外 pretraining data 和训练计算，且如果 pretraining data 含有不良行为，混入这些数据也可能重新引入问题。

## 数据构造与标注

### Prompt 来源

训练 prompt 主要来自两类：

1. labeler-written prompts，用于项目启动和补足 API 中不常见的 instruction 类型；
2. 早期 InstructGPT model 在 OpenAI API Playground 上接收的 prompts。

API prompt 不是 production API 数据；用户被告知其数据可能用于训练后续模型。训练集 prompt 会做启发式去重、每个 user ID 最多保留 200 个，并按 user ID 划分 train/validation/test，避免同一用户的相似 prompt 同时出现在不同 split。训练 split 还过滤 personally identifiable information（PII）。

labeler-written prompts 分为三类：

- `Plain`：任意任务，但要求足够多样；
- `Few-shot`：instruction 加多个 query/response examples；
- `User-based`：根据 API waitlist application 中的 use case 编写 prompt。

### 三种数据集

论文将数据按用途拆成 SFT、RM 和 PPO 三个集合：

| Dataset | Prompt 来源 | Train prompts | Valid prompts |
|---|---|---:|---:|
| SFT | labeler | 11,295 | 1,550 |
| SFT | customer/API | 1,430 | 103 |
| RM | labeler | 6,623 | 3,488 |
| RM | customer/API | 26,584 | 14,399 |
| PPO | customer/API | 31,144 | 16,185 |

PPO dataset 没有 human labels，而是作为 RL 环境的 prompt source。RM dataset 的 prompt 数量不等于 pair 数量，因为每个 prompt 有 $K=4$ 到 $9$ 个 output ranking，组合 pair 的训练样本数显著更多。

### Labeler 与偏好定义

论文雇佣约 40 名 contractors，通过 Upwork 和 ScaleAI 招募，并通过 screening test 筛选其识别敏感内容、判断 harmful output 和处理不同 demographic context 的能力。项目期间通过 onboarding、详细 instructions 和共享沟通渠道与 labelers 协作。

训练阶段，labelers 主要根据 inferred user intent 进行判断，并综合考虑 helpfulness。最终评估阶段则更强调 truthfulness 和 harmlessness。这种训练与评估优先级差异本身就是一个重要实验设计事实：模型最终测量的目标不一定与每个训练样本使用的权重完全相同。

训练 labelers 的 inter-annotator agreement 约为 $72.6\pm1.5\%$，held-out labelers 约为 $77.3\pm1.3\%$。这表明标签并非完全一致，但也不是随机噪声。

## 评估设计

### 为什么不能只看 Reward Model Score

论文将“alignment”拆成多个可观察代理：

- helpfulness：是否完成用户意图、遵守 instruction 和 constraints；
- truthfulness：是否少编造事实，是否在 TruthfulQA 上更真实；
- harmlessness：是否降低 inappropriate、sexual、violent、toxic 等输出；
- capability retention：是否保留传统 public NLP task 能力；
- generalization：是否对 held-out labelers、非英语和 code instruction 泛化。

没有一个单独的自动指标能覆盖这全部目标。因此论文同时进行：

1. API prompt distribution 上的 human preference 与 Likert rating；
2. closed-domain hallucination 与 TruthfulQA；
3. RealToxicityPrompts、Winogender、CrowS-Pairs；
4. SQuAD、DROP、HellaSwag、WMT、CNN/DM、TLDR、QuAC 等 public NLP datasets；
5. held-out labelers 的 preference；
6. code question、非英语 instruction 和复杂约束的 qualitative evaluation。

### API Prompt Human Evaluation

主指标是相对于 175B SFT baseline 的 win rate。评测 prompt 来自 held-out customer，并按 user ID 划分，避免训练用户泄漏。论文也在 GPT-3 prompt distribution 上评测，因为只在 InstructGPT 风格 prompt 上比较会天然不利于 GPT-3 baseline。

除了 pairwise preference，labelers 还给 response 1 到 7 的 Likert overall quality 分数，并记录：

- 是否尝试完成正确 instruction；
- 是否遵守 explicit constraint；
- 是否适合 customer assistant；
- 是否 hallucinate；
- 是否包含 sexual/violent content；
- 是否 denigrate protected class；
- 是否提供 harmful advice；
- 是否表达 moral judgment 或 opinion。

这种 metadata evaluation 使“偏好更高”可以进一步拆成具体行为变化，而不是停留在一个 aggregate win rate。

## 实验结果

### InstructGPT 相对 GPT-3 的偏好

在 API prompt 分布上，模型表现呈现出清晰的梯度：

```text
GPT-3
  -> few-shot prompted GPT-3
  -> SFT
  -> PPO
  -> PPO-ptx / InstructGPT
```

175B InstructGPT 相对 175B GPT-3 的 labeler preference win rate 为 $85\pm3\%$，相对 few-shot prompted 175B GPT-3 为 $71\pm4\%$。更值得注意的是，1.3B InstructGPT 的输出也可以优于 175B GPT-3。

这个结果不应被解读为“post-training 让小模型拥有了大模型全部能力”。比较的是特定 API prompt 分布上的行为质量，且 InstructGPT 和 GPT-3 的 architecture 相同，主要差异是 human-feedback fine-tuning。模型的基础知识、推理能力和长尾能力并不会因为 SFT/PPO 自动按比例提升。

与 FLAN/T0 的比较也很有启发：论文将 175B GPT-3 在约 1M public task examples 上 instruction tuning，并发现其在 API prompt 分布上只略好于 GPT-3，仍差于 SFT 和 InstructGPT。一个解释是 public task 数据中 classification 与 QA 占比较高，而 API prompt 中 generation 和 brainstorming 更重要；另一个解释是 public datasets 的 input diversity 不足以代表真实用户需求。

### Truthfulness 与 Hallucination

在 TruthfulQA 上，PPO 模型比 GPT-3 更常生成 truthful and informative output。这个改善并不完全依赖显式的“请说真话” prompt；在带有 helpful instruction 的设置下，PPO 更倾向于避免自信地说出错误答案。

在 API 的 closed-domain 任务中，InstructGPT hallucination rate 约为 21%，GPT-3 约为 41%，大致降低一半。这里的 hallucination 指输出加入输入中不存在的信息，不等于对开放世界事实的完整 truthfulness。

论文也提醒：TruthfulQA 的自动指标曾被高估，作者在 acknowledgement 中专门说明了这一问题。因此，TruthfulQA 的 improvement 需要结合 human evaluation 和 closed-domain hallucination 结果，而不应把某一个自动分数视作最终结论。

### Toxicity 与 Bias

在 RealToxicityPrompts 上，当 prompt 明确要求模型生成 respectful output 时，InstructGPT 的 toxic output 少于 GPT-3，自动 Perspective API 与 human evaluation 都支持这一趋势。

但这个改善高度依赖 prompt condition：

- `respectful prompt`：InstructGPT 较少 toxic；
- `no prompt`：优势明显减弱；
- 明确要求生成 toxic output：InstructGPT 可能比 GPT-3 更 toxic。

这说明 instruction following 的增强具有双刃剑性质：模型更能执行用户指令，也可能更能执行有害指令。对 bias 的 Winogender 和 CrowS-Pairs 评测则没有显示稳定改善；在要求 respectful 时，模型甚至可能因为输出分布变得更确定而表现出更低 entropy、更高 measured bias。

因此，helpfulness、truthfulness、toxicity 和 bias 不是一个可以用单一 reward 统一代表的指标。一个模型在 respectful prompt 下更安全，不代表它在所有 prompt 条件和部署场景下都更 harmless。

### Alignment Tax 与 PPO-ptx

默认 PPO 在 DROP、SQuAD、HellaSwag 和 French-to-English translation 等 public NLP datasets 上出现 regression。增加 KL coefficient 可以限制 policy drift，但不一定恢复这些任务能力。

PPO-ptx 将 pretraining data language-modeling update 混入 PPO，显著缓解了这些 regression，并在 HellaSwag 上超过 GPT-3。它在 DROP、SQuADv2 和 translation 上仍有一定落后，说明 pretraining mix 是缓解策略，不是完整解决方案。

论文的关键对比是：

```text
增加 KL coefficient
  -> 只能限制偏离 SFT reference

加入 pretraining gradient
  -> 直接训练 policy 恢复原始语言分布能力
```

但 PPO-ptx 的数据比例与 coefficient 仍需要调节。附录实验显示，pretraining data ratio 过低时恢复能力不够，过高则训练时间增加且可能把 pretraining 中的不良行为重新带回 policy；论文选择 ratio 8 作为训练速度和能力保持之间的折中。

### Held-out Labelers 与泛化

模型对没有参与训练数据的 held-out labelers 仍保持相近的 preference 优势，说明结果不只是记忆 training labelers 的个别偏好。RM 的 5-fold labeler split 实验中，RM 在 held-out group 上的 preference prediction accuracy 为 $69.6\pm0.9\%$，在 training group 上为 $72.4\pm0.4\%$，存在下降但仍具有泛化能力。

这类泛化证据是积极信号，但不能推出模型对所有用户或所有价值观都 aligned。held-out labelers 来自相近的供应商和筛选体系，不是完整社会群体的代表。

### Qualitative Generalization

论文展示了 InstructGPT 在训练数据很少覆盖的任务上出现 instruction-following 泛化：

- 可以在部分非英语 instruction 下执行任务，但经常用 English 输出；
- 能够更可靠地回答和总结代码；
- 可以处理 code question，而 GPT-3 往往需要精心设计 prompt；
- 对复杂多约束指令仍容易失败；
- 可能接受 false premise，或者面对简单问题过度 hedging。

这些结果支持“instruction following 是一种可迁移行为先验”的判断，但论文没有对非英语、code 或多约束任务做完整定量 benchmark，因此应当作为 qualitative evidence，而不是与主结果同等强度的结论。

## 训练成本与工程取舍

论文给出的训练计算量可以帮助理解 alignment 的成本：

- 175B SFT model：约 4.9 petaFLOPs/s-days；
- 175B PPO-ptx model：约 60 petaFLOPs/s-days；
- GPT-3 pretraining：约 3,640 petaFLOPs/s-days。

在论文的 setting 中，RLHF fine-tuning 的计算成本显著低于从头 pretraining，但 PPO-ptx 仍比 SFT 重很多，因为它要承担 rollout、RM inference、value、PPO update 和额外 pretraining mix。这里的比较不包含完整人工标注成本和基础设施维护成本，不能简单等同于总项目成本。

## 关键结论

### SFT、RM、PPO 各自解决不同问题

```text
SFT:
  给模型一个可用的 instruction-following 起点

RM:
  学习同一 prompt 下多个 response 的相对偏好

PPO:
  让当前 policy 在 rollout 上主动提高 reward

PPO-ptx:
  在 reward optimization 之外，恢复 pretraining distribution 能力
```

把三者混成“用人类数据训练模型”会丢失最重要的工程边界：示范数据不等于 preference data，RM score 不等于真实目标，PPO update 也不等于 SFT token imitation。

### Reward 优化必须配合能力保持和多维评估

只优化 RM 可能让模型越来越擅长得到 RM 分数，却在传统任务、事实性或安全边界上退化。PPO-ptx 说明 pretraining data mix 可以降低这种代价，但也带来新的数据和权重选择问题。

因此，RLHF 的 evaluation 至少要同时看：

- target prompt distribution 上的 human preference；
- instruction/constraint following；
- truthfulness 与 closed-domain hallucination；
- toxicity、bias 和 harmful instruction；
- public NLP capability retention；
- held-out user/labeler generalization；
- 长度、拒答、过度 hedging 和 reward hacking 等行为变化。

### 论文的 alignment 对象是一个具体的 reference group

论文没有对抽象的“人类价值”完成 alignment，而是对由下列因素共同决定的偏好进行了优化：

1. training labelers 写的 demonstrations 和 rankings；
2. researchers 编写的 labeling instructions 和处理 edge cases 的方式；
3. API customers 提交的 prompt 分布；
4. 用户和客户使用 API 的特定场景。

这些因素可能互相冲突，也无法代表所有受模型影响的人群。这个边界不是附带的伦理声明，而是直接决定训练数据、reward model 和 evaluation 结果如何解释。

## 对后续 LLM 与 Agent Training 的启发

### Preference data 的核心不是数量，而是覆盖 policy 行为分布

InstructGPT 已经展示：RM 训练数据主要来自多个模型对同一 prompt 的比较，PPO 又会改变 policy 的 response 分布。因此，RM 不应只在静态、早期、窄分布数据上训练；policy 更新后需要继续收集能覆盖新行为的 comparison data。

对 long-horizon Agent，response 可以扩展为包含 tool call、observation 和多轮行动的 trajectory。此时还要决定：人类或 judge 比较完整 trajectory、局部 workflow，还是最终 outcome；reward 如何从 sequence-level 反馈分配到 action/token；如何避免对环境 observation 或格式表面特征过度奖励。

### PPO-ptx 对能力保持的启发

如果 agent post-training 只追逐任务 reward，模型可能在目标任务上提高，却损害通用语言、代码和知识能力。PPO-ptx 提供了一个直接的工程思路：保留一部分 broad pretraining distribution 的 NTP signal，作为能力保持项，而不是只依赖 KL 将 policy 拉回 reference。

但这不是简单地把数据混入 batch 就结束。需要观察：

- pretraining mix 的数据质量和领域覆盖；
- mix ratio 与 RL reward 的梯度冲突；
- 训练时间和吞吐成本；
- 是否把 pretraining 中不需要的行为重新引入；
- target agent capability 是否被过强的 retention objective 抑制。

### 评测需要区分目标效果和副作用

InstructGPT 的评测设计可以迁移到 Agent：不要只看最终任务成功率或 judge score，还要区分：

- 是否完成用户任务；
- 是否遵守工具 schema 和显式约束；
- 是否减少无依据陈述和错误操作；
- 是否引入新的长度、拒答、模板化或 reward hacking；
- 是否损害非目标能力；
- 是否对新用户、新领域和未见工具泛化。

一个更高的 aggregate reward 只有在这些副作用没有同步恶化时，才更接近真实能力提升。

## 局限与疑问

1. **训练和评估偏好并不完全一致。** 训练阶段偏向 helpfulness，最终评估更强调 truthfulness 和 harmlessness；这会影响模型行为和结果解释。
2. **Labeler 群体有限。** 约 40 名 labelers、主要英语数据和特定供应商群体不能代表所有用户和受影响者。
3. **大多数比较只由一名 contractor 标注。** 成本约束降低了多标注者一致性分析的覆盖范围。
4. **Preference 与真实用户意图可能不同。** labeler 只能根据 prompt 推断 intent，并不一定知道 customer 的部署上下文。
5. **PPO reward 仍可能被 exploit。** KL 与 pretraining mix 能缓解，但不能保证 reward model 对分布外行为可靠。
6. **PPO-ptx 不能完全消除 alignment tax。** DROP、SQuADv2 和 translation 等任务仍可能退化。
7. **Safety 改善具有条件性。** respectful prompt 下 toxicity 降低，不代表无条件或 harmful instruction 场景都安全。
8. **Bias 没有稳定改善。** 更强的 instruction following 可能让模型更确定地输出 stereotypical behavior。
9. **Qualitative generalization 证据有限。** code、非英语和复杂约束结果主要是案例展示，缺乏完整的定量覆盖。
10. **论文不是 Agent training 论文。** 其 bandit-style prompt-response environment 与多轮工具交互、环境反馈和 long-horizon credit assignment 仍有明显差距。

## 我的理解

InstructGPT 的历史意义不只是证明“人类反馈有效”，而是把 assistant 训练拆成了可以独立分析的几个对象：示范数据塑造基本行为，偏好数据训练可优化的 proxy，policy optimization 放大偏好，pretraining mix 负责保留底层能力，多维评估负责检查收益和副作用。

其中最值得警惕的是两个闭环：

```text
policy -> response distribution -> preference data -> reward model -> policy

RL objective -> capability shift
              -> public-task regression
              -> pretraining mix / KL / evaluation
```

第一个闭环说明 RM 必须面对 policy 当前实际生成的行为；第二个闭环说明 post-training 不是单向增加能力，目标能力和基础能力之间需要持续测量和权衡。

因此，对 InstructGPT 最准确的总结不是“PPO 让 GPT-3 变成了 ChatGPT”，而是：**论文建立了一种以用户 prompt 分布为中心、以 human preference 为监督、以 reward model 为中间层、以 PPO 为优化器、以 broad capability retention 和多维 evaluation 为约束的 assistant training recipe。** 后续的 DPO、GRPO、AI feedback、verifier RL 和 Agent trajectory training 都可以看作在这条 recipe 的某个环节上做替换或扩展。

## 相关知识链接

- [[training/post-training/rlhf|RLHF]]
- [[training/post-training/sft|SFT]]
- [[training/post-training/reward-model|Reward Model]]
- [[training/post-training/ppo|PPO]]
- [[training/post-training/instruction-tuning|Instruction Tuning]]
- [[training/post-training/grpo|GRPO]]
- [[fundamentals/information-theory/kl-divergence|KL Divergence]]
- [[sources/papers/2017-deep-rl-from-human-preferences|Deep Reinforcement Learning from Human Preferences]]
- [[sources/papers/2020-learning-to-summarize-from-human-feedback|Learning to summarize from human feedback]]
