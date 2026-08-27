---
title: "Internalizing the Future: A Unified Agentic Training Paradigm for World Model Planning"
created: 2026-06-30
published: 2026-06-30
modified: 2026-06-30
type: source
status: processed
source_type: paper
area: sources
tags:
  - source
  - paper
  - agent
  - world-model
  - mid-training
  - reinforcement-learning
aliases:
  - World Model Agentic Training
  - WM-AMT
  - FE-SFT
  - FC-RL
source_url: https://arxiv.org/abs/2606.27483
paper_date: "2026-06"
paper_order: "27483"
---

# Internalizing the Future: A Unified Agentic Training Paradigm for World Model Planning

## 基本信息

| 字段 | 内容 |
|---|---|
| 来源 | arXiv:2606.27483v1 |
| 标题 | Internalizing the Future: A Unified Agentic Training Paradigm for World Model Planning |
| 作者/机构 | Xuan Zhang, Zhijian Zhou, Lingfeng Qiao, Yulei Qin, Ke Li, Xing Sun, Xiaoyu Tan, Chao Qu, Yuan Qi / Fudan University, Shanghai Innovation Institute, Tencent Youtu Lab |
| 日期 | 2026-06-25 |
| 链接 | https://arxiv.org/abs/2606.27483 |
| 相关 topic | [[application/agents/planning|Planning]], [[application/agents/agent|Agent]], [[training/mid-training/continued-pretraining|Continued Pretraining]], [[training/mid-training/capability-injection|Capability Injection]], [[training/post-training/grpo|GRPO]], [[training/post-training/reward-model|Reward Model]], [[training/pretraining/reinforcement-pretraining|Reinforcement Pretraining]] |

这篇论文讨论的是 LLM agent 在长程任务中的一个基础限制：多数 agent 仍然是 reactive policy，只根据当前历史决定下一步行动，而缺少类似人类 “what-if” 的内部未来模拟能力。论文不试图引入外部 simulator 或显式 value head，而是训练同一个 autoregressive policy 在行动前生成两类文本化信号：一段 compact prospective rollout，以及一个 plan-conditioned success estimate。后者可以理解为 Q-value 的文本化近似，而不是严格的数值 value function。

论文的核心判断是：只在 post-training 阶段给模型加入 look-ahead 格式是不够的。模型可能学会 `<imaginary>...</imaginary>`、`<confidence>...</confidence>` 之类表面结构，却没有真正预测未来状态和校准成功概率的能力。作者将这一失败模式称为 **format-capability gap**。

为解决这一问题，论文提出三阶段训练范式：

```text
World Model Agentic Mid-Training (WM-AMT)
  → Format-Eliciting SFT (FE-SFT)
  → Foresight-Conditioned RL (FC-RL)
```

如果只保留一个核心观点：**agent 的 look-ahead planning 不能只靠格式监督诱导，必须先在 mid-training 阶段注入预测未来的能力，再用 SFT 结构化表达，最后用 RL 依据真实执行结果校准。**

## 研究问题

标准 Chain-of-Thought 更关注当前问题的分解与推理，而不是对未来行动后果的模拟。对于搜索、工具使用、数学解题和长程 agent 任务，真正困难的地方往往在于：采取某个计划后会遇到什么信息、可能失败在哪里、当前路径成功概率有多高、是否应该改走另一条路线。

传统 model-based RL 会学习 transition dynamics $\hat{P}(s_{t+1}\mid s_t,a_t)$ 和 reward $\hat{R}(s_t,a_t)$，再利用模型进行 look-ahead planning。但在 LLM agent 中，状态和 observation 往往是长文本、工具返回、网页内容或推理轨迹。直接逐 token 预测未来完整 trajectory 不仅计算昂贵，还会把大量容量浪费在表层文本细节上，并带来 compounding errors。

论文因此将 LLM agent 的 world model 重新定义为 **internal semantic simulator**。它不生成完整低层未来状态，而是生成压缩后的未来经验摘要：

$$
\hat{z}_t = \Psi(\tau_{\text{future}})
$$

其中 $\Psi$ 是语义压缩函数，把未来 rollout 映射为高层预测性 foresight。policy 从纯 reactive：

$$
\pi_\theta(a_t \mid s_t)
$$

转向 foresight-conditioned：

$$
\pi_\theta(a_t \mid s_t, \hat{z}_t)
$$

研究问题由此变成：如何让一个 causal language model 内化这种未来模拟和行动价值评估能力，同时避免它只是在输出格式上模仿 world model。

## 核心主张

论文的核心主张有三层。

第一，world model planning 应该内化到 policy，而不是只依赖外部模拟器。外部 world model 可以提供未来预测，但会带来额外推理开销、simulator-policy mismatch，以及复杂系统耦合。对于语言 agent，未来模拟可以被压缩成文本化计划、关键词、gap analysis 和 confidence estimate，并与 policy tokens 共用同一表示空间。

第二，future-aware format 与 future-aware capability 必须分开。SFT 擅长让模型学会某种输出格式，RL 擅长强化已有行为，但它们都不擅长从零注入新的预测能力。如果 base policy 没有未来动态先验，那么强制它输出 foresight block 只会得到看似合理但事实不接地的文本。

第三，文本化 Q-like confidence 需要在真实执行反馈中校准。论文不仅要求模型预测未来步骤，还要求它给出当前计划成功概率 $q_t\in[0,100]\%$。这个分数在 mid-training 中由 LLM 基于 future trajectory 启发式生成，但最终要在 RL 阶段用实际任务成败和 Brier score calibration 来修正。

## 方法与机制

### Format-capability Gap

论文首先指出一个实践失败模式：直接在 post-training 中使用带 look-ahead traces 的 SFT，再接普通 RL，模型可以获得很高的 format adherence，但 future simulation 的内容可能并不准确。它会生成结构完整的 `<imaginary>`、`<keyword>`、`<analysis>`、`<confidence>`，却无法可靠预测后续检索、工具结果或数学推理路径。

附录中的初步实验支持这一点。对于 Youtu-LLM-2B-Base，FE-SFT 在 Search 上达到 99.86% format adherence，但平均分从 Standard SFT 的 38.2 降到 37.4；FE-SFT + RL 虽然保持 99.97% format adherence，但平均分只从 46.1 到 47.0。数学任务中 FE-SFT 的 format adherence 也接近 98.63%，但相对 Standard SFT 的收益有限。作者还在 Llama-3.2-3B-Instruct 和 Qwen2.5-7B-Instruct 上复现了类似现象。

这说明格式正确不等于语义预测正确。论文将其归因于 post-training 的能力边界：SFT/RL 更适合 eliciting latent capabilities，而不是 injecting fundamentally new capabilities。

### World Model Agentic Mid-Training

WM-AMT 是论文的能力注入阶段。给定原始 agentic trajectory dataset：

$$
D_{\text{raw}} = \{\tau_i\}_{i=1}^N
$$

每条 trajectory 表示为：

$$
\tau = (s_1,a_1,s_2,a_2,\ldots,s_T,a_T)
$$

论文从每条多轮轨迹中采样若干 step $t$，用当前状态 $s_t$ 和未来轨迹 $\tau_{>t}$ 构造 world model block：

$$
z_t = [z_t^{(\text{traj})}\oplus z_t^{(\text{eval})}]
$$

其中 $z_t^{(\text{traj})}$ 是对未来路径的抽象 roadmap，$z_t^{(\text{eval})}$ 包含当前 gap analysis 和成功概率估计 $q\in[0,100]\%$。随后把 $z_t$ 插入原轨迹：

$$
\tilde{\tau} = (\ldots, s_t, z_t, a_t, s_{t+1}, \ldots)
$$

训练目标仍然是标准 autoregressive negative log-likelihood：

$$
L_{\text{WM-AMT}}(\theta)
=
-\mathbb{E}_{\tilde{\tau}\sim D_{\text{WM-AMT}}}
\left[
\sum_i \log P_\theta(u_i\mid u_{<i})
\right]
$$

关键在于：模型必须在只看到 $s_t$ 之前历史的条件下预测 $z_t$，因此训练目标迫使它学习“从当前状态预测未来摘要和成功估计”。

数据构造使用 DeepSeek V3.1 作为 summarization function $\Phi(\cdot)$。prompt 要求模型基于 FUTURE GROUND TRUTH 生成 world model block，但规定两条重要约束：不得泄漏具体 future entities，只能用 abstract placeholders；同时要给出具体下一步 intent 和 confidence。这样做承认了训练数据来自后验轨迹，但试图通过抽象化减少直接答案泄漏。

原始 agentic trajectory data 约 200B tokens，包括：

| 数据类型 | 规模 |
|---|---:|
| Agentic-CoT trajectories | 25B |
| Mathematical trajectories | 20B |
| Code execution trajectories | 70B |
| Deep Research trajectories | 60B |
| Tool using / function calling / planning 等其他轨迹 | 25B |

WM-AMT 主要对 Deep Research 和 mathematical trajectories 插入 world model block。增强后，Deep Research 约 62B tokens，mathematics 约 21B tokens，其余轨迹类别保持不变。

### Format-Eliciting SFT

WM-AMT 注入的是 latent look-ahead capability，但模型还不一定会在推理时稳定调用它。因此 FE-SFT 的任务是让模型学会结构化外显这种能力。

FE-SFT 的输出格式包含两个部分：

```text
World Model Span:
  <imaginary>...</imaginary>
  <keyword>...</keyword>
  <analysis>...</analysis>
  <confidence>...</confidence>

Policy Span:
  <think>...</think>
  <action>...</action>
```

Search 任务中，论文构造约 4k trajectories，并改写成约 13k single-turn samples 以适配 verl-agent 框架。数学任务中构造约 2k SFT trajectories。FE-SFT 的 foresight block 生成 prompt 与 WM-AMT 基本一致，只额外加入结构化格式；关键词抽取也由 DeepSeek V3.1 完成。

这一阶段的作用不是从零学习未来预测，而是把 WM-AMT 中形成的能力映射到固定 schema，使后续 RL 可以分别定位 world model span 和 policy span。

### Foresight-Conditioned RL

FC-RL 是论文的校准与强化阶段。它将一次 agent step 拆成两个 span：

```text
World Model Span:
  imaginary → keyword → analysis → confidence(q_t)

Policy Span:
  think → action(a_t)
```

训练时，对同一个 prompt 采样一组 $N$ 条 trajectories。FC-RL 同时优化 world model span 和 policy span。

**World model optimization** 关注 foresight grounding 和 confidence calibration。

首先是 grounding reward。模型在 world model span 中抽取若干关键词 $K_t$，论文定义：

$$
R^{i,t}_{\text{ground}}
=
R_{\text{match}}(K^{(i)}_t,\tau^{(i)}_{>t})
$$

只有当预测关键词确实在后续实际执行轨迹中出现，模型才获得该项奖励。这用于惩罚不接地的未来模拟。

其次是 calibration reward。模型给出 confidence $q_t$，episode 结束后得到终局正确性指示 $I$。论文用 Brier score 风格的奖励校准 confidence：

$$
R^{i,t}_{\text{calib}}
=
I - (q^{(i)}_t - I)^2
$$

其中 $I=1$ 表示最终答案正确，$I=0$ 表示错误。直观上，如果模型对错误路径给出高置信度，会受到惩罚；如果对正确路径高置信、错误路径低置信，则获得更好奖励。

world model step reward 为：

$$
R_{\text{WM}}(\tau^t_i)
=
R^{i,t}_{\text{calib}}
+
R^{i,t}_{\text{ground}}
$$

然后在 group 内归一化得到 world model advantage，用于优化 world model span。

**Policy optimization** 使用终局任务 reward 与 step-level confidence 的组合。令 $A_E(\tau_i)$ 是 episode reward 在 group 内归一化后的 global advantage，令 $A_S(\tau^t_i)$ 是 confidence scores $q_t$ 在对应 step group 内归一化后的 local advantage，则 policy advantage 为：

$$
A(\tau^{[\text{policy}]t}_i)
=
A_E(\tau_i)
+
\omega A_S(\tau^t_i)
$$

其中 $\omega$ 平衡最终任务成功与局部 foresight alignment。policy 使用 clipped GRPO surrogate objective 更新。论文在 search task 中使用 $\omega=0.2$，数学任务中使用 $\omega=0.1$。

这个设计的重点是：confidence 不只是生成内容，也成为 policy 的局部学习信号；但由于 confidence 同时被 Brier calibration 约束，论文认为它比未校准的自评分更不容易造成 reward hacking。

## 实验与证据

### 实验设置

基础模型是 Youtu-LLM-2B 的一个 intermediate checkpoint，称为 Youtu-LLM-2B-Init。论文比较两个 mid-training variants：

| 模型 | Mid-training |
|---|---|
| Youtu-LLM-2B-Base | 标准 agentic mid-training，使用 200B high-quality agentic trajectory data |
| Youtu-LLM-2B-Base (WM-AMT) | 在同一 trajectory data 上插入 world model block |

评测覆盖两个领域。

Search 任务包含 7 个 search-augmented QA datasets：NQ、TriviaQA、PopQA、HotpotQA、2Wiki、MuSiQue、Bamboogle。检索知识源为 2018 Wikipedia dump，retriever 为 E5，每次返回 3 个 passages。结果通过 DeepSeek V3.1 做 LLM-as-a-judge。

数学推理任务使用 AIME2024、AIME2025、AIME2026，重复评测 30 次，报告 mean@30 和 pass@30。模型可以使用 code interpreter 工具。

### 三阶段的必要性

Search 主结果显示，在 SFT 阶段，WM-AMT 相比标准 mid-training 已有收益：

| Setting | Search Avg. |
|---|---:|
| Youtu-LLM-2B-Base + SFT | 38.2 |
| Youtu-LLM-2B-Base + FE-SFT | 37.4 |
| Youtu-LLM-2B-Base (WM-AMT) + SFT | 39.9 |
| Youtu-LLM-2B-Base (WM-AMT) + FE-SFT | 41.8 |

这说明 FE-SFT 只有在 WM-AMT 已经注入未来动态先验时才真正有效。没有 WM-AMT 时，加入 foresight format 反而可能损害表现。

RL 后结果进一步支持完整 pipeline：

| Setting | Search Avg. |
|---|---:|
| SFT + RL on standard base | 46.1 |
| FE-SFT + RL on standard base | 47.0 |
| State-SFT + RL without verbalized Q | 47.1 |
| IWM + RL | 48.4 |
| WM-AMT + FE-SFT + RL | 49.1 |
| WM-AMT + FE-SFT + FC-RL | 50.6 |

完整三阶段方法达到 50.6，是表中最高结果。论文特别强调，state-only future prediction 不如加入 confidence estimation 与 FC-RL calibration 的方法，说明仅预测未来状态还不够，成功概率估计和校准也很关键。

数学任务也呈现相同趋势：

| Setting | mean@30 Avg. | pass@30 Avg. |
|---|---:|---:|
| SFT + RL on standard base | 28.0 | 52.2 |
| FE-SFT + RL on standard base | 28.0 | 51.1 |
| State-SFT + RL without verbalized Q | 27.1 | 56.7 |
| IWM + RL | 28.6 | 57.8 |
| WM-AMT + FE-SFT + RL | 28.4 | 55.5 |
| WM-AMT + FE-SFT + FC-RL | 29.5 | 60.0 |

最终方法在 mean@30 和 pass@30 上都最好。

### Multi-hop 任务收益更明显

论文观察到，world-model planning 对复杂 multi-hop search 的收益更显著。例如在 WM-AMT 模型上，FE-SFT 相比 SFT 将 MuSiQue 从 20.6 提升到 25.6，将 2Wiki 从 31.2 提升到 35.8。最终 FC-RL 模型在 HotpotQA、MuSiQue、Bamboogle 等多步任务上也有较强表现。

这符合方法动机：single-hop retrieval 更依赖直接事实检索，未来模拟的重要性有限；multi-hop QA 需要连续搜索、验证和路径调整，更需要模型在行动前预测后续信息缺口和成功概率。

### FC-RL 组件消融

论文对 $\omega$ 做敏感性分析。Search 任务中：

| $\omega$ | Search Avg. |
|---:|---:|
| 0 | 49.1 |
| 0.1 | 50.0 |
| 0.2 | 50.6 |
| 0.5 | 49.2 |

$\omega=0$ 等价于不使用 confidence 作为 local policy advantage，只做标准 GRPO。适度加入 confidence signal 带来收益，但过大 $\omega$ 会使模型过度关注局部 foresight 信号，干扰基本任务推理。

grounding reward 的消融也显示轻微收益：加入 $R_{\text{ground}}$ 后 Search Avg. 从 50.1 提升到 50.6。论文解释为 grounding reward 能帮助模型把关键词预测和后续检索证据对齐，减少 hallucinated foresight。

### RL-only 失败

附录还评估了是否可以跳过 SFT，直接用 RL 训练。结果显示 RL-only 在 Search 上严重退化：标准 base 上平均 23.3，WM-AMT base 上平均 23.7，远低于 SFT+RL 或 FE-SFT+FC-RL。

论文分析其原因是 tool-integrated search 的 sparse reward 会导致 alignment collapse。如果没有 SFT 先建立工具调用 syntax 和 reasoning schema，policy 在 RL 后期会忘记如何使用 search tool，坍缩成单轮直接猜答案。这是一个重要结论：即使有 WM-AMT，RL 仍然需要 SFT 提供结构先验。

### 推理开销

FE-SFT 引入 world model block 会增加输出 token。论文报告 Search 平均 tokens 从 2603.8 增加到 3084.5，约 +18.5%；数学推理从 4601.2 增加到 4749.1，约 +3.2%。作者认为增加有限，说明模型生成的是压缩未来模拟而不是冗长展开。

## 关键结论

论文最重要的贡献不是提出新的 tag format，而是把 agent look-ahead planning 拆成能力注入、格式诱导和 reward 校准三个阶段。这个拆分非常关键：若没有 WM-AMT，FE-SFT 只能得到高 format adherence；若没有 FE-SFT，latent world model 不一定会在推理中稳定外显；若没有 FC-RL，confidence 和 future simulation 可能无法被真实执行反馈校准。

文本化 Q-value 是论文中比较有辨识度的设计。它不像传统 value head 那样在模型外部增加标量预测器，而是让模型在自然语言 span 中显式表达成功概率，并在 RL 中把这个 self-assessed confidence 同时作为校准对象和局部 policy signal。这使 world model 更可解释，但也带来新的风险：confidence 可能变成被模型操控的中间 reward，因此论文用 Brier score 和 grounding reward 约束它。

与 AgentFounder、daVinci-Dev 一样，这篇论文继续强化了一个趋势：agent 能力不能完全依赖 post-training 末端补齐。AgentFounder 关注 agentic behavior distribution 的 continued pretraining，daVinci-Dev 关注 software engineering workflow 的 agent-native mid-training；本文则关注未来模拟和 action value estimation 的 world-model-style mid-training。三者都把一部分 agent 能力前移到 SFT/RL 之前。

## 局限与疑问

第一，训练数据使用 future ground truth 构造 world model block，存在后验信息风险。论文通过 abstract placeholders 和 no-spoilers 规则降低直接泄漏，但模型仍然从成功未来轨迹中学习预测摘要。这适合作为能力注入数据，但如果 prompt 设计不严格，可能变成把未来答案压缩进 foresight block。

第二，confidence 的初始标注由 LLM 启发式估计，而不是由多次 Monte Carlo rollout 计算。作者承认准确估计单步 Q-value 理想上需要大量 rollouts，但成本过高。因此 mid-training 阶段的 $q$ 噪声较大，需要后续 FC-RL 校准。这意味着方法对 RL 校准质量依赖很强。

第三，实验规模和任务范围有限。主模型是 Youtu-LLM-2B，任务是 search-augmented QA 和 AIME 数学推理。虽然作者在若干其他模型上验证 format-capability gap，但完整 WM-AMT 没有扩展到更大模型、code agent、web agent、OS agent 或真实 long-horizon software engineering 任务。

第四，Search 结果使用 DeepSeek V3.1 作为 LLM-as-a-judge，reward 和 evaluation 都依赖 judge 的可靠性。对于事实问答，LLM judge 可能受到表达差异、证据缺失和自身知识偏差影响。数学任务更可验证，因此结论更稳。

第五，grounding reward 的实现依赖 keyword 与后续轨迹匹配。关键词匹配是低成本 proxy，但并不等价于 causal prediction 正确。模型可能学到容易在后续文本中出现的宽泛关键词，而不是精确预测未来状态。

第六，文本化 world model 会增加推理长度。论文报告开销不大，但 Search 任务仍增加约 18.5% tokens。对于高吞吐 agent 系统，额外 token、judge、keyword extraction 和 RL rollout 成本都需要纳入系统预算。

第七，FC-RL 的 local confidence advantage 可能引入复杂优化动态。$\omega$ 过小收益有限，过大又会损害任务推理。不同任务中 confidence 与真实成功率的相关性不同，参数迁移未必稳定。

## 分析与启发

这篇论文对 agent 训练的一个重要启发是：规划能力可以进一步拆分为“生成计划”和“预测计划后果”。很多 agent 训练只监督下一步 action 或整条 trajectory，但没有显式要求模型判断当前计划未来是否顺利、还缺什么信息、成功概率如何。World model block 把这些内容变成可训练、可校准的中间对象。

format-capability gap 的提出也很有价值。它提醒后续做 reasoning / planning 数据时，不能只看模型是否遵守结构化格式。一个模型可以非常稳定地输出 `<imaginary>` 和 `<confidence>`，但其中的未来预测可能是幻觉，confidence 也可能严重失真。对 agent 数据而言，格式只是接口，能力需要在更早阶段通过大量有时序结构的数据形成。

与 RLP 相比，本文的 reward 更贴近 agent task execution。RLP 用 thought 对 next-token likelihood 的增益作为 reward，适合把强化学习目标前移到文本预测；本文则用真实任务成败、关键词 grounding 和 confidence calibration 来优化 future-aware planning，更接近 agent post-training / mid-training 的交叉地带。

对知识库中已有的 code agent 和 deep research agent 训练路线而言，这篇论文可以作为“world-model planning”分支：Agentic CPT / daVinci-Dev 更关注行为分布和工作流数据，本文更关注模型是否能够在执行前生成压缩未来模拟，并用真实反馈校准这种模拟。未来如果构造 code agent trajectory reasoning 数据，可以借鉴其分层思路：先用后验轨迹抽象未来计划，再用严格规则避免泄漏，最后用测试或环境反馈校准 confidence。

## 相关知识链接

- [[application/agents/planning|Planning]]
- [[application/agents/agent|Agent]]
- [[training/mid-training/continued-pretraining|Continued Pretraining]]
- [[training/mid-training/capability-injection|Capability Injection]]
- [[training/post-training/grpo|GRPO]]
- [[training/post-training/reward-model|Reward Model]]
- [[training/pretraining/reinforcement-pretraining|Reinforcement Pretraining]]
