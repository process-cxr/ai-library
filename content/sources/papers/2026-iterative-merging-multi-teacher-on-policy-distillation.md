---
title: "No Pain, More Gain: Iterative Merging for Effective Multi-Teacher On-Policy Distillation"
created: 2026-10-28
published: 2026-10-28
modified: 2026-10-29
type: source
status: processed
source_type: paper
area: sources
tags:
  - source
  - paper
  - on-policy-kd
  - multi-teacher
  - model-merging
  - task-vector
  - capability-integration
aliases:
  - IM-MOPD
  - Iterative Merging for MOPD
source_url: https://arxiv.org/abs/2609.34745
pdf_url: https://arxiv.org/pdf/2609.34745v1
paper_date: "2026-09"
paper_order: "34745"
---

# No Pain, More Gain: Iterative Merging for Effective Multi-Teacher On-Policy Distillation

## 基本信息

- 标题：[No Pain, More Gain: Iterative Merging for Effective Multi-Teacher On-Policy Distillation](https://arxiv.org/abs/2609.34745)
- 作者：Seonghyeon Kim、Chaeyun Jang、Noah Lee、Boseop Kim、Juho Lee
- 机构/来源：KAIST AI、Kakao
- 发表：arXiv:2609.34745v1
- 日期：2026-09
- 代码/数据：[IM-MOPD](https://github.com/shkim0824/immopd)
- 相关主题：[[training/post-training/on-policy-kd|On-policy KD]]、[[training/post-training/model-merging|Model Merging]]、[[training/post-training/knowledge-distillation|Knowledge Distillation]]、[[training/post-training/grpo|GRPO]]、[[sources/papers/2026-mopd-multi-teacher-on-policy-distillation-for-capability-integration|MOPD]]

## 研究问题：相同来源的教师，为什么仍然难以整合

多教师 on-policy distillation（Multi-Teacher On-Policy Distillation，MOPD）将 on-policy distillation（OPD，基于学生自身采样的策略蒸馏）扩展到多个领域教师：让学生先生成回答，再按问题领域选择对应教师，在学生已经生成的前缀上提供逐 token 监督。它把不同领域教师的能力汇入一个学生，不必在推理时同时运行多个专家。

这条路线有一个容易被忽略的前提：**教师要能在学生生成的状态上提供有用反馈。** 教师单独解题很强，不代表学生走到某个前缀后，教师仍能可靠理解或延续它。共同初始化可以减小差异，但不保证后续训练得到的行为分布仍然兼容。

论文研究的情形是：所有教师具有相同架构、来自同一个 reference checkpoint，但后续接受不同领域与不同训练目标的优化。其中有监督微调（Supervised Fine-Tuning，SFT）教师，也有基于可验证奖励的强化学习（Reinforcement Learning with Verifiable Rewards，RLVR）教师。

作者观察到，学生直接以 reference 初始化并开始 MOPD 时，能较好恢复金融领域能力，但医疗和工具使用能力的恢复明显不足。让学生先做 SFT warm-up 能缓解问题，但会增加一个混合领域训练阶段，而且 warm-up 后的即时分数并不能充分预测后续蒸馏结果。

论文提出 Iterative Merging for MOPD（IM-MOPD，面向 MOPD 的迭代参数合并）：先均匀合并教师参数初始化学生，在蒸馏过程中定期检查领域恢复率，对恢复不足的领域再加入固定增量的 task vector，然后继续蒸馏。

核心不是换一个蒸馏 loss，而是改变**学生进入后续学习时所处的参数状态和前缀分布**。

## 先区分三种模型与两条更新路径

### Reference、领域教师与学生

设共同 reference 的参数为 $\theta_{\mathrm{ref}}$，第 $i$ 个领域教师为 $\varphi_i$，学生当前参数为 $\theta$。

```text
Qwen3-Base
  -> general reasoning SFT on OpenThoughts3
  -> shared reference checkpoint
       -> independently train Medical / Law / Tool Use teachers with SFT
       -> independently train Finance / Instruction Following teachers with RLVR
       -> combine teachers to initialize one student
            -> student rollouts + routed teacher supervision
            -> occasional task-vector additions
            -> final single student checkpoint
```

实验中的 reference 已在 OpenThoughts3 推理数据上做过 SFT，不是未经适配的 Qwen3-Base。4B 和 1.7B 两种规模分别构造自己的 reference 和同规模教师，没有把不同尺寸模型的参数混合。

### 蒸馏更新与参数修正分别做什么

| 路径                   | 输入                              | 学生怎样变化                 |
| ---------------------- | --------------------------------- | ---------------------------- |
| MOPD 更新              | 学生回答及对应教师的逐 token 分布 | 通过梯度学习教师行为         |
| Task-vector correction | 教师与 reference 的固定参数差     | 直接给当前学生参数加一个向量 |

教师在整合阶段保持固定。参数修正不是训练一个新教师，也不是更新 teacher-routing 概率；后续仍按原有领域采样规则蒸馏。

最终产物是一份经过合并和蒸馏的学生权重。论文没有要求部署 task-specific adapters、保留多个任务 head，或在推理时调用教师。教师和 reference 是训练资源，不是学生推理的必需组件。

## MOPD 目标与实际训练方式

对 $K$ 个领域，设问题分布为 $\mathcal D_i$。从领域 $i$ 采样问题 $x$，学生生成 $y=(y_1,\ldots,y_T)$，第 $t$ 个状态为 $s_t=(x,y_{<t})$。论文式 (2.1) 为：

$$
\mathcal L_{\mathrm{MOPD}}(\theta)
=\frac1K\sum_{i=1}^{K}
\mathbb E_{\substack{x\sim\mathcal D_i\\y\sim\pi_\theta(\cdot\mid x)}}
\left[
\frac1T\sum_{t=1}^{T}
D_{\mathrm{KL}}\!\left(
\pi_\theta(\cdot\mid s_t)
\,\middle\|\,
\pi_{\varphi_i}(\cdot\mid s_t)
\right)
\right].
$$

KL 是 Kullback-Leibler divergence（KL 散度）。这里使用 reverse KL，方向是学生到当前领域教师，**不是学生到 reference 的漂移约束**。不同领域问题分别找对应教师，不是在每个 token 上把所有教师的 logits 平均。

实际采用 policy-gradient 形式实现蒸馏。可用 sampled-token 视角理解其局部信号：

$$
a_t\approx
\log\pi_{\varphi_i}(y_t\mid s_t)
-\log\pi_\theta(y_t\mid s_t).
$$

教师相对更认可的学生 token 得到正信号，相对不认可的得到负信号。这是理解采样式蒸馏的局部表达，不等于论文每一步显式计算完整词表 KL，也不定义所有序列级梯度与 trainer 细节。附录说明 advantage 的幅度截断到 5，并屏蔽截断的 completions；不添加任务 reward 或 reference-KL 项（Appendix A.2）。

每个 prompt 只采一个 completion。这里不需要为同一问题先生成一组回答，再按任务奖励作组内标准化；领域教师在训练时使用过何种强化学习算法，与整合学生时使用何种蒸馏信号，是两件事。

## 为什么合并能改变后续学习

### Task vector 与全局尺度

第 $i$ 个教师的任务向量定义为：

$$
\tau_i=\varphi_i-\theta_{\mathrm{ref}}.
$$

给定非负系数 $\alpha_i$，合并初始化为：

$$
\theta_{\mathrm{merge}}
=\theta_{\mathrm{ref}}+\sum_{i=1}^{K}\alpha_i\tau_i,
\qquad
\lambda=\sum_i\alpha_i.
$$

这里要分别看两个量：

- **相对配比**：$\alpha_i/\lambda$，决定各领域参数增量的相对贡献；
- **总体尺度**：$\lambda$，决定离开 reference 的总增量尺度。

展开后：

$$
\theta_{\mathrm{merge}}
=(1-\lambda)\theta_{\mathrm{ref}}+\sum_i\alpha_i\varphi_i.
$$

当 $\lambda=1$，它是教师权重的凸平均；当 $\lambda>1$，它不再受限于这种凸平均的 simplex。非负的 task-vector 系数并不意味着整个表达式是模型权重的凸组合。

例如，五教师均匀初始化取 $\alpha_i=0.2$。附录另外测试一个固定非均匀配置，按 Medical、Law、Finance、Instruction Following、Tool Use 排列为 $(0.5,0.3,0.2,0.3,0.8)$，其和为 2.1。将它归一化到和为 1，会改变整体尺度，而不仅是“同一个比例的另一种写法”（Figure 3c、Appendix B.1）。

### 即时分数高，不等于适合继续蒸馏

Tool Use 提供了清楚的反例（Table 8）：

| 学生准备方式                       | MOPD 前工具任务成功率 | 100 次 MOPD 更新后 |
| ---------------------------------- | --------------------: | -----------------: |
| SFT warm-up                        |                 30.7% |              31.6% |
| 固定非均匀 merge，$\lambda=2.1$    |                 13.2% |              50.9% |
| Uniform merge 初始化并执行 IM-MOPD |                  0.9% |              69.3% |

Uniform merge 刚开始甚至低于 reference 的 5.3%，但沿 IM-MOPD 路径训练后显著提升。最后一行同时包含训练中修正的作用，因此不能把其全部收益归结为初始合并。

这些观察反对“只评估刚合并后的分数，就能选出最佳蒸馏起点”的做法。合并初始化的价值还取决于它让学生访问怎样的前缀，以及后续教师反馈能否在这些前缀上被有效利用。

Figure 3a 还从同一训练 checkpoint 分出两个分支：一支继续 MOPD，一支先干预合并再继续。干预可能立即降低分数，却在之后 50 次更新中获得更好的恢复。它是训练路径效应的证据，不是每一次加向量都保证即时增益。

## IM-MOPD 算法：周期性识别欠恢复领域

### 恢复率如何定义

设领域 $i$ 的原始评测分数为 $s_i(\theta)$，定义归一化分数：

$$
\widetilde s_i(\theta)
=\frac{s_i(\theta)-s_i(\theta_{\mathrm{ref}})}
{s_i(\varphi_i)-s_i(\theta_{\mathrm{ref}})}.
$$

reference 为 0，对应领域教师为 1。这个指标衡量学生恢复了 reference 到教师之间的多少分数增量，不是 student accuracy 除以 teacher accuracy。

它可以低于 0，也可以超过 1。4B 的 Tool Use 从 reference 5.3%、教师 66.7% 到学生 69.3%，得到：

$$
\frac{69.3-5.3}{66.7-5.3}\approx1.042.
$$

因此学生在这一指标上约恢复 104.2%，并不意味着任务成功率超过 100%。五域 Norm. 为：

$$
\mathrm{Norm.}=100\times\frac1K\sum_{i=1}^{K}\widetilde s_i.
$$

论文在验证集上使用相同形式的 $\widetilde r_i$ 作决策，在最终评测上使用 $\widetilde s_i$ 报告结果；两套数据和记号不能混用。分母应为正且足够稳定，教师与 reference 的分数差太小时，归一化值会对评测噪声敏感。

### 初始化、检测、加向量、再蒸馏

Algorithm 1 的过程为：

1. 用 $\theta_0=\theta_{\mathrm{ref}}+\frac1K\sum_i\tau_i$ 初始化学生。
2. 执行常规 MOPD 更新。
3. 每隔 $H$ 次更新，在验证集上测量领域恢复率。
4. 选出 $\mathcal M=\{i:\widetilde r_i(\theta)\le\gamma\}$。
5. 直接修正当前参数：$\theta\leftarrow\theta+\delta\sum_{i\in\mathcal M}\tau_i$。
6. 从修正后的学生继续 MOPD，直到训练结束。

主实验总计 100 次更新，$H=25$、$\delta=0.3$；4B 使用 $\gamma=0.6$，1.7B 使用 $\gamma=0.4$。只在第 25、50、75 次更新之后修正，最后一次 MOPD 更新后不再直接加向量（§4.1、Appendix A.4）。

$\gamma$ 是恢复率门槛，不是原始准确率门槛；$\delta$ 是任务向量增量系数，也不是学习率。选择哪个领域是验证反馈驱动的二元决定，增量大小在各领域和干预时点共享。它不是连续优化整组 $\alpha_i$ 的元学习算法。

### 实际 4B 调度与累计系数

4B 主配置的选择如下（Table 7）：

| 时点         | 加入任务向量的领域                            |
| ------------ | --------------------------------------------- |
| Update 25 后 | Tool Use、Medical、Law、Instruction Following |
| Update 50 后 | Tool Use                                      |
| Update 75 后 | Tool Use                                      |

从每域 0.2 起步，按这个调度累计的**直接加向量系数**为：

| 领域                  | 初始系数 | 修正次数 | 累计直接系数 |
| --------------------- | -------: | -------: | -----------: |
| Medical               |      0.2 |        1 |          0.5 |
| Law                   |      0.2 |        1 |          0.5 |
| Finance               |      0.2 |        0 |          0.2 |
| Instruction Following |      0.2 |        1 |          0.5 |
| Tool Use              |      0.2 |        3 |          1.1 |

由 Table 7 和 $\delta=0.3$ 可算出总直接尺度为 2.8。Tool Use 的 1.1 不是概率，也不是表示推理时调用教师 1.1 次。

### 为什么最终学生不是按这些系数再合一次就能得到

设领域 $i$ 被修正 $m_i$ 次，可在记账意义上写：

$$
\theta_{\mathrm{final}}
=\theta_{\mathrm{ref}}
+\sum_i\left(\frac1K+m_i\delta\right)\tau_i
+\Delta_{\mathrm{MOPD,path}}.
$$

最后一项是沿实际路径发生的全部蒸馏参数更新，依赖各次干预后的学生和 rollout。改变修正时序，后续梯度和前缀也会变化，所以不能把它视为与合并无关的固定残差。

另外，每次加入的是固定 $\tau_i=\varphi_i-\theta_{\mathrm{ref}}$，而不是 $\varphi_i-\theta_{\mathrm{current}}$。它没有把学生与教师重新凸平均，也不会重建一个新学生来覆盖此前的 MOPD 学习。

## 实验设置：教师来源、数据和训练预算

### 五个教师如何得到

两种尺寸都从 OpenThoughts3 适配后的共享 reference 出发。教师训练使用 NVIDIA A100 80GB GPU（Graphics Processing Unit，图形处理器）（Appendix A.1）。

| 领域                  | 教师训练方式 | 主要数据来源及规模                                                      |
| --------------------- | ------------ | ----------------------------------------------------------------------- |
| Medical               | SFT          | Qwen3.6-35B-A3B 根据 MedQA 训练问题生成并过滤的 40,853 条正确、去重样本 |
| Law                   | SFT          | 同一生成模型根据 CaseHOLD 与律师考试问题生成的 56,123 条正确、去重样本  |
| Tool Use              | SFT          | AReaL-tau2-data 的 retail、airline、telecom 数据，过滤后 32,758 条      |
| Finance               | GRPO         | FinQA 与 TAT-QA 训练集的 7,565 个数值问答 prompts                       |
| Instruction Following | GRPO         | Nemotron-RL-Ultra-Training-Blends 中 11,728 个可验证指令 prompts        |

GRPO 是 Group Relative Policy Optimization（组相对策略优化），用于这里两个 RLVR 教师的训练。TAT-QA 是 Tabular And Textual Question Answering，面向表格与文本的数值问答数据。指令遵循记为 IF（Instruction Following），后文 IFBench、IFEval 分别是相关评测集名称。

SFT 教师采用 packing、梯度范数截断 1、零 weight decay、5% warm-up 和 cosine 学习率调度。两个 GRPO 教师用 NeMo-RL，分别按数值答案和指令约束给 reward，采用 clipped policy updates 与超长惩罚，不使用 reference KL penalty（Appendix A.1）。

Tool Use 教师保留 tool schemas，只监督最后一个 assistant turn，并移除更早的 thinking。它与其他领域的训练文本和输出形态并不相同。这里观察到 SFT 教师较难恢复，是特定教师组合下的现象，不能归纳成“SFT 能力必然无法被 OPD 转移”。

### 蒸馏数据与共同预算

MOPD 的 Medical、Law 分别使用 8,384 和 31,563 个 prompts；Finance、IF 复用前述问题池。Tool Use 使用 31,718 个历史对话 contexts，每个作为生成下一 assistant turn 的输入（Appendix A.2）。

| 配置项                  | 设置                                       |
| ----------------------- | ------------------------------------------ |
| MOPD 更新数             | 所有学生方法均为 100                       |
| 每次更新 prompt batch   | 128                                        |
| 领域采样                | 均匀采样                                   |
| 每 prompt completion 数 | 1                                          |
| 学习率                  | $10^{-6}$                                  |
| Completion 长度上限     | 16,384 tokens                              |
| Sampling                | temperature 1、top-p 1                     |
| 信号处理                | advantage 幅度截断 5，mask 截断 completion |
| 额外目标                | 无任务 reward、无 reference KL             |

Tool Use 的蒸馏训练是基于保存的对话上下文生成下一 turn；最终评测则在真实多轮工具环境中运行。这不等于训练时为每个 prompt 都执行完整交互式 Agent rollout。

SFT warm-up baseline 先用五教师生成的混合数据训练到约 30% 归一化恢复，再执行同样 100 次 MOPD。4B、1.7B warm-up 分别消耗约 167M、171M tokens（Table 6）。因此“没有额外 SFT 阶段”是 IM-MOPD 相对这一 baseline 的预算优势，不代表整个方法无需训练数据、验证计算或教师调用。

### 验证与最终评测

周期性决策中，Tool Use 从完整数据集采样 128 个验证任务；其余领域各从训练池留出的验证数据采样 256 个 prompts（Appendix A.4）。

| 领域     | Benchmark 与任务                                       | 主实验指标                  |
| -------- | ------------------------------------------------------ | --------------------------- |
| Medical  | MedQA，医疗问答，1,273 题                              | 答案准确率                  |
| Law      | CaseHOLD，法律案例 holding 选择，5,314 题              | 答案准确率                  |
| Finance  | FinQA，财报数值问答，1,147 题                          | 数值答案准确率              |
| IF       | IFBench，300 个可验证指令 prompts                      | Prompt-level loose accuracy |
| Tool Use | $\tau^2$-Telecom，电信工具交互，114 个 base-split 任务 | 任务成功率                  |

前四项报告 avg@3，即按三次采样表现平均，不是从三次里挑一次成功的 pass@3；Tool Use 报告 pass@1。评测采样 temperature 1，top-p 1，关闭 top-k。非工具问答与 IFBench 的 completion 上限分别为 26,624 和 32,768 tokens（§4.1、Appendix A.5）。

工具评测使用官方 v1.0.1 harness，32,768-token context、200-step 上限，用户模拟器为 temperature 0 的 `gpt-4.1-2025-04-14`，空响应与执行/context-limit 错误记失败。

Figure 1 的初始化分析和两域实验使用 IFEval，相关结果采用 prompt-level strict accuracy；主表使用 IFBench loose accuracy。二者不可直接当作同一列比较（Table 8、Table 11）。

## 主结果：收益来自哪里

以下原始指标与 Norm. 均按 Table 1 记录。Domain Teacher 一行是各列分别采用该领域专家的成绩，不是一个同时获得这些分数的统一教师。

### Qwen3-4B

| 方法                    | MedQA | CaseHOLD | FinQA | IFBench | $\tau^2$-Telecom | Norm. |
| ----------------------- | ----: | -------: | ----: | ------: | ---------------: | ----: |
| Reference：Qwen3-4B-OT3 |  69.5 |     61.8 |  59.1 |    24.9 |              5.3 |   0.0 |
| Domain Teacher          |  80.7 |     73.4 |  75.0 |    59.4 |             66.7 | 100.0 |
| MOPD                    |  69.1 |     64.7 |  74.3 |    56.9 |              7.9 |  42.8 |
| SFT Warm-up + MOPD      |  73.9 |     68.2 |  71.1 |    53.4 |             31.6 |  59.0 |
| Uniform Merge + MOPD    |  72.7 |     67.6 |  74.7 |    53.6 |             11.4 |  53.8 |
| IM-MOPD                 |  76.6 |     70.9 |  74.0 |    57.8 |             69.3 |  86.9 |

直接 MOPD 在 Finance 和 IF 上已取得大部分教师增益，Medical 甚至略低于 reference，Tool Use 只从 5.3 到 7.9。均匀合并改善了总体恢复，却没有解决 Tool Use。

IM-MOPD 相对 Uniform Merge 的 Norm. 从 53.8 升至 86.9，增加 **33.1 个归一化百分点**。它不表示所有 benchmark 的平均 accuracy 上升 33.1 点；最大原始变化是 Tool Use 的 11.4 到 69.3。FinQA 则由 74.7 略降至 74.0，说明能力恢复不是逐列严格支配。

### Qwen3-1.7B

| 方法                      | MedQA | CaseHOLD | FinQA | IFBench | $\tau^2$-Telecom | Norm. |
| ------------------------- | ----: | -------: | ----: | ------: | ---------------: | ----: |
| Reference：Qwen3-1.7B-OT3 |  44.1 |     44.6 |  42.4 |    18.4 |              7.9 |   0.0 |
| Domain Teacher            |  58.5 |     69.6 |  62.4 |    46.4 |             28.1 | 100.0 |
| MOPD                      |  49.0 |     52.2 |  59.1 |    36.8 |              0.0 |  34.9 |
| SFT Warm-up + MOPD        |  52.7 |     56.6 |  54.8 |    32.6 |             10.5 |  46.5 |
| Uniform Merge + MOPD      |  51.2 |     59.9 |  58.6 |    34.6 |              4.4 |  46.3 |
| IM-MOPD                   |  53.7 |     59.4 |  55.9 |    42.9 |             28.1 |  76.0 |

Norm. 相对 Uniform Merge 提升 29.7 点，相对 warm-up 提升 29.5 点。Tool Use 恢复到教师水平，IF 也明显提高，但 Law 略低于 Uniform Merge，Finance 低于直接 MOPD 和 Uniform Merge。

IM-MOPD 在两种尺寸上均优于所比较的 SFT warm-up 的全部五列。这个结论限于当前 warm-up 配方，不能解释为任何混合 SFT 准备都不如合并。

## 消融：累计系数相同，为什么时机仍然重要

### 将同一组最终系数全部前置或后置

作者取 IM-MOPD 实际产生的累计 task-vector 系数，保持 MOPD 更新数相同，将它们全部放到初始化，或全部加到已经训练完成的 Base MOPD 学生上。4B 结果为（Table 2）：

| 合并时机          | MedQA | CaseHOLD | FinQA | IFBench | $\tau^2$ | Norm. |
| ----------------- | ----: | -------: | ----: | ------: | -------: | ----: |
| 不合并：Base MOPD |  69.1 |     64.7 |  74.3 |    56.9 |      7.9 |  42.8 |
| 全部在初始化      |  76.3 |     70.8 |  74.0 |    52.4 |     57.0 |  79.0 |
| 全部在 MOPD 之后  |  68.1 |     71.0 |  59.6 |    38.2 |     10.5 |  23.3 |
| 交错加入：IM-MOPD |  76.6 |     70.9 |  74.0 |    57.8 |     69.3 |  86.9 |

最终直接注入的参数增量相同，但训练路径不同。后置合并的结果出现明显能力回退，且没有后续蒸馏消化这些变化；前置合并允许学习，却仍低于分次修正。

这支持“时序与后续学习共同重要”，但不能把结果解释成已经找到全局最优调度。对照中使用的最终系数来自 IM-MOPD 事后结果，本身也不是无需运行训练就可预知的配方。

### 超参数敏感性

4B 固定 $\gamma=0.6$，当 $\delta=0.2,0.3,0.4$ 时，Norm. 分别为 80.3、86.9、86.1。1.7B 的结果更敏感（Table 3、Table 10）：

| $\gamma$ | $\delta=0.1$ | $\delta=0.2$ | $\delta=0.3$ |
| -------- | -----------: | -----------: | -----------: |
| 0.4      |         62.2 |         57.2 |         76.0 |
| 0.6      |         57.7 |         65.0 |         69.8 |

不能据此说“修正越大越好”或“门槛越高越好”。门槛改变选中的领域及次数，步幅改变每次扰动，两者共同改变后续学习轨迹。

### 两域实验没有证明无搜索方法必然胜过所有固定 Merge

Medical 与 IF 的两域实验里，固定系数 $(0.4,0.6)$ 的初始 Norm. 最高，为 86.7，训练后为 91.3；初始只有 68.9 的 $(0.8,0.2)$，训练后达到 105.4。这再次证明即时排序会反转。

但该表中 IM-MOPD 的最终 Norm. 为 105.2，略低于最好的固定候选 105.4。合理的结论是它减少了事先完整系数搜索的需要，在此小实验中接近最好候选，而非保证超越任何固定初始化（Table 11）。

## 机制诊断：学生前缀能否让教师继续成功

### 怎样测试前缀兼容性

论文使用 student-prefix teacher-continuation 诊断：学生先生成一段前缀，再让对应教师接着生成，评估最终任务是否完成。改变学生负责的比例，可以观察教师在学生状态上是否仍有用（§5、Figure 5）。

- Medical：对每个已记录学生回答，设全长为 $L$，在前 $\lfloor aL\rfloor$ 个 tokens 后切给教师，$a\in\{0.1,0.3,0.5,0.7\}$。
- Tool Use：使用学生单独评测的平均 assistant turn 长度 $\bar L$，每 turn 预算为 $N=\operatorname{round}(a\bar L)$，到预算后由教师续写。

工具环境中，改动一个 turn 会影响之后的用户和工具反馈，不能再按原轨迹逐样本长度精确匹配，所以使用平均 turn 预算。较短 turn 可能完全由学生生成，图中的比例也不是每条对话实际严格由学生承担的 token 比例（Appendix B.3）。

0% 是教师独立表现，100% 是学生独立表现；中间点是混合生成诊断，不是完整学生 checkpoint 的准确率。

### 观察与可支持的解释

在 Medical 和 Tool Use 中，SFT warm-up 学生即使只提供较短前缀，也可能明显降低教师后续成功率；合并产生的学生前缀更容易被教师继续利用。某些比较中，合并学生独立成绩相近或更低，这个前缀优势仍存在。

这支持“学生的即时任务分数不能完整刻画其与教师的兼容性”。但它仍是诊断证据：接管生成后的任务成功率不等同于训练时每个 token 的 log-probability 监督质量，也没有隔离具体哪些参数、格式或推理习惯造成差异。因此不能把它提升为“合并必然修复教师反馈”的普遍机制定理。

## 局限、成本与复现边界

### 需要同源、可访问权重的教师

MOPD 本身可以从教师输出分布获取信号，IM-MOPD 的参数修正则必须读取教师和 reference 的对应权重。不同架构、不同参数对齐、只有远程接口的教师，不能直接套用该算法。

任务向量包含领域知识，也包含格式、风格、训练配方和其他变化，不能自动视作互不干扰的纯能力模块。

### “不加 SFT”与“没有额外成本”不同

方法省去 warm-up 阶段，但仍有三次周期性验证、任务向量存取与参数修改成本，也仍需蒸馏 prompts 和教师前向计算。论文没有给出完整 wall-clock、峰值显存或与大范围静态搜索严格等算力的对照。

阈值、步幅和检查间隔仍是超参数；4B 与 1.7B 采用不同阈值。小范围调参比连续高维系数搜索简化，但不是完全免调参。

### 工具评测存在明确的上游训练重叠风险

教师训练时移除了指定 telecom task-ID overlap，但最终 114 个 base-split 评测任务中仍有 20 个被标记可能存在上游训练重叠。论文明确说明，其数据过滤不能排除 reference 或其他上游训练中的暴露（Appendix A.5）。

因此，Tool Use 的大幅改善不能只解释为在完全未见工具任务上的泛化。周期性 Tool Use 验证又只说明从完整数据集采样 128 个任务，附录未在相邻描述中完整列出其与最终评测的拆分关系，这也是复现时需要核实的条件。

### 两个模型尺寸、五个领域尚不足以建立普适结论

主实验限于 Qwen3 的 1.7B 与 4B，以及固定领域组合；没有展示更大模型、不同架构和持续新增教师时的效果。主表也未给出多随机种子误差条；avg@3 可以降低生成采样噪声，但不等于训练三次的方差估计。

Norm. 等权平均的是相对能力增量，不是统一物理量。评估时必须同时保留原始逐域分数，检查收益是否集中在某个领域、已有领域是否回退，以及教师-reference 的小分母是否放大噪声。

### 直接改权重后的优化器语义仍需结合实现确认

Algorithm 1 明确写出了参数加法，但附录没有展开干预时 optimizer moments 是否保留、重置，以及 rollout 权重与 trainer 权重如何同步。这些是工程复现问题，不能从数学伪代码自动补全。也不能在干预后仍把旧 rollout 当作无分布变化的数据继续使用而不检查训练协议。

## 关键结论

IM-MOPD 将参数空间的任务向量注入与行为分布上的蒸馏交错执行。其有价值的观察是：**一个学生是否容易继续向教师学习，不能只由它当前答题分数判断；相同的直接参数增量，在不同时间加入也可能带来不同结果。**

在所测五域设置中，它改善了落后领域的恢复，并以同样 100 次蒸馏更新超过均匀合并和指定的 SFT warm-up baseline。它没有取消领域权衡、验证成本或同源权重要求，也没有证明所有固定合并配方都更差。

## 相关知识链接

- [[training/post-training/on-policy-kd|On-policy KD]]
- [[training/post-training/model-merging|Model Merging]]
- [[training/post-training/knowledge-distillation|Knowledge Distillation]]
- [[training/post-training/grpo|GRPO]]
- [[fundamentals/information-theory/kl-divergence|KL Divergence]]
- [[sources/papers/2026-mopd-multi-teacher-on-policy-distillation-for-capability-integration|MOPD]]
