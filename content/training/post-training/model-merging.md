---
title: Model Merging
created: 2026-09-23
published: 2026-09-23
modified: 2026-10-29
type: topic
status: growing
area: training
tags:
  - post-training
  - model-merging
  - task-vector
---

Model Merging（模型合并）通过组合多个模型的参数整合能力。它与推理时运行多个模型的 ensemble（集成）不同，也与重新联合训练所有任务不同。常见任务向量方法从共享的预训练初始化出发，组合各任务微调产生的参数变化。

## 任务向量与合并

对共享初始化 $\theta_0$ 的任务模型 $\theta_t$，定义：

$$
\tau_t=\theta_t-\theta_0,\qquad
\theta_m=\theta_0+\sum_t\lambda_t\tau_t.
$$

$\lambda_t$ 是合并系数。实际算法还可处理参数重要性、更新符号冲突和向量子空间。任务向量记录的是全部训练变化，包含任务知识及训练配方的影响，不能自动解释为相互独立的能力模块。

## 三种能力整合方式

| 方式 | 主要输入 | 能力如何整合 | 主要约束 |
|---|---|---|---|
| 联合多任务训练 | 多任务样本 | 同一个模型接收各任务梯度 | 数据可得性、采样和梯度干扰 |
| 多教师蒸馏 | Teacher 输出或分布 | Student 学习多个 teacher 行为 | 输入覆盖、teacher 调用成本与兼容性 |
| 参数合并 | 多个 checkpoints | 组合权重或任务向量 | 参数对齐、共享初始化及任务干扰 |

蒸馏与参数合并可以组合：先调整待合并模型的行为和参数性质，再执行合并。它们不必是互斥选项。

## Fisher 加权与局部行为距离

参数欧氏距离不直接等于输出行为距离。设 $F_d$ 是领域教师 $\theta_d$ 的 Fisher 信息矩阵，局部 KL（Kullback-Leibler divergence，KL 散度）在学生接近教师时满足：

$$
D_{\mathrm{KL}}(\pi_\theta\|\pi_{\theta_d})
\approx\frac12(\theta-\theta_d)^\top F_d(\theta-\theta_d).
$$

Fisher 来自 log probability 参数梯度的外积期望，刻画哪些参数方向更容易改变预测分布。若多个同源教师足够接近，多教师蒸馏目标可近似为这些二次距离的加权和，得到 Fisher merge：

$$
\theta_m=
\left(\sum_dp_dF_d\right)^{-1}
\sum_dp_dF_d\theta_d.
$$

该表达式要求相应矩阵可逆；实际可使用阻尼或其他近似。它是局部二次 surrogate 的最优解，不保证原始非线性蒸馏或下游任务准确率全局最优。完整 Fisher 具有参数方向间耦合，也不同于逐参数对角重要性。

[[sources/papers/2026-fisher-geometric-multi-teacher-on-policy-distillation|A Fisher-Geometric View of MOPD]] 进一步将参数限制为 $\theta_0+T\alpha$，把 $P\times P$ 的 Fisher 问题降为 $K\times K$ 的 task-vector 投影问题，再用教师/reference log-ratio 的二阶矩估计局部残差。其实际 Minimax 合并加入了上界近似、最坏领域目标和正则化，不能与精确 Fisher 闭式解直接等同。

### 残差选择与分支写回

同一研究的 RGBM 方法，用原始残差求合并系数，再按 reward energy 归一化的残差选择后续训练领域。把归一化提前放入优化目标会改变相对领域权重；诊断量与优化目标不是可互换的两个写法。

选中的领域从同一合并 checkpoint 独立训练，再将缩小后的参数增量写回。没有训练某个领域自己的分支，只意味着不加入它的直接更新，不能保证其他分支写回后该领域不变。分支自身指标上升，也不能单独证明其增量适合加入多域共享模型。

在校准集上检查 log-ratio 线性拟合，可用作写回尺度的代理诊断，但低回归误差不是任务无退化保证；仍需逐域验证最终模型，并计入校准生成、评分和分支训练的成本。

## 合并前的兼容性

至少需要检查 architecture、参数对应关系、初始化来源，以及输入输出接口。对语言模型还应检查 tokenizer、词表与对话模板；形状相同不代表参数语义相同。

训练配方也属于兼容性信息：学习率、训练步数、优化器、weight decay 和训练目标都会影响任务向量。应同时记录全局及逐层范数、向量夹角、源模型任务指标与输出分布。

[[sources/papers/2025-distac-conditioning-task-vectors-via-distillation-for-robust-model-merging|DisTaC]] 在视觉和语言分类设置中显示，范数失配及低置信度训练条件会影响合并表现。它先收缩异常长的向量，再用原任务模型作为 teacher 恢复行为，并以参数正则约束新锚点附近的移动。由此可将训练流程组织为：

```text
检查源模型及训练配方
  -> 诊断尺度、方向和预测分布差异
  -> 必要时进行参数调理
  -> 合并模型
  -> 评估绝对能力、能力保留率和最终校准
```

这类调理是有条件的经验方法；不能由分类结果推出所有生成模型都应降低输出熵。

## 面向后续适配的 Merge

[[sources/papers/2026-learn-to-merge-meta-learning-adaptive-multi-task-model-merging|Learn to Merge: Meta-Learning for Adaptive Multi-Task Model Merging]] 进一步改变了 merge coefficient 的优化目标。若 unified model 后续还要训练 task-specific adapter，那么合并阶段不应只优化当前模型的静态 accuracy，而应模拟 adapter 的 inner update，再用 held-out data 的 post-adaptation loss 更新合并系数。

这形成一个双层流程：

~~~
lambda -> construct unified backbone
  -> inner loop trains task adapters
  -> outer loop evaluates adapted models
  -> meta-gradient updates lambda
~~~

因此，merge coefficient 依赖最终使用协议：adapter architecture、rank、inner learning rate、inner steps 和 feature alignment loss 变化后，原系数不一定仍然有效。Model merging 的评估也应同时报告无 adapter 表现、适配后表现、adapter sample efficiency 和 worst-task performance。

## 评估口径

### 合并时机与后续学习

参数合并还可以与后续训练交错。[[sources/papers/2026-iterative-merging-multi-teacher-on-policy-distillation|IM-MOPD]] 将固定的教师 task vector 定期加入正在蒸馏的学生，再继续在修正后的模型上采样和学习。这里的向量始终是“教师减共同 reference”，而不是“教师减当前学生”。

直接累计的合并系数只能描述参数干预量，不能完整描述最终模型：训练梯度依赖每次修正后的模型与采样轨迹。即使累计加入相同的 task vectors，全部放在初始化、训练中分次加入、训练结束后一次加入，也可能产生不同结果。

因此，目标若是训练后的表现，应在相同后续训练协议下比较候选，而非只按即时合并分数排序。验证集、训练预算、修正时机、optimizer state 和训练/采样模型同步，都属于这一协议的组成部分。

### 绝对能力与恢复率

先保留每个 source model 的单任务结果，再报告合并后的逐任务绝对指标。能力保留率可以定义为合并指标除以对应源模型指标，但需要固定分母，避免源模型变弱造成比例看似改善。

若使用 reference 到教师的增量作为分母，恢复率应明确写为 $(s_{\mathrm{student}}-s_{\mathrm{ref}})/(s_{\mathrm{teacher}}-s_{\mathrm{ref}})$。它与 student/teacher 比值不同，可以小于 0 或超过 1；分母很小时也可能放大评测波动。跨任务平均恢复率不等于平均准确率。

平均指标之外，应检查最差任务、通用能力回退、校准、生成多样性和运行成本。比较不同配方时，还要计入调理所用数据、teacher forward、超参数搜索和验证集预算。

## 常见失败模式

- 范数支配：一个任务更新过大，组合方向偏向该任务。
- 参数冲突：不同任务对共享参数需要不同更新。
- 表面兼容：形状相同但初始化或参数语义不同。
- 验证集过拟合：合并系数只在少量验证任务上有效。
- 指标误读：相对保留率上升，但最终绝对能力下降。

## 相关知识

- [[training/post-training/knowledge-distillation|Knowledge Distillation]]
- [[training/post-training/on-policy-kd|On-policy KD]]
- [[fundamentals/probability/calibration-uncertainty|Calibration and Uncertainty]]
- [[sources/papers/2025-distac-conditioning-task-vectors-via-distillation-for-robust-model-merging|DisTaC]]
- [[sources/papers/2026-learn-to-merge-meta-learning-adaptive-multi-task-model-merging|MetaMerging]]
- [[sources/papers/2026-fisher-geometric-multi-teacher-on-policy-distillation|Fisher-Geometric MOPD / RGBM]]
