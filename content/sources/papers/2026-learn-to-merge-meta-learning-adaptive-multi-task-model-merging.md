---
title: "Learn to Merge: Meta-Learning for Adaptive Multi-Task Model Merging"
created: 2026-09-23
published: 2026-09-23
modified: 2026-09-23
type: source
status: processed
source_type: paper
area: sources
tags:
  - source
  - paper
  - model-merging
  - task-vector
  - meta-learning
  - multi-task-learning
  - adapter
aliases:
  - MetaMerging
publisher: "Proceedings of the 43rd International Conference on Machine Learning, PMLR 306, 2026"
source_url: https://proceedings.mlr.press/v306/
paper_date: "2026-07"
---

# Learn to Merge: Meta-Learning for Adaptive Multi-Task Model Merging

## 基本信息

- 标题：[Learn to Merge: Meta-Learning for Adaptive Multi-Task Model Merging](https://proceedings.mlr.press/v306/)
- 作者：Jun Chen、Qin Zhang、Weizhi Zhang、Xiao Luo、Philip S. Yu、Ziyue Qiao。
- 发表：[ICML 2026](https://icml.cc/Conferences/2026)，PMLR 306。
- 日期：2026-07
- 代码/数据：[MetaMerging](https://github.com/cjcj46262/MetaMerging)。
- 相关主题：[[training/post-training/model-merging|Model Merging]]、[[training/post-training/knowledge-distillation|Knowledge Distillation]]、[[training/post-training/multi-task-learning|Multi-Task Learning]]
## 研究对象与实验范围

- 研究对象：任务向量合并系数与合并后任务适配器训练的协同优化。

## 研究问题

### 静态最优的 unified model 不一定是适配后最优的 unified model

设共享预训练 backbone 为 $\theta_{\mathrm{pre}}$，第 $k$ 个任务的 fine-tuned model 为 $\theta_k$，task vector 为：

$$
V_k=\theta_k-\theta_{\mathrm{pre}}.
$$

MetaMerging 用系数 $\lambda_k$ 构造 unified model：

$$
\theta_{\mathrm{uni}}(\lambda)
=\theta_{\mathrm{pre}}+\sum_{k=1}^{K}\lambda_kV_k.
$$

传统静态 merge 直接优化合并模型的 task loss：

$$
F(\lambda)=\sum_k L_k^{\mathrm{task}}(\theta_{\mathrm{uni}}(\lambda)).
$$

但本文的实际部署链路是：

~~~
task-specific fine-tuned models
  -> merge shared backbone
  -> attach / train task-specific adapters
  -> evaluate each adapted task model
~~~

因此，更贴近实际的目标是：先在每个任务上训练 adapter，再评价 adapter 更新后的模型：

$$
G(\lambda)=\sum_k\min_{u\in\mathcal U_k}
L_k^{\mathrm{task}}(\theta_{\mathrm{uni}}(\lambda)+u),
$$

其中 $\mathcal U_k$ 是 task-specific adapter 能够进行的受限更新子空间。

如果 unified model 与任务最优点之间的误差可以被 adapter 修复，它不必在合并时被完全消除；如果误差落在 adapter 无法访问的方向上，它才是合并初始化真正需要解决的部分。因此，最小化 $F$ 和最小化 $G$ 通常不是同一个问题。

### 为什么需要 meta-learning

合并系数 $\lambda$ 决定 unified representation，adapter 参数又依赖这个 representation。adapter 训练效果反过来可以作为评价 $\lambda$ 的信号，形成一个 bilevel optimization：

~~~
outer variable: merging coefficients lambda
  -> construct unified model
  -> inner loop trains task-specific adapters
  -> outer loop evaluates adapted models
  -> meta-gradient updates lambda
~~~

论文要回答的是：能否不直接寻找一个静态表现最好的 merge，而是学习一个更容易被后续 adapter 适配的 unified initialization？

## 核心主张

### 1. Merge coefficients 应面向后续 adaptation 学习

统一模型的质量不应只通过“当前没有 adapter 时的 accuracy”判断。一个合并模型即使初始 accuracy 略低，只要保留了更好的 representation 和 adaptation potential，经过轻量 adapter training 后可能达到更高的最终性能。

### 2. MetaMerging 将 adapter training 纳入 merge coefficient 优化

MetaMerging 在 inner loop 快速模拟 task-specific adapter training，在 outer loop 用 held-out meta-test batch 评价 adapter 更新后的 representation alignment，再更新 $\lambda$。这样 merging coefficient 不再依赖人工设置的平均值，也不只围绕未适配模型优化。

### 3. 最好的 initialization 不一定是合并后立即分数最高的模型

论文观察到，MetaMerging 得到的模型在没有 adapter 时不一定优于所有 baseline，但在 adapter training 后可以取得更好的 multi-task performance。这说明 model merging 中需要区分 immediate unified-model quality、post-adaptation quality、adapter sample efficiency 和 worst-task performance。

## 方法与机制

### Unified backbone 与 task-specific adapters

作者保留 task-specific prediction head，并在共享 unified backbone 上连接 adapter $A_{\theta_{\mathrm{ada},k}}$。给定输入 batch $X$，adapter 的训练目标是使 adapted unified representation 接近原 task-specific fine-tuned model 的 representation：

$$
L_k(X;\lambda,\theta_{\mathrm{ada},k})
=\left\|f_{\theta_k}(X)-
A_{\theta_{\mathrm{ada},k}}\left(f_{\theta_{\mathrm{uni}}(\lambda)}(X)\right)\right\|_2^2.
$$

论文使用轻量残差式 adapter：

$$
A_\phi(h)=h-\operatorname{Linear}(\operatorname{ReLU}(\operatorname{Linear}(h))).
$$

主视觉实验的 adapter rank 为 64，token feature dimension 为 768。adapter 结构本身不是论文的主要贡献，而是用来模拟 merge 后的 task adaptation。

附录 E 笼统写 feature dimension 为 768，但不同 backbone 的维度配置仍需以实现核查；不能把它理解为所有模型层宽度均为 768。这里的 rank 指两层适配器的 bottleneck hidden dimension，不等于对整个模型做低秩权重分解。任务 head 不合并，推理时需要选择对应任务的 head 和 adapter。

### Inner loop：快速模拟 adapter training

在每一个 meta-update cycle 中，对每个 task $k$：

1. 从 task training set $\mathcal D_k$ 采样 batch $X$；
2. 计算 representation alignment loss $L_k$；
3. 用 inner learning rate $\alpha$ 更新 adapter：

$$
\theta'_{\mathrm{ada},k}
=\theta_{\mathrm{ada},k}
-\alpha\nabla_{\theta_{\mathrm{ada},k}}L_k(X;\lambda,\theta_{\mathrm{ada},k}).
$$

主实验通常使用一次 inner update；附录分析了 0、1、5、10 次 inner steps 的影响。Inner loop 不追求 adapter 完整收敛，而是模拟“这个 unified initialization 是否能被快速适配”。

### Outer loop：根据适配后的表现更新 merging coefficients

随后从 task meta-test / validation set $\mathcal D'_k$ 采样 $X'$，将已经 inner-updated 的 adapter 接在 unified backbone 上，重新计算：

$$
L'_k=L_k(X';\lambda,\theta'_{\mathrm{ada},k}).
$$

汇总所有任务的 meta-test loss 后，使用 meta learning rate $\beta$ 更新 merging coefficients：

$$
\lambda\leftarrow\lambda-\beta\nabla_\lambda\sum_kL'_k.
$$

梯度需要穿过 unified model、inner adapter update 和 outer loss，因此它直接回答：调整 $\lambda$ 后，adapter 是否会更容易在 held-out samples 上恢复 task-specific representation。

精确梯度含两条路径。记 adapter 参数为 $\phi_k$，则：

$$
\frac{dL'_k}{d\lambda}
=\frac{\partial L'_k}{\partial\lambda}
+\left(\frac{\partial\phi'_k}{\partial\lambda}\right)^\top
\nabla_{\phi'_k}L'_k,\qquad
\frac{\partial\phi'_k}{\partial\lambda}
=-\alpha\frac{\partial}{\partial\lambda}\nabla_{\phi_k}L_k.
$$

第二条路径涉及混合二阶导数。正文 §3.2 明确说明，GPT-2 和 ViT-L/14 使用 First-Order MAML（FOMAML，一阶 MAML）式近似，忽略二阶项以降低开销。因此不能声称所有实验都完整反传穿过 inner update。无论是否使用近似，outer loss 都是在临时适配后的模型上计算。

每次 episode 的 inner update 从 adapter 初始化出发；得到最终合并系数后，再从初始化正式训练 adapter。临时 adapter 与最后部署的 adapter 需要区分。该方法用无标签输入对齐冻结任务模型的特征，实际 outer objective 是特征距离，不是直接对分类准确率求导。

### 完整训练流程

~~~
1. 由 pretrained model 和 task vectors 构造 unified backbone
2. 为每个 task 初始化 adapter
3. 采样 task train batch，做少量 adapter inner updates
4. 采样 task meta-test batch，计算适配后 loss
5. 通过 meta-gradient 更新 merging coefficients
6. 重复 meta cycles，得到 learned lambda
7. 固定 unified backbone，分别完整训练各 task adapter
8. 评估各 task 的 adapted model
~~~

## 理论解释

若任务最优参数为 $\theta_k^*$，定义 unified model 与任务最优点的误差：

$$
e_k(\lambda)=\theta_{\mathrm{uni}}(\lambda)-\theta_k^*.
$$

若 adapter 只能在子空间 $\mathcal U_k$ 中移动，且任务损失在局部可以用 Hessian $H_k$ 二阶近似，则适配后残余误差近似只保留正交于 $\mathcal U_k$ 的部分：

$$
\min_{u\in\mathcal U_k}L_k(\theta_{\mathrm{uni}}+u)
\approx L_k(\theta_k^*)+
\frac12\left\|(I-P_k)e_k(\lambda)\right\|_{H_k}^2.
$$

其中 $P_k$ 按附录 A 是 $H_k$ 内积下的正交投影，不是一般的欧氏投影。静态目标惩罚全部误差，适配后的局部目标只保留适配器不能消除的分量；两个目标因此可以偏好不同的合并初始化。仅有一个误差落在子空间外，还不足以保证任意实例的最优系数必然不同，特殊对齐时两者仍可能重合。

论文还在简化线性 adapter 场景中使用 online gradient descent 的 regret bound 解释 adapter 学习速度。直观上，若 unified representation 使任务 adapter 的最优参数范数更小，有限步梯度下降的 regret 也更低。这个理论结果是在简化设置下得到的，作用是解释 meta objective 的方向，不是对深度网络训练速度的普适定量保证。

Regret（遗憾值）衡量累计训练损失相对事后最优固定 adapter 的差距。在零初始化、凸损失和有界梯度等条件下，论文使用：

$$
R_k\le\frac{\|\phi_k^*(\lambda)\|^2}{2\eta}+\eta G_k^2m.
$$

$\eta$ 为学习率，$G_k$ 为梯度界，$m$ 为步数。减小右侧第一项可以改善该上界，但更低的累计 regret 不直接保证最后一步 accuracy 更高。

Theorem 3.2 的附录证明进一步简化为线性 adapter 和单步优化，并把表征 $H$ 直接作为可更新量。实际 $H$ 只能通过少量合并系数改变，实际 adapter 还含 Rectified Linear Unit（ReLU，修正线性单元），损失对全体参数并非一般凸函数。此外证明中的一步更新范数式省略了学习率因子，相关半正定与步长条件也需要进一步审查。因此，本报告将其视为简化机制分析，结论强度主要依据实验证据。

## 实验与证据

### Vision 实验

主视觉实验使用 Contrastive Language-Image Pre-training（CLIP，对比式图文预训练）的 Vision Transformer（ViT，视觉 Transformer）编码器 ViT-B/32 和 ViT-L/14。八个任务包括 SUN397（场景）、Cars（汽车）、RESISC45（遥感场景）、EuroSAT（卫星图像）、Street View House Numbers（SVHN，街景数字）、German Traffic Sign Recognition Benchmark（GTSRB，交通标志）、Modified National Institute of Standards and Technology database（MNIST，手写数字）、Describable Textures Dataset（DTD，纹理）。Ties-Merging 使用参数筛选和符号冲突处理，AdaMerging 自适应学习合并系数。

作者同时报告 w/o adapters 和 w/ adapters 两种状态。ViT-B/32 的平均 accuracy 为：

| 方法 | 无 adapters | 加 adapters |
|---|---:|---:|
| Weight Averaging | 65.8 | 80.0 |
| Task Arithmetic | 70.1 | 80.9 |
| Ties-Merging | 73.6 | 83.1 |
| AdaMerging | 80.1 | 86.1 |
| MetaMerging | 67.8 | 87.2 |

MetaMerging 的 unified model 初始平均分只有 67.8，低于 AdaMerging 的 80.1，但训练 adapters 后达到 87.2，高于 AdaMerging 的 86.1。这直接支持论文的核心判断：最好的 initialization 不一定是未经适配时分数最高的模型。

在 ViT-L/14 上，MetaMerging 的最终平均 accuracy 为 93.4，高于 AdaMerging 的 92.3、Ties-Merging 的 89.3、Task Arithmetic 的 89.0 和 Weight Averaging 的 85.9。它的无 adapter 分数为 84.4，依然低于 AdaMerging 的 90.8，说明这种现象并非只出现在小模型。

这里比较的是附录 Table 6/7 给各基线添加相同适配器后的结果；AdaMerging 原始无 adapter 分数分别是 80.1 和 90.8。同 adapter 对照比直接拿 MetaMerging 与无训练的平均权重比较更能隔离初始化质量。

平均更高并非所有任务更好：ViT-B/32 的 DTD 上，MetaMerging 为 64.9，AdaMerging 加 adapter 为 73.6，前者低 8.7 个百分点。因此不能只根据平均值宣称任务全面更均衡。其 67.8→87.2 是 +19.4 个百分点；Figure 1 的 +28.6% 是相对增幅，两者单位不同。

### NLP 实验

自然语言处理（Natural Language Processing，NLP）实验使用 Generative Pre-trained Transformer 2（GPT-2）及 General Language Understanding Evaluation（GLUE，通用语言理解评测）的七项分类任务：Corpus of Linguistic Acceptability（CoLA，语言可接受性）、Stanford Sentiment Treebank 二分类任务（SST-2，情感）、Microsoft Research Paraphrase Corpus（MRPC，释义）、Quora Question Pairs（QQP，问题相似性）、Multi-Genre Natural Language Inference（MNLI，多领域推断）、Question-answering Natural Language Inference（QNLI，问答蕴含）和 Recognizing Textual Entailment（RTE，文本蕴含）。论文统一报告 accuracy，不能直接与采用其他官方任务指标的 GLUE 综合分比较。

正文 Table 3 的最终平均准确率是 MetaMerging 76.9、Task Arithmetic 70.0、Ties-Merging 70.0、RegMean 68.8、AdaMerging 59.6；独立任务模型平均为 82.0。下面则是附录 Table 10 的 inner-step 消融，不能将其中的 76.4 替换正文主结果：

| 方法 | Average accuracy |
|---|---:|
| Pretrained model | 44.5 |
| Fine-tuned model | 82.0 |
| MetaMerging，0 inner steps | 74.9 |
| MetaMerging，1 inner step | 76.4 |
| MetaMerging，5 inner steps | 75.7 |
| MetaMerging，10 inner steps | 75.8 |

一次 inner update 在这个设置下最好，但 5 和 10 steps 并没有继续提高。这说明更长的 inner simulation 不必然更好；meta objective、计算成本和 inner-loop horizon 需要一起调。

### 超参数与计算成本

ViT-B/32 上，inner step size $\alpha$ 从 1、0.1、0.01 到 0.001 时，平均 accuracy 为 86.74、87.15、86.85、86.85；meta step size $\beta$ 为 1、0.1、0.01、0.001 时，平均 accuracy 为 85.74、87.13、87.15、86.68。过大或过小的 step size 都可能偏离较优区间。

正文 Table 4 报告单张 RTX 4090 上 MetaMerging 用时 1 小时 23 分钟，AdaMerging 2 小时 5 分钟，Surgery 46 分钟，多任务学习 15 小时 53 分钟。附录超参扫描还有不同计时结果，不应合并成一个固定耗时。适配器训练按一 epoch 计，而多任务学习按十 epochs 计；已存在的任务模型训练成本也不在这个比较中，因此这是论文协议下的增量运行成本。

Table 4 报告单 adapter 有 99,136 参数；八个约为 0.793M，最终仍有任务专属 head/adapter。正文另称八任务模型共 89.05M 参数，而 Table 4 的 ViT-B/32 为 92.185M；具体是否排除了某些头或组件未说清，部署估算需核查口径。

Table 5 也报告了较大容量基线超过 MetaMerging：例如 FREE-Merging 89.7、WEMoE 89.4、EMR-Merging 88.7，而 MetaMerging 为 87.2。论文将这些列为带额外专家、路由或任务专属参数的比较。应比较实际存储、活跃参数和延迟，而不能仅据平均 accuracy 宣称 MetaMerging 全面领先。

### Adapter training 的观察

Figure 5 与 Figure 6 的 loss curves 显示，MetaMerging 的 adapter training 在多个任务上下降更快或达到更低的最终 loss。作者据此支持“更适配的统一 representation 能降低后续 adapter 学习负担”。但 adapter rank、inner step、完整 adapter epoch 数和 task-specific data 都会影响这个结论，不能只把最终 accuracy 归因给系数学习。

## 对模型合并实践的启发

### Merge quality 要绑定最终使用流程

如果部署流程是：

~~~
merge checkpoints -> attach adapter -> adapt to task -> inference
~~~

那么 merge coefficients 应围绕适配后的效果选择。只看 merge 后 zero-shot 或无 adapter accuracy，可能优化了错误目标。

### Meta-train / meta-test split 是关键

MetaMerging 使用 task train data 做 inner adapter update，用独立 meta-test / validation data 计算 outer loss，最后使用 test split 做最终评估。若 inner 与 outer 使用同一数据，系数可能只对当前 batch 的 adapter update 过拟合，无法代表真实 adaptation。

### 记录 adapter regime

Meta-learned coefficients 依赖 adapter architecture、rank、inner learning rate、inner steps、feature alignment loss 和 task data distribution。更换 adapter 结构或部署训练预算后，应重新评估 $\lambda$，不能把某次 MetaMerging 得到的系数当作通用 merge recipe。

## 局限与疑问

- 主实验是视觉和文本分类，尚未覆盖自回归生成、推理、工具使用或 Agent 轨迹。
- Unified model 保留 task-specific heads 和 adapters；结果不等于一个完全无 task metadata 的单模型 merge。
- Meta-learning 依赖各任务 training / meta-test inputs，即使不使用 labels，也需要任务数据可得。
- Inner-loop 只模拟少量 adapter updates，和实际长时间 adapter training 可能存在 mismatch。
- 系数通过 meta-validation loss 优化，任务数量、数据划分和 adapter rank 改变后需要重新调参。
- 论文展示了平均指标提升，但不同任务之间仍有波动，应同时关注 worst-task、任务平衡和适配后成本。
- 主表未报告多随机种子的误差范围，视觉同 adapter 对照中约 1.1 个百分点的平均优势仍需复现实验确认稳定性。
- 附录 E 使用训练集做 inner、验证集做 meta-test，Algorithm 1 的最终 adapter 训练也使用训练集；但 §4.5 又写 adapter 在 validation dataset 上训练一 epoch。这一分割表述差异需要结合代码核实，尤其不能将 meta-test 与最终测试集混淆。

## 研究判断

这篇论文最重要的结论是：**Model merging 的优化目标应该包含模型合并之后的使用方式。** 当 unified model 只是后续 adapter training 的 initialization 时，静态 merge accuracy 不能完整代表它的质量；更合理的 merge coefficient 应使 adapter 能够快速、稳定地恢复各个任务的能力。

MetaMerging 把这个想法落实成一个清晰的 bilevel optimization：inner loop 模拟 task adapter learning，outer loop 用 held-out data 优化 merging coefficients。它的贡献是把后续适配过程纳入合并目标，使合并系数服务于适配后的任务表现。

当前证据仍限定在共享 backbone、任务专属 head、轻量 adapter 和分类。迁移到大语言模型时，需要重新确定适配器结构、更新预算、特征对齐或任务损失，以及外层验证数据是否代表真实多轮交互。论文提供的是面向适配过程选择初始化的方法，尚未证明开放式生成能力整合的有效性。

## 相关知识链接

- [[training/post-training/model-merging|Model Merging]]
- [[training/post-training/knowledge-distillation|Knowledge Distillation]]
- [[training/post-training/on-policy-kd|On-policy KD]]
- [[training/post-training/multi-task-learning|Multi-Task Learning]]
