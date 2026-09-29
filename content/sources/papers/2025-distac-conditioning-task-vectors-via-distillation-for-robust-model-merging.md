---
title: "DisTaC: Conditioning Task Vectors via Distillation for Robust Model Merging"
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
  - distillation
  - calibration
aliases:
  - DisTaC
source_url: https://arxiv.org/abs/2508.01148
paper_date: "2025-08"
paper_order: "01148"
---

# DisTaC: Conditioning Task Vectors via Distillation for Robust Model Merging

## 基本信息

- 标题：[DisTaC: Conditioning Task Vectors via Distillation for Robust Model Merging](https://arxiv.org/abs/2508.01148)
- 作者：Kotaro Yoshida、Yuji Naraki、Takafumi Horie、Ryotaro Shimizu、Ioannis Mitliagkas、Hiroki Naganuma，以所读 PDF 首页为准。
- 发表：arXiv:2508.01148；International Conference on Learning Representations（ICLR，国际学习表征会议）2026。
- 日期：2025-08
- 代码/数据：[katoro8989/DisTaC](https://github.com/katoro8989/DisTaC)。本报告分析论文，未执行代码复现。
- 相关主题：[[training/post-training/model-merging|Model Merging]]、[[training/post-training/knowledge-distillation|Knowledge Distillation]]、[[fundamentals/information-theory/label-smoothing|Label Smoothing]]
## 研究对象与实验范围

- 原文：[arXiv:2508.01148v3](https://arxiv.org/abs/2508.01148v3)，版本日期 2026-03-02，首次公开于 2025-08。

## 研究问题与基本定义

### Task vector 表示什么

多个任务模型共享预训练参数 $\theta_{\mathrm{pre}}$。第 $t$ 个任务经过微调后得到 $\theta_t$，其任务向量为：

$$
\tau_t=\theta_t-\theta_{\mathrm{pre}}.
$$

任务向量记录整个参数空间中的变化；它既包含任务适配，也可能包含优化器、正则化、训练时长和数据分布留下的影响，不能直接等同于纯粹的“能力向量”。

Task arithmetic（任务算术）的典型合并形式是：

$$
\theta_{\mathrm{merge}}=\theta_{\mathrm{pre}}+\sum_{t=1}^{M}\lambda_t\tau_t,
$$

其中 $M$ 是任务数，$\lambda_t$ 是合并系数。本文用 $M$ 表示任务数，避免与后文蒸馏温度 $T$ 混淆。其他方法还会加入参数筛选、符号冲突处理或奇异向量变换。

这一过程属于 multi-task learning（MTL，多任务学习）的一种能力整合路线：各任务独立训练，随后组合参数。它与联合训练全部任务、推理时集成多个模型、将多个 teacher 蒸馏为单个 student 都有区别。

### 为什么统一微调配方会掩盖问题

常见 benchmark 使用同一个初始化、相同学习率与训练步数，以及相同监督目标生成任务模型。这样得到的任务向量较容易比较。实际获得的 checkpoints 则可能使用不同学习率、训练时长、weight decay 和损失函数。

论文通过两组干预检查鲁棒性：一组只提高某个任务的学习率，另一组让任务模型使用标签平滑。实验重点是观察源模型单独表现与合并后表现是否脱钩：**单任务准确率接近，并不意味着合并适配性接近。**

## 两类失败机制

### 范数失配：大任务向量支配组合方向

主实验使用 Contrastive Language-Image Pre-training（CLIP，对比式图文预训练）的 Vision Transformer（ViT，视觉 Transformer）图像编码器。相同步数下，将学习率从 $10^{-5}$ 提高到 $10^{-4}$，会使任务向量范数增大约 5 至 7 倍；附录 E.1 显示这种差异遍布多个层和参数类型。

作者依次让八个任务中的一个使用较大学习率，其他七个保持原设置，再平均这些合并实验。它不是把所有任务同时扩大，也不是直接对一个已经训练好的向量乘 10。

对两个正交向量，令：

$$
\delta=\frac{\|\tau_1\|}{\|\tau_2\|},\qquad
\tau_1\perp\tau_2,\qquad \tau_m=\tau_1+\tau_2.
$$

则：

$$
\cos(\tau_m,\tau_2)=\frac{1}{\sqrt{1+\delta^2}},\qquad
\cos(\tau_m,\tau_1)=\frac{\delta}{\sqrt{1+\delta^2}}.
$$

例如 $\|\tau_2\|=5\|\tau_1\|$ 时，两个 cosine 分别约为 0.981 和 0.196。合并方向明显靠近大向量。

这个几何结果解释了风险，但不等于“小向量能力必然完全消失”的证明。在 Neural Tangent Kernel（NTK，神经切线核）相关的局部线性近似下：

$$
f(x,\theta_{\mathrm{pre}}+\tau_m)
\approx f(x,\theta_{\mathrm{pre}})+J_f(x)\tau_m.
$$

函数变化还取决于输入相关的 Jacobian $J_f$，以及向量的方向和幅度。参数夹角本身不足以定量预测任务准确率，真实向量也不严格正交。论文的跨方法实验为这个机制提供了经验支持。

### 低置信度：校准良好不代表容易合并

Label smoothing（LS，标签平滑）将硬标签替换为较平滑的目标分布。论文在强度 $\alpha=0.1$ 下观察到预测熵最多提高约三个数量级，而合并表现明显降低。

需要区分三个量：

- 准确率：预测类别是否正确；
- 置信度与预测熵：模型输出是否集中；
- 校准：例如预测置信度为 80% 的样本是否约有 80% 正确。

较低熵不保证更正确，高熵也不自动意味着校准良好。论文通过 reliability diagram（可靠性图）观察到，增大标签平滑强度时，源模型从过度自信走向校准较好，再走向置信不足；对应的合并准确率下降。

附录 C 用局部二阶展开分析带额外校准惩罚的目标。若标准交叉熵目标附近的梯度和 Hessian 为 $g_i,H_i$，额外正则项会改变任务向量，形成扰动 $\delta_i$。组合后，交叉熵差异包含形如 $g_i^\top\delta_j$ 的一阶项，其符号未必有利，因此可能增加损失。

这是在局部可微、Hessian 可逆和小扰动等假设下的解释，不是所有校准方法都会破坏所有合并方法的定理。标签平滑、Mixup 与 focal loss 同时改变表征、margin 和优化路径，实验并没有将预测熵隔离为唯一因果变量。

## 方法与训练目标

### 先选参数锚点

固定原任务模型作为 teacher：

$$
\theta_{T,t}=\theta_{\mathrm{pre}}+\tau_t.
$$

Student 从缩放后的参数初始化：

$$
\theta_{0,t}=\theta_{\mathrm{pre}}+\kappa_t\tau_t.
$$

$\kappa_t<1$ 是向预训练模型收缩，$\kappa_t>1$ 是沿微调方向外推。论文范数失配设置中，将最长向量缩放到其余七个向量范数的平均值：

$$
\kappa_{t^*}=\frac{\frac{1}{M-1}\sum_{j\ne t^*}\|\tau_j\|}{\|\tau_{t^*}\|}.
$$

这是一种具体实验配方，而不是对所有任务都应采用平均范数的最优性证明。

### 用原模型恢复缩放后的行为

对无标签输入 $x$，teacher 和 student 分别输出 logits $z_T(x)$ 与 $z_S(x)$。定义温度化概率：

$$
p_T(x)=\operatorname{softmax}(z_T(x)/T_T),\qquad
p_S(x)=\operatorname{softmax}(z_S(x)/T_S).
$$

采用 Kullback-Leibler divergence（KL 散度），Algorithm 1 的实际目标为：

$$
\mathcal L_t(\theta)=
\frac{T_TT_S}{|B_t|}\sum_{x\in B_t}D_{\mathrm{KL}}(p_T(x)\|p_S(x))
+\beta\|\theta-\theta_{0,t}\|_2^2.
$$

$B_t$ 是对应任务的无标签 mini-batch；$\beta$ 是参数约束强度。Teacher 固定，student 更新；不使用真实类别标签的 cross-entropy（CE，交叉熵）监督。

第一项要求 student 保留 teacher 的输出行为；第二项要求参数留在新锚点附近。第二项不是将 task vector norm 严格投影到常数，也不是约束参数到零的普通 weight decay。它通过限制整体参数位移，间接控制范数漂移。

最终重新计算 $\widetilde\tau_t=\theta_{S,t}-\theta_{\mathrm{pre}}$，再把它交给原 merging method。DisTaC 是合并前处理，可以与多种合并算子组合。

### 为什么更高 student temperature 会得到更自信的模型

Low Confidence 设置保持 $\kappa_t=1$，但取 $(T_T,T_S)=(1,10)$。训练中 student 输出被除以更大的温度，为匹配 teacher，模型需要扩大自身 logit 差。

如果温度化分布精确匹配，则任意两类别 $i,j$ 满足：

$$
z_{S,i}-z_{S,j}=\frac{T_S}{T_T}(z_{T,i}-z_{T,j}).
$$

例如 teacher 的二分类 logit gap 为 1，温度化匹配可能要求 student gap 为 10。部署时将温度恢复为 1，student 分布就会更加尖锐。这个推导解释了训练方向；实际模型受参数约束、有限数据和优化误差影响，不保证每个输入都恰好放大十倍。

普通 post-hoc temperature scaling（训练后温度校准）只改变输出，不改变任务向量。DisTaC 将这种行为变化写入参数，因此能够影响随后的权重合并。

### 两种实验配方

| 配置 | 初始化缩放 | Teacher / student 温度 | 作用 |
|---|---|---|---|
| Norm Mismatch | 缩小异常长向量 | 10 / 10 | 调整尺度后恢复任务行为 |
| Low Confidence | $\kappa_t=1$ | 1 / 10 | 保持任务准确率并提高置信度 |

两种操作可在统一目标中同时启用，但主表分别测试这两类失效条件，不能将其当作所有联合失配条件均已充分验证。

## 实验设置与指标

### 模型、任务和预算

主实验使用 ViT-B-32 和 ViT-L-14，分别代表基础规模、32 像素 patch 和大规模、14 像素 patch 的视觉编码器。文本编码器冻结，类别提示产生固定分类头，只更新图像编码器。

八个视觉任务包括 Cars（汽车分类）、Describable Textures Dataset（DTD，纹理分类）、EuroSAT（卫星图像分类）、German Traffic Sign Recognition Benchmark（GTSRB，交通标志识别）、Modified National Institute of Standards and Technology database（MNIST，手写数字）、RESISC45（45 类遥感场景）、SUN397（397 类场景）和 Street View House Numbers（SVHN，街景数字）。

源模型训练使用 2,000 updates、AdamW、weight decay 0.1、200 warmup steps、cosine schedule 和有效 batch size 128。常规学习率为 $10^{-5}$，范数失配时一个任务改为 $10^{-4}$。DisTaC 使用 500 steps、学习率 $10^{-5}$ 和 $\beta=0.5$。

对照包含 task arithmetic，以及论文命名的 TIES、Consensus TA、EMR-Merging、TSVM、Iso-CTS、WUDI-Merging。其中 TIES 指 Trim、Elect Sign、Merge 的冲突处理路线；TSVM 指 Task Singular Vector Merging（任务奇异向量合并）；其余缩写在本文保留算法名称，分别作为共识筛选、调制重缩放、各向同性处理及干扰处理的对照，不据名称补写未核实的全称。

合并系数按附录 D 的协议统一约束，并基于验证集搜索。因此，“蒸馏无需标签”不等于整个超参数选择流程完全无需标签，也不等于在每个任务独立调节合并系数后仍有同样幅度的收益。

### Absolute accuracy 与 normalized accuracy

逐任务归一化准确率定义为：

$$
\mathrm{NormAcc}_t=\frac{\mathrm{Acc}(\theta_{\mathrm{merge}},\mathcal D_t^{\mathrm{test}})}{\mathrm{Acc}(\theta_t,\mathcal D_t^{\mathrm{test}})}.
$$

它衡量合并后相对于对应单任务模型的能力保留率。跨任务汇总是逐任务比值的平均，通常不等于“平均合并准确率除以平均单任务准确率”。此外，不同 source 条件与处理后模型的分母口径需要核查，不能只凭归一化分数评价绝对能力。

## 主要结果与消融

### 范数调理恢复合并表现

以下为 Table 1 的绝对准确率，单位为百分比；Original 是常规源模型条件下相应方法的结果。

| 方法 / 模型 | Original | 范数失配 | 加 DisTaC |
|---|---:|---:|---:|
| Task arithmetic / ViT-B-32 | 70.4 | 63.6 | 70.0 |
| TIES / ViT-B-32 | 74.0 | 59.1 | 73.1 |
| TSVM / ViT-B-32 | 83.3 | 72.2 | 82.9 |
| WUDI / ViT-L-14 | 91.7 | 57.9 | 91.4 |

多数严重退化能够恢复到接近 Original 的水平。附录 E.4 同时比较仅缩放和缩放后蒸馏：仅缩放只能部分修复，加入 KD 才恢复更多能力。这支持“修正参数尺度”和“恢复模型函数”需要一起考虑。

### 低置信度条件下收益更大，但存在反例

| 方法 / 模型 | Original | 低置信度 | 加 DisTaC | 绝对增益 |
|---|---:|---:|---:|---:|
| TSVM / ViT-B-32 | 83.3 | 60.7 | 81.5 | +20.8 |
| TSVM / ViT-L-14 | 90.5 | 71.6 | 89.7 | +18.1 |
| WUDI / ViT-B-32 | 85.5 | 38.0 | 73.8 | +35.8 |
| WUDI / ViT-L-14 | 91.7 | 28.0 | 91.6 | +63.6 |
| EMR / ViT-L-14 | 93.0 | 27.4 | 92.3 | +64.9 |
| Iso-CTS / ViT-B-32 | 81.0 | 72.5 | 69.0 | -3.5 |

TSVM / ViT-B-32 的归一化准确率从 68.4% 恢复到 91.8%，接近 Original 的 92.4%。但 Iso-CTS / ViT-B-32 在同一条件下下降，因此不能沿用正文中“始终提升”的强表述。

另一个需要保留的原文差异是：正文称 ViT-L-14 最大绝对提升为 63.6，但 Table 1 的 EMR 行按 92.3−27.4 计算为 64.9 个百分点。报告逐项列数，不替作者猜测修正值。

### 训练动态：保住准确率，同时改变范数或熵

Figure 2 显示约前 100 步即可恢复大量准确率或明显降低熵，完整实验采用 500 步。范数调理中，最终 task vector norm 约为缩放初始化时的 1.1 倍，说明正则项约束有效但并非严格恒定。

部分 student 超过 teacher，论文指出这些案例的缩放初始化本身往往已经改善泛化。不能将全部提升解释成蒸馏凭空产生额外知识。

### 缩短优于拉长

Figure 3 与附录 E.3 扫描缩放系数 0 至 3。适度缩短向量通常比拉长更稳健；当系数为 3 时，所测任务表现均低于零样本基线。因此对本文的尺度失配，优先收缩长向量比放大短向量更有依据。

这并不意味着任意收缩都无损：系数趋近 0 时会回到预训练模型，任务适配也随之减少。KD 的作用正是恢复收缩所损失的任务行为。

### 对其他低置信度训练方法的泛化

附录 E.2 使用 Mixup 和 focal loss。以 ViT-B-32 的 TSVM 为例，Mixup 条件下准确率从 60.9 恢复到 80.1，focal loss 条件下从 69.3 到 81.8；Original 为 83.3。证据支持问题不局限于标签平滑，但尚不能覆盖所有校准方法。

### 无标签数据量与质量

Table 3 的百分比以“使用完整无标签数据的结果”为 100%，不是分类绝对准确率，也不是前述 source-normalized accuracy。

| 每类样本数 | 范数失配：DisTaC | 低置信度：DisTaC |
|---|---:|---:|
| 100 | 96.0 | 83.9 |
| 300 | 97.3 | 90.5 |
| 500 | 99.0 | 95.0 |

完整数据平均约每类 2,490 个样本。从缩放后的任务模型初始化，比从预训练模型重新蒸馏更耐受小数据量。例如每类 100 个样本、范数失配条件下，后者只有 71.1% 的相对表现。

Table 4 通过 Gaussian blur（高斯模糊）改变蒸馏输入；最强所测模糊下，范数失配保留干净数据结果的 91.7%，低置信度为 99.9%。这说明对该类视觉腐蚀较稳健，不代表对任意跨领域或语义分布变化都稳健。

### 语言任务：有迁移证据，也有负结果

附录 E.7 使用 natural language processing（NLP，自然语言处理）模型 RoBERTa-base、RoBERTa-large 与 Llama2-7B，在 General Language Understanding Evaluation（GLUE，通用语言理解评测）的四项任务上实验：Corpus of Linguistic Acceptability（CoLA，语法可接受性）、Microsoft Research Paraphrase Corpus（MRPC，释义判断）、Recognizing Textual Entailment（RTE，文本蕴含）和 Stanford Sentiment Treebank 的二分类任务（SST-2，情感分类）。

RoBERTa-large 的 task arithmetic 在范数失配下，绝对指标由 46.0 到 64.4，归一化指标由 58.1 到 80.5；低置信度下绝对指标由 64.5 到 70.0。

但 Llama2-7B 的 task arithmetic 在低置信度下，绝对指标从 75.7 降到 73.0，表中归一化指标却从 95.1 升到 95.9。RoBERTa-base 的 TSVM 在该条件下也由 69.6 降到 67.5。**相对能力保留率提高与最终绝对能力提高不能混用。** 复现时需检查归一化分母和逐任务结果。语言附录的训练与评测细节也比视觉主实验少，不宜将其当作生成式能力整合已得到充分验证。

### 计算成本应保留复核条件

Table 5 报告两张 NVIDIA A100、每设备 batch size 64、每步约 0.0064 秒、500 步约 3.2 秒、峰值显存 7.1 GB，并称包含在线 teacher 推理、不含评测。

这些是论文报告值，尚未通过本地代码复现。如此短的训练计时需要进一步确认设备同步、计时区间、数据加载和统计口径；不能直接作为大语言模型训练预算依据。更可靠的机制结论是：以任务模型为初始化做短程恢复，比从预训练模型重新学习任务所需的优化更少。

## 与蒸馏及能力整合的关系

### 同架构蒸馏可以服务于参数调理

本文 teacher 与 student 架构相同，teacher 是原任务模型，student 是调整尺度或置信度后的模型。蒸馏约束的是函数行为，锚点约束的是参数位置；两种约束共同得到较易合并的任务向量。

“函数保持”在这里是优化目标而非数学保证：有限输入上的 KL 较小，不保证所有输入行为都保持，也不保证 teacher 的每种能力都保留。

### 与 On-policy Distillation 的区别

On-Policy Distillation（OPD，在线策略蒸馏）在 student 自己生成的序列或访问的状态上取得 teacher 监督。DisTaC 使用固定的无标签任务输入，对分类输出做蒸馏；没有自回归 student rollout，也没有基于其行为分布刷新数据。

二者都使用 teacher 分布，但目标和数据来源不同。DisTaC 直接改善待合并参数的性质，OPD 主要在 student 状态分布上迁移行为。[[sources/papers/2026-mopd-multi-teacher-on-policy-distillation-for-capability-integration|多教师 OPD]] 则直接训练统一 student，属于另一种能力整合路线，不能把本文当成 OPD 的实验证据。

### 对大语言模型的可检验启发

可先记录同初始化 specialist checkpoints 的全局和逐层任务向量范数、任务输出熵、单任务质量与校准，再比较直接合并、仅缩放、缩放加蒸馏三种条件。生成任务还应观察回答长度、推理正确性、工具执行成功率和通用能力回退。

分类模型提高 logit margin 的收益不能直接推出生成模型应统一压低 token entropy。自回归模型中，探索、多种正确解法和不同 prefix 的不确定性都有实际价值。

## 局限与研究判断

本文最扎实的结论是：合并前的单模型准确率不足以衡量 mergeability，源模型训练配方造成的参数尺度和预测分布差异需要单独诊断。DisTaC 为这些失配提供了一种可与现有算子组合的修复流程。

边界主要包括：

- 实验都是分类任务，Llama2-7B 出现在实验里也不代表已经验证开放生成、推理或 Agent 合并。
- 共享初始化和参数可对齐是重要前提，不解决异构架构、词表映射或独立预训练模型的参数排列问题。
- 蒸馏需要任务内无标签输入；验证集选择合并系数仍可能需要标签。
- 提高源模型置信度与部署可靠性是不同目标。作者建议合并后再做温度校准，但最终系统仍需独立验证校准误差与分布外表现。
- 主表和附录存在负结果，部分归一化指标与绝对指标走势不同；正文的最大增益和计时数据也值得复现核查。

因此，实践中适合采用“源模型诊断 → 参数调理 → 合并 → 最终能力与校准评测”的流程。论文支持对特定失配做有针对性的调理，而不支持将高置信度、统一范数或某种蒸馏温度当作跨模型通用配方。

## 相关知识链接

- [[training/post-training/model-merging|Model Merging]]
- [[training/post-training/knowledge-distillation|Knowledge Distillation]]
- [[training/post-training/logits-distillation|Logits Distillation]]
- [[training/post-training/on-policy-kd|On-policy KD]]
- [[fundamentals/information-theory/label-smoothing|Label Smoothing]]
- [[fundamentals/probability/calibration-uncertainty|Calibration and Uncertainty]]
