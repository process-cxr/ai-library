---
title: "A Fisher-Geometric View of Multi-Teacher On-Policy Distillation"
created: 2026-10-29
published: 2026-10-29
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
  - fisher-information
  - capability-integration
aliases:
  - RGBM
  - Residual-Governed Branch-and-Merge
source_url: https://arxiv.org/abs/2610.26525
pdf_url: https://arxiv.org/pdf/2610.26525v1
paper_date: "2026-10"
paper_order: "26525"
---

# A Fisher-Geometric View of Multi-Teacher On-Policy Distillation

## 基本信息

- 标题：[A Fisher-Geometric View of Multi-Teacher On-Policy Distillation](https://arxiv.org/abs/2610.26525)
- 作者：Kaiyuan Liu、Ziyuan Zhuang、Haoxuan Ma、Yannian Gu、Mingfei Song、Rongxiang Weng、Jieping Ye
- 机构/来源：Zhejiang University、Nanjing University、Shanghai Jiao Tong University、Meituan LongCat
- 发表：arXiv:2610.26525v1
- 日期：2026-10
- 相关主题：[[training/post-training/model-merging|Model Merging]]、[[training/post-training/on-policy-kd|On-policy KD]]、[[fundamentals/information-theory/kl-divergence|KL Divergence]]、[[fundamentals/optimization/loss-landscape|Loss Landscape]]

## 研究问题与核心结论

同一个基础模型经过不同领域训练后，会得到数学、代码、指令遵循等 specialist。如何将这些教师整合成一份可部署的学生权重，既涉及参数合并，也涉及行为蒸馏。

参数合并不需要反向训练，但合并系数怎样选、哪些能力尚未恢复，通常缺少统一依据。多教师 on-policy distillation（Multi-Teacher On-Policy Distillation，MOPD）让学生在自己的生成轨迹上学习对应教师分布，却需要持续 rollout，并可能通过共享参数更新干扰已有能力。

本文把两者联系到同一个局部几何问题：**当同源教师足够接近，且局部展开有效时，多教师 reverse KL（Kullback-Leibler divergence，KL 散度）的二阶近似就是 Fisher 加权的参数距离之和。** 因而可以先求一个低成本的合并解，再决定是否还有必要训练，以及哪些分支值得写回。

作者据此提出 Residual-Governed Branch-and-Merge（RGBM，残差驱动的分支训练与合并）：

1. 在 task-vector 子空间中，用无标签校准数据构造二次残差，求最小化最坏领域残差的合并模型。
2. 根据归一化残差决定 active domains，只独立训练这些领域的分支。
3. 用校准集上的线性诊断选择写回比例，将缩小后的分支增量加入合并模型。

Open-MOPD 上，Phase 1 的 Macro 为 32.52，高于官方 600-step 蒸馏学生的 30.93；Phase 2 达到 33.03。G-OPD（Generalized On-Policy Distillation，广义 on-policy 蒸馏）的中间教师组合上，Phase 1 为 53.34，略低于奖励外推配方 ExOPD 的 53.79，Phase 2 为 54.14（Table 1、Table 3）。因此，证据支持的是这两组同源模型上的低成本整合，不是“参数合并普遍优于充分训练的蒸馏”。

## 模型关系与最终产物

设共享 reference 参数为 $W_{\mathrm{ref}}\in\mathbb R^P$，$P$ 为参数数目；第 $d$ 个教师参数为 $W_{\varphi_d}$，共 $K$ 个领域。教师来自同一 reference，通过各自领域的强化学习（Reinforcement Learning，RL）获得专长。

任务向量与子空间定义为：

$$
\tau_d=W_{\varphi_d}-W_{\mathrm{ref}},
\qquad
T=[\tau_1,\ldots,\tau_K]\in\mathbb R^{P\times K},
$$

$$
\theta(\alpha)=W_{\mathrm{ref}}+T\alpha,
\qquad \alpha\in\mathbb R^K.
$$

RGBM 的训练与交付链路如下：

```text
shared reference + fixed domain specialists
  -> teacher-generated calibration completions and log-probability scoring
  -> Phase 1: solve merge coefficients in the task-vector subspace
  -> merged checkpoint theta_0*
  -> residual spectrum selects active domains
       -> train each active branch independently from theta_0*
       -> choose a shrinkage factor for each admitted update
  -> sum admitted updates into theta_0*
  -> one deployable generalist checkpoint
```

Phase 2 的每个 branch 都是从 Phase 1 checkpoint 复制出的独立学生，不是在不同教师上接一个 adapter。最终只部署合并后的统一模型，不需要路由到多个专家，也不要求保留分支供推理调用。

## 理论基础：Fisher 几何在这里是什么意思

### 参数距离不等于行为距离

两个模型的参数欧氏距离相同，它们的输出分布变化可能很不相同。某些方向对预测很敏感，另一些方向即使移动较多，也几乎不改变输出。

Fisher 信息矩阵用模型的 score function 描述这种敏感性：

$$
F_d=
\mathbb E_{x\sim\mathcal D_d,\,y\sim\pi_{W_{\varphi_d}}}
\left[g_d(x,y)g_d(x,y)^\top\right],
\qquad
g_d=\left.\nabla_\theta\log\pi_\theta(y\mid x)\right|_{\theta=W_{\varphi_d}}.
$$

这里 score function 是 log probability 对参数的梯度，不是 benchmark 分数。沿参数方向 $v$ 的局部行为变化用 $v^\top F_dv$ 衡量，敏感方向的代价更大。

Fisher 矩阵是半正定矩阵，可以包含方向间的耦合，不能简单理解成“给每个参数一个独立重要性权重”。完整 Fisher 与只保留对角线的近似也不是同一个对象。

### 从 Reverse KL 到二次参数目标

MOPD 的目标写为：

$$
J_{\mathrm{MOPD}}(\theta)=
\sum_{d=1}^{K}p_d\,
\mathbb E_{x\sim\mathcal D_d}
\left[
D_{\mathrm{KL}}\left(
\pi_\theta(\cdot\mid x)
\,\middle\|\,
\pi_{W_{\varphi_d}}(\cdot\mid x)
\right)
\right],
\qquad p_d\ge0,\quad\sum_dp_d=1.
$$

这里 KL 的方向为学生到教师。在自回归模型中，完整回答的 reverse KL 可以展开为沿学生生成前缀访问的逐 token 分布差异，因此与 on-policy distillation（OPD，基于学生自身采样的策略蒸馏）相联系。

对每个领域，在学生等于该教师的点 $\theta=W_{\varphi_d}$ 展开：

- KL 为 0；
- 一阶梯度为 0；
- 二阶 Hessian 等于对应 Fisher。

于是 Theorem 1 给出：

$$
J_{\mathrm{MOPD}}(\theta)
=\frac12\sum_dp_d
(\theta-W_{\varphi_d})^\top F_d(\theta-W_{\varphi_d})
+O\!\left(\max_d\|\theta-W_{\varphi_d}\|^3\right).
$$

忽略三阶余项后，求导得到：

$$
\theta^*_{\mathrm{Fisher}}
=\left(\sum_dp_dF_d\right)^{-1}
\sum_dp_dF_dW_{\varphi_d}.
$$

这就是 Fisher merge：更敏感的参数方向，对对应教师的偏离施加更大惩罚。直观上，它寻找一个在各领域行为度量下都尽量接近教师的折中点。

### 这个“最优”限定在什么范围

定理要求教师位于可共同展开的局部区域，相关映射足够光滑，三阶导数有界；误差还依赖学生到各教师的距离。共同来源和参数形状相同，并不足以保证三阶余项可忽略。

因此，闭式解是**二次近似目标的最优解**，不是原始非线性 MOPD 目标或下游准确率的全局最优解。逆矩阵形式还要求加权 Fisher 非奇异；存在零空间时，需要阻尼、伪逆或其他处理，不能默认过参数化模型的完整 Fisher 总能直接求逆（Appendix A.1）。

## 从完整 Fisher 到可计算的子空间问题

### 将 $P\times P$ 问题降到 $K\times K$

完整 Fisher 对数十亿参数模型不可直接构造和求逆。将学生限制为 $\theta(\alpha)=W_{\mathrm{ref}}+T\alpha$，记 $e_d$ 为第 $d$ 个标准基向量，则教师对应 $\theta(e_d)$。

定义投影 Fisher：

$$
S_d=T^\top F_dT\in\mathbb R^{K\times K},
$$

$$
J_{\mathrm{sub}}(\alpha)
=\frac12\sum_dp_d(\alpha-e_d)^\top S_d(\alpha-e_d).
$$

固定领域权重 $p_d$ 后，其闭式解为：

$$
\alpha^*_{\mathrm{sub}}
=\left(\sum_dp_dS_d\right)^{-1}
\sum_dp_dS_de_d.
$$

这时只有 $K$ 个合并系数，但还需要估计 $S_d$。论文下一步用输出 log probability 的差替代直接计算参数梯度。

### 隐式 Reward：不是正确性标签

对同一个 prompt-prefix $x$ 和候选 token $y$，定义：

$$
r_d(x,y)=
\log\frac{\pi_{W_{\varphi_d}}(y\mid x)}
{\pi_{W_{\mathrm{ref}}}(y\mid x)},
\qquad
r=[r_1,\ldots,r_K]^\top.
$$

这个 implicit reward 记录“教师相对 reference 改变了多少 token 偏好”，不是数学答案是否正确的外部 reward。正值只说明教师相对更偏好该 token。

对 logits 和 log-partition function 在 reference 附近作一阶展开，Proposition 2 得到：

$$
\ell_\alpha(x,y):=
\log\frac{\pi_{\theta(\alpha)}(y\mid x)}
{\pi_{W_{\mathrm{ref}}}(y\mid x)}
\approx \alpha^\top r(x,y).
$$

即在局部线性近似有效时，混合 task vectors 的系数，也近似混合其隐式 rewards。这个关系必须同时线性化 logits 与归一化项；不能只将未归一化 logits 相加，就宣称概率或 log probability 严格线性（Appendix A.2）。

### 为什么用二阶矩，而不是直接构造 Fisher

在上述近似下，$S_d$ 对应领域 $d$ 教师分布下、按前缀中心化的 reward covariance。实际流程直接估计未中心化二阶矩：

$$
M_d=
\mathbb E_{x\sim\mathcal D_{\mathrm{calib},d},\,y\sim\pi_{W_{\varphi_d}}}
[rr^\top]\in\mathbb R^{K\times K}.
$$

具体做法是：领域 $d$ 的教师生成校准 completions，reference 和**所有**教师再对这些相同前缀和 tokens 计算 log probability，得到向量 $r$。只给每个领域算自己的 $r_d^2$ 不够，$M_d$ 还需要 $r_jr_k$ 等交叉项。

设 $\mu_d(x)=\mathbb E_{y\sim\pi_{W_{\varphi_d}}}[r(x,y)]$，则：

$$
M_d\approx S_d+
\mathbb E_x[\mu_d(x)\mu_d(x)^\top].
$$

额外项半正定，所以在局部近似下，未中心化二阶矩给出投影 Fisher 二次项的保守上界。它包含均值偏移，不能直接称为精确 Fisher，也不是仅由领域之间的 correlation 决定的矩阵（Appendix A.3）。

## Phase 1：用原始残差求合并，用归一化残差作诊断

### 原始残差有一个直观解释

论文定义：

$$
L_d(\alpha)=(\alpha-e_d)^\top M_d(\alpha-e_d)
=\mathbb E_d[(\alpha^\top r-r_d)^2].
$$

右侧等式直接来自 $M_d=\mathbb E_d[rr^\top]$。因此它衡量：在领域 $d$ 的校准样本上，合并后的**近似 reward 组合**还差多少才能匹配该领域教师自己的 reward。

它不是下游错误率，也不是训练 loss 的剩余百分点。一个 residual 低的模型，仍可能答错很多问题；只是这个组合在当前 reward-space proxy 中已较好匹配教师。

### 为什么采用 Minimax

固定加权平均需要选定 $p_d$。RGBM 改为最坏领域目标：

$$
\alpha^*=
\operatorname*{arg\,min}_{\alpha\in\mathbb R^K}
\max_dL_d(\alpha).
$$

其依据是：$\max_{p\in\Delta_K}\sum_dp_dL_d=\max_dL_d$，即对领域权重取最坏情况。作者将其称为 Minimax QP（Quadratic Program，二次规划）；每个领域残差是凸二次函数，最大值仍为凸函数。

数值求解时使用 $\widetilde M_d=M_d+\lambda I$，$\lambda=10^{-4}$。这里的 $\lambda$ 是 Tikhonov 正则常数，不是合并系数的总和。正则化使问题具有稳定的正曲率；报告残差时仍使用未正则化的 $M_d$。

该阶段的实际结果是：

$$
\theta_0^*=W_{\mathrm{ref}}+T\alpha^*.
$$

**理论中的固定加权 Fisher 闭式解，与实际采用的 Minimax 求解不是同一个优化问题。** Phase 1 还使用了二阶矩 surrogate 和正则化，因此不能把实际 checkpoint 直接说成原始 MOPD 的精确驻点（§3.2）。

系数不限制在 $[0,1]$，也不要求和为 1。Open-MOPD 的系数约为 $(0.597,0.620,0.800)$，总和 2.017；它们是增量系数，不是推理路由概率或教师参数凸平均的概率权重。

### 求解之后才除以 Reward Energy

为了比较不同领域，定义 energy-normalized residual：

$$
\overline L_d(\alpha)=
\frac{L_d(\alpha)}{\mathbb E_d[r_d^2]}.
$$

分母是该教师相对 reference 的 reward energy，即自身 log-ratio 的二阶矩。$\overline L_d$ 表示相对这个量级仍未解释的残差，组成 domain residual spectrum：

$$
\overline{\mathbf L}=
[\overline L_1(\alpha^*),\ldots,\overline L_K(\alpha^*)].
$$

这里的 spectrum 是**逐领域残差列表，不是 Fisher 或 $M_d$ 的特征值谱**。

必须区分：求系数时使用 $L_d$，决定后续分支时使用 $\overline L_d$。如果先除 reward energy 再优化，相当于改变领域相对权重，得到的系数一般会变。Open-MOPD 的 energies 为 $(0.180,0.692,2.40)$，若在目标中归一化，Code 相对 Math 的权重约降低 3.8 倍，IF 的权重也明显降低（§4.3）。

### 为什么“无训练”仍有校准成本

Phase 1 使用 6,144 条校准 rollout sequences，三个领域各 2,048 条，再对对应 token 做 reference/教师 log-probability scoring。它省掉的是梯度更新，不是所有生成、推理和数据准备成本。

论文报告在 8 张 GPU（Graphics Processing Unit，图形处理器）上组装 $M_d$ 少于 15 分钟、低于 2 GPU·h，小规模求解少于 10 ms；但这不是 RGBM 全流程，包括 Phase 2 分支训练在内的总成本（§4.1、Appendix C.6）。

## Phase 2：门控、隔离训练与缩小写回

### 第一步：按最大相邻间隔选领域

将归一化残差降序排列，追加一个 0：

$$
\overline L_{(1)}\ge\cdots\ge\overline L_{(K)},
\qquad \overline L_{(K+1)}=0.
$$

选择最大相邻 gap 的中点作门槛：

$$
k^*=\operatorname*{arg\,max}_{i=1,\ldots,K}
(\overline L_{(i)}-\overline L_{(i+1)}),
\qquad
\epsilon=\frac{\overline L_{(k^*)}+\overline L_{(k^*+1)}}{2}.
$$

领域 $d$ 在 $\overline L_d>\epsilon$ 时 active，否则 passivated，即不训练或不写入该领域自己的更新。

Open-MOPD 的 $(Math,Code,IF)$ 为 $(0.128,0.052,0.015)$，最大 gap 为 $0.128-0.052=0.076$，所以 $\epsilon=0.090$，只接纳 Math。

G-OPD 的 $(0.094,0.090)$ 则因到 0 的 gap 最大，得到 $\epsilon=0.045$，两个领域都 active。这个规则是相对分离，不是固定绝对误差门槛；即使两个 residual 都很小，只要它们接近且为正，追加的 0 也可能使二者都 active。

### 第二步：从同一 Merge 训练独立 Branch

对每个 active domain：

$$
\theta_d^{(0)}=\theta_0^*,
\qquad
\Delta_d=\theta_d^{(\mathrm{end})}-\theta_0^*.
$$

每个 branch 只用自己的领域数据训练，不与其他 branch 共享梯度。Open-MOPD 主配置只训练独立 Math OPD 分支；G-OPD 为 Math、Code 分别训练 OPD 分支。

这种隔离保证“训练某个 branch 时，没有立即修改共享 merge checkpoint”，但**不保证之后把 $\Delta_d$ 加回去时，其他领域分数完全不变**。模型权重仍是共享的，没有一组专属于 Code 或 IF、可以被局部冻结的参数。

### 第三步：用线性诊断决定写回多大比例

对某个 checkpoint，计算真实合并 log-ratio $\ell$，并与 Phase 1 的预测组合 $u=\alpha^{*\top}r$ 作回归：

$$
\mathrm{Err}_{\mathrm{lin}}
=1-R^2
=\min_{a,b}
\frac{\mathbb E[(\ell-au-b)^2]}{\operatorname{Var}(\ell)}.
$$

这里 $R^2$ 是带截距 OLS（Ordinary Least Squares，普通最小二乘）回归的决定系数，衡量变化能被该线性 predictor 解释多少。对 Phase 2 的 $\theta_0^*+\gamma\Delta_d$，预测器仍固定为 $\alpha^{*\top}r$，不会重新拟合一组 task-vector 系数或把 branch 当作新的原始教师（Appendix C.1）。

候选缩放网格与容差为：

$$
\Gamma=\{0.125,0.25,0.5,0.75\},
\qquad \delta_{\mathrm{tol}}=0.10.
$$

$$
\gamma_d^*=
\max\left\{\gamma\in\Gamma:
\mathrm{Err}_{\mathrm{lin}}(\theta_0^*+\gamma\Delta_d)
\le\delta_{\mathrm{tol}}\right\}.
$$

选择的是通过无标签诊断的最大网格点，不是用 benchmark 标签选最高准确率。Open-MOPD 为 $\gamma^*_{\mathrm{math}}=0.125$；G-OPD 为 0.5。

最终统一模型为：

$$
\boxed{
\theta_{\mathrm{RGBM}}
=\theta_0^*+
\sum_{d:\overline L_d>\epsilon}\gamma_d^*\Delta_d.
}
$$

没有 active domain 时直接返回 $\theta_0^*$。Algorithm 1 未明确给出“某个 active branch 在整个 $\Gamma$ 上都不合格”时的 fallback，复现时需要核对实现，不能自行把不存在的 $\gamma_d^*$ 当成已定义值。

## 实验设置与评测口径

### 主实验 Open-MOPD

基础模型为 SmolLM3-3B，reference 为混合有监督微调（Supervised Fine-Tuning，SFT）得到的 MixSFT。三个同源 RL 专家分别面向 Math、Code、Instruction Following（IF，指令遵循）。

| 领域 | Benchmark                                                                             | 领域分数 |
| ---- | ------------------------------------------------------------------------------------- | -------- |
| Math | AIME 2024、AIME 2025，American Invitational Mathematics Examination（美国数学邀请赛） | 两项平均 |
| Code | LiveCodeBench v5、v6，程序生成与执行评测                                              | 两项平均 |
| IF   | IFEval、IFBench，可验证指令遵循评测                                                   | 两项平均 |

$$
\mathrm{Macro}=\frac{\mathrm{Math}+\mathrm{Code}+\mathrm{IF}}3.
$$

这里是领域原始分数的平均，不是 reference-to-teacher 的归一化恢复率。RouteRL oracle 按任务路由到三个教师，作为多模型参考；不是一份可部署的单模型基线。

官方 Multi-Teacher OPD 是已发布的 600-step 学生。本文的后续联合训练及各分支对照都从同一个 $\theta_0^*$ 出发，避免把不同初始化和后续更新混在一起（§4.1）。

### 次实验 G-OPD

使用 Qwen3-4B 的 Math 与 Code 专家。主表取原 G-OPD 工作中的中间 checkpoint：Math 500 steps、Code 300 steps；附录另测两者都训练到 1,200 steps 的组合。

七项评测包括四项数学任务 AIME 2024/2025、HMMT Feb/Nov，以及三项代码任务 HumanEval+、MBPP+、LiveCodeBench v6。HMMT 是 Harvard-MIT Mathematics Tournament（哈佛-麻省理工数学竞赛）；MBPP 是 Mostly Basic Python Problems，`+` 表示增强测试版本，HumanEval+ 同样使用增强测试。

G-OPD 的 Avg 是七项平均：

$$
\mathrm{Avg}_{7}=\frac{4\times\mathrm{Math}+3\times\mathrm{Code}}7,
$$

不是两个领域各占一半。SFT、OPD、ExOPD 的部分结果引用原工作，本文合并结果采用其官方协议；不能把所有对照都当作本文重新训练的统一运行。

ExOPD 是 G-OPD 工作中使用 reward extrapolation（奖励外推）的蒸馏配方。相关机制见 [[sources/papers/2026-learning-beyond-teacher-generalized-on-policy-distillation-with-reward-extrapolation|Learning beyond Teacher]]。

### 分支训练配置

Open-MOPD 分支采用 verl，在 8 张 NVIDIA A100/H800 GPU 集群上运行（Table 12）：

| 配置项                  | 设置                                                  |
| ----------------------- | ----------------------------------------------------- |
| Optimizer               | AdamW，$\beta_1=0.9,\beta_2=0.95,\varepsilon=10^{-8}$ |
| Peak learning rate      | $1.5\times10^{-6}$                                    |
| Schedule                | 5-step linear warm-up，之后 cosine decay              |
| Global batch            | 每 step 96 个 prompts                                 |
| 每 prompt rollout 数    | 8                                                     |
| Maximum sequence length | 24,576 tokens                                         |
| Maximum model context   | 32,768 tokens                                         |
| Generation temperature  | Math 0.6，Code/IF 1.0                                 |
| Weight decay            | 0.01                                                  |
| Checkpoint interval     | 每 5 steps                                            |

后续联合蒸馏的可见轨迹通常记录到 50 steps，Trust-Region 对照止于 20 steps；这些与官方 600-step 学生属于不同实验，不能混为同一训练长度。附录没有在主分支配置表中清楚给出最终写回分支的统一总更新数和完整 token 开销，单凭表格不能补出等算力比较。

## 主结果：先合并已覆盖多数收益，后续写回需要克制

### Open-MOPD 的领域分数

| 方法                                 |  Math |  Code |    IF | Macro |
| ------------------------------------ | ----: | ----: | ----: | ----: |
| MixSFT reference                     | 16.49 | 17.05 | 42.52 | 25.35 |
| RouteRL：多教师 oracle               | 27.24 | 26.27 | 53.52 | 35.68 |
| 官方 Multi-Teacher OPD               | 24.53 | 21.18 | 47.12 | 30.93 |
| Task Arithmetic                      | 24.35 | 22.81 | 46.77 | 31.31 |
| Subspace Fisher Merge                | 24.11 | 23.59 | 47.99 | 31.90 |
| RGBM Phase 1                         | 23.86 | 24.09 | 49.60 | 32.52 |
| 从 Phase 1 继续 Uniform Distillation | 23.97 | 21.36 | 47.42 | 30.91 |
| TIES：所有分支写回                   | 24.61 | 21.56 | 46.72 | 30.96 |
| DARE：所有分支写回                   | 24.58 | 22.40 | 47.28 | 31.42 |
| RGBM Phase 2                         | 23.93 | 24.63 | 50.52 | 33.03 |

原始数据来自 Table 1、Table 4。Uniform Distillation 行的数值是列出 checkpoints 的轨迹均值，不是 step 50 的最终分数；step 50 的 Macro 是 30.54（Table 9）。

Subspace Fisher Merge 使用均匀 $p_d=1/3$ 的平均残差与 $M_d$ surrogate，不是直接计算完整参数 Fisher。RGBM Phase 1 则使用最坏残差，对应不同系数。两者都优于对应的朴素参数基线，但并不逐列支配所有教师或训练结果。

此处 TIES、DARE 处理的是**从 Phase 1 训练得到的 branch updates $\Delta_d$**，不是原 specialist task vectors $\tau_d$。DARE 指 Drop And REscale（丢弃并重缩放）；TIES 是对更新裁剪、符号选择和合并的既有方法。该对照检验的是“把所有分支带回共享模型”是否有效，不构成这两类合并算法在所有场景下都较差的证据（Table 8）。

### Phase 2 的提升主要不在被训练领域

从 Phase 1 到 Phase 2：

- Math：23.86 到 23.93，仅增加 0.07 个百分点；
- Code：24.09 到 24.63，增加 0.54 点；
- IF：49.60 到 50.52，增加 0.92 点；
- Macro：32.52 到 33.03，增加 0.51 点。

因此，不能把 RGBM 描述成“诊断数学缺口，然后数学分支补齐了数学”。该小幅 Math 写回通过共享权重影响 Code 与 IF，最终增益主要来自这两个未训练自己分支的领域（§4.6、Appendix D）。

门控保护的是“不把某些领域自己的训练更新写回”，不是保证这些领域的参数或分数冻结。两者正是不同层次的保护语义。

### G-OPD：中间教师与收敛教师要分开看

| 中间教师组合上的方法 |  Math |  Code | 七项 Avg |
| -------------------- | ----: | ----: | -------: |
| 按域使用专家：oracle | 46.00 | 61.17 |    52.50 |
| SFT                  | 44.33 | 60.80 |    51.39 |
| OPD                  | 46.38 | 60.57 |    52.46 |
| ExOPD                | 47.65 | 61.97 |    53.79 |
| Task Arithmetic      | 43.05 | 60.65 |    50.59 |
| Equal-Weight Average | 22.55 | 55.67 |    36.75 |
| RGBM Phase 1         | 46.17 | 62.89 |    53.34 |
| RGBM Phase 2         | 47.55 | 62.93 |    54.14 |

Phase 1 相对 ExOPD 仍低 0.45 点，Phase 2 相对 ExOPD 高 0.35 点；相对 Phase 1 提高 0.80 点。这里没有“零训练就超过最强训练方法”的统一结论（Table 3）。

对两个都训练到 1,200 steps 的收敛教师，本文只报告 Phase 1 的七项平均 57.35，略低于 multi-teacher ExOPD 的 57.57；没有该组合的 Phase 2 实验。不能把中间教师上的 54.14 与收敛教师的分数直接比较方法优劣（Table 7）。

## 消融与诊断：哪些选择真正影响结果

### 1. 求合并系数时不能随意换成归一化残差

Table 2 比较两种目标、两种系数自由度，以及 raw/normalized residual：

| 目标与系数形式        | 优化原始 $L_d$ 的 Macro | 优化归一化 $\overline L_d$ 的 Macro |
| --------------------- | ----------------------: | ----------------------------------: |
| 平均目标，共享 scalar |                   31.96 |                               30.86 |
| 平均目标，自由 vector |                   31.90 |                               30.59 |
| Minimax，共享 scalar  |                   31.19 |                               28.99 |
| Minimax，自由 vector  |                   32.52 |                               30.66 |

这组实验中，四个原始残差方案都优于四个归一化方案。原因是归一化进入目标后，会改变领域权重；但证据限于当前数据，不能推导为所有领域整合都不应在优化时做尺度归一化。

该表也说明“更接近二阶理论”的形式未必自动对应最高 benchmark：平均目标的 scalar 31.96 略高于 vector 31.90；Minimax 下则自由 vector 明显优于 scalar。残差是一个有用 surrogate，不能被视作准确率的等价函数。

### 2. 低残差不表示这个领域单独学不动

各独立分支在自己领域的 peak gain 都是正值（Table 11）：

| 领域 | 归一化残差 | OPD 分支自身领域 peak gain |
| ---- | ---------: | -------------------------: |
| Math |      0.128 |                      +1.10 |
| Code |      0.052 |                      +0.43 |
| IF   |      0.015 |                      +0.48 |

Outcome-gated 或 RL 分支也都有自身领域收益。因此 residual 的语义是**联合折中后的 reward-space 余量**，不是某个分支继续训练的最大可获准确率。

写回被拒绝的 IF 分支 $0.5\Delta_{\mathrm{IF}}$ 后，Code 降至 20.70、Macro 降至 31.10（Appendix C.4）。这支持评估跨域写回影响的重要性；不能只看某个分支自己的验证集涨分，就决定全部加入共享模型。

### 3. 只有门控，没有 Shrinkage，仍可能退化

同一个 Math 分支的不同写回比例为（Table 10）：

| $\gamma$ |  Math |  Code |    IF | Macro | $\mathrm{Err}_{\mathrm{lin}}$ |
| -------- | ----: | ----: | ----: | ----: | ----------------------------: |
| 0.125    | 23.93 | 24.63 | 50.52 | 33.03 |                        0.0701 |
| 0.25     | 24.19 | 24.02 | 47.82 | 32.01 |                        0.1749 |
| 0.50     | 25.42 | 23.48 | 48.08 | 32.32 |                        0.2594 |
| 0.75     | 25.08 | 22.87 | 45.51 | 31.15 |                        0.6442 |
| 1.00     | 25.03 | 22.51 | 47.87 | 31.80 |                        0.7231 |

$\gamma=1$ 是不缩小的诊断对照，不在实际候选网格中。按 0.10 容差，只有 0.125 合格；更大写回可以提高 Math，却牺牲 Code 与 IF，导致 Macro 低于原 merge。

同一附录中的等位移规模、中间 checkpoint 或双点平均对照也未超过 33.03。它们用于事后检验，不是作者据 benchmark 选 $\gamma$ 的搜索过程。

### 4. 联合蒸馏的五种对照均出现回退

作者测试了 Uniform、Outcome-Gated、Outcome-Veto、Trust-Region、Hybrid Teacher-Anchor 五种后续联合训练目标。结果包括（Table 9）：

- Uniform 的 step 50 Macro 为 30.54，低于 Phase 1 的 32.52；
- Outcome-Gated 的 step 50 为 31.28；
- Trust-Region 在 step 20 为 31.25，Code 从 24.09 降至 21.18；
- 五组列出的 checkpoints 均低于 RGBM 的 33.03。

Trust-Region 对照将估计的逐 token policy drift 约束到约 $1.4\times10^{-3}$ nat/token，仍出现任务分数退化（§4.4）。这说明在所测协议中，小的平均分布漂移没有保护住全部 benchmark 行为。

但不能将它概括为“KL 只能限制各向同性距离，所以所有 KL 约束都无效”：KL 的局部二次几何本身就是 Fisher 度量，通常具有方向性；小的平均 KL 也不能保证对关键行为的逐项约束。实验支持特定配方没有解决干扰，尚不足以证明联合蒸馏在一般情况下必然失败。

### 5. 线性诊断发现明显失效，但没有覆盖全部理论误差

附录沿各领域共享 scalar 的 family $\theta(a)=W_{\mathrm{ref}}+a\sum_d\tau_d$ 检查 38M tokens：$a=0.75$ 时回归 $R^2=0.935$，$a=1.50$ 时降到 0.557，Macro 也降至 22.48（Figure 5）。

这是 isotropic family 的诊断，不是自由 vector 的 Minimax 系数扫描。它支持大步参数外推可能破坏一阶组合关系，但不能直接充当每个 $\theta_0^*$ 都满足二阶最优条件的证明。

Appendix B.4 报告隐式 reward 的 Math-Code correlation 为 0.313，Math-IF、Code-IF 分别为 0.043、0.035。它们描述校准样本上的 reward 关系；接近零相关不等于完整参数空间严格正交，也不等于加入某个领域后一定没有干扰。

## 理论、证据与工程边界

### 局部等价不等于原始目标的全局解

从 Fisher 二次目标到实际 Phase 1，经历了子空间限制、一阶 reward 近似、二阶矩上界、Minimax 替换和正则化。它们让问题可计算，但也改变了目标与可行范围。

二阶 Fisher 解是该局部 surrogate 的解，Minimax checkpoint 又不是同一个加权平均目标的驻点。因此，“从 merge 再做梯度只能优化三阶余项”只对精确二次目标的相应驻点具有近似意义，不能不加条件地用于真实 RGBM Phase 1。

### $R^2$ 是回归可解释性，不是能力安全证书

由于允许拟合斜率 $a$ 和截距 $b$，即使 $\ell=2u+b$，该诊断也可能得到完美 $R^2$。所以低 $\mathrm{Err}_{\mathrm{lin}}$ 不保证 log-ratio 数值幅度与预测完全一致，更不保证准确率、校准和安全行为不下降。

论文 Appendix D 也明确承认，这个诊断不约束 Theorem 1 的三阶余项。它在校准样本上检查一阶 reward superposition，不是整个分布空间的统一误差界。

Proposition 4 给出的 shrinkage 结论是：对 Hessian 有界的局部区域，余项的一个保守上界可随 $\gamma$ 减小而减小。这个上界单调，不代表实际余项或下游准确率必然随 $\gamma$ 单调变化；Table 10 的 Macro 已经显示非单调性。

### 多分支单独合格，不自动意味着相加后合格

式 (11) 逐个分支检验 $\theta_0^*+\gamma_d\Delta_d$，最终写回的是多个增量之和。若有多个 active domains，各分支单独通过诊断，并不能一般性推出其总和仍通过或不产生新干扰。G-OPD 的两个分支相加后有效，是该实验的经验结果，不是单分支规则提供的数学保证。

### “无标签”与“无数据、无训练、无 Rollout”不同

Phase 1 的校准无需 gold answers，但使用教师生成文本和全部模型前向评分；Phase 2 又包含真实 OPD branch training。因此“零额外联合训练步数”只是说不在共享学生上继续联合更新，不能写成整个 RGBM 没有训练成本。

附录对校准重采样报告：$N\ge1024$ 时系数标准差低于 0.012，残差排序稳定；这属于当前教师与校准池的 bootstrap 证据。对更大 $K$ 的样本复杂度和谱排序推断，还依赖分布、尾部与 gap 条件，不能只凭 $O(K\log K)$ 形式认定任意领域数都足够稳定（Appendix C.6）。

### 证据规模和收益幅度仍有限

实验覆盖 3B/4B 两个模型家族、主要为 $K=2$ 或 $3$ 的同源专家，未建立异构基础模型、几十个教师和大规模工业模型上的可迁移保证。

Phase 2 在两个主设置中的增益分别为 0.51 与 0.80 点；相对强 ExOPD 的优势在 G-OPD 中为 0.35 点。论文没有提供这些主差异的训练多种子置信区间，校准 bootstrap 也不能代替下游指标显著性检验。主要收益来自“先找到较好的合并，再避免破坏它”，而非证明一个大幅、普适的二阶段能力跃迁。

## 关键结论

这篇论文最有价值的部分，是把参数合并、输出分布匹配与跨域写回干扰放在同一个局部几何框架中，并给出了只需前向评分的残差代理量。

RGBM 的完整逻辑是：**先在可计算的 task-vector 子空间求一个联合折中，再用残差分配后续训练，最后对训练增量做有限、经过诊断的写回。** 其中，原始残差用于优化，归一化残差用于选择领域，线性回归误差用于选择写回尺度，三者不能互换；这些代理量也都不直接等同于任务能力。

## 相关知识链接

- [[training/post-training/model-merging|Model Merging]]
- [[training/post-training/on-policy-kd|On-policy KD]]
- [[fundamentals/information-theory/kl-divergence|KL Divergence]]
- [[fundamentals/optimization/loss-landscape|Loss Landscape]]
- [[sources/papers/2026-learning-beyond-teacher-generalized-on-policy-distillation-with-reward-extrapolation|Learning beyond Teacher]]
