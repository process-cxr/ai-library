---
title: "Beyond Domain-Routed Supervision: Cross-Teacher Guarding for Multi-Teacher On-Policy Distillation"
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
  - multi-task-learning
  - gradient-conflict
  - credit-assignment
aliases:
  - ROADMOPD
  - Cross-Teacher Guarding for MOPD
source_url: https://arxiv.org/abs/2610.13521
pdf_url: https://arxiv.org/pdf/2610.13521v2
paper_date: "2026-10"
paper_order: "13521"
---

# Beyond Domain-Routed Supervision: Cross-Teacher Guarding for Multi-Teacher On-Policy Distillation

## 基本信息

- 标题：[Beyond Domain-Routed Supervision: Cross-Teacher Guarding for Multi-Teacher On-Policy Distillation](https://arxiv.org/abs/2610.13521)
- 作者：Shi-Yu Tian、Yong Liu、Ximan Liu、Tianrui Zhu、Sule Bai、Xinwei Yang、Ruiqi Liu、Kun-Yang Yu、Ming Yang、Zhi Zhou、Yu-Feng Li、Yipeng Sun、Simiu Gu
- 机构/来源：Nanjing University、JD.com
- 发表：arXiv:2610.13521v2
- 日期：2026-10
- 相关主题：[[training/post-training/on-policy-kd|On-policy KD]]、[[training/post-training/multi-task-learning|Multi-Task Learning]]、[[training/post-training/ppo|PPO]]、[[training/distributed-training/fsdp|FSDP]]、[[fundamentals/optimization/gradient-clipping|Gradient Clipping]]

## 研究问题：路由了教师，却没有隔离参数更新

多教师 on-policy distillation（Multi-Teacher On-Policy Distillation，MOPD）通常按领域选择教师：数学问题交给数学教师，代码问题交给代码教师，学生在自己生成的回答上接受逐 token 监督。

这样的 domain routing 解决了“向谁学习”，但各领域梯度最终仍然更新同一份学生参数。一个领域单独训练有效的监督，放入联合训练后，可能破坏其他领域已经具备的行为。

论文比较数学、代码与医疗的组合：Math×Code 相对兼容，联合训练通常保留或提高单域成绩；Medical 与另外两个领域组合时，梯度冲突更频繁，医疗成绩出现不对称下降。对 Qwen3-4B，Vanilla MOPD 的医疗平均分甚至低于初始学生（Figure 1、Table 1）。

作者提出 Cross-Teacher Guarding for MOPD，方法名为 **ROADMOPD**。它保留 routed teacher 的主监督，同时让未被路由到的教师在**同一条学生轨迹**上担任辅助 critic：

1. **Token 层**：两位教师相对 reference 的偏好一致时，增强主教师的修正；分歧时减弱。
2. **参数层**：用辅助教师在同一轨迹上形成的梯度，限制主梯度中冲突的部分，同时保留足够的主梯度范数。

医疗领域相对 Vanilla MOPD 提高约 5.6-8.8 个百分点，四组高冲突设置的两域平均分均高于所比较的多域基线。小模型仍有数学或代码上的取舍，因此结论是改善联合训练的权衡，而不是逐项消除所有干扰（§4.2）。

## 先区分学生、教师与 Reference

### 教师角色由当前领域决定

设当前问题来自领域 $d$，学生为 $\pi_\theta$：

- **Routed teacher $T_r$**：当前领域的主教师，负责这个任务的蒸馏监督。
- **Auxiliary teacher $T_a$**：其他领域教师，在相同回答上提供兼容性信号。
- **Frozen reference $\pi_{\mathrm{ref}}$**：学生初始化的冻结副本，用来比较教师相对初始行为的变化。

同一位教师在不同 batch 中可以扮演不同角色。数学问题上，数学教师是主教师，医疗教师是辅助教师；医疗问题上，角色交换。

reference 不是第三个 specialist，也不是正在训练的学生或 rollout 时的旧学生 $\pi_{\mathrm{old}}$。实验中，4B 与 1.7B 学生分别使用对应初始化的冻结副本。

### 同一轨迹意味着什么

学生对问题 $x$ 生成 $y=(y_1,\ldots,y_T)$，前缀为 $h_t=(x,y_{<t})$。所有教师与 reference 都在相同 $h_t$ 下，对学生实际采样的 $y_t$ 计算 log probability：

$$
\ell_t^r=\log\pi_{T_r}(y_t\mid h_t),\qquad
\ell_t^a=\log\pi_{T_a}(y_t\mid h_t),\qquad
\ell_t^{\mathrm{ref}}=\log\pi_{\mathrm{ref}}(y_t\mid h_t).
$$

教师不需要各自生成一条新回答。这里是 teacher forcing 式评分，复用同一学生轨迹，增加的是模型评分与梯度计算，而不是为每个辅助教师新增 rollout。

```text
domain prompt
  -> student generates one response
  -> routed teacher scores the sampled tokens
  -> auxiliary teachers score those same tokens at the same prefixes
  -> frozen reference supplies a common baseline
  -> modulate the target signal
  -> compute target and auxiliary losses at the same student parameters
  -> project target gradients, then merge and update the student
```

最终产物仍是一份全参数训练后的学生 checkpoint。所有教师和 reference 在训练中冻结，推理部署不要求调用它们，也没有增加任务专属 adapter 或推理期教师投票（Appendix B.1）。

## 从普通 OPD 到相对 Reward 表达

### 标准 Reverse KL

On-policy distillation（OPD，基于学生自身采样的策略蒸馏）的序列目标为：

$$
\mathcal L_{\mathrm{OPD}}(\theta)
=\mathbb E_{x\sim\mathcal D}
D_{\mathrm{KL}}\left(
\pi_\theta(\cdot\mid x)\,
\middle\|\,
\pi_T(\cdot\mid x)
\right).
$$

KL 是 Kullback-Leibler divergence（KL 散度），方向为学生到教师。利用自回归分解与 KL chain rule，可将它写为学生所访问前缀上的逐 token KL 之和；因此教师需要评价学生的前缀，而不是只提供自己的标准答案（Appendix A.1）。

### Reference 为什么会出现在公式里

对相同回答，加上再减去 reference 的 log probability，有：

$$
-\mathcal L_{\mathrm{OPD}}
=\mathbb E_{\pi_\theta}
\left[
\log\frac{\pi_T(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
-\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
\right].
$$

若只将第一个相对 reward 乘以 $\lambda$：

$$
J_\lambda(\theta)=
\mathbb E_{\pi_\theta}
\left[
\lambda\log\frac{\pi_T(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
-\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
\right].
$$

$\lambda=1$ 时 reference 项抵消，回到普通 OPD；$\lambda>1$ 放大教师相对 reference 的变化，对应 ExOPD 的奖励外推思路；$0<\lambda<1$ 则减弱这种变化。

ROADMOPD 把固定 $\lambda$ 换成每个 token 的 $\lambda_t$。这不是将整个 OPD loss 乘以一个 token 权重，也不是另外在 trainer 中添加一项独立 KL reward。

## 方法一：Cross-Teacher-Guided Credit Assignment

### 相对 Reference 比较修正方向

定义两位教师的相对证据：

$$
u_t=\ell_t^r-\ell_t^{\mathrm{ref}},\qquad
v_t=\ell_t^a-\ell_t^{\mathrm{ref}}.
$$

Agreement 为：

$$
s_t=\tanh(u_t/\tau)\tanh(v_t/\tau),
\qquad s_t\in[-1,1].
$$

$\tau$ 控制证据映射的平滑尺度，不是 rollout sampling temperature。取 tanh 可以限制极端 log-ratio 的影响，并同时保留方向与相对强度。

- $s_t>0$：两位教师相对 reference 的修正方向相同。
- $s_t<0$：两位教师相对 reference 的偏好相反。
- $s_t\approx0$：至少一位教师相对 reference 几乎没有额外证据。

**一致不只指共同提高某个 token 的概率。** 如果两个教师都比 reference 更不喜欢它，$u_t,v_t$ 同为负，一致性仍为正，此时应加强的是压低该 token 的修正。

辅助教师不需要具备解决当前领域任务的最强能力。它在这里提供的是自身领域训练后形成的偏好约束；这也意味着信号可能包含风格、置信度与知识差异，不能自动解释成事实正确性判断。

### 将 Agreement 变为相对 Reward 的系数

$$
\lambda_t=\operatorname{clip}
\left(1+\rho_gs_t,\lambda_{\min},\lambda_{\max}\right).
$$

实验取 $\tau=1$、$\rho_g=0.5$，范围为 $[0.5,1.5]$。在精确算术下，$1+0.5s_t$ 本身已处于这个区间，clip 不会进一步缩窄映射（Appendix B.3）。

原理是：一致时 $\lambda_t>1$，加强主教师相对 reference 的修正；冲突时 $\lambda_t<1$，减弱修正；没有额外一致证据时回到普通 OPD。

### 实际 Advantage 与 PPO-style Loss

记录学生当前 token log probability $\ell_t^\theta$，输入策略损失的信号为：

$$
A_t=\operatorname{sg}\left[
\operatorname{clip}\left(
\lambda_t(\ell_t^r-\ell_t^{\mathrm{ref}})
-(\ell_t^\theta-\ell_t^{\mathrm{ref}}),
-10,10
\right)\right].
$$

$\operatorname{sg}$ 表示 stop-gradient，整个 advantage，包括学生 log probability 项，都先截断再 detach。忽略截断时，可以改写成：

$$
A_t=
(\ell_t^r-\ell_t^\theta)
+(\lambda_t-1)(\ell_t^r-\ell_t^{\mathrm{ref}}).
$$

第一项是普通 sampled-token OPD，第二项是相对 reference 的额外调制。因此，即使 $\lambda_t<1$，也不能说整个普通 OPD advantage 都按相同比例缩小。

对 rollout 旧策略 $\pi_{\mathrm{old}}$，构造 importance ratio：

$$
q_t(\theta)=\exp(\ell_t^\theta-\ell_t^{\mathrm{old}}),
$$

$$
\ell_t^{\mathrm{agr}}(\theta)=
-\min\left\{
q_tA_t,
\operatorname{clip}(q_t,1-\epsilon_{\mathrm{PPO}},1+\epsilon_{\mathrm{PPO}})A_t
\right\}.
$$

PPO 是 Proximal Policy Optimization（近端策略优化）。这里借用其 clipped policy surrogate，$\epsilon_{\mathrm{PPO}}=0.2$；梯度从 ratio 路径传播，而不穿过 advantage。采用 PPO-style loss 并不意味着额外训练了 critic，也不意味着要为同一问题生成一组带结果奖励的回答。

### Mask 与 Sequence-Mean Reduction

对领域 batch $B_d$，每条回答先按有效响应 tokens 求平均，再对回答求平均：

$$
L_d^{\mathrm{agr}}=
\frac1{|B_d|}\sum_{i\in B_d}
\frac{\sum_tM_{i,t}\ell_{i,t}^{\mathrm{agr}}}
{\sum_tM_{i,t}}.
$$

prompt 和 padding 不进入 loss；返回的 EOS（End of Sequence，序列结束）token 可进入响应 mask。到长度上限时保留已经生成前缀上的监督，不丢弃整条回答；完全无有效 tokens 的样本贡献 0（Appendix A.1）。

这里有三个不同的 clipping：agreement coefficient 的 $[0.5,1.5]$、advantage 的 $[-10,10]$、importance ratio 的 PPO clipping。它们控制的对象不同，不能混为一种稳定化操作。

## 方法二：同轨迹辅助梯度与预防性投影

### 辅助梯度从哪里来

目标梯度为：

$$
g_d=\nabla_\theta L_d^{\mathrm{agr}}.
$$

对同一批领域 $d$ 回答，用辅助教师替换主教师，并关闭 agreement modulation，构造普通 OPD advantage：

$$
A'_{d\to a,t}
=\operatorname{sg}[\operatorname{clip}(\ell_t^a-\ell_t^\theta,-10,10)],
\qquad
g'_d=\nabla_\theta L'_{d\to a}.
$$

两次 loss 都在同一组学生参数与同一批存储轨迹上计算。关键区别是：

| 梯度               | 使用什么问题和回答          | 使用哪位教师                     |
| ------------------ | --------------------------- | -------------------------------- |
| 目标梯度 $g_d$     | 当前领域 $d$ 的学生轨迹     | 当前领域主教师，并使用 agreement |
| 辅助梯度 $g'_d$    | **仍是同一组领域 $d$ 轨迹** | 其他领域教师，关闭 agreement     |
| 独立辅助域任务梯度 | 辅助领域自己的问题和回答    | 辅助领域教师                     |

后两项不是同一个方向。所谓 preventive，来自在当前轨迹上检查拟议修正的兼容性，不是模型预测了所有未来任务的梯度。

### 负内积为什么表示局部冲突

对一个可训练参数 tensor，取目标与辅助梯度切片 $g,g'$。若执行普通梯度步 $-\eta g$，同轨迹辅助 loss 的一阶变化为：

$$
\Delta L'\approx-\eta\langle g,g'\rangle.
$$

所以 $\langle g,g'\rangle<0$ 表示该候选梯度方向会局部增加这个辅助 loss。注意它描述的是**同轨迹辅助目标**，不直接等于辅助领域 benchmark 会下降。

没有冲突则保留 $g$；有冲突时，全投影为：

$$
g_{\mathrm{full}}
=g-\frac{\langle g,g'\rangle}{\|g'\|_2^2}g'.
$$

全投影可以去掉沿 $g'$ 的冲突分量，但如果 $g$ 与 $g'$ 几乎反向，$g_{\mathrm{full}}$ 的范数可能接近 0，主领域学习信号也随之被抹掉。

### 用 Norm-Retention Floor 限制削弱程度

ROADMOPD 只在 $g$ 与 $g_{\mathrm{full}}$ 之间选择：

$$
\widetilde g
=g-\alpha^*\frac{\langle g,g'\rangle}{\|g'\|_2^2}g',
\qquad 0\le\alpha^*\le1.
$$

在 $\|\widetilde g\|_2\ge m\|g\|_2$ 的前提下，尽可能去掉冲突。设 cosine 为：

$$
c=\frac{\langle g,g'\rangle}{\|g\|_2\|g'\|_2},
$$

沿投影线段有：

$$
\frac{\|\widetilde g(\alpha)\|_2^2}{\|g\|_2^2}
=1-(2\alpha-\alpha^2)c^2.
$$

它在 $[0,1]$ 上随 $\alpha$ 增大而下降，解为（Appendix A.2）：

$$
\alpha^*=
\begin{cases}
0,&c\ge0,\\
1,&c<0\ \text{且}\ \sqrt{1-c^2}\ge m,\\
1-\sqrt{1-\dfrac{1-m^2}{c^2}},
&c<0\ \text{且}\ \sqrt{1-c^2}<m.
\end{cases}
$$

双领域实验使用 $m=0.6$。这是**逐参数 tensor 的投影后梯度范数**保留比例，既不是保留 60% 的参数，也不是保留 60% 的任务能力。

例如，$g$ 与 $g'$ 完全反向时，全投影会得到零向量；$m=0.6$ 时只能取 $\alpha^*=0.4$，得到 $\widetilde g=0.6g$。主梯度仍然存在，辅助冲突也仍然存在。这是明确的折中，不是完全无冲突的约束求解。

一般地：

$$
\langle\widetilde g,g'\rangle
=(1-\alpha^*)\langle g,g'\rangle.
$$

当 $c<0$ 且 $\alpha^*<1$ 时，右侧仍为负。不能把方法写成“投影后所有梯度必然非冲突”。

### 投影之后怎样更新学生

各领域只累加修正后的**目标**梯度：

$$
g_{\mathrm{update}}=\frac1K\sum_d\widetilde g_d.
$$

之后做一次 global gradient norm clipping，再执行一次 AdamW 更新。辅助梯度只用来定义投影约束，用完丢弃，不作为第二个教师的普通优化梯度再加入 $g_{\mathrm{update}}$（§3.4、Algorithm 1）。

范数约束成立在 tensor 投影阶段。领域聚合可能发生抵消，global clipping 和 AdamW 的动量、预条件及 weight decay 还会改变实际位移。因此它不保证最终优化器步长保留相同比例，也不保证辅助域任务 loss 或准确率不退化。

## 多于两位教师与分布式实现

### Token Agreement 对所有 Peers 求平均

对 $K>2$，当前领域 $d$ 的辅助教师为其余 $K-1$ 位：

$$
\overline s_{d,t}=
\frac1{K-1}\sum_{j\ne d}
\tanh\left(\frac{\ell_t^d-\ell_t^{\mathrm{ref}}}{\tau}\right)
\tanh\left(\frac{\ell_t^j-\ell_t^{\mathrm{ref}}}{\tau}\right).
$$

再用相同映射得到 $\lambda_{d,t}$。不同辅助意见可以相互抵消；均值为 0 时回到普通 OPD，但这不表示每位教师都无分歧。

### 顺序投影使用同一个原始范数预算

每个 tensor 先选择与当前梯度 cosine 最负的、尚未处理的 peer，投影后重新计算剩余 peers 的 cosine。全程约束：

$$
\|g_d^{(q)}\|_2\ge m\|g_d^{(0)}\|_2.
$$

不能每次只相对上一次结果保留 $m$，否则两次后只保证 $m^2$。例如 $m=0.6$ 时，两个局部 floor 只能保证原范数的 0.36，而不是共享预算要求的 0.6。

若预算已耗尽，后续冲突 peer 的投影系数可以为 0。处理过的 peer 不再回访，后来一次投影又可能改变之前的对齐关系，所以多 peer 过程同样不保证最终与所有辅助梯度同时非冲突（Appendix A.2、A.3）。

### Tensor-wise 不是任意本地 Shard-wise

实现使用 FSDP（Fully Sharded Data Parallel，完全分片数据并行）。梯度快照保存为 BF16（BFloat16，16 位浮点格式），内积和范数用 FP64（64-bit Floating Point，64 位浮点）计算，并在数据并行 group 上归约。

因此，同一个参数 tensor 的不同 shards 使用共同的 cosine 和投影系数。仅在每张卡自己的 shard 上独立决定 $\alpha$，会得到不同算法。论文图中有 layerwise 统计，但实际修正粒度是可训练参数 tensor，不应直接理解成整个 Transformer block 共用一个投影系数。

### 理论均匀领域平均与实际 Batch 权重

主文按领域均值定义梯度，再作 $1/K$ 平均。附录说明，带 full-batch normalization metadata 的实现，切出 $B_d$ 后仍保留全 batch 的分母，因此对应：

$$
w_d=\frac{|B_d|}{|B|},
\qquad
g_{\mathrm{update}}=\sum_dw_d\widetilde g_d.
$$

目标与同轨迹辅助梯度同时乘正数 $w_d$，在精确算术下不改变 cosine 与范数保留系数，但会改变最终聚合中的领域权重。balanced batch 下与 $1/K$ 一致；随机打乱后的不平衡 batch，不能默认是完全相同的 optimizer step。三域 ROADMOPD 明确采用每域 336 个 prompts 的平衡 batch（Appendix A.3、B.3）。

## 实验设置与评测协议

### 学生与教师

学生为 Qwen3-4B 和 Qwen3-1.7B，均通过 chat template 关闭 thinking mode，但回答仍可包含中间推理。训练采用全参数更新。

教师分别为：

- Math：`Qwen3-4B-Non-Thinking-RL-Math-Step500`；
- Code：`Qwen3-4B-Non-Thinking-RL-Code-Step300`；
- Medical：基于 Qwen3-8B 的 RL checkpoint `II-Medical-8B`。

RL 是 Reinforcement Learning（强化学习）。教师与学生不必同尺寸，但使用兼容的 Qwen3 tokenizer 和词表，确保在相同 prefix 上评分相同 token IDs。异构词表场景并未在这里验证（Appendix B.1）。

### 训练 Prompts

每域采样 10,000 个 prompts，双域共 20,000 个并随机打乱（Table 5）：

| 领域    | 来源                                               |   数量 |
| ------- | -------------------------------------------------- | -----: |
| Math    | DeepMath，difficulty 至少为 6，从约 57K 候选池采样 | 10,000 |
| Code    | Eurus-RL-Code                                      | 10,000 |
| Medical | MedMCQA training split                             |  4,000 |
| Medical | MedReason                                          |  3,000 |
| Medical | Medical-R1-Distill-Data                            |  2,000 |
| Medical | m23k                                               |  1,000 |

训练使用领域标签路由教师，但模型只接收问题。gold answers 不用作完整方法蒸馏 loss 的结果 reward；这不等于完全没有领域标注或不需要校验数据。

### 共享训练配置

实现使用 verl 训练、vLLM rollout 与教师评分，运行在单节点 8 张 NVIDIA B200 GPU（Graphics Processing Unit，图形处理器）上（Table 6）：

| 配置项                                | ROADMOPD 设置                              |
| ------------------------------------- | ------------------------------------------ |
| Train batch / PPO minibatch           | 1,008 / 1,008 prompts                      |
| 每 rollout batch 的 optimizer updates | 1                                          |
| Rollout batches                       | 双域 58，三域 116                          |
| PPO epochs                            | 1                                          |
| 每 prompt 回答数                      | 1                                          |
| Sampling                              | temperature 1、top-p 1                     |
| Prompt / response 上限                | 2,048 / 16,384 tokens                      |
| Optimizer                             | AdamW，betas 为 (0.9,0.999)                |
| Learning rate                         | $10^{-5}$，constant，无 warm-up            |
| Weight decay                          | 0.01                                       |
| Global gradient clipping              | 合并领域梯度后截到 1.0                     |
| Advantage / PPO ratio clipping        | $[-10,10]$ / 0.2                           |
| Agreement 参数                        | $\tau=1,\rho_g=0.5,\lambda_t\in[0.5,1.5]$  |
| Paired norm-retention floor           | $m=0.6$                                    |
| Trainer 额外项                        | Task reward、独立 trainer KL reward 均关闭 |
| Checkpoint / evaluation               | 每 29 次更新保存，离线评测                 |

六张 GPU 承载分片学生与三个张量并行度为 2 的 rollout replicas，另两张承载教师/reference 评分。三域中 Math、Code 教师与 reference 共用一张评分 GPU，Medical 教师占另一张。额外评分复用轨迹，但并非零成本。

### 比较哪些训练方法

- **Vanilla MOPD**：只由当前领域教师提供 sampled-token OPD advantage，不做跨教师调制或投影。
- **ExOPD**：固定外推系数 1.25，相对冻结学生初始化放大教师修正，不使用跨教师 gate。
- **Open-MOPD**：按领域响应 token 占比和教师-学生 gap 调整领域 loss 权重，并在同一 rollout batch 内用当前学生刷新 advantage。
- **CaMOPD**：区分能力恢复与能力保持更新，利用教师-学生 gap 选择样本，并通过领域调度重点恢复弱域。

它们分别修改监督来源、修正幅度、领域权重或样本/更新调度；本文方法另增加同轨迹跨教师兼容性信号。具体实现和预算仍需结合附录核对，不能只凭方法名假定所有对照完全等价。

### Avg@n 不是 Pass@n

全部评测采用 temperature 1、top-p 1、最多 16,384 个输出 tokens，并关闭 thinking mode（Appendix B.5）。

| 领域    | Benchmark                                                      |                        问题数 | 每题采样数 |
| ------- | -------------------------------------------------------------- | ----------------------------: | ---------: |
| Math    | AIME 2024、2025；HMMT 2025 February、November                  |                       每集 30 |         32 |
| Code    | HumanEval+、MBPP+、LiveCodeBench v6                            |               164 / 378 / 175 |          4 |
| Medical | MedQA、MedXpertQA Text、MedMCQA validation、SuperGPQA-Medicine | 1,273 / 2,450 / 4,183 / 2,755 |          1 |

AIME 是 American Invitational Mathematics Examination（美国数学邀请赛）；HMMT 是 Harvard-MIT Mathematics Tournament（哈佛-麻省理工数学竞赛）；MBPP 是 Mostly Basic Python Problems。`+` 的代码 benchmark 使用增强测试，需通过原始及附加测试。

Math 使用最后一个 boxed answer 经 `math-verify` 判定；Medical 为多项选择答案字母匹配。LiveCodeBench 使用 2025-01-04 到 2025-04-06 的 v6 split，编译、运行错误、超时和失败测试均判错误。

报告指标为：

$$
\operatorname{Avg@n}
=\frac1N\sum_{i=1}^{N}\frac1n
\sum_{r=1}^{n}\mathbf1[\text{completion}_{i,r}\text{ 正确}].
$$

它平均每次生成是否正确，对 Math 和 Code 等于从相同样本估计 pass@1，而不是“只要 $n$ 次有一次成功”的 pass@n。Math 的 32 次采样不等于给模型挑 32 次最好的机会。

### 领域平均与全任务平均

每个领域内部对 benchmark 等权，不按问题数加权；两域 All Avg. 是两个领域均值的平均，三域同理。

Code 有三个 benchmark，Medical 和 Math 各四个，所以 Code×Medical 的 All Avg. 不等于七项 benchmark 直接平均。三域 Table 3 的均值也不等于 Table 8 中 11 项任务等权平均，比较时必须固定口径。

## 主结果：改善高冲突领域的联合训练权衡

下面保留 Table 1 的领域均值，单位均为百分比。Single OPD 一行分别取各领域单独训练 checkpoint 的域内结果，属于多个专门模型的参照，不是一份同时取得全部分数的统一学生。

### Qwen3-4B：Medical 从退化恢复到接近单域水平

| 方法         | Math×Medical：Math | Medical |   All | Code×Medical：Code | Medical |   All |
| ------------ | -----------------: | ------: | ----: | -----------------: | ------: | ----: |
| Student      |              15.37 |   41.01 | 28.19 |              52.97 |   41.01 | 46.99 |
| Single OPD   |              44.27 |   47.90 | 46.09 |              60.91 |   47.90 | 54.41 |
| Vanilla MOPD |              45.97 |   39.45 | 42.71 |              60.53 |   40.75 | 50.64 |
| ExOPD        |              47.19 |   43.87 | 45.53 |              61.65 |   44.93 | 53.29 |
| Open-MOPD    |              45.34 |   44.52 | 44.93 |              60.83 |   47.34 | 54.09 |
| CaMOPD       |              46.64 |   47.75 | 47.20 |              60.64 |   47.28 | 53.96 |
| ROADMOPD     |              47.60 |   48.23 | 47.92 |              63.02 |   47.75 | 55.38 |

Medical 的初始学生为 41.01，Vanilla MOPD 的 39.45 与 40.75 都更低；ROADMOPD 分别提高 8.78 和 7.00 点，并恢复到接近单域 47.90 的水平。相对最强多域 Medical 基线，改进则为 0.48 和 0.41 点。

同时，Math 与 Code 的领域平均也达到所比较多域方法中的最高值。但这不是每个 benchmark 都最高，例如 Code×Medical 中 ExOPD 的 LiveCodeBench 为 30.00，ROADMOPD 为 28.71。

### Qwen3-1.7B：通过 Medical 收益换取更好的总分

| 方法         | Math×Medical：Math | Medical |   All | Code×Medical：Code | Medical |   All |
| ------------ | -----------------: | ------: | ----: | -----------------: | ------: | ----: |
| Student      |               8.05 |   30.37 | 19.21 |              42.13 |   30.37 | 36.25 |
| Single OPD   |              27.60 |   35.67 | 31.64 |              51.68 |   35.67 | 43.67 |
| Vanilla MOPD |              28.36 |   31.18 | 29.77 |              51.07 |   30.93 | 41.00 |
| ExOPD        |              28.44 |   33.01 | 30.72 |              54.16 |   29.77 | 41.97 |
| Open-MOPD    |              26.67 |   30.55 | 28.61 |              50.33 |   29.92 | 40.12 |
| CaMOPD       |              26.69 |   31.03 | 28.86 |              49.98 |   31.34 | 40.66 |
| ROADMOPD     |              27.21 |   36.89 | 32.05 |              53.22 |   36.51 | 44.86 |

Medical 分别比 Vanilla 提高 5.71、5.58 点，也超过单域参照 35.67。相对最强 Medical 多域基线，提高 3.88 和 5.17 点。

但 Math 与 Code 分别低于 ExOPD 1.23 和 0.94 点。All Avg. 因 Medical 收益分别提高 1.33 与 2.89 点，体现的是等权领域平均下更有利的折中。在 1.7B 学生上，不能沿用“医疗退化到比初始学生更差”的统一描述：Vanilla 在两个组合中仍略高于初始 Medical 30.37，只是明显低于单域训练。

### 低冲突和三域设置

Qwen3-4B 的其他组合如下（Table 2、Table 3）：

| 方法         | Math×Code：Math |  Code |   All | 三域：Math |  Code | Medical |   All |
| ------------ | --------------: | ----: | ----: | ---------: | ----: | ------: | ----: |
| Student      |           15.37 | 52.97 | 34.17 |      15.37 | 52.97 |   41.01 | 36.45 |
| Single OPD   |           44.27 | 60.91 | 52.59 |      44.27 | 60.91 |   47.90 | 51.03 |
| Vanilla MOPD |           46.12 | 61.24 | 53.68 |      46.98 | 61.26 |   41.03 | 49.76 |
| ExOPD        |           48.28 | 61.82 | 55.05 |      47.09 | 61.52 |   45.62 | 51.41 |
| Open-MOPD    |           44.32 | 59.82 | 52.07 |      44.06 | 60.56 |   38.83 | 47.82 |
| CaMOPD       |           46.30 | 60.37 | 53.34 |      45.96 | 60.97 |   48.20 | 51.71 |
| ROADMOPD     |           47.73 | 63.07 | 55.40 |      47.29 | 62.94 |   48.45 | 52.89 |

Math×Code 上，Math 比 ExOPD 低 0.55 点、Code 高 1.25 点，All 高 0.35 点。三域上则三个领域平均都领先所比较多域方法，All 比 CaMOPD 高 1.18 点。

这证明方法并非只能在一个受损领域上起作用，但三域是本文规模上限，不能据此断言任意数量教师都能保持同样收益与成本。

## 消融与机制证据

### Token 分歧与同轨迹原始梯度冲突相关

固定学生 checkpoint 后，作者将四条回答组成 batch，按有效 token 的平均负 agreement：

$$
S_-=\operatorname{mean}_t[\max(-s_t,0)]
$$

在各领域内划分五个 quintiles，每组 50 个 batches，检测未做 token modulation 或投影的主教师/辅助教师全参数梯度 cosine 是否为负。

从最低到最高分歧组，Math 冲突率由 24% 到 40%，Medical 由 46% 到 76%；中间组不全单调。它支持 agreement 是一个有用的冲突线索，但不是“每个分歧 token 都造成能力退化”的因果证明（Figure 3、Appendix B.5）。

### Credit Assignment、Projection 与 Retention 各自有什么作用

Table 4 的 Qwen3-4B 消融为：

| 设置                              | Math×Medical：Math | Medical |  Avg. | Code×Medical：Code | Medical |  Avg. |
| --------------------------------- | -----------------: | ------: | ----: | -----------------: | ------: | ----: |
| 去掉 Credit Assignment            |              45.78 |   47.74 | 46.76 |              59.54 |   47.46 | 53.50 |
| 去掉 Gradient Protection          |              47.19 |   48.11 | 47.65 |              63.01 |   46.87 | 54.94 |
| 去掉 Adaptive Retention，改全投影 |              45.78 |   48.40 | 47.09 |              62.85 |   47.03 | 54.94 |
| 完整 ROADMOPD                     |              47.60 |   48.23 | 47.92 |              63.02 |   47.75 | 55.38 |

去掉 token 调制，Math 与 Code 分别下降 1.82、3.48 点；去掉梯度保护，Code×Medical 的医疗分下降 0.88 点。

改成全投影后，Math×Medical 的 Medical 反而高 0.17 点，但 Math 大幅受损，两域平均下降 0.83 点。这正好符合 norm floor 的设计目的：不要为了消除局部辅助冲突而过度擦除主领域信号。它不是保证每个领域单独都比全投影好。

### 与 PCGrad 的差别在于比较哪两条梯度

PCGrad（Projecting Conflicting Gradients，冲突梯度投影）通常比较各任务在**各自数据**上形成的梯度；ROADMOPD 比较主教师与辅助教师对**同一行为**的监督梯度。

两个独立任务梯度总体对齐，并不排除辅助教师对当前具体轨迹上的某些修正有不同意见。Figure 4 的同轨迹 cosine 与独立域 cosine 对应不同问题，不能仅以哪一条曲线更负来判断哪个系统更差。

附录归档的 PCGrad 对照也开启 agreement modulation，Table 7 以 $\dagger$ 标注，并非“普通 MOPD 加一个原版 PCGrad”的裸基线：

| 比较                  | PCGrad + Agreement | ROADMOPD |
| --------------------- | -----------------: | -------: |
| Math×Medical：Math    |              46.85 |    47.60 |
| Math×Medical：Medical |              48.03 |    48.23 |
| Code×Medical：Code    |              60.90 |    63.02 |
| Code×Medical：Medical |              47.57 |    47.75 |

四个领域均值的提升为 0.18-2.12 点。这支持同轨迹辅助约束在当前设置中比独立任务梯度比较更有用，仍不是对所有 gradient surgery 方法的普遍排序。

### 轨迹可视化和范数曲线应该怎样读

token 高亮例子展示了某些修正被增强或减弱，但主文窗口从每个 prompt 的 20 条采样中选出，使用 token-modulation-only checkpoint，两条展示轨迹都达到 2,048-token 上限。它们是解释机制的样例，不是无偏的系数分布或完整方法的任务成功率估计（Appendix C.3）。

训练日志中的 surrogate 还只记录 task-0 的主 loss contribution，可为负，不是完整多域目标，也不是非负 KL 估计。不能凭该曲线下降就认定两个领域都收敛（Figure 8）。

范数分析则显示，Math×Medical 的 Math 方向中，全投影的同梯度估计在 58 次更新里有 40 次低于 0.6，最低 0.253；自适应约束的最低值为 0.600。Medical 方向未触及 floor，两种策略估计重合。这是投影公式在实际梯度上发挥约束作用的证据；全投影曲线不是另行训练出的 baseline，测的也不是 AdamW 后的实际参数位移（Figure 9）。

## 对照预算与证据边界

### 相同 Rollout Budget 不等于相同更新数

ExOPD、CaMOPD、ROADMOPD 在双域/三域分别使用 58/116 个 rollout batches，每 batch 1,008 prompts，通常一批一次 optimizer update。Open-MOPD 则将每批拆成四个 252-prompt minibatches，达到 232/464 次更新，并在每个 minibatch 用当前学生重新计算 advantage（Appendix B.3、B.4）。

因此这些对照较接近匹配采样批次，而不是完全匹配优化步数。CaMOPD 又通过 recovery/preservation 调度改变领域样本分配：医疗高冲突双域采用 3:1 更新比例，三域 Medical:Math:Code 为 3:1:1。报告其结果应视为本文同一训练栈上的具体实现，不能当作各原算法所有配方的最优结果。

Single OPD 的预算也不统一等同联合训练：4B 取 10,000 个单域 prompts 训练 3 epochs 的域内 checkpoint；1.7B 单域各训练 50 steps，联合主结果取 58 steps。它是专门训练参照，不是严格预算相同的理论能力上限（Table 8、Appendix B.3）。

### 没有新增 Rollout，仍有额外成本

辅助教师评分、同轨迹辅助 loss 的额外 backward、梯度快照和跨 shard 内积归约都会增加开销。论文没有给出完整 wall-clock、额外显存和等算力增益曲线，不能仅根据复用 rollout 宣称计算成本与 Vanilla MOPD 相同（Appendix D）。

### Teacher Agreement 是兼容性代理，不是真值

辅助教师在当前任务上可能缺乏领域知识，reference 的选择也会改变证据符号。两位教师共同偏好某种措辞或格式，不保证该 token 对任务正确性有贡献。方法通过最终联合评测验证这种代理的实用性，并没有建立每个 token 的语义级 credit truth。

### 局部投影约束与最终能力不同

负 cosine 描述的是指定 loss 的局部关系。部分投影允许残余负内积，多 peer 顺序处理不保证同时满足全部非冲突条件，域间聚合与 AdamW 又会改变最后位移。这些限制在论文附录中已明确，不应把“guarding”表述成完整的能力不遗忘保证。

### 规模、方差与泛化范围

实验限于 Qwen3 家族的 1.7B、4B 学生、数学/代码/医疗三个文本领域，教师词表兼容，最多三个教师。对更大模型、异构 tokenizer、长程工具环境以及更多领域，收益和辅助评分成本都需要另行验证。

Avg@32 和 Avg@4 降低的是生成评测的采样噪声，不是训练多随机种子的方差估计。论文没有为各主表方法差异给出完整多种子区间；尤其低冲突场景的 0.35 点平均收益，不宜独立解释为已确立的普适优势。

## 关键结论

ROADMOPD 将未被 domain routing 选中的教师，从“这批数据完全不用的专家”变成“当前行为的兼容性评估者”。它分别在 token 相对 reward 和参数梯度层面协调监督，同时保留主领域学习所需的梯度规模。

最值得保留的区别有三个：**辅助教师评估的是同一条目标域轨迹，agreement 调制的是教师相对 reference 的 reward，norm floor 约束的是投影阶段的梯度 tensor。** 这三点解释了方法如何工作，也限定了它能提供的保护强度。

## 相关知识链接

- [[training/post-training/on-policy-kd|On-policy KD]]
- [[training/post-training/multi-task-learning|Multi-Task Learning]]
- [[training/post-training/ppo|PPO]]
- [[training/distributed-training/fsdp|FSDP]]
- [[fundamentals/optimization/gradient-clipping|Gradient Clipping]]
- [[sources/papers/2026-learning-beyond-teacher-generalized-on-policy-distillation-with-reward-extrapolation|Learning beyond Teacher / ExOPD]]
