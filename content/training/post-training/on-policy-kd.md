---
title: On-policy KD
created: 2026-03-08
published: 2026-03-08
modified: 2026-09-14
type: topic
status: mature
area: training
tags:
  - post-training
  - distillation
  - on-policy-kd
  - opd
---

On-policy KD，On-policy Knowledge Distillation，指 student model 先按照自己的当前 policy 生成样本，再让 teacher model 在 student 实际访问到的状态上提供 token distribution、log probability、correction 或其他反馈，最后用这些信号更新 student。

本页将现代 reasoning model 中最常见的 token-level OPD，On-Policy Distillation，作为主线。更广义的 On-policy KD 也可以使用 verifier、reward model 或人工系统，但只有在反馈仍然作用于 student 自己访问到的状态时，才保留 on-policy 的核心含义。

OPD 要解决的不是“如何把 teacher 的答案复制给 student”这么简单，而是一个自回归模型特有的问题：**student 在推理时会沿着自己的 prefix 继续生成，但离线训练数据通常只覆盖 teacher 或人工答案的 prefix。** 如果 student 早期生成了偏离答案的 token，后续状态就可能完全落在训练分布之外。OPD 让 student 先走到这些真实会访问的 prefix，再让 teacher 在这些位置提供纠正信号。

## 先建立一个统一视角

对一个 prompt $x$，student 按当前策略生成 response：

$$
y=(y_1,\ldots,y_T)\sim\pi_\theta(\cdot\mid x)
$$

第 $t$ 个位置的上下文记为：

$$
h_t=(x,y_{<t})
$$

在这个上下文上，teacher 和 student 分别给出下一个 token 的分布：

$$
p_T^t(v)=\pi_T(v\mid h_t),
\qquad
p_\theta^t(v)=\pi_\theta(v\mid h_t)
$$

其中 $v$ 遍历 vocabulary 中的 token。OPD 的基本闭环是：

```text
prompt x
  -> student rollout y ~ pi_theta
  -> teacher evaluates the student prefixes h_t
  -> compute token-level distribution / log-ratio signal
  -> update student on the visited prefixes
  -> refresh the student policy and sample again
```

这里的 `on-policy` 指的是：**prefix 或 response 的采样分布来自当前 student policy**，不是指 teacher 生成了一个“在线”答案。Teacher 通常只需要在 student 已经走到的 prefix 上做 forward pass，并不一定要重新生成完整 response。

### 两类常见实现

同样是 OPD，teacher signal 可以有两种不同的计算粒度：

1. **Full-vocabulary distillation**：在每个 student prefix 上，同时取得 teacher 和 student 对整个 vocabulary 的分布，再计算 forward KL、reverse KL 或 JSD。每个位置都能看到 teacher 对所有候选 token 的相对偏好，信号完整，但 logits 的计算、传输和显存成本高；
2. **Sampled-token distillation**：只读取 student 实际采样 token 的 teacher / student log probability，用 log-ratio 构造 token reward，再以 policy-gradient-style objective 更新。它不需要保存完整 vocabulary logits，更容易接入 RL trainer，但只是完整分布目标的 Monte Carlo 估计，方差和稳定化设计更重要。

这两类实现都可以是 on-policy，因为它们使用的 prefix 都来自当前 student。后文的 `soft rejection` 主要是 sampled-token OPD 的直觉；full-vocabulary OPD 更准确的描述是“在 student visited states 上做 soft-distribution matching”。

## 为什么需要 OPD

### Offline KD / SFT 覆盖的是参考 prefix

在普通 supervised fine-tuning（SFT，监督微调）或 offline knowledge distillation（offline KD，离线知识蒸馏）中，训练数据通常是固定的 $(x,y^*)$。模型在 teacher forcing 条件下看到：

$$
(x,y^*_{<t})
$$

然后学习预测 $y_t^*$。如果 student 自己在第 2 步生成了另一个 token，那么后续的 prefix 可能没有被训练数据覆盖。这个问题通常被称为 exposure bias 或 train-inference distribution mismatch。

### OPD 学习的是 student 会走到的状态

OPD 让 student 自己生成 $y$，因此 teacher 看到的是：

$$
h_t=(x,y_{<t}),\qquad y_{<t}\sim\pi_\theta
$$

即使 student 已经犯错，teacher 仍然可以在这个错误 prefix 上判断下一步哪些 token 更合理。它的优势不是让 student 复述一条理想轨迹，而是让 teacher 参与修复 student 自己的状态分布。

但 OPD 也有前提：student 至少要能产生有一定信息量的 prefix。如果 student 初始能力太弱，绝大多数 rollout 都是无意义或无法恢复的状态，teacher 的计算会被浪费在低价值样本上。因此 OPD 往往需要一个已有基本能力的 student，或者先用 offline SFT 做 cold start。

## OPD 为什么像 Soft Rejection

### 这是一个有用的直觉，不是严格的标准名称

在 sampled-token 实现中，OPD 可以被理解成一种 token-level、soft 的 rejection mechanism：student 先提出一个 token，teacher 根据自己对该 token 的偏好给出连续强度的“接受”或“拒绝”信号。

但 `soft rejection` 是帮助理解的类比，不是所有论文对 OPD 的正式命名。OPD 和 rejection sampling 仍然存在结构差异：

| 维度 | Hard rejection sampling | Token-level OPD |
|---|---|---|
| 决策粒度 | 通常是整条 response | 每个 sampled token 或每个 prefix |
| feedback | 通过 / 拒绝、或一个 response score | 连续的 teacher-student log-probability difference |
| 失败 response | 可以直接丢弃 | 不必丢弃，错误 prefix 也能提供纠正状态 |
| 同一条 response 内部 | 通常统一保留或丢弃 | 不同 token 可以同时有正、负信号 |
| 是否需要生成 teacher answer | 通常需要候选 response 和 scorer | teacher 可以只做 forward，输出 logprob / logits |
| 训练形式 | 常转成 SFT、DPO 或 RL 数据 | 可转成 KL loss、policy-gradient-style update 或混合目标 |

### Token-level soft rejection 的信号

对 student sampled token $y_t$，常见的 OPD token signal 是：

$$
r_t^{\mathrm{OPD}}
=
\log \pi_T(y_t\mid h_t)
-
\log \pi_\theta(y_t\mid h_t)
=
\log\frac{\pi_T(y_t\mid h_t)}{\pi_\theta(y_t\mid h_t)}
$$

它表达的不是“这个 token 在客观上是否正确”，而是：

- $r_t>0$：teacher 比当前 student 更偏好这个 sampled token，应该提高它在类似状态中的概率；
- $r_t<0$：teacher 比当前 student 更不偏好这个 token，应该降低它的概率；
- $r_t\approx0$：teacher 和 student 对这个 token 的判断接近，更新信号较弱。

举一个局部 token 分布的例子：

```text
token       student       teacher
a           0.60          0.10
b           0.30          0.80
c           0.10          0.10
```

如果 student 采样到 `b`，teacher-student log ratio 是 $\log(0.8/0.3)>0$，这个 token 会得到正向增强；如果 student 采样到 `a`，对应信号是 $\log(0.1/0.6)<0$，这个 token 会得到抑制。对于一条 response，前面的 reasoning token 可能是正信号，后面的错误 token 可能是负信号，所以 OPD 不需要把整条 response 粗暴地判为“保留”或“拒绝”。

### 它与 hard rejection 的根本区别

Hard rejection 通常是：

```text
student samples N responses
  -> scorer gives each response a score
  -> keep high-score responses
  -> discard low-score responses
  -> train on kept responses
```

OPD 更接近：

```text
student samples one response
  -> teacher scores every visited prefix / sampled token
  -> strengthen teacher-preferred local decisions
  -> suppress student-preferred but teacher-disfavored decisions
```

所以 OPD 的“soft”体现在两个方面：信号是连续值，不是 binary label；更新粒度可以细到 token，而不是整条 response。它仍然可以结合 rejection sampling，但 OPD 本身不等于“采样后只保留高分 response”。

## OPD 与 SFT 的区别

### Vanilla SFT 的目标

给定固定 target sequence $y^*$，SFT 通常优化：

$$
\mathcal{L}_{\mathrm{SFT}}(\theta)
=
-\sum_{t\in\mathcal A}
\log\pi_\theta(y_t^*\mid x,y^*_{<t})
$$

其中 $\mathcal A$ 表示 assistant output tokens。SFT 使用的是 hard target：每个位置主要告诉 student “参考 token 是 $y_t^*$”。前缀则是 ground-truth、teacher-generated 或 dataset 中保存的 prefix。

如果把 target token 写成 one-hot distribution $\delta_{y_t^*}$，SFT 的单 token loss也可以写成：

$$
D_{\mathrm{KL}}(\delta_{y_t^*}\|p_\theta^t)
=
-\log p_\theta^t(y_t^*)
$$

这里省略了与 student 参数无关的 target entropy。由此可以看出，SFT 与 forward-KL distillation 在数学形式上有连续关系：SFT 的 target 是只在一个 token 上有概率质量的 hard distribution；logits distillation 的 target 则是 teacher 给出的 soft distribution。OPD 在此基础上又把训练 prefix 从固定参考序列换成了 student rollout。

### OPD 的目标

OPD 使用 student 自己生成的 prefix，并在这些 prefix 上对齐 teacher 分布：

$$
\mathcal{L}_{\mathrm{OPD}}(\theta)
=
\mathbb{E}_{x}
\mathbb{E}_{y\sim\pi_\theta(\cdot\mid x)}
\left[
\frac{1}{T}
\sum_{t=1}^{T}
D\left(p_T^t\|p_\theta^t\right)
\right]
$$

或者在 sampled-token implementation 中，使用 $r_t^{\mathrm{OPD}}$ 作为局部 policy-gradient-style signal。

### 对比表

| 维度 | SFT | OPD |
|---|---|---|
| prefix 来源 | 固定数据 / reference sequence | 当前 student rollout |
| target 形式 | hard token 或固定 teacher sequence | teacher soft distribution / log ratio |
| 主要解决的问题 | 行为、格式和目标序列模仿 | student 自己状态上的纠错与分布对齐 |
| 是否看到 student 的错误 prefix | 通常看不到 | 可以直接看到 |
| teacher 是否要生成完整答案 | SFT teacher data 通常需要 | 不一定，forward logits 就可以 |
| 梯度位置 | reference prefix 下的 target tokens | student visited prefix 下的 distillation signal |
| 对 tokenizer 的要求 | 序列级 SFT 可允许不同 tokenizer | full-vocabulary KD 通常要求词表可对齐 |
| 稳定性 | 通常较高 | 依赖 student rollout 与 teacher-student compatibility |
| 计算成本 | 训练阶段不必调用 teacher | 需要 student sampling 和 teacher forward |

### 三个容易混淆的边界

1. **在 student samples 上 SFT，不自动等于 OPD。** 如果 teacher / verifier 选出一个完整 response，然后只用普通 SFT 学习这个 response，这更准确地叫 on-policy sequence distillation 或 online rejection sampling。
2. **OPD 可以使用 SFT。** 常见 recipe 是先用 teacher-generated data 做 off-policy cold start，再切换到 OPD；两者是前后阶段关系，不是互斥方法。
3. **OPD 不一定需要最终答案标签。** 只要 teacher 能在 student prefixes 上提供可靠 distribution signal，就可以训练；但 teacher distribution 本身不保证客观正确，仍可能传递风格、格式和错误偏好。

## OPD 与其他训练方式的关系

### Offline KD

Offline KD 在 teacher 生成或计算完数据后，student 在固定 corpus 上训练。它更稳定、容易复现和批处理；OPD 更能覆盖 student 当前错误，但 teacher latency、采样成本和训练动态都更复杂。

### Rejection Sampling

Rejection sampling 通常在 response level 进行筛选。它把多候选搜索的结果压缩为 chosen response；OPD 则把 teacher 的局部偏好分解到 student 已访问的 token / prefix 上。两者可以组合：先做 response-level filtering 保证基本质量，再在保留下来的或 student 当前困难的 prefix 上做 OPD。

### DPO

Direct Preference Optimization（DPO，直接偏好优化）使用 chosen / rejected response pair 学习偏好。它通常不要求 student 在当前训练 step 在线生成这些 response，也不会像 OPD 那样直接在 student 自己的错误 prefix 上提供 token-level teacher distribution。若 chosen / rejected pair 来自 student 当前 rollout，可以称为 on-policy preference data，但仍不等于 token-level OPD。

### RL / GRPO

Reinforcement Learning（RL，强化学习）通常通过 reward 优化 policy；Group Relative Policy Optimization（GRPO，组相对策略优化）使用同一个 prompt 下多个 response 的相对 reward 估计 advantage。OPD 的 teacher log-ratio 也可以写成 dense token-level reward，并用 policy-gradient-style update 实现，因此二者在优化形式上接近。

区别在于：

- OPD 的主要信号来自 teacher-student distribution discrepancy；
- GRPO 的主要信号来自 verifier、reward model 或规则系统给出的 response / token reward；
- OPD 不一定需要 group sampling、value model 或 outcome verifier；
- RL 可以优化 teacher 没有显式提供的外部目标，OPD 更像把 teacher policy 的局部知识迁移给 student。

因此，OPD 可以和 RL / GRPO 组合，但不能因为使用了 sampled-token reward 就把所有 OPD 都称为完整 RL。

## KL Loss 的方向与含义

Kullback-Leibler divergence（KL divergence，KL 散度）衡量两个概率分布之间的差异。“KL loss”不是一个单一的 loss，必须说明谁是第一个分布、谁是第二个分布，以及它作用在什么 prefix 上。

设同一个 student prefix $h_t$ 下：

$$
p=p_T^t,\qquad q=p_\theta^t
$$

### Forward KL：$D_{\mathrm{KL}}(p_T\|p_\theta)$

$$
D_{\mathrm{KL}}(p_T\|p_\theta)
=
\sum_v p_T(v)\log\frac{p_T(v)}{p_\theta(v)}
$$

它的优化部分等价于 teacher distribution 下的 cross-entropy。直觉上，teacher 是 target，student 需要覆盖 teacher 分布中有概率质量的多个 token，因此常被称为 mode-covering：

- teacher 偏好的多个合理 continuation 都可能被保留；
- teacher 给出的长尾概率也可能传给 student；
- 如果 teacher 分布比 student 更丰富，student 需要尝试覆盖更多 teacher modes。

它适合 teacher distribution 本身值得完整保留、student 容量足够、希望保留多样性的场景。风险是 student 容量不足时，可能被迫给 teacher 的多个模式都分配质量，导致每个模式都学得不够好；teacher 的低质量长尾也可能被一并蒸馏。

### Reverse KL：$D_{\mathrm{KL}}(p_\theta\|p_T)$

$$
D_{\mathrm{KL}}(p_\theta\|p_T)
=
\sum_v p_\theta(v)\log\frac{p_\theta(v)}{p_T(v)}
$$

它把 student 当前质量作为期望分布，更关心 student 已经在使用哪些 token，因此常被称为 mode-seeking：

- student 更容易集中到 teacher 高概率区域；
- 不会主动覆盖 student 没有探索到的 teacher modes；
- 对小 student 或希望输出更集中、更确定的任务可能更有利。

它的风险是可能忽略 teacher 的其他合理 modes，导致 diversity 降低或 mode collapse。若 teacher 对某个 student token 的概率极低，reverse KL 的惩罚可能非常大；工程上需要注意 log probability 的数值稳定性。

### Sampled-token OPD 与 reverse KL 的联系

在固定 prefix $h_t$、$y_t\sim p_\theta$ 的局部视角下：

$$
\mathbb{E}_{y_t\sim p_\theta}
\left[
\log p_T(y_t)-\log p_\theta(y_t)
\right]
=
-D_{\mathrm{KL}}(p_\theta\|p_T)
$$

因此，sampled-token OPD reward 可以看成负 reverse KL 的 Monte Carlo sample。实际实现可能直接使用 full-vocabulary KL，也可能使用 sampled-token reward；不能只看到“on-policy”就默认它一定是哪一个 KL 方向。

### Jensen-Shannon Divergence：JSD

Jensen-Shannon Divergence（JSD，詹森-香农散度）是一个对称、通常有界的分布差异。令：

$$
m_\beta=\beta p_T+(1-\beta)p_\theta,
\qquad 0<\beta<1
$$

则 generalized JSD 可以写为：

$$
D_{\mathrm{JSD},\beta}(p_T,p_\theta)
=
\beta D_{\mathrm{KL}}(p_T\|m_\beta)
+(1-\beta)D_{\mathrm{KL}}(p_\theta\|m_\beta)
$$

标准对称 JSD 通常是有界的。它在 forward KL 和 reverse KL 之间提供折中：比 forward KL 更不强调覆盖 teacher 的全部尾部，比 reverse KL 更不容易只追逐一个 mode。论文实验显示，最优 divergence 与任务和 decoding temperature 相关。

`JSD(0.1)`、`JSD(0.5)`、`JSD(0.9)` 中的系数取决于具体论文的定义，阅读时必须确认它放在 teacher 还是 student 一侧，不要只根据数字判断哪一边更重。

### Temperature 与 KL 不是同一个超参

Teacher distribution 常先经过 temperature scaling：

$$
p_T^{(\tau)}(v)
=
\frac{\exp(z_T(v)/\tau)}
{\sum_u\exp(z_T(u)/\tau)}
$$

其中 $\tau>0$：$\tau>1$ 让分布更平，$\tau<1$ 让分布更尖。Temperature 改变的是分布形状；forward / reverse KL 改变的是比较方向。两者需要分开调试和记录。

## 两种 KL 不能混为一谈

OPD 文献和 RL trainer 中还经常出现另一种 KL：student 与 reference model 的 policy-drift constraint。它和 teacher-student distillation KL 不是一回事。

### Distillation KL

比较的是 $D(p_T\|p_\theta)$ 或 $D(p_\theta\|p_T)$，作用是让 student 学习 teacher 的行为分布。

### Reference KL

在 policy optimization 中常见：

$$
D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})
$$

作用是限制 student 不要离开 reference policy 太远，控制更新幅度、保持已有能力并减少 reward hacking。它回答的是“student 相对 reference policy 漂移了多少”，不是“student 是否接近 teacher”。

一个组合目标可以写成：

$$
J(\theta)=
\mathbb E_{y\sim\pi_\theta}[r_{\mathrm{ext}}(x,y)]
-\beta D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})
-\gamma D_{\mathrm{distill}}(p_T,p_\theta)
$$

其中 $r_{\mathrm{ext}}$ 是外部 reward，$\beta$ 控制 reference constraint，$\gamma$ 控制 teacher distillation strength。一个实验里可能同时存在这两种 KL，报告超参时必须分别记录。

## GKD：统一 fixed data 与 student data

Generalized Knowledge Distillation（GKD，广义知识蒸馏）是 On-policy KD 的重要基础形式。它将 teacher forcing 数据和 student-generated 数据放到同一个目标中：

$$
\mathcal L_{\mathrm{GKD}}
=
(1-\lambda)\mathbb E_{(x,y)\sim\mathcal D}[D(p_T\|p_\theta)]
+
\lambda\mathbb E_{x\sim\mathcal D,\,y\sim\pi_\theta}[D(p_T\|p_\theta)]
$$

这里的 $\lambda$ 是 student-generated data fraction：$\lambda=0$ 接近 supervised KD，$\lambda=1$ 是纯 student on-policy outputs，$0<\lambda<1$ 是混合 fixed prefix 与 student prefix。

这个 $\lambda$ 不一定等于其他论文中“RL 与 KD 的 loss weight”，也不一定等于 ExOPD 的 extrapolation coefficient。GKD 把两个独立旋钮分开：状态分布由 $\lambda$ 控制，分布比较方式由 $D$ 控制。

## OPD 的典型变体

### Standard OPD

Standard OPD 通常直接使用 teacher 与 student 的 token log probability difference 或 teacher-student KL。它传递的是 teacher 的整体局部分布，可能同时包含 reasoning、language prior、format、verbosity、length 和 safety preference。因此 standard OPD 不是“只蒸馏 reasoning 能力”的保证。

### G-OPD / ExOPD

Generalized On-Policy Distillation（G-OPD，广义 On-policy Distillation）和 ExOPD，Extrapolative On-Policy Distillation，显式引入 reference model 和 reward coefficient。相关论文中的抽象形式为：

$$
J_{\mathrm{G\text{-}OPD}}(\theta)=
\mathbb E_{x,y\sim\pi_\theta}
\left[
\lambda\log\frac{\pi_T(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
-D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})
\right]
$$

在该论文的定义下，$\lambda=1$ 对应标准 OPD，$0<\lambda<1$ 是 interpolation，$\lambda>1$ 是 reward extrapolation。ExOPD 可能在 multi-teacher distillation 中更强，但系数过大也可能带来 length bias、reward hacking 和训练不稳定。这里的 coefficient 与 GKD 的 student-data fraction 不是同一个概念。

### Delta / OPD2

On-Policy Delta Distillation 关注 teacher 相对自身 base model 的变化：

$$
r_t^\Delta=
\log \pi_T(y_t\mid h_t)-\log \pi_{T,\mathrm{base}}(y_t\mid h_t)
$$

它试图提取 reasoning tuning 后新增或强化的能力增量，而不是蒸馏 teacher 的完整语言分布。由于 delta signal 不直接包含 student probability，OPD2 使用 centering，并要求 delta advantage 与传统 OPD advantage 方向一致时才更新：

$$
A_t^{\mathrm{D2}}=
\begin{cases}
A_t^\Delta, & A_t^\Delta A_t^{\mathrm{OPD}}>0\\
0, & \text{otherwise}
\end{cases}
$$

这说明 OPD 的关键设计不仅是是否 on-policy，还包括到底蒸馏 teacher 的完整行为还是某个训练阶段带来的能力增量。

### Self-distillation 与 Multi-teacher OPD

On-policy self-distillation 让同一个模型在不同上下文下扮演 teacher 和 student：teacher 可以看到 privileged solution 或 reference trace，student 只看到问题。Multi-teacher OPD 则让不同 domain 的 specialist teachers 负责不同 prompt，再在 student rollout 上做 token-level distillation。两种变体都保留了 student-state coverage，但分别引入 privileged context 或 teacher routing 的新问题。

## Agent 和长轨迹中的 OPD

在 tool-use agent 中，状态可能包含 system / user request、assistant reasoning、tool call、external observation、workspace 和 environment state。若 action 会改变环境，真正的 on-policy state 需要执行 action 后再获得 next observation；只在离线拼接文本上采样，不能完全等价于真实环境 rollout。

建议区分 loss mask：

```text
context and observations: attention only
assistant reasoning / action / answer: OPD target
external tool result: no generation loss
```

OPD 在长 trajectory 上还会遇到 signal degradation：早期错误会污染后续 prefix，teacher-student probability gap 会变大，局部 token log ratio 不一定反映整条任务是否成功，response length 还会放大 dense reward 噪声。实践中可以先做 cold-start SFT，对 reasoning、tool call 和 observation integration 分段计算，对 reward 做 centering / clipping，并结合 task-level verifier。

## 训练流程与关键超参

1. 准备真实任务、hard prompts 和 held-out prompts；
2. 选择 student、teacher 和可选 reference model；
3. 让 student rollout，记录 token、prefix、logprob、tool call 和 environment state；
4. 让 teacher 在一致的 prefix、template 和 tool context 上计算 logits / logprob；
5. 选择 forward KL、reverse-KL-style sampled reward、JSD 或 delta signal；
6. 应用 loss mask、temperature、centering、clipping 和 length normalization；
7. 更新 student，但不对离散 sampling 过程反向传播；
8. 重新采样并刷新 student state distribution；
9. 评估 teacher alignment、目标能力、general capability、response length 和 failure recovery。

需要单独记录的超参包括：student data fraction $\lambda$、teacher temperature、KL direction、JSD coefficient、reference model、distillation loss weight、external reward weight、rollout length、sampling temperature、reward clipping 和 update frequency。

## 稳定性与失败模式

- **Student rollout 质量太低**：先用 offline SFT、prompt filtering 或 minimum quality threshold 做 cold start。
- **Thinking pattern 不兼容**：监控 top-k overlap ratio、entropy gap 和 teacher-student logprob gap；必要时使用 teacher-aligned prompts。
- **Teacher bias 被完整迁移**：用 teacher-base delta signal，或混合多个 teacher / verifier。
- **Mode collapse 与长度偏置**：记录 per-token reward、response length、entropy 和 distinctness，不只看 sequence score。
- **Reward hacking 与 reference drift**：区分 distillation KL 和 reference KL，分别调节其权重。
- **Long-horizon failure**：结合 environment execution、task verifier 和 recovery supervision。

## 评估 OPD 是否真的有效

至少分四层评估：

1. **Distribution alignment**：teacher-student KL、top-k overlap、entropy gap、正负 token signal 比例；
2. **Local behavior**：格式、reasoning token、tool-call schema、argument grounding；
3. **Task capability**：数学、代码、检索、工具使用和 agent task success；
4. **Generalization and stability**：held-out / OOD prompts、response length、diversity、failure recovery 和多 seed 方差。

建议至少比较：base student、offline SFT / offline KD、on-policy sequence distillation、forward-KL OPD、sampled-token OPD、reverse-KL / JSD variant，以及 OPD + verifier / external reward。所有对照尽量固定 prompt distribution、student sampling、teacher version、训练 token、更新步数和 evaluation scaffold。

## 一句话总结

OPD 可以理解为：**让 student 先走自己的路，再让 teacher 在 student 真正走到的每个局部状态上，连续地告诉它哪些 token 值得保留、哪些 token 应该压低。**

它像 soft rejection，是因为反馈连续、局部、可正可负；它又不同于普通 rejection sampling，因为不需要把整条 response 二元保留或丢弃。它与 SFT 的根本差别在于 prefix 来源和监督形态，与不同 KL loss 的根本差别在于“覆盖 teacher 分布”还是“集中到 teacher 高概率区域”。

## 经典论文与资料

- [[sources/papers/2024-on-policy-distillation-of-language-models|On-Policy Distillation of Language Models]]
- [[sources/papers/2026-rethinking-on-policy-distillation-of-large-language-models-phenomenology-mechanism-and-recipe|Rethinking On-Policy Distillation of Large Language Models]]
- [[sources/papers/2026-learning-beyond-teacher-generalized-on-policy-distillation-with-reward-extrapolation|Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation]]
- [[sources/papers/2026-on-policy-delta-distillation|On-Policy Delta Distillation]]
- [[sources/papers/2026-self-distilled-reasoner-on-policy-self-distillation-for-large-language-models|Self-Distilled Reasoner]]
- [[sources/papers/2026-mopd-multi-teacher-on-policy-distillation-for-capability-integration|MOPD]]
- [[sources/papers/2020-learning-to-summarize-from-human-feedback|Learning to summarize from human feedback]]
- [[sources/papers/2022-instructgpt|Training language models to follow instructions with human feedback]]
- [[sources/papers/2024-deepseekmath|DeepSeekMath]]

## 相关知识链接

- [[training/post-training/knowledge-distillation|Knowledge Distillation]]
- [[training/post-training/offline-kd|Offline KD]]
- [[training/post-training/logits-distillation|Logits Distillation]]
- [[training/post-training/sequence-level-distillation|Sequence-level Distillation]]
- [[training/post-training/rejection-sampling|Rejection Sampling]]
- [[training/post-training/grpo|GRPO]]
- [[training/post-training/rlhf|RLHF]]
- [[training/post-training/dpo|DPO]]
