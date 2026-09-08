---
title: Cross Entropy
created: 2025-12-27
published: 2025-12-27
modified: 2026-09-07
type: topic
status: mature
area: fundamentals
tags:
  - math
  - information-theory
  - loss-function
  - language-modeling
aliases:
  - Cross Entropy
  - Cross-entropy Loss
---

## 概念界定

Cross Entropy，交叉熵，衡量真实数据来自分布 $p$ 时，使用模型分布 $q$ 预测或编码这些数据所需的平均代价：

$$
H(p,q)=-\sum_x p(x)\log q(x)=\mathbb{E}_{x\sim p}[-\log q(x)]
$$

它把模型分配给真实事件的概率转换成负对数代价：真实事件的预测概率越高，代价越小；概率越低，代价越大。Cross Entropy 因此连接了信息论中的编码代价、统计学习中的 Maximum Likelihood，以及大语言模型中的 Next-Token Prediction。

在语言模型训练中，需要区分三个紧密相关但层次不同的概念：

- **Cross Entropy** 是两个分布之间的期望负对数概率；
- **Negative Log-Likelihood, NLL** 是给定观测样本后得到的负对数似然；
- **Maximum Likelihood Estimation, MLE** 是选择参数使观测数据似然最大的估计原则。

对于 categorical output 和 one-hot target，单个位置的 Cross Entropy 与该 target 的 NLL 数值相同；在整个数据集上最小化这些 NLL，又等价于进行 Maximum Likelihood training。它们在 LLM 训练中常指向同一个 objective，但并不是脱离条件后完全相同的概念。

## 从 Hidden State 到 Token Loss

Transformer 在位置 $t$ 产生 hidden state $h_t\in\mathbb{R}^{d}$，LM Head 将它映射到 vocabulary size 为 $V$ 的 logits：

$$
z_t=W_{\mathrm{LM}}h_t+b,\qquad z_t\in\mathbb{R}^{V}
$$

Logits 是未归一化分数，不要求非负或总和为 1。Softmax 将它们转换为 next-token distribution：

$$
q_\theta(j\mid x_{<t})=
\frac{e^{z_{t,j}}}{\sum_{k=1}^{V}e^{z_{t,k}}}
$$

如果真实 next token 的 vocabulary index 是 $y$，one-hot target 为 $p_j=\mathbf{1}[j=y]$，则该位置的 Cross Entropy 为：

$$
L_t=-\sum_{j=1}^{V}p_j\log q_j=-\log q_y
$$

把 Softmax 展开可得：

$$
\boxed{L_t=-z_{t,y}+\operatorname{logsumexp}(z_t)}
$$

因此，LLM 中常见的三种写法分别对应同一计算的三个视角：

$$
\underbrace{-\sum_jp_j\log q_j}_{\text{Cross Entropy}}
=
\underbrace{-\log q_y}_{\text{target NLL}}
=
\underbrace{-z_y+\operatorname{logsumexp}(z)}_{\text{logits implementation}}
$$

“one-hot target 下只读取正确类别”并不意味着其他 token 与 loss 无关。$q_y$ 的分母包含整个 vocabulary 的 logits，因此提高竞争 token 的 logit 也会降低正确 token 的概率。

## 梯度为什么是 $q-p$

Softmax 与 Cross Entropy 组合后，对第 $j$ 个 logit 的梯度为：

$$
\boxed{\frac{\partial L}{\partial z_j}=q_j-p_j}
$$

对于 one-hot target：

$$
\frac{\partial L}{\partial z_y}=q_y-1,
\qquad
\frac{\partial L}{\partial z_j}=q_j\quad(j\neq y)
$$

梯度下降因此会提高真实 token 相对于其他 token 的 logit，并按模型当前分配给错误 token 的概率压低它们。模型越确信一个错误 token，作用在该 token 上的梯度越大。

梯度还满足：

$$
\sum_j\frac{\partial L}{\partial z_j}=0
$$

这与 Softmax 对 logits 共同平移的不变性一致：给所有 logits 同时加上常数 $c$，不会改变概率或 Cross Entropy。

当 target 不是 one-hot，而是 Label Smoothing 或 Knowledge Distillation 中的 soft distribution 时，梯度仍然是 $q-p$，只是每个类别都可能接收非零 target mass。

## 与 Next-Token Prediction 的关系

对于 token 序列 $x_1,\ldots,x_T$，自回归模型通过概率链式法则分解联合概率：

$$
q_\theta(x_1,\ldots,x_T)=\prod_{t=1}^{T}q_\theta(x_t\mid x_{<t})
$$

最大化序列 likelihood，等价于最小化 token-level NLL 之和：

$$
\mathcal{L}_{\mathrm{NTP}}
=-\sum_{t=1}^{T}\log q_\theta(x_t\mid x_{<t})
$$

在实际 tensor 中，输入和 label 通常错开一个位置：位置 $t$ 的 logits 用来预测 $x_{t+1}$。BOS、EOS、padding、document boundary 和 chat role tokens 是否参与目标，由数据构造与 loss mask 决定，而不是由 Cross Entropy 公式自行决定。

```text
input_ids:  [BOS, x1, x2, x3]
labels:     [ x1, x2, x3, EOS]
                       ↑
              每个位置预测右侧 token
```

## Loss Mask 与训练阶段

设 $m_{b,t}\in\{0,1\}$ 表示 batch 中第 $b$ 条序列的第 $t$ 个 target token 是否参与训练，最常见的 token-average loss 为：

$$
L_{\mathrm{token}}
=\frac{\sum_{b,t}m_{b,t}L_{b,t}}
{\sum_{b,t}m_{b,t}}
$$

Cross Entropy 只定义单个 target distribution 的代价；**哪些位置参与 loss、不同位置如何聚合，同样是训练目标的一部分。**

- **Pre-training / Mid-training**：通常让普通文本中的大多数 target tokens 参与 loss，只屏蔽 padding、无效位置或特定控制 token。
- **SFT**：常让 system、user、tool observation 作为可见条件，但只对 assistant response、reasoning 或 action tokens 计算 loss。
- **Knowledge Distillation**：可以只在目标位置拟合 teacher distribution，也可以与 hard-label CE 组合。
- **Preference / RL training**：虽然底层仍会使用 token log-prob，但 policy objective 已不再等于普通 supervised Cross Entropy。

对 agent trajectory 而言，tool result 可以影响后续 action，却通常不是模型需要生成的 policy token。它可以进入 attention context，同时通过 mask 排除在 supervised loss 之外。mask 错误会把环境 observation 误当成模型行为学习。

## Reduction 决定样本权重

训练日志中的一个 `loss` 标量通常来自大量 token losses 的聚合，但聚合口径并不唯一。

### Token Mean

$$
L_{\mathrm{token}}
=\frac{\sum_{b,t}m_{b,t}L_{b,t}}
{\sum_{b,t}m_{b,t}}
$$

每个有效 token 权重相同，因此长样本对梯度的总贡献通常更大。这是语言模型预训练中最自然、最常见的口径。

### Sequence Mean

$$
L_{\mathrm{sequence}}
=\frac{1}{B}\sum_{b=1}^{B}
\frac{\sum_t m_{b,t}L_{b,t}}{\sum_t m_{b,t}}
$$

每条序列先各自求平均，再在 batch 内平均，因此长短样本获得相同的 sequence-level 权重。这可能用于某些 SFT 或 sequence-oriented objective，但与 token mean 的优化含义不同。

在 distributed data parallel training 中，如果各 rank 的有效 token 数不同，不能先计算 rank-local token mean 再对各 rank 等权平均，否则 token 较少的 rank 会被赋予过大权重。严格的 global token mean 应聚合全局 loss numerator 与有效 token denominator，或使用数学上等价的缩放方式。

因此，比较两个训练运行的 CE 时，需要同时确认：

- mask 是否一致；
- reduction 是 token mean、sequence mean 还是其他方式；
- gradient accumulation 和 DP 是否保持全局归一化；
- padding tokens 是否进入 denominator；
- 不同数据源或样本是否另有 sample weight。

## 数值稳定实现

直接先计算 $e^{z_j}$ 再做 Softmax，可能发生 overflow 或 underflow。实际实现通常在 log space 中完成计算：

$$
\operatorname{logsumexp}(z)
=m+\log\sum_j e^{z_j-m},\qquad m=\max_jz_j
$$

减去最大 logit 后，最大的指数项为 $e^0=1$，显著扩大了稳定数值范围。概念上，PyTorch 的：

```python
torch.nn.functional.cross_entropy(logits, labels)
```

等价于稳定的 `log_softmax + negative log likelihood`。具体框架和硬件还可能融合 LM Head、log-softmax 与 NLL，以减少中间 tensor materialization 和 memory traffic。

Mixed Precision 下，即使模型参数和 activation 使用 BF16/FP16，max、sum-exp 或 loss reduction 仍可能在更高精度中累加。Cross Entropy 的数学形式相同，kernel 的计算精度与 reduction 策略却会影响数值稳定性。

## Vocabulary-Parallel Cross Entropy

LLM 的 logits tensor 形状通常为 $[B,T,V]$。当 vocabulary 很大并使用 Tensor Parallel 时，LM Head 和 vocabulary logits 可以沿 $V$ 维分片，每个 TP rank 只持有 $V/P$ 个 logits。此时没有必要为了得到一个标量 loss 而 AllGather 完整的 $[B,T,V]$ tensor。

Vocabulary-parallel Cross Entropy 可以按以下方式计算：

```text
local vocabulary logits
  -> TP AllReduce(MAX) 得到每个 token position 的 global max
  -> 计算 local exp sum
  -> TP AllReduce(SUM) 得到 global sum-exp
  -> 持有 target id 的 rank 读取 target logit，其他 rank 置零
  -> TP AllReduce(SUM) 得到 global target logit
  -> loss = log(global sum-exp) - target logit
```

Backward 仍然使用 local shard 上的 $q_j-p_j$，因此 logits 和梯度可以保持 vocabulary-sharded。不同框架会融合其中的算子或 collective，但核心目的相同：避免 materialize 和通信完整 vocabulary logits。

## Softmax 平移不变性与 Logit Z-Loss

Softmax 满足：

$$
\operatorname{softmax}(z+c\mathbf{1})=\operatorname{softmax}(z)
$$

因此普通 Cross Entropy 无法识别所有 logits 的共同偏移。这里需要准确区分：

- **共同平移** $z\mapsto z+c\mathbf{1}$ 不改变概率；
- **尺度变化** $z\mapsto \alpha z$ 通常会改变分布尖锐程度和 Cross Entropy。

部分大模型训练配方会加入 Logit Z-Loss：

$$
L_z=\lambda\left(\log\sum_j e^{z_j}\right)^2
$$

它惩罚偏离零的 log-partition function，打破 CE 的共同平移不变性，并帮助控制 logits normalization constant。Z-Loss 是训练稳定性 regularization，不是语言建模目标本身；它也不应被笼统描述为“限制所有 logits 的绝对值”。详见 [[training/optimization/logit-z-loss|Logit Z-Loss]]。

## 与 Perplexity 的关系

当 $L$ 是使用自然对数计算的 token-average NLL 时：

$$
\mathrm{PPL}=e^L
$$

例如 $L=2$ 时，$\mathrm{PPL}\approx7.39$。它可以粗略理解为模型在平均意义下面对的有效候选数量，但不表示模型真的在 7.39 个等概率 token 中选择。

PPL 只有在 tokenizer、数据集、context construction、mask 和 reduction 一致时才适合比较。不同 tokenizer 会改变预测单位和 token 数量，不能仅凭 token-level PPL 判断模型优劣。

## 与 KL Divergence 的关系

Cross Entropy 可以分解为：

$$
H(p,q)=H(p)+D_{\mathrm{KL}}(p\Vert q)
$$

当数据分布 $p$ 固定时，$H(p)$ 与模型参数无关，因此：

$$
\underset{q}{\arg\min}\ H(p,q)
=
\underset{q}{\arg\min}\ D_{\mathrm{KL}}(p\Vert q)
$$

这解释了 Maximum Likelihood 为什么会推动模型分布逼近数据分布。训练样本上的 one-hot label 是对数据分布的一次 empirical observation，并不意味着自然语言在给定上下文后的真实 next-token distribution 本身是 one-hot。

## 数值示例

假设某位置只有三个候选 token，logits 为：

$$
z=[2,1,0]
$$

第一个 token 是真实 target。Softmax probability 约为：

$$
q=[0.665,0.245,0.090]
$$

该位置的 loss 为：

$$
L=-\log0.665\approx0.408
$$

对 logits 的梯度为：

$$
q-p=[-0.335,0.245,0.090]
$$

梯度下降会提高第一个 token 的相对 logit，并压低另外两个 token；第二个错误 token 当前概率更高，因此受到的抑制也更强。

## Cross Entropy 能说明什么

较低的 held-out Cross Entropy 通常表示模型对目标数据分布具有更好的 token prediction 能力，也是 language-model scaling 和训练稳定性分析中的核心指标。但它不直接衡量：

- 事实正确性；
- 长链 reasoning；
- instruction following；
- tool use 与多轮 Agent 成功率；
- safety、preference 或真实用户价值。

这些能力会受到数据分布、context、训练阶段、sampling、verifier 和系统 scaffold 的共同影响。因而：

$$
\text{lower CE}\not\Rightarrow\text{all downstream capabilities improve}
$$

在相同 tokenizer、数据分布、mask、reduction 和评测设置下，held-out CE 仍然是重要的基础指标；超出这些条件后，应结合 task evaluation，而不是单独比较 loss 数字。

## 常见误解

- **Cross Entropy 只涉及真实 token，其他 logits 不参与。** 最终形式虽然是 $-\log q_y$，但 $q_y$ 由整个 vocabulary 的 Softmax normalization 决定。
- **模型先显式生成完整 probability tensor，再计算 CE。** 模型产生 logits；训练 kernel 通常直接在 log space 计算，vocabulary-parallel implementation 甚至不会在单卡上聚合完整 logits。
- **CE 对 logits 的绝对尺度完全不敏感。** CE 对共同平移不敏感，但乘法缩放会改变概率分布。平移和尺度不是同一件事。
- **训练日志里的 CE 天然可比。** mask、tokenizer、reduction、数据集和 context 不同，loss 数值就可能不具备可比性。
- **CE 越低，生成质量一定越高。** CE 只直接衡量目标分布上的 token prediction，高层任务能力需要单独评测。

## 相关概念

- [[fundamentals/information-theory/entropy|Entropy]]：分布自身的不确定性。
- [[fundamentals/information-theory/negative-log-likelihood|Negative Log-Likelihood]]：观测样本上的负对数概率。
- [[fundamentals/information-theory/kl-divergence|KL Divergence]]：分布差异及 Cross Entropy 分解。
- [[fundamentals/information-theory/perplexity|Perplexity]]：token-average NLL 的指数形式。
- [[fundamentals/information-theory/label-smoothing|Label Smoothing]]：将 one-hot target 改为 soft target。
- [[fundamentals/probability/maximum-likelihood|Maximum Likelihood Estimation]]：最小化数据 NLL 对应的参数估计原则。
- [[fundamentals/neural-network-basics/logits-output-head|Logits and Output Head]]：hidden state 到 vocabulary logits 的映射。
- [[training/pretraining/objective|Training Objective]]：NTP、mask 和其他预训练目标。
- [[training/post-training/sft|SFT]]：assistant-targeted masked NTP。
- [[training/distributed-training/tensor-parallel|Tensor Parallel]]：vocabulary projection 与 Cross Entropy 的分片实现。
- [[training/optimization/logit-z-loss|Logit Z-Loss]]：对 log-partition function 的稳定性 regularization。
