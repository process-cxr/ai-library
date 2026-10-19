---
title: "Scaling Laws, Carefully"
created: 2026-10-19
published: 2026-10-19
modified: 2026-10-19
type: source
status: growing
area: sources
tags:
  - source
  - blog
  - language-model
  - foundation
  - pretraining
  - transformer
  - learning-dynamics
  - scaling-laws
aliases:
  - Scaling Laws, Carefully
source_url: https://lilianweng.github.io/posts/2026-06-24-scaling-laws/
source_type: personal-blog
publisher: Lil’Log
author: Lilian Weng
source_date: "2026-06-24"
---

# Scaling Laws, Carefully

## 基本信息

- 标题：[Scaling Laws, Carefully](https://lilianweng.github.io/posts/2026-06-24-scaling-laws/)
- 作者：Lilian Weng
- 来源：Lil’Log
- 日期：2026-06-24
- 类型：个人技术博客 / 研究综述
- 标签：Language Model、Foundation、Pretraining、Transformer、Learning Dynamics
- 阅读时间：约 25 分钟

Scaling laws 是深度学习中最重要的经验发现之一。它的形式很简单：当我们扩大模型规模 $N$、数据集规模 $D$ 和计算量 $C$ 时，训练损失 $L$ 会按照幂律曲线可预测地下降；在双对数图上，这条曲线表现为一条直线。我们可以把 scaling laws 看作描述计算量、损失、模型规模和数据之间关系的框架；其核心问题，是如何在 $N$ 和 $D$ 之间最优地分配宝贵的计算资源。

这种可预测性让 scaling laws 在实践中非常有价值。一种常见工作流是：在少量小规模运行上拟合 scaling laws，然后外推更大模型所需的 token 数和计算量。

| 符号            | 说明                                                                                                                                                                 |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| $N$             | 模型规模，以参数数量衡量。                                                                                                                                           |
| $D$             | 训练数据集规模，通常以 token 数量衡量。                                                                                                                              |
| $C$             | 训练计算量，以 FLOPs 衡量。一个有用的近似是 $C \approx 6ND$（[Kaplan et al. 2020](https://arxiv.org/abs/2001.08361)），其中 $2ND$ 对应前向传播，$4ND$ 对应反向传播。 |
| $E$             | 不可约损失（irreducible loss）。                                                                                                                                     |
| $L, \hat{L}(.)$ | 测试损失 / 测试损失预测函数；也可以指训练损失，因为二者高度相关。                                                                                                    |
| $\epsilon$      | 泛化误差。                                                                                                                                                           |

# 早期：ML 损失的可预测性（Early days: ML loss predictability）

在 scaling laws 成为主流概念之前，人们已经研究过泛化误差如何随规模变化而具有可预测性。

[Amari et al.（1992）](https://ieeexplore.ieee.org/document/6796972) 使用贝叶斯方法和 annealed approximation 推导了四种学习曲线：

1. 确定性学习算法、无噪声数据、唯一解：$\epsilon \sim c \cdot D^{-1}$，其中 $c$ 是某个常数。
2. 确定性学习算法、无噪声数据、多个等价解：$\epsilon \sim c \cdot D^{-2}$；随着每个新数据点到来，学习速度会更快，因为模型只需要学习参数的最优流形，而不必寻找单个解点。
3. 确定性学习算法、有噪声数据：$\epsilon \sim c \cdot D^{-1/2}$；数据中的噪声让学习更加困难。
4. 随机学习算法、有噪声数据：$\epsilon \sim c \cdot D^{-1} + E$；这里的不可约损失 $E$ 是随机学习器无法进一步降低的残余误差，例如模型在面对大量数据时容量耗尽的情况。

这四种学习曲线都遵循幂律：

$$
\epsilon \sim c \cdot D^\alpha + E
$$

其中 $E$ 可以为 0，而 $\alpha = -2, -1, -1/2$。虽然他们的理论设置基于一个简化的二分类任务，但这为构建经验性的 ML 损失预测模型指明了有用方向。

[Hestness et al.（2017）](https://arxiv.org/abs/1712.00409) 的一项早期经验研究解释了泛化误差、模型规模和数据之间的关系。对于给定的训练数据规模，他们通过网格搜索找出最合适的模型规模，然后绘制损失随训练数据集规模变化的曲线。在深度学习的四个不同领域（神经机器翻译、图像分类、语言建模和语音识别）中，他们反复观察到以下模式：

- 泛化误差会随着一组因素（例如数据规模）按照幂律变化。
- 模型改进会平移误差曲线，但似乎不会影响幂律指数。
- 有趣的是，架构会改变幂律拟合的偏移量（$E$），却不会改变指数（$\alpha$）。幂律的斜率似乎是问题领域的属性，而不是模型架构的属性。
- 拟合规模为 $D$ 的数据集所需的模型参数数量 $N$ 也遵循幂律变化。

![Deep-Speech-2（DS2）和 attention speech model 的学习曲线（左），以及不同规模 DS2 模型的学习曲线（右）。当训练数据变大时，小模型的损失会进入平台期。](https://lilianweng.github.io/posts/2026-06-24-scaling-laws/hestness-1.png)

_Deep-Speech-2（DS2）和 attention speech model 的学习曲线（左），以及不同规模 DS2 模型的学习曲线（右）。当训练数据变大时，小模型的损失会进入平台期。（图片来源：[Hestness et al. 2017](https://arxiv.org/abs/1712.00409)）_

一张概念图将学习曲线分成三个阶段。在小数据区域，学习信号不足，模型的表现只比随机猜测略好。在中间的“幂律区域”，我们观察到损失、数据和模型规模之间的幂律关系。最后的不可约误差区域，则可以归因于数据噪声等因素。

![幂律学习曲线阶段示意图](https://lilianweng.github.io/posts/2026-06-24-scaling-laws/hestness-2.png)

_幂律学习曲线阶段示意图。（图片来源：[Hestness et al. 2017](https://arxiv.org/abs/1712.00409)）_

[Rosenfeld et al.（2020）](https://arxiv.org/abs/1909.12673) 进一步尝试将误差建模为模型规模 $N$ 和数据规模 $D$ 的联合函数，覆盖多种架构（ResNet、WRN、LSTM、Transformer）和优化器（Adam、SGD 的变体）。他们在经验上观察到，当固定一个轴时，误差会随着另一个轴按照幂律衰减：

$$
\hat{L}(D,N) \approx \frac{A}{N^{\alpha}} + E_N,\quad
\hat{L}(D,N) \approx \frac{B}{D^{\beta}} + E_D
$$

这两个形式可以合并成联合形式：

$$
\hat{L}(D, N) \approx \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}} + E
$$

其中 $A > 0$、$B > 0$、$\alpha \geq 0$、$\beta \geq 0$ 是标量常数，而 $E$ 不依赖于 $N$ 或 $D$。

![数据规模、模型规模和泛化误差在 log-log-log 坐标中的 3D 等高线图。蓝点来自经验实验，曲面是蓝点之间的线性插值。](https://lilianweng.github.io/posts/2026-06-24-scaling-laws/rosenfeld-1.png)

_数据规模、模型规模和泛化误差在 log-log-log 坐标中的 3D 等高线图。蓝点来自经验实验，曲面是蓝点之间的线性插值。（图片来源：[Rosenfeld et al. 2020](https://arxiv.org/abs/1909.12673)）_

因此，他们可以构建一个简单参数函数形式的预测模型，令 $\boldsymbol{\theta} = \langle A, B, E, \alpha, \beta \rangle$，只在较小的训练配置 $(D, N)$ 上训练，就能预测超过某些阈值的 $(D, N)$ 所对应的预期损失。

![在小规模配置上拟合参数误差模型，并外推到更大的模型/数据区域：（a）实验设置示意图；（b）ImageNet、（c）WikiText-103 和（d）CIFAR100 上的实验结果。误差估计覆盖三种架构（WRN、VGG、DenseNet）和两种优化器（SGD、Adam）。](https://lilianweng.github.io/posts/2026-06-24-scaling-laws/rosenfeld-2.png)

_在小规模配置上拟合参数误差模型，并外推到更大的模型/数据区域：（a）实验设置示意图；（b）ImageNet、（c）WikiText-103 和（d）CIFAR100 上的实验结果。误差估计覆盖三种架构（WRN、VGG、DenseNet）和两种优化器（SGD、Adam）。（图片来源：[Rosenfeld et al. 2020](https://arxiv.org/abs/1909.12673)）_

补充一点：这些早期工作依赖 VC dimension 等经典学习理论直觉，将其作为容量的代理指标。VC dimension 是模型能够打散（shatter）的最大点集的基数。但在现代深度学习研究中，VC dimension 往往过于粗糙，无法解释实际行为；相比理论提供的最坏情况界，经验幂律反而更加干净且实用。

# 无限数据区域中的 Scaling Laws（Scaling Laws in Data-Infinite Region）

## Kaplan et al. 的 Scaling Laws

[Kaplan et al.（2020）](https://arxiv.org/abs/2001.08361) 在语言建模社区中普及了 scaling laws 的概念。他们发现，交叉熵测试损失 $L$ 会随着模型规模 $N$（不包括 embedding 层）、数据集规模 $D$ 和训练计算量 $C$，在多个数量级范围内分别按照幂律变化。这些发现与上一节的早期工作一致，但 Kaplan et al. 聚焦于 Transformer 语言模型，并通过更大规模的经验实验将概念形式化：模型规模从 768M 到 1.5B 个非 embedding 参数，数据集规模从 22M 到 23B tokens。论文中的所有训练运行都使用了这样的学习率调度：先进行 3000 步线性 warmup，然后余弦衰减到零。

关键发现如下：

- 损失 $L$ 分别随 $N$、$D$ 和 $C$ 按照幂律变化；要获得最优性能，三者必须同步扩展。
- 训练曲线遵循可预测的幂律，其参数大致独立于模型规模。
- 更大的模型具有更高的样本效率，也就是说，它们用更少的优化步数和更少的数据点就能达到给定损失。
- 架构细节（宽度、宽高比等）不如纯粹的规模重要。
- 训练损失和测试损失正相关。（这听起来很明显，但它是预训练工作的基础。另一方面，预训练损失的改善能否迁移到后训练评估，还需要单独研究。）
- 在固定计算预算下，训练一个非常大的模型并在**收敛之前**停止，比训练一个较小模型直到收敛更加高效。**这正是 Chinchilla scaling laws 与之不一致的地方：Kaplan et al. 拟合出的指数更大，因此高估了最优模型规模。**

他们用一个方程总结 $N$ 和 $D$ 的联合依赖关系：

$$
\hat{L}(N,D) = \left[ \left(\frac{a}{N}\right)^{\frac{\alpha}{\beta}} + \frac{b}{D} \right]^{\beta}
$$

这个形式有一个很好的推论：过拟合程度（即模型复杂或数据量较小时）主要取决于比值 $N^{\alpha / \beta} / D$。这说明，为了避免训练受数据限制，数据需要随着模型规模的增长按照特定比例增长。

![测试损失随计算量、数据集规模和参数量按幂律变化，覆盖多个数量级。](https://lilianweng.github.io/posts/2026-06-24-scaling-laws/kaplan-1.png)

_测试损失随计算量、数据集规模和参数量按幂律变化，覆盖多个数量级。（图片来源：[Kaplan et al. 2020](https://arxiv.org/abs/2001.08361)）_

最有影响力、事后看来也最有争议的结论，是计算最优分配。Kaplan et al. 得到 $N_\text{opt} \propto C^{0.73}$，并据此认为模型规模的增长速度应该快于数据规模。具体来说，当计算量增加 10 倍时，他们建议将模型规模扩大约 5.5 倍，而训练 token 只增加约 1.8 倍。之后的 Chinchilla 论文推翻了这一建议，认为它会让大模型严重**训练不足**。

Kaplan et al. 的另一项有用分析，是根据 $D$ 和 $N$ 近似估算所需的训练 FLOPs。每次乘加被计为约 2 FLOPs。

![不同 Transformer 架构组件的参数量和计算量估计。变量包括层数 $n_\text{layer}$、模型宽度 $d_\text{model}$（等于 $d_\text{embed}$，原表中的记号不一致）、前馈层维度 $d_\text{ff}$（通常等于 $4d_\text{model}$）、attention 维度 $d_\text{attn}$（通常等于 $d_\text{model}$）、上下文长度 $n_\text{ctx}$ 和词表大小 $n_\text{vocab}$。](https://lilianweng.github.io/posts/2026-06-24-scaling-laws/kaplan-2.png)

_不同 Transformer 架构组件的参数量和计算量估计。变量包括层数 $n_\text{layer}$、模型宽度 $d_\text{model}$（等于 $d_\text{embed}$，原表中的记号不一致）、前馈层维度 $d_\text{ff}$（通常等于 $4d_\text{model}$）、attention 维度 $d_\text{attn}$（通常等于 $d_\text{model}$）、上下文长度 $n_\text{ctx}$ 和词表大小 $n_\text{vocab}$。（图片来源：[Kaplan et al. 2020](https://arxiv.org/abs/2001.08361)）_

给定标准配置 $d_\text{attn} = d_\text{model} = d_\text{ff}/4$，并且将 embedding 层排除在 $N$ 和每 token 的前向计算之外：

$$
\begin{align}
N &= n_\text{layer} d_\text{model} 3 d_\text{attn} + n_\text{layer} d_\text{attn} d_\text{model} + n_\text{layer} 2 d_\text{model} d_\text{ff} && \small{\text{；不含 embedding 层}} \\
&= 2\;n_\text{layer} d_\text{model}(2d_\text{attn} + d_\text{ff}) \\
&= 12\;n_\text{layer} d_\text{model}^2 \\
\\
C_\text{fwd} &= 2 n_\text{layer} (d_\text{model} 3 d_\text{attn} + n_\text{ctx}d_\text{attn} + d_\text{attn}d_\text{embed} + 2 d_\text{model} d_\text{ff}) \\
&= 2 n_\text{layer} (12 d_\text{model}^2 + n_\text{ctx}d_\text{attn}) \\
&= 2N + 2 n_\text{layer}n_\text{ctx}d_\text{attn} \\
&\approx 2N \quad\quad \small{\text{；假设 }n_\text{ctx} < 12 d_\text{model}\text{，且 }n_\text{ctx}\text{ 项相对较小。}}
\end{align}
$$

然后，我们将反向传播 FLOPs 计为前向传播 FLOPs 的两倍，因为反向传播分别针对输入激活和权重的梯度运行两次矩阵乘法。因此，每个 token 的训练 FLOPs 总量约为 $6N$，在 $D$ 个 token 上训练的总 FLOPs 为 $C \approx 6ND$。

## Chinchilla Scaling Laws

Chinchilla 论文（[Hoffmann et al. 2022](https://arxiv.org/abs/2203.15556)）在更加谨慎的实验设计下，研究了固定计算预算 $C$ 时最优模型规模 $N$（包括 embedding 在内的总参数量）与 token 数 $D$ 之间的关系，并得出了与 Kaplan et al. 略有不同的答案。

![你应该知道 Chinchilla 长什么样 😊](https://lilianweng.github.io/posts/2026-06-24-scaling-laws/animal.png)

_你应该知道 Chinchilla 长什么样 😊（图片来源：ChatGPT 生成）_

核心问题是：在约束 $\text{FLOPs}(N, D) = C \approx 6ND$ 下，如何最优地分配资源。换句话说，当我们的 FLOPs 有限（给定数量的 GPU 运行给定时间）时，应该如何在更多数据 token 和更多模型参数之间做选择？

$$
N_\text{opt}(C), D_\text{opt}(C) = \operatorname*{arg\,min}_{\text{s.t. } \text{FLOPs}(N,D) = C} \hat{L}(N, D)
$$

Chinchilla 论文提出了三种设计严谨的 scaling laws 拟合方法。

经验实验扫描了 400 多个模型，规模从 70M 到超过 16B 参数，训练 token 从 5B 到 500B。实验假设每个训练 token 都是唯一的（无限数据区域）。所有运行都使用余弦学习率调度，在整个训练过程中衰减 10 倍。扫描模型规模可以描出计算最优前沿。

### 方法 1：固定模型规模，改变 token 预算（Method 1: Fix model sizes, vary the token budget）

对于每一个参数量 $N$，使用不同 token 预算训练多次运行，并记录每个 FLOP 预算 $C$ 下能够达到的最小损失。

![Chinchilla 方法 1：扫描不同模型规模时，训练损失随 FLOP 预算变化的曲线。](https://lilianweng.github.io/posts/2026-06-24-scaling-laws/chinchilla-1.png)

_Chinchilla 方法 1：扫描不同模型规模时，训练损失随 FLOP 预算变化的曲线。（图片来源：[Hoffmann et al. 2022](https://arxiv.org/abs/2203.15556)）_

### 方法 2：IsoFLOP 曲线（Method 2: IsoFLOP profiles）

固定计算预算 $C$，绘制最终损失随参数量 $N$ 变化的曲线。每条 iso-FLOP 曲线在对数空间中大致是一条抛物线，其最低点标记了该计算预算下的最优模型规模。随后在不同预算上重复这一过程，就能在图中描出一条幂律线。

![Chinchilla 方法 2：IsoFLOP 抛物线；每条曲线的最低点是该预算下的计算最优模型规模。](https://lilianweng.github.io/posts/2026-06-24-scaling-laws/chinchilla-2.png)

### 方法 3：参数拟合（Method 3: Parametric fit）

直接拟合 [Rosenfeld et al.（2020）](https://arxiv.org/abs/1909.12673) 使用的同一个参数函数：

$$
\hat{L}(N, D) = \frac{A}{N^\alpha} + \frac{B}{D^\beta} + E
$$

我们实际上可以在约束 $\text{FLOPs}(N,D) = C \approx 6ND$ 下最小化 $\hat{L}(N, D)$，从而得到最优 $N_\text{opt}(C), D_\text{opt}(C)$ 的闭式近似。

首先，将表达式化简为只包含 $N$：

$$
\begin{align}
\hat{L}(N) &= A N^{-\alpha} + B \Big(\frac{C}{6}\Big)^{-\beta}N^\beta + E \\
\hat{L}'(N) &= -\alpha A N^{-\alpha-1} + \beta B \Big(\frac{C}{6}\Big)^{-\beta} N^{\beta -1} = 0 && \small{\text{；关于 }N\text{ 的导数应为零。}} \\
\text{因此}\quad & \alpha A N^{-\alpha-1} = \beta B \Big(\frac{C}{6}\Big)^{-\beta} N^{\beta -1} \\
& \alpha A = \beta B \Big(\frac{C}{6}\Big)^{-\beta} N^{\alpha + \beta} \\
& N_\text{opt} = \Big(\frac{\alpha A}{\beta B}\Big)^{\frac{1}{\alpha + \beta}} \Big(\frac{C}{6}\Big)^{\frac{\beta}{\alpha+\beta}} \\
& D_\text{opt} = \frac{C}{6 N_\text{opt}} = \Big(\frac{\beta B}{\alpha A}\Big)^{\frac{1}{\alpha + \beta}} \Big(\frac{C}{6}\Big)^{\frac{\alpha}{\alpha+\beta}}
\end{align}
$$

当 $\alpha \approx \beta$ 时，模型规模和训练 token 应当以相同速率增长。

为了找到最优的 $\boldsymbol{\theta} = \langle A, B, E, \alpha, \beta\rangle$，Chinchilla 论文采用了 Huber loss（对异常值具有鲁棒性；$\delta=10^{-3}$）和 L-BFGS 算法（适合参数数量较少的曲线拟合）。

$$
\begin{align}
\min_{A,B,E,\alpha,\beta} \sum_{\text{runs }\{i\}} \text{Huber}_\delta (\log \hat{L}(N_i, D_i) - \log L_i) \\
\text{其中 }\text{Huber}_\delta (x) = \begin{cases}\frac{1}{2} x^2 & \text{当 }\vert x \vert \leq \delta \\
\delta \cdot (\vert x \vert - \frac{1}{2}\delta), & \text{否则。}\end{cases}
\end{align}
$$

Chinchilla 通过三种互补方法得到答案，并且三种方法的最终结果彼此一致，这也是结果相当有说服力的原因之一。

![三种方法都同意计算最优前沿为 $N_\text{opt} \propto C^{0.5}$，但与 Kaplan et al. 不一致。注意，方法 3 的结果与另外两种方法略有偏差，后文会解释原因。](https://lilianweng.github.io/posts/2026-06-24-scaling-laws/chinchilla-3.png)

![三种方法得到的 Chinchilla 预测，以及 Kaplan et al.（2020）的预测。三种方法都表明，当时的一些主流 LLM 训练不足。](https://lilianweng.github.io/posts/2026-06-24-scaling-laws/chinchilla-4.png)

_三种方法都同意计算最优前沿为 $N_\text{opt} \propto C^{0.5}$，但与 Kaplan et al. 不一致。注意，方法 3 的结果与另外两种方法略有偏差，后文会解释原因。（图片来源：[Hoffmann et al. 2022](https://arxiv.org/abs/2203.15556)）_

_三种方法得到的 Chinchilla 预测，以及 Kaplan et al.（2020）的预测。三种方法都表明，当时的一些主流 LLM 训练不足。（图片来源：[Hoffmann et al. 2022](https://arxiv.org/abs/2203.15556)）_

Chinchilla 论文关于大多数大模型（在当时，也就是约 2022 年）训练不足的结论，有一个著名的演示作为支持：在与 Gopher 相同的计算预算下（[Rae et al. 2021](https://arxiv.org/abs/2112.11446)；280B 参数、300B token 预算），他们训练了 Chinchilla（70B 参数、1.4T token 预算）。Chinchilla 的规模小了 4 倍，但训练 token 大约多了 4 倍，并且在各项指标上都超过了 Gopher。

## 调和 Kaplan 与 Chinchilla（Reconciling Kaplan and Chinchilla）

Chinchilla scaling laws 与 Kaplan et al. 的差异如下：

- 不是“让模型增长得比数据快”（$N_\text{opt} \propto C^{0.73}$），而是模型规模每翻倍时，训练 token 数也应当翻倍（$N_\text{opt} \propto C^{0.5}$）。
- 不是“训练一个大模型并在收敛前停止”，而是应该在更多数据上训练一个更小的模型。

两篇论文仍然认同同一个底层原则，但对于模型规模与 token 数之间的最优权衡位置存在分歧。为什么它们会有这么大的差异？

**差异 1：Kaplan et al. 主要在小模型上做实验。** Kaplan et al. 的实验主要覆盖较小模型，而 Chinchilla 论文的实验规模扩大了十倍以上。在对数—对数空间中外推时，拟合上的微小差异可能导致巨大的结果差异（参见[Toy simulation](#toy-simulation)）。

**差异 2：Embedding 参数量对小模型很重要。** 在小参数规模区域，embedding 参数占总参数的比例不可忽略，因此是否计入它们会产生影响。[Pearce & Song（2024）](https://arxiv.org/abs/2406.12907) 沿着这一方向进行了详尽分析。让我们用 $N_{\setminus E}, C_{\setminus E}$ 表示排除 embedding 时的模型规模和计算量，用 $N, C$ 表示包含全部参数时的数量。

- Kaplan et al.：$N^*_{\setminus E} \propto C^{0.73}_{\setminus E}$（不含 embedding）；
- Chinchilla：$N^* \propto C^{0.50}$（包含全部参数）。

为了连接二者，他们用某个常数 $\omega$ 拟合了总参数 $N_T$ 与非 embedding 参数 $N_{\setminus E}$ 之间的关系：

$$
N = N_{\setminus E} + \omega\, N_{\setminus E}^{1/3}。
$$

这个形式有一些很好的性质：它严格递增，并且 $\lim_{N \to \infty} N = N_{\setminus E}$（因为 $\frac{N}{N_{\setminus E}} = 1 + \omega {N_{\setminus E}}^{- \frac{2}{3}}$，且 $\lim_{N_{\setminus E} \to \infty} \frac{N}{N_{\setminus E}} = 1$）。

将它代入 Chinchilla laws 方程：

$$
\begin{align}
L(N_{\setminus E}, C_{\setminus E}) &= A(N_{\setminus E} + \omega\, N_{\setminus E}^{1/3})^{-\alpha} + B \Big(\frac{C_{\setminus E}}{6}\Big)^{-\beta} N_{\setminus E}^\beta + E \\
L'(N_{\setminus E}, C_{\setminus E}) &= - \alpha A (N_{\setminus E} + \omega N_{\setminus E}^{1/3})^{-\alpha -1}(1 + \frac{\omega}{3}N_{\setminus E}^{-2/3}) + \beta B \Big(\frac{C_{\setminus E}}{6}\Big)^{-\beta} N_{\setminus E}^{\beta -1} = 0 && \small{\text{；关于 }N_{\setminus E}\text{ 的导数应为零。}} \\
\text{整理得到 }& \alpha A (N^{*}_{\setminus E} + \omega {N^{*}_{\setminus E}}^{1/3})^{-\alpha -1}(1 + \frac{\omega}{3} {N^{*}_{\setminus E}}^{-2/3}) = \beta B \Big(\frac{C_{\setminus E}}{6}\Big)^{-\beta} {N^{*}_{\setminus E}}^{\beta -1} \\
& 6^{-\beta}\frac{\alpha A}{\beta B} ({N^{*}_{\setminus E}} + \omega {N^{*}_{\setminus E}}^{1/3})^{-\alpha -1}(1 + \frac{\omega}{3}{N^{*}_{\setminus E}}^{-2/3}) {N^{*}_{\setminus E}}^{1 - \beta} = C_{\setminus E}^{-\beta} \\
& 6 \Big(\frac{\beta B}{\alpha A}\Big)^{\frac{1}{\beta}} ({N^{*}_{\setminus E}} + \omega {N^{*}_{\setminus E}}^{1/3})^{\frac{1 + \alpha}{\beta}} ({N^{*}_{\setminus E}} + \frac{\omega}{3}{N^{*}_{\setminus E}}^{1/3})^{-\frac{1}{\beta}} {N^{*}_{\setminus E}} = C_{\setminus E}
\end{align}
$$

上式中的 $C_{\setminus E}$ 与 $N_{\setminus E}$ 的关系不再是干净的幂律。我们只能将它局部近似为 $N^*_{\setminus E} \overset{\propto}{\sim} C_{\setminus E}^g$，其中 $g$ 是基于一阶导数的局部指数（$\overset{\propto}{\sim}$），而不是全局幂律指数，因此 $g = \frac{\mathrm{d} \log C_{\setminus E}}{\mathrm{d} \log N_{\setminus E}}$。关于如何近似指数 $g$ 的完整细节，见 [Pearce & Song（2024）](https://arxiv.org/abs/2406.12907) 的附录 A.1。

![局部幂律指数 $g$ 随 $C_{\setminus E}$ 增长的可视化](https://lilianweng.github.io/posts/2026-06-24-scaling-laws/pearce-1.png)

_局部幂律指数 $g$ 随 $C_{\setminus E}$ 增长的可视化。（图片来源：[Pearce & Song 2024](https://arxiv.org/abs/2406.12907)）_

如上图所示，随着 $C_{\setminus E}$ 增大，$g$ 会收敛到 Chinchilla 的估计。通过使用上述方程生成合成训练曲线，并在 Kaplan et al. 的模型规模范围（768M 到 1.5B）内进行实验，他们估计该区域中的 $g$ 接近 Kaplan 的 0.73 系数。

## 为什么是幂律？（Why power law?）

幂律在 AI 之外的许多领域都被广泛观察到，例如 [Zipf 定律](https://en.wikipedia.org/wiki/Zipf%27s_law)、[无标度网络](https://en.wikipedia.org/wiki/Scale-free_network)、[城市规模定律](https://en.wikipedia.org/wiki/Urban_scaling)以及许多其他复杂系统。反复出现的模式是：大事件很少见，小事件很常见，而规模与频率之间的关系在双对数坐标上往往呈现一条直线。

**为什么 LLM 的 scaling laws 也呈现幂律形状？**

受到不同领域具有不同指数这一现象的启发（[Hestness et al. 2017](https://arxiv.org/abs/1712.00409)），[Sharma & Kaplan（2020）](https://arxiv.org/abs/2004.10802) 提出了一种早期解释：语言建模可以被看作在一个低维数据流形上进行回归。更多模型参数可以对数据流形进行更细的划分，从而降低泛化误差。最简单地说，如果有效规模为 $N$ 的模型将一个 $d$ 维流形划分成 $O(N)$ 个区域，那么典型的线性分辨率会按照 $\sim N^{-1/d}$ 变化。这与上面的 scaling laws 具有相似的幂律形式。这个理论最适用于无限数据、欠拟合的区域，但在现实中，估计数据流形的内在维度非常困难。

后来的一个假设（[Michaud et al. 2023](https://arxiv.org/abs/2303.13506)、[Brill 2024](https://arxiv.org/abs/2412.07942)）认为，知识或技能是以离散块（“量化的”）形式学习的，而这些技能的频率分布遵循幂律。模型先学习常见技能，之后学习罕见技能，从而使损失呈现平滑的幂律衰减。

这里只列出了两种假设，但还有更多研究尝试通过数据的谱尾、核特征值、自然语言统计或训练动力学中的相变来解释幂律 scaling 的形状。

# 数据受限区域中的 Scaling Laws（Scaling Laws in Data-Limited Region）

经典 scaling laws 假设实际上拥有**无限的唯一数据**，没有重复，也没有多 epoch 训练。随着模型规模显著增长，我们正在耗尽足够多的高质量唯一 token。事实上，关于 AI 的 scaling 还能持续多久的一些论证，正是围绕我们是否正在撞上“数据墙”展开的。

还需要强调的是，$D$ 背后的数据集应当已经被清洗。预训练数据管线往往是有效预训练管线的重要组成部分，常见步骤包括去重（精确和模糊）、质量过滤、模板文本移除、安全过滤、PII/版权遮蔽、benchmark 去污染，以及根据语言、质量、内容类型等因素仔细重新加权数据混合组件。即使两个数据集包含相同数量 $D$ 的 token，高质量数据集与由互联网垃圾组成的数据集也可能产生截然不同的计算效率。

[Hernandez et al.（2022）](https://arxiv.org/abs/2205.10487) 研究了一个受控版本：数据集大部分唯一，只有一小部分数据重复。从一个大型数据集开始，他们保留 90% 的非重复数据，将剩余 10% 替换成原始数据中极小一部分的重复数据。通过训练 Transformer 模型 100B 个 token，他们观察到 double-descent 现象：随着重复数据权重增加，测试损失实际上会先**变差**，然后再次变好；重复比例越高，这一效应越明显。

![随着重复比例增加，测试损失出现 double-descent（左侧重复 90%，右侧重复 50%）](https://lilianweng.github.io/posts/2026-06-24-scaling-laws/hernandez-1.png)

_随着重复比例增加，测试损失出现 double-descent（左侧重复 90%，右侧重复 50%）。（图片来源：[Hernandez et al. 2022](https://arxiv.org/abs/2205.10487)）_

训练中间阶段的平坦或上升趋势，可能是由重复数据的记忆化导致的。这样的学习曲线会让 scaling law 拟合变得不准确。他们还得出结论：重复数据会损害一部分 OOD 评估和下游微调。不过，他们的数据混合是在更像实验室的设置中构造的，而现实世界中的重复通常更加复杂（例如，不同数据具有不同程度的重复、语义重复等）。

与其说数据重复会损害训练，我们更感兴趣的是：当唯一的高质量数据并非无限、训练中很可能必须重复数据时，应该如何拟合 scaling laws。

[Muennighoff et al.（2023）](https://arxiv.org/abs/2305.16264) 研究了在模型训练受数据约束时，应该如何最优分配计算量。具体来说，他们通过约 400 个实验，经验性地研究了数据重复的影响；模型规模从 10M 到 9B，数据规模最高 900B tokens，最多训练 1500 个 epoch。每个 epoch 都重复完全相同的数据集，在 epoch 之间打乱，并在留出的测试集上评估。

关键的建模调整，是将 token 总量 $D$ 分解为两部分：（i）唯一 token 数量 $U_D$；（ii）重复次数 $R_D$（即 epoch 数减 1）。因此有 $D = U_D(1 + R_D)$。给定唯一数据预算 $D_\text{uniq}$，按定义 $U_D = \min \{{ D_\text{uniq}, D\}}$，$R_D = (D / U_D) - 1$。他们使用 Chinchilla scaling laws 找到拟合 $U_D$ 的最优模型规模 $U_N$，并通过重复定义额外模型规模 $R_N = (N / U_N) - 1$。

随后，他们更新 Chinchilla 的参数拟合（[方法 3](#方法-3参数拟合method-3-parametric-fit)），用有效的（折扣后的）数据 $D'$ 和模型规模 $N'$ 替代原始量：

$$
\hat{L}(N, D) = \frac{A}{N'^\alpha} + \frac{B}{D'^\beta} + E
\quad\text{其中}\quad
D' = U_D + U_D\, r_D\left(1 - \exp\!\left(-\frac{R_D}{r_D}\right)\right)。
$$

直觉是，一个 token 的价值会随着重复而**指数衰减**。在他们的模型中，每次重复都会让 token 剩余价值损失 $(1 - 1/r_D)$，其中 $r_D$ 是可学习的“半衰期”参数。当 $R_D = 0$ 或 $R_D \ll r_D$ 时，可以恢复 $D' \approx D$。

对额外模型规模也采用对称形式：$N' = U_N + U_N r_N(1 - \exp(-R_N / r_N))$。它表达了“更大的模型在重复数据上更快过拟合”和“模型可能对于数据集来说过大”的想法。这一部分不太直观，我没有找到令人满意的解释，说明为什么模型规模需要以和重复数据类似的对称形式出现。后来的 [Lovelace et al.（2026）](https://arxiv.org/abs/2605.01640) 改变了这一假设。

他们的经验拟合发现，**额外参数的价值比重复数据衰减得更快**，即 $r_N < r_D$。因此，与增加模型参数相比，我们应该把更多资源分配给更多 epoch。这种建模的一个弱点是，正如作者指出的，它会显著低估失败模型的最终测试损失（也就是训练中途损失上升的模型），例如训练 44 个 epoch 的模型。

![重复条件下的数据受限 scaling 比不考虑数据的拟合更好地捕捉实验结果；重复 token 的价值向上限指数衰减。随着 epoch 增加，拟合变差，因为高重复会导致测试损失在训练中途上升，图中未显示这一点。](https://lilianweng.github.io/posts/2026-06-24-scaling-laws/muennighoff-1.png)

_重复条件下的数据受限 scaling 比不考虑数据的拟合更好地捕捉实验结果；重复 token 的价值向上限指数衰减。随着 epoch 增加，拟合变差，因为高重复会导致测试损失在训练中途上升，图中未显示这一点。（图片来源：[Muennighoff et al. 2023](https://arxiv.org/abs/2305.16264)）_

最近，[Lovelace et al.（2026）](https://arxiv.org/abs/2605.01640) 用不同方法重新研究了同一个问题。他们没有把过参数化建模为有效模型规模的边际收益递减，而是显式建模模型规模与数据重复之间的交互。经验上，他们训练了约 300 个模型，规模从 15M 到 1B 参数，唯一 token 从 50M 到 6B。

当他们绘制固定模型规模在不同数据重复程度下的拟合残差时，观察很直观：epoch 越多，损害越大；有趣的是，**更大的模型对重复更加敏感**。这暗示损失惩罚很可能同时取决于模型规模和数据规模。

![有效规模拟合的残差表明，过拟合损害会随着 epoch 数和模型规模同时增长。](https://lilianweng.github.io/posts/2026-06-24-scaling-laws/lovelace-1.png)

_有效规模拟合的残差表明，过拟合损害会随着 epoch 数和模型规模同时增长。（图片来源：[Lovelace et al. 2026](https://arxiv.org/abs/2605.01640)）_

他们引入了一个显式的过拟合惩罚项，并围绕**容量比** $N / U_D$（参数量相对于唯一 token 数量）构建：

$$
\hat{L}(N, U_D, R_D) = E + \frac{A}{N^\alpha} + \frac{B}{\big(U_D (1 + R_D)\big)^\beta} + \color{red}{P \cdot R_D^\delta \cdot \left(\frac{N}{U_D}\right)^\kappa}
$$

其中：

- $R_D$ 是重复次数；
- 标量 $P$ 是可学习参数；
- 指数 $\kappa$（第 2 个可学习参数）使惩罚能够随容量比 $N / U_D$ 非线性变化；
- 重复次数上的独立指数 $\delta$（第 3 个可学习参数）将重复的非线性与 $\kappa$ 解耦。

新增的项（红色部分）是一个直接的过拟合惩罚，它会随着数据重复次数增加，以及模型相对于可用唯一数据的过参数化程度增加而增长。

他们还研究了 weight decay 如何影响数据受限条件下的训练，发现较强的 weight decay 能够降低数据重复造成的过拟合惩罚。

![较强的 weight decay 能够降低数据重复带来的过拟合惩罚。](https://lilianweng.github.io/posts/2026-06-24-scaling-laws/lovelace-2.png)

_较强的 weight decay 能够降低数据重复带来的过拟合惩罚。（图片来源：[Lovelace et al. 2026](https://arxiv.org/abs/2605.01640)）_

Muennighoff et al. 和 Lovelace et al. 的两种建模方法都建立在经验曲线拟合之上，因此我们仍然不清楚，为什么数据受限的 scaling laws 应该恰好具有这些形式，以及为什么每个自由参数都是必要的。我很期待看到这条线上的更多理论工作。

# 现实中拟合 Scaling Laws 的棘手之处（Trickiness of Fitting Scaling Laws in Reality）

尽管 scaling law 的形式很简洁，但在实践中，拟合过程可能对看似琐碎的流程选择异常敏感，例如如何计算参数量、如何进行精度舍入、如何对损失求和或取平均等。

因为 scaling law 只能在我们负担得起训练的、相对较小且相对便宜的模型上拟合，而预测却要**外推**到大几个数量级的模型。在这种设置下，看起来像舍入误差的选择，也可能导致预测出现巨大差异。

与此同时，scaling-law 拟合假设唯一变化的因素是**规模**，这意味着模型架构、优化器、学习率调度、batch ramp、数据混合、tokenizer 和其他设计选择都应该保持不变。另一个潜在假设是，所有这些设置都应该被仔细调优，因为欠训练模型等情况可能导致不同结论。

Kaplan et al. 和 Chinchilla 的结果分歧，就是展示 scaling laws 拟合棘手性的一个例子。

第二个例子，是一项后续分析，它研究了为什么 Chinchilla [方法 3](#方法-3参数拟合method-3-parametric-fit) 与另外两种方法略有偏差。[Besiroglu et al.（2024）](https://arxiv.org/abs/2404.10102) 从 Hoffmann et al.（2022）的 Figure 4 中提取原始的 $(N, D, L)$ 数据点，并重新运行方法 3 的参数拟合。他们发现了几个具体问题：

- L-BFGS-B 最小化器中的损失尺度过高，这是因为对每个样例的 Huber loss 取平均而不是求和，导致优化过早终止。原始拟合和 bootstrap 中损失最小化的提前停止，产生了不一致的估计和不合理地窄的置信区间。
- 报告的 $\alpha$ 和 $\beta$ 被舍入到 2 位精度，这使得推导出的 $A, B$ 看起来比实际偏差更大。

## Toy simulation

这里有一个由 ChatGPT 创建的玩具模拟器，用来展示三种具体的失败模式。

我们假设真实函数为：

$$
\hat{L}(N, D) = 482.01 \cdot N^{-0.3478} + 2085.43 D^{-0.3658} + 1.8172
$$

因此 $N_\text{opt} \propto C^{0.5126}$，$D_\text{opt} \propto C^{0.4874}$。这是 [Besiroglu et al.（2024）](https://arxiv.org/abs/2404.10102) 的估计。

模拟器绘制损失预测 $\hat{L}$ 随数据集规模 $D$ 的变化，同时提供一组滑块来展示以下情况：

- **损失精度**：将损失从高位小数舍入到低位小数，可能改变拟合出的参数值。
- **损失噪声**：只用千分之一损失（0.001）量级的乘法扰动损失值，也会导致不同的拟合结果。
- **拟合区域敏感性**：只拟合小模型、只拟合中等模型，或拟合全部模型，会得到不同的表面 scaling laws。

原文此处提供一个交互式模拟器，包含以下可见控件：拟合区域（仅小模型 / 仅中等模型 / 所有测量模型）、损失精度、损失噪声、外推目标，以及拟合后的 $\alpha$、$\beta$、$E$、$N_\text{opt}$ 指数、$D_\text{opt}$ 指数、预测损失、真实损失和误差读数。由于该控件依赖原文页面中的 JavaScript，本文保留其数学设定和控件语义，不在静态 Markdown 页面中重复嵌入脚本。

# 引用（Citation）

请按如下方式引用本文：

> Weng, Lilian. “Scaling Laws, Carefully”. Lil’Log (Jun 2026). https://lilianweng.github.io/posts/2026-06-24-scaling-laws/

或者使用 BibTeX：

```bibtex
@article{weng2026scaling,
  title = {Scaling Laws, Carefully},
  author = {Weng, Lilian},
  journal = {lilianweng.github.io},
  year = {2026},
  month = {June},
  url = "https://lilianweng.github.io/posts/2026-06-24-scaling-laws/"
}
```

# 参考文献（References）

[1] S. Amari, N. Fujita, and S. Shinomoto. [“Four Types of Learning Curves.”](https://ieeexplore.ieee.org/document/6796972) _Neural Computation_, 4(4):605–618, 1992。

[2] Hestness et al. [“Deep Learning Scaling is Predictable, Empirically.”](https://arxiv.org/abs/1712.00409) arXiv preprint arXiv:1712.00409, 2017。

[3] Rosenfeld et al. [“A Constructive Prediction of the Generalization Error Across Scales.”](https://arxiv.org/abs/1909.12673) ICLR 2020。

[4] Kaplan et al. [“Scaling Laws for Neural Language Models.”](https://arxiv.org/abs/2001.08361) arXiv preprint arXiv:2001.08361, 2020。

[5] Hoffmann et al. [“Training Compute-Optimal Large Language Models.”](https://arxiv.org/abs/2203.15556) NeurIPS 2022。

[6] Pearce and Song. [“Reconciling Kaplan and Chinchilla Scaling Laws.”](https://arxiv.org/abs/2406.12907) TMLR 2024。

[7] Bahri et al. [“Explaining Neural Scaling Laws.”](https://arxiv.org/abs/2102.06701) arXiv preprint arXiv:2102.06701, 2021。

[8] Sharma and Kaplan. [“A Neural Scaling Law from the Dimension of the Data Manifold.”](https://arxiv.org/abs/2004.10802) arXiv preprint arXiv:2004.10802, 2020。

[9] Hernandez et al. [“Scaling Laws and Interpretability of Learning from Repeated Data.”](https://arxiv.org/abs/2205.10487) arXiv preprint arXiv:2205.10487, 2022。

[10] Muennighoff et al. [“Scaling Data-Constrained Language Models.”](https://arxiv.org/abs/2305.16264) NeurIPS 2023。

[11] Lovelace et al. [“Prescriptive Scaling Laws for Data Constrained Training.”](https://arxiv.org/abs/2605.01640) arXiv preprint arXiv:2605.01640, 2026。

[12] Besiroglu et al. [“Chinchilla Scaling: A Replication Attempt.”](https://arxiv.org/abs/2404.10102) arXiv preprint arXiv:2404.10102, 2024。

[13] Michaud et al. [“The Quantization Model of Neural Scaling.”](https://arxiv.org/abs/2303.13506) NeurIPS 2023。

[14] Brill. [“Neural Scaling Laws Rooted in the Data Distribution.”](https://arxiv.org/abs/2412.07942) arXiv preprint arXiv:2412.07942, 2024。

[15] Rae et al. [“Scaling Language Models: Methods, Analysis & Insights from Training Gopher.”](https://arxiv.org/abs/2112.11446) arXiv preprint arXiv:2112.11446, 2021。
