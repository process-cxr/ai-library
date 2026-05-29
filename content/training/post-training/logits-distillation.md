---
title: Logits Distillation
created: 2026-03-08
published: 2026-03-08
modified: 2026-05-29
type: topic
status: mature
area: training
tags:
  - post-training
  - distillation
  - logits
---

Logits Distillation 是让 student model 学习 teacher model 在 token vocabulary 上输出分布的蒸馏方法。与普通 SFT 只学习一个 hard target token 不同，logits distillation 试图让 student 接近 teacher 对所有或部分候选 token 的相对概率判断。

在后训练中，logits distillation 可以用于把大模型的语言偏好、格式倾向、回答不确定性和多答案分布迁移给小模型。但由于大词表 logits 存储和计算成本很高，实际系统常使用 top-k logits、online teacher forward 或只在关键阶段使用 logits KD。

## 目标与问题

Hard-label SFT 对每个位置只告诉 student：“下一个 token 是这个”。但语言生成中经常有多个合理 token。例如中文回答中“因此”“所以”“由此可见”都可能合理。Teacher logits 提供了更丰富的信息：

- 哪些 token 是高概率替代；
- 哪些 token 明显不合适；
- teacher 对格式、风格和语义延续的偏好；
- 不确定性和多模态答案空间；
- 类别之间的相似性。

Logits distillation 的目标是让 student 分布 $p_\theta$ 接近 teacher 分布 $q_\phi$。

## Temperature

蒸馏常使用 temperature $T$ 软化分布：

$$
p_T(i) =
\frac{\exp(z_i/T)}
{\sum_j \exp(z_j/T)}
$$

其中 $z_i$ 是 logit。$T>1$ 会让分布更平滑，使非最大 token 的相对概率更可见；$T<1$ 会让分布更尖锐。

Temperature 的意义是调节 teacher 信号的信息密度：

- 太低：接近 hard label，蒸馏信息少；
- 适中：保留替代 token 的相对偏好；
- 太高：分布过平，噪声增加。

## KL Loss

常见 logits KD loss：

$$
\mathcal{L}_{\text{KD}}
= T^2
\sum_{t}
\mathrm{KL}
\left(
q_{\phi,T}(\cdot\mid x,y_{<t})
\|
p_{\theta,T}(\cdot\mid x,y_{<t})
\right)
$$

也可以写成 cross-entropy：

$$
\mathcal{L}_{\text{KD}}
= -T^2 \sum_t \sum_{v\in V}
q_{\phi,T}(v\mid h_t)
\log p_{\theta,T}(v\mid h_t)
$$

这里省略了 teacher distribution 的 entropy 项；在 teacher 固定时，该项与 student 参数无关，因此最小化 KL 与最小化 teacher soft targets 下的 cross-entropy 在优化方向上等价。

实践中常与 hard-label loss 混合：

$$
\mathcal{L}
= \alpha \mathcal{L}_{\text{KD}}
+ (1-\alpha)\mathcal{L}_{\text{CE}}
$$

这样既学习 teacher 分布，也保持目标序列的明确监督。

## Token-level 对齐

Logits distillation 通常要求 teacher 和 student 在同一上下文位置上输出分布。这里有几个重要前提：

- tokenizer 最好相同，否则 vocabulary 对齐困难；
- 输入 chat template 必须一致；
- teacher forcing 序列必须明确；
- loss mask 要决定哪些位置蒸馏；
- 对 tool outputs、user tokens、system tokens 是否蒸馏要谨慎。

如果 teacher 和 student tokenizer 不同，可以做 vocabulary mapping，但复杂度和误差都会上升。因此许多 practical KD 更偏向 sequence-level。

## Top-k / Top-p Logits

完整 vocabulary logits 很大。例如词表 100k、序列长度 4k、样本数百万时，保存完整 logits 几乎不可承受。常见压缩方式：

- 只保存 top-k logits；
- 保存 top-p token 集合；
- 保存 logit difference 或 normalized probabilities；
- online 蒸馏时不落盘 logits；
- 对关键 token 或 assistant tokens 才做 KD。

Top-k KD 的风险是遗漏 teacher 分布长尾信息，但通常比完整 logits 更可行。

## Offline 与 Online 实现

Offline logits KD：

1. teacher 对训练数据 forward；
2. 保存 logits 或 top-k logits；
3. student 读取缓存 logits 训练。

优点是可复现、teacher 不参与训练 loop；缺点是存储大、数据固定。

Online logits KD：

1. 每个 batch 同时跑 teacher 和 student；
2. 直接计算 KL；
3. 不保存 logits。

优点是节省存储、可以配合 on-policy；缺点是训练成本高，需要 teacher 常驻。

## 适用场景

Logits distillation 适合：

- teacher 和 student tokenizer 相同；
- 需要细粒度模仿 teacher 分布；
- student 容量足以吸收分布信息；
- 训练预算允许 teacher forward 或 logits 存储；
- 希望保留多答案不确定性，而不是只学一个 teacher sample。

在 instruction tuning 或 reasoning trace 场景中，sequence-level KD 更常见；在模型压缩和同族模型蒸馏中，logits KD 更有吸引力。

## 失败模式与边界

- **存储成本高**：完整 logits 几乎不可扩展到大规模数据。
- **Tokenizer mismatch**：不同词表导致分布对齐困难。
- **Teacher bias**：student 会模仿 teacher 的错误概率分布。
- **过度平滑**：temperature 过高会让 student 学到噪声。
- **容量不足**：小 student 无法匹配大 teacher 分布，KL 可能优化困难。
- **模板错配**：chat template 不一致会让同一 token 位置语义不同。

## 经典论文与资料

- [[sources/papers/2016-sequence-level-knowledge-distillation|Sequence-Level Knowledge Distillation]]
- [[sources/papers/2023-distilling-step-by-step|Distilling Step-by-Step]]

## 相关概念

- [[training/post-training/knowledge-distillation|Knowledge Distillation]]
- [[training/post-training/sequence-level-distillation|Sequence-level Distillation]]
- [[training/post-training/offline-kd|Offline KD]]
- [[fundamentals/information-theory/kl-divergence|KL Divergence]]
