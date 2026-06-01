---
title: YaRN
created: 2026-02-07
published: 2026-02-07
modified: 2026-05-31
type: topic
status: mature
area: architecture
tags:
  - positional-encoding
  - long-context
  - yarn
---

YaRN 是一种面向 [[architecture/positional-encoding/rope|RoPE]] 模型的长上下文扩展方法。它的目标是在不从零训练长上下文模型的情况下，通过调整 RoPE 的位置频率/缩放方式，并配合少量继续训练，使模型能够处理比原训练长度更长的上下文。

## 背景问题

RoPE 把位置 $m$ 转换成旋转角度 $m\theta_i$。如果模型训练时只见过最大长度 $L_{train}$，推理时直接扩展到远大于 $L_{train}$ 的位置，旋转角度会进入训练中没有充分覆盖的区域，可能导致 attention score 分布异常。

长上下文扩展要解决的问题是：

> 如何让 RoPE 在更长位置范围内保持相对稳定，同时尽量保留模型在原长度范围内的能力？

## Position Interpolation 直觉

一种基本思路是 position interpolation：把更长上下文中的位置压缩映射回模型熟悉的范围。例如目标长度是原训练长度的 $s$ 倍，可以把位置缩放为：

$$
m'=\frac{m}{s}
$$

然后用 $m'$ 计算 RoPE。这样远位置不会产生过大的旋转角度。

但简单插值也有代价：原本相邻 token 的位置间隔被压缩，局部位置分辨率可能下降。长上下文扩展必须在“远距离稳定”和“局部分辨率”之间折中。

## YaRN 的核心思想

YaRN 可以理解为更细致的 RoPE scaling 方案。它不是简单地对所有频率统一缩放，而是考虑不同频率维度对短距离和长距离的作用，试图保留局部位置能力，同时扩展长距离范围。

高层直觉：

- 高频维度对局部顺序敏感，不应过度破坏；
- 低频维度负责更长距离，可以更适合扩展；
- 缩放策略需要平滑过渡，而不是粗暴统一压缩；
- 位置扩展通常还需要继续训练来适配新位置分布。

这里不把 YaRN 写成单一公式，是因为具体实现会涉及 frequency ramp、scale factor、attention scaling 等细节。知识层面应抓住它的本质：**面向 RoPE 的长上下文频率重标定与继续训练方法**。

## 与 RoPE Scaling 的关系

常见 RoPE 长上下文路线包括：

- 修改 `rope_theta`；
- linear position interpolation；
- NTK-aware scaling；
- YaRN；
- 长上下文 continued pretraining。

这些方法都围绕同一个问题：训练时的 RoPE 频率范围与推理目标长度不一致。YaRN 属于更系统的缩放策略，目标是在较少额外训练下获得更好的长上下文适配。

## 适用边界

YaRN 解决的是位置编码扩展问题，不等于完整长上下文能力。即使位置机制可扩展，模型仍可能因为以下原因失败：

- 训练数据缺少长文档任务；
- attention 或 KV Cache 成本过高；
- 模型不擅长从远处检索关键信息；
- prompt 中噪声过多；
- 评测只测 needle retrieval，不能代表真实长程推理。

因此，使用 YaRN 后仍需要长上下文继续训练、评测和推理系统支持。

## 与训练和推理的关系

训练侧，YaRN 通常与 long-context continued pretraining 配合，让模型在新位置分布上调整 attention pattern。

推理侧，必须确保模型配置中的 RoPE scaling 参数和训练时一致。如果 serving 框架加载了错误的 scaling 配置，模型可能不会报 shape error，但长上下文输出质量会明显下降。

## 常见误解

- **误解：YaRN 是新的 attention 机制。** 它主要是 RoPE 长上下文扩展方法，不是替代 attention。
- **误解：套用 YaRN 就自动拥有强长文档理解。** 位置扩展只是基础，能力还取决于数据、训练和评测。
- **误解：所有频率统一缩放即可。** 简单缩放可能损害局部位置分辨率，YaRN 的价值在于更细致的缩放策略。
- **误解：只要训练能跑，推理配置无所谓。** RoPE scaling 配置必须在训练和推理中一致。

## 相关概念

- [[architecture/positional-encoding/rope|RoPE]]
- [[architecture/positional-encoding/positional-encoding|Positional Encoding]]
- [[training/mid-training/long-context-training|Long-context Training]]
- [[inference/kv-cache-and-memory/kv-cache|KV Cache]]
- [[architecture/attention/attention|Attention]]

## 经典论文与资料

- [[sources/papers/2023-yarn|YaRN]]
- [[sources/papers/2023-position-interpolation|Position Interpolation]]
- [[sources/papers/2024-longrope|LongRoPE]]
