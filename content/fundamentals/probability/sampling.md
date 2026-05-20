---
title: 采样
type: topic
status: growing
area: fundamentals
tags:
  - math
  - probability
  - decoding
aliases:
  - Sampling
---

## 概念界定

采样是从概率分布中随机抽取具体取值的过程。在语言模型推理中，采样表示根据模型给出的下一个 token 概率分布选择实际输出 token。

## 背景与问题

语言模型每一步输出的是词表上的概率分布，而不是直接输出唯一 token。如何从这个分布得到最终文本，是解码策略的问题。采样让生成具有多样性，但也会引入随机性和不稳定性。

## 定义与记号

如果模型给出分布：

```text
p_θ(X_t | x_<t)
```

采样就是从这个分布中抽取：

```text
x_t ~ p_θ(X_t | x_<t)
```

然后把 `x_t` 接到上下文后继续下一步生成。

## 直观解释

采样类似按概率抽签。概率高的 token 更容易被选中，但概率低的 token 也有机会出现。这和贪心解码不同，贪心解码每次都选择概率最高的 token。

## 基本性质

- 采样会增加生成多样性。
- 采样结果具有随机性，同一 prompt 多次生成可能不同。
- 分布越平坦，采样结果越不确定。
- 分布越尖锐，采样结果越接近贪心选择。

## 示例

假设模型输出：

```text
p(Paris) = 0.70
p(Lyon) = 0.10
p(Marseille) = 0.05
p(other) = 0.15
```

采样通常会选中 `Paris`，但仍有一定概率选中其他 token。

## 常见误解

- 误解：采样就是让模型乱说。
  - 正确理解：采样仍然受模型分布约束，只是允许非最高概率 token 被选中。
- 误解：不用采样就一定更准确。
  - 正确理解：贪心解码更稳定，但可能导致重复、保守或缺乏创造性。
- 误解：随机性来自模型参数。
  - 正确理解：在同一输入下，前向计算通常确定；随机性主要来自采样过程。

## 相关概念

- [[fundamentals/probability/distribution|概率分布]] — 采样的对象。
- [[fundamentals/probability/monte-carlo|蒙特卡罗方法]] — 用采样近似期望、概率和指标。
- [[fundamentals/probability/temperature-topk-topp|Temperature、Top-k 与 Top-p]] — 常见采样控制方法。
- [[inference/decoding/speculative-decoding|投机解码]] — 推理加速中的解码相关技术。
