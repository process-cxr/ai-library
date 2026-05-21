---
title: Bayes Rule
created: 2025-12-20
published: 2025-12-20
modified: 2025-12-20
type: topic
status: seed
area: fundamentals
tags:
  - math
  - probability
  - bayes
aliases:
  - Bayes Rule
  - Bayes Theorem
---

## 概念界定

贝叶斯公式描述如何在观察到证据后更新某个假设的概率。它把先验、似然、证据和后验连接起来，是理解概率推断和不确定性更新的基础。

## 定义与记号

贝叶斯公式：

```text
p(H | E) = p(E | H) p(H) / p(E)
```

其中：

- `H`：假设。
- `E`：观察到的证据。
- `p(H)`：先验概率。
- `p(E | H)`：似然。
- `p(H | E)`：后验概率。
- `p(E)`：证据概率或归一化常数。

## 直观解释

贝叶斯公式表达的是：看到新证据后，应该如何调整对某个假设的相信程度。如果某个证据在假设成立时更容易出现，那么观察到该证据后，这个假设的后验概率会上升。

## 示例

在分类任务中，可以把类别看作假设 `Y`，输入文本看作证据 `X`：

```text
p(Y | X) ∝ p(X | Y) p(Y)
```

现代神经网络通常不显式这样分解，但贝叶斯公式仍然是理解条件概率和概率推断的重要背景。

## 相关概念

- [[fundamentals/probability/conditional-probability|条件概率]] — 贝叶斯公式建立在条件概率定义上。
- [[fundamentals/probability/joint-marginal|联合分布与边缘分布]] — 证据概率来自边缘化。
- [[fundamentals/probability/calibration-uncertainty|校准与不确定性]] — 后验概率与置信度解释相关。
