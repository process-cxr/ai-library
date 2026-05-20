---
title: Computational Graph and Automatic Differentiation
created: 2026-02-12
published: 2026-02-12
modified: 2026-02-19
type: topic
status: growing
area: fundamentals
tags:
  - optimization
  - autodiff
  - computational-graph
aliases:
  - Computational Graph
  - Autodiff
  - Automatic Differentiation
---

## 概念界定

计算图是把模型计算表示为节点和边的有向图，自动微分是深度学习框架基于计算图自动计算梯度的机制。它们是 PyTorch、JAX、TensorFlow 等框架执行反向传播的基础。

## 背景与问题

大模型由大量张量操作组成，如果手动推导和实现每个操作的梯度几乎不可行。计算图记录前向计算依赖关系，自动微分根据这些依赖关系自动组合局部梯度。

## 定义与记号

计算图中：

- 节点表示张量或操作。
- 边表示数据依赖。
- 前向传播按依赖顺序计算输出。
- 反向传播按反向拓扑顺序传播梯度。

例如：

```text
x, W -> matmul -> y -> loss -> L
```

反向时：

```text
L -> loss -> y -> matmul -> x, W
```

## 直观解释

计算图像一张“计算账本”，记录每个结果是从哪些输入算出来的。反向传播时，框架沿着这张账本反向追踪每个参数对 loss 的影响。

## 基本性质

- 动态计算图在运行时构建，PyTorch 常用这种方式。
- 静态计算图先定义后执行，便于编译优化。
- 自动微分不是数值差分，它使用链式法则精确组合局部导数。
- 前向激活是否保存会影响训练显存。

## 示例

PyTorch 中：

```text
loss.backward()
optimizer.step()
```

`backward()` 根据计算图计算梯度，`step()` 由优化器更新参数。二者职责不同。

## 常见误解

- 误解：自动微分就是用很小扰动估计导数。
  - 正确理解：自动微分通过链式法则组合解析局部导数，不是有限差分。
- 误解：计算图只在教学里有用。
  - 正确理解：显存优化、梯度检查点、编译优化都依赖计算图视角。
- 误解：调用 backward 后参数会自动更新。
  - 正确理解：backward 只计算梯度，参数更新由 optimizer 完成。

## 相关概念

- [[backpropagation|反向传播]] — 自动微分执行反向传播。
- [[gradient-descent|梯度下降]] — 优化器使用梯度更新参数。
- [[training/optimization/mixed-precision|混合精度训练]] — 计算图中的 dtype 与梯度缩放影响稳定性。
