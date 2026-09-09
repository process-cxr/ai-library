---
title: "Fundamentals"
created: 2025-12-06
published: 2025-12-06
modified: 2025-12-06
---

LLM 学习所需的数学、概率、信息论、优化和神经网络基础。这里不追求数学大全，只沉淀理解大模型主线所必需的概念。

## Sections

- [[fundamentals/linear-algebra/|Linear Algebra]] — 向量、矩阵、范数、矩阵乘法
- [[fundamentals/probability/|Probability]] — 随机变量、条件概率、概率分布、期望
- [[fundamentals/information-theory/|Information Theory]] — 熵、交叉熵、KL 散度、困惑度
- [[fundamentals/optimization/|Optimization]] — 梯度下降、反向传播、Adam、学习率
- [[fundamentals/neural-network-basics/|Neural Network Basics]] — Embedding、激活函数、归一化、Softmax
- [[fundamentals/representation-learning/|Representation Learning]] — 连续表示、离散 token、embedding space 和对比学习
- [[fundamentals/dynamical-systems/|Dynamical Systems]] — state、observation、transition 和时间演化
- [[fundamentals/control-and-planning/|Control and Planning]] — action、反馈、规划和 model-based decision

## How to Use

- 先读 `linear-algebra/` 和 `probability/`，建立数学与分布建模基础。
- 再读 `information-theory/`，理解语言模型训练目标与评估指标。
- 然后读 `optimization/`，理解参数如何被训练出来。
- 最后读 `neural-network-basics/`，连接到 [[architecture/transformer/transformer|Transformer]] 结构。
- 进入 Multimodal 前补读 `representation-learning/`，理解不同模态如何被编码并对齐。
- 进入 World Model 前补读 `dynamical-systems/` 和 `control-and-planning/`，理解静态预测与环境决策的区别。
