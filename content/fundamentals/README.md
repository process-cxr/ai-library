# fundamentals

`fundamentals/` 负责沉淀理解大模型所需的基础概念。它采用二级结构：一级是基础领域，二级是该领域下的原子概念。

## 当前模块职责

- 建立理解大模型所需的数学、概率、信息论、优化和神经网络基础。
- 为 `../architecture/`、`../training/`、`../inference/` 提供前置知识。
- 控制粒度一致性，避免大主题和小概念长期平铺在同一级。

## 直接子项

- `index.md` — Fundamentals 网页入口与阅读顺序。
- `linear-algebra/` — 线性代数基础：向量、矩阵、范数、矩阵乘法。
- `probability/` — 概率论基础：随机变量、条件概率、概率分布、期望。
- `information-theory/` — 信息论基础：熵、交叉熵、KL 散度、困惑度。
- `optimization/` — 优化基础：梯度下降、反向传播、Adam、学习率。
- `neural-network-basics/` — 神经网络基础：Embedding、激活函数、归一化、Softmax。

## 文件契约

- 二级目录的 `index.md` 负责该基础领域的入口和概念索引。
- 二级目录下的普通 `.md` 文件负责一个相对原子的概念。
- 每篇概念笔记应说明直觉、定义、公式或机制、与 LLM 的关系、常见误解。
- 相关概念应通过 Quartz wikilink 互相连接。

## 边界

- 不写成泛数学百科，只保留服务大模型学习主线的基础内容。
- 模型结构放入 `../architecture/`。
- 训练方法和对齐算法放入 `../training/`。
- 推理优化和部署性能放入 `../inference/`。
- 应用系统、RAG、Agent 和评测放入 `../application/`。

## Direct-Call 信息

- 目录入口链接：`[[fundamentals/|Fundamentals]]`。
- 二级入口示例：`[[fundamentals/information-theory/|Information Theory]]`。
- 原子概念示例：`[[fundamentals/information-theory/cross-entropy|交叉熵]]`。
