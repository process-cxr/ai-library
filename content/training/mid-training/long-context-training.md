---
title: Long Context Training
created: 2026-02-28
published: 2026-02-28
modified: 2026-05-31
type: topic
status: mature
area: training
tags:
  - mid-training
  - long-context
---

Long Context Training 是让模型在比原始预训练更长的上下文窗口内保持建模、检索、推理和生成能力的训练阶段。它通常发生在预训练后期或 [[training/mid-training/continued-pretraining|Continued Pretraining]] 中，通过位置编码扩展、长序列数据构造和额外训练预算实现。

长上下文能力不是简单把 `max_position_embeddings` 改大。模型需要同时解决位置表示、attention 成本、长文档数据、训练稳定性和评测可信度问题。

## 目标

长上下文训练的目标可以分成几类：

- **Long-range retrieval**：在很长上下文中找到相关片段。
- **Long-document understanding**：理解长文章、论文、合同、书籍、代码仓库。
- **Multi-document synthesis**：跨多个文档整合信息。
- **Long-horizon generation**：生成长篇结构化文本或代码。
- **Agent memory / tool traces**：处理长交互历史和工具调用轨迹。

这些能力不完全相同。通过 needle-in-a-haystack 检索成功，不代表模型能做长文档推理；能处理长文档摘要，也不代表能稳定生成长代码仓库修改。

## 位置编码扩展

Transformer 需要位置表示。长上下文扩展通常要处理模型在训练时没有见过的位置范围。常见路线包括：

- **Position Interpolation**：将更长位置范围映射回原训练范围，降低外推难度。
- **RoPE scaling / NTK-aware scaling**：调整 RoPE 频率或缩放方式，以支持更长位置。
- **YaRN / LongRoPE 类方法**：对 RoPE 扩展做更细粒度的插值、重缩放或分段设计。
- **ALiBi / relative position bias**：使用更适合长度外推的位置偏置，但通常需要架构或训练阶段支持。

这些方法解决的是位置表示外推问题，不等于自动获得长上下文任务能力。模型仍需要长序列训练数据和足够优化步骤来适应新位置分布。

## 数据构造

长上下文训练数据不能只由短文档拼接而成。拼接能填满长度，但未必提供长程依赖信号。

高价值长上下文数据包括：

- 长文档：书籍、论文、法律合同、技术手册；
- 代码仓库：多文件依赖、README、issues、tests；
- 多文档 QA：问题答案需要跨文档检索；
- 长对话：多轮任务状态和偏好保持；
- 合成检索任务：needle、key-value retrieval、multi-hop retrieval；
- 长推理轨迹：数学证明、复杂代码修改、工具调用 trace。

数据构造要记录：

- sequence length 分布；
- 真实长文档比例；
- 拼接短文档比例；
- 文档边界 token；
- 是否允许跨文档 attention；
- long-range dependency 的距离分布；
- 目标任务是否需要中间位置、开头位置和末尾位置的信息。

## 训练成本

标准 attention 的时间和显存都包含与 sequence length 相关的二次项。更具体地说，attention score / probability 这类中间矩阵规模大致为：

$$
M_{\mathrm{attn\ scores}} \propto B \cdot H_{\mathrm{heads}} \cdot S^2
$$

而 attention 的矩阵乘法计算还会与每个 head 的维度相关，粗略可写为：

$$
C_{\mathrm{attn}} \propto B \cdot S^2 \cdot H_{\mathrm{model}}
$$

其中 $B$ 是 batch size，$S$ 是 sequence length，$H_{\mathrm{heads}}$ 是 attention heads 数，$H_{\mathrm{model}}$ 是 hidden size。即使使用 FlashAttention 等 kernel 避免显式 materialize 完整 attention matrix，长序列训练仍然会显著增加计算、activation memory 和通信压力。

实际训练常需要：

- 降低 micro-batch size；
- 使用 gradient checkpointing；
- 使用 sequence parallel 或 context parallel；
- 调整 learning rate 和 warmup；
- 使用更高效 attention kernel；
- 更稀疏或分段的长上下文数据 schedule。

长上下文训练应纳入 [[training/scaling/training-budget|Training Budget]] 和 [[training/optimization/training-memory-estimation|Training Memory Estimation]]，不能只看 token 数。

## Length Extrapolation 与 Position Generalization

如果模型原始训练长度为 $S_0$，目标长度为 $S_1>S_0$，直接外推通常有风险。位置编码可能在未见过的位置上产生不稳定表示，注意力模式也可能偏向局部或开头。

常见训练策略：

- 从较短扩展到中等长度，再扩展到目标长度；
- 混合短序列和长序列，避免短上下文能力下降；
- 在长序列中构造不同距离的监督信号；
- 使用 curriculum，让模型逐步适应更长位置；
- 在最后阶段用目标长度数据 annealing。

如果只训练最长长度，短上下文效率和能力可能下降；如果只用短文档拼接，模型可能学会位置存在，但没有学会长程依赖。

## 评测

长上下文评测要覆盖多种能力：

- needle-in-a-haystack retrieval；
- key-value retrieval；
- long document QA；
- multi-document QA；
- long summarization；
- code repository understanding；
- long conversation state tracking；
- lost-in-the-middle 测试；
- 不同位置、不同长度、不同干扰强度下的鲁棒性。

Needle 测试有用，但不能代表全部长上下文能力。模型可能学会检索显眼字符串，却无法进行跨段推理或综合多个证据。评测应报告随位置和长度变化的性能曲线，而不是只给单点最大上下文长度。

## 常见失败模式

- **只改位置编码不训练**：上下文窗口变大，但长程能力不可靠。
- **只拼接短文档**：长度利用率高，但缺少真实长程依赖。
- **只看 needle 结果**：忽略 long reasoning、summarization 和 multi-document synthesis。
- **短上下文能力退化**：长序列阶段改变数据分布或位置分布。
- **成本低估**：$S^2$ attention、activation 和通信成本被忽略。
- **中间位置退化**：模型能用开头和末尾信息，但 lost-in-the-middle 明显。

## 相关概念

- [[training/mid-training/continued-pretraining|Continued Pretraining]]
- [[training/data-engineering/packing|Packing]]
- [[training/pretraining/tokenizer|Tokenizer]]
- [[training/scaling/training-budget|Training Budget]]
- [[training/optimization/training-memory-estimation|Training Memory Estimation]]
- [[architecture/positional-encoding/rope|RoPE]]
- [[sources/papers/2023-position-interpolation|Position Interpolation]]
- [[sources/papers/2023-yarn|YaRN]]
- [[sources/papers/2024-longrope|LongRoPE]]
