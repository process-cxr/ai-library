---
title: Decoder-only Transformer
created: 2026-01-24
published: 2026-01-24
modified: 2026-05-31
type: topic
status: mature
area: architecture
tags:
  - transformer
  - decoder-only
---

Decoder-only Transformer 是当前大多数自回归大语言模型的基础架构形态。它只保留 Transformer decoder 的 causal self-attention 和 FFN 堆叠，不使用单独 encoder，也通常不使用 encoder-decoder 架构中的 cross-attention。模型把输入 prompt 和待生成文本统一看作一条 token 序列，并通过 [[training/pretraining/objective|Next-token Prediction]] 学习：

$$
p_\theta(x_1,\dots,x_T)=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t})
$$

这种结构的核心优势是训练目标、模型结构和推理过程高度一致：训练时每个位置预测下一个 token，推理时也逐 token 生成后续文本。

## 结构与数据流

给定 token 序列：

$$
x_1,\dots,x_T
$$

模型首先把 token 映射为 embeddings，并注入位置信息，例如 [[architecture/positional-encoding/rope|RoPE]]：

$$
H^{(0)} \in \mathbb{R}^{B \times T \times d_{model}}
$$

然后经过 $L$ 个 decoder-only Transformer blocks。现代 LLM 常见 block 可以抽象为 Pre-Norm 形式：

$$
\tilde{H}^{(\ell)} = H^{(\ell)} + \mathrm{CausalSelfAttn}(\mathrm{Norm}(H^{(\ell)}))
$$

$$
H^{(\ell+1)} = \tilde{H}^{(\ell)} + \mathrm{FFN}(\mathrm{Norm}(\tilde{H}^{(\ell)}))
$$

最后通过输出层得到每个位置的 logits：

$$
\mathrm{logits}_t = W_{out} h_t^{(L)}
$$

并对词表做 softmax 得到下一 token 分布。

## Causal Mask

Decoder-only Transformer 的关键约束是 causal mask。第 $t$ 个位置只能看到 $1,\dots,t$ 的 token，不能访问未来 token：

$$
S_{tj}=-\infty \quad \text{for } j>t
$$

softmax 后，未来位置的 attention weight 变为 0。这让模型在训练时可以并行计算所有位置的 loss，同时不泄漏未来答案。

直观地说：

```text
position 1: sees x1
position 2: sees x1, x2
position 3: sees x1, x2, x3
...
```

如果去掉 causal mask，模型会在训练时偷看目标 token 后面的上下文，得到不合法的语言建模目标。

## 与 Encoder-Decoder 的区别

原始 Transformer 用 encoder 读取源序列，用 decoder 生成目标序列，并通过 cross-attention 让 decoder 访问 encoder 输出。Decoder-only Transformer 去掉这个显式的输入/输出分离，把所有任务都组织成 prompt + continuation。

| 架构 | 输入读取 | 输出生成 | 常见用途 |
|---|---|---|---|
| Encoder-only | 双向 self-attention | 通常不是自回归生成 | 表示学习、分类、检索 |
| Encoder-Decoder | encoder 双向读取，decoder 自回归生成 | cross-attention 连接输入输出 | 翻译、摘要、seq2seq |
| Decoder-only | causal self-attention | next-token generation | LLM、对话、代码、工具调用 |

Decoder-only 的好处是统一、简单、可扩展；代价是所有输入理解、条件控制和输出格式都要通过同一条 token 序列表达。

## 为什么适合 LLM

Decoder-only Transformer 适合大模型，主要因为它把很多工程问题统一起来：

- **统一目标**：预训练、continued pretraining、SFT 都可以看成不同数据上的 token prediction。
- **统一接口**：分类、问答、翻译、代码生成、tool use 都可以转成 prompt 后续生成。
- **训练并行**：训练时所有 token 位置可以在 causal mask 下并行计算。
- **推理增量**：推理时可以用 [[inference/kv-cache-and-memory/kv-cache|KV Cache]] 复用历史 K/V。
- **扩展简单**：增加层数、宽度、数据和 token 数时，结构不需要为每个任务重写。

这也是 GPT、LLaMA、Qwen、DeepSeek 等模型家族大量采用 decoder-only 路线的原因。

## 与 Chat Template 的关系

Base decoder-only model 只学习 token continuation。要让它成为 assistant，后训练通常引入 [[training/post-training/chat-template|Chat Template]]，把多轮对话组织成带角色标记的序列：

```text
<system>...
<user>...
<assistant>...
```

结构上模型仍然是 decoder-only Transformer；变化的是数据格式、loss masking 和后训练目标。训练时如果只对 assistant token 计算 loss，user/system token 仍参与 attention，但不被直接预测。这说明 decoder-only 架构本身不区分“输入”和“输出”，这种区分来自数据模板和 loss mask。

## 推理链路

自回归推理过程是：

```text
prompt tokens
  -> prefill: compute hidden states and KV cache
  -> decode step 1: sample next token
  -> append token and update KV cache
  -> decode step 2
  -> ...
```

Prefill 阶段并行处理 prompt；decode 阶段每次只生成一个或少量新 token。由于历史 K/V 已缓存，新 token 只需计算自己的 Q/K/V，并读取历史 K/V 做 causal attention。

这带来一个重要系统后果：decoder-only LLM 的推理成本不仅由参数量决定，也由上下文长度、batch size、KV Cache、attention head 设计和 serving 调度决定。

## 复杂度与瓶颈

训练阶段，标准 full attention 的主要复杂度来自序列长度：

$$
O(T^2 d)
$$

其中 $T$ 是 sequence length，$d$ 是 hidden size 或 attention 相关维度。长上下文训练会显著增加 attention 计算和 activation memory。

推理阶段，decode 每一步只处理新 token，但需要读取历史 K/V，因此 KV Cache 显存近似随：

$$
L \times T \times n_{kv} \times d_h
$$

增长，其中 $L$ 是层数，$n_{kv}$ 是 KV heads 数，$d_h$ 是 head dimension。这解释了为什么 [[architecture/attention/grouped-query-attention|GQA]]、[[architecture/attention/multi-query-attention|MQA]]、[[architecture/attention/multi-head-latent-attention|MLA]] 和 PagedAttention 都与 decoder-only serving 强相关。

## 设计取舍

| 设计点 | 优势 | 代价 |
|---|---|---|
| Causal self-attention | 训练和生成目标一致 | 不能双向读取未来 token |
| Prompt + continuation | 任务接口统一 | 输入/输出边界依赖模板设计 |
| KV Cache | 避免重复计算历史 K/V | 长上下文和大 batch 显存压力大 |
| Decoder-only stack | 简洁、可扩展 | seq2seq 任务需要用文本格式模拟 |
| Next-token objective | 数据规模化容易 | 不直接优化偏好、安全或任务成功 |

## 常见误解

- **误解：decoder-only 只能生成，不能理解。** 它通过条件生成完成理解任务，例如分类可以组织成“给定文本，输出标签”。
- **误解：decoder-only 没有输入输出区分。** 结构上没有 encoder/decoder 区分，但训练数据、模板和 loss mask 可以定义哪些 token 是条件、哪些 token 是目标。
- **误解：训练时也是一个 token 一个 token 算。** 训练可以并行计算整段序列的所有位置，只是 causal mask 限制每个位置可见范围。
- **误解：KV Cache 只是小优化。** 对长上下文和高并发 serving，KV Cache 往往是核心显存瓶颈。

## 相关概念

- [[architecture/transformer/transformer|Transformer]]
- [[architecture/attention/self-attention|Self-Attention]]
- [[architecture/attention/multi-head-attention|Multi-Head Attention]]
- [[architecture/positional-encoding/rope|RoPE]]
- [[architecture/transformer/feed-forward|Feed Forward Network]]
- [[architecture/transformer/normalization|Normalization in Transformer]]
- [[training/pretraining/objective|Training Objective]]
- [[inference/kv-cache-and-memory/kv-cache|KV Cache]]

## 经典论文与资料

- [[sources/papers/2020-language-models-are-few-shot-learners|Language Models are Few-Shot Learners]]
- [[sources/papers/2017-attention-is-all-you-need|Attention Is All You Need]]
