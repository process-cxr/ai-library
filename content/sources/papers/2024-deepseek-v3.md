---
title: "DeepSeek-V3 Technical Report"
created: 2026-06-01
published: 2026-08-31
modified: 2026-08-31
type: source
status: processed
source_type: paper
area: sources
tags:
  - source
  - paper
  - deepseek
  - moe
  - mla
  - fp8
  - multi-token-prediction
  - pretraining
source_url: https://arxiv.org/abs/2412.19437
paper_date: "2024-12"
paper_order: "19437"
---

# DeepSeek-V3 Technical Report

## 基本信息

- 标题：[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- 版本：arXiv:2412.19437v2，2025-02-18
- 作者：DeepSeek-AI
- 研究对象：大规模 decoder-only Mixture-of-Experts language model
- 模型规模：671B total parameters，37B activated parameters per token
- 预训练数据：14.8T tokens
- 训练硬件：2,048 张 NVIDIA H800 GPU
- 相关 topic：[[architecture/model-families/deepseek|DeepSeek]]，[[architecture/sparse-and-efficient/moe|Mixture of Experts]]，[[architecture/attention/multi-head-latent-attention|Multi-Head Latent Attention]]，[[training/optimization/mixed-precision|Mixed Precision]]，[[training/distributed-training/pipeline-parallel|Pipeline Parallel]]，[[training/pretraining/data-mix|Data Mix]]，[[training/pretraining/objective|Training Objective]]

这篇技术报告的价值，不在于单独提出一个新的网络模块，而在于展示了一套大规模 MoE 模型如何把架构、路由、低精度训练、分布式通信、数据配方、长上下文扩展和后训练组织成一个可运行的整体。DeepSeek-V3 继承 DeepSeek-V2 的 MLA 与 DeepSeekMoE，在此基础上加入 auxiliary-loss-free load balancing 和 Multi-Token Prediction，并通过 FP8 mixed precision、DualPipe 与定制化 cross-node all-to-all 把 671B total parameters 的训练成本压到可接受范围。

报告给出的主线可以概括为：

> 大容量 MoE + 每 token 稀疏激活 + 低 KV Cache + 通信与计算重叠 + 低精度训练

这不是“671B 参数自然带来更强能力”的单因素故事。模型规模、14.8T 高质量 tokens、MLA / DeepSeekMoE、FP8 数值策略、专家通信系统和后续 reasoning distillation 共同决定了最终结果。阅读这篇报告时，最值得抓住的是这种算法、系统和数据之间的 co-design 关系。

## 研究问题

### 如何扩大模型容量而不按 dense 模型的成本增长

Dense Transformer 中，每个 token 都会经过全部 FFN 参数。模型规模扩大时，计算、显存和通信通常同步增加。MoE 提供了另一条路径：保留大量 total parameters，但只为每个 token 激活少数 experts。

DeepSeek-V3 的设计目标是同时获得：

- 更大的总容量；
- 较低的 per-token computation；
- 更高的多领域和多语言建模能力；
- 可在大规模集群上完成的训练吞吐；
- 可实际部署的 inference cost。

因此，671B 与 37B 必须同时看：671B 描述模型总容量、权重存储和专家集合规模，37B 更接近每个 token 的主要计算路径。37B 并不表示部署只需要保存 37B 参数。

### 如何处理 MoE 的负载均衡代价

MoE 路由的困难不是选择 top-k 这么简单。若 router 总把 token 发给少数 experts，会导致：

- 热门 expert 超载，拖慢整个 step；
- 冷门 expert 得不到足够训练信号；
- cross-node all-to-all 流量不均；
- capacity 溢出和 token dropping；
- router 为了追求均衡而牺牲内容相关性。

传统 MoE 通常用 auxiliary load-balancing loss 约束 router，但 auxiliary loss 过强会直接干扰主任务。DeepSeek-V3 的核心路由贡献，是把用于选择 expert 的负载均衡控制与参与反向传播的内容相关 gating score 分开。

### 如何让跨节点专家并行真正跑起来

当 experts 分布在多个节点时，一个 token 的路由会触发 dispatch 和 combine 通信。如果计算和通信串行，MoE 的稀疏计算收益很容易被 network overhead 吃掉。DeepSeek-V3 需要解决：

- 如何让 cross-node communication 与 attention / MLP 计算重叠；
- 如何减少 pipeline bubbles；
- 如何利用 IB 与 NVLink 的不同带宽；
- 如何让 fine-grained experts 跨节点部署；
- 如何在不使用 tensor parallelism 的情况下装下并训练模型。

### 低精度训练能否支持 trillion-token 规模

FP8 可以降低 GEMM 的计算、激活存储和通信成本，但其动态范围与累加精度不足，容易受到 outlier、underflow、overflow 和梯度异常影响。报告需要回答哪些算子可以使用 FP8、哪些路径必须保留 BF16 / FP32，以及如何设计 quantization scale 和高精度 accumulation。

## 核心主张

1. **DeepSeek-V3 是一个 671B total / 37B active 的大规模 MoE language model。** 它用 MLA 降低 inference KV Cache，用 DeepSeekMoE 扩大容量并控制 FFN 计算。
2. **Auxiliary-loss-free load balancing 可以在较少损害模型性能的情况下平衡 expert load。** 路由 bias 只用于 top-k 选择，不参与最终 gating value 的计算。
3. **Multi-Token Prediction 可以在不增加主模型 inference 成本的情况下改善 base model 表现。** 训练时增加一个额外未来 token 的预测深度，推理时可以丢弃 MTP module，或将其用于 speculative decoding。
4. **FP8 mixed precision 可以在极大规模 MoE 训练中工作。** 通过 tile-wise / block-wise quantization、周期性高精度 accumulation 和 selective high-precision operators，作者在两个规模的 MoE 对照中观察到相对 BF16 的 loss error 小于 0.25%。
5. **MoE 的主要瓶颈是系统级通信，而不是单纯的 FLOPs。** 16-way PP、64-way EP、ZeRO-1 DP、DualPipe 和定制化 all-to-all kernel 共同把通信尽量隐藏在计算之后。
6. **数据构造与训练 schedule 仍然是最终能力的重要来源。** DeepSeek-V3 使用 14.8T tokens、128K tokenizer、FIM、数学和代码增强的数据配方，并通过两阶段 long-context extension 获得 128K context。
7. **后训练中的 reasoning distillation 对数学和代码能力贡献明显。** 作者从 DeepSeek-R1 系列构造 reasoning data，再结合 expert model、rejection sampling 和 RL，在 accuracy、reasoning pattern、格式和输出长度之间做取舍。

这些结论都来自特定模型、数据和系统设置。训练成本、benchmark 和 ablation 主要由作者在内部框架中报告，不能直接视为独立复现实验结论。最终 chat model 的提升同时受到模型规模、数据规模、架构、训练系统、SFT、RL 和 distillation 的共同影响。

## 整体方法

DeepSeek-V3 的系统可以按以下链路理解：

~~~text
DeepSeekMoE + MLA + MTP
        |
        v
16-way PP + 64-way EP + ZeRO-1 DP
        |
        v
DualPipe + IB/NVLink all-to-all overlap
        |
        v
FP8 GEMM + fine-grained quantization + FP32 accumulation
        |
        v
14.8T-token pre-training + 4K -> 32K -> 128K extension
        |
        v
SFT + rule/model-based RM + GRPO + R1 reasoning distillation
~~~

每一层都对应一个瓶颈：

- MLA 主要处理 attention 的 KV Cache；
- DeepSeekMoE 主要处理模型容量与 FFN 计算；
- auxiliary-loss-free routing 主要处理专家负载和 specialization；
- DualPipe 与 all-to-all kernel 主要处理跨设备通信；
- FP8 主要处理计算吞吐、激活显存和通信带宽；
- data / schedule 主要决定预训练信息覆盖与优化轨迹；
- SFT / RL / distillation 主要把 base model 能力转成可用的 chat 和 reasoning 行为。

## Architecture

### 基本配置

DeepSeek-V3 仍然是标准 Transformer 框架，但将绝大多数 FFN 替换为 DeepSeekMoE，并在 attention 中使用 MLA：

| 配置 | 数值 |
|---|---:|
| Transformer layers | 61 |
| Hidden dimension | 7,168 |
| Attention heads | 128 |
| Per-head dimension | 128 |
| KV compression dimension $d_c$ | 512 |
| Query compression dimension $d'_c$ | 1,536 |
| Decoupled RoPE dimension per head | 64 |
| Shared experts per MoE layer | 1 |
| Routed experts per MoE layer | 256 |
| Activated routed experts per token | 8 |
| Maximum nodes per token | 4 |
| MTP depth | 1 |
| Total parameters | 671B |
| Activated parameters | 37B |

除最初三个 Transformer layers 外，其余 FFN 都使用 MoE layer。每个 MoE layer 包含一个 shared expert 和 256 个 routed experts，每个 token 选择 8 个 routed experts。

### Multi-Head Latent Attention

MLA 的核心是对 key / value 做 low-rank joint compression。对于第 $t$ 个 token：

$$
c_t^{KV}=W_D^{KV}h_t
$$

然后从低维 latent 中恢复 content key 和 value：

$$
k_t^C=W_U^Kc_t^{KV}, \qquad v_t^C=W_U^Vc_t^{KV}
$$

位置相关信息单独通过 decoupled RoPE 路径生成：

$$
k_t^R=RoPE(W_K^Rh_t)
$$

最终 key 将 content part 与 RoPE part 拼接：$k_{t,i}=[k_{t,i}^C;k_t^R]$。query 也采用 low-rank compression，再与 RoPE query 拼接。

MLA 的关键推理收益是，生成过程中不需要保存每个 head 的完整 K/V，只需要缓存 $c_t^{KV}$ 与 $k_t^R$。因此 KV Cache 的增长维度明显小于标准 MHA，同时保留多头 attention 的表达路径。

MLA 与 MoE 的分工不同：

- MoE 主要改变 FFN 的容量与 per-token computation；
- MLA 主要改变 attention 的历史状态存储与读取；
- 二者组合后，模型既能扩大总容量，又能控制长上下文 inference 的 KV Cache。

更完整的公式与实现分析见 [[architecture/attention/multi-head-latent-attention|Multi-Head Latent Attention]]。

### DeepSeekMoE

设 $u_t$ 为 FFN 输入，DeepSeekMoE 的输出由 shared experts 和 routed experts 两部分组成：

$$
h'_t=u_t+\sum_i FFN_i^{shared}(u_t)+\sum_i g_{i,t}FFN_i^{routed}(u_t)
$$

DeepSeek-V3 使用 sigmoid 计算 affinity：

$$
s_{i,t}=Sigmoid(u_t^T e_i)
$$

然后在 top-$K_r$ routed experts 内归一化 gating value。shared expert 对所有 token 生效，routed expert 则由 token-level router 选择。

这个结构的直觉是：通用变换交给 shared expert，内容相关且更细粒度的模式交给 routed experts。它既避免所有知识都必须共享，也避免所有基础能力都完全依赖动态路由。

### Auxiliary-loss-free load balancing

传统做法会把 load-balancing auxiliary loss 加到总训练目标里。DeepSeek-V3 改为为每个 routed expert 维护一个 bias $b_i$，只在 top-k routing 时使用：

$$
g'_{i,t}=
\begin{cases}
s_{i,t}, & s_{i,t}+b_i \in TopK(\{s_{j,t}+b_j\},K_r) \\
0, & otherwise
\end{cases}
$$

关键点是：

1. $s_{i,t}+b_i$ 用于决定哪个 expert 入选；
2. 入选后真正乘在 expert output 上的 gating value 仍来自原始 $s_{i,t}$；
3. bias 不通过主 loss 反向传播；
4. 每个 training step 结束时，根据 batch-level expert load 更新 bias；
5. overloaded expert 的 bias 减少 $\gamma$，underloaded expert 的 bias 增加 $\gamma$。

这种做法把系统负载均衡与模型内容选择部分解耦。router 仍能依据 affinity score 表达 specialization，bias 则提供外部的动态 load correction。

### Sequence-wise balance loss 与 node-limited routing

作者并没有完全删除所有 balance signal。为防止某个 sequence 内出现极端失衡，仍保留很小的 sequence-wise balance loss：

$$
L_{Bal}=\alpha \sum_i f_iP_i
$$

DeepSeek-V3 使用的 $\alpha=0.0001$ 很小，它的作用更接近防止单条 sequence 内的极端情况，而不是承担主要的全局负载均衡任务。

同时，node-limited routing 规定每个 token 最多发送到 $M=4$ 个节点。每个节点根据本节点 experts 的最高 affinity scores 计算节点级得分，再选择有限的目标节点。这降低了 IB traffic，并使 fine-grained experts 可以跨节点部署。

在这种 balance strategy 下，训练和 inference 都不进行 token dropping。训练阶段依赖稳定的 expert load；推理阶段则通过 redundant experts 和 expert placement 处理不同请求分布带来的局部热点。

## Multi-Token Prediction

### 为什么在 NTP 之外增加 MTP

标准 NTP 只预测当前位置的下一个 token。MTP 的动机有两个：

- 在相同输入序列上提供更密集的未来预测信号；
- 让 hidden representation 提前编码未来 token 的信息，形成一定的 pre-planning。

DeepSeek-V3 使用 sequential MTP modules，保留每个预测深度的完整 causal chain，而不是使用彼此独立的多个 output heads。

第 $k$ 个 MTP module 包含与主模型共享的 embedding layer、一个 Transformer block、projection matrix $M_k$ 和共享 output head。它把上一深度的 representation 与未来 token embedding 拼接，再预测更远位置的 token：

$$
h_i^{\prime k}=M_k[Norm(h_i^{k-1});Norm(Emb(t_{i+k}))]
$$

不同预测深度的 cross-entropy loss 平均后乘以权重 $\lambda$：

$$
L_{MTP}=\lambda\frac{1}{D}\sum_{k=1}^{D}L_{MTP}^k
$$

DeepSeek-V3 的 MTP depth $D=1$，即在主模型 next-token prediction 外，再预测一个额外未来 token。$\lambda$ 在前 10T tokens 中为 0.3，剩余 4.8T tokens 中为 0.1。

推理时 MTP module 可以直接丢弃，主模型 inference cost 不增加；也可以重新利用为 speculative decoding 的 draft predictor。报告中第二个 token 的 acceptance rate 约为 85%-90%，并据此报告约 1.8x 的 TPS 提升。

## Training Infrastructure

### 集群与并行组合

DeepSeek-V3 在 2,048 张 H800 GPU 上训练。单节点内 GPU 通过 NVLink / NVSwitch 连接，节点间使用 InfiniBand。

训练框架是从头构建的 HAI-LLM，整体并行组合为：

| 并行维度 | 配置 | 作用 |
|---|---:|---|
| Pipeline Parallelism | 16-way | 按层切分 Transformer |
| Expert Parallelism | 64-way，跨 8 nodes | 分布 routed experts |
| Data Parallelism | ZeRO-1 | 切分 optimizer states |
| Tensor Parallelism | 训练阶段不使用 | 通过其他系统优化降低内存压力 |

训练阶段没有依赖 tensor parallelism，而是用 PP、EP、ZeRO-1、activation recomputation、FP8 storage 和通信优化组合解决模型规模问题。这不表示 TP 没有价值，而是说明并行策略应根据模型结构和通信拓扑共同选择。

### DualPipe

cross-node expert parallelism 的 communication 与 computation ratio 约为 1:1。若按传统 pipeline schedule 串行执行，MoE 的 all-to-all 会明显拖慢训练。

DualPipe 把一个 forward / backward chunk 拆成 attention、all-to-all dispatch、MLP、all-to-all combine、backward for input、backward for weights 和 PP communication。随后把一组 forward chunk 与 backward chunk 重新排布，让 computation、dispatch / combine 和 PP communication 互相重叠；完整 schedule 采用 bidirectional pipeline，同时从 pipeline 两端输入 micro-batches。

它的效果是：

- 大部分 all-to-all communication 可以隐藏在 computation 中；
- PP communication 也可以被重叠；
- pipeline bubble 相比 1F1B / ZB1P 更小；
- fine-grained experts 可以跨节点部署，而不让通信成为线性额外开销。

DualPipe 需要保存两份 model parameters，但作者认为在较大 EP size 下，这部分额外内存相对可控；其 peak activation memory 约为 $2\times PP+1$ 的量级，而 1F1B 是 $1\times PP$，换取了更好的通信重叠。

### Cross-node all-to-all

训练集群内 NVLink 带宽约 160 GB/s，InfiniBand 带宽约 50 GB/s，NVLink 约为 IB 的 3.2 倍。因此 token 先通过 IB 发送到目标节点中相同 in-node index 的 GPU，再在节点内通过 NVLink 转发到真正承载 target expert 的 GPU。

每个 token 最多发送到 4 个节点，平均每个节点承载约 3.2 个 experts。虽然 DeepSeek-V3 实际只选择 8 个 routed experts，但按最多 4 个节点、每节点约 3.2 个 experts 的通信限制，理论上可以把可选 expert 数扩展到约 13 个而不显著增加节点间通信成本。

作者还使用 warp specialization、20 个 SM 划分出的 10 个 communication channels、customized PTX、auto-tuned communication chunk size，以及 communication stream 与 computation stream overlap。最终报告称只用约 20 个 SM 就可以充分利用 IB 与 NVLink 带宽，但这也意味着通信仍会占用原本可执行模型计算的 GPU 资源。

### Memory saving

训练内存优化包括：

- backward 时 recompute RMSNorm 与 MLA up-projection；
- 在 CPU 中异步维护 model parameter 的 EMA；
- 通过 physical sharing 复用 MTP 与主模型的 embedding / output head；
- 用 FP8 缓存部分 activation；
- 在分布式 DP ranks 间分片保存高精度 master weights 与 optimizer states。

这些优化共同解释了为什么模型可以在不使用 training TP 的情况下运行。单个优化都不是决定性的，整体收益来自多个小幅节省叠加。

## FP8 Mixed Precision Training

### Mixed precision policy

报告没有把所有 tensor 都强制变成 FP8，而是将算子按数值敏感性区分：

- Linear 的 Fprop、Dgrad 和 Wgrad GEMM 使用 FP8；
- embedding、output head、MoE gating、normalization 和 attention 保持 BF16 或 FP32；
- master weights、weight gradients 和关键 optimizer state 保留更高精度；
- MoE dispatch 前的部分 activation 使用 FP8，combine 保持 BF16。

这体现了 FP8 训练的核心原则：compute-dense GEMM 使用低精度，决定数值稳定性的路径保留高精度。

### Fine-grained quantization

普通 per-tensor scaling 容易受 activation outliers 影响。DeepSeek-V3 使用：

- activation：每个 token、每 128 个 channel 做 1x128 tile-wise scaling；
- weight：每 128 个 input channel、每 128 个 output channel 做 128x128 block-wise scaling；
- scale 在当前 tile / block 上在线计算，而不是只依赖历史 maximum。

H800 Tensor Core 的 FP8 GEMM accumulation 有效精度约 14 bits。作者每累计 128 个 FP8 乘积，就把 partial result promotion 到 CUDA Core 的 FP32 registers 中继续累加；选择 $N_C=128$，相当于每 4 个 WGMMA 做一次 promotion，在精度与吞吐之间取得折中。

作者还统一使用 E4M3，而不是在 Fprop、Dgrad、Wgrad 之间切换 E4M3 / E5M2。原因是 fine-grained scaling 已经部分缓解了动态范围不足，E4M3 的更多 mantissa bits 可以提供更高精度。

### Activation、optimizer 和 communication storage

- AdamW 的 first / second moments 使用 BF16；
- master weights 和用于 batch accumulation 的 gradients 使用 FP32；
- attention 后 Linear 的输入使用定制 E5M6；
- MoE SwiGLU 输入用 FP8 缓存，并在 backward 中重算 output；
- MoE up-projection 前的 activation 用 FP8 进行 dispatch；
- activation gradient 在 down-projection 前也采用类似压缩；
- forward / backward combine 保持 BF16。

### FP8 对照与失败模式

作者在两个 baseline MoE 上对比 FP8 和 BF16：约 16B total parameters、1.33T tokens；约 230B total parameters、0.9T tokens。在高精度 accumulation 和 fine-grained quantization 下，relative loss error 始终低于 0.25%。

但如果 activation gradients 也简单使用 128x128 block-wise quantization，约 16B MoE 在训练约 300B tokens 后会 divergence。作者推测，activation gradients 在不同 token 间高度不均衡，形成 token-correlated outliers，普通 block-wise scale 无法处理。

因此，FP8 不是统一 cast 到 FP8，而是对不同张量、不同方向和不同算子分别设计 quantization granularity 与 accumulation path。

## Inference 与 Deployment

### Prefill

Prefill 的最小部署单元是 4 nodes、32 GPUs：

- attention 使用 TP4 + Sequence Parallelism；
- attention 组合 DP8；
- MoE 使用 EP32；
- shallow layers 的 dense MLP 使用 1-way TP；
- 使用与训练类似的 IB -> NVLink token dispatch；
- 通过两个 workload 相近的 micro-batches 重叠 attention、MoE、dispatch 和 combine。

为了处理不同输入分布造成的 expert hotspots，系统会根据 online service 统计周期性复制高负载 experts。报告中的部署设置使用 32 个 redundant experts，每张 GPU 在原有 experts 外再承载一个额外 redundant expert。

### Decoding

Decoding 的最小部署单元是 40 nodes、320 GPUs：

- attention 使用 TP4 + Sequence Parallelism + DP80；
- MoE 使用 EP320；
- 每个 GPU 通常承载一个 expert；
- 64 GPUs 负责 redundant experts 与 shared experts；
- dispatch / combine 直接通过 IB point-to-point transfer；
- 使用 IBGDA 降低 latency。

Decoding 阶段把 shared expert 当作 routed expert 看待，因此每个 token 实际选择 9 个 experts。由于 decoding 每个 expert 的 batch 通常不超过 256 tokens，瓶颈更多是 memory access 而不是 arithmetic throughput，因此通信与 MoE 计算只分配少量 SM，避免影响 attention。

### Deployment trade-off

MoE 的 active parameters 较低并不意味着 deployment unit 可以很小。推理仍需要承担 total experts 的存储、expert dispatch、跨节点通信和 redundant expert placement。论文也承认，模型虽然训练成本高效，但小团队部署仍有较高门槛。

## Pre-Training

### Data Construction

DeepSeek-V3 的 14.8T-token corpus 相比 DeepSeek-V2 做了几类调整：

- 提高数学与 programming samples 的比例；
- 扩大 English / Chinese 之外的 multilingual coverage；
- 优化数据处理流程，减少 redundancy，同时保留 diversity；
- 使用 document packing 保持文档完整性；
- packing 时不加入 cross-sample attention masking；
- 采用 Fill-in-Middle，帮助代码与文档中的中间内容预测。

FIM 使用 Prefix-Suffix-Middle 格式：

~~~text
<|fim_begin|> f_pre <|fim_hole|> f_suf <|fim_end|> f_middle <|eos_token|>
~~~

FIM rate 为 0.1，并在 document-level、pre-packing 阶段应用。其目标是让模型学会根据 prefix 和 suffix 共同恢复 middle，而不牺牲标准 next-token prediction。

### Tokenizer

DeepSeek-V3 使用 byte-level BPE，vocabulary size 为 128K。tokenizer 与 pretokenizer 针对 multilingual compression 做了调整，并引入了组合 punctuation 与 line break 的 token。

这种组合 token 会带来 token boundary bias：如果 few-shot prompt 的多行文本没有以 terminal line break 结束，模型可能遇到训练中较少出现的边界形式。作者通过在训练中随机拆分一部分组合 token，让模型接触更多边界情况，以缓解这一问题。

### Model 与 optimizer hyperparameters

预训练使用 AdamW，$\beta_1=0.9$、$\beta_2=0.95$、weight decay = 0.1、maximum sequence length = 4K、gradient clipping norm = 1.0，共训练 14.8T tokens。

学习率 schedule 分为多个阶段：

1. 前 2K steps 从 0 线性 warm up 到 $2.2\times10^{-4}$；
2. 随后保持 $2.2\times10^{-4}$，直到消耗 10T tokens；
3. 接下来 4.3T tokens 按 cosine decay 降到 $2.2\times10^{-5}$；
4. 最后 500B tokens 中，前 333B 保持 $2.2\times10^{-5}$；
5. 最后 167B 使用 $7.3\times10^{-6}$。

batch size 在前 469B tokens 中从 3,072 逐渐增加到 15,360，之后保持 15,360。batch schedule 和 learning-rate schedule 都是训练轨迹的一部分，不能只记录最终 global batch。

### Long Context Extension

预训练完成后，context length 通过两阶段、各 1,000 steps 的 extension training 从 4K 扩展到 32K，再扩展到 128K：

| 阶段 | Sequence length | Batch size | Learning rate |
|---|---:|---:|---:|
| 4K -> 32K | 32K | 1,920 | $7.3\times10^{-6}$ |
| 32K -> 128K | 128K | 480 | $7.3\times10^{-6}$ |

作者使用 YaRN，但只应用于 decoupled shared key $k_t^R$。两阶段使用相同 YaRN 配置：$s=40$、$\alpha=1$、$\beta=32$，scaling factor 为 $0.1\ln s+1$。

在经过 SFT 后的 NIAH 测试中，DeepSeek-V3 在 2K 到 128K 的不同 context length 和 document depth 上保持较好表现。NIAH 证明的是特定 needle retrieval 压力测试中的鲁棒性，不等价于所有 128K 任务都具有相同的有效利用能力。

## Base Model Evaluation

### Evaluation protocol

Base model 评估覆盖 English / Chinese / multilingual multiple choice、language understanding、closed-book QA、reading comprehension、reference disambiguation、language modeling、math、code 和 standardized exams。

作者根据任务特点混合使用 perplexity-based 与 generation-based evaluation，并在 Pile-test 上使用 Bits-Per-Byte，以降低不同 tokenizer 对比较结果的影响。

### Representative results

下表摘录论文 Table 3 中 DeepSeek-V3-Base 的结果：

| Benchmark | DeepSeek-V3-Base |
|---|---:|
| Pile-test BPB | 0.548 |
| BBH 3-shot | 87.5 |
| MMLU 5-shot | 87.1 |
| MMLU-Pro 5-shot | 64.4 |
| DROP 3-shot F1 | 89.0 |
| HumanEval 0-shot Pass@1 | 65.2 |
| MBPP 3-shot Pass@1 | 75.4 |
| LiveCodeBench-Base 3-shot Pass@1 | 19.4 |
| GSM8K 8-shot | 89.3 |
| MATH 4-shot | 61.6 |
| C-Eval 5-shot | 90.1 |
| MMMLU-non-English 5-shot | 79.4 |

作者报告 DeepSeek-V3-Base 在大多数 benchmark 上超过 DeepSeek-V2-Base 和 Qwen2.5-72B Base，并在 multilingual、code、math 等任务上超过 LLaMA-3.1 405B Base。这个结果支持稀疏激活、更大 total capacity、更多高质量 tokens 和系统性训练优化的组合价值，但不能将提升单独归因于某一个组件。

## Base Model Ablations

### MTP ablation

作者在约 15.7B total / 2.4B active 的 Small MoE 和约 228.7B total / 20.9B active 的 Large MoE 上比较 baseline 与 1-depth MTP。训练数据和其他架构保持一致，inference 时都丢弃 MTP module。

| Benchmark | Small baseline | Small + MTP | Large baseline | Large + MTP |
|---|---:|---:|---:|---:|
| BBH | 39.0 | 41.4 | 70.0 | 70.7 |
| MMLU | 50.0 | 53.3 | 67.5 | 66.6 |
| DROP | 39.2 | 41.3 | 68.5 | 70.6 |
| HumanEval | 20.7 | 26.8 | 44.5 | 53.7 |
| GSM8K | 25.4 | 31.4 | 72.3 | 74.0 |
| MATH | 10.7 | 12.6 | 38.6 | 39.8 |

MTP 在多数指标上有收益，但并非每个指标都单调提升，例如 large model 的 MMLU 略低。这组结果更适合支持“增加未来预测训练信号具有平均收益”，而不是“每个任务必然提升”。

### Auxiliary-loss-free ablation

作者在相同规模 baseline 上比较 sequence / batch auxiliary-loss-based routing 与 auxiliary-loss-free routing：

| Benchmark | Small aux-loss | Small aux-loss-free | Large aux-loss | Large aux-loss-free |
|---|---:|---:|---:|---:|
| Pile-test BPB | 0.727 | 0.724 | 0.656 | 0.652 |
| BBH | 37.3 | 39.3 | 66.7 | 67.9 |
| MMLU | 51.0 | 51.8 | 68.3 | 67.2 |
| HumanEval | 22.0 | 22.6 | 40.2 | 46.3 |
| GSM8K | 27.1 | 29.6 | 70.7 | 74.5 |
| MATH | 10.9 | 11.1 | 37.2 | 39.6 |

结果在多数指标上支持 auxiliary-loss-free routing 的性能优势，但 large model MMLU 出现轻微下降。更关键的机制证据来自 expert specialization：batch-wise balancing 允许不同 sequence 或 domain 使用不同 experts，因此比 sequence-wise 强制平衡更能保留 specialization。

在 1B MoE 实验中，sequence-wise auxiliary loss、auxiliary-loss-free 和 batch-wise auxiliary loss 的 validation loss 分别为 2.258、2.253、2.253；在 3B MoE 中分别为 2.085、2.080、2.080。这说明收益可能主要来自 balancing 的作用范围，而不一定完全来自“是否显式写了 auxiliary loss”。

## Post-Training

### SFT data construction

作者构造约 1.5M 条 instruction-tuning instances，覆盖多个 domain，并针对不同 domain 使用不同数据生成方式。

Reasoning data 覆盖数学、code competition 和 logic puzzle 等任务，主要来自 DeepSeek-R1 系列。R1 轨迹准确率高，但存在 overthinking、格式差和长度过长的问题，因此目标不是简单复制 R1，而是保留 reasoning capability，同时控制清晰度、格式和 generation length。

数据构造流程是：

1. 针对 code、math 或 general reasoning 训练一个 expert model；
2. expert model 的训练同时使用 SFT 与 RL；
3. 对每个问题构造 original response 样本，以及带 system prompt 的 R1 response 样本；
4. system prompt 显式引导 reflection 和 verification；
5. RL 阶段使用 high-temperature sampling，让模型在没有显式 system prompt 时也吸收 R1 patterns；
6. 经过数百个 RL steps 后，用 expert model 生成最终样本；
7. 通过 rejection sampling 保留高质量、较简洁的 SFT data。

Non-reasoning data，例如 creative writing、role-play 和 simple QA，则用 DeepSeek-V2.5 生成，再由 human annotators 验证准确性和正确性。

SFT 在 DeepSeek-V3-Base 上训练两个 epochs，learning rate 从 $5\times10^{-6}$ cosine decay 到 $1\times10^{-6}$。多个样本会 packing 到一个 sequence，但通过 sample masking 保证不同样本之间互相不可见。

### Reward Model 与 GRPO

RL 阶段同时使用 rule-based RM 和 model-based RM。数学和代码任务优先使用规则验证，例如固定答案格式、compiler 或 test cases；free-form ground-truth、creative writing 和其他开放任务使用 model-based RM。

model-based RM 从 DeepSeek-V3 SFT checkpoint 训练得到。为降低 reward hacking，preference data 不只包含最终 scalar reward，也包含给出该 reward 的 chain-of-thought / feedback reasoning。

对每个 question $q$，GRPO 从 old policy 采样一组 outputs。组内 reward 为 $r_1,\ldots,r_G$，advantage 用组内均值和标准差归一化：

$$
A_i=\frac{r_i-mean(\{r_j\})}{std(\{r_j\})}
$$

policy objective 使用 clipped importance ratio，并加 reference policy KL regularization。GRPO 不维护通常与 policy 同规模的 critic，而是用 group scores 估计 baseline。RL prompt 覆盖 coding、math、writing、role-playing 和 QA，使 post-training 不只优化可验证 reasoning，也对齐更一般的 interaction preference。

### Reasoning distillation

作者在 DeepSeek-V2.5 上做 ablation，比较 short-CoT baseline 与来自 R1 expert checkpoint 的 reasoning distillation：

| Benchmark | Baseline score | R1 distill score | Baseline length | R1 distill length |
|---|---:|---:|---:|---:|
| LiveCodeBench-CoT | 31.1 | 37.4 | 718 | 783 |
| MATH-500 | 74.6 | 83.2 | 769 | 1,510 |

结果说明 long-CoT distillation 对 reasoning benchmark 有明显帮助，但同时增加输出长度，尤其在 MATH-500 上更明显。DeepSeek-V3 的数据与训练设置因此需要在准确率、reflection / verification pattern 和 generation cost 之间做取舍。

### Self-rewarding

对于无法写出可靠规则 verifier 的开放任务，作者采用 constitutional AI 思路，让 DeepSeek-V3 自身的 voting evaluation 作为 feedback source，并结合 constitutional inputs 引导 alignment。它把非结构化的开放任务判断转为可扩展 reward signal，但 reliability 仍然依赖 judge 的一致性和抗偏差能力。

## Chat Model Evaluation

### Evaluation setting

Chat model 额外评估 IFEval、FRAMES、LongBench v2、GPQA、SimpleQA / Chinese SimpleQA、SWE-Bench Verified、Aider、LiveCodeBench、Codeforces、CNMO 2024 和 AIME 2024。

代码和数学任务允许最大 8,192 output tokens；AIME 和 CNMO 使用 temperature 0.7、平均 16 次运行，MATH-500 使用 greedy decoding；SWE-Bench Verified 使用 agentless framework，并以 diff format 评估 Aider 任务。

### Representative results

论文 Table 6 报告的 DeepSeek-V3 chat model 结果如下：

| Benchmark | DeepSeek-V3 |
|---|---:|
| MMLU | 88.5 |
| MMLU-Pro | 75.9 |
| GPQA-Diamond | 59.1 |
| DROP 3-shot F1 | 91.6 |
| IFEval Prompt Strict | 86.1 |
| SimpleQA Correct | 24.9 |
| FRAMES Accuracy | 73.3 |
| LongBench v2 Accuracy | 48.7 |
| HumanEval-Mul Pass@1 | 82.6 |
| LiveCodeBench CoT Pass@1 | 40.5 |
| LiveCodeBench Pass@1 | 37.6 |
| Codeforces Percentile | 51.6 |
| SWE-Bench Verified Resolved | 42.0 |
| Aider-Edit Accuracy | 79.7 |
| Aider-Polyglot Accuracy | 49.6 |
| AIME 2024 Pass@1 | 39.2 |
| MATH-500 EM | 90.2 |
| CNMO 2024 Pass@1 | 43.2 |
| Chinese SimpleQA Correct | 64.8 |

开放式评估中，作者报告 Arena-Hard 为 85.5，AlpacaEval 2.0 length-controlled win rate 为 70.0。RewardBench 上，DeepSeek-V3 单次平均为 87.0，majority vote @6 为 89.6。

这些结果显示，DeepSeek-V3 不仅在 code / math 上强，也在 long-context、instruction following、Chinese knowledge 和 open-ended conversation 上保持竞争力。但不同 benchmark 的评估机制不同，不能把所有分数压成一个“总能力”指标。

## 训练成本与效率

论文按 H800 GPU hour 统计：

| 阶段 | H800 GPU hours | 假设单价 |
|---|---:|---:|
| Pre-training | 2,664K | $5.328M |
| Context extension | 119K | $0.238M |
| Post-training | 5K | $0.010M |
| Total | 2,788K | $5.576M |

单价假设为每张 H800 每小时 2 美元。Pre-training 每 1T tokens 约需 180K H800 GPU hours，在 2,048 张 H800 的集群上约 3.7 天；完整 pre-training 不到两个月。

成本只包含官方 DeepSeek-V3 training，不包含 architecture、algorithm、data 和 ablation research 的前置成本。因此它更适合说明 production run 的系统效率，不应理解为从零研发同等模型的全部成本。

## 论文贡献的因果边界

### 证据较直接的部分

- auxiliary-loss-free 与 auxiliary-loss-based routing 在相同规模 baseline 上的 ablation；
- MTP 与无 MTP 的小 / 大模型对照；
- FP8 与 BF16 的 loss curve 对照；
- 128K context 的 NIAH 压力测试；
- pre-training、context extension 和 post-training 的训练成本拆分。

### 不能从报告单独推出的部分

- 671B total parameters 本身对能力的独立贡献；
- 14.8T tokens 相对于更少数据的边际收益；
- MLA、MoE、FP8、DualPipe、MTP 各自对最终 chat benchmark 的独立贡献；
- R1 distillation 对全部通用任务的普遍提升；
- H800 GPU hours 是否能迁移到其他硬件、框架和网络拓扑。

最终 DeepSeek-V3 并不是一个只改变单一变量的 controlled experiment。最可靠的阅读方式是把 ablation 用于理解局部机制，把最终 benchmark 视为整个系统组合的结果。

## 对当前 Agentic Mid-training 研究的启发

这篇论文的核心对象是 base model pre-training，不是 agentic mid-training，因此不能直接把 14.8T tokens、37B active parameters 或 FP8 recipe 套到 agent trajectory 上。但它提供了几条可迁移的方法论：

1. **统计 effective tokens，而不只统计 nominal tokens。** Agent trajectory 中 observation、tool result、reasoning、action 和环境重复上下文的信息价值不同。数据规模评估应同时记录总 token、有效决策 token、重复率、任务域覆盖和质量等级。
2. **把数据配方作为训练变量。** 目标能力需要通过 mixture 和 schedule 显式规划，而不是只依赖总数据量。
3. **将 sequence integrity 与 sample isolation 分开处理。** 保留长轨迹状态上下文，与 packing 中隔离不同样本，是两种不同训练语义。
4. **把快速评测嵌入训练过程。** 应同时保留 validation loss、domain loss、tool-call 格式、短轨迹执行成功率和长任务成功率。
5. **对 reasoning data 做长度与质量控制。** reasoning 越长不等于决策越好，应同时评估 action correctness、恢复能力、任务完成率和 token cost。
6. **长任务的瓶颈可能来自系统而非 loss。** 训练时应记录数据读取、packing、attention、activation memory 和 checkpoint 代价，不要只看 NTP loss。
7. **组件 ablation 要围绕可归因变量设计。** 如果同时改变 reasoning content、tool schema、trajectory length、数据 mixture 和 tokenizer，就无法判断能力提升来自哪里。

## 局限与疑问

### 最终模型缺少完整的单变量归因

最终性能来自复杂组合，报告没有对所有组件做正交、等 compute 的完整 factorial ablation。最终 chat benchmark 无法单独说明 MTP、auxiliary-loss-free routing、FP8、DualPipe 或 R1 distillation 各自贡献多少。

### FP8 结果依赖硬件与 kernel

FP8 的精度和吞吐建立在 H800 Tensor Core、customized PTX、特定 tile / block quantization 和高精度 promotion 上。换到不支持相同硬件能力的 GPU 或其他 accelerator，relative loss error、通信开销和实现复杂度都可能变化。

### MoE deployment 门槛仍然较高

尽管每 token 只激活 37B 参数，推理仍需要大规模 expert storage、跨节点通信和 redundant expert placement。论文报告的高吞吐依赖较大的部署单元，不适用于所有团队和低并发场景。

### 长上下文评测仍有限

NIAH 能验证特定位置检索，但不能覆盖真实长文档中的多跳推理、信息冲突、跨段落聚合、工具 observation 处理和输出可靠性。128K 最大长度不应直接等价于 128K 有效理解。

### 后训练 reward 仍存在 judge bias

规则 RM 适合 math / code 等可验证任务，但 model-based RM 和 self-rewarding 会继承 judge 的偏差。preference data 中加入 reasoning feedback 可以降低部分 reward hacking，却不能保证开放任务评价完全可靠。

## 关键结论

1. **大模型效率来自共同设计。** MoE、MLA、低精度、并行策略、通信 kernel、数据配方和后训练必须作为一套系统分析。
2. **MoE 的核心难点是 routing 与 systems co-design。** 负载均衡既要避免设备过载，也不能强行抹平专家 specialization；通信拓扑和 kernel 设计与路由算法同等重要。
3. **Auxiliary-loss-free 不等于完全没有 balance constraint。** 它将主要的 batch-wise balance 放到不参与主 loss 反向传播的 bias 更新，同时保留极小的 sequence-wise balance loss 处理极端情况。
4. **FP8 training 是混合精度工程，而不是统一降精度。** 精度、scale 粒度、累加路径和算子选择需要逐一设计，并用大规模 loss curve 与 divergence monitoring 验证。
5. **MTP 是训练目标增强，也可以服务 inference acceleration。** 训练收益来自额外未来预测信号，推理收益来自可选的 speculative decoding；两者不能混为一谈。
6. **高 base capability 与后训练能力是两条连续但不同的路径。** 14.8T-token pre-training 建立基础能力，SFT / RL / R1 distillation 再将其塑造成 chat、reasoning 和 instruction-following 行为。
7. **对 Agentic mid-training 的直接启发是记录结构、质量和系统成本。** 长轨迹数据不能只按 token 数统计，也不能只用 average NTP loss 判断 agentic capability。

## 相关知识链接

- [[architecture/model-families/deepseek|DeepSeek]]
- [[architecture/sparse-and-efficient/moe|Mixture of Experts]]
- [[architecture/attention/multi-head-latent-attention|Multi-Head Latent Attention]]
- [[training/optimization/mixed-precision|Mixed Precision]]
- [[training/distributed-training/pipeline-parallel|Pipeline Parallel]]
- [[training/distributed-training/data-parallel|Data Parallel]]
- [[training/optimization/optimizer-state|Optimizer State]]
- [[training/pretraining/data-mix|Data Mix]]
- [[training/pretraining/objective|Training Objective]]
- [[training/post-training/grpo|GRPO]]
- [[training/post-training/knowledge-distillation|Knowledge Distillation]]
- [[sources/papers/2024-deepseek-v2|DeepSeek-V2]]
- [[sources/papers/2024-deepseekmoe|DeepSeekMoE]]
- [[sources/papers/2025-rlp-reinforcement-as-a-pretraining-objective|RLP: Reinforcement as a Pretraining Objective]]
