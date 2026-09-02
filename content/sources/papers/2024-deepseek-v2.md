---
title: "DeepSeek-V2"
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
  - kv-cache
  - long-context
  - distributed-training
source_url: https://arxiv.org/abs/2405.04434
paper_date: "2024-05"
paper_order: "04434"
---

# DeepSeek-V2

## 基本信息

- 标题：[DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)
- 版本：arXiv:2405.04434v5，2024-06-19
- 作者：DeepSeek-AI
- 项目与模型：[deepseek-ai/DeepSeek-V2](https://github.com/deepseek-ai/DeepSeek-V2)
- 模型规模：236B total parameters，21B activated parameters per token
- 预训练数据：8.1T tokens，中英双语
- 上下文长度：128K
- 核心架构：[[architecture/attention/multi-head-latent-attention|Multi-Head Latent Attention]] 与 [[architecture/sparse-and-efficient/moe|DeepSeekMoE]]
- 相关主题：[[architecture/model-families/deepseek|DeepSeek]]，[[inference/kv-cache-and-memory/kv-cache|KV Cache]]，[[training/distributed-training/pipeline-parallel|Pipeline Parallel]]，[[training/post-training/grpo|GRPO]]

DeepSeek-V2 是 DeepSeek 高效模型路线中的关键节点。它第一次把 MLA、DeepSeekMoE、长上下文扩展以及面向 MoE 的分布式训练优化整合到 236B 规模，并给出从模型效果到训练成本、KV Cache 和 serving throughput 的完整证据。后续 DeepSeek-V3 继续沿用 MLA 与 DeepSeekMoE，但改变了 expert 配置、负载均衡和系统实现；因此，理解 V2 不只是回顾前代模型，也是在理解 DeepSeek-V3 架构的起点。

论文的主线不是单纯追求更大的 total parameter count，而是同时处理两个不同瓶颈：

```text
Attention history states
  -> MLA low-rank KV compression
  -> smaller KV Cache and higher serving batch capacity

FFN model capacity
  -> DeepSeekMoE sparse activation
  -> more total capacity under lower per-token computation

MoE system overhead
  -> device-limited routing + balance losses + token dropping
  -> bounded communication and more stable device load
```

## 研究问题

### 如何扩大模型容量而不按 dense model 的计算成本增长

Dense Transformer 的每个 token 都经过全部 FFN 参数，扩大模型容量通常会同步提高训练 FLOPs 和 inference cost。MoE 允许模型保留较大的 total capacity，同时只为每个 token 激活少量 experts；但稀疏激活会引入 routing、负载均衡、token dispatch 和跨设备通信成本。

DeepSeek-V2 需要回答的不是“MoE 是否可用”，而是 fine-grained experts 和 shared experts 能否在 236B total parameters 上形成有效 specialization，并在真实 H800 集群上保留足够高的训练效率。

### 如何降低自回归推理中的 KV Cache

MoE 主要减少 FFN 计算，并不会自动解决 attention 的历史状态开销。随着 context length 和 serving batch size 增大，标准 MHA 为每层、每个历史 token、每个 head 缓存完整 K/V，显存容量和 memory bandwidth 很容易成为 decode 瓶颈。

MQA 和 GQA 通过共享 K/V heads 缩小缓存，但论文在受控实验中观察到，head sharing 会在若干 hard benchmarks 上损失性能。MLA 因而尝试另一种折中：保留多头 query 与较强 attention 表达能力，同时把 K/V 联合压缩为低维 latent state。

### 如何把算法上的稀疏性转化为系统吞吐

“每 token 只激活 21B parameters”只描述计算路径的一部分。236B 权重仍需存储和放置，routed tokens 仍需跨设备发送，不均衡 routing 还可能让少数设备成为 straggler。DeepSeek-V2 因此把 architecture 与 system design 联合处理：限制每个 token 触达的设备数，分别约束 expert load、device compute 和 communication load，并通过 kernel fusion 与 communication-computation overlap 提高实际 MFU。

## 核心主张

1. **MLA 可以同时降低 KV Cache 并保持较强 attention 能力。** DeepSeek-V2 每 token、每层只需缓存 compressed KV latent 和一份 shared RoPE key，其缓存元素数相当于 2.25-group GQA；论文的两个规模对照中，MLA 的 hard benchmark 表现整体优于对应 MHA 模型。
2. **DeepSeekMoE 将大容量与较低 per-token computation 结合。** 模型包含 236B total parameters，但每个 token 激活约 21B parameters；shared experts 承担通用变换，fine-grained routed experts 提供更灵活的稀疏容量。
3. **MoE 效率依赖 routing 与系统协同。** Device-limited routing、三类 auxiliary balance loss、device-level token dropping、8-way expert parallelism 和 AllToAll overlap 共同控制通信与负载。
4. **DeepSeek-V2 的效率收益同时来自 architecture 与 deployment optimization。** 相比 DeepSeek 67B，论文报告 training cost 降低 42.5%、实际部署 KV Cache 降低 93.3%、maximum generation throughput 提升到 5.76 倍；其中 KV quantization、FP8 weights 和定制 kernel 也参与了最终结果，不能把全部增益只归因于 MLA 或 MoE。
5. **强 base model 仍需单独的 alignment 阶段形成稳定 chat 行为。** 论文使用 1.5M SFT instances，再以 GRPO 进行 reasoning alignment 和 human preference alignment；RL 改善代码、数学与开放式生成，但部分标准 benchmark 出现 alignment tax。

## Architecture

### 模型配置

DeepSeek-V2 保留 decoder-only Transformer 主体结构，attention 使用 MLA，除第一层外的 FFN 均替换为 DeepSeekMoE。

| 配置 | DeepSeek-V2 |
|---|---:|
| Transformer layers | 60 |
| Hidden dimension | 5,120 |
| Attention heads | 128 |
| Per-head content dimension | 128 |
| KV compression dimension $d_c$ | 512 |
| Query compression dimension $d'_c$ | 1,536 |
| Decoupled RoPE dimension $d_h^R$ | 64 |
| Shared experts per MoE layer | 2 |
| Routed experts per MoE layer | 160 |
| Activated routed experts per token | 6 |
| Expert intermediate dimension | 1,536 |
| Total parameters | 236B |
| Activated parameters per token | 21B |

低秩压缩和细粒度 expert segmentation 会改变 layer output scale。作者在 compressed latent vectors 后增加 RMSNorm，并在 latent bottleneck 与 routed-expert intermediate states 上增加 scaling factors，以稳定大规模训练。这一细节说明，替换 attention 或 FFN 结构并不是无条件的模块互换，压缩宽度与激活尺度会直接影响优化稳定性。

### Multi-Head Latent Attention

#### 从完整 K/V 到压缩 latent

标准 MHA 对每个 token、每层需要缓存：

$$
2n_hd_hl
$$

其中 $n_h$ 是 attention heads 数量，$d_h$ 是每个 head 的维度，$l$ 是层数，系数 2 对应 key 与 value。

MLA 先把当前 hidden state $h_t$ 压缩成 K/V 共用的 latent vector：

$$
c_t^{KV}=W^{DKV}h_t
$$

再从 latent 中恢复 content key 和 value：

$$
k_t^C=W^{UK}c_t^{KV},\qquad
v_t^C=W^{UV}c_t^{KV}
$$

其中 $c_t^{KV}\in\mathbb{R}^{d_c}$，且 $d_c\ll n_hd_h$。自回归生成时，历史 token 不再需要保存所有 heads 的完整 K/V，只需缓存低维 $c_t^{KV}$。借助矩阵乘法结合律，K up-projection 可以吸收到 query-side projection，V up-projection 可以吸收到 output projection，因此 inference 不必逐 token 显式恢复完整 content K/V。

MLA 还对 query 进行低秩压缩：

$$
c_t^Q=W^{DQ}h_t,\qquad
q_t^C=W^{UQ}c_t^Q
$$

query 不作为历史状态长期缓存，所以这一步不会继续缩小 KV Cache；它的主要作用是降低 training activation memory。

#### 为什么 RoPE 必须解耦

若直接对 compressed content key 应用 [[architecture/positional-encoding/rope|RoPE]]，位置相关旋转矩阵会位于 query projection 与 key up-projection 之间。由于矩阵乘法不可交换，$W^{UK}$ 无法再被吸收到 query-side projection；生成每个新 token 时，系统可能需要为 prefix 重新构造 keys，低秩缓存的推理收益会被破坏。

DeepSeek-V2 因此将位置相关路径拆出：

$$
q_t^R=\operatorname{RoPE}(W^{QR}c_t^Q),\qquad
k_t^R=\operatorname{RoPE}(W^{KR}h_t)
$$

每个 head 的最终 query/key 由 content part 和 RoPE part 拼接：

$$
q_{t,i}=[q_{t,i}^C;q_{t,i}^R],\qquad
k_{t,i}=[k_{t,i}^C;k_t^R]
$$

$q_{t,i}^R$ 是 multi-head query，而 $k_t^R$ 在所有 heads 间共享。最终需要缓存的是 $c_t^{KV}$ 与 $k_t^R$，所以 MLA 每 token、每层的缓存元素数为：

$$
(d_c+d_h^R)l
$$

DeepSeek-V2 中 $d_c=4d_h$、$d_h^R=d_h/2$，于是缓存量为 $4.5d_hl$，相当于 2.25-group GQA 的 $2\times2.25d_hl$。这一等价只比较缓存元素数，不表示 MLA 的计算图或能力等同于 GQA。

#### Attention ablation

论文先用约 7B dense models 对比 MQA、8-group GQA 和 MHA。三组模型都训练 1.33T tokens，并通过调整层数使参数量接近；MHA 在 BBH、MMLU、C-Eval 和 CMMLU 上均优于 MQA/GQA。这组实验给出了作者不直接采用更少 KV heads 的理由。

随后，作者在两个 MoE 规模上只替换 attention mechanism：

| 模型 | KV cache elements / token | BBH | MMLU | C-Eval | CMMLU |
|---|---:|---:|---:|---:|---:|
| Small MoE + MHA | 110.6K | 37.9 | 48.7 | 51.6 | 52.3 |
| Small MoE + MLA | 15.6K | 39.0 | 50.0 | 50.9 | 53.4 |
| Large MoE + MHA | 860.2K | 46.6 | 57.5 | 57.9 | 60.7 |
| Large MoE + MLA | 34.6K | 50.7 | 59.0 | 59.2 | 62.5 |

Small models 约为 16B total parameters、训练 1.33T tokens；large models 约为 250B total parameters、训练 420B tokens。MLA 分别只使用对应 MHA 约 14% 和 4% 的 KV Cache，并在 8 个模型/benchmark 对照中的 7 项更高。C-Eval small-scale 例外提醒我们，MLA 不是逐任务严格支配 MHA；但这组受控实验比只比较最终模型更直接地支持了“压缩缓存并未以明显能力退化为代价”的主张。

### DeepSeekMoE

DeepSeekMoE 的两个核心设计是：

- **Fine-grained expert segmentation**：把相同总 expert 参数切成更多、更小的 experts，使一个 token 可以组合更细粒度的知识与变换。
- **Shared expert isolation**：保留所有 token 都经过的 shared experts，承载通用知识，减少 routed experts 之间重复学习相同模式。

设 $u_t$ 为 FFN 输入，输出为：

$$
h'_t=u_t+
\sum_{i=1}^{N_s}\operatorname{FFN}^{(s)}_i(u_t)+
\sum_{i=1}^{N_r}g_{i,t}\operatorname{FFN}^{(r)}_i(u_t)
$$

router 根据 token hidden state 与 expert centroid 的 affinity 做 top-$K_r$ 选择：

$$
s_{i,t}=\operatorname{Softmax}_i(u_t^Te_i)
$$

未进入 top-$K_r$ 的 routed expert 权重为零。DeepSeek-V2 每层有 2 个 shared experts、160 个 routed experts，每个 token 激活 6 个 routed experts。

### Device-Limited Routing

Fine-grained segmentation 增加了每 token 组合 experts 的灵活性，也可能让 target experts 分散到更多设备。Expert parallelism 下，一个 token 的通信频率与它触达的设备数直接相关。

DeepSeek-V2 先选择包含最高 affinity experts 的 $M$ 个 devices，再只从这些 devices 上的 experts 中执行 top-k selection。论文报告，当 $M\geq3$ 时，device-limited routing 的效果与 unrestricted top-k 大致相当；正式训练采用 $M=3$。这是一种 routing search-space constraint：它不直接减少激活 expert 数，而是限制这些 experts 的物理分布范围。

### 三层负载均衡

论文把 MoE imbalance 拆成三个层次，而不是用单一 expert-count metric 处理：

1. **Expert-level balance loss**：避免少量 experts 垄断 routing，降低 routing collapse 风险，使各 experts 获得训练信号。
2. **Device-level balance loss**：聚合同一 device 上的 experts，约束不同设备之间的计算量，减少 straggler。
3. **Communication balance loss**：约束各设备接收的 token 数，避免 device-limited routing 只限制发送范围，却仍形成接收热点。

三类损失都使用实际选择频率与平均 affinity 的乘积构造，但聚合粒度不同。正式训练中的系数为：

$$
\alpha_1=0.003,\qquad
\alpha_2=0.05,\qquad
\alpha_3=0.02
$$

这些系数不能脱离 expert 数量、EP 拓扑、batch composition 和 routing 实现直接迁移。它们更重要的启示是：训练质量、设备计算负载和网络接收负载是三个相关但不等价的优化对象。

### Token Dropping

Auxiliary losses 只能鼓励均衡，不能严格保证每个 device 的负载。DeepSeek-V2 在训练中采用 device-level token dropping：

- 以 capacity factor 1.0 计算每个 device 的平均计算预算；
- 超出预算时，优先丢弃 affinity 最低的 token-expert assignments；
- 约 10% 的 training sequences 被保护，确保其中的 token 永不被丢弃；
- evaluation 阶段不丢 token。

被保护的 sequences 让模型始终看到一部分与 inference 完全一致的完整计算路径。论文称 inference 时可按效率需求决定是否 dropping，但正式 evaluation 不 drop。Token dropping 是 V2 在特定 routing 与系统条件下采用的训练加速策略，不是 DeepSeekMoE 或所有 MoE 的固有属性；DeepSeek-V3 后来通过新的负载均衡策略取消了 training / inference token dropping。

## Pre-Training

### Data Construction

DeepSeek-V2 延续 DeepSeek 67B 的数据处理阶段，在此基础上扩大互联网数据、修正清洗流程中误删的有效内容、增加中文数据，并改进 quality-based filtering。最终 corpus 为 8.1T tokens，中文 tokens 比英文 tokens 约多 12%。Tokenizer 使用 100K vocabulary 的 byte-level BPE。

作者还过滤了受特定区域文化影响的 contentious content，希望降低预训练语料中的主观偏差。附录显示，这一策略可能降低模型在 MMLU Humanity-Moral 等特定文化价值 benchmark 上的分数。三位人工标注者对 420 个 scenarios 重新标注后，与 benchmark ground truth 的 agreement 只有 42.1% 到 66.7%，标注者之间也只有 57.9% 到 69.0%。这说明 value-sensitive benchmark 的 ground truth 本身可能包含文化依赖；但该分析不能证明 filtering 已经消除 bias，也没有隔离过滤策略对其他任务的影响。

论文明确说明，DeepSeek-V2 pre-training 没有混入 SFT data。因而 base model 结果与后续 instruction tuning 结果之间仍保持相对清楚的阶段边界。

### Training Recipe

| 配置 | 数值 |
|---|---:|
| Optimizer | AdamW |
| $\beta_1 / \beta_2$ | 0.9 / 0.95 |
| Weight decay | 0.1 |
| Maximum learning rate | $2.4\times10^{-4}$ |
| Warmup | 2K steps |
| Gradient clipping norm | 1.0 |
| Batch size | 2,304 -> 9,216 sequences |
| Batch-size ramp-up | first 225B tokens |
| Training sequence length | 4K |
| Training tokens | 8.1T |

Learning rate 先在 2K steps 内线性 warmup 到峰值；训练约 60% tokens 后乘 0.316，约 90% tokens 后再次乘 0.316。Batch size 则在前 225B tokens 从 2,304 逐步增至 9,216，之后保持不变。这是 warmup、batch-size ramp-up 和 step-decay 的组合 recipe，而不是单一 cosine schedule。

### Distributed Training Infrastructure

DeepSeek-V2 使用内部 HAI-LLM framework，在 NVIDIA H800 集群上训练：

- 16-way zero-bubble pipeline parallelism；
- 8-way expert parallelism；
- ZeRO-1 data parallelism；
- training 阶段不使用 tensor parallelism；
- 部分 operators 通过 recomputation 降低 activation memory；
- shared experts computation 与 expert-parallel AllToAll communication 重叠；
- communication、routing 和跨 experts 的 fused linear computation 使用定制 CUDA kernels；
- MLA 基于改进版 FlashAttention-2 优化。

单节点内 8 张 H800 通过 NVLink / NVSwitch 连接，节点间使用 InfiniBand。作者不使用 TP 的理由，是较低的 activated parameters 与 recomputation 已使单个计算分片可容纳，并希望减少额外通信；这不构成“MoE 不需要 TP”的普遍结论。并行组合取决于 total parameter placement、activation memory、network topology、expert size 和目标 micro-batch。

### Long-Context Extension

初始 pre-training 的 maximum sequence length 为 4K。完成 8.1T-token training 后，作者用 YaRN 将目标 context window 扩展到 128K：

- 只对承载 RoPE 的 decoupled shared key $k_t^R$ 应用 YaRN；
- scale $s=40$，$\alpha=1$，$\beta=32$；
- target maximum context length 设置为 160K；
- 额外训练 1,000 steps；
- sequence length 为 32K；
- batch size 为 576 sequences。

模型只在 32K sequence length 上进行 extension training，却在 Needle In A Haystack 中测试到 128K。实验显示不同 context length 和 document depth 下都能检索 needle，证明位置扩展没有完全失效；但 NIAH 主要检验显式信息检索，不能替代长文理解、跨段推理、多轮状态保持与真实长任务评测。

## Base Model Evaluation

DeepSeek-V2 Base 的评测覆盖英文、中文、code、math、reading comprehension、knowledge 和 standardized exams。所有对比模型均使用作者内部 framework 与统一 setting 重新评测，降低了 prompt / evaluator 差异，但结果仍由论文作者自行报告。

| Benchmark | DeepSeek 67B | Qwen1.5 72B | Mixtral 8x22B | LLaMA 3 70B | DeepSeek-V2 |
|---|---:|---:|---:|---:|---:|
| MMLU | 71.3 | 77.2 | 77.6 | 78.9 | 78.5 |
| BBH | 68.7 | 59.9 | 78.9 | 81.0 | 78.9 |
| HumanEval | 45.1 | 43.9 | 53.1 | 48.2 | 48.8 |
| MBPP | 57.4 | 53.6 | 64.2 | 68.6 | 66.6 |
| GSM8K | 63.4 | 77.9 | 80.3 | 83.0 | 79.2 |
| MATH | 18.7 | 41.4 | 42.5 | 42.2 | 43.6 |
| C-Eval | 66.1 | 83.7 | 59.6 | 67.5 | 81.7 |
| CMMLU | 70.8 | 84.3 | 60.0 | 69.3 | 84.0 |

从结果看，DeepSeek-V2 相比 DeepSeek 67B 在绝大多数 benchmark 上显著提升，并以 21B activated parameters 达到当时开源模型中的较强水平。它在中文、数学和部分代码任务上表现突出；MMLU 接近 LLaMA 3 70B，但 BBH、英文 commonsense / knowledge 等任务仍存在差距。论文指出，DeepSeek-V2 使用的 English training tokens 少于 LLaMA 3 70B 的四分之一，因此中英文 data mix 是解释结果的重要背景。

这些结果不能简化为“21B 模型击败 70B 模型”。21B 是 activated parameters，而 DeepSeek-V2 仍有 236B total parameters；MoE 与 dense model 的计算、存储、通信和容量不能由单一参数数字公平对齐。最终能力也同时受 8.1T data、tokenizer、training recipe 和 architecture 影响。

## Training and Inference Efficiency

### Training cost

在 H800 集群的实际训练中：

| 模型 | GPU hours / trillion tokens |
|---|---:|
| DeepSeek 67B | 300.6K |
| DeepSeek-V2 | 172.8K |

DeepSeek-V2 因而比 DeepSeek 67B 节省 42.5% training cost。该数字包含 sparse activation、MoE communication overhead、operator optimization 和集群实现的共同结果，不是只由 21B activated parameters 线性推导出来的理论 FLOPs 比例。

### Inference cost

部署时，作者还将 model parameters 转为 FP8，并把 KV Cache 平均量化到 6 bits / element。在 MLA 与这些优化共同作用下，论文报告：

- 实际 deployed KV Cache 比 DeepSeek 67B 减少 93.3%；
- 单节点 8 张 H800 上 maximum generation throughput 超过 50K tokens/s；
- generation throughput 达到 DeepSeek 67B 最大值的 5.76 倍；
- prompt input throughput 超过 100K tokens/s。

吞吐测试使用实际 DeepSeek 67B 服务中的 prompt / generation length distribution，比固定长度 microbenchmark 更接近在线 workload。不过论文没有完整披露 batch scheduling、latency percentile、并发分布和 baseline deployment stack，因此这些数字适合证明同一内部系统中的相对收益，不宜直接外推到其他 serving engine 或硬件。

## Alignment

### SFT

作者构造 1.5M instruction-tuning instances，其中 1.2M 用于 helpfulness、0.3M 用于 safety，覆盖 math、code、writing、reasoning 和安全等任务。SFT 训练 2 epochs，learning rate 为 $5\times10^{-6}$。相比前代数据，作者重点提高数据质量，以减少 hallucination 并改善 writing。

### GRPO

RL 使用 DeepSeekMath 中提出的 [[training/post-training/grpo|Group Relative Policy Optimization]]。对每个 question $q$，old policy 采样一组 outputs，使用组内 rewards 的均值和标准差构造 standardized advantage：

$$
A_i=\frac{r_i-\operatorname{mean}(r_1,\ldots,r_G)}
{\operatorname{std}(r_1,\ldots,r_G)}
$$

GRPO 保留 PPO-style clipped objective 与 reference-policy KL regularization，但不训练与 policy 同规模的 critic model，降低大模型 RL 的显存与计算负担。

训练分为两个阶段：

1. **Reasoning alignment**：使用 code 与 math prompts。Code preference 来自 compiler feedback，math preference 来自 ground-truth labels，再训练 reasoning reward model。
2. **Human preference alignment**：组合 helpfulness reward model、safety reward model 和 rule-based reward model。

Reward models 从 DeepSeek-V2 Chat (SFT) 初始化，以 point-wise 或 pair-wise loss 训练。RL 系统采用 training / inference 不同并行策略的 hybrid engine，以 vLLM 承担 large-batch rollout，并在 CPU offloading 与 GPU loading 之间调度模型状态。这部分已具备后续大规模 online RL 系统的早期形态。

### Chat model results

| Benchmark | Chat (SFT) | Chat (RL) | 变化 |
|---|---:|---:|---:|
| HumanEval | 76.8 | 81.1 | +4.3 |
| MBPP | 70.4 | 72.0 | +1.6 |
| LiveCodeBench | 28.7 | 32.5 | +3.8 |
| GSM8K | 90.8 | 92.2 | +1.4 |
| MATH | 52.7 | 53.9 | +1.2 |
| BBH | 81.3 | 79.7 | -1.6 |
| MMLU | 78.4 | 77.8 | -0.6 |
| MT-Bench | 8.62 | 8.97 | +0.35 |
| AlpacaEval 2.0 LC win rate | 30.0 | 38.9 | +8.9 |
| AlignBench overall | 7.74 | 7.91 | +0.17 |

RL 对 code、math 和 open-ended preference metrics 的改善较稳定，但并非所有 benchmark 都提高。C-Eval 从 80.9 降至 78.0，CMMLU 从 82.4 降至 81.6，BBH 也下降。论文将其称为 alignment tax，并通过 data processing、多 reward 和 training strategy 将损失控制在可接受范围，但没有消除这种权衡。

论文还报告，少于 10K SFT instances 时 IFEval 明显下降，说明模型规模增大可能降低特定技能的数据需求，但不会让 instruction data 完全失去必要性；writing 和 open-ended tasks 尤其依赖数据质量。作者观察到 online RL 明显优于 offline RL，因此投入在线框架，但没有给出完整受控对照，并明确指出结论可能随场景变化。

## DeepSeek-V2-Lite

DeepSeek-V2-Lite 是面向公开研究的较小版本，用于降低 MLA 与 DeepSeekMoE 的实验门槛：

| 配置 | DeepSeek-V2-Lite |
|---|---:|
| Layers / hidden dimension | 27 / 2,048 |
| Attention heads | 16 |
| KV compression dimension | 512 |
| Query compression | 不使用 |
| Shared / routed experts | 2 / 64 |
| Activated routed experts | 6 |
| Total / activated parameters | 15.7B / 2.4B |
| Pre-training tokens | 5.7T |
| Extended context | 32K |

Lite 的 experts 不跨设备部署，因此不使用 device-level 与 communication balance loss，只保留 $\alpha_1=0.001$ 的 expert-level balance loss。Base model 的 MMLU / CMMLU 为 58.3 / 64.3，Chat model 为 55.7 / 62.5。它明显优于论文中的 DeepSeek 7B 与 DeepSeekMoE 16B baselines，但训练 token 从 2T 增至 5.7T，architecture 与 data budget 同时改变，不能把全部增益单独归因于 MLA。

## 证据边界与局限

### 架构贡献没有被完全因果拆分

MLA 有相对清楚的 MHA 对照，但最终 DeepSeek-V2 与 DeepSeek 67B 的比较同时改变了 dense / MoE architecture、参数容量、data amount、data quality、中文比例、training recipe 和 serving optimization。42.5% training-cost saving 和 5.76x throughput 是整个系统的结果，不能用于估计单一模块的独立贡献。

### Long-context evaluation 仍然单一

128K 主要通过 NIAH 证明。该测试适合检测不同深度位置的信息是否仍可访问，但无法充分衡量长文档 synthesis、跨段 causal reasoning、long-horizon agent state tracking、位置偏差或上下文利用率。

### MoE 的 deployment cost 不能只看 activated parameters

21B active 有利于估计 per-token FLOPs，却不能代表 model weight memory、expert placement、AllToAll traffic 和 tail latency。论文给出 throughput，但没有完整的 latency、concurrency、failure recovery 与跨硬件复现信息。

### Alignment evaluation 含 model-based judgment

MT-Bench、AlpacaEval 和 AlignBench 依赖 judge model 或自动评估，其结论可能受 response length、style、judge preference 和 prompt format 影响。AlpacaEval 使用 length-controlled win rate 已缓解一部分长度偏差，但不能替代人工与任务成功率评估。

### 数据透明度有限

论文说明了 8.1T-token corpus 的语言比例与改进方向，但没有公开 source-level mixture、去重率、quality classifier、污染检测和详细数据许可。因而无法独立判断能力变化分别来自数据规模、质量还是领域比例。

### 模型自身限制

作者明确指出：模型无法在 pre-training 后持续更新知识，仍可能生成不实信息、未经验证的建议与 hallucination；训练数据以中文和英文为主，其他语言能力有限；模型仅支持 text modality。

## 对训练与推理系统的启发

### 将能力、计算和通信分开建模

DeepSeek-V2 最可迁移的经验，是不要用一个参数量或一个 FLOPs 数字概括系统：

- total parameters 决定模型容量与权重存储；
- activated parameters 影响每 token 的主要计算；
- routing fan-out 决定 token 触达多少 experts；
- device fan-out 决定 expert parallel communication；
- load skew 决定设备利用率和 tail step time；
- KV representation 决定长上下文 serving memory 与 bandwidth。

设计 MoE 训练实验时，至少应同时记录 expert load、device load、tokens sent / received、drop rate、AllToAll time、MFU 和 validation quality。只看 language-model loss 或 theoretical FLOPs，无法判断 sparse architecture 是否真正转化为训练收益。

### Attention efficiency 与 FFN efficiency 是两条独立轴

DeepSeekMoE 解决的是 FFN capacity / computation，MLA 解决的是 attention history state / KV Cache。大规模 agent、long-context 和 high-concurrency serving 同时受两者约束；只缩小 active FFN parameters，并不会自动降低随着 context length 线性增长的 cache。

### Routing constraint 可以显式编码系统拓扑

Device-limited routing 把网络拓扑直接写进 expert selection：先限制设备，再选择 experts。它说明 router 不必只优化 semantic affinity，也可以在保持能力的前提下接受 hardware-aware constraint。后续策略可进一步把 node locality、link bandwidth、expert replication 和 request distribution 纳入 routing，但每种约束都需要 quality ablation，而不能只看通信量。

### 长上下文扩展应区分“可寻址”与“可推理”

V2 从 4K pre-training 出发，只用 1,000 steps 的 32K extension training 获得 128K NIAH 能力，说明 position extrapolation 和较低成本 adaptation 可以迅速恢复长位置可访问性。但真实 agent trajectory 训练还需要验证：模型能否识别关键 observation、保持任务状态、跨越长距离完成 credit assignment，并忽略大量低价值环境 token。NIAH 成功只是最低门槛。

### 训练阶段的 token dropping 需要语义审计

按低 affinity 丢弃 assignments 能固定 device compute budget，但被丢 token 的 expert path 与完整 inference path 不一致。V2 通过保护约 10% sequences 缓解这一问题。迁移到长轨迹或高价值样本时，除了总体 drop rate，还应检查被丢位置是否集中在稀有语言、代码边界、tool result、错误恢复或其他关键 token 上。

### 评测必须覆盖优化的副作用

V2 的 RL 同时展示了 code/math/open-ended improvement 与 BBH/MMLU/中文考试集下降。类似地，数据 debiasing 也改变了 value-sensitive benchmark。任何 data mix、routing、long-context extension 或 alignment 方案都应同时保留目标能力集、通用能力回归集和系统效率指标，否则容易把局部收益误认为整体改进。

## 关键结论

1. DeepSeek-V2 的核心不是单一 MoE scaling，而是 MLA、DeepSeekMoE 与 distributed system 的协同设计。
2. MLA 通过 low-rank joint KV compression 和 decoupled RoPE，将 inference cache 从完整 multi-head K/V 改为 compressed latent 与 shared positional key；其原始受控实验支持了显著缓存节省与较强能力可以同时实现。
3. DeepSeekMoE 用 fine-grained routed experts 提供组合容量，用 shared experts 承担通用变换；但真正的训练效率依赖 device-limited routing、分层 balance objectives 和系统级 communication overlap。
4. 236B total / 21B active 必须同时报告。Active parameters 不能替代 total weight memory、communication 与 deployment complexity。
5. 128K context 来自 4K pre-training 后的 YaRN extension；NIAH 验证了可访问性，但不足以代表完整长上下文推理能力。
6. SFT 与 GRPO 进一步释放了 code、math 和开放式生成能力，同时暴露 alignment tax；能力提升和通用 benchmark 回归需要联合评估。
7. 论文最稳定的工程结论，是 architecture、data、training system 与 serving system 必须共同设计和共同测量。

## 相关知识链接

- [[architecture/attention/multi-head-latent-attention|Multi-Head Latent Attention]]
- [[architecture/sparse-and-efficient/moe|Mixture of Experts]]
- [[architecture/model-families/deepseek|DeepSeek]]
- [[inference/kv-cache-and-memory/kv-cache|KV Cache]]
- [[architecture/positional-encoding/rope|RoPE]]
- [[training/distributed-training/pipeline-parallel|Pipeline Parallel]]
- [[training/distributed-training/zero|ZeRO]]
- [[training/post-training/grpo|GRPO]]
- [[sources/papers/2024-deepseekmoe|DeepSeekMoE]]
- [[sources/papers/2024-deepseek-v3|DeepSeek-V3 Technical Report]]
