---
title: "DeepSeek V4: Towards Highly Efficient Million-Token Context Intelligence"
created: 2026-05-28
published: 2026-05-28
modified: 2026-10-08
type: source
status: processed
source_type: technical-report
area: sources
tags:
  - source
  - report
  - deepseek
  - moe
  - long-context
  - agent
  - on-policy-distillation
  - inference-system
aliases:
  - DeepSeek V4 Technical Documentation
  - DeepSeek V4 Report
source_url: https://arxiv.org/abs/2606.19348
publisher: DeepSeek-AI
report_date: "2026-04"
---

# DeepSeek V4: Towards Highly Efficient Million-Token Context Intelligence

## 基本信息

- 标题：[DeepSeek V4: Towards Highly Efficient Million-Token Context Intelligence](https://arxiv.org/abs/2606.19348)
- 作者：DeepSeek-AI
- 机构/来源：DeepSeek-AI
- 发表：arXiv:2606.19348v1
- 日期：2026-04
- 代码/数据：[DeepSeek-V4 模型集合](https://huggingface.co/collections/deepseek-ai/deepseek-v4)、[推理实现](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference)
- 相关主题：[[architecture/model-families/deepseek|DeepSeek]]、[[architecture/sparse-and-efficient/moe|Mixture of Experts]]、[[architecture/attention/hybrid-attention|Hybrid Attention]]、[[training/optimization/muon|Muon Optimizer]]、[[training/post-training/on-policy-kd|On-Policy Distillation]]、[[application/agents/agent|Agent]]、[[application/evaluation/benchmark|Benchmark]]

## 报告范围与整体方案

DeepSeek-V4 试图解决的问题很集中：当上下文长度进入百万 token 量级，普通 dense attention 的计算、KV cache 和推理服务成本会先于模型能力成为瓶颈。与此同时，reasoning 和 agent 任务又在不断增加 test-time compute，模型需要在更长的轨迹、更大的代码仓库和更多工具结果上保持状态。

所以 V4 的核心不是简单扩大参数，而是同时改造四个环节：

| 环节 | 主要变化 | 直接目标 |
|---|---|---|
| Attention | Compressed Sparse Attention（CSA，压缩稀疏注意力）与 Heavily Compressed Attention（HCA，重压缩注意力） | 把百万上下文的 attention 计算和 KV cache 压下来 |
| 深层结构与优化 | Manifold-Constrained Hyper-Connections（mHC，流形约束超连接）与 Muon optimizer | 提高深层信号传播和大规模训练稳定性 |
| 训练与推理系统 | 专家并行 overlap、TileLang、确定性 kernel、contextual parallelism、异构 KV cache | 让复杂架构可以训练、调试和服务 |
| 后训练 | specialist SFT/RL、multi-teacher On-Policy Distillation（OPD，on-policy distillation） | 将不同领域能力收敛到一个统一模型 |

### 模型变体与阶段产物

V4 v1 讨论两个 preview 模型：

| 模型 | Total parameters | Activated parameters | 主要定位 |
|---|---:|---:|---|
| DeepSeek-V4-Flash | 284B | 13B / token | 更低成本的长上下文与 agent 模型 |
| DeepSeek-V4-Pro | 1.6T | 49B / token | 更强的知识、推理和复杂 agent 任务 |

两者都支持 1M token context。total parameters 代表模型总容量和需要管理的权重规模，activated parameters 只代表每个 token 的主要计算路径，不能直接当成完整部署成本。

V4 的训练产物关系如下：

~~~text
DeepSeek-V4-Flash / Pro base model
  -> 32T / 33T token pre-training
  -> 4K -> 16K -> 64K -> 1M context training
  -> domain specialists: SFT + GRPO-based RL
  -> multiple math / code / agent / instruction experts
  -> multi-teacher full-vocabulary reverse-KL OPD
  -> unified DeepSeek-V4 model
  -> Non-think / Think High / Think Max modes
  -> FP4-aware deployment and agent serving
~~~

这里有一个容易混淆的阶段关系：**最终模型的 mixed RL 被 OPD 替代，但 specialist 分支仍然使用 SFT + RL。** RL 先分别把不同领域的专家训练出来，OPD 再让一个 student 在自己的 rollout 上吸收这些专家的输出分布，最终形成单一 unified model。V4 不是在线路由多个 specialist，也不是把多个 checkpoint 做参数平均。

## 模型架构与关键机制

### 从 DeepSeek-V3 继承的部分

V4 保留 DeepSeekMoE 和 Multi-Token Prediction（MTP，多 token 预测）路线，但对 MoE routing 做了几处调整：

- affinity score 从 V3 的 sigmoid 改为 `Sqrt(Softplus(.))`；
- 继续使用 auxiliary-loss-free load balancing，并保留很小的 sequence-wise balance loss；
- 去掉 V3 中每个 token 最多发送到固定节点数的约束，改由新的并行系统承担通信效率；
- 所有 Transformer block 都使用 MoE，最初三个 MoE layer 使用 Hash routing，根据 token id 的预定义 hash 函数选择专家；
- MTP 配置保持与 V3 一致，depth 为 1。

这说明 V4 并不是从 V3 重新设计一套模型，而是在 V3 的稀疏容量和训练目标上，把主要创新集中到残差连接、attention、优化器与系统实现。

### mHC：把 residual mapping 约束到稳定流形

Hyper-Connections（HC，超连接）先把 residual stream 的宽度扩展为 $n_{hc}$ 份，并用三个映射控制层间信息：输入映射 $A_l$、残差变换 $B_l$ 和输出映射 $C_l$。其更新可以写成：

$$
X_{l+1}=B_lX_l+C_lF_l(A_lX_l).
$$

普通 HC 的问题是，堆叠很多层后 residual mapping 可能出现数值不稳定。mHC 的关键改动是将 $B_l$ 约束到 doubly stochastic matrices 的集合，也就是 Birkhoff polytope：

$$
\mathcal{M}=\{M\mid M\mathbf{1}=\mathbf{1},\;\mathbf{1}^{T}M=\mathbf{1}^{T},\;M\geq 0\}.
$$

这会限制 residual transformation 的谱范数，并使深层堆叠时的信息传播更稳定。实际参数先生成 unconstrained raw matrices，再通过 sigmoid 和 Sinkhorn-Knopp normalization 投影到满足约束的参数空间。$A_l$ 与 $C_l$ 通过 sigmoid 保证非负且有界，$B_l$ 通过最多 20 次 Sinkhorn-Knopp 迭代得到 doubly stochastic matrix。

mHC 会增加 activation memory 和 pipeline stage 间通信。V4 使用 fused kernel、选择性 activation recomputation 和调整 DualPipe 1F1B schedule，将 mHC 的 wall-time overhead 控制在 overlapped pipeline stage 的 6.7%。

### Hybrid Attention：CSA 与 HCA 的分工

V4 将 CSA 和 HCA 交错放入 Transformer blocks。两者都压缩 KV cache，但压缩强度和访问方式不同：

- CSA 每 $m$ 个 token 压成一个 KV entry，再用 DeepSeek Sparse Attention（DSA，DeepSeek 稀疏注意力）从压缩 entries 中选 top-k；
- HCA 每 $m'$ 个 token 压成一个 entry，其中 $m'\gg m$，但不做 sparse selection，而是在更小的压缩序列上执行 dense attention；
- 两条路径都额外保留 sliding-window attention，避免压缩 block 内的局部依赖被完全抹掉。

所以 Hybrid Attention 不是“稀疏 attention 与 dense attention 二选一”，而是用不同的压缩倍率分别处理远程和局部信息：CSA 保留更细的、可选择的远程信息，HCA 用更激进的压缩提供低成本的全局交互。

#### CSA 的工作过程

CSA 先把每个 token 映射成 content KV 和 indexer KV，再以 $m=4$ 的压缩率形成 compressed KV entries。随后 lightning indexer 为每个 query 计算 compressed block 的 index score，只保留 top-k entries：Flash 使用 512，Pro 使用 1024。

保留下来的压缩 entries 进入 Multi-Query Attention（MQA，多查询注意力）主路径。CSA 还使用 grouped output projection，避免把大量 attention heads 的输出直接投影回 hidden dimension 带来过高计算成本。

#### HCA 的工作过程

HCA 使用更大的压缩倍率 $m'=128$，将更长的 token 区间聚合成单个 KV entry。它不使用稀疏 top-k，而是在压缩后的序列上进行 MQA。HCA 也使用 grouped output projection，因此 attention 的输入序列被压缩，输出投影成本也被控制。

#### 局部依赖与 KV cache

CSA/HCA 都保留一个 sliding window，V4 的窗口大小为 128。它解决两个问题：压缩 block 内尚未形成 compressed entry 的最新 token 仍然可以参与 attention；最近 token 往往对语言建模更重要，不能完全依赖粗粒度压缩。

KV cache 使用异构布局：CSA/HCA 的 compressed entries 进入 classical KV cache，sliding-window entries 和尚未完成压缩的尾部 token 进入 state cache。不同层的压缩率不同，因此不能假设所有层拥有相同数量、相同生命周期的 KV blocks。

#### 效率结果

在 1M context 下，相对 DeepSeek-V3.2：

- V4-Pro 的 single-token inference FLOPs 约为 27%，KV cache 约为 10%；
- V4-Flash 的 single-token inference FLOPs 约为 10%，KV cache 约为 7%。

报告还指出，混合 KV 存储使用 BF16 保存 RoPE 相关维度，其余部分使用 FP8；CSA indexer 的计算使用 FP4。以 BF16 GQA8、head dimension 128 为参照，V4 在 1M context 下的 KV cache 可压缩到该基线的约 2%。这些数字是特定模型、硬件和实现下的系统结果，不应单独理解为 CSA/HCA 的理论复杂度结论。

### Muon optimizer

V4 对绝大多数参数使用 Muon，对 embedding、prediction head、mHC 的 static bias/gating factors 和所有 RMSNorm 权重保留 AdamW。

Muon 的更新流程包含 momentum、Nesterov trick、Newton-Schulz orthogonalization 和 update RMS rescaling。V4 使用 hybrid Newton-Schulz：前 8 次迭代快速逼近正交矩阵，最后 2 次迭代使用更稳定的系数把 singular values 拉到 1 附近。

Muon 与 V4 架构的关系很紧：mHC、CSA/HCA 和大规模 MoE 都增加了训练稳定性的要求，优化器不只是一个可以独立替换的超参数。V4 仍然在 attention query 和 KV entries 上使用 RMSNorm，因此不需要额外采用 QK-Clip 来避免 attention logits 爆炸。

## 数据与预训练

### 数据构建

V4 在 DeepSeek-V3 数据基础上继续扩大训练语料，最终使用超过 32T tokens。数据重点包括数学与 programming corpus、更大规模的 multilingual data、经过过滤的 web data、scientific papers、technical reports，以及在 mid-training 阶段加入的 agentic data。

Tokenizer 仍保持 128K vocabulary，在 V3 tokenizer 基础上加入 context construction 所需的 special tokens，并继续使用 token splitting 和 Fill-in-the-Middle（FIM，中间填空）。与 V3 不同，V4 pre-training 使用 sample-level attention masking，以保证不同样本在 packing 后不会互相看到。

### 两个模型的结构配置

| 配置 | DeepSeek-V4-Flash | DeepSeek-V4-Pro |
|---|---:|---:|
| Transformer layers | 43 | 61 |
| Hidden dimension | 4096 | 7168 |
| Routed experts / layer | 256 | 384 |
| Shared experts / layer | 1 | 1 |
| Activated routed experts / token | 6 | 6 |
| Expert intermediate dimension | 2048 | 3072 |
| CSA compression rate $m$ | 4 | 4 |
| HCA compression rate $m'$ | 128 | 128 |
| CSA top-k | 512 | 1024 |
| mHC expansion $n_{hc}$ | 4 | 4 |
| Total parameters | 284B | 1.6T |
| Activated parameters | 13B | 49B |

Flash 前两层使用 pure sliding-window attention，之后交错使用 CSA/HCA；Pro 前两层使用 HCA，之后交错使用 CSA/HCA。两者都在所有 Transformer blocks 使用 MoE，前 3 个 MoE layers 使用 Hash routing。

### Training schedule 与稳定性

| 项目 | Flash | Pro |
|---|---|---|
| Training tokens | 32T | 33T |
| Maximum batch size | 75.5M tokens | 94.4M tokens |
| Peak learning rate | $2.7\times10^{-4}$ | $2.0\times10^{-4}$ |
| Final learning rate | $2.7\times10^{-5}$ | $2.0\times10^{-5}$ |
| Context schedule | 4K -> 16K -> 64K -> 1M | 4K -> 16K -> 64K -> 1M |

两者都在前 2000 steps warm up，主要参数使用 Muon，embedding、prediction head 和 RMSNorm 使用 AdamW。MTP loss weight 大部分训练时间为 0.3，进入 learning-rate decay 后改为 0.1；sequence-wise balance loss weight 为 0.0001。

V4 公开了两种训练稳定性机制：

1. **Anticipatory Routing**：当前 step 用当前 backbone 参数计算 feature，但用较早的历史参数预先计算 routing indices。loss spike 出现时临时激活，并配合短 rollback，额外 wall-time 约 20%。
2. **SwiGLU clamping**：将 SwiGLU 的 linear component 截断到 $[-10,10]$，gate component 的上界限制为 10，用来抑制 MoE 层中的异常值和 loss spike。

## 训练与推理基础设施

### Expert Parallelism 的细粒度 overlap

V4 将一个 MoE layer 拆成 Dispatch、Linear-1、SwiGLU、Linear-2 和 Combine，并把 experts 划分为多个 waves。当前 wave 做计算时，下一 wave 可以通信，上一 wave 可以发送结果，从而把通信和计算组成连续流水线。

报告在非融合 baseline 上报告一般 workload 1.50-1.73x speedup，在 RL rollout 和高速 agent serving 这类 latency-sensitive workload 上最高 1.96x。对应的 MegaMoE kernel 已作为 DeepGEMM 的组件开源。

### TileLang 与确定性 kernel

V4 用 TileLang 这种 Domain-Specific Language（DSL，领域特定语言）开发融合 kernel，重点不是单一算子优化，而是降低复杂 attention、MoE 和 mHC 的 kernel 开发成本。Host Codegen 将参数检查和 argument marshaling 从 Python 路径移到生成的 host code，报告称 CPU-side validation overhead 从几十/几百微秒降到 1 微秒以内。

另一个重点是 bitwise reproducibility：训练、post-training 和 inference 使用 batch-invariant、deterministic kernels，使同一个 token 不因所在 batch 的位置变化而产生不同输出，并减少非确定性 accumulation 对 loss spike 调查的干扰。

### Training framework

V4 继承 V3 的分布式训练基础设施，但为新组件增加了三类改造：

- Muon 与 ZeRO 的混合分片：Muon 需要完整 gradient matrix，不能直接套用为 element-wise optimizer 设计的 ZeRO。V4 使用 bucket assignment 和 knapsack 分配 dense matrices；MoE experts 则按 expert 独立优化，并用 BF16 同步部分 gradients、FP32 做本地求和。
- mHC 的 fused kernel 与 selective recomputation：重算大部分 hidden states 与 normalized layer inputs，把 mHC 的 wall-time overhead 控制在约 6.7%。
- Contextual Parallelism（CP，上下文并行）和 tensor-level activation checkpointing：处理压缩 attention 的跨 rank 边界，并让开发者只标注需要 checkpoint 的 tensor，由 TorchFX 自动追踪最小 recomputation graph。

### Inference 与 KV cache

V4 的 KV cache 不能再用单一 paged layout，因为 CSA、HCA、sliding-window attention 和未完成压缩的尾部 token 拥有不同的 cache policy。系统把 cache 分为 classical KV cache 和 state cache：前者存放 CSA/HCA compressed entries，后者存放 sliding-window entries 和尚未形成 compressed entry 的尾部状态。

对于共享 prefix，V4 支持 on-disk KV cache：CSA/HCA compressed entries 直接存储，SWA entries 则可以选择 full caching、periodic checkpointing 或 zero caching，在存储成本和重复计算之间取舍。

## 后训练与统一模型

### Specialist training

V4 的后训练不是直接在一个模型上混合所有领域的 RL，而是先独立训练 domain specialists。数学、coding、agent、instruction following 等领域分别使用 domain-specific prompts、SFT data 和 reward signals；每个 specialist 再使用 GRPO 做 RL。

reasoning effort 也作为训练变量。Pro 和 Flash 都提供三种模式：

| 模式 | 行为 | 典型用途 |
|---|---|---|
| Non-think | 快速、短回答 | 日常任务和低风险请求 |
| Think High | 更长的显式 reasoning | 复杂问题和中等风险决策 |
| Think Max | 更长 context、更低 length penalty，并注入额外 system instruction | 复杂推理边界和高难度任务 |

困难的开放任务不一定有可靠的 scalar verifier。V4 因此使用 rubric-guided RL 和 Generative Reward Model（GRM，生成式奖励模型）：actor 同时承担生成和评价角色，直接用 RL 优化其 judging capability，减少对大量人工 scalar preference annotation 的依赖。

Tool-use 方面，V4 引入了基于 `<|DSML|>` 的 XML-style tool-call schema，用结构化的 invoke/parameter 表达工具调用。报告认为这种格式能降低长字符串参数中的 escaping failure 和 malformed tool call。

V4 还调整了 interleaved thinking：在 tool-calling 场景中跨 user message 保留完整 reasoning history，使长程 agent 能持续使用之前的推理状态；普通对话仍清理前一轮 reasoning，避免无效上下文膨胀。对于把工具结果伪装成 user message 的 scaffold，这条 context path 可能不会被触发。

Quick Instruction 则把搜索触发、标题生成、query generation、authority/domain classification 等辅助任务编码成 special tokens，复用已经计算出的 KV cache，不再为每个辅助任务单独启动小模型并重复 prefill。

### Multi-teacher On-Policy Distillation

专家训练完成后，V4 用 multi-teacher OPD 把多个 specialist 的能力合并到一个 unified model。设专家策略为 $\pi^{E_i}$，student 为 $\pi_\theta$，目标是：

$$
L_{OPD}(\theta)=\sum_i w_iD_{KL}(\pi_\theta\|\pi^{E_i}).
$$

这里的 KL 是 reverse KL，student 在自己的 rollout 上学习，而不是直接把 teacher trajectory 当作 offline SFT 样本。不同任务上下文可以由对应 teacher 提供分布信号，因此多个物理上独立的专家最终被压回同一个参数空间。

V4 使用 full-vocabulary logit distillation，而不是只对 sampled token 做 KL 估计。作者认为后者方差较大，容易导致训练不稳定；完整 vocabulary 的 reverse KL 提供更稳定、也更忠实的 teacher distribution。该阶段使用十多个覆盖不同领域的 teacher models。

这与传统 model merging 不同：它不是平均权重或合并 task vector，而是让 unified student 在自己的状态分布上逐 token 对齐多个 teacher。

### 后训练基础设施

- **FP4 Quantization-Aware Training（QAT，量化感知训练）**：对 MoE expert weights 和 CSA indexer 的 QK path 使用 MXFP4；QK index score 从 FP32 量化到 BF16，top-k selector 速度约提升 2x，同时 KV entry recall 保持 99.7%。
- **Full-vocabulary OPD teacher scheduling**：teacher 权重放在集中式分布式存储中，按需加载并做 ZeRO-style sharding；只缓存 teacher 的最后一层 hidden states，再按 teacher head 重建 logits，避免同时物化多个 >100K vocabulary 的完整 logits。
- **容错 rollout**：每个请求使用 token-granular Write-Ahead Log（WAL，预写日志）记录生成 token，并保存未完成请求的 KV cache。中断恢复时继续 decoding，而不是从头生成，避免短轨迹因为更容易完成而被过度采样。
- **百万上下文 RL/OPD**：rollout metadata 与 heavy per-token fields 分离，metadata 用于全局 shuffle/packing，重字段通过 shared-memory loader 按 mini-batch 加载和释放。
- **DSec sandbox**：DeepSeek Elastic Compute 提供统一 Python SDK，底层支持 function call、container、microVM 和 full VM 四种 substrate，并在生产环境管理数十万并发 sandbox。轨迹日志支持 fast-forward、provenance 和 deterministic replay。

## 评测设置与结果

### Base model

Base model 评测覆盖 world knowledge、language reasoning、coding/math 和 long-context。主要结论是：V4-Flash-Base 以 13B activated parameters 在多数指标超过 V3.2-Base 的 37B active；V4-Pro-Base 则在几乎所有类别继续领先。

| 指标 | V3.2-Base | V4-Flash-Base | V4-Pro-Base |
|---|---:|---:|---:|
| MMLU-Pro | 65.5 | 68.3 | 73.5 |
| Simple-QA Verified | 28.3 | 30.1 | 55.2 |
| HumanEval Pass@1 | 62.8 | 69.5 | 76.8 |
| LongBench-V2 | 40.2 | 44.7 | 51.5 |
| Activated parameters | 37B | 13B | 49B |

这组对比支持“架构、数据和训练优化提高了参数效率”，但不能把提升全部归因给 CSA/HCA 或 Muon，因为模型规模、数据、训练 token 和多个系统变量同时变化。

### Final model 与 reasoning effort

V4-Pro-Max 在公开 benchmark 上的代表性结果包括：MMLU-Pro 87.5、SimpleQA Verified 57.9、Chinese-SimpleQA 84.4、GPQA Diamond 90.1、LiveCodeBench 93.5、Codeforces rating 3206、Terminal-Bench 2.0 67.9、SWE Verified 80.6、BrowseComp 83.4、Toolathlon 51.8。

Reasoning mode 对结果影响很大。V4-Pro 的 MMLU-Pro 从 Non-think 的 82.9 提高到 Max 的 87.5，HLE 从 7.7 提高到 37.7，Terminal-Bench 从 59.1 提高到 67.9；Flash 也表现出类似趋势。这里的差异来自 reasoning budget、length penalty、上下文窗口和 specialist 训练配置，不应简单理解为同一模型固定能力的单点分数。

### 一百万 token context

MRCR 1M 和 CorpusQA 1M 分别测试长上下文检索与更接近真实任务的问答。V4-Pro-Max 在 1M token 上的 MRCR 平均 MMR 为 83.5，CorpusQA accuracy 为 62.0；曲线显示 128K 内性能较稳定，超过 128K 后逐步下降，但到 1M 仍保留较强能力。

### Agent 与真实任务

代码 agent 评测使用最多 500 interaction steps 和 512K context，由内部 harness 提供 bash 与 file-edit 工具。V4-Pro 在 Terminal-Bench 2.0、SWE Verified、SWE-Pro、SWE Multilingual、BrowseComp、MCPAtlas 和 Toolathlon 上均有报告结果。

报告还使用约 200 个内部研发任务，覆盖 PyTorch、CUDA、Rust、C++ 的 feature development、bug fixing、refactoring 和 diagnostics，经过筛选保留 30 个任务。DeepSeek-V4-Pro-Max 在该内部 R&D coding benchmark 上 pass rate 为 67%，低于 Opus 4.5 Thinking 的 73%，高于 Sonnet 4.5 的 47%。

白领任务使用 30 个中文专业任务，覆盖金融、教育、法律和科技等 13 个行业，人工评价 task completion、instruction following、content quality 和 formatting aesthetics。V4-Pro-Max 相对 Opus 4.6-Max 的整体 non-loss rate 为 63%，优势主要集中在任务完成和内容质量，格式约束与视觉呈现仍有不足。

## 公开边界与复现条件

V4 v1 已经公开了完整得多的架构、预训练、后训练和系统信息，并提供部分 inference implementation。但以下内容仍不是完整公开 recipe：

- 完整的 32T/33T training corpus、数据 mixture、过滤规则和 mid-training agent data；
- specialist 的全部 SFT/RL prompts、reward 配置、GRM 训练细节和 teacher 权重；
- 训练集群规模、完整并行拓扑、所有 kernel 和生产调度配置；
- 真实任务评测的完整 prompts、内部 harness、sandbox 镜像和人工标注数据；
- OPD 的 teacher weights、teacher weighting 细节和完整 full-vocabulary distillation run。

因此，这份技术报告足以理解 V4 的系统设计和公开实验，但不足以让外部团队从零复现同规模模型。更现实的复现路径是：先复现 CSA/HCA inference kernel、mHC 小模型训练稳定性、Muon optimizer implementation、OPD 小规模多 teacher 合并，以及 DSec 类 sandbox 的任务回放机制。

还需要注意，许多 final benchmark 由内部评测框架得到，真实 agent 任务依赖特定 scaffold、工具、上下文管理和 sandbox。它们可以说明系统在作者设置下有效，但不能自动等价于所有外部 agent harness 的能力。

## 关键结论

1. **V4 的首要目标是降低百万上下文的边际成本。** CSA 用“压缩 + 稀疏选择”，HCA 用“更强压缩 + dense attention”，二者交错后共同降低 attention FLOPs 和 KV cache。
2. **模型结构和训练系统是一起设计的。** mHC、Muon、contextual parallelism、tensor-level checkpointing 和异构 KV cache 分别对应信号稳定、优化更新、长序列训练和服务状态管理，单独拿出某个模块无法代表完整收益。
3. **V4 的统一能力来自 OPD，而不是参数合并。** specialist 先独立训练，student 在自己的 rollout 上用 full-vocabulary reverse KL 吸收多个 teacher，最终得到单一 unified model。
4. **reasoning effort 是训练产物和推理配置共同决定的能力。** Non-think、High、Max 并非只是在推理时改变一个开关，而是对应不同的 RL length penalty、context budget、response format 和 system instruction。
5. **Agent 能力依赖完整基础设施。** 容错 rollout、百万上下文数据加载、FP4 QAT、DSec sandbox、trajectory provenance 和 deterministic replay 都是后训练能力的一部分，不是模型之外的附属工具。
6. **评测结果支持系统整体有效，但不支持单组件因果归因。** V4 同时改变了架构、数据、训练 token、优化器、后训练和 serving system，最终 benchmark 只能作为组合证据。

## 相关知识链接

- [[architecture/model-families/deepseek|DeepSeek]]
- [[architecture/sparse-and-efficient/moe|Mixture of Experts]]
- [[architecture/attention/hybrid-attention|Hybrid Attention]]
- [[training/optimization/muon|Muon Optimizer]]
- [[training/optimization/mixed-precision|Mixed Precision]]
- [[training/post-training/on-policy-kd|On-Policy Distillation]]
- [[training/post-training/grpo|GRPO]]
- [[application/agents/agent|Agent]]
- [[application/tool-use/tool-calling|Tool Calling]]
- [[application/evaluation/benchmark|Benchmark]]

## 参考资料

- [DeepSeek V4: Towards Highly Efficient Million-Token Context Intelligence](https://arxiv.org/abs/2606.19348)
- [DeepSeek-V4 model collection](https://huggingface.co/collections/deepseek-ai/deepseek-v4)
- [DeepSeek-V4 inference implementation](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference)
- [[sources/papers/2024-deepseek-v3|DeepSeek-V3 Technical Report]]
- [[sources/papers/2026-qwen3-coder-next-technical-report|Qwen3-Coder-Next Technical Report]]
