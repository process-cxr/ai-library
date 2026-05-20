---
title: "XR Lab"
---

<div style="padding: 1.8rem 0 1rem; border-bottom: 1px solid var(--lightgray); margin-bottom: 1.6rem;">
  <p style="letter-spacing: .16em; text-transform: uppercase; font-size: .78rem; color: var(--secondary); margin: 0 0 .5rem;">XR Lab · Large Model Knowledge System</p>
  <h1 style="font-size: 2.35rem; line-height: 1.12; margin: 0 0 .8rem;">一座面向大模型全栈的知识实验室</h1>
  <p style="font-size: 1.05rem; line-height: 1.75; margin: 0; color: var(--darkgray); max-width: 46rem;">从数学基础、模型架构、训练范式、推理系统到应用工程，把零散阅读、实验复现和源码理解沉淀成可检索、可链接、可长期演化的知识网络。</p>
</div>

> *"Attention is all you need." — but understanding is what we pursue.*

## Knowledge Map

> [!abstract] 主线：从原理到系统
> 沿着大模型的核心链路组织知识：先理解基础，再进入架构与训练，最后落到推理部署和应用评测。

- **[[fundamentals/|Fundamentals]]** — 数学与深度学习基础：线性代数、概率、信息论、神经网络基础、优化。
- **[[architecture/|Architecture]]** — 模型结构与家族：Transformer、Attention、位置编码、MoE、Mamba、LLaMA、Qwen、DeepSeek。
- **[[training/|Training]]** — 训练全流程：预训练、中训练、后训练、数据工程、分布式训练、Scaling Law。
- **[[inference/|Inference]]** — 推理与服务系统：解码、KV Cache、FlashAttention、量化、Serving、性能优化。
- **[[application/|Application]]** — 应用构建与评测：Prompting、RAG、Tool Use、Agents、Benchmark、LLM-as-Judge。

## Research Workbench

> [!tip] 副线：从资料与实践反哺知识体系
> 新论文、博客、课程、报告和代码阅读先进入工作台；稳定结论再回流到主题知识地图。

- **[[sources/|Sources]]** — 外部资料分析：论文、博客、课程、技术报告、官方文档。
- **[[projects/|Projects]]** — 实践与工程记录：论文复现、Demo、部署实验、源码阅读、排障经验。

## Learning Routes

### 01 · Build the Foundation

[[fundamentals/linear-algebra/|Linear Algebra]] → [[fundamentals/probability/|Probability]] → [[fundamentals/information-theory/|Information Theory]] → [[fundamentals/neural-network-basics/|Neural Network Basics]] → [[fundamentals/optimization/|Optimization]]

### 02 · Understand the Model

[[architecture/transformer/|Transformer]] → [[architecture/attention/|Attention]] → [[architecture/positional-encoding/|Positional Encoding]] → [[architecture/model-families/|Model Families]] → [[architecture/sparse-and-efficient/|Sparse & Efficient Architecture]]

### 03 · Train and Align

[[training/pretraining/|Pretraining]] → [[training/mid-training/|Mid-training]] → [[training/post-training/|Post-training]] → [[training/distributed-training/|Distributed Training]] → [[training/scaling/|Scaling]]

### 04 · Serve and Build

[[inference/decoding/|Decoding]] → [[inference/kv-cache-and-memory/|KV Cache & Memory]] → [[inference/serving-systems/|Serving Systems]] → [[application/rag/|RAG]] → [[application/agents/|Agents]] → [[application/evaluation/|Evaluation]]

## Navigation Protocol

- 使用 `Ctrl+K` 搜索概念、论文、工程记录或具体术语。
- 用 `[[双链]]` 在概念之间跳转；右侧 Graph View 可以观察知识连接密度。
- 主题目录回答“是什么、为什么、如何工作”；`sources/` 回答“这份资料讲了什么”；`projects/` 回答“我做了什么、结果如何”。
- 阅读资料或实验结论如果变成稳定知识，应回链到对应主题目录。

> [!note] Lab Status
> 这是一个持续生长的 AI knowledge garden：先建立清晰骨架，再逐步把每个节点写成可复用的知识单元。
