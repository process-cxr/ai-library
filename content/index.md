---
title: "XR Lab"
created: 2025-12-20
published: 2025-12-20
modified: 2025-12-27
---

<div style="position: relative; overflow: hidden; padding: 2rem 1.45rem 2.15rem; border: 1px solid var(--lightgray); border-radius: 24px; background: linear-gradient(135deg, rgba(132, 165, 157, .16), rgba(255, 178, 102, .12)); margin: 1rem 0 1.8rem; box-shadow: 0 18px 45px rgba(40, 75, 99, .08);">
  <div style="font-family: 'Schibsted Grotesk', 'Avenir Next', 'Helvetica Neue', sans-serif; font-size: 2rem; line-height: 1; margin-bottom: .8rem; letter-spacing: .01em;">🍊 XR Lab 🍉</div>
  <p style="font-family: 'Schibsted Grotesk', 'Avenir Next', 'Helvetica Neue', sans-serif; letter-spacing: .18em; text-transform: uppercase; font-size: 1rem; line-height: 1.45; color: var(--secondary); margin: 0 0 .8rem;">Large Model Knowledge System</p>
  <h1 style="font-family: KaiTi, STKaiti, serif; font-size: 2rem; line-height: 1.18; margin: 0 0 .9rem; font-weight: 700;">一座面向大模型全栈的知识实验室</h1>
  <p style="font-family: KaiTi, STKaiti, serif; font-size: 1.16rem; line-height: 1.85; margin: 0; color: var(--darkgray); max-width: 50rem;">在这里，论文、源码、实验和工程经验会逐渐生长成一套可检索、可链接、可复用的大模型全栈知识系统。</p>
  <div style="margin: 1.35rem 0 0; padding-top: 1rem; border-top: 1px solid rgba(132, 165, 157, .22);">
    <p style="font-family: KaiTi, STKaiti, serif; font-size: 1.16rem; line-height: 1.7; margin: 0; color: var(--dark);">陈新冉 · 孔芮 · 李豫晨</p>
    <p style="font-family: 'Schibsted Grotesk', 'Avenir Next', 'Helvetica Neue', sans-serif; font-size: .98rem; line-height: 1.6; letter-spacing: .055em; margin: .08rem 0 0; color: var(--darkgray);">Xinran Chen · Rui Kong · Yuchen Li</p>
  </div>
</div>

> *"Attention is all you need." — but understanding is what we pursue.*

## Knowledge Map

> [!abstract] 主线：从原理到系统
> 沿着大模型的核心链路组织知识：先理解基础，再进入架构与训练，最后落到推理部署和应用评测。

- **[[fundamentals/|Fundamentals]]** — 基础理论：线性代数、概率、信息论、神经网络基础、优化。
- **[[architecture/|Architecture]]** — 模型架构：Transformer、Attention、位置编码、MoE、Mamba、LLaMA、Qwen、DeepSeek。
- **[[training/|Training]]** — 训练体系：预训练、中训练、后训练、数据工程、分布式训练、Scaling Law。
- **[[inference/|Inference]]** — 推理系统：解码、KV Cache、FlashAttention、量化、Serving、性能优化。
- **[[application/|Application]]** — 应用工程：Prompting、RAG、Tool Use、Agents、Benchmark、LLM-as-Judge。

## Research Workbench

> [!tip] 副线：从资料与实践反哺知识体系
> 新论文、博客、课程、报告和代码阅读先进入工作台；稳定结论再回流到主题知识地图。

- **[[sources/|Sources]]** — 资料分析：论文、博客、课程、技术报告、官方文档。
- **[[projects/|Projects]]** — 实践记录：论文复现、Demo、部署实验、源码阅读、排障经验。

## Learning Routes

### 01 · Build the Foundation

[[fundamentals/linear-algebra/|Linear Algebra]] → [[fundamentals/probability/|Probability]] → [[fundamentals/information-theory/|Information Theory]] → [[fundamentals/neural-network-basics/|Neural Network Basics]] → [[fundamentals/optimization/|Optimization]]

### 02 · Understand the Model

[[architecture/transformer/|Transformer]] → [[architecture/attention/|Attention]] → [[architecture/positional-encoding/|Positional Encoding]] → [[architecture/model-families/|Model Families]] → [[architecture/sparse-and-efficient/|Sparse & Efficient Architectures]]

### 03 · Train and Align

[[training/pretraining/|Pretraining]] → [[training/mid-training/|Mid-training]] → [[training/post-training/|Post-training]] → [[training/distributed-training/|Distributed Training]] → [[training/scaling/|Scaling]]

### 04 · Serve and Build

[[inference/decoding/|Decoding]] → [[inference/kv-cache-and-memory/|KV Cache and Memory]] → [[inference/serving-systems/|Serving Systems]] → [[application/rag/|RAG]] → [[application/agents/|Agents]] → [[application/evaluation/|Evaluation]]

## Navigation Protocol

- 使用 `Ctrl+K` 搜索概念、论文、工程记录或具体术语。
- 用 `[[双链]]` 在概念之间跳转；右侧 Graph View 可以观察知识连接密度。
- 主题目录回答“是什么、为什么、如何工作”；`sources/` 回答“这份资料讲了什么”；`projects/` 回答“我做了什么、结果如何”。
- 阅读资料或实验结论如果变成稳定知识，应回链到对应主题目录。

> [!note] Lab Status
> 这是一个持续生长的 AI knowledge garden：先建立清晰骨架，再逐步把每个节点写成可复用的知识单元。
