---
name: ai-knowledge-writer
description: AI Library 知识库写作规范。Use when Comate needs to create, rewrite, expand, or review Markdown knowledge notes in this Quartz-based AI knowledge base, including concept notes under content/fundamentals, architecture, training, inference, application, source-reading notes under content/sources, and practice records under content/projects. Trigger when the user asks to write knowledge points, organize notes, analyze papers/blogs/courses/reports, add Markdown/HTML-enhanced content, improve note quality, decide where a note belongs, or backfill topic notes from reading/practice notes.
---

# AI Knowledge Writer

Use this skill to write maintainable knowledge notes for the `ai-library` Quartz digital garden.

The goal is not to make every note long or structurally identical. The goal is to make each note precise, reusable, linkable, and suitable for long-term review.

---

## Core Principle

Write notes as a long-lived knowledge system, not as chat transcripts, progress logs, or temporary summaries.

- Write in Chinese by default; keep technical terms in English when they are standard.
- Use English page titles for sidebar consistency, and explain Chinese meanings in the body when helpful.
- Use Markdown first: headings, lists, callouts, tables, code fences, LaTeX, and wikilinks.
- Use inline HTML only for local visual enhancement, not for the main structure of a note.
- Structure serves content. Do not force every note into the same section order.
- Choose the lightest structure that preserves the knowledge accurately, but do not make important notes shallow.
- Add examples, trade-offs, boundaries, failure modes, and cross-links when they improve understanding.
- Do not add weak or empty sections just to satisfy a template.
- Do not add stage progress, “施工中”, temporary TODOs, or conversation references unless the user explicitly asks for a placeholder.

---

## Knowledge Layers

Place content by what the note is trying to preserve:

- `fundamentals/` — stable mathematical, probabilistic, information-theoretic, neural-network, and optimization concepts.
- `architecture/` — model structures, components, data flow, complexity, architecture variants, and model families.
- `training/` — training objectives, data, optimization, post-training, distributed training, scaling, and alignment methods.
- `inference/` — decoding, memory, KV cache, kernels, quantization, serving systems, compression, and performance.
- `application/` — prompting, RAG, tool use, agents, workflows, evaluation, and product-facing patterns.
- `sources/` — papers, blogs, courses, reports, docs, and talks as source-specific analysis.
- `projects/` — experiments, reproduction, deployment, code reading, troubleshooting, and practical validation.

Reading and project notes should backfill stable conclusions into the five main topic areas when conclusions become reusable.

---

## Note Type Selection

Before writing, classify the note. This classification determines the shape of the note; it is not a rigid template.

### 1. Mathematical or Theoretical Concept

Use for linear algebra, probability, information theory, optimization, and formal ML concepts.

Recommended sections:

- `## 概念界定` — what the concept describes and why it matters.
- `## 定义与记号` — formal definitions, variables, assumptions, and formulas.
- `## 推导与关系` — derivations or links to adjacent formulas when useful.
- `## 直观解释` — plain-language intuition that does not replace the formal definition.
- `## 示例` — concrete numeric example, mini calculation, or model scenario.
- `## 常见误解` — misconceptions and corrections.
- `## 相关概念` — wikilinks with relationship explanations.

Quality bar:

- Important formulas must explain each variable.
- Abstract concepts need at least one concrete example.
- If the concept appears in LLM training or inference, explain where it appears.

### 2. Architecture or Mechanism Note

Use for Transformer, Attention, RoPE, MoE, model families, and architecture components.

Recommended sections:

- `## 问题背景` — what limitation or need the mechanism addresses.
- `## 结构与数据流` — components, tensor shapes, forward path, and inputs/outputs.
- `## 关键机制` — core algorithmic idea or computation.
- `## 复杂度与代价` — compute, memory, context length, and scaling behavior when relevant.
- `## 变体与对比` — variants and why they differ.
- `## 设计取舍` — advantages, limitations, and failure cases.
- `## 相关概念` — links to training/inference/application implications.

Quality bar:

- Do not only describe the final architecture; explain why the design exists.
- Include shape or flow explanations when the note involves tensors.
- Connect architecture decisions to training and inference consequences.

### 3. Training Method Note

Use for pretraining, SFT, RLHF, DPO, GRPO, distillation, data engineering, scaling, and distributed training.

Recommended sections:

- `## 目标与问题` — what capability or bottleneck the method addresses.
- `## 数据与输入` — required data, labels, preference pairs, prompts, or batches.
- `## 训练目标` — objective, loss, reward, constraints, and key equations.
- `## 训练流程` — step-by-step process or pipeline.
- `## 关键超参` — knobs that change behavior.
- `## 失败模式与边界` — instability, data bias, reward hacking, overfitting, cost, or scaling limits.
- `## 相关概念` — links to fundamentals, architecture, inference, and application.

Quality bar:

- Always connect training objectives to underlying fundamentals when possible.
- Explain practical failure modes, not only mathematical objectives.
- Include when to use and when not to use the method.

### 4. Inference or Systems Note

Use for decoding, KV Cache, PagedAttention, FlashAttention, quantization, serving, batching, and performance.

Recommended sections:

- `## 系统问题` — latency, throughput, memory, cost, or deployment problem.
- `## 请求链路` — how a request flows through the system.
- `## 核心机制` — algorithm, kernel, cache layout, scheduler, or serving trick.
- `## 性能指标` — TTFT, TPOT, throughput, memory, utilization, quality impact.
- `## 工程取舍` — trade-offs across speed, memory, quality, cost, and complexity.
- `## 常见问题` — bottlenecks, pathological cases, debugging hints.
- `## 相关概念` — links to architecture, training, application, and projects.

Quality bar:

- Explain the bottleneck before explaining the optimization.
- Include metrics and what they mean operationally.
- Distinguish algorithmic ideas from framework-specific implementation details.

### 5. Application or Evaluation Note

Use for prompting, RAG, tool use, agents, workflows, benchmarks, LLM-as-judge, and evaluation systems.

Recommended sections:

- `## 使用场景` — what user or product problem this solves.
- `## Pipeline` — components and data flow.
- `## 关键设计点` — chunking, retrieval, tool schemas, memory, planning, prompts, judges, metrics.
- `## 示例` — realistic mini case or prompt/tool/evaluation example.
- `## 评测与反馈` — how to measure quality and observe failure.
- `## Failure Modes` — hallucination, retrieval miss, tool error, judge bias, brittle prompts.
- `## 相关概念` — links to inference, training, architecture, and projects.

Quality bar:

- Prefer concrete workflows over broad advice.
- Evaluation notes must explain what is measured and what is not measured.
- Agent/RAG notes should include failure modes and debugging signals.

### 6. Source Note

Use for papers, blogs, courses, reports, docs, and talks.

Recommended sections:

- `## 基本信息` — type, author/institution, date, link, related topic notes.
- `## 研究问题` — what question the source answers.
- `## 核心主张` — main claims.
- `## 方法与证据` — method, data, experiments, analysis, metrics.
- `## 关键结论` — durable conclusions.
- `## 局限与疑问` — limitations and open questions.
- `## 可沉淀到 Topic Note 的内容` — what should be backfilled into stable topic notes.

Quality bar:

- Keep source-specific claims separate from stable topic knowledge.
- Always link back to topic notes that should absorb conclusions.
- Do not over-generalize a single source into a universal rule.

### 7. Project or Code-Reading Note

Use for experiments, reproductions, demos, deployments, code reading, and troubleshooting.

Recommended sections:

- `## 目标` — what is being validated or built.
- `## 环境与输入` — codebase, model, data, hardware, command, or configuration.
- `## 步骤` — reproducible process.
- `## 结果` — observations, logs, metrics, screenshots, or outputs.
- `## 问题与排查` — issues and diagnosis.
- `## 结论与回补` — conclusions and topic notes that should be updated.

Quality bar:

- Make practical notes reproducible.
- Separate observation from conclusion.
- Link conclusions back to stable topic notes.

---

## Status Policy

Use `status` as an editorial state, not as a strict length indicator:

- `seed` — captured idea, outline, or minimal explanation. It can be short but should not contain misleading certainty.
- `growing` — readable, teachable, and linked. It should answer the core questions for future review.
- `mature` — comprehensive enough to serve as a reference: includes examples, boundaries, trade-offs, and source/project support.
- `processed` — for source notes whose reusable conclusions have been backfilled into topic notes.

Do not mark a note `growing` only because it has many sections. Mark it `growing` when it is actually useful.

---

## Writing Process

When writing or rewriting a note:

1. Decide the target directory and note type.
2. Identify the core question the note must answer.
3. Choose only the sections that serve that question.
4. Add definitions, mechanisms, examples, trade-offs, and links as needed.
5. Remove empty or weak sections.
6. Preserve frontmatter fields unless there is a reason to update them.
7. Keep the page title English for sidebar consistency.
8. Keep the body primarily Chinese, with English technical terms where standard.

---

## Quality Checklist

Before finishing a note, verify:

- The note can stand alone for future review.
- The first substantive section clearly states what the concept/method/source/project is about.
- Important formulas, mechanisms, or pipelines explain their variables and assumptions.
- At least one concrete example is included for abstract or difficult concepts.
- Trade-offs, boundaries, or failure modes are included when relevant.
- Links connect the note to upstream prerequisites and downstream applications.
- Source-specific claims are not mixed with stable topic knowledge.
- The note does not look templated simply for the sake of consistency.

---

## Markdown and HTML Policy

Markdown is the default. Inline HTML is allowed in `.md` files because Quartz/Markdown can render HTML blocks, but use it sparingly.

Use Markdown for:

- Main headings and note hierarchy.
- Paragraphs, lists, tables, callouts, code fences, LaTeX formulas.
- Wikilinks, backlinks, tags, and references.

Use inline HTML only for:

- Small visual cards or emphasis blocks that Markdown cannot express well.
- Compact comparison panels.
- Lightweight diagrams or SVG snippets.
- Landing-page-style sections in `index.md` pages.

Avoid inline HTML for:

- Entire articles.
- Core semantic structure.
- Large tables that Markdown can express.
- Content that needs frequent editing.
- Interactive UI that should be a Quartz component instead.

When using HTML:

- Keep the content readable when viewed as raw Markdown.
- Prefer semantic tags like `<section>`, `<div>`, `<figure>`, `<figcaption>`.
- Avoid inline JavaScript.
- Avoid hardcoded styles unless the visual need is local and small.
- Do not break surrounding Markdown spacing; leave blank lines before and after HTML blocks.
