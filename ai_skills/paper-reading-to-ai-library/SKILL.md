---
name: paper-reading-to-ai-library
description: Read AI/large-model papers, technical reports, blogs, courses, or official docs and turn them into AI Library knowledge-base notes. Use when the user asks Codex to read a paper or source, write a reading report, add it to the ai-library Quartz knowledge base, backfill durable conclusions into topic notes, or maintain linked source notes under content/sources.
---

# Paper Reading To AI Library

Use this skill to convert an external AI source into maintainable notes for the `ai-library` Quartz digital garden.

## Core Workflow

1. Identify the source type and access path.
   - Paper: arXiv, conference page, PDF, DOI, title, or local file.
   - Report: technical report, model card, release note, or benchmark report.
   - Blog/docs/course: official article, engineering blog, documentation, lecture, or talk.
   - If the source is remote and current metadata matters, browse or otherwise verify from primary sources.

2. Decide the reading depth.
   - `quick`: capture problem, key idea, contribution, and where it belongs.
   - `standard`: write a complete source note with methods, evidence, conclusions, limitations, and backfill targets.
   - `deep`: include mechanism-level explanation, equations or algorithms, experimental design, ablations, failure modes, and concrete topic-note updates.
   - If the user does not specify depth, use `standard`.

3. Choose the destination.
   - Papers: `content/sources/papers/`.
   - Technical reports/model releases: `content/sources/reports/`.
   - Blogs/articles/docs/courses/talks: `content/sources/blogs/` unless a more specific existing source subdirectory fits.
   - Create the destination `index.md` only when needed and follow existing frontmatter/date conventions.
   - Use stable English kebab-case filenames.

4. Write the source note in Chinese.
   - Keep the page title in English for sidebar consistency.
   - Keep technical terms in English when standard.
   - If the source was read from a local PDF path, use that path only for reading; do not write local absolute paths into public notes.
   - Preserve the public source URL in frontmatter or `## 基本信息` when the user provides one.
   - Use Markdown, wikilinks, callouts, tables, LaTeX, and code fences as appropriate.
   - Keep the source's research question, method, evidence, and limitations as the main narrative. Do not force a separate "knowledge sedimentation" essay at the end when the report itself already explains the paper.
   - Put knowledge-base links where they naturally support the discussion. If a closing link section is useful, keep it lightweight, e.g. `## 相关知识链接`, and list only related topic notes without re-explaining each concept.
   - Avoid chat transcript language, temporary TODOs, and progress-log phrasing.

5. Backfill stable knowledge.
   - Update relevant topic notes under `content/fundamentals`, `content/architecture`, `content/training`, `content/inference`, or `content/application` when the source contains reusable concepts, mechanisms, metrics, or trade-offs.
   - Keep source-specific claims in the source note; only promote durable conclusions to topic notes.
   - Link both directions where useful: source note to topic notes, and topic notes to source note when evidence or provenance matters.

6. Validate before finishing.
   - Dates must not be later than the current date.
   - `published` is the date Quartz displays in this repository unless the config changes.
   - Frontmatter should preserve existing fields unless there is a clear reason to change them.
   - Wikilinks should point to stable conceptual locations.
   - The note should be useful for future review without requiring the original conversation.

## Source Note Shape

Use only sections that serve the source. For a standard paper/report note, prefer:

```markdown
---
title: Paper or Report Title
created: YYYY-MM-DD
published: YYYY-MM-DD
modified: YYYY-MM-DD
type: source
status: seed|growing|processed
source_type: paper|report|blog|course|docs|talk
tags:
  - sources
  - relevant-topic
---

## 基本信息

- Source: ...
- Authors / Institution: ...
- Date: ...
- Link: ...
- Related topic notes: [[...]], [[...]]

## 研究问题

## 核心主张

## 方法与机制

## 实验与证据

## 关键结论

## 局限与疑问

## 相关知识链接
```

For closed or speculative sources, clearly separate public facts, paper claims, experimental evidence, and inference.
Treat the closing links section as optional navigation, not as a required analytical conclusion. Use it only when links do not fit naturally into earlier sections, and keep it concise.

## Topic Backfill Rules

- Architecture mechanisms go to `content/architecture/`.
- Training objectives, data, alignment, distillation, and scaling go to `content/training/`.
- Decoding, KV cache, quantization, serving, compression, and metrics go to `content/inference/`.
- Prompting, RAG, agents, tool use, and evaluation go to `content/application/`.
- Mathematical or ML prerequisites go to `content/fundamentals/`.
- Practical reproduction, code reading, deployment, and troubleshooting go to `content/projects/`.

When updating a topic note, prefer concise additions that improve the note's reusable knowledge: definitions, mechanism explanations, trade-offs, failure modes, examples, or links to evidence.

## Final Response

Report what was added or updated, with file links. Mention any source access limits, unresolved ambiguities, and whether topic notes were backfilled.
