---
name: technical-report-reading-to-ai-library
description: Read model and system technical reports into detailed AI Library notes covering architecture, data, training stages, infrastructure, deployment, and evidence. Use for multi-component model reports or official technical documentation, even when archived as papers. Use paper-reading-to-ai-library for a single-method research paper; do not treat original research plans or cross-paper syntheses as external technical reports.
---

# Technical Report Reading To AI Library

Explain how the reported model or system is assembled, trained, evaluated, and used. A technical report is not a collection of disconnected method summaries. Select this workflow by the source's scope, not its title or page count.

## Shared Conventions

Use [source-note conventions](../paper-reading-to-ai-library/references/source-note-conventions.md) for metadata, affiliations, chronology, Chinese writing, abbreviations, citations, and repository validation. Install this skill together with `paper-reading-to-ai-library`; both use that one maintained reference.

- Preserve an existing note's path, aliases, `source_type`, and chronology unless the user requests a migration. Reading strategy and storage category are separate decisions.
- An arXiv technical report already in `content/sources/papers/` remains there. New standalone technical documentation belongs under `content/sources/reports/`, following local conventions.
- Do not silently change a `paper` into a `report`: homepage recommendations and PDF readers may filter by `source_type`.
- A model card or release note may warrant this workflow, but its disclosure limits must remain visible. Do not invent a full training recipe from a short announcement.

## Reading Workflow

### 1. Establish the source and coverage

Read the primary report, including relevant appendices. Prefer a PDF or document file explicitly provided by the user. **Do not proactively download remote PDFs or large report files.** If only a URL is available, record the link and ask the user to provide the file before doing a detailed report; public HTML metadata or an official page may be used for basic identification, but not as a substitute for the full report. Use official code, configurations, or model cards to clarify an implementation only when needed, and identify facts that come from those additional sources.

Before drafting, make a private coverage map of the report's sections, figures, tables, and appendices. Record which ones cover architecture, datasets, training, systems, evaluation, release artifacts, and limitations. This map is a working aid, not a section of the published note.

If the source cannot be accessed completely, explain the access limit. Do not present a reorganization of an old note as a fresh verification of the original report.

### 2. Reconstruct the whole system

Identify the model variants and the lineage of checkpoints before explaining individual modules. Distinguish:

- the starting model from intermediate checkpoints and released models;
- layers or experts inside one model from separately trained teacher or specialist models;
- trainable, frozen, shared, task-specific, temporary, and discarded components;
- model weights from the scaffold, tools, environments, verifier, and serving runtime.

Use a compact pipeline or table to show the dependencies. A unified final model, a backbone plus adapters, and a routed ensemble are different deployment products. State which one is actually supported by the report.

### 3. Trace data and training stages

For each stage, establish its input checkpoint, data, update target, objective, feedback mechanism, disclosed budget, and output artifact. Explain how the output becomes the next stage's input.

Keep raw documents, unique repositories, task instances, environments, trajectories, training exposures, and tokens distinct. Record denominators and stage scope next to important quantities. Preserve differences between natural and synthetic data, offline filtering and online reward, training and evaluation environments, and real versus reported reproducibility.

Do not infer loss masking, optimizer settings, mixture ratios, adapter freezing, KL direction, on-policy sampling, or distillation algorithms when the source does not disclose them. State the specific missing information where it affects interpretation.

### 4. Explain mechanisms and systems together

For each important mechanism, explain the constraint it addresses, its location, inputs and outputs, interactions with other components, training-versus-inference behavior, and supporting evidence. Expand the key mechanisms; reference foundational notes for prerequisite detail instead of repeating whole textbooks.

Keep hardware topology, parallelism, precision, activation and optimizer memory, communication, stability, and checkpointing under training infrastructure. Keep rollout orchestration distinct from parameter-update infrastructure. Keep inference context, weight storage, active compute, latency, throughput, and environment execution cost distinct.

Total parameters and active parameters are not measured deployment costs. GPU-hours, dollar estimates, and wall-clock time need scope, hardware, workload, and exclusions.

### 5. Audit evidence and release scope

Use [the evidence checklist](references/evidence-checklist.md) before drafting conclusions. Separate final-system benchmarks, component ablations, scaling observations, and efficiency measurements. Match every result to its model version, training stage, scaffold, metric, and budget.

For a claim supported only by the authors' interpretation, say so. For a component without an isolated ablation, explain its intended role without assigning it a numerical share of the final gain. Disclose contradictions instead of selecting whichever number fits the narrative.

Distinguish open weights, inference code, training code, data, environments, evaluation harnesses, and licenses. A report describing a component does not establish that it was released. Avoid time-sensitive claims of current availability unless checked.

### 6. Write and validate

Organize a single substantial note around the system's logic. Use only applicable sections; change their order when data or environment construction is the prerequisite for understanding training. A useful outline is:

```markdown
## 基本信息
## 报告范围与整体方案
## 模型架构与关键机制
## 数据与环境构建
## 训练流程与阶段产物
## 系统实现与推理部署
## 评测设置、结果与归因
## 公开边界与复现条件
## 关键结论
## 参考资料
```

Replace generic headings with domain-specific ones where appropriate. Detailed mid-training and post-training sections may be separate. Do not repeat the same pipeline under three headings. Split into multiple notes only when independently reusable material warrants it, not because the source is long.

Keep the prose academic and self-contained, without instructions to the reader, conversation remnants, or a forced connection to the user's latest unrelated paper or project. Include natural topic links. Backfill only durable, source-supported conclusions; do not expand unrelated notes just to meet a template.

Validate the coverage map against the final note, check links and formula rendering, and run the Quartz build after changes. Preserve the previous homepage recommendation membership and order unless a requested migration intentionally changes them. Report the edited note, skill installation state, and any unresolved source gaps.
