---
name: source-code-reading-to-ai-library
description: Read and analyze large software repositories and turn them into durable AI Library source-reading projects. Use when the user asks to study a framework or codebase, establish an index and roadmap under content/projects, trace a real runtime path, explain control/data/tensor/process-group flows, compare execution backends, create reproducible labs, or update existing source-reading notes from code evidence.
---

# Source Code Reading To AI Library

Use this skill to convert a software repository into a reader-facing source-reading project in the `ai-library` Quartz knowledge base.

This skill governs repository analysis, evidence collection and project information architecture. Follow the repository's general knowledge-writing conventions for prose and Markdown style; do not duplicate generic concept-note or paper-reading workflows here.

## Core Principles

- Read a real execution path before describing the repository as a whole.
- Organize notes around runtime questions and ownership boundaries, not directory order.
- Keep public pages reader-facing. Authoring rules and AI maintenance instructions belong in this skill, not in a numbered knowledge chapter.
- Pin source claims to an explicit repository revision.
- Separate code-visible behavior, runtime-tested behavior and engineering interpretation.
- Preserve user changes in both the knowledge repository and source repository.
- Do not expose local filesystem paths, credentials, private repository names, internal data, hostnames or company-specific details in public notes.

## Workflow

### 1. Inspect Both Repositories

Identify:

- the AI Library workspace;
- the source repository;
- repository-local `AGENTS.md` or equivalent instructions;
- source branch, HEAD commit and remotes;
- current working-tree changes in both repositories;
- the public GitHub URL to use in notes.

Treat the source repository as read-only unless the user explicitly requests instrumentation or code changes. Never discard unrelated changes.

### 2. Define a Fixed Reading Baseline

Choose one concrete runtime configuration before tracing:

```text
entry command
  + algorithm or workload
  + training / inference backend
  + distributed topology
  + representative data path
  + fixed source revision
```

Prefer a minimal path that still closes the system loop. Record the public repository, full commit, branch at reading start and scope on the project `index.md`. Link source files to the fixed commit rather than a moving default branch.

When upstream has multiple generations, identify the primary implementation first. Treat deprecated paths as historical mappings unless the user specifically wants them.

### 3. Build Reader-Facing Project Architecture

Use:

```text
content/projects/<project>/
├── index.md
├── roadmap.md
├── focused-topic-pages.md
└── labs/
```

- `index.md` explains what the project is, why its architecture matters, the system map, the canonical runtime chain, the note map and the fixed reading baseline.
- `roadmap.md` contains only learning order, dependencies, core questions, source entrances, expected notes and completion criteria.
- Topic pages explain one runtime boundary, mechanism or comparison.
- Labs preserve commands, environment, traces, observations and acceptance criteria.

Do not publish a `00-conventions` page for AI-facing templates, evidence policy or maintenance instructions. See [Project Structure](references/project-structure.md) before creating or restructuring a project.

### 4. Trace a Runtime Path

Start at a runnable script, CLI, test or service request and trace downward. Maintain several synchronized ledgers when relevant:

```text
control flow     caller -> callee -> dispatch -> worker
configuration    default -> override -> resolved value -> runtime object
data flow        producer -> field -> representation -> consumer -> cleanup
tensor flow      shape -> dtype -> device -> mask -> layout transition
topology         world rank -> process groups -> local ownership -> communication
state/version    checkpoint -> policy version -> cache/state -> synchronization
```

For an ML training framework, follow at least:

```text
entry / config
  -> distributed initialization
  -> dataset / dataloader
  -> model construction
  -> forward / objective
  -> backward / communication
  -> optimizer
  -> checkpoint / evaluation
```

For an RL or agent framework, additionally follow:

```text
prompt
  -> rollout / environment
  -> trajectory representation
  -> reward
  -> old / reference / current policy data
  -> advantage / returns
  -> actor / critic update
  -> rollout weight synchronization
```

### 5. Mark Backend Boundaries

Do not generalize one backend's local behavior into a framework-wide rule.

Separate:

- controller- or protocol-level invariants;
- backend-specific batch mapping;
- process-group topology;
- forward-backward schedule;
- gradient and optimizer communication;
- checkpoint and weight-export behavior.

When comparing backends, hold the controller-level input fixed and explain how each backend consumes it. Distinguish training parallel dimensions from rollout or serving parallel dimensions.

### 6. Maintain Evidence Discipline

Read [Evidence and Tracing](references/evidence-and-tracing.md) when making source-backed claims, designing a lab, comparing backends or updating the pinned revision.

Use three evidence labels where the distinction matters:

- **Source fact**: directly visible in the pinned source revision.
- **Tested behavior**: reproduced by a unit test, minimal run, trace or profiler.
- **Engineering interpretation**: inferred design intent, trade-off, risk or expected behavior.

Do not present a config comment, symbol name, stale documentation or static code path as tested runtime behavior.

### 7. Write Source Anchors

Use public links in knowledge pages:

```text
https://github.com/<owner>/<repo>/blob/<full-commit>/<path>#L<line>
```

- Link to functions or compact source regions instead of pasting large implementations.
- Keep short code blocks for call chains, field transformations, tensor layouts and formulas.
- Never write a local source path into a public page.
- Recheck line anchors whenever the pinned revision changes.

### 8. Validate the Result

Before finishing:

1. search for links to removed or renamed pages;
2. check Markdown/frontmatter and unclosed code fences;
3. run `git diff --check`;
4. build Quartz;
5. inspect the diff and confirm unrelated user changes remain untouched;
6. state which conclusions are still static source analysis rather than runtime validation.

A topic page can move from `seed` to `growing` after its central call chain or mechanism is readable and source-grounded. Do not mark a lab complete until its required run or trace has actually executed.

## Writing Boundaries

Keep framework-specific findings in `content/projects`. Backfill stable, reusable mechanisms into `content/training`, `content/inference`, `content/architecture`, `content/application` or `content/fundamentals` only when they survive beyond the source repository.

Avoid:

- chat-style remnants such as “建议”“你可以” or conversation history;
- empty templates and repetitive completion checklists in reader-facing pages;
- broad project claims derived from one example;
- presenting local machine assumptions as framework requirements;
- documenting private implementation details in a public knowledge base.

