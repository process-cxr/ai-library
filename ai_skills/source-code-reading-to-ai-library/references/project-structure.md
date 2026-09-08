# Project Structure

Use this reference when creating or restructuring a source-reading project.

## Public Directory

```text
content/projects/<project>/
├── index.md
├── roadmap.md
├── 01-<first-runtime-trace>.md
├── <focused-topic-pages>.md
└── labs/
    ├── index.md
    └── <experiment-pages>.md
```

Number topic pages only when the numbers express a useful reading sequence. Do not reserve `00` for authoring instructions.

## Index

The index is the reader's system map. It should normally contain:

1. a precise explanation of the framework and the problem it solves;
2. why ordinary algorithm-level understanding is insufficient;
3. an architecture map organized by responsibilities;
4. one canonical end-to-end runtime chain;
5. boundaries between controller, data plane, execution backends and external systems;
6. grouped links to the project's topic pages and labs;
7. repository, pinned revision and implementation scope;
8. links to reusable topic knowledge.

The index should not become a second roadmap. Describe the knowledge landscape, not a sequence of assignments.

## Roadmap

The roadmap is the dependency-aware learning route. For each phase, record:

- the question that becomes answerable;
- prerequisites from earlier phases;
- source entrances;
- focused notes or labs produced;
- completion criteria.

Do not repeat the full project introduction. Link back to the index for repository and baseline information.

## Topic Pages

Each page should own one coherent boundary, such as:

- configuration resolution;
- data and batch lifecycle;
- algorithm-to-field mapping;
- controller and worker dispatch;
- process-group topology;
- model-engine backend comparison;
- checkpoint and weight synchronization;
- failure handling or asynchronous state.

A useful topic page usually includes a runtime context, concrete call chain, field or tensor flow, source anchors, invariants, failure modes and links back to the whole run. Use only the sections needed by that topic.

## Labs

A lab turns source interpretation into observable behavior. Record:

- hypothesis or question;
- pinned source revision;
- environment and resolved config;
- commands or instrumentation;
- trace schema;
- actual observations;
- acceptance criteria;
- conclusions that can be promoted to topic pages.

Keep planned observations visibly separate from measured results.

## Navigation Rules

- The index links to every main topic group.
- The roadmap links to the notes and labs produced in each phase.
- Topic pages link back to the main run when their role is otherwise ambiguous.
- Labs link to the claims they verify.
- Removed pages must have all incoming wikilinks updated in the same change.

