# Evidence Checklist

Use this checklist when reading the report and reviewing the draft. It is not a mandatory public report section.

## Stage and Artifact Audit

- Does each checkpoint have a clear predecessor, training purpose, and downstream use?
- Are temporary teachers, reward models, adapters, and draft modules separated from inference-time components?
- Does "expert" mean a routed network block, a task-specific checkpoint, or an external agent?
- Are parameter updates distinguished from dataset generation, rejection sampling, evaluation, and inference-time tool execution?
- Are independently trained expert branches distinguished from a sequential curriculum?
- Is an unspecified consolidation method still unspecified, rather than filled in with a familiar SFT or on-policy distillation recipe?

## Data and Budget Audit

- What is counted: unique documents, repositories, tasks, environment snapshots, trajectories, or token exposures?
- Are reported corpus sizes the full training budget or one subset?
- Are successful trajectories only one filtered view of a larger task pool?
- Are deduplication, leakage prevention, evaluation exclusion, sample boundaries, and attention/loss masks disclosed separately?
- Do context length, total trajectory length, output length, maximum turns, and wall-clock timeout have distinct meanings?
- Does the cost include teacher inference, failed environment builds, evaluation, ablations, and pre-existing model training?

## Result Audit

For every table or important comparison, recover:

1. Source section, table, or figure and the report version.
2. Model identity: base, adapted, specialist, merged, or final instruction model.
3. Evaluation setting: dataset split, scaffold/version, tools, decoding, retries, context/output budget, and metric.
4. Baseline comparability: changed variables, fixed data/compute, identical adapters or harness where applicable.
5. Strength of evidence: isolated ablation, coupled-system comparison, observational curve, or author explanation.

Do not label aggregate task completion "format accuracy", treat percentile as accuracy, compare Pass@1 to best-of-N, or turn a parameter-count advantage into a latency claim.

Numerical scale must remain explicit: percentages versus percentage points, total versus active parameters, unique tokens versus repeated exposures, and peak versus sustained throughput. Verify suspicious numbers against the original table layout, not only extracted text.

## Generalization and Reproducibility Audit

- Which domains, languages, environments, scaffolds, and model sizes were actually evaluated?
- Are curves from proxy experiments being generalized to the production run?
- Are final-system improvements incorrectly attributed to one data source or training stage?
- Does a verifier establish task success or merely success on its particular tests?
- Is an environment guard a complete isolation mechanism or a heuristic for an observed exploit?
- Which architecture settings, mixtures, losses, hyperparameters, artifacts, and infrastructure remain unavailable?

Use concrete unresolved items, not a generic statement that "more experiments are needed". Identify the minimum evidence needed to reproduce a local conclusion separately from the much larger requirements to reproduce the full model.
