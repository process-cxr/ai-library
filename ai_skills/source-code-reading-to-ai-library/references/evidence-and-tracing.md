# Evidence and Tracing

Use this reference for source-backed claims, runtime experiments, backend comparisons and revision updates.

## Pin the Source

Record:

```text
public repository
full commit SHA
branch at reading start
reading date
primary implementation path
deprecated or excluded paths
```

Use immutable commit links in published notes. A branch link may be used for the repository homepage, but not as the sole evidence for a line-level claim.

## Evidence Classes

### Source Fact

A source fact is directly supported by the pinned code:

- a function calls another function;
- a field is produced or consumed;
- a branch is selected by a resolved config value;
- a process group is initialized with explicit dimensions;
- a cleanup call occurs after a particular stage.

Trace beyond names and comments. Confirm the caller, actual branch and downstream consumer.

### Tested Behavior

Tested behavior requires executable evidence:

- a passing unit or integration test;
- a minimal end-to-end run;
- a structured field trace;
- profiler or communication output;
- checkpoint inspection;
- reproducible logs with the resolved config.

Record the environment and command. Do not promote expected shapes, timing or memory behavior to this class before execution.

### Engineering Interpretation

Interpretation explains:

- why an abstraction likely exists;
- performance and memory trade-offs;
- possible race, staleness or correctness risks;
- how a design compares with another backend;
- what a field means for training behavior.

Tie interpretation to source facts and identify what experiment would falsify it.

## Trace Ledgers

### Call-Chain Ledger

```text
stage
caller
callee
execution process or worker
sync / async boundary
source anchor
```

### Field Ledger

```text
field
producer
consumer
shape or per-row lengths
dtype
device
mask
storage location
lifetime and cleanup
```

### Parallel Topology Ledger

```text
world size
logical role
DP / TP / PP / CP / SP / EP dimensions
rank coordinates
process groups
collective or point-to-point operation
tensor ownership before and after communication
```

Never infer training DP from rollout TP. Ask the active training backend for its data-parallel group and size.

### State and Version Ledger

```text
state owner
creation point
policy or checkpoint version
mutation point
synchronization boundary
recovery behavior
cleanup
```

This is required for rollout/training overlap, replay buffers, caches, parameter transport and asynchronous systems.

## Backend Comparison

Use the same controller-level input for both backends:

```text
same global batch and fields
  -> backend A local mapping
  -> backend B local mapping
```

Compare:

- topology construction;
- data dispatch and local sample ownership;
- micro-batch formation;
- forward-backward scheduling;
- gradient synchronization;
- optimizer sharding;
- checkpoint layout;
- parameter export and rollout synchronization.

A shared API does not imply identical local tensor shapes or communication.

## Revision Updates

When changing the source baseline, first record:

```text
old revision
new revision
affected notes
renamed or removed symbols
changed field contracts
changed runtime semantics
tests or labs that must rerun
```

Update links only after mapping old and new behavior. Preserve historical conclusions when the new implementation is not semantically equivalent.

## Privacy and Publication

Before publishing:

- replace local filesystem paths with public repository links;
- remove credentials, hostnames, cluster identifiers and private endpoints;
- generalize internal datasets, model names and company-specific systems when they are not essential;
- never paste proprietary source or logs into a public note;
- log metadata rather than complete prompts, tool results or user data.

