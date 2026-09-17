# ARC-Exocortex (AREx)

**An offline-first cognitive substrate for language-model agents in ARC-AGI-3.**

> **The language model is the agent. The harness is its exocortex.**

ARC-Exocortex is an experimental Python harness for building, running, tracing, and evaluating language-model agents in ARC-AGI-3 environments.

The central architectural principle is that the harness should **augment reasoning without replacing agency**.

The model owns interpretation, hypotheses, relevance, goals, strategy, exploration, recovery diagnosis, planning intent, and action selection.

The exocortex provides the deterministic machinery around that cognition: evidence storage, retrieval, requested computation, search, simulation, budget enforcement, action execution, verification, checkpointing, replay, branching, logging, and evaluation.

It never silently substitutes model intent with a hidden policy.

## Research hypothesis

ARC-Exocortex is built around a broader research question:

> **How much reasoning capability can a deterministic cognitive substrate provide around a weaker language model without taking agency away from the model itself?**

Rather than embedding intelligence into a conventional agent framework, ARC-Exocortex externalizes the parts of cognition that do not inherently require semantic judgment.

This allows experiments such as:

```text
Weak LM + Strong Exocortex
vs.
Strong LM + Thin Harness
```

while preserving a clear boundary between **model reasoning** and **deterministic infrastructure**.

The long-term goal is to measure whether structured memory, verification, planning tools, world-model execution, recovery, and externalized computation can substitute for some amount of model capability.

---

## Architecture principle

ARC-Exocortex follows a strict ownership boundary.

### The language model owns

* semantic interpretation;
* causal reasoning;
* hypotheses and belief meaning;
* relevance judgments;
* inferred goals;
* exploration decisions;
* planning intent;
* search strategy selection;
* resource-allocation strategy;
* error diagnosis;
* recovery strategy;
* action selection; and
* decisions about what information is worth acquiring.

### The exocortex owns

* immutable observations and transitions;
* exact state deltas;
* structured memory;
* retrieval requested by the model;
* deterministic arithmetic;
* budget accounting and enforcement;
* deterministic search and simulation;
* world-model execution;
* prediction verification;
* action authorization enforcement;
* checkpoints and branches;
* recovery trace reconstruction;
* replay;
* experiment logging; and
* deterministic evaluation.

The guiding rule is:

> **Harness computes. Agent interprets.**

---

## What is implemented

The current implementation includes:

* Strict, versioned Pydantic contracts for observations, decisions, actions, hypotheses, tasks, tools, plans, predictions, verification, recovery, and evaluation.
* A synchronous orchestrator with replaceable environment and model protocols.
* Official local ARC environment adaptation plus deterministic fake adapters.
* Schema-constrained model output with bounded, model-owned repair attempts.
* A local Ollama backend tested with `qwen3.5:9b`.
* Hash-chained append-only JSONL events and atomic checksummed checkpoints.
* Typed resource budgets for actions, model calls, tokens, search, simulations, time, and memory.
* Exact state deltas, prediction verification, action chunks, soft interrupts, and objective hard stops.
* Versioned beliefs and hypotheses.
* Persistent task DAGs.
* Declarative world models.
* Bounded BFS and A* search.
* Exploration accounting.
* Recovery branches and first-divergence reconstruction.
* Offline replay adapters.
* Deterministic evaluation.
* Typed ablations.
* Rebuildable SQLite/FTS5 indexes.
* Seeded aggregate statistics.
* JSONL, CSV, and optional Parquet exports.
* Hardware and model-asset manifests.
* Offline wheelhouse tooling.
* A thin Kaggle-parity notebook.

Raw events and the run manifest are authoritative.

SQLite indexes, summaries, metrics tables, and analytical exports are disposable derived artifacts that can be regenerated from the canonical trace.

---

## Cognitive loop

At runtime, ARC-Exocortex supports a loop conceptually similar to:

```text
Environment
    ↓
Observation
    ↓
Immutable evidence capture
    ↓
Context projection
    ↓
Language-model Cognitive Executive
    ↓
Decision / hypothesis / plan / tool request
    ↓
Deterministic exocortex operations
    ↓
Authorized execution
    ↓
Prediction verification
    ↓
Environment
```

The model decides **what something means and what should happen next**.

The exocortex makes sure the evidence, computation, execution, and resulting trace are reliable.

---

## Verification

Verification is deliberately strict.

Wherever a claim is mechanically checkable, verification is deterministic.

For example:

```text
predicted_state = T_hat(state, action)
actual_state    = environment.step(action)

mismatch = compare(predicted_state, actual_state)
```

The exocortex reports the discrepancy.

The model determines its meaning.

A failed prediction therefore does **not** automatically weaken a hypothesis or change the plan. Instead:

```text
prediction
    → deterministic verification
    → LM interpretation
    → semantic belief update
    → deterministic belief arithmetic
```

The exocortex may automatically block objectively invalid execution, but it never independently decides the replacement strategy.

---

## Action execution

The model controls execution granularity.

It may authorize:

```text
one action
a bounded sequence of explicit actions
```

An action chunk contains at most 64 explicit steps. Each step may carry a required
pre-action state hash and an expected outcome. The exocortex executes the
authorized sequence step by step while continuously validating that contract.

Execution can result in:

```text
CONTINUE
SOFT_INTERRUPT
HARD_STOP
```

A hard stop occurs when authorization becomes objectively invalid, such as:

* an action becoming illegal;
* a plan precondition failing;
* an invariant being violated;
* a hard budget limit being exceeded;
* the environment becoming terminal; or
* a model-defined execution tolerance being violated.

The central rule is:

> **The agent authorizes a chunk. The exocortex continuously validates that authorization.**

---

## Exploration

Exploration is treated as **budgeted evidence acquisition**.

The agent decides what uncertainty is worth investigating.

The exocortex can calculate deterministic quantities such as information weights, action costs, search costs, available budget, and measured reward.

Conceptually:

```text
exploration_value =
    decision_relevant_information
    + expected_goal_progress
    + hypothesis_discrimination
    - action_cost
    - risk_cost
    - compute_cost
```

Exploration continues while another probe is expected to improve the eventual decision enough to justify its marginal cost.

It may stop when:

* useful information gain saturates;
* the current executable path becomes sufficiently strong;
* the best remaining exploration opportunity is worse than acting;
* the exploration budget is exhausted; or
* further exploration is unlikely to change the policy.

The semantic judgment remains with the model.

---

## Recovery and backtracking

Failures are preserved as evidence.

The exocortex records:

* executed actions;
* predicted states;
* actual states;
* state deltas;
* active plan versions;
* active world-model versions;
* hypothesis snapshots;
* task state;
* budgets;
* checkpoints;
* branches; and
* mismatch events.

When a failure occurs, it can deterministically locate the earliest useful divergence:

```text
first point where prediction != observation
```

and reconstruct the surrounding context for the model.

The model then decides whether the failure came from:

* an incorrect world model;
* an incorrect hypothesis;
* a bad plan;
* failed execution;
* an incorrect goal assumption;
* or some interaction between them.

The exocortex reconstructs **how we failed**.

The model decides **why**.

A further distinction is important:

> **History is always rewindable. The environment may not be.**

Cognitive state can always be reconstructed from the trace. Physical actions are only reversed when the environment actually supports reversal.

---

## Research trace

Every meaningful operation is persisted as a typed event.

The trace may include:

* observations;
* raw states;
* parsed states;
* LM-visible context;
* model decisions;
* hypotheses;
* epistemic updates;
* task updates;
* plans;
* search requests;
* search results;
* tool calls;
* tool outputs;
* world-model versions;
* predictions;
* actions;
* state deltas;
* verification results;
* budget changes;
* interrupts;
* failures;
* recovery operations;
* checkpoints;
* branches;
* evaluation metrics; and
* run metadata.

Each run is associated with provenance including:

```text
experiment_id
run_id
harness commit
configuration hash
model identity
model digest
model generation settings
environment version
game / level
seed
budget configuration
enabled components
ablation configuration
Python and platform metadata
```

Hardware profiles and checksummed model-asset manifests are available as separate
deployment records; they are not silently inferred into every run manifest.

This makes trajectories suitable for later replay, ablation studies, failure analysis, and research-paper figures.

---

## Repository layout

```text
src/arc_agi_3/
  adapters/       Environment and inference boundaries
  cognition/      Beliefs, hypotheses, tasks, and workspace state
  context/        Exact retrieval and context projection
  contracts/      Serialized domain contracts
  deployment/     Hardware, assets, and offline entrypoints
  exploration/    LM-parameterized exploration calculations
  planning/       Bounded deterministic search
  recovery/       Evidence reconstruction and branch tracking
  research/       Derived indexes, exports, and aggregates
  runtime/        Episode orchestration and authorized execution
  trace/          Canonical events and checkpoints
  world_model/    Versioned declarative simulation

tests/             Unit, property, integration, replay, and golden tests
notebooks/         Local and Kaggle-parity experiment entrypoints
scripts/           Wheelhouse build and offline-install smoke tests
```

Competition data, upstream reference code, model assets, run artifacts, and research/planning documents are intentionally not versioned in this repository.

---

## Requirements

* Python 3.12
* [`uv`](https://docs.astral.sh/uv/)
* The official ARC-AGI-3 dataset and wheels for real environment runs
* Ollama with `qwen3.5:9b` for the current local Qwen workflow

Install the core development environment:

```bash
uv sync
```

Install the optional official ARC adapter dependencies:

```bash
uv sync --extra arc
```

Install optional analytics support such as Parquet export:

```bash
uv sync --extra analytics
```

---

## Deterministic smoke run

The fake environment exercises the complete runtime without network access or model weights.

```bash
uv run arc-agi-3 fake-run --output runs --run-id smoke
uv run arc-agi-3 verify-trace runs/smoke/events.jsonl
```

Build disposable research artifacts from the canonical trace:

```bash
uv run arc-agi-3 research index-trace \
  runs/smoke/events.jsonl \
  --output runs/smoke/events.sqlite3

uv run arc-agi-3 research export-run \
  runs/smoke \
  runs/smoke/metrics.jsonl
```

---

## Local Qwen workflow

`notebooks/arc_agi_3_local_qwen_kaggle_parity.ipynb` is the centralized local experiment entrypoint.

Its first code cell contains configuration for:

* paths;
* game;
* seeds;
* budgets;
* model settings;
* context size;
* repair count; and
* turn limit.

The notebook:

1. optionally installs the harness and ARC dependencies from attached wheel directories using `pip --no-index`;
2. resolves and records the exact Ollama model digest;
3. loads a bundled ARC environment in official offline mode;
4. composes the ARC, Ollama, structured-decision, and runtime adapters;
5. executes the episode and persists the complete research trace;
6. verifies the hash chain and LM/action causal boundary; and
7. builds disposable SQLite indexes and metrics exports.

Start the local model before running the notebook:

```bash
ollama pull qwen3.5:9b
ollama serve
```

Optional environment variables include:

```text
ARC_DATASET_ROOT
ARC_WHEELS
HARNESS_WHEELS
ARC_OUTPUT_ROOT
```

Defaults target the local competition bundle and switch to attached-wheel installation when `KAGGLE_KERNEL_RUN_TYPE` is present.

Ollama is currently a local-development backend.

A Kaggle kernel cannot use a model server running on the developer workstation. The production submission should retain the same harness interfaces while replacing the Ollama adapter with a backend loading attached model weights directly inside the competition environment.

---

## Offline packaging

Build the core wheelhouse:

```bash
./scripts/build_wheelhouse.sh
```

Verify that the harness can be installed without an external package index:

```bash
./scripts/offline_smoke.sh
```

Wheelhouses containing native dependencies are platform-specific.

The submission wheelhouse should therefore be built on a Linux environment compatible with the target Kaggle runtime.

Model weights and their checksummed manifest should be attached separately.

---

## Reproducibility contract

Every run directory contains:

```text
manifest
events.jsonl
checkpoints
derived artifacts
```

The normalized manifest describes the experiment.

`events.jsonl` is immutable and authoritative.

Checkpoints are atomic and checksummed.

Derived indexes and exports may always be rebuilt.

Recorded adapters allow model and environment results to be replayed offline, while the agency audit verifies that executed environment actions correspond exactly to actions previously authorized by the model.

This provides a hard research boundary between:

```text
what the LM decided
what the harness computed
what the environment did
what the evaluator measured
```

---

## Evaluation

Evaluation is completely outside the agent loop.

The evaluator consumes immutable traces and produces deterministic metrics.

Tracked quantities can include:

* ARC outcome;
* RHAE when the required competition baselines are available;
* environment action count;
* repeated actions;
* model calls;
* tokens;
* latency;
* search expansions;
* simulations;
* budget utilization;
* prediction error;
* verification failures;
* soft interrupts;
* hard stops;
* recovery attempts;
* recovery success;
* branch count; and
* completion efficiency.

The agent cannot alter how its run is scored or selectively suppress events from the evaluator.

---

## Validation status

The current baseline has been validated with:

* **57 passing tests**;
* **89.54% measured coverage**;
* strict mypy validation;
* Ruff linting and formatting;
* successful source and wheel builds;
* a fresh `pip --no-index` wheelhouse installation;
* a **20-event deterministic golden-trace replay**;
* an official offline `ls20` smoke run using local `qwen3.5:9b`; and
* a complete eight-cell notebook execution with **zero agency-boundary violations**.

The Qwen/`ls20` run was a **one-turn integration smoke test**.

It is not a solved-game claim and should not be interpreted as a benchmark result.

---

## Current limits

* Production model and quantization choices are not yet fixed.
* The local Ollama adapter is not the Kaggle inference backend.
* Official environment rewind is unavailable through the current adapter; recovery records unsupported restores instead of pretending they succeeded.
* Declarative world models and bounded BFS/A* are implemented.
* MCTS remains optional until experiments justify the additional complexity and compute.
* Learned world models are intentionally deferred.
* Human action baselines and private evaluator integration must be supplied by the competition evaluation environment before RHAE claims can be reported.

---

## Design philosophy

ARC-Exocortex is intentionally not an attempt to hide an increasingly capable deterministic agent behind a language-model interface.

The research question depends on preserving genuine language-model agency.

A useful implementation test is therefore:

> **If the LM were replaced with a random policy, would this component still be making an important semantic or strategic decision?**

If the answer is yes, that responsibility probably belongs to the model.

The exocortex should make reasoning **more reliable, persistent, computationally capable, and measurable** without deciding what the agent should believe or want.

That distinction is the core of the project.
