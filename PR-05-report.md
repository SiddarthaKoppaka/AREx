# PR-05 — Experiment ledger, repeated-call identity, epistemic learning

> **Status:** Implemented. Scope narrowed from the original design after
> finding a real correctness problem in the first draft - see below.

## What already existed (no duplication needed)

Checked before writing anything: `runtime/experiments.py:record_experiment`
already stamps every `EXPERIMENT` event with
`causal_refs=(decision.event_id, action.event_id)`, and
`RetrievalQuery.causal_ref`/`event_types` already let any tool query them
back. `evidence/repeats.py:detect_repeat` already classifies prior
attempts by `RepeatKind` (same-state/same-prediction/same-hypotheses).
This is most of "experiment ledger... prediction-based novelty and repeat
evidence... retrieval by experiment/hypothesis/state/version" already —
confirmed with a new regression test
(`test_experiment_ledger_retrieval.py`) rather than rebuilt.

## What's new: exact-result caching for pure tools

**A caught design flaw, not just caution:** my first draft keyed the
cache on `(tool_name, arguments, total_event_count)`, reasoning that any
harness state change happens via an appended event. That's true but
useless as a cache key: every tool call appends its own
`TOOL_REQUEST`/`TOOL_RESULT` events, so the event count always differs
between two calls to the very same tool — a repeat could *never* hit
under that design. Tried narrowing to "count of non-tool-bookkeeping
events," but that's actually wrong in the other direction:
`record_exploration_outcome`'s effect is a `TOOL_RESULT` write that
mutates `workspace.exploration`, so excluding all tool events would
produce a false cache hit for `evaluate_exploration` across a call that
genuinely changed the relevant state.

**Final design: cache only tools that are providably pure given their
arguments alone, no event-count signal at all.** `CACHEABLE_TOOLS =
{retrieve_artifact, simulate_world_model, search_world_model}`:
- `retrieve_artifact` reads a content-addressed, write-once store
  (`trace/artifacts.py:EventArtifactStore`) — same ref always returns the
  same content.
- `simulate_world_model`/`search_world_model` look up an immutable
  `WorldModel` by an exact `model_id`/`model_version` already pinned in
  their own arguments (`world_model/store.py` never mutates an existing
  version).

Deliberately excluded: `retrieve_events`/`retrieve_evidence`/
`check_prediction_history`/`inspect_frame_region` (genuinely depend on
how much history exists) and `evaluate_exploration` (depends on
separately-mutable exploration state unless the caller supplies every
input explicitly) — narrower but correct beats broader but wrong.
`archive_artifact`/`record_exploration_outcome` (writes) and
`consult_specialist`/`compare_specialist_reports` (model-generating) stay
permanently excluded per the gate: "do not dedupe... model-generating
specialist calls automatically."

`runtime/tools.py:run_tool` still records a full `TOOL_REQUEST`/
`TOOL_RESULT` event pair on a cache hit — the result payload carries
`cache_hit: true` and `cached_from_event_id` rather than concealing that
a repeat happened.

## Gate verification

- **"pure exact repeat avoids recompute while staying observable"** —
  `test_an_exact_repeat_is_a_cache_hit_and_stays_observable`: second
  identical `retrieve_artifact` call is `cache_hit: true`, visible in the
  trace, identical output.
- **"version/state change bypasses cache"** —
  `test_a_different_argument_bypasses_the_cache`: a different `keys`
  filter on the same artifact ref is a fresh computation.
- **"old negative outcome is queryable"** — confirmed via the existing
  machinery above, regression-tested.
- **"action policy remains LM-owned"** — unaffected by construction:
  caching touches none of the 3 non-action-authorizing tool types that
  read already-immutable data; actions are a separate code path never
  touched here.
- `test_a_write_tool_is_never_cached_even_on_an_exact_repeat` — explicit
  negative case for the write-exclusion rule.

## Validation

Full suite (375 tests), `ruff check`, `ruff format --check`, `mypy` all
pass. Golden fixture unchanged (no new config fields this time). No
change to action authorization or evaluation behavior.
