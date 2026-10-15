# PR-04 — Programmatic model replay, counterexamples, version lineage

> **Status:** Implemented, built directly on PR-03's sandbox executor.

## What was built

`src/arc_agi_3/executable_model/`:
- `replay_contracts.py` — `TransitionCase` (before/action/after embedded
  directly alongside their source event-id refs, plus a `train`/`holdout`
  split label), `CaseResult`, `ModelTestReport` (per-split accuracy,
  mismatch/error counts, counterexample case IDs).
- `replay_cases.py` — `build_transition_cases()` reuses
  `evidence/history.py`'s existing `transition_records`/`.observations()`
  rather than re-deriving before/action/after a second way; the
  train/holdout split is a deterministic, seeded shuffle, never an LM
  choice.
- `replay.py` — `replay_candidate()`: for each case, runs
  `encode_observation` on both `before`/`after` and `transition` on the
  encoded `before` + action, all through PR-03's sandbox, then compares
  predicted vs. actual encoded state. Every failure mode becomes a
  `CaseResult(status="error", error_category=...)`, never an exception.
  **Reads cases, returns an in-memory report — never appends to the
  trace.** That's what makes "no synthetic result written as a real
  observation" true by construction, not by a runtime check.
- `registry.py` — `ExecutableModelRegistry`, mirroring
  `world_model/store.py`'s `WorldModelStore` shape: enforces a strict
  parent-version chain per `model_id`, lets competing `model_id`s coexist.

## Gate verification (handoff.md's exact required items)

Built a real recorded trace (`FakeShiftEnvironment`, alternating actions
1/2 across 16 turns) via the existing `ReactiveModel`/`build_runner`
machinery — no new test fixture mechanism invented.

- **"a deliberate false model is falsified"** —
  `test_a_deliberately_wrong_model_is_falsified_on_holdout`: a candidate
  that ignores the action argument gets `holdout_accuracy < 1.0` with
  non-empty counterexamples.
- **"revised candidate obtains new version and improves held-out
  accuracy"** — `test_a_revised_candidate_improves_holdout_accuracy_over_its_parent`:
  v1 (wrong) registered, v2 (correct, `parent_version=1`) registered;
  `v2.holdout_accuracy > v1.holdout_accuracy`, lineage enforced by the
  registry.
- **"no synthetic result written as a real observation"** —
  `test_replay_never_appends_to_the_trace_or_changes_its_content`: the
  run's own `events.jsonl` file content is byte-identical before and
  after replay.

Also: a correct candidate matches every case (train and holdout), and
`test_executable_model_registry.py` covers lineage/duplicate-version/
coexistence rules independent of replay.

## Explicitly not done

No CLI command and no wiring into the live decision loop yet — per
PR-03's own note, that integration is PR-06's job (the full scientist-loop
fixture), once there's a reason for the LM to see these reports turn by
turn rather than as an offline analysis tool.

## Validation

Full suite (368 tests), `ruff check`, `ruff format --check`, `mypy` all
pass. Golden fixture unchanged (no new config fields this time). No
change to action authorization or evaluation behavior.
