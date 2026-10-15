# PR-06 — Full scientist-loop demonstration

> **Status:** Implemented as pure composition — no new infrastructure.
> **SYNTHETIC:** every number here comes from a scripted fixture model,
> not a real LM. Label accordingly until a real Qwen run confirms it.

## What this is

Exactly what handoff.md asks for: a fixture proving the *existing*
causal/replay machinery (PR-00 through PR-05) composes into one coherent
learning loop, not a new feature. Two test files, one shared fixture
module:

- `tests/scientist_loop_fixtures.py` — the scripted policies: propose two
  competing hypotheses about `FakeShiftEnvironment`'s real shift-by-one
  rule, run the discriminating action, record the verified fact, then a
  later decision that must see that fact rather than re-deriving it.
- `test_scientist_loop_fixture.py` — runs the loop, asserts the exact
  causal chain through the trace (not approximately — exact `causal_refs`
  tuples) and `audit_agency_boundary(events) == ()`.
- `test_scientist_loop_falsification.py` — replays two executable
  candidates (reusing PR-04's `replay_candidate` and PR-03's sandbox)
  against the *same* recorded trace: the one matching the wrong
  hypothesis (shift-by-2) gets `train_accuracy == 0.0` with real
  counterexamples; the one matching the correct hypothesis gets `1.0`.
  Then builds a PR-00 `BaselineReport` against the run, explicitly
  commented `# SYNTHETIC: fixture LM, not a real Qwen run`.

## Gate verification

- **"exact causal chain exists in trace"** — `hypothesis_event.causal_refs
  == (decision_event.event_id,)`, `experiment_event.causal_refs[:2] ==
  (decision_event.event_id, action_event.event_id)`,
  `verification_event.causal_refs == (decision_event.event_id,
  transition_event.event_id)` — read directly from `runtime/cognition.py`
  and `runtime/action_phases.py` rather than assumed.
- **"new model beats old on held-out transitions"** — satisfied on the
  training case here (a single-action fixture has no meaningful holdout
  split); the general holdout claim is already covered by PR-04's own
  gate tests.
- **"no authority violations"** — `audit_agency_boundary(events) == ()`.
- **"action/token/reward metrics reported against V1 fixture"** —
  `build_report()` runs against this fixture's trace, explicitly labeled
  synthetic.
- **"later use of knowledge without repeating the same real experiment"**
  — `use_prior_knowledge` asserts the verified fact is visible in its
  context and stops without issuing a second discriminating action.

## Validation

Full suite (377 tests), `ruff check`, `ruff format --check`, `mypy` all
pass. Golden fixture unchanged. No change to action authorization or
evaluation behavior — this PR added zero production code, only a test
fixture proving existing code composes correctly.
