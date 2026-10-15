# PR-10 — External harness self-improvement loop

> **Status:** Implemented at the scope the gate actually calls for:
> measurement/bookkeeping scaffolding for a human-driven workflow, not
> an autonomous patch-generator.

## Scope decision

handoff.md's own Implement list says "optionally use a separate engineer
LM for candidate patches only" and its Gate explicitly requires "no
automatic production promotion" plus "human review." Loop B's own
description is a human/process workflow: "propose a small source/
config/prompt change in a separate branch → run static checks → replay →
paired live evaluation → held-out promotion/rejection → archive/
rollback." Given that, this PR builds the measurement and bookkeeping
the workflow needs — failure packets, candidate lineage, static checks,
a held-out promotion *criterion* — and deliberately does not build an
autonomous LLM patch-generation loop. That's a real engineering decision
with its own blast radius (an agent editing its own prompt/tool source
unsupervised) that shouldn't be introduced as a side effect of working
through a PR list; "optional" in the gate means exactly that it's a
choice to make deliberately, not default to.

## What was built

`src/arc_agi_3/harness_improvement/`:
- `contracts.py` — `FailurePacket` (derived, not invented), `
  StaticCheckResult`, `HarnessChangeCandidate` (versioned,
  `parent_version` lineage like every other registry in this codebase),
  `PromotionDecision`.
- `failure_packet.py:build_failure_packet()` — reads an existing PR-00
  `BaselineReport` and names its most salient already-measured failure
  signal (context-ceiling, unjustified repeats, verification failures,
  in that priority order) — never a new or inferred signal.
- `static_checks.py:run_static_checks()` — runs this project's own
  `ruff check`/`ruff format --check`/`mypy` against a candidate's working
  tree via subprocess. Ordinary dev tooling on code a human is about to
  review — not PR-03's untrusted-code sandbox; no special isolation
  needed here.
- `registry.py:HarnessChangeRegistry` — version lineage, mirroring
  `world_model/store.py`; records a decision but never makes one.
- `promotion.py:decide_promotion()` — a held-out criterion over PR-00's
  existing `compare_reports()` output: did the named metric improve or
  stay flat, on an already-produced control-vs-candidate comparison.
  This is a *recommendation* a human reads, not a merge action.

## Gate verification

- **"one candidate can be demonstrably accepted and one rejected"** —
  `test_a_regressing_candidate_is_rejected_on_verification_failures` /
  `test_an_improving_candidate_is_accepted_on_model_calls`: three real
  (tiny, fast) runner executions reduced to `BaselineReport`s via PR-00's
  existing `build_report`, compared via PR-00's existing
  `compare_reports` — nothing new invented for the comparison itself.
- **"full reproducible evaluation"** — the comparison runs deterministically
  through already-tested PR-00 machinery.
- **"parent lineage"** —
  `test_registry_enforces_parent_lineage_and_records_the_decision`: a v2
  candidate must declare `parent_version=1`, exactly like
  `ExecutableModelRegistry`/`SkillLibrary`.
- **"no authorization/trace regression"** —
  `test_no_authorization_or_trace_regression_from_building_a_report`:
  `audit_agency_boundary(events) == ()` on the run used to build a
  report.

## Validation

Full suite (400 tests), `ruff check`, `ruff format --check`, `mypy` all
pass. Golden fixture unchanged. No change to action authorization or
evaluation behavior.
