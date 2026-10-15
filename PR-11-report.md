# PR-11 — Comparative evaluation protocol (cross-cutting)

> **Status:** Implemented at the scope a *protocol* PR actually has:
> closing one concrete gap in the existing comparison machinery, and
> documenting the rest against what's already measured.

## What this PR is

handoff.md itself calls PR-11 "cross-cutting" and says it "spans all
stages" — it's a discipline every comparison should already follow, not
a standalone feature. So the job here was narrower than most PRs: find
what the *existing* comparison machinery (PR-00's `compare_reports`,
PR-10's `decide_promotion`) was silently assuming, fix that, and
document the rest against metrics that already exist rather than
inventing new ones to check a box.

## The one real gap: comparability was never checked

`compare_reports(control, candidate)` would happily diff two
`BaselineReport`s with different `game_id`s, different seeds, or
different model metadata, with nothing in the output saying so — exactly
the "always evaluate against matched game versions, seed, model
weights..." requirement being silently assumed rather than verified.

**Fixed**: `research/report_comparability.py:check_comparable()` checks
`game_id`/`seed`/`model`/`environment`/`ablations` and returns a tuple of
specific mismatch warnings (never blocks — a deliberate cross-game or
cross-seed comparison is sometimes exactly what's being studied, so this
makes the mismatch *visible*, not forbidden). Wired into
`compare_reports()`'s own output (`ReportComparison.comparability_warnings`)
and surfaced in `render_markdown()`.

**Also fixed**: `ReportComparison.sample_size_note` (default `"n=1 run
per side; no uncertainty estimate"`) is now always rendered — handoff.md's
"report sample size/uncertainty" requirement, made honest rather than
implicit. `research/aggregate.py`'s existing seeded bootstrap CI
(`aggregate_rows`) is the already-built path to a real multi-sample
comparison when more than one seed/run is available per side; this PR
doesn't duplicate it, just makes the single-run case stop looking like
more evidence than it is.

## The rest, checked against what already exists

| Protocol requirement | Already covered by |
|---|---|
| completion | `metrics.succeeded`/`levels_completed` |
| partial reward | `metrics.rhae_level_scores`/`rhae_environment_score` |
| action efficiency | `metrics.environment_actions`/`model_calls` |
| inference/total cost | `metrics.input_tokens`/`output_tokens`/`wall_time_ms` |
| novelty | `metrics.consecutive_no_novelty_turns`/`unique_tool_requests` |
| world-model holdout accuracy | PR-04's `ModelTestReport.holdout_accuracy` |
| repeats | `metrics.repeated_tool_requests`/`repeated_experiments` |
| failure counts / reliability | `metrics.failures`/`structured_output_failures`/`errors` |
| "useful retrieval" | approximated by existing tool-use counters; no dedicated metric exists — noted, not invented, per PR-08's own same decision |
| "reserve held-out games... do not optimize repeatedly" | a process discipline, not code — recorded here as the convention to follow, not something to enforce in this harness |
| ablation ladder (V1 → tool cues → executable models → ...) | `AblationConfig` already has `world_models`, `executable_models` (PR-03); `tool_cues`/skills/batches have no ablation flag because PR-07/09 concluded there's no separate mechanism to toggle yet (batches: no concurrency added; skills: not wired into the live loop) |

## Validation

Full suite (403 tests), `ruff check`, `ruff format --check`, `mypy` all
pass. Golden fixture unchanged. No change to action authorization or
evaluation behavior.
