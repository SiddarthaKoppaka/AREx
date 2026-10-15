# PR-08 — Queryable compact long-horizon memory

> **Status:** Implemented at the scope actually justified — mostly a
> regression suite against already-existing infrastructure, plus one
> explicitly deferred piece.

## What already existed

`research/index.py:TraceIndex` is already exactly the infrastructure
handoff.md asks for: a rebuildable SQLite + FTS5 full-text index over the
immutable trace, atomically swapped, zero hosted-embedding or vector
dependency — satisfying "do not introduce hosted embeddings or large
default vector dependencies" by construction, not by restraint. It had a
basic rebuild test (`test_research.py`) but nothing exercising it at the
scale or with the specific recall claim handoff.md's gate names.

"Index experiments... by exact causal keys" is likewise already true:
every event type, `EXPERIMENT` included, is indexed with its
`causal_refs`-bearing payload; `retrieve_events`'s `causal_ref`/
`event_types` filters already serve exact-key lookup (confirmed again in
PR-05's `test_experiment_ledger_retrieval.py`).

## What's new: the actual gate, measured at scale

`tests/test_long_horizon_retrieval.py`:

- **"recover an old negative experiment after 50+ unrelated turns"** —
  built a 55-turn trace, negative `EXPERIMENT` event at turn 0, and
  recovered it through *both* paths: the live `retrieve_events` tool
  (`test_retrieve_events_recovers_an_old_negative_experiment_after_many_turns`)
  and the offline `TraceIndex` FTS search
  (`test_trace_index_also_recovers_the_same_old_negative_experiment`) —
  proving the claim at both the per-episode and cross-episode-analysis
  layers, not just one.
- **"no cross-game leakage"** —
  `test_a_second_games_trace_never_leaks_into_the_first_games_results`:
  two separate traces (`game-a`/`game-b`), each with its own negative
  experiment; querying game A's run never surfaces game B's event. True
  by construction (`JsonlEventStore` is one file per run directory, no
  shared state), confirmed rather than assumed.
- **"retrieval metrics measured rather than assumed"** — every assertion
  above is an exact count or exact `event_id` match, not a "probably
  works" — `index.rebuild()`'s own return value (`UNRELATED_TURNS + 2`)
  is checked directly.

## Explicitly deferred, not forgotten

"Index... code versions" for `ExecutableModelCandidate` (PR-03/04) isn't
addressed here: those candidates aren't wired into the live decision loop
yet (no `EventType` exists for proposing one — `ExecutableModelRegistry`
is in-memory only), so there's nothing trace-persisted to index. Same
logic as PR-03/04's own "not wired into the live loop yet" notes — adding
a new event type for a capability with no live caller would be
speculative. Revisit once/if a PR wires candidate proposal into
`CognitiveDecision`. "Skills" (PR-09) don't exist as a concept yet
either — nothing to index until PR-09 defines them.

## Validation

Full suite (383 tests), `ruff check`, `ruff format --check`, `mypy` all
pass. Golden fixture unchanged. No change to action authorization or
evaluation behavior — this PR added zero production code, only tests
against already-existing infrastructure.
