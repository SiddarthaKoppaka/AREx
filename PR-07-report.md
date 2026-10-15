# PR-07 — Bounded independent simulation/search batches

> **Status:** Implemented. Scope: a typed batch summary plus a
> regression suite — most of the gate already held by construction.

## What already existed

Traced through before writing anything:

- `runtime/cognition.py`'s `for tool_request in decision.tool_requests:`
  loop is already sequential, and `runtime/tools.py:run_tool` already
  wraps each call's execution in its own `try/except` — one failing tool
  call already can't abort the rest; the loop just continues.
- `adapters/tool_contract_check.py:check_tool_requests` already rejects
  duplicate `request_id`s *before* a decision is even accepted (part of
  the repair loop, same as the belief/scratchpad checks) — malformed
  batches never reach execution at all.
- Every tool call in one decision already receives the exact same
  `observation` argument — the "immutable snapshot inputs" requirement
  is already satisfied structurally, not something to add.
- `BudgetLedger` already aggregates resource spend across the whole
  episode (including every tool call in a batch), so "aggregate resource
  budgets" needed no new mechanism.

## Decision: no concurrent execution

handoff.md asks for "concurrent execution of independent pure tools when
supported... and serial fallback." Checked what the 8 potentially
cacheable/pure tools actually are: all of them — `retrieve_*`,
`inspect_frame_region`, `check_prediction_history`,
`simulate_world_model`, `search_world_model`, `evaluate_exploration` —
are in-memory, CPU-bound, pure-Python computations with no I/O. Python's
GIL means threading them wouldn't parallelize real work, and
multiprocessing's serialization overhead isn't justified for these
typically-small payloads with no measured bottleneck. Implementing
concurrency here would be exactly the kind of "unmeasured complexity"
ADR-V2-009 says to reject. Serial execution is the only mode; that mode
is already correct (see above), so "equal outputs across serial/parallel
modes" is trivially satisfied by there being one mode.

## What's new

`contracts/tool_batch.py:ToolBatchSummary` + `evidence/tool_batch.py:
summarize_tool_batch()` — a typed, queryable summary (requested/
succeeded/failed/cache_hits/request order) of one decision's tool batch,
read straight from the trace. This is PR-07's "batch contract" — genuine,
dogfooded by the new tests rather than unused.

## Gate verification

- **"no missing/duplicate event refs"** —
  `test_no_missing_or_duplicate_event_refs`: every `TOOL_REQUEST` has
  exactly one `TOOL_RESULT`, no source event is reused.
- **"bounded partial failures"** —
  `test_partial_failure_is_isolated_and_order_is_deterministic`: a batch
  of 3 (`ok-1`, a call against a well-formed but nonexistent artifact
  ref, `ok-2`) still runs all 3, in request order, with exactly 1 failure
  isolated from the 2 successes.
- **"deterministic ordering... equal outputs across serial/parallel
  modes"** — `test_the_same_batch_produces_identical_results_across_runs`:
  the same batch run twice in separate episodes produces an identical
  `ToolBatchSummary`.
- **"no race to live environment"** — unaffected by construction: no
  tool in `CACHEABLE_TOOLS` or this batch touches the environment;
  actions stay on their existing, separate, always-sequential path.

## Validation

Full suite (380 tests), `ruff check`, `ruff format --check`, `mypy` all
pass. Golden fixture unchanged. No change to action authorization or
evaluation behavior.
