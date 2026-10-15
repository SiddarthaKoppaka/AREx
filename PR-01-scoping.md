# PR-01 Scoping Document — Tool Affordances & Repeat Visibility

> **Status:** Scoping only. No implementation yet.
> **Depends on:** PR-00 (`ec5b0ba`, merged to `main`).
> **Scope lock:** read-side prompt content + one evidence-ordering fix. No
> agent-policy change, no action-authorization change, no new world-model
> execution framework, no automated behavior.

## Context

Before touching the prompt, I audited what the model is actually shown
today and what the harness already computes but never surfaces live. This
document is that audit plus a narrow, reviewable proposal — nothing here
has been implemented.

## Explicit exclusions (restated from the brief)

- No automated strategy switching.
- No forced tool usage or experimentation.
- No new world-model execution framework.
- No generalized stagnation detection. `consecutive_no_novelty_turns`
  (`evaluation_tool_repeats.py:39-48`) is treated here only as a
  **research input** already computed for the PR-00 report — its
  semantics (what a "novel" tool call means, whether it's a good stagnation
  proxy) are explicitly **not** investigated or wired live in this PR.

## 1. Audit of all 12 registered tools

Source of truth: `tool_registry.py:TOOL_ARGUMENTS` (12 entries),
cross-referenced against `runtime/tool_handlers.py`'s dispatch and each
tool's own handler module for real prerequisites/outputs — not inferred
from naming.

| Tool | Purpose | Prerequisites | Output | In `TOOLS` prose today? |
|---|---|---|---|---|
| `archive_artifact` | Persist a durable, purpose-tagged artifact keyed by evidence refs, for later cheap retrieval. | `evidence_refs` must be non-empty existing refs. | Artifact ref. | Yes |
| `retrieve_artifact` | Read back a previously archived artifact within a token budget. | Artifact must exist (`archive_artifact` ran first). | Artifact content, possibly key-filtered. | Yes |
| `retrieve_evidence` | Bounded, LM-chosen view (`metadata`/`summary`/`transition_delta`/`rle_frame`/`region`/`full`) over up to 8 named events. | Target `event_ids` must exist. | View-shaped payload within `token_budget` (byte bound, not token count — already documented via `Field(description=...)` on `EvidenceQuery.token_budget`). | Yes |
| `inspect_frame_region` | Dedicated crop of one event's frame by explicit `min_row/max_row/min_col/max_col`. | Target event must have frame data. | Cropped region payload. | Yes |
| `check_prediction_history` | Retrospective compatibility check of a prediction against past transitions (compatible/contradicting/not_evaluable). | None beyond a well-formed `ExpectedOutcome`. | `RetrodictionResult` counts, never asserts truth. | Yes |
| `retrieve_events` | Exact, filtered event lookup (by type/id/causal ref/component/text/sequence range). | None. | Up to `limit` (≤200) `ContextEvent`s. | Yes |
| **`simulate_world_model`** | Run one or more actions through an already-authored declarative `WorldModel` and get predicted resulting states, without touching the live environment. | **A `WorldModel` with this exact `model_id`/`model_version` must already exist** — authored by the model itself via `world_model_updates` on a prior `CognitiveDecision` (`contracts/decision_core.py:40`), *not* via any tool. Calling this before authoring a model fails lookup. Gated by `ablations.world_models`. | `SimulationStep` sequence (`applied`/`invalid`/`ambiguous` per step) + `final_state`. Spends `Resource.SIMULATIONS`. | **No** |
| **`search_world_model`** | BFS/A* search over the same declarative model toward `goal_facts`, instead of the model manually chaining single simulations. | Same world-model prerequisite as above, plus `ablations.search`. | `SearchResult`: status (`found`/`partial`/`exhausted`), action sequence, cost, node expansions. Spends `Resource.SEARCH_NODES`. | **No** |
| **`evaluate_exploration`** | Pure arithmetic scoring of candidate "probes" (candidate next moves/states) by a weighted benefit-minus-cost formula, to decide whether further exploration is worth it versus taking the already-known best plan. | None — the model must supply its own per-probe estimates (see `drig` below). No ablation gate. | Per-probe `value`, the maximizing probe(s), `exploration_dominated` (is the known plan already at least as good as exploring?), `saturated` (have recent marginal gains been at/under threshold?). | **No** |
| **`record_exploration_outcome`** | Append one realized probe outcome (`state_hash`, `marginal_value`) to the harness's running exploration history, so future `evaluate_exploration` calls see real visit counts/recent marginal values instead of the model re-asserting them every time. | A prior `evaluate_exploration` call is the normal source of a probe worth recording, but nothing enforces that order. | A snapshot of the full exploration history (`visits`, `marginal_values`). | **No** |
| `consult_specialist` | Ask one ExoCortex faculty (hypothesis/dynamics/critic) for an independent view of current evidence. | A specialist backend must be configured for the run (`specialist_backend is not None`); otherwise this tool is unavailable. | A `SpecialistReport`. | Yes |
| `compare_specialist_reports` | Ask one faculty to critique other faculties' prior reports this episode. | Same specialist-backend prerequisite; at least one prior report to compare against. | A `SpecialistReview` with critiques. | Yes |

**The exact gap, restated precisely:** 4 of 12 tools — all four read-side
*and* their argument contracts carry zero `description=` metadata anywhere
in `world_model/contracts.py`, `planning/contracts.py`,
`exploration/contracts.py` — get no prose explanation of purpose or
prerequisites, and no per-field schema hints. The model sees their bare
JSON shape only. The most severe single case:

### `drig`, investigated from the implementation

`exploration/contracts.py:17`: `ProbeEstimate.drig: float`, never explained
in any docstring, comment, or commit message (checked full history via
`git log --all -- exploration/contracts.py exploration/value.py`: both
files were added whole, already containing `drig`, in the initial harness
commit). Its meaning is fully recoverable from its one call site,
`exploration/value.py:15-20`:

```python
benefit = (
    request.weights.information * probe.drig
    + request.weights.reward_progress * probe.expected_reward_progress
    + request.weights.hypothesis_discrimination * probe.hypothesis_discrimination
)
```

`drig` is structurally parallel to `expected_reward_progress` and
`hypothesis_discrimination` — each is a per-probe *estimate* the model
must supply, weighted by the matching `ExplorationWeights` field, and
summed into `benefit` before costs are subtracted. By construction,
**`drig` is the probe's estimated information gain** — the quantity
`weights.information` is meant to price. The literal expansion of the
acronym is not recoverable from the code (likely house terminology from
whatever "Miro exploration-value formulation" the module's docstring
names); the **operational** meaning — what number the model should put
there and why — is fully determined and is what gets documented.

## 2. Proposed additions (text only, not yet applied)

**`prompt_text.py`'s `TOOLS` block** gets one new paragraph (after the
existing tool-name sentence, before `CURRENT_OBSERVATION omits...`):

> World-model tools operate on a model *you* author via
> `world_model_updates`, not on the live environment: `simulate_world_model`
> previews one or more actions against your existing declarative model
> without spending a real action; `search_world_model` finds an action
> sequence toward named goal facts over that same model instead of chaining
> single simulations by hand. Both require a `world_model_updates` entry
> with the matching `model_id`/`model_version` to already exist from an
> earlier decision. `evaluate_exploration` scores candidate next moves you
> estimate yourself (including `drig`, your estimated information gain from
> that probe) against the plan you already have, and reports whether
> further exploration is worth it or already dominated; call
> `record_exploration_outcome` afterward so later `evaluate_exploration`
> calls see real visit history instead of you re-asserting it.

**Field-level `description=`** (mirrors the existing pattern on
`EvidenceQuery.token_budget`):
- `ProbeEstimate.drig`: *"Estimated information gain from this probe, in the same units `weights.information` is meant to price — not a measured value."*
- `ExplorationRequest.saturation_window` / `saturation_threshold`: *"How many of the most recent recorded marginal values must be at or below this threshold for `saturated` to report exploration has stopped paying off."*
- `SimulationRequest.model_id` / `model_version`: *"Must match an existing `world_model_updates` entry from an earlier decision; this tool does not create a model."*
- `SearchRequest.model_id` / `model_version`: same note; `goal_facts`: *"Facts the search treats as satisfying the objective — a subset of a reachable `SymbolicState.facts`."*

Both additions are additive and reviewed together: prose explains the
*when/why*, field descriptions explain the *what* at the exact point the
model reads the schema. Neither duplicates `EPISTEMICS`'s already-adequate
experiment/hypothesis guidance (see §4).

## 3. `tool_request_evidence()` ordering fix

**Current behavior** (`evidence/tool_cues.py:46-47`): fingerprints are
grouped into a plain dict in first-seen order, then
`list(groups.items())[:limit]` keeps the **first** `limit` (default 8)
distinct fingerprints ever made. Past 8 distinct tool calls in an episode,
the live `TOOL_EVIDENCE` section can only show ever-older entries — a
genuinely new repeat pattern from turn 40 is invisible if 8 different
fingerprints were already used earlier. Confirmed this is the exact call
path the live prompt uses (`runtime/working_set.py:39`, default `limit`).

**Proposed fix** — order by recency of each fingerprint's *last* event,
not by first sight, while leaving every other semantic untouched:

```python
def tool_request_evidence(history, current_observation_event_id, *, limit=8):
    # ... unchanged grouping loop ...
    ordered = sorted(groups, key=lambda key: groups[key][-1].sequence, reverse=True)
    table = []
    for key in ordered[:limit]:
        events = groups[key]  # full history for this fingerprint
        request = requests[key]
        from_current = sum(
            observation_at.get(e.event_id) == current_observation_event_id
            for e in events
        )
        table.append(
            ToolRequestEvidence(
                tool_name=request.tool_name,
                arguments=request.arguments,
                fingerprint=key,
                attempts=len(events),  # full count, unchanged
                attempts_from_current_state=from_current,  # full count, unchanged
                last_request_event_id=events[-1].event_id,
            )
        )
    return tuple(table)
```

`attempts`/`attempts_from_current_state` still count from the *entire*
matching history for that fingerprint (never truncated) — only **which**
`limit` fingerprints get a row changes, from "earliest created" to "most
recently touched." Row count, schema, and event-provenance fields
(`last_request_event_id`) are all unchanged, so the context-budget
contract and "mechanical, not a decision" character of this table are
preserved exactly. `EventEnvelope.sequence` (already a required field,
`contracts/events.py:17`) is the recency key — no new state, no new event
type.

## 4. Is action-repeat evidence already sufficient? (yes — no change)

Checked `evidence/action_cues.py` and `evidence/repeats.py` for the same
staleness risk:

- `action_cues.py:33-43` seeds its table from the **current observation's**
  `available_actions` first — a small, closed, per-turn-bounded set (ARC
  actions, not open-ended tool+argument combinations) — so insertion order
  never crowds out anything the way an unbounded tool-fingerprint space can.
- `recent_outcomes=tried[-outcome_limit:]` (`action_cues.py:63`) **already**
  slices from the end — most recent first — not the start.
- `evidence/repeats.py:detect_repeat()` iterates `reversed(records)`
  (`repeats.py:28`) and stops at `limit` matches — also recency-correct by
  construction.

**Conclusion: no change to `action_cues.py` or `repeats.py`.** They don't
share `tool_cues.py`'s bug because the action space is small and bounded
while the tool-fingerprint space is not, and because their code already
reads from the recent end where it matters. This directly satisfies "avoid
duplicating existing functionality" — the fix is scoped to exactly the one
function that has the problem.

## 5. Test plan (deterministic, no live model calls)

All in the existing `tests/` style (`JsonlEventStore` + hand-appended
events, or `ScriptedModel`/`ScriptedBackend` fixtures already used
throughout this codebase):

1. **Tool documentation coverage** — a test asserting every entry in
   `TOOL_ARGUMENTS` either has a non-empty prose mention in `TOOLS` *or*
   every one of its fields carries a `description` (fails loudly if a 13th
   tool is ever added and forgotten, rather than silently repeating today's
   gap).
2. **`drig`/field-description presence** — `tool_schemas()["evaluate_exploration"]["properties"]["probes"]...["drig"]["description"]` is non-empty (and similarly for the other newly-described fields).
3. **Evidence freshness** — hand-build >`limit` distinct tool-call
   fingerprints across many turns, then make an old fingerprint recur;
   assert the recurring old fingerprint's row is present (previously it
   would have been evicted by `[:limit]` on first-seen order) and that the
   least-recently-touched fingerprint beyond `limit` is the one dropped.
4. **Unchanged-state repeat** — identical fingerprint called twice from the
   same observation; `attempts == 2`, `attempts_from_current_state == 2`,
   exactly one row (regression guard matching the existing
   `test_tool_request_evidence_counts_attempts_from_current_state`).
5. **Changed-state request** — same fingerprint called before and after a
   new `OBSERVATION` event; `attempts_from_current_state` reflects only
   the post-observation calls (existing test already covers this shape;
   extended to assert ordering is unaffected by an intervening unrelated
   fingerprint).
6. **Prompt-size impact** — byte length of `tool_schemas()`'s rendered
   JSON and of `TOOLS` before/after the prose addition; assert the delta is
   small and bounded (a concrete number, not just "didn't explode") so a
   future change can't silently bloat every turn's prompt.
7. **Backward compatibility** — re-run the full existing
   `tests/test_tool_repeat_evidence.py` suite unmodified against the new
   ordering implementation; all current assertions (single-fingerprint
   cases, which are order-invariant) must keep passing with no edits.

## 6. Anticipated file changes (not yet made)

| File | Change |
|---|---|
| `adapters/prompt_text.py` | Add the one `TOOLS` paragraph above. |
| `exploration/contracts.py` | Add `description=` to `drig`, `saturation_window`, `saturation_threshold`. |
| `world_model/contracts.py` | Add `description=` to `SimulationRequest.model_id`/`model_version`. |
| `planning/contracts.py` | Add `description=` to `SearchRequest.model_id`/`model_version`/`goal_facts`. |
| `evidence/tool_cues.py` | The ordering fix in §3 (≈3 changed lines). |
| `tests/test_tool_repeat_evidence.py` | Add cases 3 and 5 (extended); no edits to existing assertions. |
| New: `tests/test_tool_documentation_coverage.py` | Cases 1, 2, 6. |

All files stay within the existing 100-line cap; `tool_cues.py` is 63
lines today and the fix adds roughly 3.

## 7. Acceptance criteria

- Every one of the 12 tools is named with purpose+prerequisite in `TOOLS`
  prose, or has field-level descriptions covering its non-obvious inputs.
- `tool_request_evidence()` surfaces the most recently touched fingerprints
  up to `limit`, never silently drops a fresh repeat in favor of a stale
  one, and every existing test passes unmodified.
- `drig`'s documented meaning is confirmed against this document before
  being written into the schema (operational semantics only — the author
  should correct the acronym's literal expansion if known).
- No change to `action_cues.py`, `repeats.py`, action authorization,
  `EvaluationMetrics`, or any decision/runtime behavior.
- Full suite + `ruff` + `mypy` green; prompt byte-size delta reported, not
  just checked non-regressive.

## 8. Risks

- **Prompt growth**: every added sentence repeats every turn. Mitigated by
  keeping the addition to one paragraph and measuring the exact delta
  (test 6) rather than assuming it's negligible.
- **False precision on `drig`**: the operational meaning is sound from the
  arithmetic, but the *name* itself is undocumented house terminology — the
  description text deliberately describes behavior ("estimated information
  gain... not a measured value"), not a claimed acronym expansion, to avoid
  asserting something unverified.
- **Ordering fix changes which fingerprints are visible**: by design — this
  is the fix's whole point — but any future code that assumed first-seen
  order (none found in this audit) would need updating. Test 7 exists
  specifically to catch that.

## After approval

Implement exactly this document — no agent-policy change, no change to
action authorization, no change to `EvaluationMetrics`/evaluation behavior,
no change to deterministic execution guarantees. Full validation
(`pytest`, `ruff check`, `ruff format --check`, `mypy`) before reporting
back, same as PR-00.
