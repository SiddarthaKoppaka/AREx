# PR-02 Scoping Document — Context & Memory Reliability

> **Status:** Scoping only. No implementation yet.
> **Depends on:** PR-00 (`ec5b0ba`), PR-01 (`a51644c`), both merged to `main`.
> **Scope lock:** investigation + proposal only. No vector databases, no
> embeddings, no additional memory agents. The harness manages evidence and
> memory; the LM owns interpretation and strategy — nothing proposed here
> changes who decides what.

## 1. Source-backed map of the context/memory lifecycle

**Reconstruction pipeline, every turn, from scratch:** `runtime/context.py:project_context()`
reads the *entire* trace (`events.read()` — unbounded, O(episode length),
re-verifies the full hash chain every call), splits it into `recent`
(bounded window) and `older` (everything else), then builds `AgentContext`
by combining bounded views with several workspace-derived sections.
`compact_for_backend()` (`adapters/context_compaction.py`) then prunes
*only two* of those fields if the rendered prompt exceeds budget. The final
prompt is assembled by `adapters/prompts.py:decision_prompt()` as fixed
prose (`CONTRACT`+`EPISTEMICS`+`TOOLS`+`TOOL_CONTRACTS`) + the sections
below.

| Section (prompt key) | Source | Always included? | Bounded by | Touched by compaction? |
|---|---|---|---|---|
| `CURRENT_OBSERVATION` | live `Observation` | Yes | `compact_observation` flag strips raw frame | No (not variable-length in the relevant sense) |
| `RECENT_TURNS` | `recent_turn_events()` over `visible` history | Yes | `raw_recent_turns`/`recent_event_limit` at build time; re-sliced to last 2 steps / 12 events under pressure | **Yes** — the primary lever |
| `RELEVANT_EPISODES` | `context/episodic.py:summarize_episode()` | Only if `persistent_memory` ablation on | `episodic_retrieval_limit` (default 6-8) items, word-overlap+recency scored from *all* older salient events | **Yes** — items trimmed further under pressure |
| `HYPOTHESIS_LEDGER` | `evidence/ledger.py` via `hypothesis_ledger()` | If `hypotheses` ablation on | Has its own display limit + `omitted_hypothesis_ids` | No |
| `ACTION_EVIDENCE` | `evidence/action_cues.py` | Yes | `limit=8` (seeded from the *current* legal action set, so naturally small); `recent_outcomes` already slices the last 3 | No (already small) |
| `TOOL_EVIDENCE` | `evidence/tool_cues.py` | Yes | `limit=8` distinct fingerprints — **PR-01 fixed this to be recency-ordered**, no longer first-seen | No |
| `TASK_GRAPH` | `cognition/tasks.py:TaskGraph.views()` | If `tasks` ablation on | **None** — one view per distinct `task_id` ever created, including completed/abandoned | **No** |
| `MODEL_AUTHORED_WORLD_MODELS` | `world_model/store.py:WorldModelStore.current` | If `world_models` ablation on | Latest version per distinct `model_id` (bounded by distinct-model count, not version count) | **No** |
| `COGNITIVE_SYNTHESIS` | `evidence/cognitive_synthesis.py:build_cognitive_synthesis()` | Whenever any specialist report exists | **None at all** — every report, every review, every hypothesis inside every report, every critique inside every review, for the whole episode | **No** |
| `WORKING_SCRATCHPAD` | `cognition/scratchpad.py` | Yes | Hard `scratchpad_token_budget` (default 2048) enforced **at write time** | No (nothing to compact; it's already small by construction) |
| `UNRESOLVED_CONTRADICTIONS` | derived from hypothesis ledger | If hypotheses on | Same bound as ledger | No |
| `OTHER_CONTEXT` (budget/stats/recovery) | misc | Yes | Fixed shape | No |
| `TOOL_CONTRACTS` | `tool_schemas()` over all 12 tools | Yes, every turn | Fixed (grows only when tools/fields are added) | No |

## 2. Confirmed defects vs. architectural limitations

### Confirmed defect: a scratchpad update can crash the whole episode instead of being repaired

Traced the exact call path. `adapters/generation_loop.py`'s `validate_extra`
runs `check_tool_requests` + `check_decision_context` — the latter exists
*specifically* because (per its own docstring) a raw hypothesis-contract
`ValueError` used to surface only inside `apply_cognitive_decision`,
"after the decision was already accepted, which killed the whole episode
instead of being repaired the same cheap way as a malformed tool call."
That fix was applied **only for hypothesis mutations**. The exact same
late-crash shape still exists for scratchpad updates:
`cognition.py:apply_cognitive_decision()` calls
`workspace.scratchpad.prepare(decision.scratchpad_updates, observed)`
directly (`cognition.py:32-34`), with no pre-check equivalent to
`check_decision_context`. `ScratchpadStore.prepare()`
(`cognition/scratchpad.py:55-58`) raises bare `ValueError` if the budget
is exceeded; `scratchpad_merge.py:merge_facts()` raises bare `ValueError`
for an invalid `evidence_refs` set, a fact superseding itself, duplicate
`supersedes`, a dropped-evidence consolidation, or a wrong `version`. None
of these are caught anywhere between `apply_cognitive_decision` and
`runtime/runner.py:31-38`'s top-level `except Exception` — confirmed by
reading `session_decide.py` and `session.py`, there is no intermediate
handler. **Concretely: a model that cites a stale `evidence_ref`, or tries
to add one fact too many without consolidating via `supersedes`, ends the
episode with a `FAILURE` event instead of getting a cheap repair prompt.**
This directly answers "can verified facts be lost" with something worse
than loss — a crash.

### Confirmed, narrower defect: old world-model versions become unreachable after a fork/recovery

`WorldModelStore.snapshot()` serializes only `.current` (latest version
per `model_id`); `restore()` rebuilds `_history` as `{model_id: [item]}` —
**one entry each**, discarding every older version
(`world_model/store.py:39-46`). After any `RECOVER`→`FORK_CHECKPOINT`, the
agent's `simulate_world_model`/`search_world_model` calls against a
pre-fork, non-latest `model_version` will `KeyError` (`store.py:30-34`),
even though that version's content is still sitting in the immutable
`events.jsonl` as a `WORLD_MODEL` event — nothing currently reconstructs
`_history` from the raw trace, only from the checkpoint's snapshot. This
is real but narrow: it only bites after a recovery fork, and only for
non-current model versions.

### Architectural limitation (not a bug): episodic recall is not guaranteed, by design

`context/episodic.py:summarize_episode()` scores older steps by literal
word-overlap with a query built from the current objective/open
tasks/hypothesis claims (`runtime/context.py:43-49`), takes the top
`episodic_retrieval_limit`. If an important older step shares no
vocabulary with the current query, it will not surface automatically —
matching handoff.md's own flagged risk. Critically, **nothing is actually
destroyed**: the full event is still in the immutable trace and
`retrieve_events`/`retrieve_evidence` can fetch it by ID or `causal_ref`
on demand. The limitation is recall being automatic-and-approximate
rather than guaranteed, not data loss. Confirmed experiments specifically
*are* reliably retrievable: `runtime/experiments.py:record_experiment()`
stamps every `EXPERIMENT` event with `causal_refs=(decision.event_id,
acted.event_id)` (`experiments.py:58-64`), and `RetrievalQuery` supports
filtering by `causal_ref` directly.

### Architectural limitation: two sections grow with episode length, unbounded

`COGNITIVE_SYNTHESIS` (every specialist report/review ever, confirmed
above) and `TASK_GRAPH` (every distinct task ever created, including
finished/abandoned ones, confirmed above) are the only two context
sections with **no size limit of any kind** and **no compaction touch
point**. For a short ARC episode (bounded by `max_turns`/`max_model_calls`)
this is unlikely to matter today — but it is the same shape of problem
PR-01 already hit once for fixed prose, and `hypothesis_versions`
(`working_set.py:60-62`, explicitly commented "Unbounded, unlike
`hypotheses` above") is a third growth path, though it's a validation
dict rather than something rendered into the prompt directly.

### Confirmed clean: checkpoint/restore preserves the full cognitive workspace

`CognitiveWorkspace.snapshot()`/`.restore()` (`cognition/workspace.py:28-51`)
cover beliefs, tasks, world models, exploration history, *and* the
scratchpad together; wired end-to-end through
`runtime/io.py:87-95` (`save_checkpoint(..., self.workspace.snapshot())`)
and `runtime/recovery_fork.py:32` (`workspace.restore(checkpoint.cognitive_state)`).
**The scratchpad does survive checkpoint/restore.** The one caveat is the
world-model version-history loss noted above — restore brings back every
subsystem's *current* state correctly, just not non-current world-model
versions.

## 3. The PR-01 test-fixture investigation (17,300 → 19,200)

Quantified precisely, not estimated: my PR-01 changes added 683 characters
to `TOOLS` prose and 689 bytes to the rendered `TOOL_CONTRACTS` JSON (both
measured directly, before vs. after, via `len()`/`json.dumps()` on the
actual values). `tests/test_memory_edges.py::CharacterCounter` models a
token counter as raw character count, with `max_input_tokens=17300` —
and its own comment already says this ceiling exists specifically "to
allow room for the fixed... prose/schema... none of which is compactable."
Raising it to 19,200 was a direct, traceable consequence of that fixed
content growing by ~1,372 characters — not a hidden behavior change
elsewhere; I confirmed this by running the full suite before and after
and finding exactly one failure, in exactly the test whose own docstring
names the mechanism.

**That said, the investigation surfaced a real process gap.** This is the
**third** time in this project's history a "headroom" constant needed
raising because fixed prompt content grew (`d80c724` and `7923ecc`'s
commit messages both say "raise colab_transformers headroom"; now PR-01).
And the constants in `CharacterCounter` (`soft_input_limit=6000,
max_input_tokens=17300→19200`) are **not derived from or checked against
real production defaults** — `RuntimeFields` (`settings_fields.py:25-30`)
sets `soft_input_limit=12000`, `hard_input_limit=30720` in production;
the test's numbers are ad hoc, chosen only to make that one stress test
pass. My own PR-01 test
(`test_tool_documentation_coverage.py::test_prompt_and_schema_sizes_stay_within_a_bounded_budget`)
only covers `TOOLS`+`TOOL_CONTRACTS` against a loose, disconnected bound
(3,500 chars / 20,000 bytes) that the 1,372-character growth easily
passed — it did not and could not have caught the `CharacterCounter`
regression, because the two tests check unrelated ceilings.

**Conclusion: no, existing tests do not adequately distinguish legitimate
recalibration from a real context-budget regression.** There is currently
no single test that (a) measures the *total* fixed, non-compactable
overhead (`CONTRACT`+`EPISTEMICS`+`TOOLS`+`TOOL_CONTRACTS`) and (b) checks
it against the *real* `soft_input_limit`/`hard_input_limit` defaults, with
a message that says *why* a change grew or shrank it. Today, the signal is
an opaque failure inside an unrelated stress test, diagnosed by hand each
time. Proposed in §4.

## 4. Proposed improvements (not implemented)

**A. A scratchpad pre-check, mirroring `check_decision_context`.** Add a
`ScratchpadContractError` (same `ValueError` subclass pattern as
`BeliefContractError`/`ToolContractError`) and a
`check_scratchpad_update(decision, context)` function that re-derives the
exact same validation `ScratchpadStore.prepare()` already does (budget
estimate, evidence-ref validity, supersede rules, version) *before*
acceptance, wired into `generation_loop.py`'s `validate_extra`. This
converts a currently episode-ending crash into the same cheap repair path
every other contract violation already gets. No change to the scratchpad's
actual rules — only *when* they're enforced.

**B. Bound `COGNITIVE_SYNTHESIS`**, mirroring the `tool_cues.py`/
`action_cues.py` pattern already in this codebase: most-recent-N reports
and reviews (by `turn`/event sequence, not first-seen), with an explicit
omitted count, kept retrievable in full by event ID exactly as unbounded
`retrieve_events` already allows. Needs a decision (for review, not
auto-decided here) on whether "most recent" or "most recent per faculty"
is the right ordering, since a faculty consulted once early could
otherwise be crowded out entirely.

**C. Bound `TASK_GRAPH`** — **correction found during implementation:**
`cognition/tasks.py:TaskGraph.validate_many()` already hard-caps the
graph at 32 distinct tasks total (`raise ValueError("task graph exceeds
32 tasks")`, line 47), including completed/abandoned ones. This section's
original "unbounded... never pruned" framing was imprecise — the total
is already small and fixed, not growing with episode length. No change
made; implementing an additional display bound here would be complexity
without demonstrated need (ADR-V2-009).

**D. A dedicated fixed-prompt-overhead test**, tied to real settings
defaults instead of an ad hoc local constant: build the real
`CONTRACT+EPISTEMICS+TOOLS+TOOL_CONTRACTS` string, assert its length
against an explicit fraction of `RuntimeFields().soft_input_limit` (e.g.
"fixed overhead must stay under 25% of the soft limit"), and give a
failure message naming the exact byte delta if it doesn't. This is a
process fix, not a behavior fix — its job is to make the *next* PR-01-style
growth show up as one clear, dedicated failure instead of a side effect of
an unrelated stress test.

**E. Open question, not a decision:** should old world-model versions be
reconstructed from the raw trace after a restore (replaying `WORLD_MODEL`
events since episode start into `_history`), or is "only the current
version survives a fork" an acceptable, documented limitation? Flagging
for review rather than picking an answer, since reconstructing from trace
on every restore has its own cost/complexity trade-off worth discussing
before committing to it.

## 5. Explicitly not proposed

No vector database, no embeddings, no additional memory/retrieval agent.
Episodic recall stays word-overlap + recency; exact retrieval stays
event-ID/causal-ref based. Nothing here changes who decides what — all
four proposals are visibility/robustness fixes to data the harness already
collects, not new interpretation.

## 6. Deterministic stress-test scenarios (for the eventual PR)

1. **Long history**: hundreds of turns of a scripted model; assert
   `recent_events`/`episodic_memory` stay bounded at every turn and the
   per-turn `events.read()` cost is accounted for (not necessarily fixed,
   but measured, so a future change can see if it got worse).
2. **Many world-model versions**: one `model_id` revised dozens of times;
   assert `MODEL_AUTHORED_WORLD_MODELS` stays at exactly one entry (latest
   version), `simulate_world_model` resolves the current version, and a
   fork/recovery in the middle demonstrates the confirmed version-loss
   limitation (reproducing §2's defect as a named, tracked test rather
   than leaving it undemonstrated).
3. **Many specialist reports/reviews**: dozens of `SPECIALIST_REPORT`/
   `SPECIALIST_REVIEW` events; demonstrate `COGNITIVE_SYNTHESIS` growing
   unbounded today (a documenting test, expected to need updating once §4-B
   ships), then (after B) assert it stays bounded with a correct omitted
   count.
4. **Large retrieved artifacts**: a `retrieve_events` result exceeding
   `MAX_ITEM_CHARACTERS` (4096); assert `retrieved_projection.py`'s
   automatic-context bound holds and the full result is still fetchable
   by event ID (already implemented — this is a regression guard, not a
   new behavior).
5. **Checkpoint/fork content fidelity**: author a scratchpad fact, a
   world model, a task, and a hypothesis; checkpoint; fork; assert *all
   four* survive with correct values. This exercises code already
   confirmed correct in §2 but currently has **no test asserting content
   fidelity** — `test_recovery.py`'s existing fork test only checks
   branch/action mechanics, not workspace content.
6. **Scratchpad contract violation**: a decision with a `scratchpad_updates`
   citing a stale `evidence_ref`; before §4-A, assert it currently ends
   the episode (documenting the confirmed defect); after §4-A, assert it
   is instead repaired the same cheap way as a malformed tool call.
7. **Fixed-overhead budget**: the §4-D test itself — build the real fixed
   prose+schema string, assert against the real `soft_input_limit`
   fraction.

## 7. Files changed (implemented)

| File | Change |
|---|---|
| New: `cognition/scratchpad_prepare.py` | `prepare_scratchpad()`, the shared pure merge+estimate function extracted out of `ScratchpadStore.prepare` so the real commit path and the new early check can never drift apart (A). |
| `cognition/scratchpad.py` | `prepare()` now delegates to the shared function (A). |
| New: `adapters/scratchpad_contract_check.py` | `check_scratchpad_update()` (A). |
| `adapters/generation_loop.py` | Added the new check to `validate_extra` (A). |
| `adapters/generation_errors.py`, `contracts/generation.py` | `ScratchpadContractError` + `scratchpad_contract_violation` category, classified like `BeliefContractError` (A). New: `adapters/repair_instructions.py` — extracted `CATEGORY_INSTRUCTIONS` to stay under the 100-line cap after adding the new category's instruction text. |
| `contracts/decision.py`, `runtime/context.py` | New `AgentContext.known_event_ids`/`scratchpad_token_budget` fields (not rendered in the prompt, same precedent as `hypothesis_versions`), needed so the early check can validate without a live store reference (A). |
| `evidence/cognitive_synthesis.py`, `contracts/cognitive_synthesis.py` | Bounded to the most recent 8 reports/reviews each, with `omitted_report_count`/`omitted_review_count` (B). |
| **`cognition/tasks.py`** | **Not changed** — already hard-capped at 32 tasks; see the §4-C correction above (C was scoped out as unnecessary). |
| New: `tests/test_fixed_prompt_overhead.py` | Scenario 7 (D) — tied to the real `RuntimeFields().soft_input_limit` default, not an ad hoc constant. |
| New: `tests/test_scratchpad_contract_repair.py` | Scenario 6 — reproduces both contract-violation shapes and confirms cheap repair. |
| New: `tests/test_cognitive_synthesis_bounds.py` | Scenario 3 — confirms the §4-B bound and omitted counts. |
| New: `tests/test_checkpoint_content_fidelity.py` | Scenario 5 — scratchpad+world-model+task+hypothesis all survive a fork; previously untested. |
| New: `tests/test_world_model_fork_limitation.py` | Scenario 2 — documents the confirmed, narrow §2 limitation (old world-model versions unreachable after a fork); **E is still an open question and was not fixed**, this only makes the existing behavior a named, tracked test instead of undemonstrated. |
| Scenario 1 (long history) | **Not added separately** — already reasonably exercised by the existing `test_memory_edges.py` compaction stress test; no new file needed. |
| Scenario 4 (large retrieved artifact) | **Not added** — already covered by the existing `tests/test_bounded_retrieval_projection.py`; would have duplicated it. |

All files stayed within the 100-line cap.

## 8. Acceptance criteria

- Scratchpad contract violations are repaired the same cheap way as tool/
  belief violations; zero behavior change for valid scratchpad updates.
- `COGNITIVE_SYNTHESIS`/`TASK_GRAPH` growth is bounded with a visible
  omitted count, same transparency pattern as every other bounded section.
- The fixed-prompt-overhead test exists, is tied to real settings
  defaults, and would have caught PR-01's growth with a clear message.
- No change to action authorization, evaluation behavior, or deterministic
  execution guarantees. Golden fixture unchanged. Full suite + `ruff` +
  `mypy` green.

## 9. Risks

- **Bounding `COGNITIVE_SYNTHESIS`/`TASK_GRAPH` could hide something the
  model needed.** Mitigated the same way every other bounded section in
  this codebase already is: an explicit omitted count plus full
  retrievability by event ID — never silent loss.
- **A scratchpad pre-check duplicates validation logic** between the new
  check and `ScratchpadStore.prepare()`. Resolved as planned: both now
  call the single shared `prepare_scratchpad()` function
  (`cognition/scratchpad_prepare.py`), so they cannot drift apart —
  the early check and the final invariant run the exact same code.
- **World-model version reconstruction-from-trace (E) adds real
  complexity** for a fork-only edge case — left as an open question,
  not implemented; `test_world_model_fork_limitation.py` documents the
  current behavior as a named, tracked test instead.

## Implementation status

A, B, and D are implemented (see §7); C was scoped out after finding
`TaskGraph` already has a hard 32-task cap; E remains an open question,
not implemented. Full suite (327 tests), `ruff check`, `ruff format
--check`, and `mypy` all pass; the golden fixture is unchanged; no
agent-policy, action-authorization, or evaluation-behavior change was
made anywhere in this work.
