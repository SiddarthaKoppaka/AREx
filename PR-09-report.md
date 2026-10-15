# PR-09 — Reusable skill extraction and preconditions

> **Status:** Implemented. A genuinely new concept, but built as a thin
> wrapper around existing, unmodified execution machinery rather than a
> second way to run actions.

## Key design decision: a skill never executes anything itself

Checked `contracts/execution.py`/`runtime/session_chunk.py` before
designing anything: `ActionChunk`/`ChunkStep` already give exactly what a
"skill" needs to execute safely — each step still goes through
`authorize_action` individually, `chunk_guards.py:invalid_reason` already
hard-stops on a stale `required_before_hash` precondition
(`"stale_precondition"`), and interrupts/checkpoints already work
per-chunk. So `skills/instantiate.py:instantiate_skill()` does exactly
one thing: check the skill's own precondition and per-game eligibility,
and if both hold, hand back the skill's stored `ActionChunk` *unchanged*.
There is no second execution path to keep in sync with the first — this
is what makes "cannot bypass the action firewall" true by construction,
not by a check added after the fact.

## What's new

- `skills/contracts.py` — `Skill` (versioned, `game_id`-scoped,
  `transfer_eligible: bool = False` by default, `evidence_refs:
  tuple[str, ...] = Field(min_length=1)` — structurally impossible to
  construct a skill with no real evidence) and `SkillPrecondition`
  (required hypothesis IDs + minimum confidence, required verified fact
  IDs — checked against the model's *own* hypothesis ledger and
  scratchpad, never a hardcoded game solution).
- `skills/preconditions.py` — `check_precondition()` (pure, returns the
  specific unmet requirement or `None`) and `is_eligible_for_game()`
  (per-game isolation: the handoff.md invariant "no leakage of inferred
  hidden game dynamics across isolated games... general skills must
  declare eligibility" implemented as a one-line rule).
- `skills/registry.py` — `SkillLibrary`, mirroring
  `world_model/store.py`'s version-lineage pattern.
- `skills/instantiate.py` — the thin wrapper described above.

## Gate verification

- **"validated macro reduces a repeated sequence cost"** —
  `test_a_validated_skill_reduces_model_call_cost_for_a_repeated_sequence`:
  3 actions via one skill-derived chunk costs 2 fewer `MODEL_CALLS` than
  3 separate single-action decisions (measured, not assumed).
- **"aborts on violated precondition"** —
  `test_precondition_violation_blocks_instantiation`: a missing required
  hypothesis correctly refuses instantiation with a specific reason.
- **"cannot bypass the action firewall"** —
  `test_a_skill_chunk_still_hard_stops_on_a_violated_per_step_precondition`:
  a skill's chunk with a stale `required_before_hash` on its second step
  still hard-stops after exactly one real action, through the
  *unmodified* runner path.
- **"never imports hidden game solutions"** —
  `test_a_skill_cannot_be_constructed_without_real_evidence`: empty
  `evidence_refs` raises `ValidationError` at construction time.
- Per-game isolation and registry lineage each get a dedicated test.

## Explicitly not done

Not wired into `CognitiveDecision` or any LM-facing tool yet — same
pattern as PR-03/04/05: land the contract and its proven-correct behavior
first, wire it into the live loop once there's a concrete need (likely
alongside PR-10, once the harness-improvement loop needs something to
actually reuse).

## Validation

Full suite (390 tests), `ruff check`, `ruff format --check`, `mypy` all
pass. Golden fixture unchanged. No change to action authorization or
evaluation behavior — `instantiate_skill` only ever returns an unmodified
`ActionChunk`.
