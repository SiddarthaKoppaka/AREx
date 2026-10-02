"""Per-action evidence table: what each action did before, from which state."""

from arc_agi_3.contracts.epistemic import ActionEvidence, ActionOutcomeSummary
from arc_agi_3.contracts.observation import Action, Observation
from arc_agi_3.trace.canonical import canonical_json

from .history import TransitionRecord
from .targets import targeted


def _translations(record: TransitionRecord) -> tuple[str, ...]:
    moved = record.transition_event.payload.get("translations")
    if not isinstance(moved, list):
        return ()
    return tuple(
        f"value {m.get('value')} at {[m.get('from_row'), m.get('from_col')]} "
        f"shifted {[m.get('d_row'), m.get('d_col')]}"
        for m in moved[:3]
        if isinstance(m, dict)
    )


def action_evidence(
    records: list[TransitionRecord],
    targets: dict[str, tuple[str, ...]],
    observation: Observation,
    *,
    limit: int = 8,
    outcome_limit: int = 3,
) -> tuple[ActionEvidence, ...]:
    grouped: dict[str, list[TransitionRecord]] = {}
    actions: dict[str, Action] = {}
    for action_id in observation.available_actions:
        if action_id != 6:
            action = Action(action_id=action_id)
            actions[canonical_json(action)] = action
    for record in records:
        key = canonical_json(record.action)
        actions.setdefault(key, record.action)
        grouped.setdefault(key, []).append(record)
    current = observation.observation_hash
    table: list[ActionEvidence] = []
    for key, action in list(actions.items())[:limit]:
        tried = grouped.get(key, [])
        hypotheses = sorted({h for record in tried for h in targeted(record, targets)})
        table.append(
            ActionEvidence(
                action=action,
                attempts=len(tried),
                attempts_from_current_state=sum(
                    r.before_hash == current for r in tried
                ),
                hypotheses_targeted=tuple(hypotheses[:8]),
                recent_outcomes=tuple(
                    ActionOutcomeSummary(
                        transition_event_id=record.transition_event.event_id,
                        step_id=record.transition_event.step_id,
                        from_current_state=record.before_hash == current,
                        changed_cells=record.changed_cells,
                        translations=_translations(record),
                        verification_status=record.verification_status,
                    )
                    for record in tried[-outcome_limit:]
                ),
            )
        )
    return tuple(table)
