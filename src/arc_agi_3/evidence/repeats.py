"""Flag likely repeated experiments; never blocks or overrides the LM."""

from arc_agi_3.contracts.epistemic import PriorAttempt, RepeatCue, RepeatKind
from arc_agi_3.contracts.observation import Action
from arc_agi_3.contracts.prediction import ExpectedOutcome, Experiment

from .history import TransitionRecord
from .targets import targeted
from .verdict import prediction_signature


def detect_repeat(
    records: list[TransitionRecord],
    targets: dict[str, tuple[str, ...]],
    *,
    before_hash: str,
    action: Action,
    expected: ExpectedOutcome | None,
    experiment: Experiment | None,
    limit: int = 4,
) -> RepeatCue:
    signature = prediction_signature(expected) if expected else None
    hypotheses = {
        *(expected.hypothesis_ids if expected else ()),
        *(experiment.hypothesis_ids if experiment else ()),
    }
    prior: list[PriorAttempt] = []
    for record in reversed(records):
        if record.action != action:
            continue
        match: RepeatKind | None = None
        if record.before_hash == before_hash:
            match = "same_state_same_action"
        elif signature and record.verification.get("prediction_signature") == signature:
            match = "same_action_same_prediction"
        elif hypotheses & targeted(record, targets):
            match = "same_action_same_hypotheses"
        if match is None:
            continue
        prior.append(
            PriorAttempt(
                action_event_id=record.action_event.event_id,
                transition_event_id=record.transition_event.event_id,
                match=match,
                changed_cells=record.changed_cells,
                verification_status=record.verification_status,
            )
        )
        if len(prior) >= limit:
            break
    return RepeatCue(
        equivalent_prior=tuple(prior),
        justification=experiment.repeat_justification if experiment else None,
    )
