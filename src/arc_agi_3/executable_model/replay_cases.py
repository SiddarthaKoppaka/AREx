"""Build typed replay cases from the real, already-recorded trace.

Reuses `evidence/history.py`'s `transition_records` rather than
re-deriving before/action/after from events a second way. The train/
holdout split is a deterministic, seeded shuffle, not an LM choice.
"""

import random

from arc_agi_3.evidence.history import TransitionRecord

from .replay_contracts import TransitionCase


def build_transition_cases(
    records: list[TransitionRecord], *, seed: int = 0, holdout_fraction: float = 0.3
) -> tuple[TransitionCase, ...]:
    resolvable = [
        (record, observed)
        for record in records
        if (observed := record.observations()) is not None and record.before_event
    ]
    order = list(range(len(resolvable)))
    random.Random(seed).shuffle(order)
    holdout_count = round(len(order) * holdout_fraction)
    holdout_indices = set(order[:holdout_count])
    cases = []
    for index, (record, (before, after)) in enumerate(resolvable):
        assert record.before_event is not None
        cases.append(
            TransitionCase(
                case_id=record.transition_event.event_id,
                before_observation_event_id=record.before_event.event_id,
                action_event_id=record.action_event.event_id,
                after_transition_event_id=record.transition_event.event_id,
                before=before,
                action=record.action,
                after=after,
                split="holdout" if index in holdout_indices else "train",
            )
        )
    return tuple(cases)
