"""Reconstruct executed transitions from the immutable event trace."""

from dataclasses import dataclass
from typing import Any

from pydantic import JsonValue

from arc_agi_3.contracts.enums import EventType
from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.observation import Action, Observation


@dataclass(frozen=True)
class TransitionRecord:
    action_event: EventEnvelope
    transition_event: EventEnvelope
    before_event: EventEnvelope | None
    after_event: EventEnvelope
    verification_event: EventEnvelope | None

    @property
    def action(self) -> Action:
        return Action.model_validate(self.action_event.payload.get("action"))

    @property
    def before_hash(self) -> str | None:
        value = self.action_event.payload.get("before_hash")
        return value if isinstance(value, str) else None

    @property
    def changed_cells(self) -> int:
        value = self.transition_event.payload.get("changed_cells")
        return value if isinstance(value, int) else 0

    @property
    def verification(self) -> dict[str, JsonValue]:
        return self.verification_event.payload if self.verification_event else {}

    @property
    def verification_status(self) -> Any:
        return self.verification.get("status")

    @property
    def tested_hypotheses(self) -> tuple[str, ...]:
        value = self.verification.get("hypothesis_ids")
        return tuple(str(item) for item in value) if isinstance(value, list) else ()

    def observations(self) -> tuple[Observation, Observation] | None:
        if self.before_event is None:
            return None
        return (
            Observation.model_validate(self.before_event.payload),
            Observation.model_validate(self.after_event.payload),
        )


def transition_records(events: list[EventEnvelope]) -> list[TransitionRecord]:
    """Every action whose outcome was observed, oldest first."""
    by_id = {event.event_id: event for event in events}
    verifications = {
        ref: event
        for event in events
        if event.event_type is EventType.VERIFICATION
        for ref in event.causal_refs
    }
    latest_by_hash: dict[str, EventEnvelope] = {}
    before_of: dict[str, EventEnvelope | None] = {}
    records: list[TransitionRecord] = []
    for event in events:
        payload = event.payload
        if event.event_type is EventType.OBSERVATION:
            latest_by_hash[str(payload.get("observation_hash"))] = event
        elif event.event_type is EventType.ACTION:
            before_of[event.event_id] = latest_by_hash.get(
                str(payload.get("before_hash"))
            )
        elif event.event_type is EventType.TRANSITION and len(event.causal_refs) > 1:
            action, after = (by_id.get(ref) for ref in event.causal_refs[:2])
            if action is None or after is None or action.event_id not in before_of:
                continue
            records.append(
                TransitionRecord(
                    action,
                    event,
                    before_of[action.event_id],
                    after,
                    verifications.get(event.event_id),
                )
            )
    return records
