"""Result of a deterministic recovery operation."""

from dataclasses import dataclass

from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.observation import Observation


@dataclass(frozen=True)
class RecoveryOutcome:
    observation: Observation
    observed_event: EventEnvelope
    stop_reason: str | None = None
