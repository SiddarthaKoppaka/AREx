"""Epistemic portion of the bounded working set, derived from the exact trace."""

from typing import Any

from arc_agi_3.cognition import CognitiveWorkspace
from arc_agi_3.config import AblationConfig
from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.observation import Observation
from arc_agi_3.contracts.transition import TransitionEvidence
from arc_agi_3.contracts.verification import VerificationResult
from arc_agi_3.evidence import (
    action_evidence,
    experiment_targets,
    hypothesis_ledger,
    transition_records,
    unresolved_contradictions,
)


def epistemic_fields(
    history: list[EventEnvelope],
    observation: Observation,
    workspace: CognitiveWorkspace,
    ablations: AblationConfig,
) -> dict[str, Any]:
    """Latest transition/verification, hypothesis ledger, and action cues."""
    records = transition_records(history)
    targets = experiment_targets(history)
    fields: dict[str, Any] = {
        "action_evidence": action_evidence(records, targets, observation)
    }
    if records:
        latest = records[-1]
        payload = latest.transition_event.payload
        fields["latest_transition"] = TransitionEvidence.model_validate(payload)
        fields["latest_transition_event_id"] = latest.transition_event.event_id
        if "status" in latest.verification:
            fields["latest_verification"] = VerificationResult.model_validate(
                {**latest.verification, "delta": payload}
            )
    if ablations.hypotheses:
        entries, omitted = hypothesis_ledger(
            workspace.beliefs.current, records, targets
        )
        fields["hypothesis_ledger"] = entries
        fields["hypotheses"] = tuple(item.hypothesis for item in entries)
        fields["unresolved_contradictions"] = unresolved_contradictions(entries)
        fields["omitted_hypothesis_ids"] = omitted
    return fields
