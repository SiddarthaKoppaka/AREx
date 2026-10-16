"""Epistemic portion of the bounded working set, derived from the exact trace."""

from typing import Any

from arc_agi_3.cognition import CognitiveWorkspace
from arc_agi_3.config import AblationConfig
from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.observation import Observation
from arc_agi_3.contracts.transition import TransitionEvidence
from arc_agi_3.contracts.verification import VerificationResult
from arc_agi_3.evaluation_tool_repeats import (
    current_no_novelty_streak,
    tool_requests_since_last_action,
)
from arc_agi_3.evidence import (
    action_evidence,
    build_cognitive_synthesis,
    experiment_targets,
    hypothesis_ledger,
    tool_request_evidence,
    transition_records,
    unresolved_contradictions,
    untried_actions_from_current_state,
)

from .recent_events import current_observation_event_id


def epistemic_fields(
    history: list[EventEnvelope],
    observation: Observation,
    workspace: CognitiveWorkspace,
    ablations: AblationConfig,
) -> dict[str, Any]:
    """Latest transition/verification, hypothesis ledger, and action cues."""
    records = transition_records(history)
    targets = experiment_targets(history)
    current_observation = current_observation_event_id(
        history, observation.observation_hash
    )
    fields: dict[str, Any] = {
        "action_evidence": action_evidence(records, targets, observation),
        "cognitive_synthesis": build_cognitive_synthesis(history),
        "tool_evidence": tool_request_evidence(history, current_observation),
        "no_novelty_streak": current_no_novelty_streak(history),
        "tool_requests_since_last_action": tool_requests_since_last_action(history),
        "untried_actions_from_current_state": untried_actions_from_current_state(
            records, observation
        ),
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
        # Unbounded, unlike `hypotheses` above: validates proposals/updates
        # even against a hypothesis the ledger's display limit omitted.
        fields["hypothesis_versions"] = {
            item.hypothesis_id: item.version for item in workspace.beliefs.current
        }
    return fields
