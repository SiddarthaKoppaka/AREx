"""Exact transition evidence and checks of LM-authored observable predictions."""

from arc_agi_3.contracts.observation import Observation
from arc_agi_3.contracts.transition import TransitionEvidence
from arc_agi_3.evidence.transition import build_transition
from arc_agi_3.evidence.verdict import prediction_signature as prediction_signature
from arc_agi_3.evidence.verdict import summarize_checks as summarize_checks
from arc_agi_3.evidence.verdict import verify_outcome as verify_outcome


def compare_observations(before: Observation, after: Observation) -> TransitionEvidence:
    return build_transition(before, after)
