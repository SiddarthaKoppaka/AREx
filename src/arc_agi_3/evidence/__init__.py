"""Deterministic, semantics-free evidence services for the LM's epistemic loop."""

from .action_cues import action_evidence, untried_actions_from_current_state
from .cognitive_synthesis import build_cognitive_synthesis
from .history import TransitionRecord, transition_records
from .ledger import hypothesis_ledger, unresolved_contradictions
from .repeats import detect_repeat
from .retrodiction import retrodict
from .specialist_memory import recent_reports
from .targets import experiment_targets
from .tool_batch import summarize_tool_batch
from .tool_cues import tool_fingerprint, tool_request_evidence
from .transition import build_transition

__all__ = [
    "TransitionRecord",
    "action_evidence",
    "build_cognitive_synthesis",
    "build_transition",
    "detect_repeat",
    "experiment_targets",
    "hypothesis_ledger",
    "recent_reports",
    "retrodict",
    "summarize_tool_batch",
    "tool_fingerprint",
    "tool_request_evidence",
    "transition_records",
    "untried_actions_from_current_state",
    "unresolved_contradictions",
]
