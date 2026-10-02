"""Decision interface and the bounded working set shown to the LM agent."""

from pydantic import Field

from arc_agi_3.world_model.contracts import WorldModel

from . import cognition
from .base import Contract
from .decision_core import BudgetRequest as BudgetRequest
from .decision_core import CognitiveDecision as CognitiveDecision
from .decision_core import PublicSummary as PublicSummary
from .enums import Resource
from .epistemic import ActionEvidence, ContextStats, HypothesisLedgerEntry
from .execution import ExpectedOutcome as ExpectedOutcome
from .memory import EpisodeMemory
from .model_io import ModelAttempt as ModelAttempt
from .model_io import ModelResponse as ModelResponse
from .model_io import ModelUsage as ModelUsage
from .model_io import PromptReport as PromptReport
from .observation import Observation
from .recovery import RecoveryEvidence
from .retrieval import ContextEvent
from .scratchpad import WorkingScratchpad
from .transition import TransitionEvidence
from .verification import VerificationResult


class AgentContext(Contract):
    turn: int = Field(ge=0)
    observation: Observation
    budget: dict[Resource, int]
    recent_event_refs: tuple[str, ...] = ()
    recent_events: tuple[ContextEvent, ...] = ()
    working_scratchpad: WorkingScratchpad | None = None
    episodic_memory: EpisodeMemory | None = None
    hypotheses: tuple[cognition.Hypothesis, ...] = ()
    tasks: tuple[cognition.TaskView, ...] = ()
    world_models: tuple[WorldModel, ...] = ()
    recovery_evidence: RecoveryEvidence | None = None
    latest_transition: TransitionEvidence | None = None
    latest_transition_event_id: str | None = None
    latest_verification: VerificationResult | None = None
    hypothesis_ledger: tuple[HypothesisLedgerEntry, ...] = ()
    unresolved_contradictions: tuple[str, ...] = ()
    action_evidence: tuple[ActionEvidence, ...] = ()
    context_stats: ContextStats | None = None
