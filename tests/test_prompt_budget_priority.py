"""Compaction trims recent turns and episodes first; live evidence survives."""

from arc_agi_3.adapters.context_compaction import compact_for_backend
from arc_agi_3.contracts.cognition import Hypothesis
from arc_agi_3.contracts.decision import AgentContext, CognitiveDecision
from arc_agi_3.contracts.enums import EventType
from arc_agi_3.contracts.memory import EpisodeMemory, EpisodeMemoryItem
from arc_agi_3.contracts.retrieval import ContextEvent
from arc_agi_3.contracts.transition import TransitionEvidence
from arc_agi_3.contracts.verification import VerificationResult
from arc_agi_3.testing import FakeLineEnvironment
from tests.generation_fixtures import TokenCounter


def recent(count: int) -> tuple[ContextEvent, ...]:
    return tuple(
        ContextEvent(
            event_id=f"event-{index}",
            event_hash=f"hash-{index}",
            sequence=index,
            step_id=index,
            event_type=EventType.TRANSITION,
            component="test",
            causal_refs=(),
            payload={"note": "x" * 80},
        )
        for index in range(count)
    )


def episode_items(count: int) -> tuple[EpisodeMemoryItem, ...]:
    return tuple(
        EpisodeMemoryItem(
            step_id=index,
            turn_range=(index, index),
            first_sequence=index,
            last_sequence=index,
            summary=f"Turn {index} summary " + "x" * 60,
            evidence_refs=(f"event-{index}",),
            source_event_refs=(f"event-{index}",),
        )
        for index in range(count)
    )


def heavy_context() -> AgentContext:
    transition = TransitionEvidence(before_hash="a", after_hash="b", changed_cells=1)
    return AgentContext(
        turn=9,
        observation=FakeLineEnvironment().reset(),
        budget={},
        recent_events=recent(40),
        episodic_memory=EpisodeMemory(
            source_event_count=20,
            summarized_event_count=20,
            omitted_event_count=0,
            items=episode_items(20),
        ),
        hypotheses=(
            Hypothesis(
                hypothesis_id="h1", version=1, claim="live claim", probability=0.6
            ),
        ),
        latest_transition=transition,
        latest_transition_event_id="transition-event",
        latest_verification=VerificationResult(
            status="matched", checked=True, passed=True, delta=transition
        ),
        unresolved_contradictions=("h1: unresolved",),
    )


def test_compaction_trims_low_priority_context_before_live_evidence() -> None:
    context = heavy_context()
    schema = CognitiveDecision.model_json_schema()
    compacted = compact_for_backend(context, TokenCounter(4000), schema)
    assert len(compacted.recent_events) < len(context.recent_events)
    memory = compacted.episodic_memory
    assert memory is not None and len(memory.items) < 20
    assert compacted.hypotheses == context.hypotheses
    assert compacted.latest_transition == context.latest_transition
    assert compacted.latest_verification == context.latest_verification
    assert compacted.unresolved_contradictions == context.unresolved_contradictions
    assert compacted.working_scratchpad == context.working_scratchpad
