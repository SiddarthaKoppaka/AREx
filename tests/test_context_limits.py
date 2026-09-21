"""Soft compaction removes optional memory before the hard input guard."""

from types import SimpleNamespace

import pytest

from arc_agi_3.adapters.context_compaction import compact_for_backend
from arc_agi_3.adapters.prompts import decision_prompt
from arc_agi_3.adapters.transformers_limits import InputTokenLimitError
from arc_agi_3.contracts.decision import AgentContext, CognitiveDecision
from arc_agi_3.contracts.enums import EventType
from arc_agi_3.contracts.memory import EpisodeMemory, EpisodeMemoryItem
from arc_agi_3.contracts.retrieval import ContextEvent
from arc_agi_3.contracts.scratchpad import WorkingScratchpad
from arc_agi_3.testing import FakeLineEnvironment


def event(number: int) -> ContextEvent:
    return ContextEvent(
        event_id=f"event-{number}",
        event_hash=f"hash-{number}",
        sequence=number,
        step_id=number,
        event_type=EventType.TRANSITION,
        component="test",
        causal_refs=(),
        payload={"note": "x" * 80},
    )


def context() -> AgentContext:
    items = tuple(
        EpisodeMemoryItem(
            step_id=index,
            turn_range=(index, index),
            first_sequence=index,
            last_sequence=index,
            summary=f"Turn {index}",
            facts_learned=(),
            failed_approaches=(),
            evidence_refs=(f"event-{index}",),
            source_event_refs=(f"event-{index}",),
            events=(event(index),),
        )
        for index in range(5)
    )
    return AgentContext(
        turn=6,
        observation=FakeLineEnvironment().reset(),
        budget={},
        recent_events=tuple(event(index + 10) for index in range(5)),
        working_scratchpad=WorkingScratchpad(objective="Keep this required state."),
        episodic_memory=EpisodeMemory(
            source_event_count=5,
            summarized_event_count=5,
            omitted_event_count=0,
            items=items,
        ),
    )


class Counter:
    def __init__(self, soft: int, hard: int) -> None:
        self.config = SimpleNamespace(
            soft_input_limit=soft, max_input_tokens=hard, model_name="test-model"
        )

    def input_token_count(self, prompt: str, schema: dict[str, object]) -> int:
        return len(prompt)


def test_soft_limit_prunes_optional_history_and_keeps_required_context() -> None:
    original = context()
    schema = CognitiveDecision.model_json_schema()
    base = original.model_copy(update={"recent_events": original.recent_events[-2:]})
    soft = len(decision_prompt(base.model_copy(update={"episodic_memory": None}))) + 100
    compacted = compact_for_backend(original, Counter(soft, soft + 1000), schema)
    assert compacted.observation == original.observation
    assert compacted.working_scratchpad == original.working_scratchpad
    assert len(compacted.recent_events) == 2
    assert compacted.episodic_memory is not None
    assert len(compacted.episodic_memory.items) < 5


def test_hard_limit_refuses_when_required_context_is_too_large() -> None:
    original = context().model_copy(update={"episodic_memory": None})
    with pytest.raises(InputTokenLimitError, match="turn=6"):
        compact_for_backend(
            original, Counter(1, 2), CognitiveDecision.model_json_schema()
        )
