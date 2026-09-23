"""Pressure reduction and emergency refusal are separate limits."""

from types import SimpleNamespace

import pytest

from arc_agi_3.adapters.context_compaction import compact_for_backend
from arc_agi_3.adapters.prompts import decision_prompt
from arc_agi_3.adapters.transformers_limits import InputTokenLimitError
from arc_agi_3.contracts.decision import AgentContext, CognitiveDecision
from arc_agi_3.contracts.enums import EventType
from arc_agi_3.contracts.retrieval import ContextEvent
from arc_agi_3.testing import FakeLineEnvironment


class Counter:
    def __init__(self, pressure: int, target: int, emergency: int) -> None:
        self.config = SimpleNamespace(
            compaction_pressure_start=pressure,
            active_context_target=target,
            max_input_tokens=emergency + 1000,
            max_context_tokens=emergency + 128,
            max_new_tokens=64,
            template_and_generation_margin=64,
            model_name="test",
        )

    def input_token_count(self, prompt: str, schema: dict[str, object]) -> int:
        return len(prompt)


def _context() -> AgentContext:
    recent = tuple(
        ContextEvent(
            event_id=f"event-{index}",
            event_hash=f"hash-{index}",
            sequence=index,
            step_id=index,
            event_type=EventType.TRANSITION,
            component="test",
            causal_refs=(),
            payload={"note": "x" * 100},
        )
        for index in range(5)
    )
    return AgentContext(
        turn=5,
        observation=FakeLineEnvironment().reset(),
        budget={},
        recent_event_refs=tuple(event.event_id for event in recent),
        recent_events=recent,
    )


def test_pressure_prunes_derived_recent_history_before_target() -> None:
    context = _context()
    original_size = len(decision_prompt(context))
    counter = Counter(original_size - 1, original_size + 100, original_size + 500)
    compacted = compact_for_backend(
        context, counter, CognitiveDecision.model_json_schema()
    )
    assert len(compacted.recent_events) == 2
    assert compacted.observation == context.observation
    assert compacted.recent_event_refs == tuple(
        event.event_id for event in compacted.recent_events
    )


def test_emergency_uses_model_window_minus_output_and_margin() -> None:
    context = _context()
    counter = Counter(1, 2, 3)
    with pytest.raises(InputTokenLimitError) as caught:
        compact_for_backend(context, counter, CognitiveDecision.model_json_schema())
    assert caught.value.limit == 3
    assert caught.value.actual > 3
