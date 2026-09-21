"""Large turns and requested evidence remain bounded and useful."""

from types import SimpleNamespace

from arc_agi_3.adapters.context_compaction import compact_for_backend
from arc_agi_3.context import compact_context_event
from arc_agi_3.contracts.decision import AgentContext, CognitiveDecision
from arc_agi_3.contracts.enums import EventType
from arc_agi_3.contracts.retrieval import ContextEvent
from arc_agi_3.testing import FakeLineEnvironment
from arc_agi_3.trace.store import JsonlEventStore


class CharacterCounter:
    config = SimpleNamespace(
        soft_input_limit=6000, max_input_tokens=9000, model_name="test"
    )

    def input_token_count(self, prompt: str, schema: dict[str, object]) -> int:
        return len(prompt)


def test_soft_compaction_bounds_a_long_single_turn() -> None:
    recent = tuple(
        ContextEvent(
            event_id=f"event-{index}",
            event_hash=f"hash-{index}",
            sequence=index,
            step_id=2,
            event_type=EventType.ACTION,
            component="executor",
            causal_refs=(),
            payload={"action": index},
        )
        for index in range(80)
    )
    context = AgentContext(
        turn=3,
        observation=FakeLineEnvironment().reset(),
        budget={},
        recent_event_refs=tuple(item.event_id for item in recent),
        recent_events=recent,
    )
    backend = CharacterCounter()
    compacted = compact_for_backend(
        context, backend, CognitiveDecision.model_json_schema()
    )
    assert len(compacted.recent_events) <= 12
    assert compacted.recent_event_refs == tuple(
        item.event_id for item in compacted.recent_events
    )
    assert compacted.observation == context.observation


def test_requested_small_frame_is_visible_in_tool_result(tmp_path) -> None:
    store = JsonlEventStore(tmp_path / "events.jsonl", "run", "episode", "main")
    event = store.append(
        EventType.TOOL_RESULT,
        "tools",
        1,
        {
            "result": {
                "status": "complete",
                "output": {
                    "retrieval": {
                        "matches": [
                            {
                                "event_id": "prior-observation",
                                "event_type": "observation",
                                "payload": {"frame": [[[1, 2], [3, 4]]]},
                            }
                        ]
                    }
                },
            }
        },
    )
    projected = compact_context_event(event)
    result = projected.payload["result"]
    assert isinstance(result, dict)
    matches = result["retrieved_matches"]
    assert isinstance(matches, list)
    assert matches[0]["payload"]["frame"] == [[[1, 2], [3, 4]]]
