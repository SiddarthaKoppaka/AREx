"""Offline action-consistency checks never enter the live Kaggle callback."""

from arc_agi_3.contracts.decision import AgentContext, CognitiveDecision
from arc_agi_3.contracts.enums import DecisionMode
from arc_agi_3.contracts.observation import Action
from arc_agi_3.research.context_consistency import compare_next_action
from arc_agi_3.testing import FakeLineEnvironment


def test_next_action_consistency_compares_two_offline_projections() -> None:
    context = AgentContext(
        turn=1,
        observation=FakeLineEnvironment().reset(),
        budget={},
    )
    compact = context.model_copy(update={"recent_event_refs": ("prior-event",)})
    calls = 0

    def decide(_context: AgentContext) -> CognitiveDecision:
        nonlocal calls
        calls += 1
        return CognitiveDecision(
            assessment="Move once.",
            intent="Advance.",
            mode=DecisionMode.EXECUTE,
            action=Action(action_id=1),
        )

    result = compare_next_action(decide, context, compact)
    assert calls == 2
    assert result.same_mode and result.same_action_authorization
    assert result.full_context_hash != result.compact_context_hash
