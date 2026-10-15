"""A scratchpad update that violates its contract is repaired as cheaply
as malformed JSON, not surfaced as a fatal crash that kills the episode.
See PR-02-scoping.md section 2 (confirmed defect) and section 4-A."""

from arc_agi_3.adapters.structured_model import StructuredModelAdapter
from arc_agi_3.contracts.decision import AgentContext, CognitiveDecision
from arc_agi_3.contracts.enums import DecisionMode
from arc_agi_3.contracts.scratchpad import (
    ScratchpadUpdates,
    VerifiedFact,
    WorkingScratchpad,
)
from arc_agi_3.testing import FakeLineEnvironment
from tests.generation_fixtures import ScriptedBackend


def context_with(
    known_event_ids: tuple[str, ...] = ("obs-1",), token_budget: int = 2048
) -> AgentContext:
    return AgentContext(
        turn=2,
        observation=FakeLineEnvironment().reset(),
        budget={},
        working_scratchpad=WorkingScratchpad(),
        known_event_ids=known_event_ids,
        scratchpad_token_budget=token_budget,
    )


def decision_json(**fields: object) -> str:
    decision = CognitiveDecision(
        assessment="x", intent="x", mode=DecisionMode.INVESTIGATE, **fields
    )
    return decision.model_dump_json()


def test_citing_an_unknown_evidence_ref_is_repaired() -> None:
    bad = decision_json(
        scratchpad_updates=ScratchpadUpdates(
            add_verified_fact=(
                VerifiedFact(
                    fact_id="f1",
                    version=1,
                    fact="x",
                    confidence=0.9,
                    evidence_refs=("model-decision-1",),
                ),
            )
        )
    )
    good = decision_json(
        scratchpad_updates=ScratchpadUpdates(
            add_verified_fact=(
                VerifiedFact(
                    fact_id="f1",
                    version=1,
                    fact="x",
                    confidence=0.9,
                    evidence_refs=("obs-1",),
                ),
            )
        )
    )
    backend = ScriptedBackend([bad, good])
    response = StructuredModelAdapter(backend).decide(context_with())
    assert [a.role for a in response.attempts] == ["primary", "repair"]
    assert response.attempts[0].error_category == "scratchpad_contract_violation"
    assert response.attempts[1].valid is True


def test_exceeding_the_scratchpad_token_budget_is_repaired() -> None:
    long_fact = "x" * 400
    bad = decision_json(
        scratchpad_updates=ScratchpadUpdates(
            add_verified_fact=(
                VerifiedFact(
                    fact_id="f1",
                    version=1,
                    fact=long_fact,
                    confidence=0.9,
                    evidence_refs=("obs-1",),
                ),
            )
        )
    )
    good = decision_json(
        scratchpad_updates=ScratchpadUpdates(set_objective="a shorter objective")
    )
    backend = ScriptedBackend([bad, good])
    response = StructuredModelAdapter(backend).decide(context_with(token_budget=64))
    assert response.attempts[0].error_category == "scratchpad_contract_violation"
    assert "token budget" in (response.attempts[0].validation_error or "")
    assert response.attempts[1].valid is True
