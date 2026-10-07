"""A hypothesis mutation that conflicts with the current belief state is
repaired as cheaply as malformed JSON, not surfaced as a fatal BeliefStore
crash that kills the whole episode. See 'Contextual Decision Contracts v1'."""

from arc_agi_3.adapters.structured_model import StructuredModelAdapter
from arc_agi_3.contracts.cognition import BeliefUpdate, Hypothesis
from arc_agi_3.contracts.decision import AgentContext, CognitiveDecision
from arc_agi_3.contracts.enums import BeliefOperation, DecisionMode
from arc_agi_3.testing import FakeLineEnvironment
from tests.generation_fixtures import ScriptedBackend


def context_with(hypothesis_versions: dict[str, int]) -> AgentContext:
    return AgentContext(
        turn=2,
        observation=FakeLineEnvironment().reset(),
        budget={},
        hypothesis_versions=hypothesis_versions,
    )


def decision_json(**fields: object) -> str:
    decision = CognitiveDecision(
        assessment="x", intent="x", mode=DecisionMode.INVESTIGATE, **fields
    )
    return decision.model_dump_json()


def test_reproposing_an_existing_id_is_repaired_into_an_update() -> None:
    bad = decision_json(
        hypothesis_proposals=(
            Hypothesis(
                hypothesis_id="h1", version=1, claim="new claim", probability=0.5
            ),
        )
    )
    good = decision_json(
        hypothesis_updates=(
            BeliefUpdate(
                hypothesis_id="h1",
                expected_version=1,
                operation=BeliefOperation.SUPPORT,
                strength="moderate",
                evidence_refs=(),
            ),
        )
    )
    backend = ScriptedBackend([bad, good])
    response = StructuredModelAdapter(backend).decide(context_with({"h1": 1}))
    assert [a.role for a in response.attempts] == ["primary", "repair"]
    assert response.attempts[0].error_category == "belief_contract_violation"
    assert response.attempts[1].valid is True


def test_stale_expected_version_is_repaired() -> None:
    bad = decision_json(
        hypothesis_updates=(
            BeliefUpdate(
                hypothesis_id="h1",
                expected_version=1,
                operation=BeliefOperation.SUPPORT,
                strength="moderate",
                evidence_refs=(),
            ),
        )
    )
    good = decision_json(
        hypothesis_updates=(
            BeliefUpdate(
                hypothesis_id="h1",
                expected_version=2,
                operation=BeliefOperation.SUPPORT,
                strength="moderate",
                evidence_refs=(),
            ),
        )
    )
    backend = ScriptedBackend([bad, good])
    response = StructuredModelAdapter(backend).decide(context_with({"h1": 2}))
    assert response.attempts[0].error_category == "belief_contract_violation"
    assert "expected_version" in (response.attempts[0].validation_error or "")
    assert response.attempts[1].valid is True


def test_same_id_in_proposals_and_updates_is_repaired() -> None:
    bad = decision_json(
        hypothesis_proposals=(
            Hypothesis(hypothesis_id="h1", version=1, claim="x", probability=0.5),
        ),
        hypothesis_updates=(
            BeliefUpdate(
                hypothesis_id="h1",
                expected_version=1,
                operation=BeliefOperation.SUPPORT,
                strength="moderate",
                evidence_refs=(),
            ),
        ),
    )
    good = decision_json(
        hypothesis_proposals=(
            Hypothesis(hypothesis_id="h1", version=1, claim="x", probability=0.5),
        )
    )
    backend = ScriptedBackend([bad, good])
    response = StructuredModelAdapter(backend).decide(context_with({}))
    assert response.attempts[0].error_category == "belief_contract_violation"
    assert response.attempts[1].valid is True


def test_supersedes_an_unknown_hypothesis_is_repaired() -> None:
    bad = decision_json(
        hypothesis_proposals=(
            Hypothesis(
                hypothesis_id="h2",
                version=1,
                claim="x",
                probability=0.5,
                supersedes=("h1",),
            ),
        )
    )
    good = decision_json(
        hypothesis_proposals=(
            Hypothesis(hypothesis_id="h2", version=1, claim="x", probability=0.5),
        )
    )
    backend = ScriptedBackend([bad, good])
    response = StructuredModelAdapter(backend).decide(context_with({}))
    assert response.attempts[0].error_category == "belief_contract_violation"
    assert response.attempts[1].valid is True
