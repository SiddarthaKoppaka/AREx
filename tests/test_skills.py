"""PR-09 gate: a skill cannot be constructed without real evidence,
aborts when its precondition is violated, respects per-game isolation
unless explicitly transfer-eligible, and the registry enforces version
lineage. See test_skills_firewall.py for the runner-level gate items."""

import pytest
from pydantic import ValidationError

from arc_agi_3.contracts.cognition import Hypothesis
from arc_agi_3.contracts.decision import AgentContext
from arc_agi_3.contracts.epistemic import HypothesisLedgerEntry
from arc_agi_3.skills import SkillLibrary, SkillPrecondition, instantiate_skill
from arc_agi_3.testing import FakeLineEnvironment
from tests.skills_fixtures import CHUNK, skill


def _context(**overrides: object) -> AgentContext:
    fields: dict[str, object] = {
        "turn": 0,
        "observation": FakeLineEnvironment().reset(),
        "budget": {},
    }
    fields.update(overrides)
    return AgentContext(**fields)


def test_a_skill_cannot_be_constructed_without_real_evidence() -> None:
    with pytest.raises(ValidationError):
        skill(evidence_refs=())


def test_precondition_violation_blocks_instantiation() -> None:
    required = skill(
        precondition=SkillPrecondition(
            required_hypothesis_ids=("h1",), min_confidence=0.5
        )
    )
    result = instantiate_skill(required, _context(), "fake-line")
    assert not result.applicable
    assert result.refusal_reason is not None and "h1" in result.refusal_reason


def test_precondition_met_instantiates_the_real_chunk() -> None:
    required = skill(precondition=SkillPrecondition(required_hypothesis_ids=("h1",)))
    h1 = Hypothesis(hypothesis_id="h1", version=1, claim="x", probability=0.9)
    context = _context(
        hypotheses=(h1,), hypothesis_ledger=(HypothesisLedgerEntry(hypothesis=h1),)
    )
    result = instantiate_skill(required, context, "fake-line")
    assert result.applicable
    assert result.chunk is CHUNK


def test_a_skill_is_not_eligible_on_a_different_game_unless_transfer_eligible() -> None:
    assert instantiate_skill(skill(), _context(), "other-game").refusal_reason
    result = instantiate_skill(skill(transfer_eligible=True), _context(), "other-game")
    assert result.applicable


def test_skill_library_enforces_version_lineage() -> None:
    library = SkillLibrary()
    library.add(skill(version=1))
    with pytest.raises(ValueError, match="requires version"):
        library.add(skill(version=3))
    library.add(skill(version=2))
    assert library.latest("walk").version == 2
