"""Belief math is explicit, normalized only within authored groups."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from arc_agi_3.cognition import BeliefStore
from arc_agi_3.config import BeliefConfig
from arc_agi_3.contracts.cognition import BeliefUpdate, Hypothesis
from arc_agi_3.contracts.enums import BeliefOperation


def hypothesis(key: str, probability: float, group: str | None = None) -> Hypothesis:
    return Hypothesis(
        hypothesis_id=key,
        version=1,
        claim=f"claim {key}",
        probability=probability,
        belief_group=group,
    )


@given(
    left=st.floats(min_value=1e-6, max_value=1, allow_nan=False),
    right=st.floats(min_value=1e-6, max_value=1, allow_nan=False),
)
def test_group_probabilities_remain_normalized(left: float, right: float) -> None:
    store = BeliefStore(BeliefConfig())
    store.add_many((hypothesis("left", left, "g"), hypothesis("right", right, "g")))
    store.apply(
        BeliefUpdate(
            hypothesis_id="left",
            expected_version=1,
            operation=BeliefOperation.SUPPORT,
            strength="strong",
            evidence_refs=("event-1",),
        )
    )
    assert sum(item.probability for item in store.current) == pytest.approx(1.0)


def test_independent_beliefs_are_not_normalized_together() -> None:
    store = BeliefStore(BeliefConfig())
    store.add_many((hypothesis("a", 0.5), hypothesis("b", 0.8)))
    store.apply(
        BeliefUpdate(
            hypothesis_id="a",
            expected_version=1,
            operation=BeliefOperation.SUPPORT,
            strength="moderate",
            evidence_refs=("evidence",),
        )
    )
    assert store.current[0].probability == pytest.approx(2 / 3)
    assert store.current[1].probability == 0.8


def test_stale_update_is_rejected_without_mutation() -> None:
    store = BeliefStore(BeliefConfig())
    store.add_many((hypothesis("a", 0.5),))
    request = BeliefUpdate(
        hypothesis_id="a",
        expected_version=2,
        operation=BeliefOperation.WEAKEN,
        strength="weak",
        evidence_refs=(),
    )
    with pytest.raises(ValueError, match="stale"):
        store.apply(request)
    assert store.current[0].version == 1
