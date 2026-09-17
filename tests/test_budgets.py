"""Budget reservation and hard-limit properties."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from arc_agi_3.budgets import BudgetExceeded, BudgetLedger
from arc_agi_3.contracts.enums import Resource


@given(st.integers(min_value=0, max_value=20))
def test_action_budget_never_overdraws(requested: int) -> None:
    ledger = BudgetLedger({Resource.ACTIONS: 10})
    if requested <= 10:
        if requested:
            ledger.spend(Resource.ACTIONS, requested)
        assert ledger.snapshot().consumed.get(Resource.ACTIONS, 0) == requested
    else:
        with pytest.raises(BudgetExceeded):
            ledger.spend(Resource.ACTIONS, requested)


def test_release_refunds_reservation() -> None:
    ledger = BudgetLedger({Resource.ACTIONS: 2}, id_factory=lambda: "r1")
    reservation = ledger.reserve({Resource.ACTIONS: 2})
    assert ledger.remaining()[Resource.ACTIONS] == 0
    ledger.release(reservation.reservation_id)
    assert ledger.remaining()[Resource.ACTIONS] == 2
