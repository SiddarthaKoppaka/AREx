from arc_agi_3.contracts.observation import Action
from arc_agi_3.duck.history import ActionHistory


def act(action_id: int, **data: int) -> Action:
    return Action(action_id=action_id, data=data)


def test_empty_until_an_action_was_observed() -> None:
    history = ActionHistory()
    assert history.note() == ""
    history.decided(act(1))
    assert history.note() == ""


def test_reports_effect_of_last_action() -> None:
    history = ActionHistory()
    history.decided(act(1))
    history.observed(7)
    assert history.note() == "LAST_ACTION: ACTION1 -> changed 7 cells"
    history.decided(act(2))
    history.observed(0)
    assert "ACTION2 -> NO EFFECT" in history.note()


def test_warns_on_repeats_and_idle_streaks() -> None:
    history = ActionHistory()
    for _ in range(3):
        history.decided(act(1))
        history.observed(0)
    note = history.note()
    assert "ACTION1 repeated 3 times" in note
    assert "last 3 actions changed nothing" in note


def test_changing_action_resets_repeat_and_clicks_are_labelled() -> None:
    history = ActionHistory()
    for _ in range(3):
        history.decided(act(1))
    history.decided(act(6, x=4, y=5))
    history.observed(2)
    assert history.note() == "LAST_ACTION: ACTION6(4,5) -> changed 2 cells"
