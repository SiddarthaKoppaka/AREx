"""Translation to/from TypeSafe's real /v1/systemone wire shape is exact,
not approximate - the internal contract must stay vendor-neutral."""

from arc_agi_3.adapters.jev_wire import build_request_body, parse_response_body
from arc_agi_3.contracts.system1 import System1DecisionRequest
from tests.system1_fixtures import two_candidate_request, typesafe_response


def request(**overrides: object) -> System1DecisionRequest:
    payload = {**two_candidate_request(), **overrides}
    return System1DecisionRequest.model_validate(payload)


def test_select_candidate_is_always_a_choice_question() -> None:
    body = build_request_body("jev-latest", request())
    question = body["questions"]["select_candidate"]
    assert question["type"] == "choice"
    assert set(question["criteria"]) == {"inspect-region", "move-down"}


def test_optional_questions_are_omitted_when_not_asked() -> None:
    body = build_request_body(
        "jev-latest",
        request(
            ask_stalled=False,
            ask_need_more_perception=False,
            ask_information_gain=False,
        ),
    )
    assert set(body["questions"]) == {"select_candidate"}


def test_all_optional_questions_present_when_asked() -> None:
    body = build_request_body("jev-latest", request())
    assert body["questions"]["stalled"]["type"] == "noul"
    assert body["questions"]["need_more_perception"]["type"] == "noul"
    assert body["questions"]["information_gain"]["type"] == "score"


def test_state_carries_goal_and_state_summary_not_the_full_candidate_set() -> None:
    body = build_request_body("jev-latest", request())
    assert body["state"] == {
        "goal": "Determine the most useful immediate step.",
        "state_summary": "Left failed twice; movement mapping otherwise known.",
    }


def test_parse_response_extracts_choice_and_probabilities() -> None:
    result = parse_response_body(request(), typesafe_response("inspect-region"))
    assert result.selected_candidate_id == "inspect-region"
    assert result.confidence == 0.84
    assert result.stalled_probability == 0.91


def test_parse_response_tolerates_missing_optional_answers() -> None:
    result = parse_response_body(
        request(), typesafe_response("move-down", with_stalled=False)
    )
    assert result.stalled_probability is None
    assert result.need_more_perception_probability is None
    assert result.information_gain_score is None
