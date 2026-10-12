"""Shared fixtures for System-1 (JEV) tests."""

from pydantic import JsonValue

from arc_agi_3.contracts.system1 import System1DecisionRequest, System1DecisionResult


class FakeSystem1Backend:
    """Implements System1Backend; queues one scripted result per call."""

    def __init__(self, results: list[System1DecisionResult | Exception]) -> None:
        self.results = list(results)
        self.requests: list[System1DecisionRequest] = []

    @property
    def metadata(self) -> dict[str, JsonValue]:
        return {"provider": "fake-system1"}

    def decide(self, request: System1DecisionRequest) -> System1DecisionResult:
        self.requests.append(request)
        item = self.results.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def two_candidate_request(trial_id: str = "trial-1") -> dict[str, object]:
    return {
        "trial_id": trial_id,
        "goal": "Determine the most useful immediate step.",
        "state_summary": "Left failed twice; movement mapping otherwise known.",
        "execution_mode": "suggest",
        "candidates": [
            {
                "candidate_id": "inspect-region",
                "description": "Inspect the collision region.",
                "tool_name": "inspect_frame_region",
            },
            {
                "candidate_id": "move-down",
                "description": "Move down to test directionality.",
                "action_id": 2,
            },
        ],
    }


def result_selecting(
    candidate_id: str, trial_id: str = "trial-1"
) -> System1DecisionResult:
    return System1DecisionResult(
        trial_id=trial_id,
        selected_candidate_id=candidate_id,
        candidate_probabilities=(
            {"candidate_id": "inspect-region", "probability": 0.6},
            {"candidate_id": "move-down", "probability": 0.4},
        ),
        confidence=0.8,
        stalled_probability=0.7,
    )
