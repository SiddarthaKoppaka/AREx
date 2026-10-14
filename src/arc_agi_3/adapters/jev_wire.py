"""Translate between System1DecisionRequest/Result and TypeSafe's real
/v1/systemone wire format (verified against api.typesafe.ai/openapi.json
2026-10-14). The internal contract stays vendor-neutral - this module is
the only place that knows TypeSafe's shape.

TypeSafe has no native "pick one of N candidates" primitive; a
ChoiceQuestion (criteria={name: description}, answer picks one with a
full probability distribution) is the closest fit and maps directly onto
our candidate set. ask_stalled/ask_need_more_perception become
NoulQuestions (yes/no probability); ask_information_gain becomes a
ScoreQuestion over a fixed ordered scale - TypeSafe returns a
probability-weighted index into that scale, passed through unscaled.
"""

from typing import Any

from arc_agi_3.contracts.system1 import (
    CandidateProbability,
    System1DecisionRequest,
    System1DecisionResult,
)

STALLED_CRITERIA = {
    "true": "the current approach is stalled and yielding no new information",
    "false": "the current approach still has promising next steps",
}
PERCEPTION_CRITERIA = {
    "true": "additional perception or inspection would likely help right now",
    "false": "current evidence is sufficient to decide",
}
INFORMATION_GAIN_LEVELS = [
    "no new information expected",
    "marginal information",
    "moderate information",
    "substantial information",
    "decisive information",
]


def build_request_body(model: str, request: System1DecisionRequest) -> dict[str, Any]:
    questions: dict[str, Any] = {
        "select_candidate": {
            "type": "choice",
            "instructions": (
                "Select the candidate operation that makes the most progress right now."
            ),
            "criteria": {c.candidate_id: c.description for c in request.candidates},
        }
    }
    if request.ask_stalled:
        questions["stalled"] = {"type": "noul", "criteria": STALLED_CRITERIA}
    if request.ask_need_more_perception:
        questions["need_more_perception"] = {
            "type": "noul",
            "criteria": PERCEPTION_CRITERIA,
        }
    if request.ask_information_gain:
        questions["information_gain"] = {
            "type": "score",
            "criteria": INFORMATION_GAIN_LEVELS,
        }
    return {
        "model": model,
        "state": {"goal": request.goal, "state_summary": request.state_summary},
        "questions": questions,
    }


def parse_response_body(
    request: System1DecisionRequest, raw: dict[str, Any]
) -> System1DecisionResult:
    answers = raw["answers"]
    choice = answers["select_candidate"]
    probabilities = choice.get("probabilities", {})
    return System1DecisionResult(
        trial_id=request.trial_id,
        selected_candidate_id=choice["choice"],
        candidate_probabilities=tuple(
            CandidateProbability(candidate_id=candidate_id, probability=probability)
            for candidate_id, probability in probabilities.items()
        ),
        confidence=choice.get("confidence"),
        stalled_probability=answers.get("stalled", {}).get("noul"),
        need_more_perception_probability=answers.get("need_more_perception", {}).get(
            "noul"
        ),
        information_gain_score=answers.get("information_gain", {}).get("score"),
    )
