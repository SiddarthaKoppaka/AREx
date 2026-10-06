"""The specialist backend's scripted outputs for the ExoCortex e2e test."""

import json


def hypothesis_report() -> str:
    return json.dumps(
        {
            "specialist_id": "S-hypothesis",
            "faculty": "hypothesis",
            "turn": 1,
            "assessment": "value 11 behaves like a static goal region",
            "hypotheses": [
                {"claim": "value 11 is a goal", "confidence": 0.7, "evidence_refs": []}
            ],
        }
    )


def critic_report() -> str:
    return json.dumps(
        {
            "specialist_id": "S-critic",
            "faculty": "critic",
            "turn": 2,
            "assessment": "the goal reading looks unsupported",
        }
    )


def critic_review() -> str:
    return json.dumps(
        {
            "specialist_id": "S-critic",
            "faculty": "critic",
            "turn": 3,
            "critiques": [
                {
                    "target_specialist_id": "S-hypothesis",
                    "target_claim": "value 11 is a goal",
                    "critique": "evidence only proves cells changed, not goal status",
                }
            ],
        }
    )


def specialist_outputs() -> list[str]:
    return [hypothesis_report(), critic_report(), critic_review()]
