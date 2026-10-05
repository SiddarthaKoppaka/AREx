"""Fixtures for the Scientist-vs-Skeptic classroom end-to-end test."""

import json

from arc_agi_3.contracts.decision import CognitiveDecision
from arc_agi_3.contracts.enums import DecisionMode
from arc_agi_3.contracts.observation import Action
from arc_agi_3.testing import ScriptedModel


def one_shot_teacher() -> ScriptedModel:
    return ScriptedModel(
        [
            CognitiveDecision(
                assessment="advance",
                intent="finish",
                mode=DecisionMode.EXECUTE,
                action=Action(action_id=1),
            )
        ]
    )


def scientist_report() -> str:
    return json.dumps(
        {
            "student_id": "S-scientist",
            "role": "scientist",
            "turn": 1,
            "assessment": "value 11 behaves like a static goal region",
            "hypotheses": [
                {"claim": "value 11 is a goal", "confidence": 0.7, "evidence_refs": []}
            ],
        }
    )


def world_modeler_report() -> str:
    return json.dumps(
        {
            "student_id": "S-world_modeler",
            "role": "world_modeler",
            "turn": 1,
            "assessment": "ok",
        }
    )


def skeptic_report() -> str:
    return json.dumps(
        {
            "student_id": "S-skeptic",
            "role": "skeptic",
            "turn": 1,
            "assessment": "the goal reading is unsupported",
            "contradictions": [
                {
                    "description": "11 changed value; a static goal would not change",
                    "evidence_refs": [],
                }
            ],
        }
    )


def peer_review(student_id: str, role: str) -> str:
    critiques = (
        [
            {
                "target_student_id": "S-scientist",
                "target_claim": "value 11 is a goal",
                "critique": "evidence only proves cells changed, not goal status",
            }
        ]
        if student_id == "S-skeptic"
        else []
    )
    return json.dumps(
        {"student_id": student_id, "role": role, "turn": 1, "critiques": critiques}
    )


def student_outputs() -> list[str]:
    return [
        scientist_report(),
        world_modeler_report(),
        skeptic_report(),
        peer_review("S-scientist", "scientist"),
        peer_review("S-world_modeler", "world_modeler"),
        peer_review("S-skeptic", "skeptic"),
    ]
