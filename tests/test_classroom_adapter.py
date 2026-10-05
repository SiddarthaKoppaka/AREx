"""ClassroomModelAdapter records one full round, then delegates to the Teacher."""

import json
from pathlib import Path

from arc_agi_3.config import RunConfig
from arc_agi_3.contracts.enums import DecisionMode, EventType, Resource, StudentRole
from arc_agi_3.runtime import build_session
from arc_agi_3.testing import FakeLineEnvironment, ScriptedModel, successful_script
from tests.generation_fixtures import ScriptedBackend


def report(student_id: str, role: str, turn: int = 1) -> str:
    return json.dumps(
        {"student_id": student_id, "role": role, "turn": turn, "assessment": "ok"}
    )


def review(student_id: str, role: str, turn: int = 1) -> str:
    return json.dumps({"student_id": student_id, "role": role, "turn": turn})


def student_outputs() -> list[str]:
    return [
        report("S-scientist", "scientist"),
        report("S-world_modeler", "world_modeler"),
        report("S-skeptic", "skeptic"),
        review("S-scientist", "scientist"),
        review("S-world_modeler", "world_modeler"),
        review("S-skeptic", "skeptic"),
    ]


def test_classroom_round_records_exactly_three_reports_reviews_and_one_synthesis(
    tmp_path: Path,
) -> None:
    students = ScriptedBackend(student_outputs(), input_tokens=10, output_tokens=5)
    teacher = ScriptedModel(successful_script())
    config = RunConfig(run_id="classroom", experiment_id="test", output_dir=tmp_path)
    session = build_session(config, FakeLineEnvironment(), teacher, students=students)
    session.start(FakeLineEnvironment().reset())
    action = session.next_action()
    assert action is not None

    events = session.io.events.read()
    kinds = [e.event_type for e in events]
    assert kinds.count(EventType.STUDENT_REPORT) == 3
    assert kinds.count(EventType.PEER_REVIEW) == 3
    assert kinds.count(EventType.CLASSROOM_SYNTHESIS) == 1
    decision = next(e for e in events if e.event_type is EventType.MODEL_DECISION)
    assert decision.payload["decision"]["mode"] == DecisionMode.EXECUTE


def test_classroom_charges_every_generations_usage_to_the_budget(
    tmp_path: Path,
) -> None:
    students = ScriptedBackend(student_outputs(), input_tokens=10, output_tokens=5)
    teacher = ScriptedModel(successful_script())
    config = RunConfig(
        run_id="classroom-budget", experiment_id="test", output_dir=tmp_path
    )
    session = build_session(config, FakeLineEnvironment(), teacher, students=students)
    session.start(FakeLineEnvironment().reset())
    session.next_action()
    consumed = session.io.ledger.consumed
    # 6 classroom generations x 10 input tokens; the Teacher adds its own on top.
    assert consumed[Resource.INPUT_TOKENS] >= 60


def test_teacher_prompt_receives_the_classroom_synthesis(tmp_path: Path) -> None:
    from arc_agi_3.contracts.decision import (
        AgentContext,
        CognitiveDecision,
        ModelResponse,
    )

    class CapturingTeacher:
        metadata = {"provider": "test"}

        def __init__(self) -> None:
            self.contexts: list[AgentContext] = []

        def decide(self, context: AgentContext) -> ModelResponse:
            self.contexts.append(context)
            return ModelResponse(
                decision=CognitiveDecision(
                    assessment="x", intent="x", mode=DecisionMode.STOP
                )
            )

    students = ScriptedBackend(student_outputs())
    teacher = CapturingTeacher()
    config = RunConfig(
        run_id="classroom-ctx", experiment_id="test", output_dir=tmp_path
    )
    session = build_session(config, FakeLineEnvironment(), teacher, students=students)
    session.start(FakeLineEnvironment().reset())
    session.next_action()
    classroom = teacher.contexts[0].classroom
    assert classroom is not None
    assert {r.role for r in classroom.reports} == {
        StudentRole.SCIENTIST,
        StudentRole.WORLD_MODELER,
        StudentRole.SKEPTIC,
    }
    assert len(classroom.peer_reviews) == 3
