"""The entire Teacher-Student round, behind the existing ModelAdapter protocol.

Everything downstream (session.py, session_decide.py, decision.py) stays
unmodified: `request_decision` calls `model.decide(context)` exactly as
before. `io` is captured at construction, not threaded through `decide`'s
signature, so `ModelAdapter` itself never changes. See ADR-0008 and
docs/implementation/classroom_v1_plan.md.
"""

from pydantic import JsonValue

from arc_agi_3.adapters.inference import InferenceBackend
from arc_agi_3.adapters.protocols import ModelAdapter
from arc_agi_3.classroom_config import ClassroomConfig
from arc_agi_3.contracts.decision import AgentContext, ModelResponse
from arc_agi_3.contracts.enums import EventType
from arc_agi_3.evidence import build_synthesis

from .classroom_events import run_review, run_student
from .io import EpisodeIO


class ClassroomModelAdapter:
    def __init__(
        self,
        io: EpisodeIO,
        teacher: ModelAdapter,
        student_backend: InferenceBackend,
        config: ClassroomConfig,
    ) -> None:
        self.io, self.teacher = io, teacher
        self.student_backend, self.config = student_backend, config

    @property
    def metadata(self) -> dict[str, JsonValue]:
        return {
            "adapter": "classroom",
            "teacher": self.teacher.metadata,
            "students": self.student_backend.metadata,
            "roles": list(self.config.roles),
        }

    def decide(self, context: AgentContext) -> ModelResponse:
        history = self.io.events.read()
        reports = tuple(
            run_student(
                self.io, self.student_backend, self.config, role, context, history
            )
            for role in self.config.roles
        )
        reviews = tuple(
            run_review(self.io, self.student_backend, self.config, report, reports)
            for report in reports
        )
        synthesis = build_synthesis(context.turn, reports, reviews)
        self.io.append(
            EventType.CLASSROOM_SYNTHESIS,
            "classroom",
            context.turn,
            synthesis.model_dump(mode="json"),
            tuple(r.student_id for r in reports),
        )
        augmented = context.model_copy(update={"classroom": synthesis})
        return self.teacher.decide(augmented)
