"""Model-facing StudentReport/PeerReview schemas, stripped for repeated sending."""

from functools import cache
from typing import Any

from arc_agi_3.contracts.classroom import StudentReport
from arc_agi_3.contracts.peer_review import PeerReview

from .schema_render import strip_schema


@cache
def student_report_schema() -> dict[str, Any]:
    stripped: dict[str, Any] = strip_schema(StudentReport.model_json_schema())
    return stripped


@cache
def peer_review_schema() -> dict[str, Any]:
    stripped: dict[str, Any] = strip_schema(PeerReview.model_json_schema())
    return stripped
