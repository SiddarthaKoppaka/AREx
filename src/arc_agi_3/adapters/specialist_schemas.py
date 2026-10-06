"""Model-facing SpecialistReport/SpecialistReview schemas, stripped for
repeated sending."""

from functools import cache
from typing import Any

from arc_agi_3.contracts.specialist_report import SpecialistReport
from arc_agi_3.contracts.specialist_review import SpecialistReview

from .schema_render import strip_schema


@cache
def specialist_report_schema() -> dict[str, Any]:
    stripped: dict[str, Any] = strip_schema(SpecialistReport.model_json_schema())
    return stripped


@cache
def specialist_review_schema() -> dict[str, Any]:
    stripped: dict[str, Any] = strip_schema(SpecialistReview.model_json_schema())
    return stripped
