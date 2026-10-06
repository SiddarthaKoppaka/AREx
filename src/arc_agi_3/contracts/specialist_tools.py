"""Arguments for the two tools that let the core agent consult ExoCortex
specialist faculties on demand. See ADR-0009."""

from pydantic import Field

from .base import Contract
from .cognitive_faculty import CognitiveFaculty


class SpecialistConsultRequest(Contract):
    faculty: CognitiveFaculty
    question: str | None = Field(default=None, max_length=240)


class SpecialistCompareRequest(Contract):
    primary_faculty: CognitiveFaculty
    other_faculties: tuple[CognitiveFaculty, ...] = Field(min_length=1, max_length=2)
