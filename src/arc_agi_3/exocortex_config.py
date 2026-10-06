"""ExoCortex specialist-consultation settings; see ADR-0009.

No live model objects live here — the specialist backend is passed to
build_session separately, the same way the core agent's model already is.
"""

from pydantic import Field

from arc_agi_3.contracts.base import Contract
from arc_agi_3.contracts.enums import CognitiveFaculty


class ExoCortexConfig(Contract):
    faculties: tuple[CognitiveFaculty, ...] = (
        CognitiveFaculty.HYPOTHESIS,
        CognitiveFaculty.DYNAMICS,
        CognitiveFaculty.CRITIC,
    )
    max_repairs: int = Field(default=2, ge=0, le=2)
    persist_invalid_output: bool = False
    preview_chars: int = Field(default=300, gt=0)
    own_history_limit: int = Field(default=2, ge=0)
