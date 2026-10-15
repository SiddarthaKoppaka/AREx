"""LM-authored executable transition-model candidates. See ADR-V2-002/003.

Alongside, not replacing, the declarative `world_model/` path. A candidate
is provisional until replay/held-out tests confirm its scope (ADR-V2-003).
"""

from typing import Literal

from pydantic import Field, JsonValue, model_validator

from arc_agi_3.contracts.base import Contract

ValidationStatus = Literal["untested", "passed", "falsified", "error"]

REQUIRED_FUNCTIONS = ("encode_observation", "transition", "terminal_or_goal")


class ExecutionProfile(Contract):
    """Resource ceilings for one sandboxed call. See `sandbox_executor.py`."""

    cpu_seconds: float = Field(default=2.0, gt=0, le=30)
    wall_seconds: float = Field(default=5.0, gt=0, le=60)
    memory_bytes: int = Field(default=256 * 1024 * 1024, gt=0)
    output_bytes: int = Field(default=65_536, gt=0)


class ExecutableModelCandidate(Contract):
    model_id: str = Field(min_length=1)
    version: int = Field(ge=1)
    parent_version: int | None = Field(default=None, ge=1)
    source_artifact_ref: str
    source_sha256: str = Field(min_length=64, max_length=64)
    interface_version: str = "v1"
    state_schema: dict[str, JsonValue] = Field(default_factory=dict)
    assumptions: tuple[str, ...] = ()
    hypothesis_ids: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    validation_status: ValidationStatus = "untested"
    test_suite_ref: str | None = None
    holdout_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    counterexample_refs: tuple[str, ...] = ()
    execution_profile: ExecutionProfile = Field(default_factory=ExecutionProfile)

    @model_validator(mode="after")
    def valid_lineage(self) -> "ExecutableModelCandidate":
        if self.parent_version is not None and self.parent_version >= self.version:
            raise ValueError("parent_version must be strictly less than version")
        return self
