"""Vendor-neutral System-1 decision backend protocol. See ADR-0010.

Deliberately not `InferenceBackend`: System-1 answers a typed question
with typed structured judgments, not free-text generation against a JSON
schema, so it gets its own minimal protocol rather than reusing/stretching
the LM-generation one.
"""

from typing import Protocol

from pydantic import JsonValue

from arc_agi_3.contracts.system1 import System1DecisionRequest, System1DecisionResult


class System1Backend(Protocol):
    @property
    def metadata(self) -> dict[str, JsonValue]: ...

    def decide(self, request: System1DecisionRequest) -> System1DecisionResult: ...
