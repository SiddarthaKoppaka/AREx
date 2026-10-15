"""Versioned merge logic for the compact working scratchpad."""

from pydantic import JsonValue, TypeAdapter

from arc_agi_3.contracts.scratchpad import ScratchpadUpdates, WorkingScratchpad

from .scratchpad_prepare import prepare_scratchpad


class ScratchpadStore:
    def __init__(
        self, capabilities: dict[str, bool] | None = None, token_budget: int = 2048
    ) -> None:
        self.current = WorkingScratchpad(capabilities=capabilities or {})
        self.token_budget = token_budget

    def apply(
        self, updates: ScratchpadUpdates, valid_evidence: set[str]
    ) -> WorkingScratchpad:
        updated = self.prepare(updates, valid_evidence)
        self.current = updated
        return updated

    def prepare(
        self, updates: ScratchpadUpdates, valid_evidence: set[str]
    ) -> WorkingScratchpad:
        updated, estimated_tokens = prepare_scratchpad(
            self.current, updates, valid_evidence
        )
        if estimated_tokens > self.token_budget:
            raise ValueError(
                f"working scratchpad exceeds {self.token_budget} token budget"
            )
        return updated

    def commit(self, prepared: WorkingScratchpad) -> None:
        self.current = prepared

    def snapshot(self) -> dict[str, JsonValue]:
        return self.current.model_dump(mode="json")

    def restore(self, value: object) -> None:
        if value:
            self.current = TypeAdapter(WorkingScratchpad).validate_python(value)
