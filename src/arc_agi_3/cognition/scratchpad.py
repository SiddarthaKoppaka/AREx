"""Versioned merge logic for the compact working scratchpad."""

from pydantic import JsonValue, TypeAdapter

from arc_agi_3.contracts.scratchpad import ScratchpadUpdates, WorkingScratchpad
from arc_agi_3.trace.canonical import canonical_json


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
        before = self.current
        facts = {item.fact_id: item for item in before.verified_facts}
        for fact in updates.add_verified_fact:
            if not set(fact.evidence_refs) <= valid_evidence:
                raise ValueError(
                    "verified scratchpad facts require known evidence refs"
                )
            if fact.fact_id in fact.supersedes:
                raise ValueError("a scratchpad fact cannot supersede itself")
            if len(set(fact.supersedes)) != len(fact.supersedes):
                raise ValueError("superseded scratchpad facts must be unique")
            superseded = [facts[key] for key in fact.supersedes if key in facts]
            if len(superseded) != len(fact.supersedes):
                raise ValueError("superseded scratchpad facts must exist")
            inherited = {
                reference for item in superseded for reference in item.evidence_refs
            }
            if not inherited <= set(fact.evidence_refs):
                raise ValueError("consolidated facts must retain source evidence")
            prior = facts.get(fact.fact_id)
            expected = 1 if prior is None else prior.version + 1
            if fact.version != expected:
                raise ValueError(f"scratchpad fact requires version {expected}")
            for item in superseded:
                del facts[item.fact_id]
            facts[fact.fact_id] = fact
        questions = [
            item
            for item in before.open_questions
            if item not in updates.resolve_open_questions
        ]
        questions.extend(
            item for item in updates.add_open_questions if item not in questions
        )
        values = {
            "version": before.version + 1,
            "objective": updates.set_objective or before.objective,
            "verified_facts": tuple(facts[key] for key in sorted(facts)),
            "action_model": {**before.action_model, **updates.action_model_updates},
            "active_plan": updates.revise_plan
            if updates.revise_plan is not None
            else before.active_plan,
            "open_questions": tuple(questions),
            "last_useful_result": updates.set_last_useful_result
            or before.last_useful_result,
            "next_test": updates.set_next_test or before.next_test,
            "capabilities": before.capabilities,
        }
        updated = before.model_copy(update=values)
        estimated_tokens = (len(canonical_json(updated)) + 3) // 4
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
