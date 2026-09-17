"""Versioned hypothesis storage and explicitly requested belief arithmetic."""

from pydantic import JsonValue

from arc_agi_3.config import BeliefConfig
from arc_agi_3.contracts.cognition import BeliefUpdate, Hypothesis

from .belief_math import normalize_group
from .belief_operations import stage_update


class BeliefStore:
    def __init__(self, config: BeliefConfig) -> None:
        self.config = config
        self._history: dict[str, list[Hypothesis]] = {}

    @property
    def current(self) -> tuple[Hypothesis, ...]:
        return tuple(self._history[key][-1] for key in sorted(self._history))

    def history(self, hypothesis_id: str) -> tuple[Hypothesis, ...]:
        return tuple(self._history.get(hypothesis_id, ()))

    def restore(self, records: tuple[Hypothesis, ...]) -> None:
        ids = [item.hypothesis_id for item in records]
        if len(ids) != len(set(ids)):
            raise ValueError("restored hypothesis IDs must be unique")
        self._history = {item.hypothesis_id: [item] for item in records}

    def add_many(self, proposals: tuple[Hypothesis, ...]) -> tuple[Hypothesis, ...]:
        if not proposals:
            return ()
        ids = [item.hypothesis_id for item in proposals]
        if len(ids) != len(set(ids)) or any(key in self._history for key in ids):
            raise ValueError("hypothesis proposal IDs must be new and unique")
        if any(item.version != 1 for item in proposals):
            raise ValueError("new hypotheses must start at version 1")
        before = {item.hypothesis_id: item for item in self.current}
        staged = {**before, **{item.hypothesis_id: item for item in proposals}}
        groups = {
            group for item in proposals if (group := item.belief_group) is not None
        }
        for group in sorted(groups):
            normalize_group(staged, group)
        changed = self._commit(staged, before, set(ids))
        return changed

    def apply(self, request: BeliefUpdate) -> tuple[Hypothesis, ...]:
        before = {item.hypothesis_id: item for item in self.current}
        staged, touched = stage_update(self.config, before, request)
        groups = {
            group for key in touched if (group := staged[key].belief_group) is not None
        }
        for group in sorted(groups):
            normalize_group(staged, group)
        return self._commit(staged, before, touched)

    def _commit(
        self,
        staged: dict[str, Hypothesis],
        before: dict[str, Hypothesis],
        touched: set[str],
    ) -> tuple[Hypothesis, ...]:
        changed: list[Hypothesis] = []
        for key in sorted(staged):
            prior, record = before.get(key), staged[key]
            if prior == record and key not in touched:
                continue
            version = 1 if prior is None else prior.version + 1
            committed = record.model_copy(update={"version": version})
            self._history.setdefault(key, []).append(committed)
            changed.append(committed)
        return tuple(changed)

    def snapshot(self) -> list[JsonValue]:
        return [item.model_dump(mode="json") for item in self.current]
