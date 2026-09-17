"""Exact visitation and marginal-return history without semantic labels."""

from pydantic import JsonValue

from .contracts import ExplorationHistorySnapshot


class ExplorationHistory:
    def __init__(self) -> None:
        self._visits: dict[str, int] = {}
        self._marginal_values: list[float] = []

    def visits(self, state_hash: str | None) -> int:
        return 0 if state_hash is None else self._visits.get(state_hash, 0)

    def record(self, state_hash: str | None, marginal_value: float) -> None:
        if state_hash is not None:
            self._visits[state_hash] = self.visits(state_hash) + 1
        self._marginal_values.append(marginal_value)

    @property
    def marginal_values(self) -> tuple[float, ...]:
        return tuple(self._marginal_values)

    def snapshot(self) -> dict[str, JsonValue]:
        return ExplorationHistorySnapshot(
            visits=self._visits,
            marginal_values=tuple(self._marginal_values),
        ).model_dump(mode="json")

    def restore(self, value: object) -> None:
        snapshot = ExplorationHistorySnapshot.model_validate(value)
        if any(count < 0 for count in snapshot.visits.values()):
            raise ValueError("exploration visit counts must be non-negative")
        self._visits = dict(snapshot.visits)
        self._marginal_values = list(snapshot.marginal_values)
