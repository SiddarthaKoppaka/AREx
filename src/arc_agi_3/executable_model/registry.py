"""Immutable candidate versions with parent lineage, coexisting per model
and across competing model_ids. Mirrors `world_model/store.py`'s shape."""

from .contracts import ExecutableModelCandidate


class ExecutableModelRegistry:
    def __init__(self) -> None:
        self._history: dict[str, list[ExecutableModelCandidate]] = {}

    def add(self, candidate: ExecutableModelCandidate) -> None:
        history = self._history.setdefault(candidate.model_id, [])
        if any(item.version == candidate.version for item in history):
            raise ValueError(
                f"version {candidate.version} already exists for {candidate.model_id!r}"
            )
        expected_parent = history[-1].version if history else None
        if history and candidate.parent_version != expected_parent:
            raise ValueError(f"candidate must declare parent_version={expected_parent}")
        history.append(candidate)

    def model_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._history))

    def versions(self, model_id: str) -> tuple[ExecutableModelCandidate, ...]:
        return tuple(self._history.get(model_id, ()))

    def get(self, model_id: str, version: int) -> ExecutableModelCandidate:
        for item in self._history.get(model_id, ()):
            if item.version == version:
                return item
        raise KeyError(f"unknown candidate {model_id!r} version {version}")

    def latest(self, model_id: str) -> ExecutableModelCandidate:
        history = self._history.get(model_id)
        if not history:
            raise KeyError(f"unknown model_id {model_id!r}")
        return history[-1]
