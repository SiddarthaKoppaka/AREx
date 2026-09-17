"""Immutable history for LM-authored world-model versions."""

from pydantic import JsonValue

from .contracts import WorldModel


class WorldModelStore:
    def __init__(self) -> None:
        self._history: dict[str, list[WorldModel]] = {}

    @property
    def current(self) -> tuple[WorldModel, ...]:
        return tuple(self._history[key][-1] for key in sorted(self._history))

    def add_many(self, models: tuple[WorldModel, ...]) -> tuple[WorldModel, ...]:
        seen: set[str] = set()
        for model in models:
            if model.model_id in seen:
                raise ValueError("a world model may be updated only once per decision")
            seen.add(model.model_id)
            history = self._history.get(model.model_id, [])
            expected = 1 if not history else history[-1].version + 1
            if model.version != expected:
                raise ValueError(f"world model requires version {expected}")
        for model in models:
            self._history.setdefault(model.model_id, []).append(model)
        return tuple(sorted(models, key=lambda item: item.model_id))

    def get(self, model_id: str, version: int) -> WorldModel:
        for model in self._history.get(model_id, ()):
            if model.version == version:
                return model
        raise KeyError(f"unknown world model {model_id!r} version {version}")

    def history(self, model_id: str) -> tuple[WorldModel, ...]:
        return tuple(self._history.get(model_id, ()))

    def restore(self, models: tuple[WorldModel, ...]) -> None:
        ids = [item.model_id for item in models]
        if len(ids) != len(set(ids)):
            raise ValueError("restored world-model IDs must be unique")
        self._history = {item.model_id: [item] for item in models}

    def snapshot(self) -> list[JsonValue]:
        return [item.model_dump(mode="json") for item in self.current]
