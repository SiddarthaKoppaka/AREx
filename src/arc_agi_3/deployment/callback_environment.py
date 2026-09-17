"""Environment capability descriptor for externally-owned episode loops."""

from pydantic import JsonValue

from arc_agi_3.contracts.observation import Action, Observation


class CallbackEnvironment:
    def __init__(self, game_id: str, framework_version: str = "unknown") -> None:
        self.game_id, self.framework_version = game_id, framework_version

    @property
    def metadata(self) -> dict[str, JsonValue]:
        return {
            "adapter": "kaggle-callback",
            "game_id": self.game_id,
            "framework_version": self.framework_version,
            "owns_environment_loop": False,
            "restore_supported": False,
        }

    def reset(self) -> Observation:
        raise RuntimeError("the official framework owns reset")

    def step(self, action: Action) -> Observation:
        raise RuntimeError("the official framework owns step")

    def checkpoint(self) -> dict[str, JsonValue] | None:
        return None

    def restore(self, state: dict[str, JsonValue]) -> Observation:
        raise NotImplementedError("competition environments cannot be restored")
