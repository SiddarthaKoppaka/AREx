"""In-memory projection rebuildable from immutable cognitive events."""

from pydantic import JsonValue, TypeAdapter

from arc_agi_3.config import BeliefConfig
from arc_agi_3.contracts.cognition import Hypothesis, TaskRecord
from arc_agi_3.exploration import ExplorationHistory
from arc_agi_3.world_model import WorldModel, WorldModelStore

from .beliefs import BeliefStore
from .scratchpad import ScratchpadStore
from .tasks import TaskGraph


class CognitiveWorkspace:
    def __init__(
        self,
        belief_config: BeliefConfig,
        capabilities: dict[str, bool] | None = None,
        scratchpad_token_budget: int = 2048,
    ) -> None:
        self.beliefs = BeliefStore(belief_config)
        self.tasks = TaskGraph()
        self.world_models = WorldModelStore()
        self.exploration = ExplorationHistory()
        self.scratchpad = ScratchpadStore(capabilities, scratchpad_token_budget)

    def snapshot(self) -> dict[str, JsonValue]:
        return {
            "hypotheses": self.beliefs.snapshot(),
            "tasks": self.tasks.snapshot(),
            "world_models": self.world_models.snapshot(),
            "exploration": self.exploration.snapshot(),
            "working_scratchpad": self.scratchpad.snapshot(),
        }

    def restore(self, value: dict[str, JsonValue]) -> None:
        hypotheses = TypeAdapter(tuple[Hypothesis, ...]).validate_python(
            value.get("hypotheses", [])
        )
        tasks = TypeAdapter(tuple[TaskRecord, ...]).validate_python(
            value.get("tasks", [])
        )
        models = TypeAdapter(tuple[WorldModel, ...]).validate_python(
            value.get("world_models", [])
        )
        self.beliefs.restore(hypotheses)
        self.tasks.restore(tasks)
        self.world_models.restore(models)
        self.exploration.restore(value.get("exploration", {}))
        self.scratchpad.restore(value.get("working_scratchpad"))
