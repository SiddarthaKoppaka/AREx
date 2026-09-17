"""Minimal loop that never substitutes a policy for the LM agent."""

from pydantic import JsonValue

from arc_agi_3.adapters.protocols import EnvironmentAdapter, ModelAdapter
from arc_agi_3.budgets import BudgetLedger
from arc_agi_3.config import RunConfig
from arc_agi_3.contracts.enums import EnvironmentState, EventType
from arc_agi_3.evaluation import evaluate
from arc_agi_3.manifest import RunManifest
from arc_agi_3.trace.checkpoints import CheckpointStore
from arc_agi_3.trace.store import JsonlEventStore

from .io import EpisodeIO
from .state import RunResult
from .turn import execute_turn


class EpisodeRunner:
    def __init__(
        self,
        *,
        config: RunConfig,
        environment: EnvironmentAdapter,
        model: ModelAdapter,
        events: JsonlEventStore,
        checkpoints: CheckpointStore,
        ledger: BudgetLedger,
        manifest: RunManifest,
    ) -> None:
        self.config, self.environment, self.model = config, environment, model
        self.events, self.manifest = events, manifest
        self.io = EpisodeIO(config, environment, events, checkpoints, ledger)

    def _episode(self, start_id: str) -> tuple[str, int]:
        observation = self.environment.reset()
        observed = self.io.append(
            EventType.OBSERVATION,
            "environment",
            0,
            observation.model_dump(mode="json"),
            (start_id,),
        )
        for step in range(1, self.config.max_turns + 1):
            if observation.state in {
                EnvironmentState.WIN,
                EnvironmentState.GAME_OVER,
            }:
                return "terminal", step
            outcome = execute_turn(self.io, self.model, step, observation, observed)
            observation, observed = outcome.observation, outcome.observed_event
            if outcome.stop_reason:
                return outcome.stop_reason, step
        return "max_turns", self.config.max_turns

    def run(self) -> RunResult:
        start = self.io.append(
            EventType.RUN_STARTED,
            "runtime",
            0,
            {"manifest": self.manifest.model_dump(mode="json")},
        )
        try:
            reason, step = self._episode(start.event_id)
        except Exception as error:
            reason, step = "failure", 0
            payload: dict[str, JsonValue] = {
                "error_type": type(error).__name__,
                "message": str(error),
            }
            details = getattr(error, "trace_payload", None)
            if isinstance(details, dict):
                payload["details"] = details
            self.io.append(
                EventType.FAILURE,
                "runtime",
                step,
                payload,
            )
        self.io.append(EventType.RUN_FINISHED, "runtime", step, {"reason": reason})
        metrics = evaluate(self.events.read(), self.config.evaluator)
        self.io.append(
            EventType.EVALUATION,
            "evaluator",
            step,
            metrics.model_dump(mode="json"),
        )
        return RunResult(
            run_id=self.config.run_id,
            stop_reason=reason,
            trace_path=str(self.events.path),
            metrics=metrics,
        )
