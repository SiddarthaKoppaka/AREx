"""Callback-driven cognition independent of environment-loop ownership."""

from arc_agi_3.adapters.protocols import ModelAdapter
from arc_agi_3.contracts.enums import EnvironmentState
from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.observation import Action, Observation
from arc_agi_3.manifest import RunManifest

from .action_phases import record_action_result
from .action_types import AuthorizedAction
from .io import EpisodeIO
from .session_chunk import accept_chunk_result, next_chunk_action
from .session_decide import decide_until_action
from .session_lifecycle import finish_run, initialize_run, record_failure
from .session_types import ActiveChunk
from .state import RunResult


class CognitiveSession:
    def __init__(
        self, io: EpisodeIO, model: ModelAdapter, manifest: RunManifest
    ) -> None:
        self.io, self.model, self.manifest = io, model, manifest
        self.observation: Observation | None = None
        self.observed: EventEnvelope | None = None
        self.pending: AuthorizedAction | None = None
        self.chunk: ActiveChunk | None = None
        self.turn = 0
        self.result: RunResult | None = None

    def start(self, observation: Observation) -> None:
        if self.observation is not None:
            raise RuntimeError("session already started")
        self.observation = observation
        self.observed = initialize_run(self.io, self.manifest, observation)

    def observe(self, observation: Observation) -> None:
        if self.pending is None:
            raise RuntimeError("session has no action awaiting an observation")
        outcome = record_action_result(self.io, self.pending, observation)
        self.observation = observation
        self.observed = outcome.observed_event
        self.pending = None
        if self.chunk and not accept_chunk_result(
            self.io, self.turn, self.chunk, outcome
        ):
            self.chunk = None

    def next_action(self) -> Action | None:
        if self.pending is not None:
            raise RuntimeError("previous action result has not been observed")
        if self.result is not None:
            return None
        observation = self._current()
        assert self.observed is not None
        if observation.state is EnvironmentState.WIN:
            self.turn += 1
            self.finish("terminal")
            return None
        if self.chunk is not None:
            return self._authorize_chunk(observation)
        outcome = decide_until_action(
            self.io, self.model, observation, self.observed, self.turn
        )
        self.turn, self.observation, self.observed = (
            outcome.turn,
            outcome.observation,
            outcome.observed,
        )
        self.pending, self.chunk = outcome.pending, outcome.chunk
        if outcome.stop_reason:
            self.finish(outcome.stop_reason)
            return None
        assert self.pending is not None
        return self.pending.action

    def _authorize_chunk(self, observation: Observation) -> Action | None:
        assert self.chunk is not None
        self.pending = next_chunk_action(self.io, self.turn, observation, self.chunk)
        if self.pending is None:
            self.chunk = None
            return self.next_action()
        return self.pending.action

    def _current(self) -> Observation:
        if self.observation is None or self.observed is None:
            raise RuntimeError("session has not started")
        return self.observation

    def fail(self, error: Exception) -> RunResult:
        record_failure(self.io, self.turn, error)
        return self.finish("failure")

    def finish(self, reason: str) -> RunResult:
        if self.result is None:
            self.result = finish_run(self.io, self.turn, reason)
        return self.result
