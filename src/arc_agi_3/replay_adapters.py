"""Offline adapters backed entirely by an immutable recorded trace."""

from pydantic import JsonValue

from arc_agi_3.contracts.decision import ModelResponse
from arc_agi_3.contracts.enums import EventType
from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.contracts.observation import Action, Observation


class RecordedEnvironment:
    def __init__(self, events: list[EventEnvelope]) -> None:
        observations = [
            event for event in events if event.event_type is EventType.OBSERVATION
        ]
        if not observations:
            raise ValueError("trace has no observations")
        self._initial = Observation.model_validate(observations[0].payload)
        by_action = {
            event.causal_refs[0]: Observation.model_validate(event.payload)
            for event in observations
            if event.causal_refs
        }
        self._steps = []
        for event in events:
            if event.event_type is not EventType.ACTION:
                continue
            after = by_action.get(event.event_id)
            if after is None:
                raise ValueError("recorded action lacks its causal observation")
            payload = event.payload.get("action")
            self._steps.append((Action.model_validate(payload), after))
        self._states = [self._initial, *(after for _, after in self._steps)]
        self._cursor = self._state_index = 0

    @property
    def metadata(self) -> dict[str, JsonValue]:
        return {"adapter": "recorded-environment", "steps": len(self._steps)}

    def reset(self) -> Observation:
        self._cursor = self._state_index = 0
        return self._initial

    def step(self, action: Action) -> Observation:
        if self._cursor >= len(self._steps):
            raise RuntimeError("recorded environment has no step remaining")
        expected, after = self._steps[self._cursor]
        if action != expected:
            raise ValueError("replay action diverged from recorded action")
        self._cursor += 1
        self._state_index = self._cursor
        return after

    def checkpoint(self) -> dict[str, JsonValue]:
        return {"state_index": self._state_index}

    def restore(self, state: dict[str, JsonValue]) -> Observation:
        index = state.get("state_index")
        if not isinstance(index, int) or not 0 <= index < len(self._states):
            raise ValueError("invalid recorded checkpoint")
        self._state_index = index
        return self._states[index]


class RecordedModel:
    def __init__(self, events: list[EventEnvelope]) -> None:
        self._responses = []
        for event in events:
            if event.event_type is not EventType.MODEL_DECISION:
                continue
            self._responses.append(
                ModelResponse.model_validate(
                    {
                        "decision": event.payload.get("decision"),
                        "usage": event.payload.get("usage", {}),
                        "attempts": event.payload.get("attempts", []),
                    }
                )
            )

    @property
    def metadata(self) -> dict[str, JsonValue]:
        return {"adapter": "recorded-model", "responses": len(self._responses)}

    def decide(self, context: object) -> ModelResponse:
        if not self._responses:
            raise RuntimeError("recorded model has no response remaining")
        return self._responses.pop(0)
