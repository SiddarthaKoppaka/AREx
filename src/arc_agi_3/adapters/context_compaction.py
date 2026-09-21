"""Token-aware pruning of optional context before structured generation."""

from typing import Any, Protocol

from arc_agi_3.contracts.decision import AgentContext

from .prompts import decision_prompt
from .transformers_limits import InputTokenLimitError


class TokenAwareBackend(Protocol):
    config: Any

    def input_token_count(self, prompt: str, schema: dict[str, Any]) -> int: ...


def compact_for_backend(
    context: AgentContext,
    backend: object,
    schema: dict[str, Any],
    *,
    prior_output: str | None = None,
    validation_error: str | None = None,
) -> AgentContext:
    counter = getattr(backend, "input_token_count", None)
    config = getattr(backend, "config", None)
    configured_soft = getattr(config, "soft_input_limit", None)
    hard = getattr(config, "max_input_tokens", None)
    if (
        not callable(counter)
        or not isinstance(configured_soft, int)
        or not isinstance(hard, int)
    ):
        return context
    soft = min(configured_soft, hard)

    def tokens(value: AgentContext) -> int:
        prompt = decision_prompt(
            value, prior_output=prior_output, validation_error=validation_error
        )
        return int(counter(prompt, schema))

    actual = tokens(context)
    if actual <= soft and actual <= hard:
        return context
    before = actual

    def report(after: int) -> None:
        reporter = getattr(backend, "reporter", None)
        if reporter is not None:
            reporter.stage(
                "context_compaction",
                before_tokens=before,
                after_tokens=after,
                soft_limit=soft,
                hard_limit=hard,
            )

    steps = sorted({event.step_id for event in context.recent_events})[-2:]
    recent = tuple(event for event in context.recent_events if event.step_id in steps)[
        -12:
    ]
    compacted = context.model_copy(
        update={
            "recent_events": recent,
            "recent_event_refs": tuple(event.event_id for event in recent),
        }
    )
    memory = compacted.episodic_memory
    if memory is not None:
        for remaining in range(len(memory.items), -1, -1):
            items = memory.items[-remaining:] if remaining else ()
            summarized = sum(len(item.source_event_refs) for item in items)
            candidate = memory.model_copy(
                update={
                    "items": items,
                    "summarized_event_count": summarized,
                    "omitted_event_count": memory.source_event_count - summarized,
                }
            )
            compacted = compacted.model_copy(update={"episodic_memory": candidate})
            actual = tokens(compacted)
            if actual <= soft:
                report(actual)
                return compacted
    actual = tokens(compacted)
    if actual > hard:
        model = str(getattr(config, "model_name", "unknown"))
        raise InputTokenLimitError(actual, hard, model, f"turn={context.turn}")
    report(actual)
    return compacted
