"""Token-aware pruning of optional context before structured generation."""

from typing import Any

from arc_agi_3.contracts.decision import AgentContext, PromptReport

from .context_limits import resolve_limits
from .prompts import decision_prompt
from .transformers_limits import InputTokenLimitError


def compact_with_report(
    context: AgentContext,
    backend: object,
    schema: dict[str, Any],
    *,
    prior_output: str | None = None,
    validation_error: str | None = None,
) -> tuple[AgentContext, PromptReport | None]:
    limits = resolve_limits(backend)
    if limits is None:
        return context, None

    def tokens(value: AgentContext) -> int:
        prompt = decision_prompt(
            value, prior_output=prior_output, validation_error=validation_error
        )
        return int(limits.counter(prompt, schema))

    def finish(value: AgentContext, after: int) -> tuple[AgentContext, PromptReport]:
        memory = value.episodic_memory
        report = PromptReport(
            tokens_before_compaction=before,
            tokens_after_compaction=after,
            compacted=value is not context,
            recent_events_after=len(value.recent_events),
            episodic_items_after=len(memory.items) if memory else 0,
        )
        reporter = getattr(backend, "reporter", None)
        if reporter is not None and report.compacted:
            reporter.stage(
                "context_compaction",
                before_tokens=before,
                after_tokens=after,
                compaction_pressure_start=limits.pressure,
                active_context_target=limits.target,
                emergency_ceiling=limits.hard,
            )
        return value, report

    before = tokens(context)
    if before <= limits.pressure and before <= limits.hard:
        return finish(context, before)
    steps = sorted({event.step_id for event in context.recent_events})[-2:]
    recent = tuple(event for event in context.recent_events if event.step_id in steps)
    recent = recent[-12:]
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
            if actual <= limits.target:
                return finish(compacted, actual)
    actual = tokens(compacted)
    if actual > limits.hard:
        model = str(getattr(getattr(backend, "config", None), "model_name", "unknown"))
        raise InputTokenLimitError(actual, limits.hard, model, f"turn={context.turn}")
    return finish(compacted, actual)


def compact_for_backend(
    context: AgentContext,
    backend: object,
    schema: dict[str, Any],
    *,
    prior_output: str | None = None,
    validation_error: str | None = None,
) -> AgentContext:
    return compact_with_report(
        context,
        backend,
        schema,
        prior_output=prior_output,
        validation_error=validation_error,
    )[0]
