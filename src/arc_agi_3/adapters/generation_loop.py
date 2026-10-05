"""Teacher/CognitiveDecision generation: the thin, context-aware caller.

The bounded primary-plus-repair engine itself lives in
`structured_generation.run_structured` and is shared with Students and
peer reviews (`classroom_adapter`). This module only supplies what is
specific to a CognitiveDecision: context compaction for the primary
prompt, and the tool-contract check as `validate_extra`.
"""

from arc_agi_3.contracts.decision import (
    AgentContext,
    CognitiveDecision,
    ModelResponse,
    PromptReport,
)

from .context_compaction import compact_with_report
from .decision_schema import decision_schema
from .inference import InferenceBackend
from .prompts import decision_prompt
from .structured_generation import run_structured
from .structured_output import StructuredOutputError as StructuredOutputError
from .tool_contract_check import check_tool_requests


def run_generation(
    backend: InferenceBackend,
    context: AgentContext,
    max_repairs: int,
    *,
    persist_invalid_output: bool,
    preview_chars: int,
) -> ModelResponse:
    schema = decision_schema()

    def primary() -> tuple[str, PromptReport | None]:
        projected, report = compact_with_report(context, backend, schema)
        return decision_prompt(projected), report

    decision, usage, attempts, report = run_structured(
        backend,
        schema,
        CognitiveDecision,
        primary,
        max_repairs,
        persist_invalid_output=persist_invalid_output,
        preview_chars=preview_chars,
        turn=context.turn,
        validate_extra=check_tool_requests,
    )
    return ModelResponse(
        decision=decision, usage=usage, attempts=attempts, prompt=report
    )
