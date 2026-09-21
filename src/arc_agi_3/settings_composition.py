"""Translate one resolved profile into existing runtime contracts."""

from pathlib import Path

from arc_agi_3.adapters.transformers import TransformersConfig
from arc_agi_3.config import BudgetConfig, RunConfig
from arc_agi_3.contracts.enums import Resource

from .settings_models import RuntimeSettings


def transformers_config(
    settings: RuntimeSettings, model_path: str | Path | None = None
) -> TransformersConfig:
    return TransformersConfig(
        model_path=str(model_path or settings.model_path),
        model_name=settings.model_name,
        model_digest=settings.model_digest,
        revision=settings.revision,
        dtype=settings.dtype,
        device_map=settings.device_map,
        attention_implementation=settings.attention_implementation,
        local_files_only=settings.local_files_only,
        trust_remote_code=settings.trust_remote_code,
        quantization=settings.quantization,
        max_new_tokens=settings.max_new_tokens,
        max_input_tokens=settings.hard_input_limit,
        max_context_tokens=settings.max_context_tokens,
        soft_input_limit=settings.soft_input_limit,
        max_time_seconds=settings.generation_timeout_seconds,
    )


def budget_config(settings: RuntimeSettings) -> BudgetConfig:
    limits = {
        Resource.ACTIONS: settings.max_actions,
        Resource.MODEL_CALLS: settings.max_model_calls,
        Resource.INPUT_TOKENS: settings.input_token_budget,
        Resource.OUTPUT_TOKENS: settings.output_token_budget,
    }
    if settings.wall_time_budget_seconds is not None:
        limits[Resource.WALL_TIME_MS] = int(settings.wall_time_budget_seconds * 1000)
    return BudgetConfig(limits=limits)


def run_config(
    settings: RuntimeSettings,
    run_id: str,
    game_id: str,
    seed: int,
    experiment_id: str | None,
) -> RunConfig:
    return RunConfig(
        run_id=run_id,
        experiment_id=experiment_id or settings.profile,
        game_id=game_id,
        seed=seed,
        max_turns=settings.max_turns,
        output_dir=settings.output_dir,
        checkpoint_every=settings.checkpoint_interval,
        recent_event_limit=settings.recent_event_limit,
        scratchpad_token_budget=settings.scratchpad_token_budget,
        raw_recent_turns=settings.raw_recent_turns,
        episodic_retrieval_limit=settings.episodic_retrieval_limit,
        context_compaction=settings.compact_context,
        budget=budget_config(settings),
    )
