"""Cross-field validation for resolved runtime settings."""

from .settings_fields import RuntimeFields


def validate_runtime[Settings: RuntimeFields](settings: Settings) -> Settings:
    if settings.max_new_tokens > settings.output_token_budget:
        raise ValueError("max_new_tokens exceeds output_token_budget")
    emergency = (
        settings.max_context_tokens
        - settings.max_new_tokens
        - settings.template_and_generation_margin
    )
    if emergency <= 0:
        raise ValueError("context window leaves insufficient generation room")
    if settings.compaction_pressure_start >= settings.active_context_target:
        raise ValueError("soft_input_limit must precede active context target")
    if settings.active_context_target > emergency:
        raise ValueError("active context target exceeds emergency ceiling")
    if settings.soft_input_limit != settings.compaction_pressure_start:
        raise ValueError("soft_input_limit must match compaction_pressure_start")
    if settings.hard_input_limit != emergency or settings.max_input_tokens != emergency:
        raise ValueError("input limits must match computed emergency ceiling")
    return settings
