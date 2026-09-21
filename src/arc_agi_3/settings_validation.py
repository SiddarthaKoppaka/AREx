"""Cross-field validation for resolved runtime settings."""

from .settings_fields import RuntimeFields


def validate_runtime[Settings: RuntimeFields](settings: Settings) -> Settings:
    if settings.max_new_tokens > settings.output_token_budget:
        raise ValueError("max_new_tokens exceeds output_token_budget")
    if settings.soft_input_limit >= settings.hard_input_limit:
        raise ValueError("soft_input_limit must be less than hard_input_limit")
    if settings.hard_input_limit >= settings.max_context_tokens:
        raise ValueError("hard_input_limit must be less than max_context_tokens")
    if (
        settings.hard_input_limit + settings.max_new_tokens
        > settings.max_context_tokens
    ):
        raise ValueError("hard_input_limit leaves insufficient generation room")
    return settings
