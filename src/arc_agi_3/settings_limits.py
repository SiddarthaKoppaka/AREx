"""Normalize legacy input limits to context pressure and model headroom."""

from collections.abc import Mapping


def normalize_context_limits(
    values: dict[str, object],
    source: Mapping[str, str],
    overrides: Mapping[str, object],
) -> None:
    def supplied(name: str) -> bool:
        return name in overrides or f"AREX_{name.upper()}" in source

    pressure = supplied("compaction_pressure_start")
    legacy_soft = supplied("soft_input_limit")
    if (
        pressure
        and legacy_soft
        and int(str(values["compaction_pressure_start"]))
        != int(str(values["soft_input_limit"]))
    ):
        raise ValueError("conflicting pressure and legacy soft input limits")
    if legacy_soft and not pressure:
        values["compaction_pressure_start"] = values["soft_input_limit"]
    else:
        values["soft_input_limit"] = values["compaction_pressure_start"]

    legacy_hard = supplied("hard_input_limit") or supplied("max_input_tokens")
    margin = supplied("template_and_generation_margin")
    context = int(str(values["max_context_tokens"]))
    output = int(str(values["max_new_tokens"]))
    if legacy_hard:
        hard = int(str(values["hard_input_limit"]))
        inferred_margin = context - output - hard
        if hard >= context:
            raise ValueError("hard_input_limit must be less than max_context_tokens")
        if inferred_margin < 0:
            raise ValueError("hard_input_limit leaves insufficient generation room")
        if margin and inferred_margin != int(
            str(values["template_and_generation_margin"])
        ):
            raise ValueError("conflicting emergency ceiling and legacy input limit")
        values["template_and_generation_margin"] = inferred_margin
    emergency = context - output - int(str(values["template_and_generation_margin"]))
    values["hard_input_limit"] = emergency
    values["max_input_tokens"] = emergency
