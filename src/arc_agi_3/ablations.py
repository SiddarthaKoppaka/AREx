"""Named, typed research ablations; no capability is disabled implicitly."""

from enum import StrEnum

from arc_agi_3.config import AblationConfig, RunConfig


class AblationPreset(StrEnum):
    FULL = "full"
    THIN = "thin"
    MEMORY_ONLY = "memory_only"


PRESETS = {
    AblationPreset.FULL: AblationConfig(),
    AblationPreset.THIN: AblationConfig(
        persistent_memory=False,
        hypotheses=False,
        tasks=False,
        search=False,
        world_models=False,
        prediction_verification=False,
        selective_retrieval=False,
        multi_candidate_inference=False,
    ),
    AblationPreset.MEMORY_ONLY: AblationConfig(
        search=False,
        world_models=False,
        prediction_verification=False,
        multi_candidate_inference=False,
    ),
}


def apply_preset(config: RunConfig, preset: AblationPreset) -> RunConfig:
    return config.model_copy(update={"ablations": PRESETS[preset]})


def require_capability(enabled: bool, capability: str) -> None:
    if not enabled:
        raise ValueError(f"capability disabled by ablation: {capability}")
