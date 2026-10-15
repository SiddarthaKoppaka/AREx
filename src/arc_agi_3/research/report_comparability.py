"""PR-11: never silently compare two runs that aren't actually matched.
handoff.md's protocol: "always evaluate against matched game versions,
seed, model weights, backend... and comparable hardware." This doesn't
block a mismatched comparison (sometimes that's exactly what's being
studied) - it just makes the mismatch visible rather than assumed.
"""

from arc_agi_3.contracts.baseline_report import BaselineReport


def check_comparable(
    control: BaselineReport, candidate: BaselineReport
) -> tuple[str, ...]:
    warnings = []
    if control.game_id != candidate.game_id:
        warnings.append(
            f"game_id differs: {control.game_id!r} vs {candidate.game_id!r}"
        )
    if control.seed != candidate.seed:
        warnings.append(f"seed differs: {control.seed} vs {candidate.seed}")
    if control.model != candidate.model:
        warnings.append("model metadata differs")
    if control.environment != candidate.environment:
        warnings.append("environment metadata differs")
    if control.ablations != candidate.ablations:
        warnings.append("ablations differ")
    return tuple(warnings)
