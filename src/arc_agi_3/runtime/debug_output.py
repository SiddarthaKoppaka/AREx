"""Keep optional invalid-output previews outside the canonical trace."""

from pathlib import Path

from pydantic import JsonValue

from arc_agi_3.trace.canonical import canonical_json


def persist_debug_previews(
    details: dict[str, JsonValue], run_dir: Path
) -> dict[str, JsonValue]:
    attempts = details.get("attempts")
    if not isinstance(attempts, list):
        return details
    cleaned: list[JsonValue] = []
    previews: list[dict[str, JsonValue]] = []
    for item in attempts:
        if not isinstance(item, dict):
            cleaned.append(item)
            continue
        without_preview = dict(item)
        preview = without_preview.pop("output_preview", None)
        if isinstance(preview, str):
            previews.append(
                {"attempt": item.get("attempt"), "output_preview": preview[:512]}
            )
        cleaned.append(without_preview)
    if previews:
        (run_dir / "invalid_model_outputs.json").write_text(
            canonical_json(previews) + "\n"
        )
    return {**details, "attempts": cleaned}
