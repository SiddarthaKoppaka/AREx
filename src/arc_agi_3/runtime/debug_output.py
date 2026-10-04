"""Keep optional invalid-output previews outside the canonical trace."""

from pathlib import Path

from pydantic import JsonValue

from arc_agi_3.trace.canonical import canonical_json

_PREVIEW_KEYS = ("output_preview", "output_preview_tail")


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
        found = {
            key: text[:512]
            for key in _PREVIEW_KEYS
            if isinstance((text := without_preview.pop(key, None)), str)
        }
        if found:
            previews.append({"attempt": item.get("attempt"), **found})
        cleaned.append(without_preview)
    if previews:
        (run_dir / "invalid_model_outputs.json").write_text(
            canonical_json(previews) + "\n"
        )
    return {**details, "attempts": cleaned}
