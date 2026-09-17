"""Numerically stable probability helpers with no semantic policy."""

from arc_agi_3.contracts.cognition import Hypothesis


def update_odds(probability: float, multiplier: float) -> float:
    if probability <= 0.0:
        return 0.0
    if probability >= 1.0:
        return 1.0
    odds = probability / (1.0 - probability)
    updated = odds * multiplier
    return updated / (1.0 + updated)


def normalize_group(records: dict[str, Hypothesis], group: str) -> None:
    active = [
        record
        for record in records.values()
        if record.belief_group == group and record.status == "active"
    ]
    if not active:
        return
    total = sum(record.probability for record in active)
    if total <= 0:
        raise ValueError(f"active belief group {group!r} has zero total weight")
    for record in active:
        records[record.hypothesis_id] = record.model_copy(
            update={"probability": record.probability / total}
        )
