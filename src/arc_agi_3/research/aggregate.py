"""Deterministic grouped statistics with seeded bootstrap intervals."""

import random
from collections import defaultdict

from pydantic import Field

from arc_agi_3.contracts.base import Contract


class AggregateResult(Contract):
    group: dict[str, str]
    metric: str
    count: int = Field(ge=1)
    mean: float
    ci_low: float
    ci_high: float
    seed: int
    bootstrap_samples: int = Field(gt=0)


def bootstrap_interval(
    values: list[float], seed: int, samples: int = 2_000
) -> tuple[float, float]:
    if not values or samples <= 0:
        raise ValueError("values and positive samples are required")
    randomizer = random.Random(seed)
    means = sorted(
        sum(randomizer.choice(values) for _ in values) / len(values)
        for _ in range(samples)
    )
    return means[int(0.025 * (samples - 1))], means[int(0.975 * (samples - 1))]


def aggregate_rows(
    rows: list[dict[str, object]],
    group_by: tuple[str, ...],
    metric: str,
    *,
    seed: int = 0,
    bootstrap_samples: int = 2_000,
) -> list[AggregateResult]:
    groups: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if not isinstance(value, int | float):
            raise ValueError(f"metric {metric!r} must be numeric")
        groups[tuple(str(row.get(key, "")) for key in group_by)].append(float(value))
    results = []
    for index, (key, values) in enumerate(sorted(groups.items())):
        low, high = bootstrap_interval(values, seed + index, bootstrap_samples)
        results.append(
            AggregateResult(
                group=dict(zip(group_by, key, strict=True)),
                metric=metric,
                count=len(values),
                mean=sum(values) / len(values),
                ci_low=low,
                ci_high=high,
                seed=seed + index,
                bootstrap_samples=bootstrap_samples,
            )
        )
    return results
