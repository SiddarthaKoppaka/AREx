"""Versioned harness-change candidates with parent lineage, mirroring
world_model/store.py's shape. Promotion is recorded here, never decided
here - see promotion.py and PR-10-report.md's "no automatic production
promotion" note."""

from .contracts import HarnessChangeCandidate


class HarnessChangeRegistry:
    def __init__(self) -> None:
        self._history: dict[str, list[HarnessChangeCandidate]] = {}

    def add(self, candidate: HarnessChangeCandidate) -> None:
        history = self._history.setdefault(candidate.candidate_id, [])
        if any(item.version == candidate.version for item in history):
            raise ValueError(
                f"version {candidate.version} already exists for "
                f"{candidate.candidate_id!r}"
            )
        expected = 1 if not history else history[-1].version + 1
        if candidate.version != expected:
            raise ValueError(
                f"candidate {candidate.candidate_id!r} requires version {expected}"
            )
        history.append(candidate)

    def get(self, candidate_id: str, version: int) -> HarnessChangeCandidate:
        for item in self._history.get(candidate_id, ()):
            if item.version == version:
                return item
        raise KeyError(f"unknown candidate {candidate_id!r} version {version}")

    def record_decision(self, candidate_id: str, version: int, accepted: bool) -> None:
        candidate = self.get(candidate_id, version)
        history = self._history[candidate_id]
        index = history.index(candidate)
        history[index] = candidate.model_copy(
            update={"promotion_status": "accepted" if accepted else "rejected"}
        )
