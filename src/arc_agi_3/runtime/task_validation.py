"""Validate provenance of LM-authored task handoffs before mutation."""

from arc_agi_3.contracts.cognition import TaskRecord
from arc_agi_3.contracts.enums import TaskStatus

from .io import EpisodeIO


def validate_task_updates(
    io: EpisodeIO, updates: tuple[TaskRecord, ...], known_events: set[str]
) -> None:
    for task in updates:
        refs = set(task.evidence_refs)
        if task.handoff is not None:
            refs.update(task.handoff.evidence_refs)
            for reference in task.handoff.artifact_refs:
                io.events.artifacts.read(reference)
        if not refs <= known_events:
            raise ValueError("task evidence must reference prior events")
        if task.status is TaskStatus.COMPLETE and task.handoff is None:
            raise ValueError("completed tasks require a compact handoff")
