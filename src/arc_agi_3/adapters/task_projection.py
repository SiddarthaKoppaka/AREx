"""Bounded task and handoff view for model prompt construction."""

from arc_agi_3.contracts.cognition import TaskView
from arc_agi_3.contracts.enums import TaskStatus


def task_prompt_view(tasks: tuple[TaskView, ...]) -> dict[str, object]:
    active = [view for view in tasks if view.task.status is not TaskStatus.COMPLETE]
    active.sort(key=lambda view: (-view.task.priority, view.task.task_id))
    completed = [view for view in tasks if view.task.status is TaskStatus.COMPLETE]
    completed.sort(key=lambda view: view.task.task_id)
    return {
        "active_tasks": [
            {
                "task_id": view.task.task_id,
                "version": view.task.version,
                "purpose": view.task.purpose,
                "success_criteria": view.task.success_criteria,
                "dependencies": view.task.dependencies,
                "status": view.task.status,
                "structurally_blocked": view.structurally_blocked,
            }
            for view in active[:8]
        ],
        "completed_handoffs": [
            {
                "task_id": view.task.task_id,
                "summary": view.task.handoff.summary if view.task.handoff else None,
                "artifact_refs": view.task.handoff.artifact_refs
                if view.task.handoff
                else (),
                "evidence_refs": view.task.handoff.evidence_refs
                if view.task.handoff
                else (),
            }
            for view in completed[-4:]
        ],
        "omitted_active_count": max(0, len(active) - 8),
        "omitted_completed_count": max(0, len(completed) - 4),
        "all_task_ids": [view.task.task_id for view in tasks],
    }
