"""LM-authored task records in a mechanically validated versioned DAG."""

from pydantic import JsonValue

from arc_agi_3.contracts.cognition import TaskRecord, TaskView
from arc_agi_3.contracts.enums import TaskStatus


class TaskGraph:
    def __init__(self) -> None:
        self._history: dict[str, list[TaskRecord]] = {}

    @property
    def current(self) -> tuple[TaskRecord, ...]:
        return tuple(self._history[key][-1] for key in sorted(self._history))

    def history(self, task_id: str) -> tuple[TaskRecord, ...]:
        return tuple(self._history.get(task_id, ()))

    def restore(self, records: tuple[TaskRecord, ...]) -> None:
        tasks = {item.task_id: item for item in records}
        if len(tasks) != len(records):
            raise ValueError("restored task IDs must be unique")
        self._validate(tasks)
        self._history = {item.task_id: [item] for item in records}

    def apply_many(self, updates: tuple[TaskRecord, ...]) -> tuple[TaskRecord, ...]:
        prepared = self.validate_many(updates)
        for update in prepared:
            self._history.setdefault(update.task_id, []).append(update)
        return prepared

    def validate_many(self, updates: tuple[TaskRecord, ...]) -> tuple[TaskRecord, ...]:
        if not updates:
            return ()
        ids = [item.task_id for item in updates]
        if len(ids) != len(set(ids)):
            raise ValueError("a task may be updated only once per decision")
        staged = {item.task_id: item for item in self.current}
        for update in updates:
            prior = staged.get(update.task_id)
            expected = 1 if prior is None else prior.version + 1
            if update.version != expected:
                raise ValueError(f"task {update.task_id!r} requires version {expected}")
            staged[update.task_id] = update
        if len(staged) > 32:
            raise ValueError("task graph exceeds 32 tasks")
        self._validate(staged)
        return tuple(sorted(updates, key=lambda item: item.task_id))

    @staticmethod
    def _validate(tasks: dict[str, TaskRecord]) -> None:
        for task in tasks.values():
            missing = set(task.dependencies).difference(tasks)
            if missing:
                raise ValueError(f"task {task.task_id!r} has missing dependencies")
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(task_id: str) -> None:
            if task_id in visiting:
                raise ValueError("task dependencies must form a DAG")
            if task_id in visited:
                return
            visiting.add(task_id)
            for dependency in tasks[task_id].dependencies:
                visit(dependency)
            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in sorted(tasks):
            visit(task_id)

    def views(self, superseded_refs: set[str] | None = None) -> tuple[TaskView, ...]:
        tasks = {item.task_id: item for item in self.current}
        superseded = superseded_refs or set()
        results = []
        for task in tasks.values():
            blocked = any(
                tasks[key].status is not TaskStatus.COMPLETE
                for key in task.dependencies
            )
            linked = (*task.evidence_refs, *task.hypothesis_refs)
            results.append(
                TaskView(
                    task=task,
                    structurally_blocked=blocked,
                    stale=any(reference in superseded for reference in linked),
                )
            )
        return tuple(sorted(results, key=lambda item: item.task.task_id))

    def snapshot(self) -> list[JsonValue]:
        return [item.model_dump(mode="json") for item in self.current]
