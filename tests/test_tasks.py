"""Task lifecycle is LM-authored while graph facts remain mechanical."""

import pytest

from arc_agi_3.cognition import TaskGraph
from arc_agi_3.contracts.cognition import TaskRecord
from arc_agi_3.contracts.enums import TaskStatus


def task(key: str, dependencies: tuple[str, ...] = ()) -> TaskRecord:
    return TaskRecord(
        task_id=key,
        version=1,
        purpose=f"purpose {key}",
        success_criteria=f"criteria {key}",
        dependencies=dependencies,
    )


def test_cycle_is_rejected_transactionally() -> None:
    graph = TaskGraph()
    with pytest.raises(ValueError, match="DAG"):
        graph.apply_many((task("a", ("b",)), task("b", ("a",))))
    assert graph.current == ()


def test_blocked_and_stale_are_views_not_status_mutations() -> None:
    graph = TaskGraph()
    source = task("source")
    child = task("child", ("source",)).model_copy(
        update={"evidence_refs": ("event@1",)}
    )
    graph.apply_many((source, child))
    views = {item.task.task_id: item for item in graph.views({"event@1"})}
    assert views["child"].structurally_blocked
    assert views["child"].stale
    assert views["child"].task.status is TaskStatus.OPEN


def test_completion_requires_explicit_next_version() -> None:
    graph = TaskGraph()
    graph.apply_many((task("a"),))
    completed = task("a").model_copy(
        update={"version": 2, "status": TaskStatus.COMPLETE}
    )
    graph.apply_many((completed,))
    assert graph.current[0].status is TaskStatus.COMPLETE
