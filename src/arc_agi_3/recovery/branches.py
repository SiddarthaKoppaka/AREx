"""Branch identity and lifecycle without diagnosing failure meaning."""

from arc_agi_3.contracts.recovery import BranchRecord


class BranchTracker:
    def __init__(self, initial_branch_id: str) -> None:
        self.active_branch_id = initial_branch_id
        self._records = {
            initial_branch_id: BranchRecord(
                branch_id=initial_branch_id,
                status="active",
            )
        }

    @property
    def records(self) -> tuple[BranchRecord, ...]:
        return tuple(self._records[key] for key in sorted(self._records))

    def fork(
        self, new_branch_id: str, checkpoint_id: str
    ) -> tuple[BranchRecord, BranchRecord]:
        if new_branch_id in self._records:
            raise ValueError("branch ID already exists")
        parent_id = self.active_branch_id
        frozen = self._records[parent_id].model_copy(update={"status": "frozen"})
        child = BranchRecord(
            branch_id=new_branch_id,
            parent_branch_id=parent_id,
            checkpoint_id=checkpoint_id,
            status="active",
        )
        self._records[parent_id] = frozen
        self._records[new_branch_id] = child
        self.active_branch_id = new_branch_id
        return frozen, child

    def abandon(self) -> BranchRecord:
        current = self._records[self.active_branch_id].model_copy(
            update={"status": "abandoned"}
        )
        self._records[self.active_branch_id] = current
        return current
