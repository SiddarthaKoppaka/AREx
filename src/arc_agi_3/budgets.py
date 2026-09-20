"""Typed deterministic resource accounting."""

from collections.abc import Callable
from uuid import uuid4

from pydantic import Field

from arc_agi_3.contracts.base import Contract
from arc_agi_3.contracts.enums import Resource


class BudgetExceeded(RuntimeError):
    pass


class BudgetSnapshot(Contract):
    limits: dict[Resource, int] = Field(default_factory=dict)
    reserved: dict[Resource, int] = Field(default_factory=dict)
    consumed: dict[Resource, int] = Field(default_factory=dict)


class Reservation(Contract):
    reservation_id: str
    amounts: dict[Resource, int]


class BudgetLedger:
    def __init__(
        self,
        limits: dict[Resource, int],
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        if any(value < 0 for value in limits.values()):
            raise ValueError("budget limits must be non-negative")
        self.limits = dict(limits)
        self.consumed: dict[Resource, int] = {}
        self.reservations: dict[str, dict[Resource, int]] = {}
        self.id_factory = id_factory or (lambda: str(uuid4()))

    def _reserved(self, resource: Resource) -> int:
        return sum(values.get(resource, 0) for values in self.reservations.values())

    def remaining(self) -> dict[Resource, int]:
        return {
            resource: limit - self.consumed.get(resource, 0) - self._reserved(resource)
            for resource, limit in self.limits.items()
        }

    def reserve(self, amounts: dict[Resource, int]) -> Reservation:
        if any(value <= 0 for value in amounts.values()):
            raise ValueError("reservation amounts must be positive")
        for resource, amount in amounts.items():
            if resource in self.limits and amount > self.remaining()[resource]:
                raise BudgetExceeded(f"{resource} budget exhausted")
        reservation = Reservation(
            reservation_id=self.id_factory(), amounts=dict(amounts)
        )
        self.reservations[reservation.reservation_id] = dict(amounts)
        return reservation

    def commit(self, reservation_id: str) -> None:
        for resource, amount in self.reservations.pop(reservation_id).items():
            self.consumed[resource] = self.consumed.get(resource, 0) + amount

    def release(self, reservation_id: str) -> None:
        self.reservations.pop(reservation_id)

    def spend(self, resource: Resource, amount: int = 1) -> None:
        reservation = self.reserve({resource: amount})
        self.commit(reservation.reservation_id)

    def record_actual(self, resource: Resource, amount: int) -> None:
        """Record measured work even when a generation crossed its limit."""
        if amount < 0:
            raise ValueError("actual usage must be non-negative")
        self.consumed[resource] = self.consumed.get(resource, 0) + amount

    def snapshot(self) -> BudgetSnapshot:
        resources = set(self.limits) | set(self.consumed)
        reserved = {resource: self._reserved(resource) for resource in resources}
        return BudgetSnapshot(
            limits=self.limits,
            reserved={key: value for key, value in reserved.items() if value},
            consumed=self.consumed,
        )
