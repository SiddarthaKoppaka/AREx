"""Retrospective prediction-compatibility request and result contracts."""

from pydantic import Field

from .base import Contract
from .observation import Action
from .prediction import ExpectedOutcome


class RetrodictionRequest(Contract):
    prediction: ExpectedOutcome
    action: Action | None = None
    limit: int = Field(default=16, gt=0, le=64)


class RetrodictionItem(Contract):
    transition_event_id: str
    action_event_id: str
    step_id: int = Field(ge=0)
    mismatches: tuple[str, ...] = ()


class RetrodictionResult(Contract):
    """Compatibility of a prediction with past transitions; truth is not inferred."""

    compatible: tuple[RetrodictionItem, ...] = ()
    contradicting: tuple[RetrodictionItem, ...] = ()
    not_evaluable: tuple[RetrodictionItem, ...] = ()
    examined: int = Field(default=0, ge=0)
