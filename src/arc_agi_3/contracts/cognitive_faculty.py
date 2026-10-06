"""Cognitive faculties the core agent may consult; no hierarchy implied."""

from enum import StrEnum


class CognitiveFaculty(StrEnum):
    HYPOTHESIS = "hypothesis"
    DYNAMICS = "dynamics"
    CRITIC = "critic"
