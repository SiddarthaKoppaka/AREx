"""The small, dedicated prompt used for every repair generation.

A repair is not a new agent decision: it never sees the observation, the
working set, or episode history, only the malformed text and why it failed.
This keeps a bad parse cheap to fix instead of re-running the full cognitive
prompt, which is what let one failing decision balloon to dozens of full-size
generations in the live run that motivated this module.
"""

from arc_agi_3.contracts.generation import ErrorCategory

from .generation_errors import CATEGORY_INSTRUCTIONS

MAX_MALFORMED_ECHO = 4000

_BASE = (
    "Repair this output into a valid CognitiveDecision. Preserve the original "
    "semantic intent. Do not reconsider the environment. Do not add new "
    "strategy. Return only the repaired JSON object matching the schema "
    "below, with no markdown or commentary.\n"
)


def _echo(malformed_output: str) -> str:
    if len(malformed_output) <= MAX_MALFORMED_ECHO:
        return malformed_output
    return malformed_output[:MAX_MALFORMED_ECHO] + "...[truncated for repair]"


def build_repair_prompt(
    category: ErrorCategory, error_text: str, malformed_output: str
) -> str:
    return (
        _BASE
        + CATEGORY_INSTRUCTIONS[category]
        + f"\nERROR_CATEGORY: {category}\nERROR_DETAIL:\n{error_text}\n"
        "MALFORMED_OUTPUT:\n" + _echo(malformed_output)
    )
