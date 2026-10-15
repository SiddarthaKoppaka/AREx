"""The fixed, non-compactable portion of every prompt (CONTRACT+EPISTEMICS+
TOOLS+TOOL_CONTRACTS) stays a bounded fraction of the real soft_input_limit
default - not an ad hoc test-local constant. See PR-02-scoping.md section 3:
this is the dedicated regression this project lacked before, after PR-01's
prose/schema growth required bumping an unrelated stress test's own
disconnected constant with no earlier warning."""

from arc_agi_3.adapters.prompt_render import render
from arc_agi_3.adapters.prompt_text import CONTRACT, EPISTEMICS, TOOLS
from arc_agi_3.adapters.tool_schemas import tool_schemas
from arc_agi_3.settings_fields import RuntimeFields


def estimated_tokens(text: str) -> int:
    # Same rough chars-per-token estimate `cognition/scratchpad.py` already
    # uses for its own token-budget check, absent a live tokenizer here.
    return (len(text) + 3) // 4


def test_fixed_overhead_stays_within_a_bounded_fraction_of_soft_input_limit() -> None:
    fixed = (
        CONTRACT + EPISTEMICS + TOOLS + f"TOOL_CONTRACTS:\n{render(tool_schemas())}\n"
    )
    budget = RuntimeFields().soft_input_limit
    tokens = estimated_tokens(fixed)
    fraction = tokens / budget
    assert fraction < 0.5, (
        f"fixed, non-compactable prompt overhead is ~{tokens} estimated tokens "
        f"({fraction:.0%} of the {budget}-token soft_input_limit default) - "
        "raise this bound deliberately if content genuinely needs to grow, "
        "rather than letting it creep toward the limit unnoticed"
    )
