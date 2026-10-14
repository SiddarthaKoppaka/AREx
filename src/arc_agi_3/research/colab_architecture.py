"""Print which ExoCortex backends are actually active before a run starts.

A Colab notebook cell that silently loads the old Qwen-8B specialist
backend when the intent was a System-1-only experiment is exactly the
kind of mistake this exists to make visible at a glance.
"""

from arc_agi_3.adapters.inference import InferenceBackend
from arc_agi_3.adapters.system1 import System1Backend


def print_architecture(
    specialist_backend: InferenceBackend | None,
    system1_backend: System1Backend | None,
) -> None:
    if system1_backend is not None:
        architecture = "core-agent-plus-system1"
    elif specialist_backend is not None:
        architecture = "core-agent-plus-exocortex-specialists"
    else:
        architecture = "core-agent-only"
    print("Architecture:", architecture, flush=True)
    print(
        "ExoCortex specialist backend:",
        specialist_backend.metadata if specialist_backend else "NONE",
        flush=True,
    )
    print(
        "System-1 backend:",
        system1_backend.metadata if system1_backend else "NONE",
        flush=True,
    )
