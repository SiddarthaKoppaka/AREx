"""Call the model, releasing failed-generation tensors and reporting GPU state.

Separated from `TransformersBackend` so the failure-cleanup path (release
references, let the allocator reuse freed blocks, report what happened) is
easy to audit on its own, independent of tokenization and decoding.
"""

from contextlib import nullcontext
from importlib import import_module
from time import perf_counter
from typing import Any

from arc_agi_3.contracts.decision import ModelUsage

from . import gpu_memory
from .generation_errors import GenerationBackendError
from .transformers_limits import StageReporter


def _inference_context() -> Any:
    try:
        torch = import_module("torch")
    except ImportError:
        return nullcontext()
    return torch.inference_mode()


def _report_memory(reporter: StageReporter | None, enabled: bool, stage: str) -> None:
    if not enabled or reporter is None:
        return
    memory = gpu_memory.snapshot()
    if memory is not None:
        reporter.stage(stage, **memory.model_dump(mode="json"))


def run_model(
    model: Any,
    inputs: Any,
    input_tokens: int,
    *,
    max_new_tokens: int,
    max_time_seconds: float,
    reporter: StageReporter | None = None,
    memory_instrumentation: bool = False,
) -> tuple[Any, int]:
    """Return (generated token slice, latency_ms), or raise GenerationBackendError."""
    started = perf_counter()
    _report_memory(reporter, memory_instrumentation, "generation_memory_before")
    if reporter:
        reporter.stage("generation_start", input_tokens=input_tokens)
    try:
        with _inference_context():
            raw = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                max_time=max_time_seconds,
                do_sample=False,
            )
        output = raw[0][input_tokens:]
    except Exception as error:
        del inputs
        memory = gpu_memory.empty_cache_after_failure()
        if reporter:
            reporter.stage("generation_failed", error=str(error), memory_after=memory)
        raise GenerationBackendError(
            str(error), ModelUsage(input_tokens=input_tokens)
        ) from error
    finally:
        if reporter:
            reporter.stage(
                "generation_finish", seconds=round(perf_counter() - started, 3)
            )
    _report_memory(reporter, memory_instrumentation, "generation_memory_after")
    return output, round((perf_counter() - started) * 1000)
