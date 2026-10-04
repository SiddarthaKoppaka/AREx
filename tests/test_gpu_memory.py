"""GPU memory telemetry and cleanup are optional and never require CUDA."""

from arc_agi_3.adapters import gpu_memory
from arc_agi_3.adapters.generation_errors import GenerationBackendError
from arc_agi_3.adapters.transformers import TransformersBackend, TransformersConfig
from tests.test_transformers import Model, Tokenizer


def test_snapshot_is_none_without_cuda() -> None:
    assert gpu_memory.snapshot() is None


def test_empty_cache_after_failure_is_a_no_op_without_cuda() -> None:
    assert gpu_memory.empty_cache_after_failure() is None


class RecordingReporter:
    def __init__(self) -> None:
        self.stages: list[tuple[str, dict[str, object]]] = []

    def stage(self, name: str, **details: object) -> None:
        self.stages.append((name, details))


class FailingModel(Model):
    def generate(self, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("CUDA out of memory")


def test_generation_instrumentation_is_cpu_safe_when_disabled() -> None:
    config = TransformersConfig(model_path="/weights", max_new_tokens=4)
    backend = TransformersBackend(config, model=Model(), tokenizer=Tokenizer())
    reporter = RecordingReporter()
    backend.reporter = reporter
    backend.generate("choose", {"type": "object"})
    assert not any(name.startswith("generation_memory") for name, _ in reporter.stages)


def test_gpu_memory_instrumentation_flag_is_harmless_on_cpu() -> None:
    config = TransformersConfig(
        model_path="/weights", max_new_tokens=4, gpu_memory_instrumentation=True
    )
    backend = TransformersBackend(config, model=Model(), tokenizer=Tokenizer())
    reporter = RecordingReporter()
    backend.reporter = reporter
    backend.generate("choose", {"type": "object"})
    assert not any(name.startswith("generation_memory") for name, _ in reporter.stages)


def test_a_failed_generation_releases_tensors_and_raises_backend_error() -> None:
    config = TransformersConfig(model_path="/weights", max_new_tokens=4)
    backend = TransformersBackend(config, model=FailingModel(), tokenizer=Tokenizer())
    reporter = RecordingReporter()
    backend.reporter = reporter
    try:
        backend.generate("choose", {"type": "object"})
    except GenerationBackendError as error:
        assert "CUDA out of memory" in str(error)
        assert error.usage.input_tokens == 3
    else:
        raise AssertionError("expected GenerationBackendError")
    assert any(name == "generation_failed" for name, _ in reporter.stages)
