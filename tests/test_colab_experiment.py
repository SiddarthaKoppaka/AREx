"""One-process workflow verifies smoke output and keeps failed-run artifacts."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from arc_agi_3.adapters.inference import BackendGeneration
from arc_agi_3.contracts.decision import ModelUsage
from arc_agi_3.research import colab_experiment
from arc_agi_3.settings import resolve_runtime
from arc_agi_3.settings_models import RuntimeSettings


def _prepare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, smoke_text: str
) -> RuntimeSettings:
    settings = resolve_runtime(
        "local_smoke",
        overrides={
            "model_path": tmp_path / "weights",
            "output_dir": tmp_path / "runs",
            "log_dir": tmp_path / "logs",
            "live_trace_mode": "silent",
        },
        environ={},
    )
    arcade = SimpleNamespace(make=lambda *args, **kwargs: object())
    arc = SimpleNamespace(
        Arcade=lambda **kwargs: arcade,
        OperationMode=SimpleNamespace(OFFLINE="offline"),
    )
    monkeypatch.setattr(colab_experiment, "import_module", lambda name: arc)
    monkeypatch.setattr(colab_experiment, "gpu_vram_gib", lambda: 80.0)
    monkeypatch.setattr(
        colab_experiment, "stage_model", lambda source, cache, **kwargs: source
    )

    class Backend:
        def __init__(self, config, reporter):  # type: ignore[no-untyped-def]
            self.config = config

        def generate(self, prompt, schema):  # type: ignore[no-untyped-def]
            return BackendGeneration(text=smoke_text, usage=ModelUsage())

    monkeypatch.setattr(colab_experiment, "TransformersBackend", Backend)
    return settings


def test_invalid_smoke_stops_before_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _prepare(tmp_path, monkeypatch, '{"status":"bad"}')
    with pytest.raises(ValueError, match="status=ok"):
        colab_experiment.run_colab_experiment(settings, tmp_path / "arc")
    assert not (tmp_path / "runs").exists()


def test_captured_failure_still_writes_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _prepare(tmp_path, monkeypatch, '```json\n{"status":"ok"}\n```')

    def fake_runner(self, run_id, game_id, environment, backend, **kwargs):  # type: ignore[no-untyped-def]
        run_dir = self.output_dir / run_id
        run_dir.mkdir(parents=True)
        result = {"stop_reason": "failure"}

        def fail():  # type: ignore[no-untyped-def]
            raise RuntimeError("episode failed")

        return SimpleNamespace(session=SimpleNamespace(result=result), run=fail)

    monkeypatch.setattr(RuntimeSettings, "build_runner", fake_runner)
    with pytest.raises(RuntimeError, match="episode failed"):
        colab_experiment.run_colab_experiment(settings, tmp_path / "arc")
    artifacts = list((tmp_path / "runs").glob("*/result.json"))
    assert len(artifacts) == 1
    assert json.loads(artifacts[0].read_text()) == {"stop_reason": "failure"}
