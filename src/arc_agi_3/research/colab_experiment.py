"""One-process Colab workflow using a resolved runtime profile."""

from importlib import import_module
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from arc_agi_3.adapters.arc import ArcEnvironmentAdapter
from arc_agi_3.adapters.structured_output import parse_json_object
from arc_agi_3.adapters.transformers import TransformersBackend
from arc_agi_3.runtime.reporting import LiveReporter
from arc_agi_3.settings_models import RuntimeSettings
from arc_agi_3.trace.canonical import canonical_json

from .gpu import gpu_vram_gib
from .model_staging import stage_model


def _load_arc_wrapper(arc_agi: Any, root: Path, game_id: str, seed: int) -> Any:
    arcade = arc_agi.Arcade(
        operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=str(root)
    )
    wrapper = arcade.make(game_id, seed=seed, include_frame_data=True)
    if wrapper is None:
        raise FileNotFoundError(
            f"ARC offline environment {game_id} is missing under {root}; "
            "run the explicit bootstrap cell before loading the model"
        )
    return wrapper


def run_colab_experiment(
    settings: RuntimeSettings,
    dataset_root: Path,
    *,
    game_id: str = "ls20",
    seed: int = 0,
) -> dict[str, object]:
    if settings.profile not in ("colab_transformers", "local_smoke"):
        raise ValueError("Transformers workflow requires a local or Colab profile")
    run_id = f"{settings.profile}-{game_id}-{uuid4().hex[:12]}"
    run_dir = settings.output_dir / run_id
    reporter = LiveReporter(
        settings.live_trace_mode, settings.log_dir / f"{run_id}.log"
    )
    print("Resolved AREx configuration:", canonical_json(settings), flush=True)
    reporter.stage("resolved_config", settings=canonical_json(settings))
    vram = gpu_vram_gib()
    if settings.require_gpu and vram is None:
        raise RuntimeError("selected Transformers profile requires a CUDA GPU")
    reporter.stage("gpu", vram_gib=vram)
    arc_agi = import_module("arc_agi")
    started = perf_counter()
    wrapper = _load_arc_wrapper(arc_agi, dataset_root, game_id, seed)
    reporter.stage("arc_environment_finish", seconds=round(perf_counter() - started, 3))
    source = settings.model_path
    cache = settings.model_cache_dir or Path("/content/arex-model-cache")
    model_path = stage_model(
        source,
        cache,
        enabled=settings.model_staging == "copy_if_space",
        digest=settings.model_digest,
    )
    reporter.stage("model_staging", source=source, active_path=model_path)
    backend = TransformersBackend(
        settings.to_transformers_config(model_path=model_path), reporter=reporter
    )
    started = perf_counter()
    smoke = backend.generate(
        'Return exactly {"status":"ok"} as JSON.',
        {"type": "object", "required": ["status"]},
    )
    reporter.stage(
        "smoke_finish",
        seconds=round(perf_counter() - started, 3),
        input_tokens=smoke.usage.input_tokens,
        output_tokens=smoke.usage.output_tokens,
    )
    if parse_json_object(smoke.text).get("status") != "ok":
        raise ValueError("deterministic model smoke did not return status=ok")
    runner = settings.build_runner(
        run_id,
        game_id,
        ArcEnvironmentAdapter(wrapper),
        backend,
        seed=seed,
        reporter=reporter,
        raise_on_failure=True,
    )
    started = perf_counter()
    try:
        result = runner.run()
    finally:
        reporter.stage("episode_finish", seconds=round(perf_counter() - started, 3))
        if runner.session.result is not None:
            (run_dir / "result.json").write_text(
                canonical_json(runner.session.result) + "\n"
            )
    return {"run_id": run_id, "result": result.model_dump(mode="json")}
