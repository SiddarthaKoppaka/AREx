"""Compose official ARC, Qwen, and the existing AREx runner."""

import json
from importlib import import_module

from arc_agi_3.adapters.arc import ArcEnvironmentAdapter
from arc_agi_3.adapters.structured_model import StructuredModelAdapter
from arc_agi_3.adapters.transformers import TransformersBackend, TransformersConfig
from arc_agi_3.config import BudgetConfig, RunConfig
from arc_agi_3.contracts.enums import Resource
from arc_agi_3.runtime import build_runner
from arc_agi_3.trace.canonical import canonical_json
from arc_agi_3.trace.store import JsonlEventStore

from .colab_smoke import ColabSmokeConfig, gpu_vram_gib, resolve_quantization


def execute_smoke(settings: ColabSmokeConfig) -> dict[str, object]:
    arc_agi = import_module("arc_agi")
    vram = gpu_vram_gib()
    quantization = resolve_quantization(settings.quantization, vram)
    backend = TransformersBackend(
        TransformersConfig(
            model_path=settings.model,
            model_name=settings.model,
            model_digest=settings.revision,
            revision=settings.revision,
            max_new_tokens=settings.max_new_tokens,
            local_files_only=False,
            quantization=quantization,
        )
    )
    arcade = arc_agi.Arcade(
        operation_mode=arc_agi.OperationMode.OFFLINE,
        environments_dir=str(settings.dataset_root),
    )
    wrapper = arcade.make(settings.game_id, seed=settings.seed, include_frame_data=True)
    if wrapper is None:
        raise RuntimeError(f"ARC environment did not load: {settings.game_id}")
    budget = BudgetConfig(
        limits={
            Resource.ACTIONS: settings.max_actions,
            Resource.MODEL_CALLS: settings.max_model_calls,
            Resource.INPUT_TOKENS: settings.max_input_tokens,
            Resource.OUTPUT_TOKENS: settings.max_output_tokens,
        }
    )
    config = RunConfig(
        run_id=settings.run_id,
        experiment_id="colab-qwen3-8b-smoke",
        game_id=settings.game_id,
        seed=settings.seed,
        max_turns=settings.max_actions,
        output_dir=settings.output_dir,
        budget=budget,
    )
    result = build_runner(
        config,
        ArcEnvironmentAdapter(wrapper),
        StructuredModelAdapter(backend, settings.max_repairs),
    ).run()
    run_dir = settings.output_dir / settings.run_id
    events = JsonlEventStore(
        run_dir / "events.jsonl", "verify", "verify", "verify"
    ).read()
    (run_dir / "result.json").write_text(canonical_json(result) + "\n")
    (run_dir / "metrics.json").write_text(canonical_json(result.metrics) + "\n")
    return {
        "result": json.loads(result.model_dump_json()),
        "event_count": len(events),
        "last_event_hash": events[-1].event_hash,
        "quantization": quantization,
        "gpu_vram_gib": vram,
    }
