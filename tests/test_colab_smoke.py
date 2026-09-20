"""Colab model-loading and smoke configuration contracts."""

import json
from pathlib import Path
from types import SimpleNamespace

from arc_agi_3.adapters.transformers import TransformersBackend, TransformersConfig
from arc_agi_3.settings import resolve_runtime


def test_experiment_zero_defaults() -> None:
    settings = resolve_runtime(
        "colab_transformers", overrides={"model_path": "/weights"}, environ={}
    )
    assert settings.max_turns == settings.max_model_calls == 8
    assert settings.max_new_tokens == 1536
    assert settings.max_input_tokens == 32768


def test_remote_nf4_loading_is_explicit(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls: dict[str, dict[str, object]] = {}
    tokenizer = object()
    model = SimpleNamespace(eval=lambda: model)

    def load_tokenizer(path, **kwargs):  # type: ignore[no-untyped-def]
        calls["tokenizer"] = {"path": path, **kwargs}
        return tokenizer

    def load_model(path, **kwargs):  # type: ignore[no-untyped-def]
        calls["model"] = {"path": path, **kwargs}
        return model

    transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model),
        BitsAndBytesConfig=lambda **kwargs: {"bnb": kwargs},
    )
    torch = SimpleNamespace(float16="float16")
    monkeypatch.setattr(
        "arc_agi_3.adapters.transformers_loader.import_module",
        lambda name: transformers if name == "transformers" else torch,
    )
    config = TransformersConfig(
        model_path="Qwen/Qwen3-8B", local_files_only=False, quantization="nf4"
    )
    backend = TransformersBackend(config)
    assert calls["tokenizer"]["local_files_only"] is False
    assert calls["model"]["quantization_config"] == {
        "bnb": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": "float16",
        }
    }
    assert backend.metadata["quantization"] == "nf4"


def test_colab_notebook_is_a_thin_current_kernel_driver() -> None:
    notebook = json.loads(
        Path("notebooks/arc_agi_3_colab_transformers.ipynb").read_text()
    )
    assert [cell["id"] for cell in notebook["cells"]] == [
        "intro",
        "mount-drive",
        "sync-repository",
        "install-python312",
        "prepare-arc-assets",
        "run-experiment",
        "artifacts",
    ]
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "drive.mount('/content/drive')" in source
    assert "f'{REPO}[colab]'" in source
    assert "'uv'" in source and "'3.12'" in source
    assert "arc_agi_3.research.arc_bootstrap" in source
    assert "arc_agi_3.research.colab_entry" in source
    assert "colab_transformers" in source
    assert "REPLACE_WITH_MODEL_DIRECTORY" in source
    assert "%pip" not in source
    assert "run_env" not in source


def test_local_notebook_uses_shared_profile() -> None:
    notebook = json.loads(
        Path("notebooks/arc_agi_3_local_transformers.ipynb").read_text()
    )
    source = "\n".join(str(cell["source"]) for cell in notebook["cells"])
    assert "local_smoke" in source
    assert "arc_agi_3.research.colab_entry" in source
    assert "model_path" in source
