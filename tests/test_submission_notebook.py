"""Generated Kaggle artifacts preserve the official submission contract."""

import json
from pathlib import Path

from scripts.build_submission_notebook import AGENT, CONFIG, build


def test_submission_notebook_contract() -> None:
    config = json.loads(CONFIG.read_text())
    notebook = build(config, AGENT.read_text())
    cells = notebook["cells"]
    assert [item["metadata"]["id"] for item in cells] == [
        "arex-intro",
        "offline-install",
        "write-agent",
        "gateway-run",
        "dummy-submission",
    ]
    assert cells[0]["metadata"]["language"] == "markdown"
    assert all(item["metadata"]["language"] == "python" for item in cells[1:])
    assert notebook["metadata"]["kaggle"] == {
        "accelerator": "nvidiaRtx6000",
        "isInternetEnabled": False,
        "isGpuEnabled": True,
        "language": "python",
        "sourceType": "notebook",
    }
    source = "\n".join(str(item["source"]) for item in cells)
    assert "KAGGLE_IS_COMPETITION_RERUN" in source
    assert "from .swarm import Swarm" in source
    assert "python', 'main.py', '--agent', 'myagent" in source
    assert "/kaggle/working/submission.parquet" in source
    assert "AREX_MODEL_PATH" in source


def test_central_config_declares_all_attached_resources() -> None:
    config = json.loads(CONFIG.read_text())
    kernel = config["kernel"]
    assert kernel["competition_sources"]
    assert kernel["dataset_sources"]
    assert kernel["model_sources"]
    assert Path(config["runtime"]["wheelhouse"]).is_absolute()
    assert Path(config["runtime"]["model_path"]).is_absolute()
    _, slug, framework, variation, version = kernel["model_sources"][0].split("/")
    expected_mount = f"/kaggle/input/{slug}/{framework}/{variation}/{version}"
    assert config["runtime"]["model_path"] == expected_mount
