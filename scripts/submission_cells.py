"""Construct deterministic cells for the ARC-AGI-3 submission notebook."""

from textwrap import dedent


def cell(kind: str, cell_id: str, source: str) -> dict[str, object]:
    language = "python" if kind == "code" else "markdown"
    result: dict[str, object] = {
        "cell_type": kind,
        "id": cell_id,
        "metadata": {"id": cell_id, "language": language},
        "source": source,
    }
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build(config: dict[str, object], agent_source: str) -> dict[str, object]:
    runtime = config["runtime"]
    assert isinstance(runtime, dict)
    environment = "\n".join(
        f"os.environ['AREX_{key.upper()}'] = {str(value)!r}"
        for key, value in runtime.items()
        if key != "wheelhouse"
    )
    install = (
        "%pip install --no-index --find-links "
        "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels "
        "arc-agi python-dotenv\n"
        f"%pip install --no-index --find-links {runtime['wheelhouse']} "
        "arc-agi-3 transformers accelerate safetensors\n"
    )
    run = "\n".join(
        [
            "import os",
            "import subprocess",
            environment,
            "",
            "if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):",
            "    gateway = 'http://gateway:8001/api/games'",
            "    subprocess.run(['curl', '--fail', '--retry', '120', "
            "'--retry-all-errors', '--retry-delay', '5', gateway], check=True)",
            "    source = '/kaggle/input/competitions/arc-prize-2026-arc-agi-3/'"
            " + 'ARC-AGI-3-Agents'",
            "    root = __import__('pathlib').Path('/kaggle/working/'"
            " + 'ARC-AGI-3-Agents')",
            "    subprocess.run(['cp', '-r', source, str(root)], check=True)",
            "    target = root / 'agents/templates/my_agent.py'",
            "    subprocess.run(['cp', '/tmp/my_agent.py', str(target)], check=True)",
            '    agents = "from .agent import Agent, Playback\\nfrom '
            ".swarm import Swarm\\nfrom .templates.my_agent import MyAgent\\n"
            "AVAILABLE_AGENTS = "
            "{'myagent': MyAgent}\\n\"",
            "    (root / 'agents/__init__.py').write_text(agents)",
            "    env = 'SCHEME=http\\nHOST=gateway\\nPORT=8001\\n' + "
            "'ARC_API_KEY=test-key-123\\nARC_BASE_URL=http://gateway:8001/\\n' + "
            "'OPERATION_MODE=online\\nRECORDINGS_DIR=/kaggle/working/"
            "server_recording\\n'",
            "    (root / '.env').write_text(env)",
            "    subprocess.run(['python', 'main.py', '--agent', 'myagent'], "
            "cwd=root, check=True)",
        ]
    )
    dummy = dedent("""\
        import os
        if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            import pandas as pd
            columns = ['row_id', 'game_id', 'end_of_game', 'score']
            submission = pd.DataFrame([['1_0', '1', True, 1]], columns=columns)
            submission.to_parquet('/kaggle/working/submission.parquet', index=False)
        """)
    kernel = config["kernel"]
    assert isinstance(kernel, dict)
    metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
        "kaggle": {
            "accelerator": kernel["accelerator"],
            "isInternetEnabled": False,
            "isGpuEnabled": True,
            "language": "python",
            "sourceType": "notebook",
        },
    }
    cells = [
        cell("markdown", "arex-intro", "# AREx ARC-AGI-3 submission"),
        cell("code", "offline-install", install),
        cell("code", "write-agent", "%%writefile /tmp/my_agent.py\n" + agent_source),
        cell("code", "gateway-run", run),
        cell("code", "dummy-submission", dummy),
    ]
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": metadata, "cells": cells}
