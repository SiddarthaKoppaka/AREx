"""Duck vision, serving backends, dead-click rule, Kaggle wiring, real game."""

import io
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from arc_agi_3.contracts.observation import Action, Observation
from arc_agi_3.duck.agent import DuckAgent
from arc_agi_3.duck.bridge import DuckBridge
from arc_agi_3.duck.chat import latest_image_only, strip_think
from arc_agi_3.duck.dead import DeadClicks
from arc_agi_3.duck.fit import fits, weights_gb
from arc_agi_3.duck.hf_chat import hf_messages
from arc_agi_3.duck.image import render_png
from arc_agi_3.duck.openai_chat import OpenAICompatChat
from arc_agi_3.duck.segment import click_candidates, signature_at
from arc_agi_3.duck.server import server_command, wait_ready
from arc_agi_3.duck.settings import DuckSettings

GRID = [[0, 0, 0, 0], [0, 7, 0, 0], [0, 0, 0, 3], [0, 0, 0, 0]]


def obs(level: int = 0) -> Observation:
    return Observation.build(
        game_id="g", frame=[GRID], state="NOT_FINISHED", levels_completed=level,
        win_levels=2, available_actions=(1, 6),
    )  # fmt: skip


class Scripted:
    def __init__(self, *replies: str) -> None:
        self.replies, self.seen = list(replies), []

    def chat(self, messages: list[dict[str, Any]]) -> str:
        self.seen.append(list(messages))
        return self.replies.pop(0)


def block(code: str) -> str:
    return f"```python\n{code}\n```"


def test_png_and_only_latest_image_kept() -> None:
    png = render_png(GRID, scale=2)
    assert Image.open(io.BytesIO(png)).size == (8, 8)
    log = [{"role": "user", "content": "a", "image": png}] * 2
    kept = latest_image_only(log)
    assert "image" not in kept[0] and "image" in kept[1]
    assert strip_think("<think>x</think> ok") == "ok"


def test_hf_and_openai_message_formats() -> None:
    log = [{"role": "user", "content": "hi", "image": render_png(GRID)}]
    part = hf_messages(log)[0]["content"]
    assert part[0]["type"] == "image" and part[1] == {"type": "text", "text": "hi"}
    payload = OpenAICompatChat().payload(log)
    url = payload["messages"][0]["content"][0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_agent_attaches_image_when_enabled() -> None:
    chat = Scripted(block("act(1)"))
    DuckAgent(chat, use_images=True).decide(obs())
    assert "image" in chat.seen[0][-1]


def test_server_command_and_readiness_timeout() -> None:
    sg = server_command("sglang", "/m", port=9, tp=2, context=100)
    assert "sglang.launch_server" in sg and sg[sg.index("--tp") + 1] == "2"
    vl = server_command("vllm", "/m")
    assert "--max-model-len" in vl and "--served-model-name" in vl
    with pytest.raises(TimeoutError):
        wait_ready("http://127.0.0.1:9", timeout=0.2, poll=0.05)


def test_fit_estimates() -> None:
    assert weights_gb(27, 8) == 27 and weights_gb(31, 4) == 15.5
    assert not fits(27, 8, vram_gb=32) and fits(27, 4, vram_gb=24)


def test_dead_click_rule_excludes_and_resets_on_level() -> None:
    dead = DeadClicks()
    assert signature_at(GRID, 1, 1) == (7, 1)
    dead.decided(GRID, Action(action_id=6, data={"x": 1, "y": 1}))
    dead.observed(changed_cells=0, level=0)
    assert dead.dead == {(7, 1)}
    assert click_candidates(GRID, exclude=dead.dead) == [(3, 2)]
    dead.observed(changed_cells=0, level=1)
    assert dead.dead == set()


def test_agent_learns_dead_click_across_turns() -> None:
    chat = Scripted(block("act(6, x=1, y=1)"), block("act(1)"))
    agent = DuckAgent(chat)
    agent.decide(obs())
    agent.decide(obs())  # identical grid: the click did nothing
    assert (7, 1) in agent.dead.dead
    assert "DEAD_CLICK_TYPES" in chat.seen[1][-1]["content"]


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DUCK_MODEL_PATH", "/m")
    monkeypatch.setenv("DUCK_BACKEND", "sglang")
    monkeypatch.setenv("DUCK_MAX_ACTIONS", "50")
    monkeypatch.setenv("DUCK_USE_IMAGES", "true")
    settings = DuckSettings.from_env()
    assert (settings.backend, settings.max_actions, settings.use_images) == (
        "sglang", 50, True,
    )  # fmt: skip
    monkeypatch.delenv("DUCK_MODEL_PATH")
    with pytest.raises(RuntimeError):
        DuckSettings.from_env()


def test_bridge_stops_on_win_value_and_action_cap() -> None:
    bridge = DuckBridge("g", DuckAgent(Scripted()), max_actions=1)
    won = types.SimpleNamespace(state=types.SimpleNamespace(value="WIN"))
    live = types.SimpleNamespace(state=types.SimpleNamespace(value="NOT_FINISHED"))
    assert bridge.is_done(won) and not bridge.is_done(live)
    bridge.actions_taken = 1
    assert bridge.is_done(live)


def test_kaggle_agent_class_delegates_to_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeAgent:
        def __init__(self, card_id: str, game_id: str, *rest: Any) -> None:
            self.game_id = game_id

    module = types.ModuleType("agents.agent")
    module.Agent = FakeAgent  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "agents", types.ModuleType("agents"))
    monkeypatch.setitem(sys.modules, "agents.agent", module)
    from arc_agi_3.duck import kaggle_agent

    settings = DuckSettings(model_path="/m", max_actions=500)
    monkeypatch.setattr(kaggle_agent, "make_agent", lambda s: DuckAgent(Scripted()))
    cls = kaggle_agent.build_agent_class(settings)
    agent = cls("card", "ls20", "name", "url", False, None)
    assert cls.MAX_ACTIONS == 500 and agent.bridge.max_actions == 500
    won = types.SimpleNamespace(state=types.SimpleNamespace(value="WIN"))
    assert agent.is_done([], won)


def test_real_game_episode_with_scripted_policy(tmp_path: Path) -> None:
    arc_agi = pytest.importorskip("arc_agi")
    root = Path("arc-prize-2026-arc-agi-3/environment_files")
    if not root.exists():
        pytest.skip("bundled ARC environments are unavailable")
    from arc_agi_3.adapters.arc import ArcEnvironmentAdapter
    from arc_agi_3.duck.local import run_episode

    arcade = arc_agi.Arcade(
        operation_mode=arc_agi.OperationMode.OFFLINE,
        environments_dir=str(root),
        recordings_dir=str(tmp_path),
    )
    wrapper = arcade.make("ls20", seed=0, save_recording=False)
    policy = block(
        "a = [x for x in available_actions if x != 0]\n"
        "import random\n"
        "c = click_candidates(grid)\n"
        "k = random.choice(a)\n"
        "act(k, *(c[0] if k == 6 and c else (32, 32)) if k == 6 else ())"
    )
    agent = DuckAgent(Scripted(*[policy] * 40), use_images=True)
    levels = run_episode(ArcEnvironmentAdapter(wrapper), agent, max_steps=40)
    assert levels >= 0 and agent.model_calls == 40 and agent.fallbacks == 0


def test_preflight_rejects_oversized_model_only_with_cuda() -> None:
    from arc_agi_3.duck.fit import preflight

    preflight(27, 8, vram_gb=0)  # no CUDA visible: skipped
    preflight(None, 8, vram_gb=16)  # size unknown: skipped
    with pytest.raises(RuntimeError, match="27B @ 8-bit"):
        preflight(27, 8, vram_gb=16)


def test_openrouter_requires_key_and_sends_bearer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from arc_agi_3.duck.factory import build_chat

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        build_chat(DuckSettings(backend="openrouter"))
    chat = OpenAICompatChat("https://x/v1", "m", api_key="k")
    request = chat._request([{"role": "user", "content": "hi"}])
    assert request.get_header("Authorization") == "Bearer k"


def test_reasoning_effort_in_payload_and_trace_written(tmp_path: Path) -> None:
    chat = OpenAICompatChat(reasoning_effort="medium")
    assert chat.payload([{"role": "user", "content": "x"}])["reasoning"] == {
        "effort": "medium"
    }
    assert "reasoning" not in OpenAICompatChat().payload([])
    trace = tmp_path / "t.jsonl"
    DuckAgent(Scripted(block("act(1)")), trace_path=str(trace)).decide(obs())
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    assert "reply" in rows[0] and rows[1]["action"]["action_id"] == 1


def test_post_enforces_wall_clock_limit_on_a_trickling_server() -> None:
    import http.server
    import threading
    import time

    class Slow(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # keep the connection alive, never finish
            self.send_response(200)
            self.end_headers()
            for _ in range(50):
                self.wfile.write(b" ")
                self.wfile.flush()
                time.sleep(0.1)

        def log_message(self, *_: object) -> None: ...

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Slow)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    chat = OpenAICompatChat(
        f"http://127.0.0.1:{server.server_port}/v1", timeout=0.5, retries=0
    )
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        chat.chat([{"role": "user", "content": "x"}])
    assert time.monotonic() - started < 2
    server.shutdown()
