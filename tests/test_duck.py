"""Duck agent: REPL, eviction, grid helpers, and the act() contract."""

from arc_agi_3.contracts.observation import Observation
from arc_agi_3.duck.agent import DuckAgent
from arc_agi_3.duck.context import evict
from arc_agi_3.duck.render import ascii_grid, changes
from arc_agi_3.duck.repl import Repl
from arc_agi_3.duck.segment import click_candidates, components, zoom

GRID = [[0, 0, 0, 0], [0, 7, 0, 0], [0, 0, 0, 3], [0, 0, 0, 0]]


def obs(grid: list[list[int]] = GRID, actions: tuple[int, ...] = (1, 6)) -> Observation:
    return Observation.build(
        game_id="g",
        frame=[grid],
        state="NOT_FINISHED",
        levels_completed=0,
        win_levels=2,
        available_actions=actions,
    )


class Scripted:
    def __init__(self, *replies: str) -> None:
        self.replies, self.seen = list(replies), []

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.seen.append(list(messages))
        return self.replies.pop(0)


def block(code: str) -> str:
    return f"thinking\n```python\n{code}\n```"


def test_ascii_and_changes() -> None:
    assert ascii_grid([[0, 10], [15, 1]]) == "0A\nF1"
    assert changes(GRID, [[0] * 4] * 4) == (2, [(1, 1, 7, 0), (2, 3, 3, 0)])
    assert changes(None, GRID) == (0, [])


def test_components_and_click_candidates() -> None:
    found = components(GRID)
    assert {(c.color, c.size) for c in found} == {(7, 1), (3, 1)}
    assert click_candidates(GRID) == [(1, 1), (3, 2)]
    assert zoom(GRID, 1, 1, 2, 3) == "700\n003"


def test_repl_persists_state_and_reports_errors() -> None:
    repl = Repl(timeout=1)
    assert repl.run("x = 41\nprint(x + 1)") == "42"
    assert repl.run("print(x)") == "41"
    assert "ZeroDivisionError" in repl.run("1 / 0")
    assert "TimeoutError" in repl.run("while True: pass")


def test_evict_keeps_system_and_starts_on_user() -> None:
    log = [{"role": "system", "content": "s"}]
    for i in range(6):
        log.append({"role": "user" if i % 2 == 0 else "assistant", "content": "x" * 10})
    kept = evict(log, budget_chars=35)
    assert kept[0]["role"] == "system" and kept[1]["role"] == "user"
    assert len(kept) < len(log)


def test_agent_acts_after_inspecting() -> None:
    chat = Scripted(block("print(len(components(grid)))"), block("act(6, x=1, y=1)"))
    action = DuckAgent(chat).decide(obs())
    assert (action.action_id, action.data) == (6, {"x": 1, "y": 1})
    assert "2" in chat.seen[1][-1]["content"]  # REPL output fed back


def test_illegal_action_is_rejected_then_fixed() -> None:
    chat = Scripted(block("act(3)"), block("act(1)"))
    assert DuckAgent(chat).decide(obs()).action_id == 1
    assert "not in [1, 6]" in chat.seen[1][-1]["content"]


def test_falls_back_to_legal_action_without_act() -> None:
    agent = DuckAgent(Scripted("no code", "no code"), max_inner=2)
    action = agent.decide(obs(actions=(2,)))
    assert action.action_id == 2 and agent.fallbacks == 1


def test_prompt_has_status_not_grid_and_repl_has_diff(capfd) -> None:  # type: ignore[no-untyped-def]
    chat = Scripted(block("act(1)"), block("diff(prev_grid, grid)\nact(1)"))
    agent = DuckAgent(chat)
    agent.decide(obs())
    agent.decide(obs([[0] * 4] * 4))
    status = chat.seen[1][-1]["content"]
    assert "AVAILABLE_ACTIONS" in status and "GRID" not in status
    assert "7->0: 1 cells, rows 1-1, cols 1-1" in agent.repl.run(
        "diff(prev_grid, grid)"
    )


def test_run_episode_stops_on_win() -> None:
    from arc_agi_3.duck.local import run_episode

    class Env:
        def __init__(self) -> None:
            self.steps = 0

        def reset(self) -> Observation:
            return obs()

        def step(self, action: object) -> Observation:
            self.steps += 1
            won = self.steps == 2
            return Observation.build(
                game_id="g",
                frame=[GRID],
                state="WIN" if won else "NOT_FINISHED",
                levels_completed=self.steps,
                win_levels=2,
                available_actions=(1,),
            )

    agent = DuckAgent(Scripted(*[block("act(1)")] * 5))
    assert run_episode(Env(), agent, max_steps=5) == 2  # type: ignore[arg-type]


def test_empty_frame_reuses_previous_grid_or_resets() -> None:
    empty = Observation.build(
        game_id="g", frame=[], state="NOT_FINISHED", levels_completed=0,
        win_levels=1, available_actions=(1,),
    )  # fmt: skip
    agent = DuckAgent(Scripted(block("act(1)")))
    assert agent.decide(empty).action_id == 1  # nothing to show: legal action
    agent.decide(obs())
    chat = Scripted(block("act(1)"))
    agent.chat = chat
    assert agent.decide(empty).action_id == 1 and chat.seen  # reused prior grid


def test_describe_diff_groups_by_color_change() -> None:
    from arc_agi_3.duck.render import describe_diff

    moved = [[0, 7, 0, 0], [0, 0, 0, 0], [0, 0, 0, 3], [0, 0, 0, 0]]
    text = describe_diff(GRID, moved)
    assert "0->7: 1 cells, rows 0-0, cols 1-1" in text
    assert "7->0: 1 cells, rows 1-1, cols 1-1" in text
    assert describe_diff(GRID, GRID) == "no cells changed"
    assert describe_diff(None, GRID) == "no previous frame"
