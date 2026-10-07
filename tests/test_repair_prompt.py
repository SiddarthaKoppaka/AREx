"""The repair prompt is architecturally separate from the cognitive prompt."""

from arc_agi_3.adapters.generation_attempts import record_attempt
from arc_agi_3.adapters.repair_prompt import build_repair_prompt
from arc_agi_3.adapters.structured_model import StructuredModelAdapter
from arc_agi_3.testing import successful_script
from tests.generation_fixtures import ScriptedBackend, context

VALID = successful_script()[0].model_dump_json()


def test_repair_prompt_excludes_frame_and_working_history() -> None:
    malformed = '{"frame": [[1,2,3]], "mode": "execute"'
    prompt = build_repair_prompt(
        "malformed_json",
        "unexpected end of input",
        malformed,
        target_type="CognitiveDecision",
    )
    assert "CURRENT_OBSERVATION" not in prompt
    assert "WORKING_SCRATCHPAD" not in prompt
    assert "RECENT_TURNS" not in prompt
    assert "RELEVANT_EPISODES" not in prompt
    assert malformed in prompt


def test_repair_prompt_is_small_relative_to_a_cognitive_prompt() -> None:
    from arc_agi_3.adapters.prompts import decision_prompt

    cognitive = decision_prompt(context())
    repair = build_repair_prompt(
        "no_json_found",
        "No valid JSON object",
        "prose only",
        target_type="CognitiveDecision",
    )
    assert len(repair) < len(cognitive)
    assert len(repair) < 2000


def test_repair_prompt_echoes_malformed_output_to_preserve_intent() -> None:
    malformed = '{"assessment": "advance toward the target", "mode": "execute"'
    prompt = build_repair_prompt(
        "malformed_json", "err", malformed, target_type="CognitiveDecision"
    )
    assert "advance toward the target" in prompt
    assert "Preserve the original semantic intent" in prompt


def test_oversized_malformed_output_is_truncated_not_unbounded() -> None:
    huge = "x" * 10_000
    prompt = build_repair_prompt(
        "malformed_json", "err", huge, target_type="CognitiveDecision"
    )
    assert len(prompt) < 5_000
    assert "[truncated for repair]" in prompt


def test_repair_prompt_names_the_actual_target_type_not_cognitive_decision() -> None:
    prompt = build_repair_prompt(
        "malformed_json", "err", "{bad", target_type="SpecialistReport"
    )
    assert "Repair this output into a valid SpecialistReport." in prompt
    assert "CognitiveDecision" not in prompt


def test_the_live_repair_call_never_carries_the_full_context() -> None:
    backend = ScriptedBackend(["not json", VALID])
    StructuredModelAdapter(backend).decide(context())
    assert "CURRENT_OBSERVATION" in backend.prompts[0]
    assert "CURRENT_OBSERVATION" not in backend.prompts[1]


def test_invalid_output_preview_is_bounded_at_each_end() -> None:
    text = "y" * 900
    attempt = record_attempt(
        1, text, "malformed_json", "err", False, persist_invalid_output=True
    )
    assert len(attempt.output_preview or "") == 300
    assert len(attempt.output_preview_tail or "") == 300


def test_preview_is_absent_by_default_privacy_preserving() -> None:
    attempt = record_attempt(1, "secret text", "malformed_json", "err", False)
    assert attempt.output_preview is None
    assert attempt.output_preview_tail is None
