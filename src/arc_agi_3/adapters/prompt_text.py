"""Fixed decision instructions; kept short because they repeat every turn."""

CONTRACT = (
    "You are the agent. Return exactly one CognitiveDecision as raw JSON "
    "matching the provided schema. Do not use Markdown code fences, XML tags, "
    "commentary, or text outside the JSON object. "
    "Own all semantic interpretation, intent, strategy, and action choice. "
    "Do not provide private chain-of-thought or hidden reasoning. Use assessment, "
    "intent, considered_options, decision_summary, observation_summary, and "
    "expected_result only for concise public decision rationale and summaries. "
    "Structural invariants: execute mode must set exactly one of action or "
    "action_chunk; every other mode must set both to null. Recover mode must set "
    "recovery; every other mode must set recovery to null. Update working memory "
    "only through scratchpad_updates; capabilities are harness-owned. "
    "scratchpad next_test is your planned experiment, not a lock: revise or clear "
    "it with a reason whenever your plan changes.\n"
)

EPISTEMICS = (
    "EPISTEMICS: Keep these distinct. Observation = mechanical evidence below. "
    "Hypothesis = your interpretation (hypothesis_proposals, with your own "
    "probability). Prediction = expected_outcome with observable claims "
    "(changed-cell range, cell_values, component translations, no_op, state, "
    "levels) plus hypothesis_ids. Verified fact = only with environment or "
    "verifier event refs, never your own decisions. Initial readings of objects, "
    "players, or goals are hypotheses. An unexpected transition may falsify your "
    "interpretation, not only reveal an obstacle; repeated mismatches mean "
    "re-examine assumptions, and any conclusion may be reopened. After a "
    "mismatch, revise explicitly with hypothesis_updates (set confidence, cite "
    "the verification event in evidence_refs). The harness never changes your "
    "confidence. Frame investigative actions with experiment and a checkable "
    "expected_outcome; prefer experiments whose outcomes differ across live "
    "hypotheses. Check ACTION_EVIDENCE before repeating an action; if you repeat "
    "an equivalent experiment, state experiment.repeat_justification.\n"
)

TOOLS = (
    "Optional task DAG via task_updates with success criteria and compact "
    "handoffs. Tools: retrieve_evidence (event_ids, purpose, token_budget, view: "
    "metadata, summary, transition_delta, rle_frame, region, full); "
    "inspect_frame_region; check_prediction_history (prediction, optional "
    "action, limit) to test a prediction against past transitions; "
    "archive_artifact / retrieve_artifact; retrieve_events. Older evidence is "
    "omitted from context but always retrievable by event ID. "
    "CURRENT_OBSERVATION omits the exact grid by default: use retrieve_evidence "
    "with event_ids=[current_observation_event_id] (view=full or rle_frame) or "
    "inspect_frame_region for a crop whenever the mechanical summary is not "
    "enough.\n"
)
