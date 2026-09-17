"""Independent deterministic metrics computed only from immutable events."""

from arc_agi_3.config import EvaluatorConfig
from arc_agi_3.contracts.enums import EnvironmentState, EventType
from arc_agi_3.contracts.evaluation import EvaluationMetrics
from arc_agi_3.contracts.events import EventEnvelope
from arc_agi_3.evaluation_counts import event_research_counts
from arc_agi_3.rhae import environment_score, level_action_counts, level_score

__all__ = ["environment_score", "evaluate", "level_score"]


def evaluate(events: list[EventEnvelope], config: EvaluatorConfig) -> EvaluationMetrics:
    observations = [
        event for event in events if event.event_type is EventType.OBSERVATION
    ]
    final = observations[-1].payload if observations else {}
    actions = sum(event.event_type is EventType.ACTION for event in events)
    decisions = [
        event for event in events if event.event_type is EventType.MODEL_DECISION
    ]
    usage = [
        value
        for event in decisions
        if isinstance((value := event.payload.get("usage", {})), dict)
    ]
    verification_failures = sum(
        event.event_type is EventType.VERIFICATION
        and not bool(event.payload.get("passed", False))
        for event in events
    )
    scores: tuple[float, ...] = ()
    result: float | None = None
    if config.human_action_baselines:
        counts = level_action_counts(events)
        computed = [
            level_score(human, counts[index], config.level_score_cap)
            if index < len(counts)
            else 0.0
            for index, human in enumerate(config.human_action_baselines)
        ]
        scores = tuple(computed)
        result = environment_score(scores, len(counts))
    research = event_research_counts(events)
    return EvaluationMetrics(
        succeeded=final.get("state") == EnvironmentState.WIN,
        levels_completed=(
            value if isinstance((value := final.get("levels_completed", 0)), int) else 0
        ),
        environment_actions=actions,
        model_calls=len(decisions),
        input_tokens=sum(
            value if isinstance((value := item.get("input_tokens", 0)), int) else 0
            for item in usage
        ),
        output_tokens=sum(
            value if isinstance((value := item.get("output_tokens", 0)), int) else 0
            for item in usage
        ),
        verification_failures=verification_failures,
        failures=sum(event.event_type is EventType.FAILURE for event in events),
        checkpoints=sum(event.event_type is EventType.CHECKPOINT for event in events),
        search_node_expansions=research["search_node_expansions"],
        simulations=research["simulations"],
        wall_time_ms=research["wall_time_ms"],
        prediction_mismatches=research["prediction_mismatches"],
        soft_interrupts=research["soft_interrupts"],
        hard_stops=research["hard_stops"],
        recovery_attempts=research["recovery_attempts"],
        recovery_failures=research["recovery_failures"],
        branch_forks=research["branch_forks"],
        repeated_actions=research["repeated_actions"],
        world_model_versions=research["world_model_versions"],
        rhae_level_scores=scores,
        rhae_environment_score=result,
    )
