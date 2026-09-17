"""Pure arithmetic for the Miro exploration-value formulation."""

from .contracts import ExplorationRequest, ExplorationResult, ProbeScore
from .history import ExplorationHistory


def evaluate_exploration(
    request: ExplorationRequest,
    history: ExplorationHistory | None = None,
) -> ExplorationResult:
    history = history or ExplorationHistory()
    scored = []
    for probe in request.probes:
        visits = history.visits(probe.candidate_state_hash)
        benefit = (
            request.weights.information * probe.drig
            + request.weights.reward_progress * probe.expected_reward_progress
            + request.weights.hypothesis_discrimination
            * probe.hypothesis_discrimination
        )
        cost = (
            probe.action_cost
            + probe.inference_cost
            + probe.risk_cost
            + probe.repeat_cost
            + visits * probe.repeat_cost_per_visit
        )
        scored.append(
            ProbeScore(
                probe_id=probe.probe_id,
                value=benefit - cost,
                prior_visits=visits,
                novel_state=probe.candidate_state_hash is not None and visits == 0,
            )
        )
    maximum = max(item.value for item in scored)
    maximizing = tuple(item.probe_id for item in scored if item.value == maximum)
    recent = request.recent_marginal_values[-request.saturation_window :]
    saturated = len(recent) == request.saturation_window and all(
        value <= request.saturation_threshold for value in recent
    )
    return ExplorationResult(
        request_id=request.request_id,
        scores=tuple(scored),
        maximum_value=maximum,
        maximizing_probe_ids=maximizing,
        exploration_dominated=(
            request.executable_plan_utility >= maximum + request.safety_margin
        ),
        saturated=saturated,
    )
