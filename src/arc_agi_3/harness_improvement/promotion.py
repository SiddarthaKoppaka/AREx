"""A held-out promotion criterion over an existing ReportComparison
(PR-00). This is a recommendation, not a promotion: "no automatic
production promotion" - a human still reviews before anything merges.
"""

from arc_agi_3.contracts.report_comparison import ReportComparison

from .contracts import PromotionDecision

HigherIsBetter = bool


def decide_promotion(
    candidate_id: str,
    comparison: ReportComparison,
    metric: str,
    *,
    higher_is_better: HigherIsBetter = False,
) -> PromotionDecision:
    delta = next((d for d in comparison.deltas if d.name == metric), None)
    if delta is None:
        return PromotionDecision(
            candidate_id=candidate_id,
            metric=metric,
            accepted=False,
            reason=f"metric {metric!r} is not in this comparison",
            control_value=None,
            candidate_value=None,
        )
    if delta.control is None or delta.candidate is None:
        return PromotionDecision(
            candidate_id=candidate_id,
            metric=metric,
            accepted=False,
            reason=f"metric {metric!r} is missing on at least one side",
            control_value=delta.control,
            candidate_value=delta.candidate,
        )
    improved = (
        delta.candidate > delta.control
        if higher_is_better
        else delta.candidate < delta.control
    )
    unchanged = delta.candidate == delta.control
    accepted = improved or unchanged
    reason = (
        f"{metric} went from {delta.control} to {delta.candidate} "
        f"({'improved' if improved else 'unchanged' if unchanged else 'regressed'})"
    )
    return PromotionDecision(
        candidate_id=candidate_id,
        metric=metric,
        accepted=accepted,
        reason=reason,
        control_value=delta.control,
        candidate_value=delta.candidate,
    )
