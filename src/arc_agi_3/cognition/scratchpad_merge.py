"""Pure merge helpers for verified facts and the planned next experiment."""

from arc_agi_3.contracts.scratchpad import (
    NextTest,
    PlanRevision,
    ScratchpadUpdates,
    VerifiedFact,
    WorkingScratchpad,
)


def merge_facts(
    before: WorkingScratchpad, updates: ScratchpadUpdates, valid_evidence: set[str]
) -> tuple[VerifiedFact, ...]:
    """Facts must cite environment/verifier/tool evidence, never bare reasoning."""
    facts = {item.fact_id: item for item in before.verified_facts}
    for fact in updates.add_verified_fact:
        if not set(fact.evidence_refs) <= valid_evidence:
            raise ValueError(
                "verified scratchpad facts require known observational evidence refs"
            )
        if fact.fact_id in fact.supersedes:
            raise ValueError("a scratchpad fact cannot supersede itself")
        if len(set(fact.supersedes)) != len(fact.supersedes):
            raise ValueError("superseded scratchpad facts must be unique")
        superseded = [facts[key] for key in fact.supersedes if key in facts]
        if len(superseded) != len(fact.supersedes):
            raise ValueError("superseded scratchpad facts must exist")
        inherited = {
            reference for item in superseded for reference in item.evidence_refs
        }
        if not inherited <= set(fact.evidence_refs):
            raise ValueError("consolidated facts must retain source evidence")
        prior = facts.get(fact.fact_id)
        expected = 1 if prior is None else prior.version + 1
        if fact.version != expected:
            raise ValueError(f"scratchpad fact requires version {expected}")
        for item in superseded:
            del facts[item.fact_id]
        facts[fact.fact_id] = fact
    return tuple(facts[key] for key in sorted(facts))


def merge_next_test(
    before: WorkingScratchpad, updates: ScratchpadUpdates
) -> tuple[NextTest | None, tuple[PlanRevision, ...]]:
    """Replace or clear the planned experiment, recording any revision."""
    revised = (
        None if updates.clear_next_test else updates.set_next_test or before.next_test
    )
    revisions = before.plan_revisions
    if before.next_test is not None and revised != before.next_test:
        revision = PlanRevision(
            scratchpad_version=before.version + 1,
            previous=before.next_test,
            revised=revised,
            reason=updates.next_test_revision_reason,
        )
        revisions = (*revisions, revision)[-4:]
    return revised, revisions
