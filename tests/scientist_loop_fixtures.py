"""Shared fixture-LM policies for the PR-06 scientist-loop demonstration.

SYNTHETIC: a scripted model, not a real LM - see test_scientist_loop_*.py.
"""

from arc_agi_3.contracts.cognition import Hypothesis
from arc_agi_3.contracts.decision import AgentContext, CognitiveDecision
from arc_agi_3.contracts.enums import DecisionMode, EventType
from arc_agi_3.contracts.observation import Action
from arc_agi_3.contracts.prediction import (
    ExpectedOutcome,
    Experiment,
    TranslationExpectation,
)
from arc_agi_3.contracts.scratchpad import ScratchpadUpdates, VerifiedFact
from tests.executable_model_fixtures import CORRECT_SOURCE

VERIFIED_FACT_TEXT = "action 1 shifts the marker by exactly one column"

WRONG_SHIFT_SOURCE = CORRECT_SOURCE.replace(
    "return {'column': min(6, state['column'] + 1)}",
    "return {'column': min(6, state['column'] + 2)}",
)


def propose_and_test(context: AgentContext) -> CognitiveDecision:
    h_correct = Hypothesis(
        hypothesis_id="h-correct",
        version=1,
        claim="action 1 shifts by 1 column",
        probability=0.5,
    )
    h_wrong = Hypothesis(
        hypothesis_id="h-wrong",
        version=1,
        claim="action 1 shifts by 2 columns",
        probability=0.5,
    )
    return CognitiveDecision(
        assessment="two competing hypotheses about the shift distance",
        intent="run a discriminating action",
        mode=DecisionMode.EXECUTE,
        action=Action(action_id=1),
        hypothesis_proposals=(h_correct, h_wrong),
        experiment=Experiment(
            experiment_id="e1",
            question="by how much does action 1 shift?",
            hypothesis_ids=("h-correct", "h-wrong"),
        ),
        expected_outcome=ExpectedOutcome(
            hypothesis_ids=("h-correct",),
            translations=(TranslationExpectation(value=3, d_row=0, d_col=1),),
        ),
    )


def record_fact(context: AgentContext) -> CognitiveDecision:
    verification_id = next(
        e.event_id
        for e in context.recent_events
        if e.event_type is EventType.VERIFICATION
    )
    fact = VerifiedFact(
        fact_id="f1",
        version=1,
        fact=VERIFIED_FACT_TEXT,
        confidence=0.95,
        evidence_refs=(verification_id,),
    )
    return CognitiveDecision(
        assessment="record the confirmed rule",
        intent="advance",
        mode=DecisionMode.INVESTIGATE,
        scratchpad_updates=ScratchpadUpdates(add_verified_fact=(fact,)),
    )


def use_prior_knowledge(context: AgentContext) -> CognitiveDecision:
    assert context.working_scratchpad is not None
    verified = any(
        f.fact == VERIFIED_FACT_TEXT for f in context.working_scratchpad.verified_facts
    )
    assert verified, "a later decision must see the verified fact, not re-derive it"
    return CognitiveDecision(
        assessment="reuse the known rule without re-testing it",
        intent="stop",
        mode=DecisionMode.STOP,
    )
