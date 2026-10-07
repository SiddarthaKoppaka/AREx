"""Canonical argument contract for every LM-invokable tool.

Single source of truth: the model prompt renders these exact schemas
(`adapters.tool_schemas`) and the generation loop validates every
`ToolRequest` against them before a decision is accepted
(`adapters.tool_contract_check`), so the model-facing contract and the
execution-time contract (`runtime.tool_handlers`) can never drift apart.
Adding a tool means adding one entry here; nowhere else defines the shape.
"""

from arc_agi_3.contracts.artifacts import ArchiveArtifactRequest, ReadArtifactRequest
from arc_agi_3.contracts.base import Contract
from arc_agi_3.contracts.evidence import EvidenceQuery, FrameRegionRequest
from arc_agi_3.contracts.retrieval import RetrievalQuery
from arc_agi_3.contracts.retrodiction import RetrodictionRequest
from arc_agi_3.exploration.contracts import ExplorationOutcome, ExplorationRequest
from arc_agi_3.planning.contracts import SearchRequest
from arc_agi_3.world_model.contracts import SimulationRequest

from .contracts.specialist_tools import (
    SpecialistCompareRequest,
    SpecialistConsultRequest,
)

TOOL_ARGUMENTS: dict[str, type[Contract]] = {
    "archive_artifact": ArchiveArtifactRequest,
    "retrieve_artifact": ReadArtifactRequest,
    "retrieve_evidence": EvidenceQuery,
    "inspect_frame_region": FrameRegionRequest,
    "check_prediction_history": RetrodictionRequest,
    "retrieve_events": RetrievalQuery,
    "simulate_world_model": SimulationRequest,
    "search_world_model": SearchRequest,
    "evaluate_exploration": ExplorationRequest,
    "record_exploration_outcome": ExplorationOutcome,
    "consult_specialist": SpecialistConsultRequest,
    "compare_specialist_reports": SpecialistCompareRequest,
}
