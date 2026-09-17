"""Stable serialized enum values."""

from enum import StrEnum


class EnvironmentState(StrEnum):
    NOT_PLAYED = "NOT_PLAYED"
    NOT_FINISHED = "NOT_FINISHED"
    WIN = "WIN"
    GAME_OVER = "GAME_OVER"


class DecisionMode(StrEnum):
    EXPLORE = "explore"
    INVESTIGATE = "investigate"
    SIMULATE = "simulate"
    PLAN = "plan"
    EXECUTE = "execute"
    RECOVER = "recover"
    STOP = "stop"


class Resource(StrEnum):
    ACTIONS = "actions"
    MODEL_CALLS = "model_calls"
    INPUT_TOKENS = "input_tokens"
    OUTPUT_TOKENS = "output_tokens"
    SEARCH_NODES = "search_nodes"
    SIMULATIONS = "simulations"
    WALL_TIME_MS = "wall_time_ms"
    MEMORY_BYTES = "memory_bytes"


class EventType(StrEnum):
    RUN_STARTED = "run_started"
    OBSERVATION = "observation"
    MODEL_DECISION = "model_decision"
    TOOL_REQUEST = "tool_request"
    TOOL_RESULT = "tool_result"
    HYPOTHESIS = "hypothesis"
    BELIEF_UPDATE = "belief_update"
    TASK_UPDATE = "task_update"
    WORLD_MODEL = "world_model"
    CHUNK_STARTED = "chunk_started"
    CHUNK_FINISHED = "chunk_finished"
    INTERRUPT = "interrupt"
    RECOVERY = "recovery"
    BRANCH_FROZEN = "branch_frozen"
    BRANCH_FORKED = "branch_forked"
    BUDGET = "budget"
    ACTION = "action"
    TRANSITION = "transition"
    VERIFICATION = "verification"
    CHECKPOINT = "checkpoint"
    FAILURE = "failure"
    RUN_FINISHED = "run_finished"
    EVALUATION = "evaluation"


class BeliefOperation(StrEnum):
    SUPPORT = "support"
    WEAKEN = "weaken"
    CONTRADICT = "contradict"
    REFINE = "refine"
    BRANCH = "branch"
    MERGE = "merge"
    SUSPEND = "suspend"
    REJECT = "reject"


class TaskStatus(StrEnum):
    OPEN = "open"
    BLOCKED = "blocked"
    COMPLETE = "complete"
    ABANDONED = "abandoned"


class SearchMethod(StrEnum):
    BFS = "bfs"
    ASTAR = "astar"


class ExecutionStatus(StrEnum):
    COMPLETE = "complete"
    SOFT_INTERRUPT = "soft_interrupt"
    HARD_STOP = "hard_stop"


class RecoveryOperation(StrEnum):
    FORK_CHECKPOINT = "fork_checkpoint"
    ABANDON_BRANCH = "abandon_branch"
