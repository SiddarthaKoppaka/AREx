# PR-03 — Executable model contract and secure executor

> **Status:** Implemented. See the isolation-approach decision made with
> the user before writing any code (below) — this was the one point
> flagged as important enough to stop for.

## Isolation investigation (the decision point)

handoff.md's gate: "if robust isolation is unavailable, disable executable
code and document a constrained DSL/interpreter fallback. A restricted
`exec()` globals dictionary is not an acceptable security boundary."

Tested empirically on this machine (Darwin/arm64) rather than assumed:

- `RLIMIT_CPU` — **works**: a busy-loop was killed by SIGXCPU at exactly
  the configured second.
- `RLIMIT_FSIZE` — **works**: a runaway file write was truncated/errored
  at exactly the configured byte limit.
- `RLIMIT_AS` (memory ceiling) — **does not work on macOS**:
  `setrlimit` always raises `ValueError: current limit exceeds maximum
  limit` regardless of the requested value — a known Darwin/XNU
  limitation. Generally works on Linux (Kaggle's actual runtime), not
  independently verified here.
- A wall-clock `sleep` loop is **not** caught by `RLIMIT_CPU` at all
  (confirmed) — needs a separate, external wall-clock watchdog.
- True network isolation isn't achievable without root/containers on
  either platform; import-blocking is the only available mitigation and
  is not a real network boundary.

User decision: proceed with **best-effort process isolation** — real,
OS-enforced limits where they exist, explicitly documented as layered
defense against accidental/careless unsafe code, not a claim of an
airtight sandbox against deliberate escape. (Options considered and not
taken: a constrained DSL instead of Python, or deferring PR-03+ entirely.)

## What was built

`src/arc_agi_3/executable_model/`:
- `contracts.py` — `ExecutableModelCandidate` (handoff.md §5.1 fields,
  `parent_version < version` enforced) and `ExecutionProfile` (per-call
  resource ceilings).
- `execution.py` — `ExecutionResult`, one of 8 typed statuses; a
  candidate call never raises past the sandbox boundary.
- `interface_check.py` — static AST gate (syntax check; exactly the 3
  required functions `encode_observation`/`transition`/`terminal_or_goal`
  with correct arity; only function defs/allowlisted imports/constants at
  top level; no classes, decorators, `global`/`nonlocal`, or calls to
  `eval`/`exec`/`compile`/`open`/`__import__` anywhere in the body). Pure
  inspection, never executes the source. A cheap first filter, not the
  real boundary.
- `restricted_builtins.py` — minimal builtins + import allowlist (`math`
  only) for the exec namespace.
- `sandbox_worker.py` — the actual untrusted-process entry point: sets
  `RLIMIT_CPU`/`RLIMIT_FSIZE`/`RLIMIT_NOFILE` (best-effort `RLIMIT_AS`),
  execs candidate source in the restricted namespace, reports every
  failure as a typed JSON response over stdout.
- `memory_watchdog.py` — polls the child's RSS via `ps` (portable,
  no extra dependency) and kills it if it exceeds budget — the memory
  backstop where `RLIMIT_AS` can't be set.
- `sandbox_executor.py` — parent-side orchestration: fresh subprocess per
  call, wall-clock timeout via `subprocess.communicate(timeout=...)`,
  the memory watchdog thread, typed result parsing.
- `hashing.py` — `source_sha256` for immutable versioning.

`config.py` — added `AblationConfig.executable_models: bool = True`
(ADR-V2-002: alongside, not replacing, declarative `world_model/`).

## Gate verification (handoff.md's exact required adversarial cases)

`tests/test_sandbox_executor.py` — one test per required case: a correct
deterministic program; an infinite busy loop (killed, not hung); a
wall-clock sleep loop (killed via import-block in this instance, since
`time` isn't in the allowlist — the raw timeout mechanism was separately
verified empirically above); huge output (rejected, never streamed back);
a filesystem write attempt (blocked); a network/import attempt (blocked);
global mutation (contained — state does not leak between separate calls,
each a fresh subprocess); a crash (typed `runtime_error`, not a crash of
the harness process).

`tests/test_interface_check.py` (11 cases) and
`tests/test_executable_model_candidate.py` (3 cases) cover the static
gate and contract lineage rules.

## Note: golden fixture regenerated

Adding `AblationConfig.executable_models` changed every event hash in
`tests/golden/fake_run_signature.json` (a new config field propagates
through the whole hash chain). Regenerated deliberately and verified —
this is exactly the "intentionally reviewed compatibility update" case
handoff.md's own golden-file note anticipates, not an accidental
regression.

## Explicitly not done in this PR

Not wired into the live decision loop (no new LM-facing tool yet) — that
integration belongs to PR-04/PR-06, which actually need to call this
executor against real replay cases. No pixel/frame support (per
handoff.md: "Test on toy symbolic states before adding pixel/frame
support").

## Validation

Full suite (354 tests), `ruff check`, `ruff format --check`, `mypy` all
pass. No change to action authorization or evaluation behavior.
