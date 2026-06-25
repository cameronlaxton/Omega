# Omega — Project Guidelines (cross-agent entrypoint)

**This is the authoritative cross-agent instruction file.** Every agent working in this repo
(Claude, ChatGPT, Codex, API agents, etc.) reads `AGENTS.md` first. The root `CLAUDE.md` is a shim
that points here and carries no independent rules.

### Canonical references (read before acting)

- **[`prompts/reference/output_modes.md`](prompts/reference/output_modes.md)** — output-mode
  semantics (`RESEARCH_CANDIDATE` vs `ACTIONABLE`), downgrade discipline, and the engine-execution
  rule. **Single source of truth; do not restate it elsewhere.**
- [`prompts/reference/presentation_contract.md`](prompts/reference/presentation_contract.md) —
  narrative-first user-facing response shape for both `ACTIONABLE` and `RESEARCH_CANDIDATE` modes.
  Owns the *shape* of the reply; authorization stays in `output_modes.md`.
- [`prompts/reference/engine_output_validation.md`](prompts/reference/engine_output_validation.md) —
  post-`analyze()` nullability / null-data-audit procedure.
- [`prompts/reference/markov_evidence_vocab.md`](prompts/reference/markov_evidence_vocab.md) —
  approved Markov `signal_type` vocabulary and the ±15% cap.
- [`OMEGA_DATA_SOURCES.md`](OMEGA_DATA_SOURCES.md) — data sourcing, fallbacks, freshness rules.
- [`docs/historical_calibration_backfill.md`](docs/historical_calibration_backfill.md) — historical
  replay → calibration backfill runbook (ingest → replay → fit → parity → promote).
- [`src/omega/core/calibration/CLAUDE.md`](src/omega/core/calibration/CLAUDE.md) — calibration method
  catalog and promotion rules. Supported methods: `none`, `shrinkage`, `cap`, `combined`, `isotonic`,
  and **`market_aware`** (issue #28 WS4) — blends the model toward the closing-line implied probability
  scaled by a coarse liquidity-deference factor; requires a `market_prob` at apply time and is fail-safe
  identity without one. New methods require evidence of held-out improvement (see that file's Method rules).
- [`docs/issue28_clv_loop.md`](docs/issue28_clv_loop.md) — the continuous, operator-gated CLV evidence
  loop (score by closing-line value → fit reliability + lifecycle recommendations → review → graduate).

### Output mode ⊥ engine execution (summary — full rule in output_modes.md)

`RESEARCH_CANDIDATE` is an output-authorization mode, not an execution mode. If the engine is
available, `analyze()` still runs, a `sandbox-` trace_id is still minted, and the trace still
persists to `var/omega_traces.db` — only the user-facing betting numbers are withheld/downgraded.
**Never skip the engine just because output is research-only**; doing so starves the calibration
loop. See [`prompts/reference/output_modes.md`](prompts/reference/output_modes.md).

## Omega operating model

Omega is a sports analytics system with bounded autonomy.

The LLM may control:
- reasoning
- planning
- routing
- evidence arbitration
- explanation
- downgrade decisions

The deterministic engine owns:
- simulation
- probability calibration
- edge calculation
- staking
- backtesting
- grading

Do not move deterministic responsibilities into LLM logic. This restriction applies to formal Omega outputs:
- Bet Cards / BetSlips / EdgeDetail rows
- model probabilities, calibrated probabilities, fair-price / no-vig price
- EV% / edge% / expected value calculations
- Kelly fractions, recommended units, staking sizes
- confidence tiers (A / B / C / Pass)
- trace_ids (always begin with `sandbox-` and are minted by the engine)

**Hard rule:** The LLM is forbidden from generating any of the above values via text. They must come from Python execution through the local MCP server or `omega.core.contracts.service.analyze`. There is no "estimated", "rough", "ballpark", `[LLM-ESTIMATED]`, or "estimated lean" mode for these fields. If the engine is unavailable, the response is qualitative-only: matchup narrative, news, recent form, listed sportsbook lines from a cited source — never a Bet Card with placeholder numbers. The full output-mode contract (including the engine-runs-in-RESEARCH_CANDIDATE rule) lives in [`prompts/reference/output_modes.md`](prompts/reference/output_modes.md).

The LLM may still perform a best-effort exploratory market scan with public web data, but any candidate it surfaces is labeled as a **research-only lean** or **missing-data watchlist** item with NO edge%, EV%, Kelly, units, confidence tier, or trace_id. The previous "estimated lean" label is retired.

The master runtime instruction for any LLM acting as the Omega agent (any LLM front-end: Claude.ai, ChatGPT, Codex, API agent) is [`prompts/system_prompt.txt`](prompts/system_prompt.txt). That file is authoritative for agent behavior. (The legacy `OMEGA_HANDBOOK.md` and `OMEGA_RUN_RECIPE.md` have been retired to `archive/historical/` — non-authoritative.)

**Deployment-specific instructions:** Use **one** instruction set based on deployment:
- **Any LLM agent / API (no local access)** → [`prompts/system_prompt.txt`](prompts/system_prompt.txt)
- **Cowork Project (local repo access)** → [`OMEGA_COWORK.md`](OMEGA_COWORK.md)

Do NOT combine both files into a single Project; they assume different execution contexts (sandbox vs. local VM, manual vs. automated pipelines).

## Phase 6 objective

Phase 6 delivers:
1. trace persistence
2. historical replay
3. calibration learning

Implement these incrementally, with minimal drift from the current architecture.

## Source-of-truth evaluation model

Omega has two evaluation planes.

### Quant plane
Purpose:
- forecast quality
- calibration quality
- edge quality
- staking quality

This is the benchmark source of truth.

Inputs:
- frozen historical artifacts
- frozen odds snapshots
- frozen normalized contexts
- known actual outcomes

Path:
- normalize
- simulate
- calibrate
- edge
- stake
- grade

### Replay plane
Purpose:
- routing quality
- evidence selection quality
- downgrade discipline
- trace completeness
- refusal discipline

This is sampled audit only. It is not the default benchmark path.

Inputs:
- historical knowable-at-the-time evidence bundles

Path:
- orchestrator replay with live fetching disabled

## Hard rules

- Do not duplicate edge, calibration, staking, or grading logic in a second path
- Do not make orchestrator replay the default benchmark path
- Do not add network calls to backtest or replay fixtures
- Do not invent new top-level architecture unless current modules cannot support the requirement
- Prefer extending existing packages over introducing parallel systems
- Every persistence format must be versioned
- Every replay path must be reproducible
- Every calibration fit must be attributable to a specific dataset and profile version
- Do not invent progress, files, integrations, benchmarks, or capabilities that do not exist
- Ground recommendations in the actual repo state before proposing structural change

## Batch analysis rule

When a session requires more than 3 analyses (daily slates, prop sweeps, etc.):

- **MCP available:** use `omega_run_batch` — one call handles odds resolution (with prop_type fallback chain), seed derivation, gate enforcement, and export-block writing to `var/inbox/traces/`. Do not loop individual `omega_analyze_prop` / `omega_analyze_game` calls and do not write a Python script.
- **MCP unavailable:** a batch Python script is the authorized fallback. It **must** call `cowork_preflight.run_formal_output_gate()` before any `analyze()` call, derive every seed deterministically as `int.from_bytes(hashlib.sha256(f"{prompt}|{date}".encode("utf-8")).digest()[:4], "big")`, include at least one `EvidenceSignal` per trace (or document why evidence is empty), and must not hardcode `trace_quality.aggregate_quality` or mutate shared state (e.g. deleting the odds cache) as a side effect.

Scripts that bypass the gate or hardcode quality scores produce uncalibrated traces and contaminate the calibration loop.

## Anti-overengineering rule

Before adding a model, service, abstraction, or package, ask:

Does this directly prevent bad sim inputs, bad recommendations, bad replay, or bad backtests?

If no, defer it.

## Required invariants

- the same frozen quant artifact must always produce the same simulation seed
- backtest and production paths must share the same calibration selection policy
- traces must be persistable without depending on request/response wrapper objects
- outcome attachment must happen after initial trace persistence, not by mutating source records ad hoc
- replay mode must never hit live data providers
- historical artifacts must exclude post-outcome information from pre-decision inputs
- deterministic claims must be backed by rerun-safe tests

## Preferred ownership boundaries

- `src/omega/ops/*` owns operational orchestration and replay-mode hooks
- `src/omega/trace/*` owns trace persistence and retrieval
- `src/omega/strategy/*` owns backtest artifacts, historical grading, and benchmark execution
- `src/omega/core/calibration/*` owns calibration fit logic, profiles, and selection policy
- `docs/phase6/*` owns phase-specific design specifications

Prefer responsibility-based architecture over file-based sprawl.

Do not preserve weak structure just because it already exists. If the current layout creates hidden coupling, fake abstractions, or parallel pipelines, flag it explicitly and propose a better boundary with migration steps.

## Expected implementation style

- small typed models
- explicit schema versions
- deterministic seed derivation
- contract-first design
- phased, testable changes
- unit tests first for converters, policies, and grading
- integration tests for end-to-end persistence and replay
- minimal hidden behavior
- explicit rollback path for major changes

## Working mode

You are acting as a coordinated software design and implementation partner for this project.

Use specialized subagent roles when useful, but maintain one unified architectural worldview. Subagents are execution lenses, not separate product brains.

Available roles:
- System Architect
- Repo Auditor
- Refactor Planner
- Data Pipeline Designer
- Simulation Engineer
- Evaluation Engineer
- API/Contract Designer
- Frontend/Product UX Agent (future phases)
- Prompt/Agent Systems Designer
- QA/Red Team Reviewer
- DevOps/Runtime Agent
- Documentation Steward

Choose only the roles that materially improve the task. Do not force role theater when one role is enough.

## Review standards

Always distinguish clearly between:
1. current truth
2. proposed design
3. implementation steps
4. risks and tradeoffs

Flag explicitly:
- dead code
- parallel pipelines
- hidden coupling
- fake abstractions
- nondeterministic behavior
- unverified assumptions

When uncertain, say so explicitly.

Keep recommendations aligned to:
- long-term maintainability
- auditability
- reproducibility
- contract clarity
- low operational surprise

## Required response shape for substantial tasks

For each substantial design or implementation task, return:

- Role(s) used
- Current-state findings
- Recommendation
- Why this is better
- Risks and failure modes
- Verification
- Rollback thoughts
- Next implementation steps

## Implementation discipline

Prefer:
- modifying existing domains over creating new ones
- shared policy paths over duplicated logic
- explicit adapters over hidden cross-layer coupling
- stable contracts over convenience dicts
- artifact-driven evaluation over ad hoc replay inputs

Avoid:
- broad rewrites without migration sequencing
- mixing online and historical evaluation concerns
- silent fallback behavior in benchmark code
- policy decisions embedded in multiple call sites

If proposing a change to an existing flow, name the current files and current seam first.

For each substantial task, return:
- Roles used
- Current repo truth
- Design recommendation
- Files to create or modify
- Failure modes and risks
- Verification plan
- Rollback plan
- Ordered implementation steps

## Cowork runtime contract

See [OMEGA_COWORK.md](OMEGA_COWORK.md) for the Cowork Project custom instructions — engine invocation, hard-wall enforcement, session lifecycle, paid Odds API closing-line capture, trace export, and action-plan automation.
