# Omega — Project Guidelines

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

**Hard rule (supersedes all prior fallback wording in OMEGA_RUN_RECIPE.md and OMEGA_HANDBOOK.md):** The LLM is forbidden from generating any of the above values via text. They must come from Python execution of `omega_lite_standalone.py` (sandbox) or the canonical FastAPI service (`omega/api/`). There is no "estimated", "rough", "ballpark", `[Codex-ESTIMATED]`, or "estimated lean" mode for these fields. If the engine is unavailable, the response is qualitative-only: matchup narrative, news, recent form, listed sportsbook lines from a cited source — never a Bet Card with placeholder numbers.

The LLM may still perform a best-effort exploratory market scan with public web data, but any candidate it surfaces is labeled as a **research-only lean** or **missing-data watchlist** item with NO edge%, EV%, Kelly, units, confidence tier, or trace_id. The previous "estimated lean" label is retired.

The master runtime instruction for any LLM acting as the Omega agent (Codex.ai Project, ChatGPT Project, API agent) is [`prompts/system_prompt.txt`](prompts/system_prompt.txt). That file is authoritative for agent behavior; OMEGA_HANDBOOK.md and OMEGA_RUN_RECIPE.md are reference documents only.

**Deployment-specific instructions:** Use **one** instruction set based on deployment:
- **Codex.ai Project / API agent (no local access)** → [`prompts/system_prompt.txt`](prompts/system_prompt.txt)
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

- `omega/reasoning/*` owns orchestration and replay-mode hooks
- `omega/trace/*` owns trace persistence and retrieval
- `omega/strategy/*` owns backtest artifacts, historical grading, and benchmark execution
- `omega/core/calibration/*` owns calibration fit logic, profiles, and selection policy
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