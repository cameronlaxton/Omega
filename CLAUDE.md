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

Do not move deterministic responsibilities into LLM logic.

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