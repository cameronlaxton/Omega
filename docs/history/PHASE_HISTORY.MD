# Omega — Phase History

## Project summary

Omega is a sports analytics platform being incrementally reshaped toward a bounded-autonomy architecture.

Core model:
- the LLM controls reasoning, planning, arbitration, and explanation
- the deterministic engine owns simulation, calibration, edge logic, staking, and backtesting

This document is historical reference, not the canonical operating contract. The FastAPI service (`omega/api/`), all external-API collectors (`omega/evidence/`), and the orchestrator-based reasoning pipeline (`omega/reasoning/orchestrator.py` and friends) referenced below have since been retired. See `AGENTS.md` (cross-agent entrypoint), `prompts/system_prompt.txt` (no-local-access agents), and `OMEGA_RUNTIME.md` (local VM / MCP runtime) for the current runtime model.

## Architecture baseline

The repo was restructured into five layers:

| Layer | Module | Purpose |
|---|---|---|
| Conversation | `omega/api/` | FastAPI, sessions, SSE streaming |
| Reasoning | `omega/reasoning/` | intent, routing, planning, gathering, evaluation, orchestration |
| Evidence | `omega/evidence/` | collectors, resolution, validation, fusion, retrieval |
| Execution | `omega/core/` | simulation, archetypes, contracts, calibration, staking |
| Synthesis | `omega/synthesis/` | response composition |

## Phase 0 — Cleanup

**Status:** completed

### Delivered
- removed empty `omega/infra/` stubs
- removed unused OpenAI provider stub
- cleaned dead code from orchestrator and surrounding areas

### Why it mattered
This removed misleading structure and reduced fake architecture before deeper changes began.

## Phase 1 — Architecture Boundary Fixes

**Status:** completed

### Delivered
- established the 5-layer architecture
- moved research-agent responsibilities into `omega/reasoning/`
- moved data and evidence responsibilities into `omega/evidence/`
- extracted response composition into `omega/synthesis/composer.py`
- updated imports across the repo

### Why it mattered
This created clearer ownership boundaries and reduced cross-layer confusion.

## Phase 2 — Data Collection Redesign

**Status:** completed

### Delivered
- introduced a `Collector` protocol and `CollectorResult` model
- added registry-based evidence retrieval instead of hardcoded source chains
- implemented or stubbed 7 collectors across trust tiers:
  - ESPN
  - Odds API
  - Team Form
  - Context
  - Injury
  - News Signal
  - Fallback Search
- introduced `CollectorRegistry`
- introduced `EntityResolver` with multi-stage alias and fuzzy resolution
- rewrote retrieval flow to use registry dispatch

### Why it mattered
The pipeline stopped being a hardcoded procedural chain and became an explicit evidence retrieval system.

### Testing
- 32 new tests added
- total at that point: 169 passing

## Gating Review — Post-Phase 2

**Status:** completed

### Finding
A major safety gap was identified:
- `CollectorResult.data` remained a loose `Dict[str, Any]`
- collector output could flow into simulation via blind merges
- no boundary guard protected the engine from garbage or malformed values

### Outcome
This was treated as a blocking issue and triggered Phase 2.5.

## Phase 2.5 — Evidence Hardening

**Status:** completed

### Delivered
- added `validate_sim_context()` in `omega/core/simulation/validation.py`
- added `validate_collector_numeric_fields()` in `omega/evidence/collectors/base.py`
- validated and bounded simulation inputs before entering Monte Carlo
- validated known numeric collector fields before caching
- hardened `FallbackSearchCollector` confidence when numeric evidence was thin
- wired validation into retrieval and orchestration with minimal changes

### Hardened path
`Collector → validate_collector_numeric_fields → cache → orchestrator → blind merge → validate_sim_context → simulation engine`

### Why it mattered
This directly reduced the chance of bad collector output causing bad sim inputs or bad recommendations.

### Testing
- 47 new tests added
- total at that point: 216 passing

## Phase 3A — Reasoning Pipeline Hardening

**Status:** completed

### Delivered
- added handlers for all 11 `OutputPackage` types in the composer
- fixed vacuous-truth bugs in evaluator and gatherer
- changed planner slot design to one slot per `(entity, data_type)`
- fixed league detection ambiguity for college football versus generic football
- corrected orchestrator variable ordering around validation

### Why it mattered
This fixed silent output loss, invalid truth assumptions, and planner-reality mismatch.

### Testing
- 29 new tests added
- total at that point: 245 passing

## Phase 3B — LLM-Enhanced Routing

**Status:** completed

### Delivered
- routed COMPARE without betting intent to RESEARCH instead of NATIVE_SIM
- added narrowly-scoped LLM arbitration for specific ambiguous situations
- improved player extraction for non-team queries
- added query-focus detection for player/aspect/temporal signals
- added `focus_hint` to `GatherSlot`
- passed LLM client into router and planner where needed

### Why it mattered
This improved routing correctness without handing deterministic responsibilities to the LLM.

### Testing
- 28 new tests added
- total at that point: 273 passing

## Phase 5 — Production Hardening

**Status:** completed

### Delivered
- populated `ExecutionTrace` across the full seven-stage pipeline
- added deterministic simulation seeding support
- added strict validation mode
- added downgrade tracking in evaluator
- returned trace in normal responses and SSE streaming

### Why it mattered
This made the pipeline inspectable and created the basis for trace persistence, replay, and future calibration learning.

### Testing
- 22 new tests added
- total at that point: 295 passing

## Phase 8 — Operator Console (read-only)

**Status:** in progress (read-only console live; V2 re-skin + first wave of data
visualizations landed)

### Delivered
- a standalone, loopback-fail-closed, **read-only** FastAPI console (`omega-console`,
  `src/omega/ui/` + `src/omega/ops/console_server.py`) for auditing traces, bets, sessions,
  calibration, signals, diagnostics, review queue, and closing-line value — no mutation path
  (the service refuses a writable store; every route is GET)
- a V2 "mission-control" re-skin of the console pages (hero headers, restructured ledgers,
  dark design-token system)
- **first wave of data visualizations** (server-computed SVG geometry; no client charting
  library; one explicit unit per chart; honest empty/low-n states):
  - per-bet **comparison strip / dumbbell** — Omega P(selection) vs market implied
    probability (Edge Scanner, Trace Detail, Bet Detail)
  - **market-movement ribbon** — taken → closing implied with outcome cap (CLV page)
  - **calibration reliability diagram** — model-probability bucket vs realized hit rate with
    the y=x diagonal; low-n buckets suppressed (Calibration page)
  - **CLV scatter** — closing-line value vs net result with quadrant guides (CLV page)
  - **data-quality heatmap** — per-league evidence / closing-line / outcome coverage on a new
    `/data-quality` page
- the reusable geometry primitives behind them (`_strip_x` / `_prob_strip`,
  `_scatter_geometry`) plus the `comp_strip` / `reliability_chart` / `clv_scatter` /
  `quality_heatmap` Jinja macros

### Deferred (by design)
The remaining visualization backlog is specced in
[`docs/phase8/CONSOLE_V2_VISUALS.md`](../phase8/CONSOLE_V2_VISUALS.md) and intentionally not
built in this wave:
- bankroll / PnL equity curve (top of backlog)
- trace-lifecycle coverage funnel (reframed from a Sankey)
- signal coverage-vs-usefulness scatter (reframed from a radar)
- evidence coverage bar (reframed from a contribution "waterfall")
- review-queue kanban; per-trace lifecycle timeline enhancement; league/market treemap
- explicitly out of scope (no backing data): odds-freshness column, live multi-book quote
  spread, native-unit spread/total/prop dumbbell

### Why it mattered
The console turns the engine's already-persisted, audit-grade data into something an operator
can read at a glance — *did we beat the close, where is Omega over/under-confident, where is
coverage thin, is good process being rewarded* — without ever recomputing a protected
quantity or fabricating a value when data is absent.

### Testing
- `tests/ui` at 215 passing (25 added for the visualization first wave)

## Current historical milestones

### Completed phases
- Phase 0: Cleanup
- Phase 1: Architecture Boundary Fixes
- Phase 2: Data Collection Redesign
- Phase 2.5: Evidence Hardening
- Phase 3A: Reasoning Pipeline Hardening
- Phase 3B: LLM-Enhanced Routing
- Phase 5: Production Hardening

### Upcoming and Current Phases

| Phase | Description | Status |
|---|---|---|
| Phase 6h | Calibration and Runtime Hardening (current phase) | Active |
| Phase 7 | Multi-Sport Expansion (Soccer, Tennis, NFL) | Deferred (except explicitly marked experimental parts) |
| Phase 8 | Operator Console (read-only) — V2 re-skin + data-visualization first wave | Active; visualization backlog deferred (see `docs/phase8/CONSOLE_V2_VISUALS.md`) |
| Phase 4 | Research Sandbox — exploratory mode, model comparison, what-if scenarios | lowest priority |

## Deferred by design

The following were deliberately deferred under the anti-overengineering rule:
- typed evidence models per data type
- `BaseEvidence` inheritance hierarchy
- field-level provenance tracking
- collector-internal schema enforcement
- `completeness` field on provider results
- dead model cleanup in `omega/evidence/models.py`
- odds-specific typed validation beyond existing contract boundaries

### Why these were deferred
They did not directly prevent bad sim inputs, bad recommendations, bad replay, or bad backtests at the time they were considered.

## Test progression

| Milestone | Total passing tests |
|---|---:|
| Before major restructure | 137 |
| After Phase 2 | 169 |
| After Phase 2.5 | 216 |
| After Phase 3A | 245 |
| After Phase 3B | 273 |
| After Phase 5 | 295 |
| Phase 8 — operator console (`tests/ui` suite) | 215 |

> Note: the Phase 8 figure is the `tests/ui` suite only (the console is an isolated
> read-only surface), not the repository-wide total, which has grown well beyond the
> Phase-5 baseline across the Phase 6/7 work tracked outside this file.

## Notes for future readers

- this file is historical context
- `AGENTS.md` is the canonical operating contract (`CLAUDE.md` is a thin shim to it)
- there is no separate current-phase execution brief; `AGENTS.md` plus this file's
  "Upcoming and Current Phases" table above are the current-state sources
- update this file when a phase closes instead of creating overlapping handoff files