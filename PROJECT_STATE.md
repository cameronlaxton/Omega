# Omega - Project State

## Current Phase: Phase 6h (Calibration and Runtime Hardening)

Omega is currently in Phase 6h. The goal of this phase is to ensure trace persistence, historical replay, and calibration learning are stable and rigorous before any broader multi-sport or advanced risk-management expansion.

Do not relabel the project as Phase 7 yet. Any Phase 7 items (e.g., Tennis) should be documented as deferred or future-facing unless they are already implemented and stable. Soccer game-plane support has been implemented ahead of Phase 7 (see the Soccer section below).

## Product Doctrine

Omega is a **research assistant + betting decision engine that continuously improves** through traceable outcomes, calibration, replay, evidence scoring, and disciplined QA.
- **First User:** Cameron, the bettor/operator.
- **First Paid-Quality Promise:** Better picks.
- **Supporting Promises:** traceability, evidence hygiene, bankroll/risk discipline, calibration learning, auditability, and reproducibility.

## Runtime Authority

- **Canonical Runtime:** MCP-first local runtime.
- **Deterministic Entrypoint:** `omega.core.contracts.service.analyze`.
- **Requirements:** Analyze calls require explicit `session_id` and `bankroll`.
- **Model Version:** Current model version remains `omega-core-phase6h` unless intentionally migrated.

## Source-of-Truth Authority

Canonical source-of-truth rules must be consistent everywhere:
- `omega_traces.db`: Canonical numeric ledger for predictions, outcomes, grading, calibration data, optional wager/CLV metadata. **Live DB is runtime state, not source-controlled truth.**
- **Sidecars:** Canonical session/process narrative.
- **Reports:** Generated, derived, non-canonical. Any generated reports committed to the repo must include metadata showing `canonical: false`.
- **Trace Exports:** Import-only transfer artifacts. Live processed traces are runtime artifacts.
- **JSONL Mirrors:** Recovery mirrors, not canonical.
- **Fixtures:** Curated test examples only. Committed examples must live under `fixtures/`.
- **Runtime artifacts:** Should not pollute normal source review.

## Output Language Rules

Omega must never use hype language for betting outputs.

**Banned terminology:**
- "best bet" / "Best Bet"
- "lock"
- "smash"
- "Tier A" / "Tier B"
- "engine-confirmed"
- "actionable bet" / "Actionable Bet"

**Preferred terminology:**
- "highest-confidence opportunity"
- "qualified play"
- "tracked lean"
- "model-supported lean"
- "confidence band"
- "calibration status"
- "risk flags"
- "edge / EV / confidence metrics"

## Valid Session Requirements

A valid session requires:
- Genuine data gathering.
- Structured context.
- Evidence present or explicit downgrade rationale.
- Context/evidence affecting reasoning or bet selection.
- Trace emitted for every model-backed recommendation.
- Outcome-ready identity.
- Sidecar/audit process trail.
- Continuous-improvement value.

## Calibration Eligibility Rules

Keep calibration strict. Free-text reasoning alone must not make a trace calibration-eligible. LLM reasoning counts as useful context only when translated into:
- structured `game_context`
- structured `player_context`
- typed `EvidenceSignal`
- explicit downgrade rationale
- trace-quality metadata

Do not loosen `context_source="provided"` to mean "the LLM thought about it." It should mean the model received structured, decision-time context.

## Actionability Modes

- **Hard fail / block formal output when:** required context is missing, evidence is empty with no downgrade rationale, odds are stale and unreplaced, engine status is skipped/error but output presents a play, identity fields are missing, or no trace is emitted for a model-backed recommendation.
- **Warning-only when:** optional CLV metadata is missing, no bet record exists because no bet was taken, reasoning narrative is missing but structured trace/evidence is present, evidence sample size is thin, identity is metadata-recovered but marked.
- Warnings become hard failures only when the issue is repairable, the schema/contract is stable, and failing prevents bad picks, calibration, or audit data.

## Phase 6h Exit Criteria

1. At least one league/market calibration profile can be fitted from eligible graded traces.
2. Session lifecycle is reproducible: preflight -> odds/context -> analyze -> trace export -> ingest -> outcome attach -> report -> audit/replay.
3. Trace eligibility rules are stable and documented.
4. Sidecar/report/audit artifacts are reliable and validated.
5. Runtime/generated artifact policy is settled.
6. `PROJECT_STATE.md` is canonical.
7. CI or local validation checks cover tests, sidecars, export shapes, report metadata, and artifact policy.

## Soccer (in progress — promoted from Phase 7)

Soccer is now wired end-to-end at the game plane:
- Simulation: `_sim_soccer()` (Dixon-Coles) in `omega/core/simulation/engine.py`.
- Leagues: MLS, EPL, LA_LIGA, BUNDESLIGA, SERIE_A, LIGUE_1, CHAMPIONS_LEAGUE, LIGA_MX
  in `omega/core/config/leagues.py`.
- Outcomes: `omega/integrations/espn_soccer.py` (per-competition ESPN scoreboard) +
  `scripts/fetch_outcomes_soccer.py`, dispatched by `scripts/fetch_outcomes_all.py`.
- 3-way result: draws are graded by `TraceStore.attach_outcome` (equal scores → `draw`)
  and calibrated via `CalibrationFitter.extract_draw_pairs`.

Still open for soccer: player-prop grading (ESPN soccer boxscore not yet parsed, so soccer
is **not** in `fetch_outcomes_props.py::_SUPPORTED_LEAGUES`), and a first fitted soccer
calibration profile once eligible graded traces accumulate.

## Deferred to Phase 7

- Tennis multi-league expansion.
- Soccer player-prop grading (game plane is in progress above).
- Portfolio/risk guard (simple exposure caps can be documented as temporary operator rules for now).

## Superseded / Historical Docs

- `OMEGA_HANDOFF_MANIFESTO.MD` (Historical)
- Any previous architectural handoff documents that contradict this state.
