# Omega - Project State

## Current Phase: Phase 7 hardening (multi-sport landed; calibration/provenance hardening)

Phase 6h (trace persistence, historical replay, calibration learning) is **complete**, and the
Phase 7 multi-sport expansion (Milestones 0–4: backend registry, WNBA, Soccer, Tennis, NFL) is
**merged to `main` with green tests** (3709 passed). Schema is at V18; all sport backends are
registered. The project is now in a **hardening pass**: making the calibration loop, sidecar/
provenance attribution, and runtime-artifact policy production-trustworthy before betting goes
live on the new sports.

This supersedes the earlier "do not relabel as Phase 7 / Phase 7 deferred" guidance, which is no
longer accurate — that work shipped. Remaining production gaps are tracked under the calibration
caveats below and in the audit remediation plan, not as "deferred Phase 7."

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
- `var/omega_traces.db`: Canonical numeric ledger for predictions, outcomes, grading, calibration data, optional wager/CLV metadata. **Live DB is runtime state, not source-controlled truth.**
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

## Calibration Promotion Authority (fail-closed)

Promotion to PRODUCTION is fail-closed. `CalibrationRegistry.promote()` evaluates the
shared gate in `omega/core/calibration/promotion.py` and raises `PromotionGateError`
unless every gate passes: `SAMPLE_SIZE >= 100`, `ECE_FLOOR <= 0.05`, Brier improvement and
log-loss no-regression vs incumbent, plus operator-confirmed `BACKTEST_PARITY` /
`CLV_NON_REG`. There is no `--force` bypass.

**Current PRODUCTION profiles (as of 2026-06-17, post audit-remediation):** exactly one
profile in `omega/core/calibration/profiles.json` carries `status: production`:

| profile_id | league / market | n | CV-ECE | promoted |
|---|---|---|---|---|
| `iso_mlb_v7_*` | MLB / game | 130 | 0.049 | 2026-06-16 |

MLB game is the one market with a genuinely-applied production profile.

**Demoted in this remediation (were briefly `production`, now `rejected`):**
- `iso_world_cup_draw_v1_*` (WORLD_CUP/draw, n=1171) and `iso_fifa_friendly_draw_v1_*`
  (FIFA_FRIENDLY/draw, n=1232). They had good CV-ECE but were **never applied at runtime** and
  were **backend-mismatched** — see caveats below. They await a backend-matched re-fit under the
  `FIFA_INTL` bucket. The earlier `iso_nba_prop_v1` / `shrink_mlb_prop_v2` remain `rejected`.

**Why the soccer-draw profiles were demoted (open remediation):**
1. **League-key mismatch.** They were keyed `WORLD_CUP` / `FIFA_FRIENDLY`, but live World Cup
   traces carry `league="FIFA_WORLD_CUP_2026"`. `CalibrationRegistry.get_production` now resolves
   a canonical calibration bucket first (`omega/core/calibration/league_buckets.py`:
   `FIFA_WORLD_CUP_2026` → `FIFA_INTL`), so once a `FIFA_INTL` draw profile is re-fit it will
   apply to live World Cup traces automatically.
2. **Backend mismatch.** `iso_world_cup_draw_v1` was fit from the legacy `fast_score` `WORLD_CUP`
   replay set; live `FIFA_WORLD_CUP_2026` runs `soccer_bivariate_poisson_dc`. Calibration is
   per-model, so it must be re-fit on backend-matched data (requires xG-enriched historical
   replay — see the audit remediation plan, Phase C3).
3. **Operator-attested gates.** `BACKTEST_PARITY` / `CLV_NON_REG` have no automated check yet —
   they are `--confirm-*` attestations. "Production" status does not by itself guarantee
   backtest/CLV safety until those checks are automated or a parity artifact is recorded.

Every market not backed by a genuinely-applied, gate-passing profile correctly classifies as
`RESEARCH_CANDIDATE`. Closing caveats 1–3 is the open work for Exit Criterion 1.

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
  `omega-fetch-outcomes-soccer`, dispatched by `omega-fetch-outcomes-all`.
- 3-way result: draws are graded by `TraceStore.attach_outcome` (equal scores → `draw`)
  and calibrated via `CalibrationFitter.extract_draw_pairs`.

Soccer player-prop grading is partially wired for ESPN-summary-backed leagues through
`fetch_outcomes_props.py`: EPL/PREMIER_LEAGUE, LA_LIGA/LALIGA, BUNDESLIGA, SERIE_A,
LIGUE_1, CHAMPIONS_LEAGUE, LIGA_MX, MLS, WORLD_CUP, and FIFA_WORLD_CUP_2026 can
attach `goals`, `assists`, `shots`, `shots_on_target`, `yellow_cards`, and `red_cards`
from ESPN roster stats. Still open for soccer: unsupported competitions without ESPN
slug mappings, soccer player-prop odds-resolution/provider-market mappings, and a first
fitted soccer calibration profile once eligible graded traces accumulate.

## Shipped in Phase 7 (formerly "deferred")

- **Tennis (ATP/WTA)** — `tennis_markov_iid` game backend + `tennis_prop_serve` prop backend
  merged and registered; `priors_tennis` / `priors_tennis_pressure` seeded. No production
  calibration profile yet (expected — rides identity profile until volume accrues).
- **NFL** — `nfl_neg_binom` game backend + `prop_neg_binom` + teasers merged. `priors_nfl_dispersion`
  is still empty (fit not yet run), so NFL NB props fall back to the distribution router.

## Still open / deferred

- Expanded soccer player-prop source coverage and odds-resolution mappings beyond the
  ESPN-summary-backed grading subset above.
- Portfolio/risk guard (simple exposure caps can be documented as temporary operator rules for now).
- First backend-matched soccer-draw calibration profile under the `FIFA_INTL` bucket (see the
  calibration caveats above).

## Superseded / Historical Docs

- `OMEGA_HANDOFF_MANIFESTO.MD` (Historical)
- Any previous architectural handoff documents that contradict this state.
