# Omega — Current Handoff

## Project identity

Omega is a sports analytics platform with:
- a Monte Carlo simulation engine covering 9 sports
- an LLM-powered reasoning and orchestration layer
- an evidence collection pipeline (LLM web search primary, structured APIs optional)
- a strategy and backtest system with frozen artifact support
- a FastAPI backend with SSE streaming
- an observer-style skills system (trace-recorder, evidence-validator, data-quality-grader)

The repo follows a bounded-autonomy model:
- the LLM controls reasoning, planning, arbitration, evidence gathering, and explanation
- the deterministic engine owns simulation, calibration, backtesting, edge logic, and staking

**Repo root:** `C:\Users\camer\OneDrive\Desktop\Omega`
**Test command:** `python -m pytest tests/ -v`
**Current test status:** 383 passing; no API keys required for the test suite
**Branch:** `main`

## What's been completed

### Skills system (pre-Phase 6)
- Observer-style skill architecture: `SkillBase` ABC with `observe()` wrapper that never raises
- Three shipped skills: `trace-recorder`, `evidence-validator`, `data-quality-grader`
- Skills hook into the orchestrator at stages 4, 5, and 7 via `_emit_skill()` — crash-safe, disabled-safe
- Skills config in `omega/skills/config.json`; only `trace-recorder` is enabled by default

### Calibration drift bug fix (pre-Phase 6)
- Production service used `method="combined"` with gate check; backtest used `method="shrinkage"` without gate check
- Fixed by extracting `apply_calibration()` in `omega/core/calibration/probability.py` as single source of truth
- Both `omega/core/contracts/service.py` and `omega/strategy/backtest/engine.py` now delegate to it
- Parity test in `tests/core/test_engine.py::test_calibration_parity_service_and_backtest`

### Phase 6 design
- Red-team analysis of two-plane evaluation architecture in `docs/phase6/RED_TEAM.md`
- Full design plan in `docs/phase6/DESIGN_PLAN.md` covering Parts 1-3

### Phase 6a: Trace persistence (COMPLETE)
- `omega/trace/__init__.py` — package init
- `omega/trace/schema.py` — SQLite DDL v1 (traces, outcomes, schema_versions tables + indexes)
- `omega/trace/store.py` — `TraceStore` class:
  - `persist()`: idempotent on trace_id via INSERT OR IGNORE
  - `attach_outcome()`: validates trace exists, derives result string, separate table
  - `get_trace()`, `query_traces()`: filters by league, time range, has_outcome, execution_mode
  - `get_graded_traces()`: joins traces with outcomes for calibration input
  - WAL mode, foreign keys ON
- `omega/skills/trace_recorder.py` — updated `_write_trace()`: SQLite primary, JSONL fallback
- `tests/trace/test_trace_store.py` — 22 tests covering schema, persist, outcome, query, graded
- `tests/skills/test_skills.py` — updated TraceRecorder tests for SQLite-primary + JSONL-fallback (26 tests total)

### Phase 6b: Frozen artifacts + API independence (COMPLETE)

#### Part A: Evidence pipeline — API independence
- LLM web search promoted to primary data path (tier 2 for structured results)
- Search fallback chain: Perplexity Sonar → OpenAI web search → Anthropic web_search
- ESPN remains as optional tier-1 accelerator (always registered, no auth needed)
- Odds API now conditional — only registered when `ODDS_API_KEY` env var is set
- `FallbackSearchCollector` confidence promoted: structured JSON with >= 3 numeric fields → tier 2, confidence up to 0.85
- Files modified: `omega/evidence/collectors/search.py`, `omega/evidence/registry.py`

#### Part B: Frozen artifacts for backtest integration
- `omega/strategy/artifacts.py` — `FrozenArtifact` Pydantic model:
  - Typed, versioned backtest input derived from ExecutionTraces
  - Deterministic `artifact_id` via sha256 of event identity (home_team + away_team + league + date)
  - Decision-time data only; outcome attached separately at grading time
- `trace_to_artifact()` — converts persisted trace dict to FrozenArtifact
- `compat_dict_to_artifact()` — shim for legacy HistoricalGame dicts (backward compatible)
- `omega/strategy/backtest/engine.py` — updated:
  - `run()` accepts `List[FrozenArtifact]` or `List[dict]` (auto-converts via shim)
  - `_process_artifact()` replaces internal processing
  - Collects `trace_ids` from artifacts into BacktestResult
- `omega/strategy/models.py` — BacktestResult gains: `artifact_schema_version`, `calibration_policy`, `trace_ids`
- `tests/strategy/test_artifacts.py` — 16 tests: model creation, round-trip, deterministic ID, trace conversion, legacy compat, backtest parity

### Phase 6c: Calibration learning (COMPLETE)

- `omega/core/calibration/profiles.py` — `CalibrationProfile` Pydantic model + `ProfileStatus` enum (CANDIDATE → PRODUCTION → ARCHIVED/REJECTED lifecycle)
- `omega/core/calibration/registry.py` — `CalibrationRegistry` JSON-backed storage:
  - Atomic writes (.tmp + rename), re-reads on every operation
  - `get_production(league)` → single active profile or None
  - `promote()` auto-archives incumbent for same league
  - `reject()` with documented reason
- `omega/core/calibration/fitter.py` — `CalibrationFitter`:
  - `extract_pairs()` — extracts (prediction, outcome) from graded traces
  - `fit_isotonic()` — bins + Pool-Adjacent-Violators (PAV) for monotonic calibration map
  - `fit_shrinkage()` — grid search shrink_factor [0.3..1.0] minimizing Brier score
  - `evaluate()` — Brier score, ECE, log loss
  - `compare()` — candidate vs incumbent with promotion recommendation
  - Minimum 30 samples required for fitting
- `omega/core/calibration/probability.py` — `apply_calibration()` now accepts `league: Optional[str] = None`:
  - Profile-driven path: when league provided, looks up production profile via registry
  - Static fallback unchanged (backward compatible when league=None)
  - Fixed isotonic string-key bug (JSON round-trip converts float keys to strings)
- `omega/core/contracts/service.py` — `_calibrate()` updated: accepts `league` param, 5 call sites thread `request.league`
- `omega/strategy/backtest/engine.py` — 2 call sites thread `league` through `apply_calibration()`
- `omega/core/calibration/__init__.py` — exports: `apply_calibration`, `CalibrationProfile`, `ProfileStatus`, `CalibrationRegistry`, `CalibrationFitter`
- `tests/core/test_calibration_profiles.py` — 22 tests: model round-trip, registry CRUD, promote workflow, reject, fitter extract/fit/evaluate/compare, backward compat, profile selection, service-backtest parity

## What's next

### Phase 7: Planning required

Phase 6 (trace persistence + frozen artifacts + calibration learning) is fully complete. Potential next directions:

1. **Calibration operationalization** — CLI/script to fit profiles from production traces, evaluate, and promote
2. **Backtest expansion** — more markets (spread, totals), multi-sport validation
3. **Orchestrator replay mode** — replay-plane evaluation from historical evidence bundles
4. **Strategy versioning** — strategy registry with promotion/rejection workflow
5. **Frontend/dashboard** — visualization of calibration quality, backtest results, trace history

User should decide priority based on what prevents the most bad outcomes.

## Key files

| File | Role |
|---|---|
| `omega/trace/store.py` | SQLite trace persistence (Phase 6a) |
| `omega/trace/schema.py` | DDL definitions, CURRENT_VERSION = 1 |
| `omega/strategy/artifacts.py` | FrozenArtifact model + converters (Phase 6b) |
| `omega/strategy/backtest/engine.py` | Backtest engine (accepts FrozenArtifact or legacy dicts) |
| `omega/strategy/models.py` | Strategy + BacktestResult models (with trace linkage) |
| `omega/evidence/collectors/search.py` | LLM web search: Perplexity → OpenAI → Anthropic |
| `omega/evidence/registry.py` | Collector registry (web search primary, APIs optional) |
| `omega/skills/trace_recorder.py` | Trace-recorder skill (SQLite primary, JSONL fallback) |
| `omega/core/calibration/profiles.py` | CalibrationProfile model + ProfileStatus enum (Phase 6c) |
| `omega/core/calibration/registry.py` | CalibrationRegistry — JSON storage, promote/reject (Phase 6c) |
| `omega/core/calibration/fitter.py` | CalibrationFitter — fit from graded traces (Phase 6c) |
| `omega/core/calibration/probability.py` | `apply_calibration()` shared policy (profile-aware, Phase 6c) |
| `omega/core/contracts/service.py` | Production service entry point |
| `omega/reasoning/orchestrator.py` | 7-stage pipeline with skill hooks |
| `docs/phase6/DESIGN_PLAN.md` | Full Phase 6 design (Parts 1-3) |
| `docs/phase6/RED_TEAM.md` | Two-plane architecture red-team analysis |

## Known state

- 383 tests passing, 0 failing
- `omega_traces.db` exists at repo root from dev runs (32 traces, all with null predictions/odds — expected without API keys)
- No existing JSONL trace files need migration (SQLite is now primary)
- All changes are on `main`, uncommitted (git status shows modifications)
- System works fully without any sports API keys — LLM web search covers all evidence types

## Constraints reminder

- Do not duplicate edge, calibration, staking, or grading logic in a second path
- Every persistence format must be versioned
- Outcome attachment must happen after initial trace persistence, not by mutating source records
- Replay mode must never hit live data providers
- Anti-overengineering: does this directly prevent bad sim inputs, bad recommendations, bad replay, or bad backtests? If no, defer it.
