# Omega Repository — Conversation Handoff

## What This Project Is

OmegaSportsAgent is a sports analytics platform with a Monte Carlo simulation engine covering 9 sports, a data collection pipeline, an LLM-powered reasoning layer, and a FastAPI backend with SSE streaming. The repo was audited against a "bounded autonomy" target operating model and is being incrementally restructured.

## What Has Been Completed

### Phase 0: Cleanup (Done)
- Removed empty `omega/infra/` stubs
- Removed OpenAI provider stub
- Cleaned dead code from orchestrator

### Phase 1: Architecture Boundary Fixes (Done)
- Established 5-layer architecture: Conversation (`omega/api/`), Reasoning (`omega/reasoning/`), Evidence (`omega/evidence/`), Execution (`omega/core/`), Synthesis (`omega/synthesis/`)
- Moved `omega/research/agent/` modules into `omega/reasoning/`: intent.py, router.py, planner.py, evaluator.py, gatherer.py, orchestrator.py, llm/client.py
- Moved `omega/research/data/` modules into `omega/evidence/`: models.py, collectors/, sources/, validation/, fusion/, pipeline/
- Extracted response composition into `omega/synthesis/composer.py`
- Updated all imports. All tests passing.

### Phase 2: Data Collection Redesign (Done)
- **Collector protocol** (`omega/evidence/collectors/base.py`): `Collector` protocol + `CollectorResult` model. Every collector returns `CollectorResult(data=Dict[str,Any], source, method, trust_tier, confidence, ...)`.
- **7 collectors implemented:**
  - `EspnCollector` (tier 1) — wraps existing ESPN schedule/standings functions
  - `OddsApiCollector` (tier 1) — wraps existing odds functions
  - `TeamFormCollector` (tier 2) — stub wrapping ESPN standings
  - `ContextCollector` (tier 2) — stub, returns None
  - `InjuryCollector` (tier 3) — stub, returns None
  - `NewsSignalCollector` (tier 3) — stub, returns None
  - `FallbackSearchCollector` (tier 3) — wraps Perplexity/web search, caps confidence at 0.70
- **CollectorRegistry** (`omega/evidence/registry.py`): dispatches by `(data_type, league)` ordered by trust tier. `build_default_registry()` wires all 7 collectors.
- **Entity resolution** (`omega/evidence/entity/`): `EntityResolver` with 6-tier resolution (canonical → alias → abbreviation → substring → fuzzy → pass-through). Alias DB covers 120+ teams across NBA/NFL/MLB/NHL. League-scoped disambiguation handles cross-league collisions (e.g., "Cardinals").
- **Pipeline rewrite** (`omega/evidence/pipeline/retrieval.py`): replaced hardcoded ESPN/OddsAPI/search fallback chain with registry-based dispatch. Entity resolution → cache → iterate collectors by trust tier → first success wins. Public API `retrieve_facts(slots)` unchanged.
- **Tests:** 32 new tests for entity resolution, collector protocol, registry dispatch. 169 total tests passing.

### Gating Review (Done)
Identified the critical gap: `CollectorResult.data` is `Dict[str, Any]` and flows unvalidated into simulation inputs via blind `dict.update()` in orchestrator. This was assessed as a blocking issue before further expansion.

### Phase 2.5: Evidence Hardening (Done)
- **Sim-input validation** (`omega/core/simulation/validation.py`): `validate_sim_context()` validates, coerces, and bounds-checks context dicts before they enter the Monte Carlo engine. `SIM_INPUT_BOUNDS` table covers all archetype keys across 9 sports. Drops invalid values (non-numeric, NaN/Inf, out-of-bounds) and strips unknown keys, letting the engine fall through to safe archetype defaults. `strict=False` param reserved for future research/production mode split.
- **Collector output validation** (`omega/evidence/collectors/base.py`): `validate_collector_numeric_fields()` type-checks known numeric keys in stat and odds data types, coerces string-numerics, drops garbage. Non-stat types (schedule, injury, news) pass through unchanged.
- **Orchestrator wiring** (`omega/reasoning/orchestrator.py`): 2-line change — `validate_sim_context()` called on `home_ctx` and `away_ctx` after the blind `dict.update()` merge, before entering the quant core.
- **Retrieval wiring** (`omega/evidence/pipeline/retrieval.py`): 1-line change — `validate_collector_numeric_fields()` called after collector returns, before caching.
- **FallbackSearchCollector hardening** (`omega/evidence/collectors/search.py`): confidence capped at 0.30 when fewer than 2 numeric values survive in the result, preventing thin LLM extractions from producing high-confidence bet recommendations.
- **Tests:** 47 new tests across `tests/core/test_sim_validation.py` and `tests/evidence/test_collector_validation.py`. Covers all 9 archetypes, type coercion, bounds rejection, unknown key stripping, garbage rejection, edge cases (None/empty context, unknown league), and bounds table integrity.
- **Total:** ~150 lines validation logic, ~100 lines bounds table, ~200 lines tests. 6 lines changed in 3 existing files. 3 new files. 216 total tests passing.

**The hardened data path:**
```
Collector → [validate_collector_numeric_fields] → cache → orchestrator
  → blind merge → [validate_sim_context: type + bounds + strip] → simulation engine
```

### Phase 3A: Reasoning Pipeline Hardening (Done)
- **Composer — all 11 OutputPackage types handled** (`omega/synthesis/composer.py`): Added handlers for COMPACT_SUMMARY, LIMITED_CONTEXT_ANSWER, BANKROLL_GUIDANCE, NEWS_DIGEST, SCENARIO_ANALYSIS, ALTERNATIVE_BETS. Previously these 6 types were silently dropped, producing empty response sections.
- **Evaluator — empty facts edge case fixed** (`omega/reasoning/evaluator.py`): `apply_quality_gate([])` no longer treats empty facts as "all critical inputs filled" via vacuous truth. Empty or missing facts now correctly trigger BET_CARD removal and plan downgrades.
- **Gatherer — vacuous truth fix** (`omega/reasoning/gatherer.py`): `critical_inputs_filled()` and `important_inputs_filled()` now return False when no slots of that importance tier exist, instead of True (vacuous truth).
- **Planner — slot consolidation** (`omega/reasoning/planner.py`): Replaced per-archetype-key GatherSlots (N slots for N keys) with one slot per (entity, data_type). Matches collector reality: collectors return all stats in one dict. Importance = CRITICAL if archetype has any critical keys.
- **Intent — football ambiguity fixed** (`omega/reasoning/intent.py`): `_detect_league()` now checks longer keywords first (sorted by length descending), so "college football" matches NCAAF before "football" matches NFL.
- **Orchestrator — variable ordering fix** (`omega/reasoning/orchestrator.py`): `league` variable is now defined before `validate_sim_context()` calls.
- **Tests:** 29 new tests covering all 11 composer package types, quality gate edge cases (empty facts, ultra-low data), gatherer vacuous truth, planner slot consolidation, intent edge cases (college football, "at" pattern, champions league).
- **Total:** 245 tests passing. 6 files modified.

## Key Files for Context

| File | Role |
|---|---|
| `omega/core/simulation/archetypes.py` | Defines `critical_team_keys`, `required_team_keys`, `optional_team_keys` per sport. Source of truth for what keys the sim reads. |
| `omega/core/simulation/validation.py` | `validate_sim_context()` + `SIM_INPUT_BOUNDS`. Sim-input boundary guard. |
| `omega/core/simulation/engine.py` | Monte Carlo engine. `run_fast_game_simulation()` at line 823. Archetype simulators at lines 322+ read from context via `.get(key, default)`. |
| `omega/core/contracts/service.py` | `analyze_game()` entry point. Wraps simulation + calibration + edge calc + Kelly staking. |
| `omega/core/contracts/schemas.py` | `GameAnalysisRequest` with `home_context: Optional[Dict[str, Any]]`. |
| `omega/reasoning/intent.py` | Intent understanding. Heuristic parser + LLM tool-calling. League detection sorted by keyword length. |
| `omega/reasoning/router.py` | Answer strategist. Deterministic routing + comparison guard + LLM arbitration for 2 ambiguity patterns. |
| `omega/reasoning/planner.py` | Requirement planner. One GatherSlot per (entity, data_type). Archetype-driven + query focus detection for player/aspect/temporal slots. |
| `omega/reasoning/evaluator.py` | Quality gate. Degrades answer type, not math. Empty-facts guard. |
| `omega/reasoning/gatherer.py` | Fact gathering delegation + quality helpers. Vacuous truth fixed. |
| `omega/reasoning/orchestrator.py` | End-to-end pipeline. Validation calls after blind merge, before sim. |
| `omega/synthesis/composer.py` | Response composer. All 11 OutputPackage types handled. |
| `omega/evidence/pipeline/retrieval.py` | `retrieve_facts()` → `_fill_slot()`. Registry-based collector dispatch. |
| `omega/evidence/collectors/base.py` | `Collector` protocol + `CollectorResult` model + `validate_collector_numeric_fields()`. |
| `omega/evidence/collectors/search.py` | `FallbackSearchCollector` — highest risk source, LLM extraction. |
| `omega/evidence/registry.py` | `CollectorRegistry` + `build_default_registry()`. |
| `omega/evidence/entity/resolver.py` | `EntityResolver` with alias DB. |

### Phase 3B: LLM-Enhanced Routing (Done)
- **Router comparison guard** (`omega/reasoning/router.py`): COMPARE goal without betting intent now routes to RESEARCH instead of NATIVE_SIM. Prevents simulation-generated bet cards for pure informational comparisons.
- **Router LLM arbitration** (`omega/reasoning/router.py`): Two narrowly-scoped ambiguity patterns trigger optional LLM override: (1) NATIVE_SIM with zero entities (can't build contexts), (2) SUMMARIZE + NATIVE_SIM without betting (likely a recap). LLM picks correct mode via tool-call; deterministic plan stands if LLM unavailable or fails.
- **Intent player extraction** (`omega/reasoning/intent.py`): Heuristic parser now detects player names (2-3 capitalized words) when no vs/at team pattern found. "How is LeBron James performing?" → player entity with SUBJECT role. Also fixed: "vs" removed from COMPARE goal signals (it's a matchup indicator, not a comparison signal); "analyze" added as explicit ANALYZE signal.
- **Planner query focus** (`omega/reasoning/planner.py`): `_detect_query_focus()` identifies player focus, aspect focus ("defense", "shooting", etc.), and temporal signals ("lately", "recently"). `_add_focus_slots()` adds player_stat + player_game_log slots for player queries and sets `focus_hint` on team_stat slots for aspect queries.
- **GatherSlot focus_hint** (`omega/core/models.py`): New optional `focus_hint` field signals which aspect the user cares about, for downstream composition.
- **Orchestrator wiring** (`omega/reasoning/orchestrator.py`): LLM client now passed to `build_answer_plan()` and `build_gather_list()` (2-line change).
- **Tests:** 28 new tests covering comparison routing, ambiguity detection, LLM arbitration (mocked), player entity extraction, focus detection, adaptive slots, and integration regressions.
- **Total:** 273 tests passing. 6 files modified, 1 new test file.

## What Comes Next

| Phase | Description | Status |
|---|---|---|
| Phase 5 | Production Hardening — execution traces, reproducibility, audit trail, strict validation mode | Ready |
| Phase 4 | Research Sandbox — exploratory mode, model comparison, what-if scenarios | Lowest priority |

## Anti-Overengineering Constraint (User-Imposed)

The user explicitly directed: *"Does this directly prevent bad sim inputs, bad recommendations, or bad backtests? If the answer is no, it should probably be deferred."*

Deliberately deferred:
- Typed evidence models per data type
- `BaseEvidence` inheritance hierarchy
- Field-level provenance tracking
- Collector-internal schema enforcement
- `completeness` field on ProviderResult
- Dead model cleanup in `evidence/models.py`
- Odds-specific typed validation (OddsInput already exists and is typed)
- Research vs production mode wiring

## Test Status

273 tests passing (137 original + 32 Phase 2 + 47 Phase 2.5 + 29 Phase 3A + 28 Phase 3B). Test suite runs in ~3s.
