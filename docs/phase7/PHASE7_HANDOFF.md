# Phase 7 Handoff — Multi-Sport Expansion

**Purpose:** continue Phase 7 implementation in a fresh session. This document is
the single source of truth for *what is built*, *what the seams are*, and *exactly
how to implement the remaining milestones*. Read it alongside the locked design
plan `docs/phase7/MULTI_SPORT_EXPANSION.md` (authoritative for design intent).

Last updated after Milestone 1, plus an **M0 registry-hardening pass (2026-06-01)**.
All work below is committed to `main`.

---

## 1. Status at a glance

| Milestone | Scope | Status | Commits |
|-----------|-------|--------|---------|
| **M0** | Backend registry + shared ETL harness | ✅ Done · hardened (see §2.2–§2.3) | `f2ccf80` |
| **M1** | WNBA (backend, integration, early-line isolation, wehoop history) | ✅ Done | `80134b3` (A), `f128711` (B), `b709c99` (C) |
| **M2** | Soccer (World Cup) — bivariate Poisson + Dixon-Coles | ⬜ Not started | — |
| **M3** | Tennis (ATP/WTA) — IID Markov + pressure coefficients | ⬜ Not started | — |
| **M4** | NFL — Gamma-Poisson + NB props + Wong teasers | ⬜ Not started | — |

**Test baseline:** `python -m pytest tests/ -q` → **993 passed** (last green run;
744 at M1 close, the rest added by later milestones + the M0 hardening pass).
Phase 6 NBA + MLB replay determinism is bit-identical. Do not regress this.

> ⚠️ **Deadline:** M2 (Soccer) has a hard external deadline of **2026-06-11**
> (World Cup kickoff). Build M2 before M3, per the plan.

---

## 2. The seams you will build on (created in M0/M1)

These are the reusable foundations. New milestones plug into them — do **not**
re-create parallel versions.

### 2.1 Game-backend registry — `omega/core/simulation/backends.py`
- `GAME_BACKENDS`, `register_game_backend(name, backend)`, `resolve_game_backend(name)`.
- A backend is any object with `backend_name`, `component_version`, and
  `run(GameSimulationInput) -> dict` satisfying `enforce_game_backend_contract`
  (V10 distribution rows on success).
- **Register at engine import time.** See the bottom of
  `omega/core/simulation/engine.py`:
  ```python
  from omega.core.simulation.markov_wnba import MarkovWNBAGameSimulationBackend
  register_game_backend("markov_state_wnba", MarkovWNBAGameSimulationBackend())
  ```
  New sport modules do a **lazy import** of any engine helper inside `run()` to
  avoid an import cycle (markov_wnba.py is the reference pattern).

### 2.2 Prop-backend registry — same file
- `PROP_BACKENDS`, `register_prop_backend`, `resolve_prop_backend`.
- `PropSimulationInput` dataclass (has `prior_payload` for NB `k`, SPW%, etc.; the
  distribution router also reads a caller `distribution` override and `dud_prob`
  out of `prior_payload`).
- `DEFAULT_PROP_BACKEND_BY_LEAGUE_STAT` + `resolve_default_prop_backend(league, stat)`.
  NFL yardage routes to `"prop_neg_binom"`; tennis aces to `"tennis_prop_serve"`.
- **Live wiring (M0 hardening).** `service.analyze_player_prop` now dispatches
  through the registry — `resolve_default_prop_backend` → `resolve_prop_backend` —
  instead of calling `run_player_simulation` directly. Any routed name that is
  **not yet registered** (`prop_neg_binom`, `tennis_prop_serve`) **falls back to
  `prop_distribution_router`** and appends a note to the response. So unregistered
  routes price correctly via the router today instead of breaking, and M3/M4
  registering the real backend is a drop-in upgrade. The fallback is bit-identical
  to the prior direct call for NBA/MLB props (verified in
  `tests/core/test_backend_registry.py`).

### 2.3 Dispatch + league default — `omega/core/contracts/service.py`
- `analyze_game` dispatches via `resolve_game_backend(_effective_game_backend_name(request))`,
  then passes the resolved backend **per call** into
  `_engine.run_fast_game_simulation(..., backend=backend)`, so a single shared
  `_engine` runs every backend. *(M0 hardening removed the old
  `backend_name == "fast_score"` branch that built a fresh `OmegaSimulationEngine`
  for non-fast-score backends; replay stays bit-identical.)*
- `_effective_game_backend_name(request)`: when the caller leaves
  `simulation_backend="fast_score"` (the default), the league's
  `default_game_backend` from `leagues.py` is used. **This is how a sport
  auto-selects its backend.** Set `default_game_backend` in the league config and
  it just works.
- **Evidence routing reads a backend capability, not its name (M0 hardening).**
  Every game backend declares `evidence_mode` on the `GameSimulationBackend`
  Protocol — `"plane_adjustment"` (fast-score path: handler adjustments applied
  to contexts) or `"markov_transition"` (markov path: transition modifiers).
  `service._uses_transition_modifiers(backend)` reads it via
  `getattr(backend, "evidence_mode", "plane_adjustment")`. This replaced the old
  `_is_markov_family(name)`/`startswith("markov_state")` name sniff, so a
  Markov-family backend no longer has to be *named* `markov_state*` to route
  evidence correctly.
  **⚠️ Decision point for M2–M4:** soccer/tennis/NFL backends default to
  `"plane_adjustment"` (the fast-score evidence path). If a new backend needs the
  transition-modifier path — or a third mode — set its `evidence_mode` and extend
  the handling in `_game_evidence_plan_for` rather than special-casing call sites.

### 2.4 Shared Markov body — `omega/core/simulation/engine.py`
- `run_markov_game_simulation(request, *, backend_name, component_version, context_source="provided", baseline_used=False)`.
- Reused by `MarkovGameSimulationBackend` and `MarkovWNBAGameSimulationBackend`.
  Tennis (M3) is its **own** absorbing-Markov math, not this body — do not force
  it through here.

### 2.5 Sport baselines — `omega/core/sport_baselines.py`
- `LEAGUE_BASELINES[league]` tuning constants + `basketball_context_defaults(league)`.
- Add per-sport baseline dicts here (do not scatter magic numbers in backends).

### 2.6 Shared ETL harness — `omega/integrations/_etl.py`
**Every** new adapter uses these three helpers (the Part 5B standards, implemented once):
1. `@cached_fetch(source, ttl_seconds, fmt, cache_root)` — raw→cache before
   transform; zero-refetch within TTL; calls `assert_not_replay_mode` on a cache
   **miss** only (frozen caches still readable in replay). `fmt` ∈
   {parquet, json, html, text}; pass `cache_key=` at call time.
2. `validate_records(rows, PydanticModel, source=, session_path=)` — fail-loud
   with `SourceSchemaDriftError` + a `fail`-status `data_provenance` sidecar
   event. Never coerce a renamed column to `None`.
3. `resolve_entity(name, alias_table)` / `resolve_entities(...)` +
   `load_alias_table(league)` — exact → `normalize_player_name` → alias table →
   unresolved (excluded + `warn` event). Alias tables live in `data/aliases/<league>.json`.
- Reference implementation: `omega/integrations/wehoop.py` uses all three.
- **CI guard:** `tests/integrations/test_replay_mode_guard.py` statically asserts
  every module under `omega/integrations/` references `assert_not_replay_mode`.
  Any new adapter that omits it fails CI.

### 2.7 Persistence conventions — `omega/trace/`
- Schema is now at **V11** (`omega/trace/schema.py` `CURRENT_VERSION = 11`).
  New tables are forward-additive: add `SCHEMA_V<n>`, bump `CURRENT_VERSION`,
  wire it in `store.py::_ensure_schema`, and **bump the version-pin tests**
  (`tests/trace/test_schema_v7.py`, `test_schema_v9.py` assert the literal).
- New priors tables (`priors_xg`, `priors_dixon_coles`, `priors_tennis`,
  `priors_tennis_pressure`, `priors_nfl_dispersion`) will be V12+.
- `early_market_snapshots` (V11) is the template for an isolated capture table.

---

## 3. What M1 delivered (file map)

**New (engine/baselines):**
- `omega/core/sport_baselines.py`
- `omega/core/simulation/markov_wnba.py`

**New (integration/scripts):**
- `omega/integrations/espn_wnba.py` (live), `omega/integrations/wehoop.py` (historical)
- `scripts/capture_early_lines.py`, `scripts/refresh_wehoop.py`
- `data/aliases/WNBA.json`

**Modified:**
- `omega/core/simulation/engine.py` (shared markov helper + WNBA registration)
- `omega/core/contracts/service.py` (league-default resolution, markov-family check)
- `omega/core/config/leagues.py` (WNBA backend + `liquidity_profile: low`)
- `omega/integrations/espn_boxscore.py` (`WNBA_STAT_KEYS`, WNBA summary URL)
- `omega/trace/schema.py`, `omega/trace/store.py`, `omega/trace/market_snapshot.py` (V11)
- `omega/core/calibration/fitter.py` (`include_early_snapshots`, `EARLY_MARKET_SLICE`)

**Tests:** `tests/core/test_replay_wnba.py`, `test_calibration_early_snapshot_isolation.py`,
`tests/trace/test_early_market_snapshots.py`, `tests/integrations/test_espn_wnba.py`,
`test_wehoop.py`; version-pin + replay-guard scans updated.

**M1 acceptance — all met:** WNBA replays deterministically; CLV reads only
`closing_lines` (bit-identical with 0 vs 50 early rows); calibration does not
drift on inflated early-line EV; opt-in routes early traces to a dedicated slice.

---

## 4. How to run things

```bash
# Full test suite (must stay green)
python -m pytest tests/ -q

# Targeted M1 suites
python -m pytest tests/core/test_replay_wnba.py tests/integrations/test_wehoop.py -q

# Early-line capture (needs OMEGA_ODDS_API_KEY; --dry-run is safe)
python scripts/capture_early_lines.py --leagues WNBA --dry-run

# WNBA backtest artifacts from wehoop (network on cold pull; cached after)
python scripts/refresh_wehoop.py --season 2025 --dry-run
```

Replay safety: set `OMEGA_REPLAY_MODE=1` to block all live fetches (cached pulls
still served).

---

## 5. Implementing the remaining milestones

General recipe for each new sport (mirror M1):
1. **Backend** in `omega/core/simulation/<sport>.py` conforming to
   `GameSimulationBackend`, emitting V10 distribution rows. Register at engine
   import time.
2. **League config(s)** in `leagues.py` with `default_game_backend`,
   `default_prop_backend`, and any sport flags.
3. **Odds API sport keys** in `omega/integrations/odds_api.py` `SPORT_KEY_MAP`.
4. **Adapters** in `omega/integrations/` using the ETL harness (cached + validate
   + alias). Live and historical as needed.
5. **Priors tables** (new schema version) + **fit scripts** in `scripts/`.
6. **Gatherer wiring**: inject priors into the request (see §5.5 seam note).
7. **Tests**: replay determinism (`tests/core/test_replay_<sport>.py`), backend
   contract, adapter ETL tests, calibration bootstrap; bump version-pin tests.

### 5.1 Milestone 2 — Soccer (World Cup) — DEADLINE 2026-06-11
- Backend `omega/core/simulation/soccer_bivariate_poisson.py`,
  `backend_name="soccer_bivariate_poisson_dc"`. Bivariate Poisson + Dixon-Coles
  low-score correction. **`supports_draw=True`** is the key contract delta — the
  `soccer` archetype already exists; confirm `result_type="team_score"` works and
  draws propagate through `_build_team_score_result` (it already handles
  `supports_draw`).
- **`rho` is a required dynamic prior** — backend raises
  `MissingDixonColesPriorError` if absent (fail closed → `status="skipped"`,
  `missing_requirements=["rho_prior"]`). **Do not hardcode a default rho.**
- New: `priors_dixon_coles` table (schema V12) + `scripts/fit_dixon_coles.py`
  (minimise DC negative log-likelihood per competition profile, e.g.
  `fifa_intl_v1`). Fit from StatsBomb Open Data.
- Adapters: `omega/integrations/statsbomb.py` (historical, primary fit source),
  `understat.py` (current season), `fbref.py` (redundancy) → `priors_xg` table.
- Edge consumer: `omega/core/edge/soccer_derivatives.py` (AH/BTTS/correct-score/
  first-half-total) reading score-distribution rows. Add `SoccerDerivativeMarket`
  enum to `schemas.py`.
- League config `FIFA_WORLD_CUP_2026` with `rho_fit_profile: "fifa_intl_v1"`,
  `supports_draw: True`, `home_advantage: 0.0`.
- Acceptance: 10 EPL/UCL matches replay with non-zero `draw_prob`; DC low-score
  cell tests; skip-when-rho-missing test; `rho` provenance in sidecar.

### 5.2 Milestone 3 — Tennis (Wimbledon target 2026-06-29)
- Backend `omega/core/simulation/tennis_markov.py`,
  `backend_name="tennis_markov_iid"`. Closed-form game/set/match, with a
  per-game finite-state Markov chain at pressure nodes. Tennis archetype is
  `individual_matchup` (required keys `serve_win_pct`, `return_win_pct`) — the
  generic markov body does **not** apply; write tennis math directly.
  Emit `distribution_type="markov_closed_form"` so calibration uses exact probs.
  `draw_prob=0.0`, no team_score.
- `pressure_coefficients` injected via `request.prior_payload` (see §5.5).
- Also register prop backend `"tennis_prop_serve"` (already routed for
  `player_aces`; until registered, `player_aces` falls back to the distribution
  router with a recorded note — registering it is a drop-in upgrade).
- New: `priors_tennis`, `priors_tennis_pressure` tables + adapter
  `omega/integrations/tennis_sackmann.py` + `scripts/fit_tennis_pressure_coefficients.py`
  (group fallback below N=500 charted points; never silent 0.0).
- League configs `ATP`/`WTA` already exist — add `default_game_backend`.
- Acceptance: 20 mixed-surface matches replay; closed-form vs 100k MC within
  0.5%; pressure-coefficient ablation shifts derivative markets; provenance in sidecar.

### 5.3 Milestone 4 — NFL (kickoff 2026-09-10)
- Backends: `omega/core/simulation/nfl_neg_binom.py`
  (`backend_name="nfl_neg_binom"`, Gamma-Poisson team scores, discrete margin
  distribution over {-21..+21}) and `omega/core/simulation/prop_neg_binom.py`
  (`backend_name="prop_neg_binom"`, NB sampler — **register it**, it is already
  routed for NFL yardage; until then NFL yardage falls back to the distribution
  router with a recorded note — correct, but not NB-accurate for the over-dispersed
  tail).
- New: `priors_nfl_dispersion` table + adapter `omega/integrations/nflverse.py`
  + `scripts/fit_nfl_dispersion.py` with **mandatory hierarchical Bayesian
  shrinkage** (`w(n)=n/(n+n0)`, `n0=8`; `nb_k_source` ∈ player|position_group|league).
  Backend stays shrinkage-agnostic — all hierarchy is offline.
- Edge: `omega/core/edge/nfl_teasers.py` (Wong legs from the margin distribution).
- `leagues.py` NFL: set `default_game_backend="nfl_neg_binom"`, `teaser_evaluation: True`.
- Also build `omega/integrations/espn_nfl.py` + NFL stat keys.
- Acceptance: 20 games replay; teaser EV on classic Wong ranges; shrinkage unit
  test (rookie WR n=30 → `position_group`, w<0.6); tail-prob test.

### 5.4 Prop backend stub still needed
`omega/core/simulation/prop_distribution_router.py` is referenced in the design
file list but the router currently lives as `PropDistributionRouterBackend` in
`engine.py` (registered). That is fine — no separate module required unless you
prefer to extract it. `prop_neg_binom` and `tennis_prop_serve` are **routed but
not registered**; register them in M4/M3 respectively. **As of the M0 hardening
pass the live prop path falls back to `prop_distribution_router` for these
unregistered names (with a note on the response), so nothing breaks in the
interim** — registering the real backend silently upgrades the math.

### 5.5 ⚠️ Open seam: how game-level priors reach the backend
`GameSimulationInput` (`backends.py`) currently has **no `prior_payload` field** —
only `PropSimulationInput` does. The design says soccer `rho` and tennis
`pressure_coefficients` flow through `request.prior_payload`. **Decide early in
M2:** either
- (a) add `prior_payload: dict | None = None` to `GameSimulationInput` and have
  `service.analyze_game` populate it from the gatherer, **or**
- (b) route these priors through `home_context`/`away_context` (the plan's
  default style for `xg_*`, `spw_pct`, etc.).
The plan text uses both phrasings; (a) is cleaner for non-team-scoped priors like
`rho`. Whichever you pick, keep it consistent across M2–M4 and document it.

---

## 6. Conventions / gotchas (learned in M0–M1)

- **Register backends at engine import time**; new sport modules lazy-import
  engine helpers inside `run()` to avoid cycles.
- **Game backends declare `evidence_mode`** (`"plane_adjustment"` |
  `"markov_transition"`); evidence dispatch reads that attribute, never the
  backend name. New backends default to plane-adjustment unless they set it.
- **Bump version-pin tests** whenever `CURRENT_VERSION` changes
  (`tests/trace/test_schema_v7.py`, `test_schema_v9.py`).
- **`TraceStore.schema_version()` is a method, not a property.**
- **ETL replay guard scan** will fail CI if a new `omega/integrations/*.py`
  omits `assert_not_replay_mode`.
- **`data/cache/` is gitignored**; `data/aliases/` and generated
  `data/backtest_artifacts/` are not cache — aliases are versioned, artifacts are
  regenerable (not committed).
- **Bounded autonomy (CLAUDE.md) is unchanged**: backends/scripts produce all
  probabilities; the LLM never authors edge/EV/Kelly/tier/trace_id.
- **Determinism**: every backend must produce identical output for the same seed.
  Add a `tests/core/test_replay_<sport>.py` proving it.

---

## 7. Quick-start for the next session

1. `git log --oneline -6` — confirm you are on/after `b709c99`.
2. `python -m pytest tests/ -q` — confirm green (993+).
3. Read `docs/phase7/MULTI_SPORT_EXPANSION.md` §M2 + this §5.1.
4. Resolve the §5.5 prior-payload seam decision first.
5. Start M2 (Soccer) — it has the binding 2026-06-11 deadline.
