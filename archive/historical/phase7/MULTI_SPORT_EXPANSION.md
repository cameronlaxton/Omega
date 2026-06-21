> [!NOTE]
> This document is from a legacy phase that has been implemented and merged to `main`. It is retained here for historical reference.

# Phase 7 Design Plan — Multi-Sport Expansion (WNBA, Soccer, Tennis, NFL)

## Context

Phase 6 closed with stable trace persistence, atomic session sidecars (`audit_events`), and a working calibration loop. The engine is sport-agnostic at every layer except simulation: `TraceStore`, calibration profiles, FrozenArtifacts, and session sidecars all key on a free-form `league` string (`omega/trace/store.py:85`, `omega/core/calibration/profiles.py:42`, `omega/strategy/artifacts.py:21`, `omega/trace/session_sidecar.py:45`).

The single seam that blocks multi-sport expansion is **simulation**. Two backends exist today — `FastScoreSimulationBackend` and `MarkovGameSimulationBackend` (`omega/core/simulation/engine.py:1091`, `:1206`) — dispatched by a hardcoded string check at `omega/core/contracts/service.py:819`. Player props live in standalone functions (`engine.py:505`) and have no Negative Binomial sampler. Both gaps must close before Phase 7 sports can land.

Phase 7 adds four sports in this order — WNBA (immediate), Soccer (2026-06-11 World Cup deadline), Tennis (Wimbledon onward), NFL (fall build) — with bounded, sport-appropriate math:

- **WNBA** — Markov possession model (NBA fork).
- **Soccer** — Bivariate Poisson with Dixon-Coles low-score correction.
- **Tennis** — IID Markov chain over serve points (closed-form game/set/match).
- **NFL** — Negative Binomial yardage props, Gamma-Poisson team scores, Wong-teaser-aware spread/total evaluation.

The bounded-autonomy invariant from `CLAUDE.md` is unchanged: the LLM never produces edge%, EV%, Kelly, units, tier, or trace_id. Every probability and every distribution row in this document comes from the deterministic Python engine.

---

## Architectural decisions

Eight decisions, locked before design, drive every section below:

1. **Backend registry** replaces the hardcoded `{"fast_score", "markov_state"}` switch in `service.py:819`. New sport backends register at import time; dispatch is a single dictionary lookup.
2. **`PropSimulationBackend` Protocol** parallel to `GameSimulationBackend`. Existing prop functions migrate to a default `prop_distribution_router` backend. Negative Binomial and bivariate-Poisson samplers ship as new backends, not as branches inside `select_distribution()`.
3. **External-priors adapters are part of Phase 7**: Jeff Sackmann CSV (tennis SPW%/RPW%), Understat + FBref (soccer xG), nflverse-derived dispersion (NFL NB `k`). Each writes to a dedicated SQLite priors table read by its respective backend.
4. **Implementation order: WNBA → Soccer → Tennis → NFL.** Soccer is built before tennis because the World Cup has a fixed external kickoff date (2026-06-11). Tennis ships in time for Wimbledon (2026-06-29).
5. **Dixon-Coles `rho` is a dynamic prior, not a config constant** (red-team finding 1). Fit per competition profile by `omega-fit-dixon-coles`, persisted in `priors_dixon_coles`, injected via `request.prior_payload`. Backend fails closed if absent.
6. **NFL NB dispersion uses hierarchical Bayesian shrinkage** (red-team finding 2). Per-player `k` is shrunk toward `(position_group, stat_type)` posteriors in `omega-fit-nfl-dispersion`. Backend stays shrinkage-agnostic; provenance (`nb_k_source`, `nb_k_shrinkage_weight`) flows through the priors table.
7. **Tennis SPW% is state-dependent** (red-team finding 3). `TennisMarkovBackend` accepts a `pressure_coefficients` dict in `request.prior_payload` and applies per-state additive deltas at pressure nodes (break points, tiebreaks, set/match points). Fit by `omega-fit-tennis-pressure-coefficients`; group-fallback used below the N=500 charted-points threshold.
8. **Early WNBA captures are segregated from CLV and calibration** (red-team finding 4). A dedicated `early_market_snapshots` table holds low-liquidity early lines. `closing_lines` is unchanged. CLV reads only `closing_lines`. The calibration fitter excludes early snapshots by default; opt-in forces a separate `context_slice="early_market_low_liq"` partition.

---

## Part 1 — Goals and non-goals

### Goals

- Four sport-specific game-simulation backends conforming to `GameSimulationBackend` Protocol (`omega/core/simulation/backends.py:69`).
- One new prop-simulation Protocol with Negative Binomial + distribution-router backends.
- League configs and Odds API mappings for WNBA, ATP, WTA, FIFA World Cup 2026, and updated NFL.
- External-priors adapters with replay-mode guarding.
- Replay-determinism tests per sport, mirroring `tests/core/test_replay_0526_mlb.py`.
- Session-sidecar event parity: every new backend and adapter emits to `var/inbox/sessions/<session_id>.json` via the unmodified writer.

### Non-goals

- **No in-game / live betting model** for any new sport in Phase 7. Pre-match and pre-event only.
- **No futures-market modeling** (championship odds, MVP, top-scorer). Game-level + props only.
- **No prediction-market arbitrage** (Polymarket, Kalshi) for the World Cup despite obvious interest.
- **No new orchestrator/reasoning routing** — `league` remains a free-form string. Evidence handlers and gather slots stay generic.
- **No new top-level package**. All new files live under `omega/core/simulation/`, `omega/core/edge/`, and `omega/integrations/`. Phase 6's `schema_version=10` is sufficient; trace schema is not bumped.

---

## Part 2 — Backend registry refactor (precondition)

This refactor must land first. Every subsequent milestone depends on it, and it cannot break NBA or MLB regressions.

### Changes to `omega/core/simulation/backends.py`

Add a `GAME_BACKENDS` registry and `register_game_backend()` function:

```python
GAME_BACKENDS: dict[str, GameSimulationBackend] = {}

def register_game_backend(name: str, backend: GameSimulationBackend) -> None:
    if name in GAME_BACKENDS:
        raise ValueError(f"game backend {name!r} already registered")
    GAME_BACKENDS[name] = backend

def resolve_game_backend(name: str) -> GameSimulationBackend | None:
    return GAME_BACKENDS.get(name)
```

Add a parallel `PropSimulationBackend` Protocol, `PropSimulationInput` dataclass, and `PROP_BACKENDS` registry:

```python
@dataclass(frozen=True)
class PropSimulationInput:
    player_name: str
    league: str
    stat_type: str
    line: float
    projection_mean: float
    n_iter: int
    seed: int | None = None
    projection_std: float | None = None
    prior_payload: dict[str, Any] | None = None  # xG, SPW%, NB k, etc.

class PropSimulationBackend(Protocol):
    backend_name: str
    component_version: str
    def run(self, request: PropSimulationInput) -> dict[str, Any]: ...

PROP_BACKENDS: dict[str, PropSimulationBackend] = {}

DEFAULT_PROP_BACKEND_BY_LEAGUE_STAT: dict[tuple[str, str], str] = {
    # NBA/WNBA: existing distribution router covers everything.
    # NFL: yardage stats use Negative Binomial; discrete count stats use Poisson via router.
    ("NFL", "rushing_yards"):    "prop_neg_binom",
    ("NFL", "receiving_yards"):  "prop_neg_binom",
    ("NFL", "passing_yards"):    "prop_neg_binom",
    ("NFL", "longest_rush"):     "prop_neg_binom",
    ("NFL", "longest_reception"):"prop_neg_binom",
    ("NFL", "passing_tds"):      "prop_distribution_router",
    ("NFL", "rushing_tds"):      "prop_distribution_router",
    # Tennis: ace / total games handled by tennis-aware prop backend (Milestone 3).
    ("ATP", "player_aces"):      "tennis_prop_serve",
    ("WTA", "player_aces"):      "tennis_prop_serve",
}

def resolve_default_prop_backend(league: str, stat_type: str) -> str:
    return DEFAULT_PROP_BACKEND_BY_LEAGUE_STAT.get(
        (league, stat_type), "prop_distribution_router"
    )
```

The `PropSimulationBackend` contract enforces V10 distribution rows for the prop's stat target. Reuse `validate_distribution_rows()` from `backends.py:79`.

### Changes to `omega/core/contracts/service.py:819`

Replace:

```python
if request.simulation_backend not in {"fast_score", "markov_state"}:
    return GameAnalysisResponse(..., status="skipped", ...)
use_markov = request.simulation_backend == "markov_state"
engine = (
    OmegaSimulationEngine(game_backend=MarkovGameSimulationBackend())
    if use_markov
    else _engine
)
```

with:

```python
backend = resolve_game_backend(request.simulation_backend)
if backend is None:
    return GameAnalysisResponse(
        matchup=matchup,
        league=request.league,
        analyzed_at=now,
        status="skipped",
        skip_reason=f"Unsupported simulation_backend={request.simulation_backend!r}",
        missing_requirements=["simulation_backend"],
        context_source="missing",
    )
engine = OmegaSimulationEngine(game_backend=backend) if backend.backend_name != "fast_score" else _engine
```

The `fast_score` special-case preserves the shared singleton `_engine` and keeps Phase 6 replay tests bit-identical.

> **Implementation note — M0 hardening (2026-06-01).** This `backend_name != "fast_score"` special-case was later removed. `OmegaSimulationEngine.run_fast_game_simulation` now takes a per-call `backend` argument, so the shared `_engine` runs *any* registered backend with no name check and no per-call re-instantiation. Replay stays bit-identical. Current state: `PHASE7_HANDOFF.md` §2.3.

### Changes to `omega/core/simulation/engine.py`

At module import time, register the two existing backends under their existing names:

```python
register_game_backend("fast_score",   FastScoreSimulationBackend())
register_game_backend("markov_state", MarkovGameSimulationBackend())
```

Wrap `run_player_simulation()` (`engine.py:505`) as a `PropDistributionRouterBackend` and register it:

```python
register_prop_backend("prop_distribution_router", PropDistributionRouterBackend())
```

`PropDistributionRouterBackend.run()` delegates verbatim to `select_distribution()` + the existing sampler logic. No behavior change for NBA/MLB props.

> **Implementation note — M0 hardening (2026-06-01).** `service.analyze_player_prop` now dispatches through the prop registry (`resolve_default_prop_backend` → `resolve_prop_backend`) rather than calling `run_player_simulation` directly. Routed-but-unregistered names (`prop_neg_binom`, `tennis_prop_serve`) fall back to `prop_distribution_router` with a note on the response. The router forwards the caller `distribution` override and `dud_prob` through `prior_payload`, keeping NBA/MLB props bit-identical. Current state: `PHASE7_HANDOFF.md` §2.2.

### Acceptance gate for Milestone 0

- `tests/core/test_replay_0526_mlb.py` passes unchanged.
- All existing NBA replay-determinism tests pass unchanged.
- Unit test in `tests/core/test_backend_registry.py`:
  - `resolve_game_backend("fast_score")` and `("markov_state")` both return non-None.
  - `resolve_game_backend("nonexistent")` returns None.
  - `register_game_backend("fast_score", ...)` raises `ValueError`.

---

## Part 3 — Schema changes

### `omega/core/config/leagues.py`

**WNBA** (already present at `leagues.py:28`): keep numeric fields; add three keys.

```python
"WNBA": {
    "sport": "basketball",
    "archetype": "basketball",
    "periods": 4,
    "period_length_min": 10,
    "scoring": "points",
    "avg_total": 160.0,
    "avg_pace": 80.0,
    "distribution": "normal",      # legacy fallback for FastScore path
    "home_advantage": 2.5,
    "std": 10.0,
    "default_game_backend": "markov_state_wnba",
    "default_prop_backend": "prop_distribution_router",
    "liquidity_profile": "low",    # triggers early-morning line capture
},
```

**ATP / WTA** (new):

```python
"ATP": {
    "sport": "tennis",
    "archetype": "tennis",
    "scoring": "points",
    "default_game_backend": "tennis_markov_iid",
    "default_prop_backend": "prop_distribution_router",
    "default_match_format": "best_of_3",   # Grand Slam override per-event
    "supports_draw": False,
},
"WTA": {
    "sport": "tennis",
    "archetype": "tennis",
    "scoring": "points",
    "default_game_backend": "tennis_markov_iid",
    "default_prop_backend": "prop_distribution_router",
    "default_match_format": "best_of_3",
    "supports_draw": False,
},
```

Tennis has no team-score, no `avg_total`, no `avg_pace`. The Markov backend reads SPW%/RPW%/surface from `request.home_context` / `request.away_context`.

**FIFA_WORLD_CUP_2026** (new):

```python
"FIFA_WORLD_CUP_2026": {
    "sport": "soccer",
    "archetype": "soccer",
    "scoring": "goals",
    "avg_total": 2.7,
    "home_advantage": 0.0,          # neutral venue
    "distribution": "bivariate_poisson",
    "default_game_backend": "soccer_bivariate_poisson_dc",
    "default_prop_backend": "prop_distribution_router",
    "supports_draw": True,
    "rho_fit_profile": "fifa_intl_v1",  # selects empirically-fit rho profile
},
```

Note: `rho` (Dixon-Coles correlation) is **not** stored in the static league config. International tournament soccer has materially different draw propensity and scoring variance than domestic club competition, so a single hardcoded value is brittle. Instead, the config carries a `rho_fit_profile` key that selects an empirically-fit Dixon-Coles profile from a new `priors_dixon_coles` table. The fitted `rho` is loaded by the gatherer and passed into the backend via `request.prior_payload["rho"]`. See Part 4 for the backend contract and Part 5 for the fit pipeline.

Existing soccer leagues (`MLS`, `EPL`, etc.) are untouched in Milestone 2; they continue to use `fast_score` until separately re-tuned and would carry their own `rho_fit_profile` when promoted.

**NFL** (already present at `leagues.py:77`): switch default backend; keep Normal-distribution fields as the FastScore fallback.

```python
"NFL": {
    ...existing fields...
    "default_game_backend": "nfl_neg_binom",
    "teaser_evaluation": True,      # consumed by omega/core/edge/nfl_teasers.py
},
```

### `omega/core/contracts/schemas.py`

- `GameAnalysisRequest`: **no new required fields**. Sport-specific priors flow through the existing free-form `home_context` / `away_context` dicts (`xg_home`, `xg_against_home`, `spw_pct`, `rpw_pct`, `nb_dispersion_k`, etc.). This matches the current style for `pts_mean` / `pass_yds_mean` and avoids a schema bump.
- `PlayerPropRequest`: add `prior_payload: dict | None = None`. Backwards compatible — existing callers omit it.
- New enum `SoccerDerivativeMarket`:

```python
class SoccerDerivativeMarket(str, Enum):
    asian_handicap          = "asian_handicap"
    total_goals_over_under  = "total_goals_over_under"
    both_teams_to_score     = "both_teams_to_score"
    correct_score           = "correct_score"
    first_half_total        = "first_half_total"
```

Used by `omega/core/edge/soccer_derivatives.py`. `EdgeDetail` itself is unchanged.

- **No changes** to `BetCard`, `BetSlip`, `EdgeDetail`, `SimulationResult`, or trace schemas.

---

## Part 4 — Engine backend stubs

All four backends live in `omega/core/simulation/`, each in its own module, each conforming to `GameSimulationBackend` Protocol and emitting V10 distribution rows (`backends.py:33`) so `TraceStore` writes through unchanged.

### `omega/core/simulation/markov_wnba.py`

```python
class MarkovWNBAGameSimulationBackend:
    backend_name = "markov_state_wnba"
    component_version = "markov_wnba_v1"

    def run(self, request: GameSimulationInput) -> dict[str, Any]:
        """Reuses MarkovSimulator with WNBA-tuned pace/efficiency.

        Tuning targets (anchored to 2025 WNBA league averages):
          - possessions_per_game_baseline = 80
          - off_efficiency_baseline       = 100.0 pts / 100 poss
          - period_length_min             = 10   (per leagues.py:33)
          - free_throw_rate, three_point_rate, turnover_rate read from
            league_baselines["WNBA"] in omega/core/sport_baselines.py.

        Wraps the same _build_team_score_result() helper used by
        MarkovGameSimulationBackend (engine.py:1206). No new math.
        """
```

Lowest-risk backend — same engine, different constants.

### `omega/core/simulation/soccer_bivariate_poisson.py`

```python
class SoccerPoissonBackend:
    backend_name = "soccer_bivariate_poisson_dc"
    component_version = "soccer_bvp_dc_v1"

    def run(self, request: GameSimulationInput) -> dict[str, Any]:
        """Bivariate Poisson with Dixon-Coles low-score correction.

        Inputs (from request.home_context / away_context):
          - xg_home, xg_away             : team xG attack (Understat/FBref)
          - xg_against_home, xg_against_away : defensive xG conceded

        Inputs (from request.prior_payload — REQUIRED, not in static config):
          - rho                          : Dixon-Coles correlation parameter,
                                           empirically fit per competition profile
                                           (e.g. "fifa_intl_v1" vs "epl_v3").
                                           Loaded from priors_dixon_coles by the
                                           gatherer; backend raises
                                           MissingDixonColesPriorError if absent
                                           rather than using a default.
          - rho_profile_id, rho_as_of_date : provenance for audit events.

        Rationale for dynamic rho: International competition exhibits
        meaningfully different draw propensity and scoring variance than
        domestic club soccer. A single hardcoded league rho is unstable
        across competitions; per-competition empirical fits are mandatory.

        Math:
          1. lambda_home = xg_home * (xg_against_away / league_avg_xga)
             lambda_away = xg_away * (xg_against_home / league_avg_xga)
             Normalize to league mean total (2.7 for World Cup).
          2. Sample N=request.n_iterations from a bivariate Poisson
             with Dixon-Coles tau adjustment on cells (0,0), (1,0), (0,1), (1,1).
             tau(0,0) = 1 - lambda_home*lambda_away*rho
             tau(1,0) = 1 + lambda_away*rho
             tau(0,1) = 1 + lambda_home*rho
             tau(1,1) = 1 - rho
          3. Derive home_win_prob, away_win_prob, draw_prob,
             predicted_total, predicted_spread. supports_draw=True is the
             key contract delta vs. NBA/MLB backends.
          4. Emit V10 distribution rows for:
               total_goals, home_goals, away_goals,
               home_clean_sheet, away_clean_sheet, both_teams_to_score.

        Standard Poisson is forbidden — it under-predicts draws and 0-0.
        """
```

Asian Handicap and Over/Under derivative markets are evaluated downstream by `omega/core/edge/soccer_derivatives.py`, which consumes the score-distribution rows. The backend is not aware of handicap lines — separation of concerns matches existing edge logic.

### `omega/core/simulation/tennis_markov.py`

```python
class TennisMarkovBackend:
    backend_name = "tennis_markov_iid"
    component_version = "tennis_markov_iid_v1"

    def run(self, request: GameSimulationInput) -> dict[str, Any]:
        """IID Markov chain over serve points; closed-form match probabilities.

        Inputs (from request.home_context / away_context):
          - spw_pct_home, spw_pct_away   : server's point-win % (Sackmann),
                                           the baseline (state-free) value.
          - rpw_pct_home, rpw_pct_away   : returner's point-win %
            Per-match priors blended via Bradley-Terry-style opponent
            adjustment so individual signal is preserved even though the
            population-level identity is rpw_X == 1 - spw_opp.
          - match_format                 : "best_of_3" | "best_of_5"
            (league config + per-event tournament override)
          - surface                      : "hard" | "clay" | "grass" — applies
            additive surface coefficient to SPW% from priors_tennis.

        Inputs (from request.prior_payload):
          - pressure_coefficients        : dict of state -> additive delta on
                                           SPW%, fit empirically per player
                                           (or per surface/tour fallback) by
                                           omega/integrations/tennis_sackmann.py.
                                           Required states:
                                             "break_point_against"     (server facing BP)
                                             "set_point_serving"       (server at SP)
                                             "match_point_serving"     (server at MP)
                                             "tiebreak"                (game-level)
                                             "serving_for_set"         (5-3, 5-4 etc.)
                                             "serving_for_match"       (final set)
                                           Each value is a signed delta applied to
                                           the baseline SPW% for that point only.
                                           Missing keys default to 0.0 (IID).

        Rationale for state-dependent p: Point outcomes in ATP/WTA matches
        are NOT strictly IID. Pressure states measurably shift serving
        distributions. A flat SPW% misprices derivative markets (set winner,
        first-set games handicap, game-by-game live derivatives) and tail
        outcomes (5-7 deciders). The backend stays computationally cheap by
        applying pressure-state deltas only at points whose state is one of
        the named keys; non-pressure points use baseline SPW%.

        Math:
          1. p_server_holds = closed-form polynomial in SPW% (Newton 1962),
             evaluated at the *baseline* SPW%. For games containing pressure
             points (break points, etc.) the closed-form is replaced with a
             per-game finite-state Markov chain that applies the per-state
             SPW% delta only at the relevant nodes; result is exact, not MC.
          2. p_set_won = absorbing Markov chain over games at 6-6 tiebreak.
          3. p_match_won = absorbing Markov chain over sets,
             respecting best-of-3 / best-of-5.
          4. Sample N=request.n_iterations Bernoulli match outcomes for
             distribution-row consistency. Also emit closed-form
             probabilities with distribution_type="markov_closed_form" so
             calibration uses exact values, not noisy MC.

        Tennis has no draws, no team_score. SimulationResult fields:
          - home_win_prob   = p_match_player_home
          - away_win_prob   = 1 - home_win_prob
          - draw_prob       = 0.0
          - predicted_home_score, predicted_away_score = expected sets won
          - predicted_total = expected total games (over/under games market)
          - predicted_spread = predicted_home_score - predicted_away_score
        """
```

Tennis distribution rows: `match_winner`, `set_winner_set_1`, `total_games_match`, `total_games_set_1`, `player_a_wins_a_set` (insurance market).

### `omega/core/simulation/nfl_neg_binom.py`

```python
class NflSimulationBackend:
    backend_name = "nfl_neg_binom"
    component_version = "nfl_nb_v1"

    def run(self, request: GameSimulationInput) -> dict[str, Any]:
        """Game-level: Gamma-Poisson team scores. Margin distribution exposed.

        Game-level math:
          1. Team-score lambda priors from request.home_context (offensive
             EPA/play, plays_per_game, opponent defensive EPA-against).
          2. Sample team_score ~ NegativeBinomial(mean=lambda, k=dispersion)
             rather than Normal(mean=lambda, sd=10). Captures the heavy upper
             tail (40+ point outputs) that Normal under-prices.
          3. Emit predicted_total, predicted_spread, win probs.

        Wong teaser support:
          Distribution rows include p(home_margin == n) for n in {-21..+21}
          so omega/core/edge/nfl_teasers.py can evaluate non-linear value at
          the 3-point and 7-point crossings. The simulation backend exposes
          the discrete margin distribution; teaser EV math is done outside
          the backend.
        """
```

### `omega/core/simulation/prop_neg_binom.py`

```python
class NegBinomPropBackend:
    backend_name = "prop_neg_binom"
    component_version = "prop_nb_v1"

    def run(self, request: PropSimulationInput) -> dict[str, Any]:
        """Negative Binomial sampler for over-dispersed prop targets.

        Routed to for NFL yardage and longest-play markets via
        DEFAULT_PROP_BACKEND_BY_LEAGUE_STAT. Also routed for any prop whose
        request.prior_payload includes an explicit nb_dispersion_k.

        Math:
          mean = request.projection_mean
          k    = request.prior_payload["nb_dispersion_k"]   # smaller k => fatter tail
          p    = k / (k + mean)
          sample ~ NegativeBinomial(k, p)
        Emits over_prob, under_prob, p10/p50/p90, distribution_params for V10.

        Source of k: priors_nfl_dispersion table, populated by
        omega-fit-nfl-dispersion. The fitter mandates Bayesian
        shrinkage toward position-group means (see Part 5). Backend
        treats k as opaque; all hierarchical-pooling logic lives in the
        offline fitter so per-call latency stays flat. The
        prior_payload also carries nb_k_shrinkage_weight and
        nb_k_source ("player" | "position_group" | "league") for
        audit-event provenance.
        """
```

### `omega/core/simulation/prop_distribution_router.py`

Thin shim wrapping the existing `select_distribution()` + sample logic from `engine.py:505`. Behavior is bit-identical to today; the migration is mechanical.

---

## Part 5 — Data integration plan per sport

### WNBA — fully reused infrastructure + one new module

- **Odds API**: WNBA already mapped at `odds_api.py:55` (`basketball_wnba`). No change.
- **ESPN scoreboard + boxscore (live)**: new file `omega/integrations/espn_wnba.py`, cloned from `espn_nba.py` with URL `basketball/wnba/scoreboard` and a WNBA team-alias map. Reuse `parse_box_score()` (`espn_boxscore.py:211`) with a new `WNBA_STAT_KEYS` constant. This is the in-season fetch path.
- **`wehoop` (backtest/historical)**: the richer historical source for WNBA replay artifacts — play-by-play, box scores, shot locations (`github.com/sportsdataverse/wehoop`). `wehoop` is R-native; port via its data exports / API rather than an R bridge, following the same snapshot-and-load pattern used for nflverse. Feeds backtest artifacts and calibration, not the live request path. See Part 5B.
- **Player-prop normalization**: WNBA stat keys identical to NBA (pts/reb/ast/stl/blk/3pm/pra). Reuse the NBA distribution router.
- **Closing-line capture**: extend `omega-fetch-closing-lines` to schedule an extra 6am ET pass for any league with `liquidity_profile == "low"`. Add a cron-callable shim `omega-capture-early-lines` that takes `--leagues WNBA` and writes to a dedicated `early_market_snapshots` table — **not** the canonical `closing_lines` table. Early WNBA markets are highly inefficient and move violently on sharp action, so blending them into CLV destroys the metric. The dedicated table is keyed on `(trace_id, league, captured_at)` with a `liquidity_profile` column copied from the league config. The canonical CLV computation reads only `closing_lines`; the `early_market_snapshots` table is consumed separately by an "early-market lean" analysis and is **excluded from calibration fits by default**. To opt a profile in, the calibration fitter must explicitly pass `include_early_snapshots=True`, which forces a separate calibration profile slice (`context_slice="early_market_low_liq"`) so promoted profiles do not inherit phantom edges from pre-move pricing. See Part 8 for the CLV-distortion red-team detail.
- **Replay-mode guard**: new module must call `assert_not_replay_mode()` from `omega/integrations/_guards.py:11`. The shape-validator test in `tests/integrations/test_replay_mode_guard.py` will be extended to enforce this for every new integration module.

### Soccer (World Cup) — new external xG sources

- **Odds API**: add `"FIFA_WORLD_CUP_2026": "soccer_fifa_world_cup"` to `SPORT_KEY_MAP` (`odds_api.py:53`). Markets: `h2h_3way` (3-way moneyline including draw), `spreads` (Asian handicaps), `totals` (over/under goals), `btts` (both teams to score).
- **ESPN**: no parity for player-level soccer stats. Skip.
- **New file `omega/integrations/statsbomb.py` (backtest/historical)**: adapter over StatsBomb Open Data (`github.com/statsbomb/open-data`) — event-level xG, freeze-frame pitch data, and pass-sequence context. Phase 7 consumes only team offensive/defensive xG aggregates into `priors_xg` and into the Dixon-Coles fit dataset; freeze-frame and pass context are reserved for a future shot-quality refinement. This is the primary source for the `fifa_intl_v1` historical fit because it is free, frozen, and reproducible.
- **New file `omega/integrations/understat.py` (current season)**: adapter that pulls team and player xG from understat.com. HTML-backed; cache aggressively. Writes to `priors_xg` indexed by `(team, season, last_updated)`.
- **New file `omega/integrations/fbref.py`**: FBref scraper, used as a redundancy source. Understat, FBref, and StatsBomb xG agree within a few percent at the season level; we keep them and surface disagreement in the session sidecar as a `data_provenance` audit event.
- Both integrations call `assert_not_replay_mode()`. Both bound to a daily-refresh cadence; in-tournament refresh after each matchday. Priors freeze at tournament kickoff so mid-event source breakage cannot poison live decisions.
- **New file `omega-fit-dixon-coles` + `priors_dixon_coles` table**: Dixon-Coles `rho` is fit empirically per *competition profile*, not per league. Profile codes use the form `<competition>_<scope>_<version>`, e.g. `fifa_intl_v1` (FIFA-level international tournament), `epl_v3`, `ucl_v2`. The fitter consumes a historical match dataset filtered to the competition profile (5+ seasons of FIFA-tier internationals for `fifa_intl_v1`) and minimises the Dixon-Coles negative log-likelihood on (home_goals, away_goals) pairs. Output row: `(profile_id, rho, n_matches, fit_loss, as_of_date)`. Refit cadence: quarterly during a tournament cycle; pinned to a frozen `as_of_date` for the duration of a tournament so live decisions cannot drift on a refit. The gatherer reads the active profile by `rho_fit_profile` from `leagues.py` and injects `rho` + provenance fields into `request.prior_payload`. If no production profile exists for the active `rho_fit_profile`, the gatherer raises `MissingDixonColesPriorError` and the engine returns `status="skipped"` with `missing_requirements=["rho_prior"]` — fail closed rather than guess.

### Tennis (ATP/WTA) — Jeff Sackmann CSVs

- **Odds API**: add `"ATP": "tennis_atp"` and `"WTA": "tennis_wta"` to `SPORT_KEY_MAP`. Markets: `h2h`, `spreads` (sets / games handicaps), `totals` (total games). Player props (`player_aces`, `player_total_games`) covered via explicit market strings.
- **New file `omega/integrations/tennis_sackmann.py`**: loads the Jeff Sackmann match-by-match CSVs from `https://github.com/JeffSackmann/tennis_atp` and `tennis_wta`. The standard open dataset for ATP/WTA stats. Cached into `data/external/sackmann/` and refreshed via a manual `omega-refresh-sackmann` (weekly cadence; freeze for an event's lookahead window once a tournament starts).
- Computes per-player rolling SPW%/RPW% by surface (12-month half-life), writes to a new `priors_tennis` table indexed by `(player, surface, as_of_date)`.
- **New file `omega-fit-tennis-pressure-coefficients` + `priors_tennis_pressure` table**: fits per-player additive SPW% deltas for each pressure state (`break_point_against`, `set_point_serving`, `match_point_serving`, `tiebreak`, `serving_for_set`, `serving_for_match`) using point-by-point Match Charting Project data from the Sackmann ecosystem. Players with fewer than N=500 charted points fall back to a tour+surface group mean (`atp_clay`, `wta_hard`, etc.) — flat 0.0 deltas are never used silently. The fit writes signed-delta values typically in the range `[-0.05, +0.02]` (servers usually take a small hit on pressure). The gatherer joins `priors_tennis_pressure` rows into `request.prior_payload["pressure_coefficients"]` along with the source (`player` | `group_fallback`) for audit events. Without these coefficients, derivative markets (set-winner, set-handicap, total-games-in-a-set) will be systematically mispriced; the IID assumption is a known failure mode of flat closed-form tennis models.
- No ESPN tennis module. Tennis match metadata (court, surface, R16/QF/etc.) read from the Odds API event payload.

### NFL — reuse + new ESPN module + nflverse adapter

- **Odds API**: NFL already at `odds_api.py:58`. Markets: `h2h`, `spreads`, `totals`, `alternate_spreads` (needed for teaser leg pricing), and the standard player-prop markets.
- **New file `omega/integrations/espn_nfl.py`**: cloned from `espn_nba.py`, URL `football/nfl/scoreboard`. Adds `NFL_STAT_KEYS` to `espn_boxscore.py`.
- **New file `omega/integrations/nflverse.py`**: Python-side adapter over the nflverse ecosystem / `nflreadpy` (`github.com/nflverse`) — play-by-play, EPA, WPA, and roster data. Extracts team-level EPA (team-score lambda priors) and player-level yardage variance (NB dispersion `k`). Writes to `priors_nfl_dispersion` keyed by `(player_or_team, stat_type, season)`. The fit step ships as `omega-fit-nfl-dispersion`, run once per week.
- **Hierarchical Bayesian shrinkage is mandatory in `omega-fit-nfl-dispersion`.** Estimating `k` per-player from small NFL sample sizes (17 games, fewer for rookies and backups) produces unstable values that over-fit individual outlier games. The fitter must:
  1. Compute a group-level posterior for `k` per `(position_group, stat_type)` (e.g. `WR/receiving_yards`, `RB/rushing_yards`, `QB/passing_yards`) using all players in the group across multiple seasons.
  2. Compute a per-player posterior for `k` shrunk toward the group posterior via a conjugate prior (Gamma prior over `k`) with shrinkage weight `w(n) = n / (n + n0)` where `n` is per-player game count and `n0` is a tuned pseudocount per position group (initial: `n0=8`).
  3. Persist `nb_dispersion_k`, `nb_k_shrinkage_weight`, `nb_k_source` (`"player"` if `w >= 0.6`, `"position_group"` if `0.2 <= w < 0.6`, `"league"` for cold starts) in `priors_nfl_dispersion`.
  4. Emit a per-fit audit log row showing group means, sample sizes, and the distribution of player-level shrinkage weights so drift can be reviewed weekly.
  - Rationale: without shrinkage, a backup RB's `k` from 30 carries is meaningless — applied to a `longest_rush` over 40.5 yards, the resulting NB tail will trigger false-positive edges. Shrinkage forces small-sample players toward defensible position-group behavior. The backend remains shrinkage-agnostic; all hierarchical logic lives in the offline fitter, and runtime cost is unchanged.
- Source data preference: snapshot the nflverse data files and load via a CSV-based adapter rather than an R bridge. Avoids R/Python-port drift.

---

## Part 5B — Backtestable data sources and ETL pipeline standards

The integrations above cover *live* line and box-score fetching. Phase 7 also introduces **backtestable historical datasets** — the frozen, knowable-at-the-time inputs that feed the quant plane (calibration fits, replay artifacts, dispersion/pressure/xG priors). These are open repositories, not live APIs, and are pulled into a local cache rather than queried per-request.

### Canonical repositories

| Sport | Repository / dataset | Primary data focus | Source |
|-------|----------------------|--------------------|--------|
| Tennis (ATP/WTA) | JeffSackmann `tennis_atp` / `tennis_wta` | Match-by-match results, player statistics, ranking histories, Match Charting Project point-by-point | `github.com/JeffSackmann/tennis_atp`, `.../tennis_wta` |
| NFL | nflverse / `nflreadpy` | Play-by-play, expected points added (EPA), win probability added (WPA), roster data | `github.com/nflverse` |
| Soccer | StatsBomb Open Data | Expected goals (xG), freeze-frame pitch data, pass-sequence context | `github.com/statsbomb/open-data` |
| NBA | `NBA_Play_Types_16_23` (DomSamangy) | Synergy play-type frequencies, points per possession, percentile ranks | `github.com/DomSamangy/NBA_Play_Types_16_23` |
| MLB | `pybaseball` | Statcast pitch-level data, FanGraphs aggregate metrics, Retrosheet historical schedules | `github.com/jldbc/pybaseball` |
| WNBA | `wehoop` | Play-by-play, box scores, shot locations (R package; port via API or Python wrapper) | `github.com/sportsdataverse/wehoop` |

NBA and MLB are existing sports; their repos are listed because Phase 7 uses them to **enrich backtest and calibration data** (Synergy play-type priors for NBA props, Statcast/Retrosheet for richer MLB backtests), not to add new backends. WNBA's `wehoop` data is richer than the ESPN scoreboard scrape and is the preferred *historical* source for WNBA replay artifacts; the live `espn_wnba.py` module remains the in-season fetch path.

### Integration adapters & ETL pipelines

Each source gets a standalone adapter at `omega/integrations/<source>.py` that fetches, normalizes, validates, and stores data locally. Adapters never feed the request path directly — they populate the priors/backtest tables that backends and the strategy plane read.

- **Tennis** — `omega/integrations/tennis_sackmann.py` parses the Sackmann CSVs to compute rolling SPW%/RPW% segmented by surface (12-month half-life) → `priors_tennis`. The Match Charting Project subset feeds `omega-fit-tennis-pressure-coefficients` → `priors_tennis_pressure`.
- **NFL** — `omega/integrations/nflverse.py` extracts team-level EPA and player-level yardage variance. `omega-fit-nfl-dispersion` fits the Negative Binomial dispersion `k` with hierarchical shrinkage and runs weekly, committing to `priors_nfl_dispersion`. EPA/WPA also feed the team-score lambda priors consumed by `NflSimulationBackend`.
- **Soccer** — `omega/integrations/statsbomb.py` (backtest/historical) and `omega/integrations/understat.py` (current season) extract baseline team offensive and defensive xG → `priors_xg`, feeding the bivariate-Poisson engine. StatsBomb freeze-frame/pass-context data is reserved for future shot-quality refinement; Phase 7 consumes only team xG aggregates. `omega/integrations/fbref.py` remains the redundancy cross-check.
- **NBA** — `omega/integrations/nba_play_types.py` loads Synergy play-type frequencies and points-per-possession percentiles → a new `priors_nba_play_types` table, available to the existing NBA prop router for context slicing. No new backend.
- **MLB** — `omega/integrations/pybaseball_adapter.py` wraps `pybaseball` for Statcast/FanGraphs/Retrosheet pulls used to build richer MLB backtest artifacts. No new backend.

### Cross-cutting ETL standards (mandatory for every adapter)

These three constraints are non-negotiable and apply to all adapters, new and existing:

1. **Local caching layer before transform.** Repositories that scrape live sites (e.g. `pybaseball` pulling from Baseball Savant) are rate-limited and risk IP bans. Every adapter must persist the **raw** upstream response to a local cache (Parquet for tabular pulls, raw JSON/HTML otherwise) under `data/cache/<source>/` *before* any transform. Transforms read from the cache, never re-fetch on retry. A cached pull within its TTL must not hit the network at all. This also makes backtests reproducible: the frozen cache *is* the knowable-at-the-time snapshot.

2. **Pydantic schema validation at the ingestion boundary.** Public datasets rename columns without warning. Each adapter defines a Pydantic model for the upstream shape and validates every row/record at ingestion. On validation failure the adapter **fails the job loudly** — raises a typed `SourceSchemaDriftError`, writes a `fail`-status `data_provenance` event to the session sidecar (`var/inbox/sessions/<session_id>.json`), and exits non-zero. It must **never** silently coerce a missing/renamed field to `None` and pass it downstream into the calibration pipeline. Garbage priors are worse than a halted job.

3. **Cross-sport entity resolution via centralized alias tables.** Player and team names differ across databases ("Patrick Mahomes II" vs "Patrick Mahomes", accented vs ASCII tennis names, club name variants). A centralized alias table per league at `data/aliases/<league>.json` intercepts and resolves every entity name **before** it is written to a priors table. Resolution order: exact match → `normalize_player_name()` (`espn_boxscore.py:122`) → alias table → unresolved. Unresolved entities emit a `data_provenance` warning and are **excluded** from the priors write rather than written under an ambiguous key. The alias tables are versioned in git and reviewed when a new source is onboarded.

A shared helper module `omega/integrations/_etl.py` provides the caching decorator, the Pydantic-validate-or-fail wrapper, and the alias-resolution function so the three standards are implemented once, not re-derived per adapter. All adapters also call `assert_not_replay_mode()` (`_guards.py:11`) before any network fetch.

---

## Part 6 — Ordered implementation milestones

Each milestone is a self-contained, mergeable slice. No milestone leaves the engine in a broken state for other sports.

### Milestone 0 — Registry refactor (precondition)

- Implement `GAME_BACKENDS` and `PROP_BACKENDS` registries in `omega/core/simulation/backends.py`.
- Register `FastScoreSimulationBackend` and `MarkovGameSimulationBackend` under their existing names at engine-module import time.
- Replace `service.py:819` dispatch with a registry lookup.
- Add `PropSimulationBackend` Protocol, `PropSimulationInput` dataclass, and `PropDistributionRouterBackend` wrapping the existing `run_player_simulation()` logic.
- Build the shared ETL harness `omega/integrations/_etl.py` (Parquet caching decorator, Pydantic validate-or-fail wrapper, alias resolver). Every Phase 7 adapter depends on it, so it lands in the precondition milestone alongside the registry. Add the three ETL-standard tests.
- **Gate**: all Phase 6 replay-determinism tests pass bit-identically (NBA + MLB); ETL harness tests green.
- **Hardening (post-merge, 2026-06-01).** Closed four residual seams left by the scaffolding: (1) removed the residual `fast_score` dispatch branch — `run_fast_game_simulation` takes a per-call `backend=`; (2) added `evidence_mode` to the `GameSimulationBackend` Protocol so evidence routing reads a capability instead of sniffing the name (`_is_markov_family` → `_uses_transition_modifiers`); (3) wired `analyze_player_prop` through the prop registry with a `prop_distribution_router` fallback for unregistered routes; (4) the router now forwards `distribution`/`dud_prob` to keep that routing bit-identical. A follow-up pass then (5) **resolved the §5.5 game-level prior seam (option a)** — added `GameSimulationInput.prior_payload` + `GameAnalysisRequest.prior_payload` and the request→engine→backend plumbing (see `PHASE7_HANDOFF.md` §5.5); and (6) added **fail-loud registration validation** (`register_*_backend` rejects an incomplete backend surface at import time). Full suite green at **993 passed** (plus the new seam/validation cases); coverage in `tests/core/test_backend_registry.py`. Current state: `PHASE7_HANDOFF.md` §2.2–§2.3, §5.5.

### Milestone 1 — WNBA

- Implement and register `MarkovWNBAGameSimulationBackend` with WNBA-tuned pace/efficiency constants in `omega/core/sport_baselines.py`.
- Wire `leagues.py:28` to point to the new backend; add `liquidity_profile: "low"` flag.
- Build `omega/integrations/espn_wnba.py` (live); add `WNBA_STAT_KEYS` to the boxscore parser. Add `omega-refresh-wehoop` to load `wehoop` historical PBP/box/shot data into backtest artifact storage for WNBA replay.
- Create `early_market_snapshots` table (separate from `closing_lines`) with `(trace_id, league, captured_at, liquidity_profile, market, price)` columns.
- Extend `omega-fetch-closing-lines` and add `omega-capture-early-lines` so early-morning low-liquidity captures land in `early_market_snapshots`, never `closing_lines`.
- Update the calibration fitter to exclude `early_market_snapshots`-derived traces by default; add `include_early_snapshots=True` flag that forces the `context_slice="early_market_low_liq"` partition so promoted profiles never inherit phantom edges from pre-move pricing.
- **Acceptance**: 5 historical WNBA games replay deterministically; closing-line capture cron tested end-to-end against the test Odds API key; CLV computation reads only `closing_lines` and ignores `early_market_snapshots`; calibration fit on a synthetic dataset with intentionally inflated early-line EV does **not** drift the production calibration profile.

### Milestone 2 — Soccer (deadline: 2026-06-11)

- Implement and register `SoccerPoissonBackend`. Backend **requires** `rho` in `request.prior_payload`; no static fallback. Raises `MissingDixonColesPriorError` if absent.
- Build `omega/integrations/statsbomb.py` (historical xG, primary fit source), `omega/integrations/understat.py` (current season), and `omega/integrations/fbref.py` (redundancy); create `priors_xg` SQLite table.
- Build `omega-fit-dixon-coles` + `priors_dixon_coles` table; fit and promote a `fifa_intl_v1` profile before the World Cup deadline using 5+ seasons of FIFA-tier international matches from StatsBomb Open Data. Freeze `as_of_date` for the tournament duration.
- Wire the gatherer to read the active `rho_fit_profile` from `leagues.py`, look up the production profile in `priors_dixon_coles`, and inject `rho` + provenance into `request.prior_payload`.
- Build `omega/core/edge/soccer_derivatives.py` for Asian Handicap, BTTS, correct-score, first-half-total markets.
- Add `FIFA_WORLD_CUP_2026` league config; add the soccer sport key to Odds API map.
- Bootstrap calibration profile for soccer with identity profile; promote after N=100 graded matches per the existing Phase 6 policy.
- **Acceptance**: 10 historical EPL or UCL matches replay deterministically with non-zero `draw_prob`; bivariate-Poisson + Dixon-Coles unit tests for low-score cells; assert engine returns `status="skipped"` with `missing_requirements=["rho_prior"]` when the Dixon-Coles profile is missing; assert `rho` provenance fields land in the session sidecar `data_provenance` event.

### Milestone 3 — Tennis (Wimbledon target: 2026-06-29)

- Implement and register `TennisMarkovBackend` with closed-form game/set/match probabilities and per-pressure-state Markov nodes.
- Build `omega/integrations/tennis_sackmann.py`; create `priors_tennis` table with surface-segmented rolling SPW%/RPW%.
- Build `omega-fit-tennis-pressure-coefficients` + `priors_tennis_pressure` table; fit additive SPW% deltas for the six pressure states from Match Charting Project data. Players below the N=500-point threshold fall back to a tour+surface group mean; no silent 0.0 defaults.
- Wire the gatherer to join `priors_tennis_pressure` into `request.prior_payload["pressure_coefficients"]` with `pressure_coefficient_source` (`player` | `group_fallback`) for the audit event.
- Add `ATP`/`WTA` league configs; add tennis sport keys to Odds API map.
- Calibration: tennis matches show higher player-level variance than basketball; expect the system to ride the identity profile longer before promotion.
- **Acceptance**: 20 historical ATP/WTA matches (mixed surfaces) replay deterministically; closed-form match-win probability matches N=100k Monte Carlo within 0.5%; ablation test asserts derivative-market probabilities (first-set winner, set-handicap) shift in the expected direction when pressure coefficients are toggled off vs. on; assert `pressure_coefficient_source` reaches the session sidecar `data_provenance` event.

### Milestone 4 — NFL (kickoff target: 2026-09-10)

- Implement and register `NflSimulationBackend` (Gamma-Poisson team scores) and `NegBinomPropBackend`.
- Build `omega/integrations/espn_nfl.py`; add NFL stat keys to boxscore parser.
- Build `omega/integrations/nflverse.py`; create `priors_nfl_dispersion` table with `nb_dispersion_k`, `nb_k_shrinkage_weight`, `nb_k_source` columns.
- Build `omega-fit-nfl-dispersion` with mandatory hierarchical Bayesian shrinkage: per-player `k` is shrunk toward `(position_group, stat_type)` posteriors via a conjugate Gamma prior with shrinkage weight `w(n) = n / (n + n0)`, `n0=8` initial. Cold-start players inherit the league mean.
- Build `omega/core/edge/nfl_teasers.py` evaluating Wong teaser legs from the discrete margin distribution.
- **Acceptance**: 20 historical NFL games replay deterministically; teaser-leg EV computation tested on the classic Wong leg ranges (`-1.5 → +4.5`, `+1.5 → +7.5`, `-8.5 → -1.5`); shrinkage unit test asserts a rookie WR with 30 receptions gets `nb_k_source="position_group"` and `w < 0.6`; tail-probability test confirms `longest_reception over 40.5` for that rookie does not produce EV more than X bps above the position-group baseline.

---

## Part 7 — Verification plan

End-to-end checks for each sport, mirroring the `tests/core/test_replay_0526_mlb.py` pattern:

1. **Replay determinism per sport** — new files `tests/core/test_replay_<date>_<league>.py`, one per milestone. Snapshot 5–20 production traces, re-run `analyze_game()` with the same seed, assert simulation fields match within tolerance and edge rows are bit-identical.
2. **Backend contract validation** — every new backend must pass `enforce_game_backend_contract()` (`backends.py:94`) and emit all `REQUIRED_DISTRIBUTION_FIELDS`.
3. **Soccer-specific** — assert `supports_draw=True` propagates end-to-end; assert Dixon-Coles low-score correction shifts cells `(0,0)`, `(1,0)`, `(0,1)`, `(1,1)` in the expected direction relative to unadjusted bivariate Poisson.
4. **Tennis-specific** — closed-form match-win probability vs. N=100k Monte Carlo agreement within 0.5%; best-of-3 vs. best-of-5 transitions tested independently; surface coefficient applied correctly.
5. **NFL-specific** — NB sampler empirical-variance check vs. `mean + mean²/k`; teaser-leg EV unit tests on known historical Wong opportunities; assert margin distribution sums to 1.0 over `{-21..+21}`.
6. **Integration adapter tests** — each new integration in `tests/integrations/`, each calling `assert_not_replay_mode()` correctly. Mock `url_opener` for HTTP-bound tests; the guard is environment-driven so tests must not set `OMEGA_REPLAY_MODE=1`.
7. **Calibration profile bootstrap** — assert each new league has an identity profile registered at `status=PRODUCTION` until N graded outcomes are collected. Promotion follows the existing Phase 6 policy in `omega/core/calibration/registry.py`.
8. **Session sidecar parity** — every new backend emits an `engine_run` audit event to `var/inbox/sessions/<session_id>.json` via the unmodified `append_audit_events()` writer (`omega/trace/session_sidecar.py:147`). Every adapter emits a `data_provenance` event. No changes to the sidecar schema.

### Red-team-finding verification tests

9. **Soccer dynamic `rho`** — unit test that `SoccerPoissonBackend.run()` raises `MissingDixonColesPriorError` when `request.prior_payload` lacks `rho`. Integration test that the gatherer loads `rho` from the production `fifa_intl_v1` profile and the resulting session sidecar `data_provenance` event records `rho_profile_id` and `rho_as_of_date`. Differential test that swapping in an `epl_v3` profile vs `fifa_intl_v1` measurably changes `draw_prob` for the same xG inputs.

10. **NFL hierarchical shrinkage** — fitter unit test that a synthetic player with `n=30` carries gets `nb_k_source="position_group"` and `nb_k_shrinkage_weight < 0.6`, while a synthetic player with `n=200` carries gets `nb_k_source="player"` and `w >= 0.6`. Backend test that the `longest_rush over 40.5` tail probability for the small-sample synthetic player matches the position-group baseline within tolerance, not the player's noisy raw `k`.

11. **Tennis pressure coefficients** — closed-form-with-pressure vs. closed-form-flat ablation test: assert first-set-winner and set-handicap probabilities shift by at least X bps for a player with non-trivial pressure deltas. Fallback test that a player below the N=500 threshold receives `pressure_coefficient_source="group_fallback"` and the group means are used, not 0.0.

12. **WNBA early-line CLV isolation** — assert that early captures land only in `early_market_snapshots`, never `closing_lines`. CLV-computation test that the metric is bit-identical whether `early_market_snapshots` contains 0 rows or 1000 rows. Calibration-fitter test that running on a synthetic dataset with intentionally inflated early-line EV does not drift the production profile; the same dataset with `include_early_snapshots=True` produces a separate `context_slice="early_market_low_liq"` profile candidate.

### ETL-standard verification tests

13. **Caching layer** — adapter test that a second pull within TTL serves from `data/cache/<source>/` and makes zero network calls (assert the mocked `url_opener` is invoked exactly once across two `fetch()` calls).
14. **Schema-drift fail-loud** — feed each adapter a fixture with a renamed/missing column; assert it raises `SourceSchemaDriftError`, writes a `fail`-status `data_provenance` sidecar event, exits non-zero, and writes **nothing** to the priors table.
15. **Entity resolution** — assert "Patrick Mahomes II" and "Patrick Mahomes" resolve to the same key via `data/aliases/NFL.json`; assert an unknown entity is excluded from the priors write and emits a `data_provenance` warning.

---

## Part 8 — Red-team assessment of data sources and modeling assumptions

### Modeling-assumption failure modes (mitigated in this design)

These four findings were raised in red-team review and the mitigations are now part of Phase 7 scope rather than deferred.

- **Soccer — static Dixon-Coles `rho` is brittle.** International tournament soccer (World Cup) has materially different draw propensity and scoring variance than domestic club leagues. Hardcoding a single `rho` in `leagues.py` was the original sketch and is now removed. `rho` is treated as a dynamic prior, fit per *competition profile* (`fifa_intl_v1`, `epl_v3`, etc.) by `omega-fit-dixon-coles`, persisted in `priors_dixon_coles`, and injected into `request.prior_payload`. The backend fails closed (`status="skipped"`, `missing_requirements=["rho_prior"]`) if no production profile exists; no implicit default. Refits are frozen at tournament kickoff for the duration of the event so live decisions cannot drift.

- **NFL — per-player NB dispersion `k` over-fits on small samples.** Estimating `k` strictly from individual nflverse per-player data is unstable: rookie wide receivers, backup running backs, and snap-share-limited players generate noisy `k` values that produce false-positive tail edges (longest-reception, longest-rush). The mitigation is mandatory hierarchical Bayesian shrinkage in `omega-fit-nfl-dispersion`: per-player `k` is shrunk toward `(position_group, stat_type)` posteriors via a conjugate Gamma prior with shrinkage weight `w(n) = n / (n + n0)`. The `priors_nfl_dispersion` table carries `nb_k_source` (`player` | `position_group` | `league`) and `nb_k_shrinkage_weight` so audits can trace whether a high-EV tail call was driven by genuine player signal or group prior. The backend remains shrinkage-agnostic; all hierarchy lives offline.

- **Tennis — IID SPW% misprices pressure-state markets.** Closed-form Markov chains under an IID SPW% assumption are computationally cheap but ignore the measurable shift in serving distributions during break points, tiebreaks, set points, match points, and serving-for-set/match nodes. The flat-SPW% backend would systematically misprice set-winner, first-set games-handicap, and live derivative markets. The mitigation is per-pressure-state additive SPW% deltas fit by `omega-fit-tennis-pressure-coefficients` from Match Charting Project point-by-point data, persisted in `priors_tennis_pressure`, and injected as `request.prior_payload["pressure_coefficients"]`. The Markov chain replaces the closed-form game polynomial with a finite-state game-level chain only at games containing pressure points — result is exact, runtime cost is small. Players with fewer than N=500 charted points fall back to a tour+surface group mean; flat 0.0 is never silently applied.

- **WNBA — early-morning line capture distorts CLV and calibration.** Capturing low-liquidity early WNBA lines is necessary to lock in size before sharp action moves the market, but those early lines do not reflect closing probability — they move violently and the implied EV is phantom. Blending early captures into the canonical `closing_lines` table would destroy CLV as a metric and bias calibration toward unrealistic edges. Mitigation: early captures land in a dedicated `early_market_snapshots` table flagged with `liquidity_profile`. The CLV computation reads only `closing_lines` and ignores `early_market_snapshots` entirely. The calibration fitter excludes `early_market_snapshots`-derived traces by default; opting in (`include_early_snapshots=True`) forces a separate calibration profile slice (`context_slice="early_market_low_liq"`) so promoted profiles inherit only closing-line-grounded calibration.

### Data-source failure modes

### Odds API

- **Rate limits**: paid tier has per-second and per-month caps. Mitigation: existing `fetch_closing_lines.py` already batches per league; extend the same batching to the new sport keys.
- **Sport-key churn**: the-odds-api occasionally renames sports mid-season (e.g., conference reshuffles). Mitigation: cache last-good sport list on disk, fall back to cached on 404, emit a `data_provenance` warning.
- **Market-availability gaps**: low-liquidity WNBA games and niche tennis tournaments may not have spreads or props at all books. Mitigation: degrade to qualitative-only output (no BetCard) rather than emitting empty edge rows. The bounded-autonomy invariant from `CLAUDE.md` already requires this.
- **3-way moneyline format**: soccer uses 3-way (`home`, `draw`, `away`) where the rest of the system uses 2-way. Mitigation: explicit `h2h_3way` market handling in the soccer edge consumer; existing 2-way handlers untouched.

### ESPN public endpoints

- **No SLA**: ESPN can change schema without warning. Box-score shape differs across sports more than expected — NFL has nested team/player tables; tennis is not available at all.
- Mitigation: per-sport shape validators with explicit version pinning, fail loud on shape drift (raise a typed `EspnSchemaDriftError`), never silently default a missing field. Tests pin known-good payloads as fixtures.

### Understat

- **HTML-scraped, fragile**: page structure can change with a single deployment.
- Mitigation: FBref redundancy check (independent source). Cross-source xG agreement check emits an audit event when disagreement exceeds 15%. Priors freeze at tournament kickoff so mid-event breakage cannot poison live decisions.

### FBref

- **Cloudflare interventions, ToS changes**: scraping risk.
- Mitigation: aggressive caching (24-hour TTL), weekly refresh cadence, fall back to last-known-good with a stale-prior audit event tagged `freshness=stale`.

### Jeff Sackmann CSVs

- **Stable but lagged**: typically updated weekly with a several-day lag for current matches.
- Mitigation: explicit `as_of_date` field in `priors_tennis`; treat priors older than 14 days as stale and emit a freshness audit event. Recent matches within the lag window blend prior-season data weighted by surface.

### nflverse

- **R-native source**: Python ports drift. Direct R-bridge calls are fragile.
- Mitigation: snapshot the relevant nflverse Parquet/CSV exports; load via a CSV-based adapter rather than an R bridge. Refresh script (`omega-refresh-nflverse`) runs weekly.

### Open-repository ETL — rate limits & IP bans

- Repositories that scrape live sites (`pybaseball` → Baseball Savant especially, also Understat/FBref HTML) enforce strict rate limits and will IP-ban aggressive pulls. StatsBomb, nflverse, Sackmann, and Synergy are git/file-hosted and lower-risk, but bulk clones still warrant courtesy throttling.
- Mitigation (mandatory, see Part 5B standard 1): a local caching layer persists the **raw** upstream response as Parquet (tabular) or raw JSON/HTML *before* any transform, under `data/cache/<source>/`. Transforms read the cache; retries never re-fetch within TTL. The frozen cache doubles as the knowable-at-the-time snapshot that makes backtests reproducible. Implemented once in `omega/integrations/_etl.py`.

### Open-repository ETL — schema drift

- Public datasets rename columns without warning (nflverse renames between seasons; Sackmann adds/reorders columns; FanGraphs metric keys change). A silent `None` coerced from a renamed column poisons the calibration pipeline far downstream where it is hard to trace.
- Mitigation (mandatory, see Part 5B standard 2): each adapter validates every ingested record against a Pydantic model at the boundary. On failure it raises a typed `SourceSchemaDriftError`, writes a `fail`-status `data_provenance` event to the session sidecar, and exits non-zero. The ETL job fails loud rather than passing `None` into priors. Tests pin known-good upstream payloads as fixtures.

### Cross-source player-identity drift

- The `normalize_player_name()` function (`espn_boxscore.py:122`) handles accents, suffixes, and punctuation, but is not infallible across leagues — "Patrick Mahomes II" vs "Patrick Mahomes", and tennis players especially (accented names, hyphens, doubles-pair compound names).
- Mitigation (mandatory, see Part 5B standard 3): a centralized per-league alias table at `data/aliases/<league>.json` resolves every entity name *before* it is written to a priors table. Resolution order: exact match → `normalize_player_name()` → alias table → unresolved. Unresolved entities emit a `data_provenance` warning and are excluded from the priors write rather than written under an ambiguous key. Resolution is implemented once in `omega/integrations/_etl.py`; alias tables are versioned in git.

### Replay-mode discipline

- Easy to forget `assert_not_replay_mode()` in a new module — the failure is silent until a backtest accidentally hits the network.
- Mitigation: extend `tests/integrations/test_replay_mode_guard.py` to import every module under `omega/integrations/` and assert each one references the guard symbol. Static check, runs in CI on every change.

---

## Part 9 — Rollback plan

Each layer of Phase 7 is reversible without rolling back the others.

- **Backend registry refactor** reverts by replacing the registry lookup in `service.py` with the inline switch and unregistering the new backends. The new backend modules can remain on disk dormant.
- **Per-sport gating** is via `default_game_backend` in `leagues.py`. Setting the value to `"fast_score"` falls back to the generic backend — the sport remains usable in a degraded form rather than broken.
- **New external-priors tables** are append-only. Flushing a table causes the relevant backend to fail closed (request hits the engine, response is `status="skipped"` with `missing_requirements`) instead of producing bad numbers. This is the same failure mode as a missing context dict today.
- **Calibration profiles** for the new sports start as identity profiles. Rolling back is `UPDATE profiles SET status='ARCHIVED' WHERE league=...`; the selection policy (`omega/core/calibration/registry.py:88`) falls back to identity automatically.
- **Dixon-Coles `rho` profiles** roll back via `UPDATE priors_dixon_coles SET status='ARCHIVED' WHERE profile_id=...`. The gatherer fails closed (`MissingDixonColesPriorError`), the engine returns `status="skipped"`, and the sport is dormant rather than producing bad numbers. Promoting an older `as_of_date` is the same UPDATE in reverse.
- **NFL dispersion** rollback to a previous fit is a row-level UPDATE on `priors_nfl_dispersion`. Position-group-only fallback is achievable by setting `nb_k_source='position_group'` for all rows in the table — the runtime is unchanged because the backend reads only `nb_dispersion_k`.
- **Tennis pressure coefficients** rollback to flat IID is achieved by truncating `priors_tennis_pressure`. The gatherer then injects an empty dict for `pressure_coefficients`, and the backend defaults all deltas to 0.0 — i.e. exactly the flat closed-form Markov chain. No code change needed.
- **WNBA early-line isolation** rollback (if a future analysis legitimately wants early lines in CLV) is a one-line change to the CLV query plus an `ALTER TABLE` to add a `source` column. The dedicated `early_market_snapshots` table remains valid; the question is whether the consumer reads it.

---

## Files this plan will create or modify

### New files

Engine and edge:
- `omega/core/simulation/markov_wnba.py`
- `omega/core/simulation/soccer_bivariate_poisson.py`
- `omega/core/simulation/tennis_markov.py`
- `omega/core/simulation/nfl_neg_binom.py`
- `omega/core/simulation/prop_neg_binom.py`
- `omega/core/simulation/prop_distribution_router.py`
- `omega/core/edge/soccer_derivatives.py`
- `omega/core/edge/nfl_teasers.py`
- `omega/core/sport_baselines.py` (per-league Markov tuning constants)

Integrations — live fetch:
- `omega/integrations/espn_wnba.py`
- `omega/integrations/espn_nfl.py`
- `omega/integrations/understat.py` (current-season soccer xG)
- `omega/integrations/fbref.py` (soccer xG redundancy)

Integrations — shared ETL + backtestable open-repository adapters:
- `omega/integrations/_etl.py` — shared caching decorator (raw → Parquet), Pydantic validate-or-fail wrapper, alias resolver. Implements the three Part 5B standards once.
- `omega/integrations/tennis_sackmann.py` — JeffSackmann `tennis_atp`/`tennis_wta` + Match Charting Project.
- `omega/integrations/nflverse.py` — nflverse / `nflreadpy` PBP, EPA, WPA, roster.
- `omega/integrations/statsbomb.py` — StatsBomb Open Data (historical xG, freeze-frame reserved).
- `omega/integrations/nba_play_types.py` — DomSamangy `NBA_Play_Types_16_23` Synergy play-type priors.
- `omega/integrations/pybaseball_adapter.py` — `pybaseball` Statcast/FanGraphs/Retrosheet for MLB backtests.

Offline fit / refresh scripts:
- `omega-fit-dixon-coles` — per-competition Dixon-Coles `rho` fits (soccer).
- `omega-fit-tennis-pressure-coefficients` — per-player pressure-state SPW% deltas with group-fallback (tennis).
- `omega-fit-nfl-dispersion` — hierarchical Bayesian NB `k` with position-group shrinkage (NFL).
- `omega-capture-early-lines` — low-liquidity early-line cron writing to `early_market_snapshots` (WNBA, future low-liq sports).
- `omega-refresh-sackmann`, `omega-refresh-nflverse`, `omega-refresh-statsbomb`, `omega-refresh-wehoop` — weekly priors/backtest-data refresh.

Data / config (versioned in git):
- `data/aliases/<league>.json` — per-league entity alias tables (`WNBA`, `ATP`, `WTA`, `FIFA_WORLD_CUP_2026`, `NFL`, plus existing `NBA`/`MLB`).
- `data/cache/<source>/` — raw upstream response cache (Parquet/JSON/HTML); git-ignored.
- `data/external/sackmann/` — cloned Sackmann CSVs.

New SQLite tables:
- `priors_xg` — StatsBomb/Understat/FBref soccer xG.
- `priors_dixon_coles` — fitted `rho` per competition profile.
- `priors_tennis` — surface-segmented rolling SPW%/RPW%.
- `priors_tennis_pressure` — per-state SPW% deltas with group fallback.
- `priors_nfl_dispersion` — NB `k` with `nb_k_source` and `nb_k_shrinkage_weight`.
- `priors_nba_play_types` — Synergy play-type frequencies / PPP percentiles (NBA prop context).
- `early_market_snapshots` — segregated from `closing_lines`, excluded from CLV.

WNBA historical (`wehoop`) data is loaded into existing backtest artifact storage via `omega-refresh-wehoop`; it does not need a new priors table.

### Modified files

- `omega/core/simulation/backends.py` — add `GAME_BACKENDS`, `PROP_BACKENDS`, `PropSimulationBackend` Protocol, `PropSimulationInput`, `DEFAULT_PROP_BACKEND_BY_LEAGUE_STAT`.
- `omega/core/simulation/engine.py` — register existing backends; wrap prop functions as `PropDistributionRouterBackend`.
- `omega/core/contracts/service.py` — registry-based dispatch at `:819`.
- `omega/core/contracts/schemas.py` — `prior_payload` on `PlayerPropRequest`; `SoccerDerivativeMarket` enum.
- `omega/core/config/leagues.py` — tune WNBA, add `ATP`/`WTA`/`FIFA_WORLD_CUP_2026`, flag NFL teaser-eval.
- `omega/integrations/odds_api.py` — add ATP / WTA / FIFA World Cup sport keys.
- `omega-fetch-closing-lines` — early-line capture for low-liquidity leagues.
- `tests/integrations/test_replay_mode_guard.py` — assert every integration module references the guard.

### New tests

One replay-determinism test per milestone:

- `tests/core/test_replay_wnba.py`
- `tests/core/test_replay_soccer_world_cup.py`
- `tests/core/test_replay_tennis.py`
- `tests/core/test_replay_nfl.py`

ETL-standard tests (shared harness in `tests/integrations/`):

- `tests/integrations/test_etl_cache.py` — caching/TTL, zero-refetch.
- `tests/integrations/test_etl_schema_drift.py` — Pydantic fail-loud per adapter.
- `tests/integrations/test_etl_aliases.py` — entity-resolution and exclusion-on-unresolved.

Plus contract / math unit tests under `tests/core/simulation/` and adapter tests under `tests/integrations/` per sport.

