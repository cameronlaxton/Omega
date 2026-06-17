# Historical Replay → Calibration Backfill Runbook

Grow calibration profile sample sizes by replaying historical fixtures through the
real Omega engine **outcome-blind**, attaching the actual result, and fitting
**candidate** profiles from the enlarged pool — without contaminating the
production trace DB or leaking post-game data into pre-game context.

```
source CSV → ingest (manifest+normalize+quarantine) → validate → replay (dedicated DB)
          → fit candidate (historical) → parity gates → promote (fail-closed)
```

## Non-negotiable safety rules (enforced in code)

- **DB isolation.** Replay writes to a dedicated `var/historical/replay_<league>.db`.
  `omega-replay-history` and `ReplayConfig` refuse the production DB
  (`omega.paths.is_production_trace_db`). Production `var/omega_traces.db` stays live-only.
- **Raw probabilities.** Persisted `predictions` are the raw simulation output; calibration is
  applied only at fit/eval time, never written back.
- **Provenance, not context.** Replay traces are tagged `execution_mode=historical_replay`;
  `context_source` stays `"provided"` (eligibility depends on it — do **not** set it to
  `historical_replay`).
- **Odds timing.** Per-source `odds_timing_class` gates **betting only** (selection/staking/ROI/CLV),
  never probability calibration. `decision_time_safe` may stake; `closing_only` is CLV-only;
  `timing_unknown` is excluded from ROI/CLV. Default for unrecognized sources is the conservative
  `timing_unknown`.
- **Promotion is fail-closed.** Candidates register as CANDIDATE only; `omega-promote-profile`
  always evaluates gates (no `--force`). Historical-only candidates additionally require a
  **distribution-parity PASS** (below).

## Worked example (EPL game + draw + CLV)

```bash
# 0. drop a football-data.co.uk season CSV at data/historical/raw/EPL/football_data/2324_E0.csv
# 1. ingest -> manifest_id (football_data is decision_time_safe + has closing odds)
python -m omega.ops.ingest_historical_dataset --source football_data --league EPL \
    --games data/historical/raw/EPL/football_data/2324_E0.csv
# 2. validate (fail-closed on hash drift)
python -m omega.ops.validate_historical_dataset --manifest-id <manifest_id> --strict
# 3. replay -> dedicated DB + RUN_AUDIT.md + replay_summary.json
python -m omega.ops.replay_history --league EPL --manifest-id <manifest_id> \
    --db var/historical/replay_epl.db --mode calibration
# 4. fit candidate(s) from historical only (dry-run first)
python -m omega.ops.fit_calibration --league EPL --plane game \
    --historical-only --historical-db var/historical/replay_epl.db --dry-run
python -m omega.ops.fit_calibration --league EPL --plane game \
    --historical-only --historical-db var/historical/replay_epl.db
python -m omega.ops.fit_calibration --league EPL --plane draw \
    --historical-only --historical-db var/historical/replay_epl.db
```

### Leakage-safe date windows

```bash
python -m omega.ops.fit_calibration --league EPL --plane game \
    --historical-only --historical-db var/historical/replay_epl.db \
    --train-start 2022-08-01 --train-end 2023-05-31 \
    --holdout-start 2023-08-01 --holdout-end 2024-05-31
```
Windows split on the event `decision_time` (not the replay run-time). Overlapping windows are
refused unless `--allow-same-season-shadow` (shadow diagnostics only; never promotable).

## Player props (league-scoped player-stat markets)

Props remain league-scoped (no standalone "props league"). Decision-time lines/prices
(`HistoricalPropMarket`) drive the prediction; the realized `stat_value`
(`HistoricalPropOutcome`) is attached only as the outcome; void/DNP props are excluded from
calibration.

```bash
python -m omega.ops.ingest_historical_dataset --source csv_games --league NBA \
    --games games.csv --player-stats player_stats.csv --prop-markets prop_lines.csv \
    --prop-context prop_context.json
python -m omega.ops.replay_history --league NBA --manifest-id <id> \
    --db var/historical/replay_nba.db
python -m omega.ops.fit_calibration --league NBA --plane prop \
    --historical-only --historical-db var/historical/replay_nba.db
```

`prop_context.json` maps `"<event_key>|<player>|<stat_type>"` → `{"<stat>_mean": .., "<stat>_std": ..}`
(decision-time, as-of). Producing this from raw box scores (as-of player context backfill) is the
one remaining prop enhancement; for now supply it explicitly.

## Promotion gates

```bash
# 1. covariate-shift gate: historical inputs must resemble live inputs
python -m omega.ops.historical_live_parity --league EPL --market game \
    --historical-db var/historical/replay_epl.db --live-db var/omega_traces.db --min-live-n 200
#    PASS -> promotable; INCONCLUSIVE (live_n < min) / FAIL -> shadow-only
# 2. candidate-vs-incumbent calibration quality (refuses ECE-only wins)
python -m omega.ops.backtest_parity --candidate-id <cand> --league EPL \
    --plane game --historical-db var/historical/replay_epl.db
# 3. ROI/CLV non-regression: use the existing walk-forward backtest report
python -m omega.ops.run_walk_forward_backtest --manifest-id <id> \
    --backtest-db var/historical/replay_epl.db
# 4. promote only when (1) PASS and (2)/(3) support it
python -m omega.ops.promote_profile --candidate-id <cand> \
    --confirm-backtest-parity --confirm-clv-non-regression
```

**Rule:** confirm `--confirm-backtest-parity` / `--confirm-clv-non-regression` only when the
parity reports above support it. A historical-only candidate must have a PASS from
`historical-live-parity`; INCONCLUSIVE or FAIL keeps it shadow-only (or requires a live-weighted
refit — a documented, deferred enhancement).

## Deferred enhancements

- As-of player-context backfill from raw box scores (props currently need supplied `prop_context`).
- `--exact-eval` on the replay path (the `analyze()` path is MC; exact-eval is wired in
  `BacktestEngine` and used by the walk-forward / parity ROI path).
- Live-weighted refit for INCONCLUSIVE/FAIL parity (v1 keeps such candidates shadow-only).
