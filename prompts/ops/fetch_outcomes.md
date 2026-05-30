# Fetch Outcomes — Daily Outcome Loop

Run this after all games for the day are final. It attaches ESPN final scores and box-score stats to all ungraded traces in the DB, then scores evidence signals and refreshes the calibration health report.

**When to run:**
- NBA: after ~midnight Eastern (games typically finish 11:30pm ET latest)
- MLB: after ~2am Eastern (late West Coast games)
- Safe to run early — idempotent; will only attach confirmed final scores

---

## Run the outcome loop

```bash
python scripts/run_action_plan.py inbox/action_plans/templates/daily_outcome_evidence_loop.json
```

This dispatches four steps in sequence:
1. `ingest_traces` — drains `inbox/traces/*.json` into the DB so outcomes see the latest exports
2. `fetch_outcomes` (NBA + MLB + props) — attaches final scores and box-score stats
3. `score_evidence_signals` — retrospectively scores signal predictiveness
4. `report_calibration --league NBA --window-days 30` — writes `reports/latest.md`

---

## Expected output

```
[ingest_traces] Processed N traces, 0 failed
[fetch_outcomes] NBA: M outcomes attached, K unmatched
[fetch_outcomes] MLB: M outcomes attached, K unmatched
[fetch_outcomes] props: M outcomes attached, K unmatched
[score_evidence_signals] scored N signal instances
[report_calibration] wrote reports/latest.md
```

**If unmatched > 20% of traces:** check alias table or date range. ESPN aliases change with trades — add new mappings to `omega/integrations/espn_nba.py::TEAM_ALIASES` or `espn_mlb.py::MLB_TEAMS`.

**If a sub-step fails:** check `inbox/traces/failed/` for `.error.txt` sidecars. Common causes: missing `home_team`/`away_team`/`game_date` on prop traces, or `ingest_traces` validation errors.

---

## Dry run first (optional)

```bash
python scripts/run_action_plan.py inbox/action_plans/templates/daily_outcome_evidence_loop.json --dry-run
```

Shows what would be dispatched without writing to the DB.

---

## Per-league options

If you only need outcomes for one league:

```bash
python scripts/fetch_outcomes_all.py --leagues nba
python scripts/fetch_outcomes_all.py --leagues mlb
python scripts/fetch_outcomes_all.py --leagues props
python scripts/fetch_outcomes_all.py --since 2026-05-25 --until 2026-05-27  # backfill range
python scripts/fetch_outcomes_all.py --dry-run
```

### WNBA coverage

WNBA is fully wired for both planes and is included in the default
`fetch_outcomes_all.py` league set and the daily outcome loop.

- **WNBA games** (moneyline/spread): `scripts/fetch_outcomes_wnba.py`
  (ESPN WNBA scoreboard). Run alone with:

  ```bash
  python scripts/fetch_outcomes_all.py --leagues wnba
  # or directly:
  python scripts/fetch_outcomes_wnba.py --since yesterday
  ```

- **WNBA player props**: graded by the `props` step — `fetch_outcomes_props.py`
  covers NBA/WNBA/MLB. Run WNBA props only with:

  ```bash
  python scripts/fetch_outcomes_props.py --league WNBA
  ```

Unmapped-team warnings mean a WNBA team alias is missing from
`omega/integrations/espn_wnba.py::WNBA_TEAMS` (kept in sync with ESPN's 15-team
list, including the 2026 expansion Portland Fire and Toronto Tempo).

---

## ⚠ Replay mode guard

If you are evaluating **historical** traces (not live today's games), set `OMEGA_REPLAY_MODE=1` first:

```bash
# WRONG — live fetch during historical evaluation:
python scripts/fetch_outcomes_all.py --since 2026-05-01 --until 2026-05-10

# CORRECT — for historical evaluation, use frozen fixtures:
export OMEGA_REPLAY_MODE=1   # blocks all live ESPN/Odds API fetches
# ... use frozen fixtures path instead
unset OMEGA_REPLAY_MODE       # restore for live operations
```

`OMEGA_REPLAY_MODE=1` blocks ESPN and Odds API fetches at the integration layer and raises `OmegaReplayModeError`. Never set it for live daily outcome runs.

---

## After the loop completes

Read the refreshed `reports/latest.md`:
- §3 game Brier/ECE — flag if ECE > 0.05
- §3B prop pairs — watch for progress toward 100+ (required for first calibration fit)
- §6B evidence signals — update your confidence weights for the next session
