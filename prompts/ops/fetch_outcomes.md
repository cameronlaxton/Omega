# Fetch Outcomes — Daily Outcome Loop

Run this after all games for the day are final. It attaches ESPN final scores and box-score stats to all ungraded traces in the DB, then scores evidence signals and refreshes the calibration health report.

**When to run:**
- NBA: after ~midnight Eastern (games typically finish 11:30pm ET latest)
- MLB: after ~2am Eastern (late West Coast games)
- Safe to run early — idempotent; will only attach confirmed final scores

---

## Run the outcome loop

```bash
omega-run-action-plan fixtures/action_plans/daily_outcome_evidence_loop.json
```

This dispatches the tracked daily outcome/evidence loop in sequence:
1. `omega-ingest-traces` — drains `var/inbox/traces/*.json` into the DB so outcomes see the latest exports
2. `omega-fetch-outcomes` (NBA + WNBA + MLB + soccer + props) — attaches final scores and box-score stats
3. `omega-settle` - settles pending user-confirmed ledger rows with attached outcomes
4. `omega-score-evidence-signals` - retrospectively scores signal predictiveness
5. `omega-report-calibration --league NBA --window-days 30` - writes `var/reports/latest.md`

---

## Expected output

```
[ingest_traces] Processed N traces, 0 failed
[fetch_outcomes] NBA: M outcomes attached, K unmatched
[fetch_outcomes] WNBA: M outcomes attached, K unmatched
[fetch_outcomes] MLB: M outcomes attached, K unmatched
[fetch_outcomes] soccer: M outcomes attached, K unmatched
[fetch_outcomes] props: M outcomes attached, K unmatched
[omega-settle] settled S pending ledger rows
[omega-score-evidence-signals] scored N signal instances
[omega-report-calibration] wrote var/reports/latest.md
```

**If unmatched > 20% of traces:** check alias table or date range. ESPN aliases change with trades/team naming. Add new mappings to `omega/integrations/espn_nba.py::TEAM_ALIASES`, `espn_wnba.py::WNBA_TEAMS`, `espn_mlb.py::MLB_TEAMS`, or `espn_soccer.py::SOCCER_TEAM_ALIASES`.

**If a sub-step fails:** check `var/inbox/traces/failed/` for `.error.txt` sidecars. Common causes: missing `home_team`/`away_team`/`game_date` on prop traces, or `omega-ingest-traces` validation errors.

---

## Dry run first (optional)

```bash
omega-run-action-plan fixtures/action_plans/daily_outcome_evidence_loop.json --dry-run
```

Shows what would be dispatched without writing to the DB.

---

## Per-league options

If you only need outcomes for one league:

```bash
omega-fetch-outcomes-all --leagues nba
omega-fetch-outcomes-all --leagues mlb
omega-fetch-outcomes-all --leagues soccer
omega-fetch-outcomes-all --leagues props
omega-fetch-outcomes-all --since 2026-05-25 --until 2026-05-27  # backfill range
omega-fetch-outcomes-all --dry-run
```

### WNBA coverage

WNBA is fully wired for both planes and is included in the default
`fetch_outcomes_all.py` league set and the daily outcome loop.

- **WNBA games** (moneyline/spread): `omega-fetch-outcomes-wnba`
  (ESPN WNBA scoreboard). Run alone with:

  ```bash
  omega-fetch-outcomes-all --leagues wnba
  # or directly:
  omega-fetch-outcomes-wnba --since yesterday
  ```

- **WNBA player props**: graded by the `props` step — `omega-fetch-outcomes-props`
  covers NBA/WNBA/MLB/supported soccer. Run WNBA props only with:

  ```bash
  omega-fetch-outcomes-props --league WNBA
  ```

Unmapped-team warnings mean a WNBA team alias is missing from
`omega/integrations/espn_wnba.py::WNBA_TEAMS` (kept in sync with ESPN's 15-team
list, including the 2026 expansion Portland Fire and Toronto Tempo).

---

### Soccer coverage

Soccer game outcomes are included in the daily loop through
`omega-fetch-outcomes-soccer`. Soccer player props are graded by the shared
`props` step only for ESPN-summary-backed leagues and stat keys documented in
`prompts/reference/prop_stat_keys.md`.

```bash
omega-fetch-outcomes-all --leagues soccer props
omega-fetch-outcomes-soccer --since yesterday
omega-fetch-outcomes-props --league EPL
```

Unsupported soccer competitions or unsupported stat keys remain pending for
manual triage; do not grade absent/DNP player props as zero-stat losses. Use
`omega_trace_void_prop` for DNP/no-action voids.

---

## ⚠ Replay mode guard

If you are evaluating **historical** traces (not live today's games), set `OMEGA_REPLAY_MODE=1` first:

```bash
# WRONG — live fetch during historical evaluation:
omega-fetch-outcomes-all --since 2026-05-01 --until 2026-05-10

# CORRECT — for historical evaluation, use frozen fixtures:
export OMEGA_REPLAY_MODE=1   # blocks all live ESPN/Odds API fetches
# ... use frozen fixtures path instead
unset OMEGA_REPLAY_MODE       # restore for live operations
```

`OMEGA_REPLAY_MODE=1` blocks ESPN and Odds API fetches at the integration layer and raises `OmegaReplayModeError`. Never set it for live daily outcome runs.

---

## After the loop completes

Read the refreshed `var/reports/latest.md`:
- §3 game Brier/ECE — flag if ECE > 0.05
- §3B prop pairs — watch for progress toward 100+ (required for first calibration fit)
- §6B evidence signals — update your confidence weights for the next session
