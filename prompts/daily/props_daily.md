# Player Props Daily Session (NBA + WNBA + MLB)

Props analysis is safe regardless of game analysis bug status. Follow [OMEGA_COWORK.md](../../OMEGA_COWORK.md) and [prompts/system_prompt.txt](../system_prompt.txt) throughout.

For prop stat key → resolve_odds mapping, see [reference/prop_stat_keys.md](../reference/prop_stat_keys.md).

---

## Step 0 — Regenerate calibration report

```bash
python scripts/report_calibration.py --league NBA --window-days 30
```

Read `reports/latest.md` §3B (prop plane pairs) and §6B (evidence signal performance). Metrics in §3B are suppressed if fewer than 10 usable pairs — that's normal while accumulating data.

---

## Step 1 — Preflight + session ID

```bash
python scripts/cowork_preflight.py --formal-output-gate
```

Session ID format: `sess-YYYYMMDD-prp1` (e.g. `sess-20260528-prp1`) or append to an existing game session if props are part of the same analysis run.

---

## Step 2 — Resolve prop odds

```bash
python scripts/resolve_odds.py --kind prop --league NBA --player "Player Name" --prop-type pts --line 22.5
```

Supported prop_type values by league — see [reference/prop_stat_keys.md](../reference/prop_stat_keys.md) for the full mapping.

WNBA prop rule: query direct sportsbook/Odds API prop boards first through the
typed resolver (for example `--league WNBA --prop-type pts`). Standard O/U
player props are valid when returned by the typed odds path. Milestone props
and manually estimated lines are research-only and must not be passed to
`analyze()` as formal odds.

---

## Step 3 — Gather player context

Minimum required inputs for a calibration-eligible prop trace:

| Field | Source | Notes |
|---|---|---|
| `pts_mean` (or `reb_mean`, etc.) | Rolling 10-game average from ESPN/BR | Use `window="last_10"` |
| `pts_std` | Standard deviation from same sample | If unavailable, impute 0.3×mean and note downgrade |
| `sample_size` | Number of games in the average | < 8 → quality degraded |
| `is_playoff` | Matchup context | Always required |
| `rest_days` | Days since last game | 0 = back-to-back |

Injury/news protocol: translate injury status, minutes limits, and role changes
into the player context (`pts_mean`, usage/minutes-derived means, observed
standard deviations), `game_context`, and typed evidence before calling
`analyze()`. If you cannot quantify the impact, append a
`quality_gate/null_data_audit` event and downgrade to research-only.

**Minimum stat fields by prop_type:**

```
pts → pts_mean, pts_std
reb → reb_mean, reb_std
ast → ast_mean, ast_std
blk → blk_mean, blk_std
stl → stl_mean, stl_std
pra → pts_mean, pts_std, reb_mean, reb_std, ast_mean, ast_std
3pm → threes_mean, threes_std
hits (MLB) → hits_mean, hits_std
strikeouts_pitched (MLB) → strikeouts_pitched_mean, strikeouts_pitched_std
```

---

## Step 4 — Run engine

```python
analyze({
    "player_name": "Player Name",
    "league": "NBA",
    "prop_type": "pts",
    "line": 22.5,
    "home_team": "Boston Celtics",   # required for grading — never omit
    "away_team": "Indiana Pacers",   # required for grading — never omit
    "game_date": "YYYY-MM-DD",       # required for grading — never omit
    "odds_over": -115,
    "odds_under": -105,
    "player_context": {
        "pts_mean": 23.1,
        "pts_std": 6.2,
        "sample_size": 10
    },
    "game_context": {
        "is_playoff": True,
        "rest_days": 2,
        "opponent_def_rank": 8
    },
    "evidence": [...],
    "n_iterations": 10000,
    "seed": <sha256_seed>,
}, session_id="sess-20260528-prp1", bankroll=1000.0)
```

**Critical:** `home_team`, `away_team`, and `game_date` are **mandatory** on every prop trace. `ingest_traces.py` rejects prop+bet exports missing these fields.

---

## Step 5 — Downgrade thresholds for props

| Condition | Action |
|---|---|
| `sample_size` < 5 | Downgrade: low confidence; note imputed_fraction |
| `pts_std` imputed (not observed) | Append downgrade rationale |
| `aggregate_quality` < 0.7 | Downgrade to research lean |
| `aggregate_quality` < 0.3 and < 3 real facts | Research-only text, no Bet Card |
| Engine returns `status: skipped` | Repair inputs or research-only |

---

## Step 6 — Single-trace bet confirmation

When the user confirms a bet was placed:

1. Re-export the **same trace file** with `bet_record` populated
2. **Reuse the original `trace_id`** — do NOT call `analyze()` again
3. Keep the original `input_snapshot` with all game identity fields
4. `bet_record` must include: `selection_descriptor`, `odds`, `units_risked`, `stake_usd`

```json
{
  "trace": { "...original analyze output..." },
  "bet_record": {
    "selection_descriptor": "Jayson Tatum over 22.5 pts",
    "odds": -115,
    "units_risked": 1.0,
    "stake_usd": 100.0
  }
}
```

Traces with `bet_record` but missing `home_team`/`away_team`/`game_date` are rejected by `ingest_traces.py` and routed to `inbox/traces/failed/`.

---

## Post-session

```bash
python scripts/run_action_plan.py inbox/action_plans/templates/daily_trace_intake.json
python scripts/run_action_plan.py inbox/action_plans/templates/render_session_audits.json
```

After games are final, run the outcome loop — see [ops/fetch_outcomes.md](../ops/fetch_outcomes.md).
