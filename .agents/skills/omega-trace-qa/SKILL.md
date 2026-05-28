---
name: omega-trace-qa
description: Full trace QA checklist for Omega sessions. Use after each analysis or before session close to verify trace completeness, NULL/missing-data audit, sidecar integrity, evidence signal status, ingest readiness, and closing-line gating. Covers every field that ingest_traces.py validates.
---

# Omega Trace QA Checklist

---

## 1. NULL / Missing-Data Audit

Run before `analyze()` if critical inputs are missing. Never fabricate — downgrade instead.

**Thresholds:**
- `aggregate_quality >= 0.7` → quality gate permits proceeding to formal-output checks
- `0.3 <= quality < 0.7` → narrative only; no Bet Card
- `quality < 0.3` or < 3 confirmed facts → ultra-low-data text only

**Log the NULL audit to the sidecar:**
```python
from omega.trace.session_sidecar import append_null_data_audit
from pathlib import Path

append_null_data_audit(
    Path(f"inbox/sessions/{session_id}.json"),
    missing_variables=["sample_size", "starter_era"],  # variable names only
    critical=False,       # True blocks analysis entirely
    trace_ids=["sandbox-xxxx"],
)
```

`append_null_data_audit` writes `event_type: "quality_gate"`, `step: "null_data_audit"`. Accepts variable **names** only — engine numeric values (`edge_pct`, etc.) raise `ProtectedValueError`.

In `reasoning_downgrade_rationale`:
```
"Downgraded: sample_size unavailable; pts_std from 3-game window only."
```

---

## 2. Trace Identity Completeness

### Prop traces — ingest rejects bet_records if any of these are missing from `input_snapshot`:
`player_name`, `prop_type`, `line`, `home_team`, `away_team`, `game_date`

### Game traces — required in `input_snapshot`:
`home_team`, `away_team`, `league`, `seed`, `session_id`, `bankroll`

**Quick check:**
```python
snap = trace_output.get("input_snapshot", {})
required = ["player_name", "prop_type", "line", "home_team", "away_team", "game_date"]
missing = [f for f in required if f not in snap]
if missing:
    print("MISSING from input_snapshot:", missing)  # fix before export
```

---

## 3. Required Export Block Fields

Every `inbox/traces/<trace_id>.json` must include:

```json
{
  "trace": { "...analyze() output..." },
  "reasoning_inputs": {
    "sources": ["espn.com"],
    "fields_gathered": ["pts_mean", "pts_std", "is_playoff", "rest_days"],
    "missing_fields": [],
    "market_context": {"book": "betmgm", "odds_over": -110, "odds_under": -110}
  },
  "reasoning_downgrade_rationale": null,
  "reasoning_narrative": "2-4 sentence summary.",
  "trace_quality": { "aggregate_quality": 0.74 }
}
```

---

## 4. Single-Trace Policy

When user confirms a bet: **reuse the original `trace_id`**. Do NOT call `analyze()` again.

- `bet_record.line_taken` differs from `input_snapshot.line` by > 1.0 → warning logged
- `bet_record.odds_taken` differs from snapshot odds by > 25 American points → warning logged

---

## 5. Evidence Signal QA

**Valid signal types** (in HANDLER_REGISTRY): `recent_form`, `series_avg`, `home_away_split`, `last_game_outlier`, `opponent_stat_rank`, `def_matchup_weak`, `def_matchup_strong`, `pitcher_matchup`, `starter_era`, `rest_advantage`, `elimination_game`, `motivation_edge`, `blowout_risk`, `b2b_fatigue`, `weather_wind`, `usage_spike`, `usage_role_change`, `pace_up`, `pace_down`, `win_streak`, `series_lead`

**Invalid → correct mapping:**
- `consecutive_games_over_line` → `recent_form`
- `matchup_advantage` → `def_matchup_weak` / `def_matchup_strong`
- `back_to_back_adjustment` → `b2b_fatigue`
- `home_court_advantage` → `home_away_split`

**Valid `window` values ONLY:** `last_1`, `last_3`, `last_5`, `last_10`, `season`, `series`, `h2h`, `matchup`

**Shadow mode (BUG-EVIDENCE-SHADOW-001 active):** All signals recorded, none applied to sim. Add to `reasoning_downgrade_rationale`:
```
"Evidence policy mode=shadow — signals recorded for scoring, no sim adjustment applied."
```

Empty `evidence: []` → tagged `evidence_status: empty` → excluded from retrospective scoring.

---

## 6. Sidecar Integrity

### Required fields before session close:
`session_id`, `opened_at`, `closed_at`, `model_version`, `purpose`, `bankroll`, `bankroll_confirmed`, `exec_stats`, `agent_notes`, `audit_events`

### Validate:
```bash
python scripts/validate_session_sidecars.py
```

### Protected fields — NEVER in audit event `inputs`/`outputs`:
`edge_pct`, `ev_pct`, `kelly_fraction`, `units`, `confidence_tier`, `fair_price`, `no_vig_price`, `model_probability`, `over_prob`, `under_prob`

Writer raises `ProtectedValueError` and rejects the append atomically.

### Valid `event_type` values:
`preflight`, `data_provenance`, `engine_run`, `candidate_rejected`, `downgrade`, `quality_gate`, `rationale`, `bug`, `command`, `step`, `note`

### Valid `status` values: `ok`, `warn`, `fail`, `skipped`

---

## 7. Ingest Readiness

```bash
# SQLite workaround first only if on FUSE/network mount or TraceStore redirects
cp omega_traces.db /tmp/omega_traces.db
export OMEGA_TRACE_DB=/tmp/omega_traces.db

# Dry-run
python scripts/ingest_traces.py --verbose --dry-run --db "$OMEGA_TRACE_DB"

# Full ingest
python scripts/ingest_traces.py --verbose --db "$OMEGA_TRACE_DB"
```

If `OMEGA_TRACE_DB` is unset, omit `--db` and let `TraceStore` resolve the repo
default.

Check `inbox/traces/failed/` — each rejected file has a `.error.txt` sidecar. Fix identity fields, re-drop.

---

## 8. Closing-Line and Outcome Gating

```bash
# After placing bets (within 2h of game close)
python scripts/fetch_closing_lines.py --dry-run
python scripts/fetch_closing_lines.py

# After games complete
python scripts/fetch_outcomes_all.py          # all leagues, idempotent
```

Both are idempotent — re-running is safe.

---

## 9. Session Close Sequence

```bash
python scripts/ingest_traces.py --verbose --db "$OMEGA_TRACE_DB"
python scripts/validate_session_sidecars.py
python scripts/render_session_audits.py --session-id <session_id>
python scripts/report_calibration.py --db "$OMEGA_TRACE_DB" --league NBA --window-days 30
python scripts/fetch_closing_lines.py --dry-run
```

If `OMEGA_TRACE_DB` is unset, omit `--db`.

---

## 10. Common Failure Modes

| Symptom | Root cause | Fix |
|---|---|---|
| Prop trace rejected at ingest | Missing identity in `input_snapshot` | Copy from result; re-drop |
| `bet_record` rejected | Second analyze() call for confirmation | Merge onto original trace_id |
| `ProtectedValueError` on sidecar | Engine value in audit event | Remove numeric value; keep in DB |
| draw_prob > 0 on MLB | Run sentinel — check BUG-MLB-DRAW-PROB-001 | Suppress Bet Card if active |
| Evidence signals all skipped | Invalid signal_type or window | Fix against HANDLER_REGISTRY |
| `aggregate_quality` absent | No quality pass ran or `trace_quality` block missing from export | Run the quality pass or omit it intentionally; never invent a score |
| Calibration missing session | Sidecar not written or invalid | Write sidecar; re-validate |

---

## References

- `omega/trace/session_sidecar.py` — `append_audit_events`, `append_null_data_audit`, `ProtectedValueError`
- `omega/trace/persistable.py` — `PersistableTrace.from_analyze_output()`
- `scripts/ingest_traces.py` — ingest validation and failure routing
- `OMEGA_COWORK.md §6` — trace export contract, evidence signal spec
- `omega-known-bug-sentinel` skill — active bug status and gate enforcement
