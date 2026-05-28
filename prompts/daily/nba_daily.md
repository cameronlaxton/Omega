# NBA Daily Analysis Session

Follow [OMEGA_COWORK.md](../../OMEGA_COWORK.md) and [prompts/system_prompt.txt](../system_prompt.txt) throughout.

---

## Step 0 — Regenerate calibration report

Run this before anything else so §6B evidence weights are current:

```bash
python scripts/report_calibration.py --league NBA --window-days 30
```

Then read `reports/latest.md` — specifically:
- §3: current Brier/ECE on game plane (flag if ECE > 0.05)
- §3B: prop plane pair count (metrics suppressed if < 10)
- §6B: evidence signal performance — trust `predictive`, discount `noise`, treat `insufficient_n` as unproven and positive `cal_gap` as overconfident

---

## Step 1 — Preflight

```bash
python scripts/cowork_preflight.py
```

Mint session ID: `sess-YYYYMMDD-nba1` (replace date with today's date, e.g. `sess-20260528-nba1`).

Bootstrap sidecar and append first audit event:
- `event_type=preflight`, `step=cowork_preflight`, `status=ok|warn|fail`
- notes: engine green/warn/fail + bankroll confirmed

---

## Step 2 — Resolve odds

```bash
python scripts/resolve_odds.py --kind game --league NBA --home-team "Team A" --away-team "Team B"
```

Default book: BetMGM. Append `data_provenance` events per source used. Never expose `OMEGA_ODDS_API_KEY` in notes.

---

## Step 3 — Gather evidence (NBA)

Express all material evidence as typed `EvidenceSignal` objects. Never bake adjustments into hand-tuned context means.

**For game plane (`plane="game"`) — Markov-approved signal types only:**

| signal_type | effect | direction required? |
|---|---|---|
| `pace_up` | +6% game pace | no |
| `pace_down` | -8% game pace | no |
| `rest_advantage` | +4% scoring rate rested team | yes (`home`/`away`) |
| `b2b_fatigue` | -6% scoring rate fatigued team | yes (`home`/`away`) |
| `def_matchup_weak` | +5% offense vs. weak defender | yes (`home`/`away`) |
| `def_matchup_strong` | -5% offense vs. strong defender | yes (`home`/`away`) |
| `usage_role_change` | -7% team rate when key player restricted/elevated | yes (`home`/`away`) |
| `blowout_risk` | -2% momentum variance | no |

All other signal types are audit-only (scored retrospectively, no Markov effect).

**`game_context` is mandatory for calibration:**

```python
game_context={
    "is_playoff": True,   # bool — always required; False for regular season
    "rest_days": 1,       # int — days since last game; 0 = back-to-back
}
```

---

## Step 4 — Run engine

NBA default backend: `simulation_backend="markov_state"`. Use `"fast_score"` only if Markov skips.

```python
analyze({
    "league": "NBA",
    "home_team": "...",
    "away_team": "...",
    "game_date": "YYYY-MM-DD",
    "home_context": {"off_rating": ..., "def_rating": ..., "pace": ...},
    "away_context": {"off_rating": ..., "def_rating": ..., "pace": ...},
    "game_context": {"is_playoff": True, "rest_days": 2},
    "odds": {...},
    "evidence": [...],
    "simulation_backend": "markov_state",
    "n_iterations": 10000,
    "seed": <sha256_seed>,
}, session_id="sess-20260528-nba1", bankroll=1000.0)
```

Seed derivation: `int.from_bytes(hashlib.sha256(f"{prompt}|{date}".encode()).digest()[:4], "big")`

---

## Step 5 — Export traces

Write to `inbox/traces/<trace_id>.json` after each analyze call. Nest `reasoning_inputs`, `reasoning_narrative`, `reasoning_downgrade_rationale`, and `trace_quality` **inside** the inner `trace` block.

---

## Step 6 — Audit events

Append after each major step:
- `engine_run` per analyze call (include `trace_ids`, note what was provided/missing)
- `candidate_rejected` for any game/prop dropped before analysis
- `downgrade` when confidence tier was lowered (include rationale)
- `bug` for any anomaly (reference file + trace_id)
- **Never** put edge%, EV%, Kelly, units, confidence tier, model probabilities into event notes

---

## Step 7 — Bet confirmation (optional)

If user confirms a bet: re-export the **same trace** with `bet_record` populated. Reuse the original `trace_id` and `input_snapshot`. Do NOT re-run `analyze()`.

---

## Step 8 — Close session

Set `closed_at`, write final `agent_notes` summary, stop. Do not run `ingest_traces` or `render_audit` from inside this session.

---

## Post-session (run separately after session closes)

```bash
python scripts/run_action_plan.py inbox/action_plans/templates/daily_trace_intake.json
python scripts/run_action_plan.py inbox/action_plans/templates/render_session_audits.json
```

After games are final (~midnight ET), also run the outcome loop (see [ops/fetch_outcomes.md](../ops/fetch_outcomes.md)).
