# MLB Daily League Session

Use this prompt for the complete MLB betting surface: moneylines, run lines,
totals, team totals, first-five markets where supported, and player props. Do
not run a separate props prompt. Game bets and props must share the
same pitcher, lineup, park, weather, bullpen, odds, sidecar, and audit context.

Follow [OMEGA_COWORK.md](../../OMEGA_COWORK.md) and
[prompts/system_prompt.txt](../system_prompt.txt) throughout.

---

## Step 0 - Calibration Snapshot

Run:

```bash
python scripts/report_calibration.py --league MLB --window-days 30
```

Read `reports/latest.md`:
- Section 3: game-plane calibration.
- Section 3B: prop-plane sample count.
- Section 6B: empirical evidence signal performance.

If an older bug note conflicts with current tests, trust the current tests and
trace behavior. Verify run-line safety with:

```bash
python -m pytest tests/core/test_engine_invariants.py::test_spread_edge_uses_cover_prob_not_moneyline_prob -q
```

Passing means run-line cover probability is the edge source. Failing means
run-line candidates are research-only.

---

## Step 1 - Formal Gate And Session

Run:

```bash
python scripts/cowork_preflight.py --formal-output-gate
```

If this does not print `cowork_preflight_ready`, do not emit Bet Cards.

Mint one league session ID: `sess-YYYYMMDD-mlb1`.

Bootstrap or update `inbox/sessions/<session_id>.json` and append a preflight
audit event with status and bankroll confirmation.

---

## Step 2 - Build One League Slate

Create one candidate list containing both game markets and player props:

```text
candidate_id | market_family | matchup | player | market | line | source | status
```

Market families:
- `game`: moneyline, run_line, total, team_total, first_5 where supported.
- `prop`: hits, total bases, RBI, HR, stolen bases, pitcher strikeouts,
  outs recorded, earned runs, hits allowed, walks allowed, and supported keys
  from [prop_stat_keys.md](../reference/prop_stat_keys.md).

Do not analyze MLB props in a separate session; pitcher, lineup, park, and
weather context often decides both the game market and the prop market.

---

## Step 3 - Resolve Odds

Game markets:

```bash
python scripts/resolve_odds.py --kind game --league MLB --home-team "Team A" --away-team "Team B"
```

Player props:

```bash
python scripts/resolve_odds.py --kind prop --league MLB --player "Player Name" --prop-type strikeouts_pitched --line 5.5
```

Default book is BetMGM unless the user asks for broader line shopping. Append
`data_provenance` events for sources used. Never expose API keys.

If a prop line is unavailable from the typed resolver or a direct sportsbook
board, keep it research-only. Do not pass guessed or narrative-only lines to
`analyze()`.

---

## Step 4 - League Context

Gather one shared MLB context pack:
- confirmed/probable starters
- starter ERA and recent workload
- bullpen usage over the last three days
- lineup availability and handedness matchups
- park factor
- weather and wind
- team run rates
- rest/travel where relevant
- sportsbook line source and timestamp

Structured predictive fields must go into team contexts, not prose:

```python
home_context={
    "off_rating": 4.5,          # runs scored per game
    "def_rating": 3.8,          # runs allowed per game
    "starter_era": 3.45,
    "park_factor": 1.05,
    "weather_wind_mph": 12.0,
}
away_context={
    "off_rating": 4.1,
    "def_rating": 4.2,
    "starter_era": 4.10,
}
game_context={
    "is_playoff": False,
    "rest_days": 4,
}
```

Do not feed basketball-style ratings, raw batting average, or raw ERA into
`off_rating`/`def_rating`. For MLB those fields are runs scored/allowed per
game.

---

## Step 5 - Typed Evidence

Capture material evidence as `EvidenceSignal` records. Do not hide predictive
metrics in unstructured notes.

Useful MLB evidence:

| signal_type | what to capture | direction |
|---|---|---|
| `pace_up` / `pace_down` | run environment / pitcher duel | optional |
| `rest_advantage` | rested starter or bullpen edge | `home` or `away` |
| `b2b_fatigue` | bullpen overuse | `home` or `away` |
| `def_matchup_weak` | lineup advantage vs starter handedness | `home` or `away` |
| `def_matchup_strong` | pitcher advantage vs lineup | `home` or `away` |
| `blowout_risk` | large mismatch plus depleted bullpen | optional |

Example:

```python
EvidenceSignal(
    signal_type="b2b_fatigue",
    category="fatigue",
    plane="game",
    value="bullpen used 25+ pitches last 2 days",
    source="baseballsavant.mlb.com",
    confidence=0.75,
    window="last_3",
    direction="away",
)
```

---

## Step 6 - Run Game Engine

MLB uses `simulation_backend="fast_score"`.

```python
analyze({
    "league": "MLB",
    "home_team": "...",
    "away_team": "...",
    "game_date": "YYYY-MM-DD",
    "home_context": {
        "off_rating": 4.5,
        "def_rating": 3.8,
        "starter_era": 3.45,
        "park_factor": 1.02,
        "weather_wind_mph": 8.0,
    },
    "away_context": {
        "off_rating": 4.1,
        "def_rating": 4.2,
        "starter_era": 4.10,
    },
    "game_context": {"is_playoff": False, "rest_days": 4},
    "odds": {...},
    "evidence": [...],
    "simulation_backend": "fast_score",
    "n_iterations": 10000,
    "seed": <sha256_seed>,
}, session_id="sess-YYYYMMDD-mlb1", bankroll=1000.0)
```

---

## Step 7 - Run Prop Engine

Every prop trace must include `home_team`, `away_team`, and `game_date`.

```python
analyze({
    "player_name": "Pitcher Name",
    "league": "MLB",
    "prop_type": "strikeouts_pitched",
    "line": 5.5,
    "home_team": "Texas Rangers",
    "away_team": "Houston Astros",
    "game_date": "YYYY-MM-DD",
    "odds_over": -110,
    "odds_under": -110,
    "player_context": {
        "strikeouts_pitched_mean": 5.8,
        "strikeouts_pitched_std": 2.1,
        "sample_size": 10,
        "k_per_9": 9.4
    },
    "game_context": {
        "is_playoff": False,
        "rest_days": 4,
        "park_factor": 1.02
    },
    "evidence": [...],
    "n_iterations": 10000,
    "seed": <sha256_seed>,
}, session_id="sess-YYYYMMDD-mlb1", bankroll=1000.0)
```

Minimum prop context:
- `{prop_type}_mean`
- `{prop_type}_std`, observed if possible
- `sample_size`
- matchup factors such as handedness, projected lineup contact/K rate, park,
  and weather where relevant

If a starter change, lineup uncertainty, or weather risk cannot be quantified,
append a `quality_gate/null_data_audit` event and keep the candidate
research-only.

---

## Step 7b - Engine Output Nullability Check

**Execute immediately after each `analyze()` call returns, before any other processing.**

Evaluate the engine response payload for fields returning NULL, `0.0`, `"undefined"`,
empty collections, or otherwise missing values. This is a structural validation step,
not a data quality judgment.

**Mandatory checks:**

1. **Request-level identity fields** (must be echoed):
   - `league`, `player_name` (for props), `home_team`, `away_team`, `game_date`

2. **Result object fields** (engine-owned; must exist if `status != "skipped"`):
   - `model_prob` (or `over_prob`/`under_prob` for player props)
   - `fair_price` / `no_vig_price`
   - `edge_pct`
   - `recommended_units`
   - `confidence_tier`
   - `trace_id`

3. **Input context fields** (your responsibility; must be populated before calling `analyze()`):
   - `game_context.is_playoff` and `game_context.rest_days` (non-null integers)
   - `{prop_type}_mean` and `{prop_type}_std` (for player props)
   - `home_context.off_rating`, `def_rating`, `starter_era` (for game analysis)
   - `park_factor`, `weather_wind_mph` (when relevant to game/prop outcome)
   - `sample_size` ≥ 5 (for player props, if available)

**Capture strategy:**

If any of the above are NULL or missing:
- Build a **null_fields** list with clean field paths: `["result.recommended_units", "game_context.rest_days"]`
- Append a `quality_gate` event with `event_type="quality_gate/null_data_audit"`
  and `notes="Null fields: " + ", ".join(null_fields)`
- Do **not** include numeric protected values in the event notes; only field names.

**Decision logic:**

- If result-level fields are NULL and `status != "skipped"`: **engine error → downgrade to research-only**.
- If input context fields are NULL: **your input was incomplete → downgrade to research-only** and log which context field(s) were missing.
- If `sample_size < 5` for a player prop: **research-only** unless explicitly backfilled from reliable source.
- If `park_factor` or `weather_wind_mph` are unavailable for a game analysis where they're expected: **log as missing and downgrade if material**.
- If all required fields are present: proceed to trace export.

**Example null_fields audit event:**

```json
{
  "ts": "2026-05-28T20:15:00Z",
  "event_type": "quality_gate/null_data_audit",
  "step": "engine_output_validation",
  "status": "warn",
  "notes": "Null fields: ['game_context.rest_days']. Downgraded to research-only.",
  "inputs": [],
  "outputs": [],
  "trace_ids": ["sandbox-xxxxx"]
}
```

---

## Step 8 - Pre-Export Quality Gate

Before exporting any trace or presenting a Bet Card:
- Confirm `game_context.is_playoff` and `game_context.rest_days` were populated and validated (Step 7b).
- Confirm `starter_era`, `park_factor`, and weather fields were mapped where material and available (Step 7b).
- Confirm engine output passed nullability check (Step 7b).

If critical missing data was found and logged in Step 7b, the trace is already
downgraded to research-only. Do not emit Bet Cards for research-only traces.

---

## Step 9 - Export, Confirm, Close

After each successful `analyze()` call, write `inbox/traces/<trace_id>.json`.
Nest `reasoning_inputs`, `reasoning_narrative`,
`reasoning_downgrade_rationale`, and `trace_quality` inside the inner `trace`
block.

If the user confirms a bet, re-export the same trace with `bet_record`
populated. Reuse the original `trace_id`; do not rerun `analyze()`.

Close the session by setting `closed_at` and a short `agent_notes` summary.
Do not run ingest or audit rendering inside the live betting session.

---

## Post-Session

Run separately after session close:

```bash
python scripts/run_action_plan.py inbox/action_plans/templates/daily_trace_intake.json
python scripts/run_action_plan.py inbox/action_plans/templates/render_session_audits.json
```

After games are final, run the outcome loop described in
[ops/fetch_outcomes.md](../ops/fetch_outcomes.md).
