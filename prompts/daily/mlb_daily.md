# MLB Daily League Session

## Session Guard

Apply the `omega-failure-budget` skill before setup, slate building, odds resolution, analysis, trace export, ingest, or outcome attachment.
If setup/preflight/DB/trace/outcome checks exceed the failure budget, stop and produce the required failure report. Do not continue into analysis or candidate generation after a hard setup failure.

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
omega-report-calibration --league MLB --window-days 30
```

**Mandatory:** read [`prompts/reference/output_modes.md`](../reference/output_modes.md) before
producing any output, and read the machine-readable `output_modes` map from the
`var/reports/latest.md` **frontmatter** — it carries a mode **per market** (`output_modes.game` and
`output_modes.prop`, each `research_candidate` or `actionable`). Authorize **per market**: a
`RESEARCH_CANDIDATE` game market and an `ACTIONABLE` prop market (or vice versa) can coexist, so
suppress or release Bet Cards for games and props off their own market's mode. `RESEARCH_CANDIDATE`
restricts only what you show the user for that market — it never skips the engine. (The scalar
`output_mode` is a conservative fallback only; prefer the per-market map.)

Read `var/reports/latest.md`:
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
omega-cowork-preflight --formal-output-gate
```

If this does not print `cowork_preflight_ready`, do not emit Bet Cards.

Mint one league session ID: `sess-YYYYMMDD-mlb1`.

Bootstrap or update `var/inbox/sessions/<session_id>.json` and append a preflight
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
omega-resolve-odds --kind game --league MLB --home-team "Team A" --away-team "Team B"
```

Player props:

```bash
omega-resolve-odds --kind prop --league MLB --player "Player Name" --prop-type strikeouts_pitched --line 5.5
```

Default book is BetMGM unless the user asks for broader line shopping. Append
`data_provenance` events for sources used. Never expose API keys.

The source book is recorded on every persisted bet (`bet_ledger.bookmaker`).
When you line-shop (`--line-shopping` / `--all-books`), the resolver payload
includes a `best_prices` block — surface it as the advisory "Best available"
line on the Bet Card per
[`prompts/reference/output_modes.md`](../reference/output_modes.md#book-provenance--line-shopping-in-the-bet-card).
It is advisory only: never recompute edge/EV/Kelly against a shopped price.

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

Both `home_context` and `away_context` must contain all required team context keys (e.g., `off_rating`, `def_rating` for MLB) to guarantee `context_source="provided"` and satisfy the calibration eligibility gate.

---

## Step 5 - Typed Evidence

Capture material evidence as `EvidenceSignal` records. Do not hide predictive
metrics in unstructured notes.

Useful MLB evidence:

| signal_type | what to capture | plane | category | direction |
|---|---|---|---|---|
| `park_factor_evidence` | Ballpark hitter/pitcher friendliness | game/player | situational | optional |
| `weather_wind` | Wind speed/direction effect on run environment | game | situational | optional |
| `pitcher_matchup` | Batter-vs-pitcher handedness/history edge | player | matchup | required (over/under/neutral) |
| `starter_era` | Starting pitcher's ERA for the directional team | game | matchup | required (home/away) |
| `rest_advantage` | Rested starter or bullpen edge | game | situational | required (home/away) |
| `blowout_risk` | Probability game is non-competitive | game | situational | optional |

Example:

```python
EvidenceSignal(
    signal_type="weather_wind",
    category="situational",
    plane="game",
    value="15mph blowing out to center field",
    source="weather.gov",
    confidence=0.90,
    window="matchup",
    direction="over",
)
```

---

## Step 6 - Run Game Engine

**Always run the engine when it is available, regardless of the Step 0 output mode.**
`RESEARCH_CANDIDATE` only restricts what you present to the user; it never means "skip
`analyze()`". Running the engine and persisting traces is how calibration data accumulates.

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

Follow the canonical procedure in
[`prompts/reference/engine_output_validation.md`](../reference/engine_output_validation.md).
For MLB, the sport-specific input-context fields to verify are `starter_era`, `park_factor`, and
`weather_wind_mph` (where material). Downgrades here are user-facing only — the engine already ran
and the trace still persists (see [`output_modes.md`](../reference/output_modes.md)).

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

After each successful `analyze()` call, write `var/inbox/traces/<trace_id>.json`.
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
omega-run-action-plan fixtures/action_plans/daily_trace_intake.json
omega-run-action-plan fixtures/action_plans/render_session_audits.json
```

After games are final, run the outcome loop described in
[ops/fetch_outcomes.md](../ops/fetch_outcomes.md).
