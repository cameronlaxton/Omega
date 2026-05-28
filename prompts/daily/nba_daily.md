# NBA Daily League Session

Use this prompt for the complete NBA betting surface: game moneylines, spreads,
totals, team totals where supported, and player props. Do not run a separate
props prompt. Props and game bets share the same league context,
injury map, rest map, odds provenance, sidecar, and session audit.

Follow [OMEGA_COWORK.md](../../OMEGA_COWORK.md) and
[prompts/system_prompt.txt](../system_prompt.txt) throughout.

---

## Step 0 - Calibration Snapshot

Run:

```bash
python scripts/report_calibration.py --league NBA --window-days 30
```

Read `reports/latest.md` before analysis:
- Section 3: game-plane Brier/ECE; flag ECE > 0.05.
- Section 3B: prop-plane pair count; metrics may be suppressed if n < 10.
- Section 6B: evidence signal performance; trust `predictive`, discount
  `noise`, treat `insufficient_n` as unproven.

---

## Step 1 - Formal Gate And Session

Run:

```bash
python scripts/cowork_preflight.py --formal-output-gate
```

If this does not print `cowork_preflight_ready`, do not emit Bet Cards.
Research-only narrative is allowed, without model probability, edge, EV,
Kelly, units, confidence tier, or trace_id.

Mint one league session ID: `sess-YYYYMMDD-nba1`.

Bootstrap or update `inbox/sessions/<session_id>.json` and append:
- `event_type=preflight`
- `step=cowork_preflight`
- `status=ok|warn|fail`
- notes with engine state and bankroll confirmation only

---

## Step 2 - Build One League Slate

Create one candidate list containing both game markets and player props:

```text
candidate_id | market_family | matchup | player | market | line | source | status
```

Market families:
- `game`: moneyline, spread, total, team_total where sourced.
- `prop`: player points, rebounds, assists, 3pm, PRA, steals, blocks, and other
  supported NBA prop types from [prop_stat_keys.md](../reference/prop_stat_keys.md).

Do not rank or discard props before the injury/rest/context pass. A player prop
can be the best NBA bet of the day even when the game market is a pass.

---

## Step 3 - Resolve Odds

Resolve direct market odds before analysis.

Game markets:

```bash
python scripts/resolve_odds.py --kind game --league NBA --home-team "Team A" --away-team "Team B"
```

Player props:

```bash
python scripts/resolve_odds.py --kind prop --league NBA --player "Player Name" --prop-type pts --line 22.5
```

Default book is BetMGM unless the user asks for broader line shopping. Append
`data_provenance` audit events for sources used. Never expose API keys.

If a prop line is unavailable from the typed resolver or a direct sportsbook
board, keep it research-only. Do not pass guessed or milestone-derived lines
to `analyze()`.

---

## Step 4 - League Context And Injury Translation

Gather one shared NBA context pack before any `analyze()` call:
- injury statuses, minute limits, questionable/probable notes
- starting lineup or role changes
- rest days and back-to-back flags
- pace environment
- defensive matchup notes
- blowout risk
- sportsbook line source and timestamp

Injury/news protocol:
1. Noticing news is not enough.
2. Translate the news into structured model inputs before analysis:
   - game plane: `off_rating`, `def_rating`, `pace`, and typed evidence
   - prop plane: `minutes`, `usage_rate`, `{stat}_mean`, `{stat}_std`, or
     explicit `injury_impact`
3. If the impact cannot be quantified from cited pre-decision sources, append
   a `quality_gate/null_data_audit` event and downgrade that candidate to
   research-only.

Basketball team contexts require possession-adjusted ratings:

```python
home_context={"off_rating": 118.0, "def_rating": 108.0, "pace": 100.0}
away_context={"off_rating": 115.0, "def_rating": 110.0, "pace": 98.0}
```

Never pass raw FG%, opponent FG%, eFG%, or other fractional proxies as
`off_rating` or `def_rating`.

`game_context` is mandatory for every NBA game and prop trace:

```python
game_context={
    "is_playoff": False,
    "rest_days": 2,
    "injury_impact": 1.0,  # include when relevant; omit only if not applicable
}
```

---

## Step 5 - Typed Evidence

Express material evidence as `EvidenceSignal` objects. Do not hide predictive
metrics in prose.

Markov-approved game-plane signal types:

| signal_type | effect | direction |
|---|---|---|
| `pace_up` | faster game environment | optional |
| `pace_down` | slower game environment | optional |
| `rest_advantage` | rested team scoring boost | `home` or `away` |
| `b2b_fatigue` | fatigued team scoring penalty | `home` or `away` |
| `def_matchup_weak` | offense vs weak defender | `home` or `away` |
| `def_matchup_strong` | offense vs strong defender | `home` or `away` |
| `usage_role_change` | key player restricted/elevated | `home` or `away` |
| `blowout_risk` | variance/routing downgrade signal | optional |

All other signal types are audit-only unless the engine maps them.

---

## Step 6 - Run Game Engine

NBA default backend: `simulation_backend="markov_state"`. Use `fast_score`
only if Markov skips and the downgrade is recorded.

```python
analyze({
    "league": "NBA",
    "home_team": "...",
    "away_team": "...",
    "game_date": "YYYY-MM-DD",
    "home_context": {"off_rating": ..., "def_rating": ..., "pace": ...},
    "away_context": {"off_rating": ..., "def_rating": ..., "pace": ...},
    "game_context": {"is_playoff": False, "rest_days": 2},
    "odds": {...},
    "evidence": [...],
    "simulation_backend": "markov_state",
    "n_iterations": 10000,
    "seed": <sha256_seed>,
}, session_id="sess-YYYYMMDD-nba1", bankroll=1000.0)
```

---

## Step 7 - Run Prop Engine

Every prop trace must include `home_team`, `away_team`, and `game_date`.

```python
analyze({
    "player_name": "Player Name",
    "league": "NBA",
    "prop_type": "pts",
    "line": 22.5,
    "home_team": "Boston Celtics",
    "away_team": "Indiana Pacers",
    "game_date": "YYYY-MM-DD",
    "odds_over": -115,
    "odds_under": -105,
    "player_context": {
        "pts_mean": 23.1,
        "pts_std": 6.2,
        "sample_size": 10,
        "minutes": 36,
        "usage_rate": 0.29
    },
    "game_context": {"is_playoff": False, "rest_days": 2},
    "evidence": [...],
    "n_iterations": 10000,
    "seed": <sha256_seed>,
}, session_id="sess-YYYYMMDD-nba1", bankroll=1000.0)
```

Minimum prop context:
- `{prop_type}_mean`
- `{prop_type}_std`, observed if possible
- `sample_size`
- usage/minutes when injury or role changes are relevant

If `sample_size < 5`, observed std is unavailable, or the injury adjustment is
not quantified, downgrade or keep research-only.

---

## Step 8 - NULL / Missing Data Audit

Before exporting any trace or presenting a Bet Card:
- Confirm `game_context.is_playoff` and `game_context.rest_days` are present.
- Confirm injury impacts were translated or explicitly audited as missing.
- Confirm engine output did not return NULL/0.0 for protected fields such as
  `recommended_units`, `model_prob`, or `edge_pct`.

If missing data is found, append a `quality_gate/null_data_audit` event listing
variable names only. Do not put edge, EV, Kelly, units, confidence tier, or
model probability in sidecar notes.

Critical missing data means research-only; drop the Bet Card.

---

## Step 9 - Export, Confirm, Close

After each successful `analyze()` call, write `inbox/traces/<trace_id>.json`.
Nest `reasoning_inputs`, `reasoning_narrative`,
`reasoning_downgrade_rationale`, and `trace_quality` inside the inner `trace`
block.

Append audit events for:
- `engine_run`
- `candidate_rejected`
- `downgrade`
- `quality_gate`
- `bug`

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
