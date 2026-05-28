# MLB Daily Analysis Session

Follow [OMEGA_COWORK.md](../../OMEGA_COWORK.md) and [prompts/system_prompt.txt](../system_prompt.txt) throughout.

---

## ⚠ Bet Card gate (check before session start)

Game analysis Bet Cards for MLB require BUG-3 and BUG-5 to be resolved in `service.py`. These are tracked in `docs/phase6/engine_cowork_issues_2026-05-18.md`. If those bugs are still open:
- **Moneyline** edges are correct — Bet Cards are safe
- **Run-line** edges are unreliable — use research-only lean or omit
- Use `best_bet` output only for moneyline market rows

Check the current fix status before trusting spread Bet Cards:
```bash
python -m pytest tests/core/test_engine_invariants.py::test_spread_edge_uses_cover_prob_not_moneyline_prob -v
```
Passing = BUG-5 is fixed. Failing = run-line edges are not trustworthy.

---

## Step 0 — Regenerate calibration report

```bash
python scripts/report_calibration.py --league MLB --window-days 30
```

Read `reports/latest.md` §6B before gathering evidence — weight signals by empirical accuracy.

---

## Step 1 — Preflight

```bash
python scripts/cowork_preflight.py --formal-output-gate
```

Mint session ID: `sess-YYYYMMDD-mlb1` (e.g. `sess-20260528-mlb1`).

---

## Step 2 — Resolve odds

```bash
python scripts/resolve_odds.py --kind game --league MLB --home-team "Team A" --away-team "Team B"
```

For prop stat key reference: see [reference/prop_stat_keys.md](../reference/prop_stat_keys.md).

---

## Step 3 — Gather evidence (MLB)

**Game plane evidence (`plane="game"`):**

| signal_type | what to capture | notes |
|---|---|---|
| `pace_up` / `pace_down` | Bullpen-heavy game / pitcher's duel | Affects expected total |
| `rest_advantage` | Starter fully rested (≥4 days) | `direction=home` or `away` |
| `b2b_fatigue` | Bullpen overused last 2 days | `direction=home` or `away` |
| `def_matchup_weak` / `strong` | Pitcher ERA vs. lineup handedness | `direction=home` or `away` |
| `blowout_risk` | Large talent gap + depleted bullpen on one side | no direction |

**Structured context fields specific to MLB:**

```python
# game_context fields remain universal calibration context
game_context={"is_playoff": False, "rest_days": 4}

# team contexts carry the predictive run environment fields consumed by engine
home_context={
    "off_rating": 4.5,          # runs scored per game
    "def_rating": 3.8,          # runs allowed per game
    "starter_era": 3.45,
    "park_factor": 1.05,
    "weather_wind_mph": 12.0,
}
```

**Probable pitcher** — capture as evidence signal:
```python
EvidenceSignal(
    signal_type="starter_era",
    category="pitching",
    plane="game",
    stat_key="era",
    value=3.45,   # season ERA
    source="baseball-reference.com",
    confidence=0.85,
    window="season",
    direction="home",  # whose starter this is
)
```

**Bullpen rest** — capture as evidence signal:
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

## Step 4 — Run engine

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
}, session_id="sess-20260528-mlb1", bankroll=1000.0)
```

MLB uses `"fast_score"` backend (Markov is NBA-optimized).

---

## Steps 5–8 — Same as NBA daily

See [nba_daily.md](nba_daily.md) Steps 5–8 for trace export, audit events, bet confirmation, and session close.

---

## Post-session

```bash
python scripts/run_action_plan.py inbox/action_plans/templates/daily_trace_intake.json
python scripts/run_action_plan.py inbox/action_plans/templates/render_session_audits.json
```

After games are final (~2am ET), run the outcome loop (see [ops/fetch_outcomes.md](../ops/fetch_outcomes.md)).

---

## Historical evaluation

When re-running analysis on historical dates (replay), always set:

```bash
export OMEGA_REPLAY_MODE=1
```

This blocks all live ESPN and Odds API fetches. Unset for live sessions.
