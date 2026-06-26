# Omega Daily All-Sports Calibration Sweep

**Purpose:** Generate traces for every matchup happening today across all
Omega-supported leagues. This is a calibration-first session — `RESEARCH_CANDIDATE`
output is expected and acceptable. The goal is a healthy daily trace volume regardless
of which sports are in season, so calibration data accumulates continuously.

For deep bet-quality analysis of a specific league, use the league-specific
prompt ([`mlb_daily.md`](mlb_daily.md), [`nba_daily.md`](nba_daily.md),
[`wnba_daily.md`](wnba_daily.md)) after this sweep has filed the day's calibration
traces.

---

## Session Guard

Apply the `omega-failure-budget` skill before proceeding. If preflight or DB
checks fail the budget, stop and file the failure report.

**MCP schema load — do this before calling any `omega_*` tool:**

```text
ToolSearch("select:mcp__plugin_omega-llm-interface_omega__omega_list_events,mcp__plugin_omega-llm-interface_omega__omega_resolve_odds,mcp__plugin_omega-llm-interface_omega__omega_run_batch")
```

Omega MCP tools are deferred at session start. Calling them without loading
their schemas first returns `InputValidationError`. If `ToolSearch` returns
nothing, the server hasn't booted — wait 30 seconds and retry once. If still
unavailable after one retry, the Python fallback in AGENTS.md §"Batch analysis
rule" is authorized; follow every constraint there (formal gate, deterministic
seeds, at least one `EvidenceSignal` per trace or explicit rationale, **no
hardcoded `aggregate_quality`**).

---

## Step 0 — Bootstrap and Calibration Snapshot

Apply the `omega-session-bootstrap` skill.

Mint session ID: `sess-YYYYMMDD-all1`

Run a cross-league calibration snapshot:

```bash
omega-report-calibration --window-days 30
```

Note which markets are `ACTIONABLE` vs `RESEARCH_CANDIDATE`. For this session,
accept `RESEARCH_CANDIDATE` on all markets — all traces still run through the
engine and accumulate calibration data.

---

## Step 1 — Discover Today's Slate

For each league in the priority groups below, call:

```python
omega_list_events(league="<LEAGUE>")
```

If a league returns `status: "empty"` or `status: "unavailable"`, skip it and
move on. Record all discovered games as `(league, home_team, away_team, event_id, commence_time)`.

**Priority 1 — Major US leagues** (analyze ALL games found today):

| League | Season window | Notes |
|---|---|---|
| `NBA` | Oct–Jun | Skip if off-season |
| `WNBA` | May–Oct | Skip if off-season |
| `MLB` | Mar–Oct | Skip if off-season |
| `NHL` | Oct–Jun | Skip if off-season |

**Priority 2 — American Football** (seasonal Sep–Feb):

| League | Notes |
|---|---|
| `NFL` | Skip if no games today |

**Priority 3 — Soccer** (analyze up to 5 games per league):

Scan in this order; stop each league at 5 games:

`EPL`, `MLS`, `LA_LIGA`, `SERIE_A`, `LIGUE_1`, `BUNDESLIGA`, `LIGA_MX`,
`CHAMPIONS_LEAGUE`, `EUROPA_LEAGUE`, `FIFA_WORLD_CUP_2026`, `COPA_AMERICA`,
`EURO`, `NATIONS_LEAGUE`

**Priority 4 — Tennis** (up to 3 featured matches per tour):

| League | Notes |
|---|---|
| `ATP` | Active only during tournament weeks |
| `WTA` | Active only during tournament weeks |
| `WIMBLEDON`, `FRENCH_OPEN`, `AUSTRALIAN_OPEN`, `US_OPEN_TENNIS` | Use specific league key during Grand Slam weeks |

**Priority 5 — Golf** (only if an active tournament round is today):

| League | Notes |
|---|---|
| `PGA` | Check if a PGA Tour event is live today |
| `LPGA` | Check if an LPGA event is live today |
| `LIV` | Check if LIV is in a tournament today |

For golf, `omega_list_events` returns head-to-head matchup props rather than
team games — treat each returned event as a player-vs-player prop analysis.

**Priority 6 — Fighting** (only if a card is today):

| League | Notes |
|---|---|
| `UFC` | Main card bouts only |
| `BOXING` | Main event + co-main only |

**Volume cap:** Hard cap at 30 game/match traces total. If discoveries exceed
30, take all Priority 1 games first, then fill remaining slots by priority order.

---

## Step 2 — Resolve Odds

For each discovered game, call:

```python
omega_resolve_odds(kind="game", league="<LEAGUE>", home_team="<HOME>", away_team="<AWAY>")
```

If odds are unavailable (`status: "unavailable"`), proceed with the analysis
anyway — set `odds={}` in the request and note in `reasoning_downgrade_rationale`
that no odds were resolved. The trace is calibration-only, not betting-grade.

---

## Step 3 — Context Pass (Lightweight)

For each game, gather minimum viable context. This is a sweep session — use league
baseline defaults when team-specific stats require more than 1–2 quick lookups.
Always record what was inferred vs. sourced.

**Context tier definitions:**

- `sourced` — actual team stats from ESPN or official source; `aggregate_quality ≥ 0.65`
- `inferred` — league baseline defaults substituted; `aggregate_quality = 0.45`

**Basketball (NBA/WNBA):**
```python
home_context = {"off_rating": <pts_per_100_or_baseline>, "def_rating": <opp_pts_per_100_or_baseline>, "pace": <possessions_or_baseline>}
```
Baselines: NBA → off=114, def=114, pace=99 | WNBA → off=100, def=100, pace=83

**Baseball (MLB):**
```python
home_context = {"off_rating": <runs_per_game_or_4.4>, "def_rating": <runs_allowed_per_game_or_4.4>, "starter_era": <era_or_4.25>}
game_context = {"is_playoff": False, "rest_days": 1, "park_factor": 1.0, "weather_wind_mph": 0.0}
```
Baseline: off=4.4, def=4.4, starter_era=4.25. Set `rest_days` to actual days since last game; if unknown use 1.

**Hockey (NHL):**
```python
home_context = {"off_rating": <goals_per_game_or_3.0>, "def_rating": <goals_allowed_per_game_or_3.0>}
game_context = {"is_playoff": False, "rest_days": 1}
```
Baseline: off=3.0, def=3.0.

**Football (NFL):**
```python
home_context = {"off_rating": <pts_per_game_or_23.0>, "def_rating": <pts_allowed_per_game_or_23.0>}
game_context = {"is_playoff": False, "rest_days": 7, "is_dome": False}
```
Baseline: off=23, def=23. Check `is_dome` from venue info.

**Soccer (all leagues):**
```python
home_context = {"off_rating": <goals_per_game_or_avg_total/2>, "def_rating": <goals_allowed_per_game_or_avg_total/2>}
away_context = {"off_rating": <goals_per_game_or_avg_total/2>, "def_rating": <goals_allowed_per_game_or_avg_total/2>}
game_context = {"is_playoff": False, "rest_days": 3}
```
Use `avg_total/2` from `leagues.py` as the per-team scoring baseline (e.g., EPL avg_total=2.7 → use 1.35 each side).

**Tennis (ATP/WTA/Slams):**
```python
home_context = {"serve_win_prob": <or_0.65>, "surface_adj": 0.0}
away_context = {"serve_win_prob": <or_0.65>, "surface_adj": 0.0}
game_context = {"is_playoff": False, "rest_days": 1}
```
Baseline: serve_win_prob=0.65. If surface and player ranking are known, adjust by ±0.02–0.05.

**Golf (player h2h matchup props):**
Use `omega_analyze_prop` with `kind="prop"`:
```python
player_name = "<Player A>",  # listed first in the matchup
prop_type = "matchup_win",
player_context = {"sg_total": <strokes_gained_total_or_0.0>, "round_std": 3.0}
game_context = {"is_playoff": False, "rest_days": 1}
```
SG:Total available at pgatour.com/stats. If unavailable, use sg_total=0.0 and note inferred.

**Fighting (UFC/Boxing):**
```python
home_context = {"win_probability": 0.5}  # adjust from moneyline implied prob if odds available
game_context = {"is_playoff": False, "rest_days": 60}
```

**`rest_days` rule:** rest_days = days since last game − 1. Back-to-back = 0. If unknown, use sport baseline (NFL=7, MLB=1, NBA/WNBA=1, NHL=1, Soccer=3, Tennis=1, Fighting=60).

---

## Step 4 — Minimal Evidence

For each game, add at least one `EvidenceSignal` if a clear directional factor is
present (rest advantage, home/away form differential, known injury impact). If none
is evident from the lightweight context pass, use `evidence=[]` and set
`reasoning_downgrade_rationale` to note the absence.

Do not spend time researching evidence for every game — this is a sweep session.
One solid signal per game is fine; zero is acceptable with a rationale.

---

## Step 5 — Batch Analysis

> **No Python scripts.** AGENTS.md §"Batch analysis rule" prohibits writing a
> script when MCP is available. Use `omega_run_batch` directly via the loaded
> MCP schema. Never post-process trace files after the fact and never hardcode
> `trace_quality.aggregate_quality` — the engine computes it from what you
> actually provided. Traces with a manually set quality score contaminate the
> calibration loop.

Collect all game entries and run in one batch:

```python
omega_run_batch(
    entries=[
        {
            "kind": "game",
            "league": "<LEAGUE>",
            "home_team": "<HOME>",
            "away_team": "<AWAY>",
            "game_date": "<YYYY-MM-DD>",
            "home_context": {...},
            "away_context": {...},
            "game_context": {"is_playoff": False, "rest_days": <N>},
            "odds": {...},           # from Step 2; empty dict if unavailable
            "evidence": [...],
        },
        # ... all discovered games
    ],
    bankroll=1000.0,
    session_id=session_id,
)
```

For golf h2h props and fighting bouts, call `omega_analyze_prop` individually
(these require player-context format, not the team-game format).

**Always run the engine regardless of output mode.** `RESEARCH_CANDIDATE` restricts
what is shown to you — it never skips `analyze()`. Running the engine is how
calibration data accumulates.

---

## Step 6 — Post-Analysis Narrative

After all traces are filed, produce a brief sweep summary (not a full Bet Card
report):

```markdown
## Daily Sweep — <YYYY-MM-DD>

Games discovered: <N> across <sport list>
Traces generated: <N>
  - ACTIONABLE: <N>   (formatted as Bet Cards below if any)
  - RESEARCH_CANDIDATE: <N>
Leagues skipped (no activity today): <list>
Context tier:
  - sourced: <N> traces
  - inferred: <N> traces

### Actionable candidates (if any)
<Bet Card format per output_modes.md — only for ACTIONABLE markets>

### Research watchlist (top 3–5 most interesting, if any)
<Brief research lean per presentation_contract.md>
```

If no markets are ACTIONABLE, note that and skip the Bet Card section. Do not
fabricate edge, EV, Kelly, units, confidence tiers, or trace IDs in prose.

---

## Step 7 — Export and Close

For each trace, write `var/inbox/traces/<trace_id>.json` with:
- `reasoning_inputs.sources` — cite each source (ESPN, Odds API, pgatour.com, etc.)
- `reasoning_inputs.fields_gathered` — list fields that were sourced vs. inferred
- `reasoning_downgrade_rationale` — if baseline context was used, state it explicitly
- Do **not** set `trace_quality.aggregate_quality` — the engine computes it from
  what you actually provided (the sources / fields_gathered / downgrade rationale
  above). A hand-set score contaminates the calibration loop (see Step 5).

Close the session:
```python
session["closed_at"] = "<ISO timestamp>"
session["agent_notes"] = "Daily sweep: <N> games across <sports>. <N> sourced, <N> inferred. <N> actionable, <N> research."
```

---

## Post-Session (run separately after session close)

```bash
omega-run-action-plan fixtures/action_plans/daily_trace_intake.json
omega-run-action-plan fixtures/action_plans/render_session_audits.json
```

After games are final (same day for afternoon games, next morning for late/overnight
games), run the outcome loop:

```bash
omega-fetch-outcomes
```

This attaches outcomes to today's traces, which are then calibration-eligible at
the next `omega-report-calibration` run.
