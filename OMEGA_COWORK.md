# OMEGA - Cowork / Local VM Instructions

**Version:** Phase 6h
**Repo:** `C:\Users\camer\OneDrive\Documents\GitHub\Omega`
**DB:** `omega_traces.db` (SQLite V6 - `traces`, `bet_records`, `closing_lines`, `outcomes`, `market_snapshots`, `prop_outcomes`)

This is the runtime instruction set for an Omega agent running with local repo access. The local VM model is the standard model. Use the local MCP server first; use direct repo imports only when MCP is unavailable in the current client.

## 1. Ownership Boundary

The LLM owns intent classification, evidence gathering, source arbitration, input mapping, downgrade decisions, narrative explanation, and local automation.

The deterministic Python engine owns simulation, probability calibration, fair/no-vig price conversion, edge, EV, Kelly fraction, recommended units, confidence tiers, backtesting, grading, and trace ID generation.

The LLM must never generate protected numeric outputs in prose. If the deterministic path cannot run, produce qualitative research only.

## 2. Runtime Preflight And Engine Invocation

Omega Cowork requires Python 3.10+. This is a hard runtime contract, not an
aspirational package metadata hint. At the start of every Cowork VM session,
verify the interpreter and install the repo dependencies before trying MCP or
direct engine imports:

```bash
python --version
python -m pip install -e .[mcp]
python scripts/cowork_preflight.py
```

If `python --version` is below 3.10, stop and switch to a Python 3.10+
interpreter. Do not bypass the package install with `sys.path` plus ad hoc
`pip install pydantic numpy`; that hides the wrong interpreter and leaves the
agent to rediscover setup failures during engine execution.

If `cowork_preflight.py` reports missing `pydantic`, `numpy`, `mcp`, or Omega
package metadata, repair setup with:

```bash
python -m pip install -e .[mcp]
python scripts/cowork_preflight.py
```

Only after the preflight prints `cowork_preflight_ready` may the agent render
formal Omega numeric output from MCP or `analyze()`.

Preferred path:

```bash
python -m omega.mcp.server
```

MCP analyze tools call `omega.core.contracts.service.analyze()` directly. MCP is an adapter over the canonical core service, not a second betting engine.

Direct smoke test when no MCP client is available:

```bash
python -m pip install -e .
python scripts/cowork_preflight.py --direct-only
```

```python
import hashlib
import sys

sys.path.insert(0, r"C:\Users\camer\OneDrive\Documents\GitHub\Omega")
from omega.core.contracts.service import analyze

prompt = "Smoke Test NBA pts prop"
date = "2026-05-18"
seed = int.from_bytes(hashlib.sha256(f"{prompt}|{date}".encode("utf-8")).digest()[:4], "big")

smoke = analyze(
    {
        "player_name": "Smoke Test",
        "league": "NBA",
        "prop_type": "pts",
        "line": 20.0,
        "home_team": "Smoke Home",
        "away_team": "Smoke Away",
        "game_date": date,
        "odds_over": -110,
        "odds_under": -110,
        "player_context": {"pts_mean": 20.0, "pts_std": 5.0},
        "n_iterations": 1000,
        "seed": seed,
    },
    session_id="sess-20260518-smok",
    bankroll=1000.0,
)
assert smoke["trace_id"].startswith("sandbox-")
assert smoke["session_id"] == "sess-20260518-smok"
assert smoke["bankroll"] == 1000.0
print("engine_ready:", smoke["trace_id"])
```

For every live `analyze()` call, the agent must generate a deterministic integer seed from `sha256(prompt + date)` and pass it in the request. `session_id` and `bankroll` are required runtime inputs. If no bankroll is configured, ask before producing a Bet Card.

## 3. Downgrade Discipline

The deleted lite quality gate is not an automated fallback path. The agent must enforce downgrade discipline before rendering any formal output:

- No Bet Card when critical inputs are missing.
- Downgrade to narrative or research-only when aggregate input quality is below `0.7`.
- Use ultra-low-data text only when fewer than 3 real facts are available and quality is below `0.3`.
- If an engine result has `status: "skipped"` or `status: "error"`, repair missing pre-decision inputs and rerun, or produce qualitative research only.
- Never emit edge, EV, Kelly, units, confidence tiers, or trace IDs unless they came from Python execution.

## 4. Current Odds Resolution

Use `omega_resolve_odds` or:

```bash
python scripts/resolve_odds.py --kind game --league NBA --home-team "Boston Celtics" --away-team "Indiana Pacers"
```

BetMGM (`betmgm`) is the default sportsbook. Use line-shopping or all-books mode only when the user explicitly asks for line shopping, consensus, market comparison, or an audit. The resolver prepares engine-ready market inputs and provenance; it does not compute protected Omega outputs.

Never print, paste, trace, report, or expose `OMEGA_ODDS_API_KEY`.

## 5. Session IDs

Mint once per conversation and reuse for every trace, bet record, and session sidecar.

Format: `sess-YYYYMMDD-XXXX`

At session start, resume the current-day session ID from workspace memory when present. If the date changed, mint a new one.

## 6. Trace Export

After every analysis, write the trace file to `inbox/traces/<trace_id>.json`:

```json
{
  "trace": {
    "trace_id": "sandbox-XXXX",
    "session_id": "sess-20260518-a1b2",
    "model_version": "omega-core-phase6h",
    "ran_at": "2026-05-18T18:00:00Z",
    "kind": "game",
    "bankroll": 1000.0,
    "input_snapshot": {},
    "result": {},
    "downgrades": []
  },
  "bet_record": null
}
```

### 6b. game_context is mandatory (required for calibration)

Every `analyze()` call **must** populate `game_context` in the request, for both game-level and prop-level analyses. This is the sole mechanism for calibration slice fitting. Omitting it pins the calibration fitter to the base profile regardless of game type.

Minimum required keys for every analysis:

| Key | Type | Notes |
|-----|------|-------|
| `is_playoff` | bool | Always required. Set `false` for regular season. |
| `rest_days` | int | Days since last game. `0` = back-to-back. |

Additional keys to supply when known:

| Key | Type | Notes |
|-----|------|-------|
| `blowout_risk` | float 0–1 | Estimated chance of non-competitive game |
| `opponent_def_rank` | int 1–30 | Opponent's defensive ranking |
| `pace_adjustment_factor` | float | Team pace ratio vs league baseline |
| `park_factor` | float | MLB only |
| `weather_wind_mph` | float | MLB/NFL only |
| `is_dome` | bool | NFL only |

Any additional matchup context (scheme advantages, defensive matchup weaknesses, etc.) may be included under any key — the engine passes all keys through to `context_labels` in the trace, where the calibration fitter can use them.

Example game request:

```python
analyze({
    "home_team": "Boston Celtics",
    "away_team": "New York Knicks",
    "league": "NBA",
    "home_context": {"off_rating": 119.2, "def_rating": 108.1, "pace": 96.5},
    "away_context": {"off_rating": 115.8, "def_rating": 110.3, "pace": 94.1},
    "odds": {"moneyline_home": -180, "moneyline_away": 150},
    "game_context": {"is_playoff": True, "rest_days": 2},
}, session_id=session_id, bankroll=bankroll)
```

If the user explicitly confirms they took a bet, include `bet_record` with actual book, market, selection, `selection_descriptor`, line, odds, stake units, and decision timestamp. Never fabricate bet metadata. The retired closing-line instruction block must not be emitted.

### 6a. Single-trace policy (required)

When the user confirms a bet, the export block **must reuse the original analysis trace's `trace_id` and `input_snapshot`**. Do **not** call `analyze()` a second time to "mint a confirmation trace"; that creates a second `trace_id` with stripped game identity and breaks automated grading (see [docs/session_bugs_20260519.md](docs/session_bugs_20260519.md), BUG-2/BUG-4).

Concretely:

- Reuse the same `trace_id` that the analysis stage wrote.
- Carry the same `input_snapshot` (player_name, prop_type, line, **home_team, away_team, game_date** for props) into the bet-confirming export.
- Attach the `bet_record` block to that same export. Never split analysis and confirmation across two trace files.

`scripts/ingest_traces.py` enforces this: a `bet_record` on a `kind: "prop"` trace missing `home_team`/`away_team`/`game_date` is **rejected** and the file is routed to `inbox/traces/failed/` with a `.error.txt` sidecar. Fix the export and re-drop the corrected file rather than working around the validation.

The ingest path also logs a warning if `bet_record.line_taken` differs from `input_snapshot.line` by more than 1.0, or `odds_taken` differs from the matching snapshot odds by more than 25 American points. Drift is allowed (line shopping is legitimate), but the warning is captured for the audit trail.

Ingest with:

```bash
python scripts/ingest_traces.py --verbose
```

Do not write to `omega_traces.db` directly.

## 7. Closing Lines And Outcomes

Closing lines are captured from the paid Odds API through:

```bash
python scripts/fetch_closing_lines.py
```

Use dry-runs when reviewing matches.

**Outcome attachment is required for calibration learning.** The calibration fitter cannot fit profiles without graded traces. Run after game windows close (same day for afternoon games, next morning for late games):

```bash
python scripts/fetch_outcomes_all.py          # all leagues, idempotent
python scripts/fetch_outcomes_all.py --dry-run  # preview only
```

Or per-league:

- `scripts/fetch_outcomes_nba.py`
- `scripts/fetch_outcomes_mlb.py`
- `scripts/fetch_outcomes_props.py`

Player props and game outcomes stay in separate tables. Outcome attachment is idempotent — re-running is safe.

## 8. Session Automation

At session start, run calibration health when enough data exists:

```bash
python scripts/report_calibration.py --league NBA --window-days 30
```

Action plans live at `inbox/action_plans/<session_id>.json`. Allowed action types are only `fit_calibration`, `promote_profile`, `report_calibration`, and `fetch_outcomes`. Dry-run before executing:

```bash
python scripts/run_action_plan.py inbox/action_plans/<session_id>.json --dry-run
python scripts/run_action_plan.py inbox/action_plans/<session_id>.json
```

### Session Sidecar Schema (required)

Write `inbox/sessions/<session_id>.json` at session end. All keys below are required; use exactly these key names.

```json
{
  "session_id": "sess-YYYYMMDD-XXXX",
  "opened_at": "2026-05-21T18:00:00Z",
  "closed_at": "2026-05-21T19:15:00Z",
  "model_version": "claude-sonnet-4-6",
  "purpose": "One-line description of session scope",
  "bankroll": 1000.0,
  "bankroll_confirmed": true,
  "exec_stats": {
    "traces_emitted": 0,
    "bets_recorded": 0,
    "webfetch_failures": 0,
    "jit_snapshots_emitted": 0
  },
  "agent_notes": "Free-text notes on session outcome, data quality issues, or anomalies."
}
```

`report_calibration.py` joins sidecar data with trace summaries by `session_id`. Missing or inconsistent keys produce silent gaps in reports.

## 9. VM Directory Map

All paths are relative to the repo root.

| Path | Purpose |
|---|---|
| `omega/core/contracts/service.py` | Canonical `analyze(request, session_id, bankroll) -> trace` entry point |
| `omega/mcp/server.py` | MCP tools over deterministic contracts |
| `omega_traces.db` | SQLite V6 - do not write directly |
| `inbox/traces/` | Trace export files -> `ingest_traces.py` |
| `inbox/sessions/` | Session sidecars |
| `inbox/action_plans/` | Action plan JSON -> `run_action_plan.py` |
| `scripts/ingest_traces.py` | Drains trace exports into trace and bet-record tables |
| `scripts/run_action_plan.py` | Validates and dispatches action plans |
| `scripts/report_calibration.py` | Calibration health and session summary report |
| `scripts/fit_calibration.py` | Fits calibration candidates |
| `scripts/promote_profile.py` | Promotes a calibration candidate |
| `scripts/fetch_closing_lines.py` | Captures closing lines through The Odds API |
| `scripts/fetch_outcomes_all.py` | Attaches outcomes for all leagues (preferred; idempotent) |
| `scripts/fetch_outcomes_nba.py` | Attaches NBA game outcomes |
| `scripts/fetch_outcomes_mlb.py` | Attaches MLB game outcomes |
| `scripts/fetch_outcomes_props.py` | Attaches player prop outcomes |
| `scripts/backfill_closing_lines.py` | Backfills missed close windows |

## 10. Human Judgment Required

Surface these to the user instead of automating around them:

- Calibration promotion with manual override.
- Team/player alias table extension.
- API key setup and rotation.
- Stake-unit confirmation for recorded bets.
