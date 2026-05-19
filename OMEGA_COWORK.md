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

Omega Cowork requires Python 3.12+. This is a hard runtime contract, not an
aspirational package metadata hint. At the start of every Cowork VM session,
verify the interpreter and install the repo dependencies before trying MCP or
direct engine imports:

```bash
python --version
python -m pip install -e .[mcp]
python scripts/cowork_preflight.py
```

If `python --version` is below 3.12, stop and switch to a Python 3.12+
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

Use dry-runs when reviewing matches. Outcome backfill sub-agents may run:

- `scripts/fetch_outcomes_nba.py`
- `scripts/fetch_outcomes_mlb.py`
- `scripts/fetch_outcomes_props.py`

Player props and game outcomes stay in separate tables.

## 8. Session Automation

At session start, run calibration health when enough data exists:

```bash
python scripts/report_calibration.py --league NBA --window-days 30
```

Action plans live at `inbox/action_plans/<session_id>.json`. Allowed action types are only `fit_calibration`, `promote_profile`, and `report_calibration`. Dry-run before executing:

```bash
python scripts/run_action_plan.py inbox/action_plans/<session_id>.json --dry-run
python scripts/run_action_plan.py inbox/action_plans/<session_id>.json
```

Session sidecars live at `inbox/sessions/<session_id>.json` and are read directly by reports.

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
