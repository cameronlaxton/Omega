# OMEGA - Cowork / Local VM Instructions

**Version:** Phase 6h
**Repo:** `C:\repos\Omega`
**DB:** `omega_traces.db` (SQLite V10 - `traces`, `simulation_distributions`, `bet_records`, `closing_lines`, `outcomes`, `market_snapshots`, `prop_outcomes`)

This is the runtime instruction set for an Omega agent running with local repo access. For product doctrine, canonical source-of-truth rules, and artifact authority, please refer to [PROJECT_STATE.md](PROJECT_STATE.md). The local VM model is the standard model. Use the local MCP server first; use direct repo imports only when MCP is unavailable in the current client.

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
python scripts/cowork_preflight.py --formal-output-gate
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

Only after the formal gate prints `cowork_preflight_ready` may the agent
render formal Omega numeric output from MCP or `analyze()`. A plain
`cowork_preflight_core_ready` state is valid for qualitative research and
debugging only; it is not betting-grade and must not produce Bet Cards.
The direct `scripts/run_analyze.py` CLI enforces this gate before execution.

If preflight reports `Source diverges from git HEAD: <file>`, the mount has
delivered a corrupt copy of a tracked file (Pattern C: a silent truncation
that drops trailing function/class definitions while leaving the file
AST-valid). Restore via:

```bash
python scripts/cowork_preflight.py --repair-from-git
```

`--repair-from-git` runs `git checkout HEAD -- <file>` through bash so the
write propagates to the sandbox mount cache. **Never use the Write tool to
repair source files in the cowork sandbox** — Windows-side writes do not
invalidate the Linux mount cache, so the next read still sees the corrupt
copy (see `docs/session_bugs_20260521_mount_corruption.md`).

#### Repair taint two-pass cycle

After `--repair-from-git` completes successfully, a **repair taint lockfile** is
written to `.omega_cache/.repair_taint`. The gate enforces a mandatory re-verification
pass before formal output is re-authorized:

| Pass | Command | Expected output | Gate state |
|---|---|---|---|
| **1 — Repair** | `--repair-from-git` | Files restored; `[repair] Taint lockfile written` | Taint set — formal output blocked |
| **2 — Verification** | `--formal-output-gate` | `cowork_preflight_taint_cleared_re-run_required` | Taint cleared — still not authorized |
| **3 — Authorization** | `--formal-output-gate` | `cowork_preflight_ready` | Gate open |

**Never skip Pass 3.** `cowork_preflight_taint_cleared_re-run_required` is a soft
failure that removes the lockfile and proves all checks pass on a clean read — but
formal output requires a subsequent clean gate run with no taint present. Skipping
Pass 3 means the agent begins formal output on a sidecar that was assembled while
the source tree was in a repaired-but-unverified state.

If Pass 2 reveals post-repair failures (e.g. a second corrupt file exposed after
the first was restored), fix the new failure before running Pass 3.

### 2b. Clean Cowork Session Hygiene

- **SQLite on a network/FUSE mount:** `TraceStore` detects FUSE/SMB/CIFS/NFS
  DB paths at open time and auto-redirects writes to a per-user local runtime
  path (`%LOCALAPPDATA%\omega\runtime\omega_traces.db` on Windows,
  `~/.omega/runtime/omega_traces.db` on POSIX). No `atexit` sync-back: archival
  back to the mount is owned by `scripts/sync_to_mount.ps1` (one-way, see §2d).
  The redirect is a safety net; the intended steady state is the local-workspace
  layout below.
- **Empty-history guard (no silent fresh DB):** because the sync is one-way, a
  fresh VM's redirected runtime DB would otherwise start EMPTY while the mount
  still holds history — the exact cause of the "0 traces now, 7 traces later"
  split. The store now **fails loud** on redirect rather than presenting
  believable-empty history: if the runtime DB is absent and the source DB is
  missing, malformed, or non-empty, `TraceStore()` raises with an actionable
  message. To populate the runtime DB from a valid source, run the explicit
  `python scripts/db_status.py --seed` (the only command that copies a DB; it
  never overwrites an existing runtime DB and never merges). To intentionally
  start empty, set `OMEGA_ALLOW_EMPTY_DB=1` — this stamps `EMPTY_HISTORY_MODE=true`
  in startup logs/reports so empty history is never mistaken for failed ingest.
- **Always check the effective DB first:** every lifecycle script now logs
  `TraceStore DB: path=… source=… trace_count=… EMPTY_HISTORY_MODE=…` at startup,
  and `python scripts/db_status.py` (read-only) reports the repo/default path,
  the would-be runtime path, existence, integrity_check, trace counts,
  divergence, latest session IDs, and a recommended action — even when the
  redirect guard would refuse to open.
- **Disk artifacts (FUSE/SQLite):** `.fuse_hidden*` files are harmless remnants
  of deleted-but-open files on the FUSE mount — ignore them (gitignored). Ad-hoc
  DB probing leaves `*.probe.db`, `tmp*.probe.db-journal`, `*.db-wal-test`; these
  are harmless and gitignored. A `*.db-journal`/`*.db-wal` that persists next to
  the live DB *after a clean exit* can indicate an open/deleted SQLite handle —
  run `scripts/db_status.py`; if it reports the source `integrity_ok=false`, treat
  the DB as malformed (the store will fail loud on redirect). Do not build a
  cleanup job; delete stray probe files manually if they accumulate.
- **Preflight repair restrictions:** `--repair-from-git` now tiers its repair
  set. Core-tier files (`omega/`, `scripts/`, MCP) are restored from `HEAD`
  even when tracked files under `tests/` have uncommitted edits. The legacy
  guard ("refusing because tracked files outside repair targets are dirty")
  still fires for dirty *core* files. Use `--force-repair` only when you
  intentionally want to clobber every divergent tracked Python file (both
  tiers).
- **Intentional core edits:** In the non-repair path, non-critical core
  divergences (e.g. `store.py`, `scripts/run_action_plan.py`) are emitted as
  `[warning]` lines rather than hard failures. Critical-file divergences
  (`omega/core/contracts/service.py`) remain hard failures. When any tracked
  files diverge but all checks pass, preflight prints
  `cowork_preflight_core_ready` (with `core_dirty=N` and/or
  `test_tier_diverged=N`) instead of `cowork_preflight_failed`. The engine
  is safe to invoke in this state.
- **Stale bytecode:** If preflight warns about host-locked `.pyc` files, add
  `PYTHONDONTWRITEBYTECODE=1` to the Windows host environment. Preflight will
  safely summarize and skip existing locked bytecode files.
- **Odds API usage:** Do not write ad hoc Python scripts that call
  `OddsApiClient._get_json()`. Use `omega_resolve_odds` or
  `scripts/resolve_odds.py` so URL construction, BetMGM defaults, provenance,
  and budget tracking stay on the typed path.

### 2c. Git Command Protocol

On Cowork/FUSE-mounted sessions, do not use `git show HEAD:path` to read blob
contents or file history. Use `git cat-file -p HEAD:path` instead, for example:

```bash
git cat-file -p HEAD:omega/core/contracts/service.py
```

This avoids git commands that can create or collide with background
`.git/index.lock` activity on the Windows-to-Linux mount. For source repair,
prefer `python scripts/cowork_preflight.py --repair-from-git`; use manual
`git cat-file -p ... > /tmp/file && cp /tmp/file target` only when the preflight
repair path explicitly fails.

### 2d. Local Workspace (preferred runtime layout)

Running Omega from the network-mounted repo (`C:\repos\Omega` on the host,
bind-mounted into the Linux sandbox) is the recurring root cause of both
BUG-FUSE-2 (SQLite WAL/DELETE both fail) and BUG-PREFLIGHT-3 (Pattern C
truncation, `index.lock` blocking `git checkout`). The supported steady-state
is to run Omega from a host-local clone and treat the mount as an append-only
archive.

**Bootstrap (Windows host PowerShell, BEFORE launching the Cowork CLI):**

```powershell
.\scripts\cowork_bootstrap.ps1
# Then:
Set-Location $env:OMEGA_LOCAL_WORKSPACE
# Launch the Cowork CLI from here.
```

`cowork_bootstrap.ps1` is idempotent. On first run it clones the upstream
remote (NOT the mount, to avoid pulling Pattern C corruption into the local
copy) into `%USERPROFILE%\.omega\workspace\Omega` and configures a `backup`
remote pointing at the bare repo on the CIFS share. On subsequent runs it
fast-forwards `main` and verifies the `backup` remote.

**Session close (Windows host PowerShell, AFTER the Cowork CLI exits):**

```powershell
.\scripts\sync_to_mount.ps1
# or, to inspect what would happen:
.\scripts\sync_to_mount.ps1 -WhatIf
```

`sync_to_mount.ps1` is one-way (local → mount). It runs
`cowork_preflight.py --direct-only` and `pytest -q --maxfail=3` first; if
either fails, sync is aborted and a log file is written under
`%USERPROFILE%\.omega\workspace\sync_failures\`. On success it:

- pushes `main` and tags to the `backup` bare repo via `git push`,
- mirrors `inbox\` and `reports\` to the mount via `robocopy /E` (append-only,
  NOT `/MIR`; explicit excludes for `*.db`, `*.db-wal`, `*.db-shm`,
  `*.db-journal`, `__pycache__`, `.pytest_cache`, `.venv`, `inbox\failed`),
- snapshots `omega_traces.db` to a timestamped path under
  `<mount>\backups\omega_traces\YYYYMMDD-HHMM.db` (write-once; never
  overwrites the live DB on the mount).

**Execution contract.** Both scripts are **host-side PowerShell**. They are
not callable from inside the Linux sandbox — PowerShell is not present there
and the scripts intentionally manipulate `%USERPROFILE%`, mapped drives, and
the CIFS share. The agent's system prompt must not attempt to invoke either
script. Schedule them via Task Scheduler or run them manually from a host
PowerShell session.

**`OMEGA_TRACE_DB` env var.** For sessions that have not yet migrated to the
local workspace, you can point `TraceStore` at any writable path with
`OMEGA_TRACE_DB=<absolute-path>`. This is the explicit-override path; the
auto-redirect described in §2b is the implicit fallback.

### 2e. Native Linux Container (preferred long-term runtime)

For sustained development or repeated formal sessions, use the native Linux
devcontainer in `.devcontainer/`. It runs the active workspace from the Docker
named volume `omega-linux-workspace`, not from the Windows `C:\repos\Omega` bind
mount, and validates:

- `python scripts/cowork_preflight.py --direct-only`
- `python scripts/bug_sentinel.py --json`
- SQLite `PRAGMA journal_mode=WAL` on a Linux-native filesystem

Use git inside the container to sync source changes back to the remote. Mirror
reports or trace exports to a Windows-visible checkout only after the session
closes; do not write live SQLite files through the FUSE mount.

Preferred path:

```bash
python -m omega.mcp.server
```

MCP analyze tools call `omega.core.contracts.service.analyze()` directly. MCP is an adapter over the canonical core service, not a second betting engine.

If the client cannot expose MCP tools, use the sanctioned direct-engine CLI for
repeatable one-off or batch calls instead of creating scratch Python under
`scripts/`:

```bash
python scripts/run_analyze.py --kind game --request-json request.json --session-id sess-YYYYMMDD-XXXX --bankroll 1000 --trace-out inbox/traces
```

Direct smoke test when no MCP client is available:

```bash
python -m pip install -e .
python scripts/cowork_preflight.py --direct-only
```

```python
import hashlib
import sys

sys.path.insert(0, r"C:\repos\Omega")
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

## 3a. Research Candidate Protocol

When any of the following conditions hold, **all formal Omega output is locked to
`RESEARCH_CANDIDATE` mode** — Bet Cards, edge%, EV%, Kelly, units, confidence
tiers, and trace IDs are prohibited for that league/session:

| Condition | Where to see it |
|---|---|
| No fitted calibration profile | §2 of `reports/latest.md` shows "static fallback" |
| 0 calibration-eligible traces in window | Coverage table: `with_predictions = 0` |
| Invalid or missing session sidecar | sidecar fails JSON parse / schema validation |

> **Bet logging is NOT an output-mode condition.** Output authorization is a
> model-evaluation decision (fitted profile + calibration-eligible coverage +
> valid sidecar). The presence or absence of a `bet_record` does **not** affect
> output mode, grading, or calibration eligibility — it is wager-tracking
> metadata only. The `with_bet` coverage count is informational; never treat
> `with_bet = 0` as a reason to downgrade to `RESEARCH_CANDIDATE`.

`scripts/report_calibration.py` emits an **"Agent Directive — Output Mode"**
section at the very top of `reports/latest.md`. **Read it before beginning any
analysis.** The classification is derived automatically from live DB counts and
the calibration registry — it is not an editorial judgment.

### In RESEARCH_CANDIDATE mode

**Permitted:**
- Qualitative matchup narrative, news synthesis, recent form discussion
- Listed sportsbook lines from a cited public source
- Research-only lean labels (no edge%, EV%, Kelly, confidence tier, or trace_id)
- Stakes up to **1u maximum**

**Forbidden language** (blocked phrases — do not emit in any output):
- "best bet" / "Best Bet"
- "Tier A" / "Tier B"
- "engine-confirmed"
- "actionable bet" / "Actionable Bet"

If a Bet Card would otherwise be warranted, replace it with a
`### Research Candidate` block containing qualitative narrative only,
clearly labeled as uncalibrated.

### Current calibration status (as of 2026-05-28)

| League | Mode | Reason | Pairs needed for fit |
|---|---|---|---|
| NBA | `RESEARCH_CANDIDATE` | Static fallback; 0 calibration-eligible traces | ~30 graded+eligible |
| MLB | `RESEARCH_CANDIDATE` | Static fallback; ~11 graded+eligible pairs accumulated | ~19 more |

These statuses update automatically each time `report_calibration.py` runs.
Regenerate before each session:

```bash
python scripts/report_calibration.py --league NBA
python scripts/report_calibration.py --league MLB
```

## 4. Current Odds Resolution

Use `omega_resolve_odds` or:

```bash
python scripts/resolve_odds.py --kind game --league NBA --home-team "Boston Celtics" --away-team "Indiana Pacers"
```

BetMGM (`betmgm`) is the default sportsbook. Use line-shopping or all-books mode only when the user explicitly asks for line shopping, consensus, market comparison, or an audit. The resolver prepares engine-ready market inputs and provenance; it does not compute protected Omega outputs.

For the full `prop_type` → stat key mapping (including MLB pitching keys and free vs. paid tier notes), see [`prompts/reference/prop_stat_keys.md`](prompts/reference/prop_stat_keys.md).

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

Injury/news protocol: noticing an injury, restriction, or role anomaly is not
enough. Before calling `analyze()`, translate it into structured model inputs:
minutes/usage shifts for player props, explicit `injury_impact` or
`pace_adjustment_factor` in `game_context`, and/or typed `EvidenceSignal`
entries such as `usage_role_change`. If the impact cannot be quantified from
pre-decision sources, append a `quality_gate/null_data_audit` event and
downgrade to research-only rather than emitting a Bet Card.

**Engine output nullability check (immediate post-analyze):**
After each `analyze()` call returns, validate the response payload for NULL,
`0.0`, or undefined values. Build a `null_fields` list if any of these are
missing: `result.model_prob`, `result.edge_pct`, `result.recommended_units`,
`result.confidence_tier`, `game_context.is_playoff`, `game_context.rest_days`,
or any `{prop_type}_mean`/`{stat}_std` on player prop requests. Append a
`quality_gate/null_data_audit` event with `notes="Null fields: " + field_list`
before exporting the trace. Do not include numeric protected values in the event;
only field names. If result-level fields are NULL and engine status is not
"skipped", downgrade to research-only (engine error). If input context fields
are NULL, downgrade to research-only (incomplete pre-analysis work).

Never use raw percentages as basketball `off_rating` or `def_rating`.
Basketball team contexts require possession-adjusted ratings on the usual
points-per-100-possessions scale. Fractional proxies such as FG% are invalid
simulation inputs and are blocked at the service boundary.

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

### 6c. Structured evidence (recommended)

Evidence routing is backend-specific. Handler-based game and player adjustments
follow `AdjustmentPolicy.mode` (`shadow` records only; `live` applies). Markov
game analysis uses transition modifiers instead of handler `off_rating` scaling.
Do not emit the same logical signal on both `plane="game"` and `plane="player"`
in one request; the service suppresses player-plane duplicates when a matching
game-plane signal is present.

Express qualitative reasoning as typed `evidence` signals on the `analyze()` request — not as free text inside `player_context` / `game_context`, which the engine ignores. The `evidence` field is a list of `EvidenceSignal` objects (see `omega/core/contracts/evidence.py`); each carries `signal_type`, `category`, `plane` (`player`/`game`), `value`, `source`, `confidence` (0–1), `window`, and optional `direction`/`stat_key`. The signal taxonomy is multi-sport — `SIGNAL_REGISTRY` declares which sport archetypes each signal type applies to.

The deterministic engine applies known signal types itself. Handler-based evidence is controlled by the versioned `AdjustmentPolicy` (currently `mode=shadow`, which records but does not apply handler factors). Markov game evidence uses backend transition modifiers. Every signal is persisted to the `evidence_signals` table and scored retrospectively by `scripts/score_evidence_signals.py`. Set `confidence` honestly; it is measured against realized outcomes.

At session start, read the "Evidence signal performance" section (§6B) of the calibration report and weight your evidence accordingly: trust signal types/sources marked `predictive`, discount `noise`, treat `insufficient_n` as unproven.

#### Markov backend — approved signal vocabulary

When calling `omega_analyze_game` with `simulation_backend="markov_state"`, **only these 8 signal_type values affect the possession-level transition matrix.** All other signal types are audited and scored but have no Markov effect (silently ignored by the modifier engine). Use the exact string keys below:

| signal_type | effect | direction required? |
|---|---|---|
| `pace_up` | +6% game pace | no |
| `pace_down` | -8% game pace | no |
| `rest_advantage` | +4% scoring rate for rested team | yes (`home`/`away`) |
| `b2b_fatigue` | -6% scoring rate for fatigued team | yes (`home`/`away`) |
| `def_matchup_weak` | +5% offense vs. weak defender | yes (`home`/`away`) |
| `def_matchup_strong` | -5% offense vs. strong defender | yes (`home`/`away`) |
| `usage_role_change` | -7% team rate when key player restricted/elevated | yes (`home`/`away`) |
| `blowout_risk` | -2% momentum acceleration; suppresses variance | no |

Rules:
- Cumulative cap: no single modifier attribute shifts by more than ±15%, regardless of stacked signals.
- Do NOT pre-adjust `home_context`/`away_context` ratings by hand to bake in these effects — the engine applies them from the signal.
- Do NOT emit the same logical signal on both `plane="game"` and `plane="player"` in one request. The service suppresses player-plane duplicates when a matching game-plane signal is present.
- Call `omega_markov_evidence_guide` (MCP prompt) for the full modifier table with scalar values.
- **Evidence routing note:** Markov transition modifiers are the Markov evidence path. Handler-based shadow/live mode applies to fast-score game and player-prop adjustments.

Example with evidence:

```python
analyze({
    "player_name": "Donovan Mitchell", "league": "NBA", "prop_type": "pts",
    "line": 26.5, "odds_over": -110, "odds_under": -110,
    "player_context": {"pts_mean": 28.4, "pts_std": 6.9},
    "home_team": "Cleveland Cavaliers", "away_team": "Detroit Pistons",
    "game_date": "2026-05-22",
    "game_context": {"is_playoff": True, "rest_days": 2},
    "evidence": [
        {"signal_type": "series_avg", "category": "player_form", "plane": "player",
         "value": 30.6, "source": "nba.com", "confidence": 0.9,
         "window": "series", "direction": "over", "stat_key": "pts"},
        {"signal_type": "last_game_outlier", "category": "player_form", "plane": "player",
         "value": True, "source": "agent_reasoning", "confidence": 0.6,
         "window": "last_3", "stat_key": "pts"},
    ],
}, session_id=session_id, bankroll=bankroll)
```

### 6d. Trace completeness (required before filing any export block)

Every export block filed to `inbox/traces/` must include structured reasoning fields alongside the `trace` output. These enable machine-auditable replay, retrospective evidence scoring, and calibration quality tracking.

**Required fields on the export block:**

```json
{
  "trace": { "...analyze() output..." },
  "reasoning_inputs": {
    "sources": ["espn.com", "nba.com"],
    "fields_gathered": ["pts_mean", "pts_std", "is_playoff", "rest_days"],
    "missing_fields": ["sample_size"],
    "market_context": {"book": "draftkings", "odds_over": -110, "odds_under": -110}
  },
  "reasoning_downgrade_rationale": "Skipped bet_card: sample_size unavailable, imputed_fraction > 0.4",
  "reasoning_narrative": "Considered recent form (5.1 pts above season avg last 5 games) and favorable matchup vs weak perimeter defense. Downgraded due to small confirmed sample — 3 of 5 recent games used imputed std.",
  "trace_quality": {
    "aggregate_quality": 0.74
  }
}
```

**Field rules:**

- `reasoning_inputs` — dict of what data was available when you called `analyze()`. At minimum include `sources`, `fields_gathered`, `missing_fields`. Include `market_context` when odds were sourced. Extra keys are allowed.
- `reasoning_downgrade_rationale` — plain-text string explaining any downgrade decision (data gap, imputation, low quality). Set to `null` if no downgrade was applied.
- `reasoning_narrative` — 2–4 sentence summary of what you considered and why. Supplemental to the structured fields.
- `trace_quality.aggregate_quality` — optional orchestrator quality score. Omit if no quality pass ran.

**Evidence signals on the request (required):**

Include at least one `EvidenceSignal` in the `evidence` field of every `omega_analyze_prop` or `omega_analyze_game` call where you evaluated a material factor (player form, matchup strength, situational risk). If no structured evidence is available, use `evidence: []` and set `reasoning_downgrade_rationale` to explain why.

Empty `evidence: []` is tagged `evidence_status: "empty"` in the persisted trace and excluded from retrospective signal scoring — this is visible in calibration reports and creates pressure to supply structured evidence.

**Why this matters:**
- `reasoning_inputs` → enables replay audit (what did the agent know at decision time?)
- `evidence` signals → flow to `evidence_signals` table → `score_evidence_signals.py` scores them retrospectively
- `trace_quality.aggregate_quality` → populates the `aggregate_quality` column in `traces` → required for calibration quality metrics

**What gets warned at ingest:** `scripts/ingest_traces.py` logs a warning for every prop trace missing any identity field (`player_name`, `home_team`, `away_team`, `game_date`, `line`), every trace with empty evidence, and every `reasoning_inputs` block missing its required keys. These are warnings only — the trace is still ingested — but they are surfaced so compliance can be tracked.

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

Action plans live at `inbox/action_plans/<session_id>.json`. Repo-local templates live under `inbox/action_plans/templates/`; see `docs/phase6/automation_playbook.md` for the trace intake, confirmed-bet closing-line, outcome/evidence, weekly shadow-review, and no-op loops.

Allowed action types are command-gated by `scripts/run_action_plan.py`: `ingest_traces`, `fetch_closing_lines`, `fetch_outcomes`, `score_evidence_signals`, `report_calibration`, `fit_calibration`, `fit_adjustment_policy`, and `promote_profile`. `fit_adjustment_policy` is shadow-only in action plans; do not schedule `promote_adjustment_policy --go-live`.

Dry-run before executing:

```bash
python scripts/run_action_plan.py inbox/action_plans/<session_id>.json --dry-run
python scripts/run_action_plan.py inbox/action_plans/<session_id>.json
```

### Session Sidecar Schema (required)

Write `inbox/sessions/<session_id>.json` via `omega.trace.session_sidecar.append_audit_events` (atomic). All top-level keys below are required; use exactly these key names.

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
    "webfetch_failures": 0
  },
  "agent_notes": "Free-text notes on session outcome, data quality issues, or anomalies.",
  "audit_events": [
    {
      "ts": "2026-05-21T18:05:00Z",
      "event_type": "preflight",
      "step": "cowork_preflight",
      "status": "ok",
      "notes": "engine green; bankroll confirmed",
      "trace_ids": []
    }
  ]
}
```

The sidecar has three distinct roles for state, no overlap:

| Field | Owner | Purpose |
|---|---|---|
| `exec_stats` | Engine-emitted counts | Deterministic numeric tallies (`traces_emitted`, `bets_recorded`, `webfetch_failures`) |
| `agent_notes` | LLM | Freeform end-of-session summary |
| `audit_events` | LLM | Structured QA log of observable steps |

**`audit_events` discipline:**
- Each event has `ts`, `event_type` (one of `preflight`, `data_provenance`, `engine_run`, `candidate_rejected`, `downgrade`, `rationale`, `bug`, `command`, `step`, `note`), `step`, and `status` (`ok` | `warn` | `fail` | `skipped`). Optional: `notes`, `inputs`, `outputs`, `assumptions`, `bugs`, `trace_ids`.
- **Never** put engine-owned quant values in `inputs`/`outputs`/`notes`: `edge_pct`, `ev_pct`, `kelly_fraction`, `units`, `confidence_tier`, `fair_price`, `no_vig_price`, `model_probability`, `over_prob`, `under_prob`. Those live in `omega_traces.db`. The writer raises `ProtectedValueError` and the append is rejected atomically — the on-disk file is untouched.
- Do **not** hand-edit the sidecar JSON. Always go through `append_audit_events(...)`. Writes are temp-file + `os.replace`; readers never observe a partial file.

Do not add inline `outcomes` or trace-level grading summaries to the sidecar.
Game outcomes belong in `outcomes`, player-prop outcomes belong in
`prop_outcomes`, and confirmed bet metadata belongs in the per-trace
`bet_record`.

**Retired:** `RUN_AUDIT.md` and `RUN_TRACE.jsonl`. Do not create either. The audit renderer at `omega/trace/audit_renderer.py` (invoked via the `render_audit` action plan step) produces `reports/run_audits/<session_id>.audit.md` from the sidecar + ledger.

`report_calibration.py` joins sidecar data with trace summaries by `session_id`. Validate sidecars before relying on report session sections:

```bash
python scripts/validate_session_sidecars.py
```

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
| `inbox/action_plans/templates/` | Repo-local action-plan templates for scheduler/manual loops |
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
| `scripts/render_session_audits.py` | Renders `reports/run_audits/<session_id>.audit.md` from sidecar + ledger |
| `omega/trace/audit_renderer.py` | Library entry point for the audit renderer |
| `omega/trace/session_sidecar.py` | Sidecar contract + `append_audit_events` atomic writer |
| `omega/trace/_atomic.py` | Atomic text-file write helper used by sidecar and renderer |
| `reports/run_audits/` | Rendered session audit markdown (output of `render_audit`) |

## 10. Human Judgment Required

Surface these to the user instead of automating around them:

- Calibration promotion with manual override.
- Team/player alias table extension.
- API key setup and rotation.
- Stake-unit confirmation for recorded bets.
