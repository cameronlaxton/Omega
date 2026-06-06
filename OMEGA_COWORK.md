# OMEGA - Cowork / Local VM Instructions

**Version:** Phase 6h
**Repo:** `C:\repos\Omega`
**DB:** `var/omega_traces.db` (SQLite V10 - `traces`, `simulation_distributions`, `bet_records`, `closing_lines`, `outcomes`, `market_snapshots`, `prop_outcomes`)

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
omega-cowork-preflight
omega-cowork-preflight --formal-output-gate
```

If `python --version` is below 3.10, stop and switch to a Python 3.10+
interpreter. Do not bypass the package install with `sys.path` plus ad hoc
`pip install pydantic numpy`; that hides the wrong interpreter and leaves the
agent to rediscover setup failures during engine execution.

If `omega-cowork-preflight` reports missing `pydantic`, `numpy`, `mcp`, or Omega
package metadata, repair setup with:

```bash
python -m pip install -e .[mcp]
omega-cowork-preflight
```

Only after the formal gate prints `cowork_preflight_ready` may the agent
render formal Omega numeric output from MCP or `analyze()`. A plain
`cowork_preflight_core_ready` state is valid for qualitative research and
debugging only; it is not betting-grade and must not produce Bet Cards.
The direct `src/omega/ops/run_analyze.py` CLI enforces this gate before execution.

If preflight reports `Source diverges from git HEAD: <file>`, the mount has
delivered a corrupt copy of a tracked file (Pattern C: a silent truncation
that drops trailing function/class definitions while leaving the file
AST-valid). Restore via:

```bash
omega-cowork-preflight --repair-from-git
```

`--repair-from-git` runs `git checkout HEAD -- <file>` through bash so the
write propagates to the sandbox mount cache. **Never use the Write tool to
repair source files in the cowork sandbox** â€” Windows-side writes do not
invalidate the Linux mount cache, so the next read still sees the corrupt
copy (see `docs/session_bugs_20260521_mount_corruption.md`).

#### Repair taint two-pass cycle

After `--repair-from-git` completes successfully, a **repair taint lockfile** is
written to `.omega_cache/.repair_taint`. The gate enforces a mandatory re-verification
pass before formal output is re-authorized:

| Pass | Command | Expected output | Gate state |
|---|---|---|---|
| **1 â€” Repair** | `--repair-from-git` | Files restored; `[repair] Taint lockfile written` | Taint set â€” formal output blocked |
| **2 â€” Verification** | `--formal-output-gate` | `cowork_preflight_taint_cleared_re-run_required` | Taint cleared â€” still not authorized |
| **3 â€” Authorization** | `--formal-output-gate` | `cowork_preflight_ready` | Gate open |

**Never skip Pass 3.** `cowork_preflight_taint_cleared_re-run_required` is a soft
failure that removes the lockfile and proves all checks pass on a clean read â€” but
formal output requires a subsequent clean gate run with no taint present. Skipping
Pass 3 means the agent begins formal output on a sidecar that was assembled while
the source tree was in a repaired-but-unverified state.

If Pass 2 reveals post-repair failures (e.g. a second corrupt file exposed after
the first was restored), fix the new failure before running Pass 3.

### 2b. Clean Cowork Session Hygiene

- **SQLite on a network/FUSE mount:** `TraceStore` detects FUSE/SMB/CIFS/NFS
  DB paths at open time and auto-redirects writes to a per-user local runtime
  path (`%LOCALAPPDATA%\omega\runtime\var/omega_traces.db` on Windows,
  `~/.omega/runtime/omega_traces.db` on POSIX). No `atexit` sync-back: archival
  back to the mount is owned by `tools/windows/sync_to_mount.ps1` (one-way, see Â§2d).
  The redirect is a safety net; the intended steady state is the local-workspace
  layout below.
- **Empty-history guard (no silent fresh DB):** because the sync is one-way, a
  fresh VM's redirected runtime DB would otherwise start EMPTY while the mount
  still holds history â€” the exact cause of the "0 traces now, 7 traces later"
  split. The store now **fails loud** on redirect rather than presenting
  believable-empty history: if the runtime DB is absent and the source DB is
  missing, malformed, or non-empty, `TraceStore()` raises with an actionable
  message. To populate the runtime DB from a valid source, run the explicit
  `omega-db-status --seed` (the only command that copies a DB; it
  never overwrites an existing runtime DB and never merges). To intentionally
  start empty, set `OMEGA_ALLOW_EMPTY_DB=1` â€” this stamps `EMPTY_HISTORY_MODE=true`
  in startup logs/reports so empty history is never mistaken for failed ingest.
- **Always check the effective DB first:** every lifecycle script now logs
  `TraceStore DB: path=â€¦ source=â€¦ trace_count=â€¦ EMPTY_HISTORY_MODE=â€¦` at startup,
  and `omega-db-status` (read-only) reports the repo/default path,
  the would-be runtime path, existence, integrity_check, trace counts,
  divergence, latest session IDs, and a recommended action â€” even when the
  redirect guard would refuse to open. For direct inspection, use
  `omega-db-status --query-traces ...` or `omega-db-status --view-ledger ...`;
  plaintext lookup output starts with `TraceStore DB Path: ... [label]`, and JSON
  lookup output carries `trace_store_db_path`, `trace_store_db_source`, and
  `workspace_identity`.
- **Disk artifacts (FUSE/SQLite):** `.fuse_hidden*` files are harmless remnants
  of deleted-but-open files on the FUSE mount â€” ignore them (gitignored). Ad-hoc
  DB probing leaves `*.probe.db`, `tmp*.probe.db-journal`, `*.db-wal-test`; these
  are harmless and gitignored. A `*.db-journal`/`*.db-wal` that persists next to
  the live DB *after a clean exit* can indicate an open/deleted SQLite handle â€”
  run `src/omega/ops/db_status.py`; if it reports the source `integrity_ok=false`, treat
  the DB as malformed (the store will fail loud on redirect). Do not build a
  cleanup job; delete stray probe files manually if they accumulate.
- **Preflight repair restrictions:** `--repair-from-git` now tiers its repair
  set. Core-tier files (`omega/`, `src/omega/ops/`, MCP) are restored from `HEAD`
  even when tracked files under `tests/` have uncommitted edits. The legacy
  guard ("refusing because tracked files outside repair targets are dirty")
  still fires for dirty *core* files. Use `--force-repair` only when you
  intentionally want to clobber every divergent tracked Python file (both
  tiers).
- **Intentional core edits:** In the non-repair path, non-critical core
  divergences (e.g. `store.py`, `src/omega/ops/run_action_plan.py`) are emitted as
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
  `src/omega/ops/resolve_odds.py` so URL construction, BetMGM defaults, provenance,
  and budget tracking stay on the typed path.

### 2c. Git Command Protocol

On Cowork/FUSE-mounted sessions, do not use `git show HEAD:path` to read blob
contents or file history. Use `git cat-file -p HEAD:path` instead, for example:

```bash
git cat-file -p HEAD:omega/core/contracts/service.py
```

This avoids git commands that can create or collide with background
`.git/index.lock` activity on the Windows-to-Linux mount. For source repair,
prefer `omega-cowork-preflight --repair-from-git`; use manual
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
.\tools\windows\cowork_bootstrap.ps1
# Then:
Set-Location $env:OMEGA_LOCAL_WORKSPACE
# Launch the Cowork CLI from here.
```

`cowork_bootstrap.ps1` is idempotent. On first run it clones the upstream
remote (NOT the mount, to avoid pulling Pattern C corruption into the local
copy) into `%USERPROFILE%\.omega\workspace\Omega` and configures a `backup`
remote pointing at the bare repo on the CIFS share. On subsequent runs it
fast-forwards `main` and verifies the `backup` remote. Cowork/sandbox preflight
fails closed when the runtime is marked as Cowork but is not executing from this
local workspace.

**Session close (Windows host PowerShell, AFTER the Cowork CLI exits):**

```powershell
.\tools\windows\sync_to_mount.ps1
# or, to inspect what would happen:
.\tools\windows\sync_to_mount.ps1 -WhatIf
```

`sync_to_mount.ps1` is one-way (local â†’ mount). It runs
`omega-cowork-preflight --direct-only` and `pytest -q --maxfail=3` first; if
either fails, sync is aborted and a log file is written under
`%USERPROFILE%\.omega\workspace\sync_failures\`. On success it:

- pushes `main` and tags to the `backup` bare repo via `git push`,
- mirrors `inbox\` and `reports\` to the mount via `robocopy /E` (append-only,
  NOT `/MIR`; explicit excludes for `*.db`, `*.db-wal`, `*.db-shm`,
  `*.db-journal`, `__pycache__`, `.pytest_cache`, `.venv`, `inbox\failed`),
- snapshots `var/omega_traces.db` to a timestamped path under
  `<mount>\backups\omega_traces\YYYYMMDD-HHMM.db` (write-once; never
  overwrites the live DB on the mount).

**Execution contract.** Both scripts are **host-side PowerShell**. They are
not callable from inside the Linux sandbox â€” PowerShell is not present there
and the scripts intentionally manipulate `%USERPROFILE%`, mapped drives, and
the CIFS share. The agent's system prompt must not attempt to invoke either
script. Schedule them via Task Scheduler or run them manually from a host
PowerShell session.

**`OMEGA_TRACE_DB` env var.** For sessions that have not yet migrated to the
local workspace, you can point `TraceStore` at any writable path with
`OMEGA_TRACE_DB=<absolute-path>`. This is the explicit-override path; the
auto-redirect described in Â§2b is the implicit fallback.

### 2e. Native Linux Container (preferred long-term runtime)

For sustained development or repeated formal sessions, use the native Linux
devcontainer in `.devcontainer/`. It runs the active workspace from the Docker
named volume `omega-linux-workspace`, not from the Windows `C:\repos\Omega` bind
mount, and validates:

- `omega-cowork-preflight --direct-only`
- SQLite `PRAGMA journal_mode=WAL` on a Linux-native filesystem

Use git inside the container to sync source changes back to the remote. Mirror
reports or trace exports to a Windows-visible checkout only after the session
closes; do not write live SQLite files through the FUSE mount.

Preferred path:

```bash
python -m omega.mcp.server
```

MCP analyze tools call `omega.core.contracts.service.analyze()` directly. MCP is an adapter over the canonical core service, not a second betting engine.

### 2f. Batch analysis (N > 3 analyses)

When MCP is available, use `omega_run_batch` for any session producing more than 3 traces.
One tool call handles odds resolution (with prop_type fallback chain), seed derivation,
formal gate enforcement, and export-block writing to `var/inbox/traces/` — no manual
looping, no round-trips, no scratch scripts required.

```python
# Example: 2 props + 1 game in one call
omega_run_batch(
    entries=[
        {"kind": "prop", "league": "MLB", "player_name": "Julio Rodríguez",
         "prop_type": ["hits", "total_bases"],   # fallback chain
         "home_team": "Seattle Mariners", "away_team": "New York Mets",
         "game_date": "2026-06-03", "player_context": {"hits_mean": 1.15, "hits_std": 0.8},
         "game_context": {"is_playoff": False, "rest_days": 1}},
        {"kind": "game", "league": "MLB",
         "home_team": "New York Yankees", "away_team": "Cleveland Guardians",
         "game_date": "2026-06-03",
         "home_context": {"off_rating": 5.15, "def_rating": 3.60, "starter_era": 0.71},
         "away_context": {"off_rating": 4.16, "def_rating": 4.06, "starter_era": 3.07},
         "game_context": {"is_playoff": False, "rest_days": 1}},
    ],
    bankroll=1000.0,
    session_id="sess-20260603-mlb1",
)
```

When MCP is unavailable and a CLI loop is impractical (N > 5), a batch Python script is
an authorized fallback. Every such script **must**:

1. Call `cowork_preflight.run_formal_output_gate()` at the top — abort if gate fails.
2. Derive every seed deterministically as `int.from_bytes(hashlib.sha256(f"{prompt}|{date}".encode("utf-8")).digest()[:4], "big")` — never `random`.
3. Include at least one `EvidenceSignal` per trace, or set `reasoning_downgrade_rationale`
   explaining why evidence is empty.
4. Not hardcode `trace_quality.aggregate_quality`; omit it or compute from input quality.
5. Not mutate shared state as a side effect (e.g., deleting the entire odds cache).

Scripts violating these rules produce uncalibrated traces and may contaminate the
calibration loop with incorrect quality scores.

If the client cannot expose MCP tools, use the sanctioned direct-engine CLI for
repeatable one-off or batch calls instead of creating scratch Python under
`src/omega/ops/`:

```bash
omega-run-analyze --kind game --request-json request.json --session-id sess-YYYYMMDD-XXXX --bankroll 1000 --trace-out var/inbox/traces
```

Direct smoke test when no MCP client is available:

```bash
python -m pip install -e .
omega-cowork-preflight --direct-only
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

For every live `analyze()` call, the agent must generate a deterministic integer seed as `int.from_bytes(hashlib.sha256(f"{prompt}|{date}".encode("utf-8")).digest()[:4], "big")` and pass it in the request. `session_id` and `bankroll` are required runtime inputs. If no bankroll is configured, ask before producing a Bet Card.

## 3. Downgrade Discipline

The deleted lite quality gate is not an automated fallback path. The agent must enforce downgrade discipline before rendering any formal output:

- No Bet Card when critical inputs are missing.
- Downgrade to narrative or research-only when aggregate input quality is below `0.7`.
- Use ultra-low-data text only when fewer than 3 real facts are available and quality is below `0.3`.
- If an engine result has `status: "skipped"` or `status: "error"`, repair missing pre-decision inputs and rerun, or produce qualitative research only.
- Never emit edge, EV, Kelly, units, confidence tiers, or trace IDs unless they came from Python execution.

## 3a. Research Candidate Protocol

All output authorization rules, downgrade discipline, and permitted/forbidden behaviors in `RESEARCH_CANDIDATE` mode are defined in the canonical reference **[`prompts/reference/output_modes.md`](prompts/reference/output_modes.md)**. All agents must read and follow that file.

Concretely, `RESEARCH_CANDIDATE` is an **output-authorization** mode, not an execution mode. If the engine is available:
- **`analyze()` still runs** and persisted traces accumulate to build calibration history.
- **User-facing betting numbers are withheld/downgraded** (no Bet Card, edge%, EV%, Kelly, units, confidence tier, or trace_id is shown in the reply).
- Database trace generation is **never** withheld.

To find the current output mode, check the `output_modes` map in the frontmatter of `var/reports/latest.md` — it carries a mode per market (`output_modes.game`, `output_modes.prop`), written automatically each time `report_calibration.py` runs. Authorize each market off its own mode; the scalar `output_mode` is a conservative fallback only.

The DB trace persists with its `sandbox-` trace_id for calibration â€” see [`output_modes.md`](prompts/reference/output_modes.md). Do not skip trace export just because the user-facing output was downgraded.

## 4. Current Odds Resolution

Use `omega_resolve_odds` or:

```bash
omega-resolve-odds --kind game --league NBA --home-team "Boston Celtics" --away-team "Indiana Pacers"
```

BetMGM (`betmgm`) is the default sportsbook. Use line-shopping or all-books mode only when the user explicitly asks for line shopping, consensus, market comparison, or an audit. The resolver prepares engine-ready market inputs and provenance; it does not compute protected Omega outputs.

For the full `prop_type` â†’ stat key mapping (including MLB pitching keys and free vs. paid tier notes), see [`prompts/reference/prop_stat_keys.md`](prompts/reference/prop_stat_keys.md).

Never print, paste, trace, report, or expose `OMEGA_ODDS_API_KEY`.

### Odds API Budget Management

The `OddsApiBudgetExceeded` error is triggered when local consumption reaches the configured cap. Note that this cap is a **local safeguard** and is distinct from the provider's API quota:
- **Cap location:** Configured via `OMEGA_ODDS_API_MONTHLY_BUDGET` environment variable (defaults to 450, or can be set to e.g. 20000 in `.env`).
- **Counter location:** Consumed budget is tracked inside `omega_odds_api_budget.json` (as a dictionary of `{"YYYY-MM": count}`).
- **Correct fix for quota warnings:** Raise the cap in `.env` (e.g. `OMEGA_ODDS_API_MONTHLY_BUDGET=20000`) and reset the consumption counter by editing `omega_odds_api_budget.json` to `{}` (or `{"2026-05": 0}`). Do NOT set the budget JSON itself to 20000; that is a counter of *used* budget, not the cap, and doing so will immediately lock you out.

## 5. Session IDs

Mint once per conversation and reuse for every trace, bet record, and session sidecar.

Format: `sess-YYYYMMDD-XXXX`

At session start, resume the current-day session ID from workspace memory when present. If the date changed, mint a new one.

## 6. Trace Export

After every analysis, write the trace file to `var/inbox/traces/<trace_id>.json`:

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
| `blowout_risk` | float 0â€“1 | Estimated chance of non-competitive game |
| `opponent_def_rank` | int 1â€“30 | Opponent's defensive ranking |
| `pace_adjustment_factor` | float | Team pace ratio vs league baseline |
| `park_factor` | float | MLB only |
| `weather_wind_mph` | float | MLB/NFL only |
| `is_dome` | bool | NFL only |

Any additional matchup context (scheme advantages, defensive matchup weaknesses, etc.) may be included under any key â€” the engine passes all keys through to `context_labels` in the trace, where the calibration fitter can use them.

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

Express qualitative reasoning as typed `evidence` signals on the `analyze()` request â€” not as free text inside `player_context` / `game_context`, which the engine ignores. The `evidence` field is a list of `EvidenceSignal` objects (see `omega/core/contracts/evidence.py`); each carries `signal_type`, `category`, `plane` (`player`/`game`), `value`, `source`, `confidence` (0â€“1), `window`, and optional `direction`/`stat_key`. The signal taxonomy is multi-sport â€” `SIGNAL_REGISTRY` declares which sport archetypes each signal type applies to.

The deterministic engine applies known signal types itself. Handler-based evidence is controlled by the versioned `AdjustmentPolicy` (currently `mode=shadow`, which records but does not apply handler factors). Markov game evidence uses backend transition modifiers. Every signal is persisted to the `evidence_signals` table and scored retrospectively by `src/omega/ops/score_evidence_signals.py`. Set `confidence` honestly; it is measured against realized outcomes.

At session start, read the "Evidence signal performance" section (Â§6B) of the calibration report and weight your evidence accordingly: trust signal types/sources marked `predictive`, discount `noise`, treat `insufficient_n` as unproven.

#### Markov backend â€” approved signal vocabulary

When using `simulation_backend="markov_state"`, only **8 `signal_type` values** adjust the possession-level transition matrix. All other signal types are audited but have no Markov effect.

The canonical list (exact string keys, transition effects, directions, and the Â±15% cumulative cap) is defined in **[`prompts/reference/markov_evidence_vocab.md`](prompts/reference/markov_evidence_vocab.md)**. Use the exact keys and rules from that file rather than restating them.

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

Every export block filed to `var/inbox/traces/` must include structured reasoning fields alongside the `trace` output. These enable machine-auditable replay, retrospective evidence scoring, and calibration quality tracking.

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
  "reasoning_narrative": "Considered recent form (5.1 pts above season avg last 5 games) and favorable matchup vs weak perimeter defense. Downgraded due to small confirmed sample â€” 3 of 5 recent games used imputed std.",
  "trace_quality": {
    "aggregate_quality": 0.74
  }
}
```

**Field rules:**

- `reasoning_inputs` â€” dict of what data was available when you called `analyze()`. At minimum include `sources`, `fields_gathered`, `missing_fields`. Include `market_context` when odds were sourced. Extra keys are allowed.
- `reasoning_downgrade_rationale` â€” plain-text string explaining any downgrade decision (data gap, imputation, low quality). Set to `null` if no downgrade was applied.
- `reasoning_narrative` â€” 2â€“4 sentence summary of what you considered and why. Supplemental to the structured fields.
- `trace_quality.aggregate_quality` â€” optional orchestrator quality score. Omit if no quality pass ran.

**Evidence signals on the request (required):**

Include at least one `EvidenceSignal` in the `evidence` field of every `omega_analyze_prop` or `omega_analyze_game` call where you evaluated a material factor (player form, matchup strength, situational risk). If no structured evidence is available, use `evidence: []` and set `reasoning_downgrade_rationale` to explain why.

Empty `evidence: []` is tagged `evidence_status: "empty"` in the persisted trace and excluded from retrospective signal scoring â€” this is visible in calibration reports and creates pressure to supply structured evidence.

**Why this matters:**
- `reasoning_inputs` â†’ enables replay audit (what did the agent know at decision time?)
- `evidence` signals â†’ flow to `evidence_signals` table â†’ `score_evidence_signals.py` scores them retrospectively
- `trace_quality.aggregate_quality` â†’ populates the `aggregate_quality` column in `traces` â†’ required for calibration quality metrics

**What gets warned at ingest:** `src/omega/ops/ingest_traces.py` logs a warning for every prop trace missing any identity field (`player_name`, `home_team`, `away_team`, `game_date`, `line`), every trace with empty evidence, and every `reasoning_inputs` block missing its required keys. These are warnings only â€” the trace is still ingested â€” but they are surfaced so compliance can be tracked.

### 6a. Single-trace policy (required)

When the user confirms a bet, the export block **must reuse the original analysis trace's `trace_id` and `input_snapshot`**. Do **not** call `analyze()` a second time to "mint a confirmation trace"; that creates a second `trace_id` with stripped game identity and breaks automated grading (see BUG-2/BUG-4).

Concretely:

- Reuse the same `trace_id` that the analysis stage wrote.
- Carry the same `input_snapshot` (player_name, prop_type, line, **home_team, away_team, game_date** for props) into the bet-confirming export.
- Attach the `bet_record` block to that same export. Never split analysis and confirmation across two trace files.

`src/omega/ops/ingest_traces.py` enforces this: a `bet_record` on a `kind: "prop"` trace missing `home_team`/`away_team`/`game_date` is **rejected** and the file is routed to `var/inbox/traces/failed/` with a `.error.txt` sidecar. Fix the export and re-drop the corrected file rather than working around the validation.

The ingest path also logs a warning if `bet_record.line_taken` differs from `input_snapshot.line` by more than 1.0, or `odds_taken` differs from the matching snapshot odds by more than 25 American points. Drift is allowed (line shopping is legitimate), but the warning is captured for the audit trail.

Ingest with:

```bash
omega-ingest-traces --verbose
```

Do not write to `var/omega_traces.db` directly.

## 7. Closing Lines And Outcomes

Closing lines are captured from the paid Odds API through:

```bash
omega-fetch-closing-lines
```

Use dry-runs when reviewing matches.

**Outcome attachment is required for calibration learning.** The calibration fitter cannot fit profiles without graded traces. Run after game windows close (same day for afternoon games, next morning for late games):

```bash
omega-fetch-outcomes          # all leagues, idempotent
omega-fetch-outcomes --dry-run  # preview only
```

Or per-league:

- `src/omega/ops/fetch_outcomes_nba.py`
- `src/omega/ops/fetch_outcomes_mlb.py`
- `src/omega/ops/fetch_outcomes_props.py`

Player props and game outcomes stay in separate tables. Outcome attachment is idempotent â€” re-running is safe.

## 8. Session Automation

At session start, run calibration health when enough data exists:

```bash
omega-report-calibration --league NBA --window-days 30
```

Action plans live at `var/inbox/action_plans/<session_id>.json`. Repo-local templates live under `fixtures/action_plans/`; see `docs/phase6/automation_playbook.md` for the trace intake, confirmed-bet closing-line, outcome/evidence, weekly shadow-review, and no-op loops.

Allowed action types are command-gated by `src/omega/ops/run_action_plan.py`: `ingest_traces`, `fetch_closing_lines`, `fetch_outcomes`, `settle_bets`, `score_evidence_signals`, `report_calibration`, `fit_calibration`, `fit_adjustment_policy`, and `promote_profile`. `fit_adjustment_policy` is shadow-only in action plans; do not schedule `promote_adjustment_policy --go-live`.

Dry-run before executing:

```bash
omega-run-action-plan var/inbox/action_plans/<session_id>.json --dry-run
omega-run-action-plan var/inbox/action_plans/<session_id>.json
```

### Session Sidecar Schema (required)

Write `var/inbox/sessions/<session_id>.json` via `omega.trace.session_sidecar.append_audit_events` (atomic). All top-level keys below are required; use exactly these key names.

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
- **Never** put engine-owned quant values in `inputs`/`outputs`/`notes`: `edge_pct`, `ev_pct`, `kelly_fraction`, `units`, `confidence_tier`, `fair_price`, `no_vig_price`, `model_probability`, `over_prob`, `under_prob`. Those live in `var/omega_traces.db`. The writer raises `ProtectedValueError` and the append is rejected atomically â€” the on-disk file is untouched.
- Do **not** hand-edit the sidecar JSON. Always go through `append_audit_events(...)`. Writes are temp-file + `os.replace`; readers never observe a partial file.

Do not add inline `outcomes` or trace-level grading summaries to the sidecar.
Game outcomes belong in `outcomes`, player-prop outcomes belong in
`prop_outcomes`, and confirmed bet metadata belongs in the per-trace
`bet_record`.

**Retired:** `RUN_AUDIT.md` and `RUN_TRACE.jsonl`. Do not create either. The audit renderer at `omega/trace/audit_renderer.py` (invoked via the `render_audit` action plan step) produces `var/reports/run_audits/<session_id>.audit.md` from the sidecar + ledger.

`report_calibration.py` joins sidecar data with trace summaries by `session_id`. Validate sidecars before relying on report session sections:

```bash
omega-validate-session-sidecars
```

## 9. VM Directory Map

All paths are relative to the repo root.

| Path | Purpose |
|---|---|
| `omega/core/contracts/service.py` | Canonical `analyze(request, session_id, bankroll) -> trace` entry point |
| `omega/mcp/server.py` | MCP tools over deterministic contracts |
| `var/omega_traces.db` | SQLite V6 - do not write directly |
| `var/inbox/traces/` | Trace export files -> `ingest_traces.py` |
| `var/inbox/sessions/` | Session sidecars |
| `var/inbox/action_plans/` | Action plan JSON -> `run_action_plan.py` |
| `fixtures/action_plans/` | Tracked action-plan templates for scheduler/manual loops |
| `src/omega/ops/ingest_traces.py` | Drains trace exports into trace and bet-record tables |
| `src/omega/ops/run_action_plan.py` | Validates and dispatches action plans |
| `src/omega/ops/report_calibration.py` | Calibration health and session summary report |
| `src/omega/ops/fit_calibration.py` | Fits calibration candidates |
| `src/omega/ops/promote_profile.py` | Promotes a calibration candidate |
| `src/omega/ops/fetch_closing_lines.py` | Captures closing lines through The Odds API |
| `src/omega/ops/fetch_outcomes_all.py` | Attaches outcomes for all leagues (preferred; idempotent) |
| `src/omega/ops/fetch_outcomes_nba.py` | Attaches NBA game outcomes |
| `src/omega/ops/fetch_outcomes_mlb.py` | Attaches MLB game outcomes |
| `src/omega/ops/fetch_outcomes_props.py` | Attaches player prop outcomes |
| `src/omega/ops/backfill_closing_lines.py` | Backfills missed close windows |
| `src/omega/ops/render_session_audits.py` | Renders `var/reports/run_audits/<session_id>.audit.md` from sidecar + ledger |
| `omega/trace/audit_renderer.py` | Library entry point for the audit renderer |
| `omega/trace/session_sidecar.py` | Sidecar contract + `append_audit_events` atomic writer |
| `omega/trace/_atomic.py` | Atomic text-file write helper used by sidecar and renderer |
| `var/reports/run_audits/` | Rendered session audit markdown (output of `render_audit`) |

## 10. Human Judgment Required

Surface these to the user instead of automating around them:

- Calibration promotion with manual override.
- Team/player alias table extension.
- API key setup and rotation.
- Stake-unit confirmation for recorded bets.
