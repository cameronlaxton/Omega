---
name: omega-session-bootstrap
description: Run the mandatory Omega session startup sequence — interpreter check, dependency install, preflight gate, SQLite workaround, session ID and sidecar creation, calibration health check. Use at the start of every Omega local runtime session before any analysis or engine invocation.
---

# Omega Session Bootstrap

Run this sequence at the start of every local runtime session. Do not invoke
`analyze()`, MCP tools, or any trace-writing script until the formal output
gate passes.

---

## Step 1 — Interpreter and Package Install

```bash
python --version
```

**Required: Python 3.10 or higher.** If below 3.10, stop.

```bash
cd C:\repos\Omega
python -m pip install -e .[mcp]
```

---

## Step 2 — Preflight

```bash
omega-cowork-preflight
omega-cowork-preflight --formal-output-gate
```

| Preflight output | Meaning | Next action |
|---|---|---|
| `cowork_preflight_ready` | All gates pass | Proceed to Step 3 |
| `cowork_preflight_core_ready` | Core OK, non-critical divergence | Qualitative/debug only |
| `cowork_preflight_failed` | Hard failure | Fix before proceeding |
| `Source diverges from git HEAD: <file>` | Mount corruption (Pattern C) | See corruption repair below |

### Corruption Repair (Pattern C)

Use `--repair-from-git` when preflight recommends mount-corruption repair. The
live Cowork contract restores syntax-corrupt tracked files through git, writes a
repair-taint lockfile, and requires the follow-up formal-output gate cycle before
formal output is authorized.

```bash
omega-cowork-preflight --repair-from-git
omega-cowork-preflight --formal-output-gate
omega-cowork-preflight --formal-output-gate
```

If `--repair-from-git` fails to change the file or verification still reports the
same corruption, fall back to manual repair for each corrupt file:

```bash
git cat-file -p HEAD:omega/core/contracts/service.py > /tmp/_service.py \
  && cp /tmp/_service.py omega/core/contracts/service.py
```

**Never use the Write tool to repair source files** — Windows-side writes do not invalidate the Linux mount cache.

---

## Step 2.5 — MCP Tool Discovery (do not skip)

**The omega MCP tools are not in the base tool list. They are deferred and must be
loaded before they can be called.** If you look once and see no `omega_*` tools, that
does NOT mean they are unavailable — it means you have not loaded them yet.

- The stdio server (`python -m omega.mcp.server`) **boots slowly on the FUSE/Windows
  mount** (cold Python process importing `numpy`/`pydantic`/`mcp`). At session start it
  may report as **"still connecting"**; its tools appear a few seconds later.
- Load the tools with **`ToolSearch`** before declaring them missing:
  - `ToolSearch` query `"omega"` (keyword) to list the whole omega tool surface, or
  - `ToolSearch` query `"select:omega_trace_query,omega_get_portfolio_summary,..."` to
    load specific tool schemas.
- If a server is shown as "still connecting," **wait and re-run `ToolSearch`** rather than
  concluding the tools were never declared. Do not fall back to hand-rolled SQL/Python on
  the assumption that MCP is unavailable.
- If tools still never appear, the likely cause is a crash-on-boot: confirm
  `python -m pip install -e .[mcp]` succeeded (preflight checks `import mcp.server.fastmcp`).

**Prefer the typed MCP tools over hand-rolled DB access.** They encapsulate the real
schema and return shapes:

- Use `omega_trace_query` instead of raw SQL. The `traces` table has **no `kind` column**
  (`kind` lives inside the `full_trace` JSON blob and in `simulation_distributions`).
- Use `omega_get_portfolio_summary` instead of calling `summarize_ledger` by hand. Its
  result key is **`active_ledgers`**, not `bets`.

---

## Step 3 — SQLite / TraceStore Workaround

Use the repo default DB path for normal local runs. If running in Cowork,
FUSE/SMB/CIFS/NFS, or if TraceStore reports a local-copy redirect, keep all
TraceStore scripts pointed at the same working DB path and sync back through the
repo-owned sync path at session close.

Manual Linux-sandbox fallback:

```bash
cp var/omega_traces.db /tmp/omega_traces.db
export OMEGA_TRACE_DB=/tmp/omega_traces.db
```

Use the configured DB path on all TraceStore scripts. If `OMEGA_TRACE_DB` is not
set, omit `--db` and let `TraceStore` resolve the repo default.

```bash
omega-report-calibration --db "$OMEGA_TRACE_DB" --league NBA --window-days 30
omega-ingest-traces --db "$OMEGA_TRACE_DB" --verbose
```

**Warning:** writes to a redirected/local copy are NOT auto-persisted. Sync with
`tools/windows/sync_to_mount.ps1` when the Cowork contract requires it.

---

## Step 4 — Mint Session ID and Open Sidecar

Format: `sess-YYYYMMDD-XXXX` (4-char alphanumeric suffix). Mint once, reuse for all traces.

```python
from omega.trace.session_sidecar import create_sidecar, bootstrap_payload, append_audit_events
from datetime import datetime, timezone
from pathlib import Path

session_id = "sess-20260528-a1b2"
path = Path(f"var/inbox/sessions/{session_id}.json")

# 1. Atomically CREATE the sidecar (temp + fsync + replace; never a partial/truncated file).
create_sidecar(path, bootstrap_payload(
    session_id,
    model_version="claude-...",
    purpose="NBA game analysis",
    bankroll=1000.0,
    bankroll_confirmed=True,
))

# 2. Append audit events as the session proceeds (atomic; mirrored to <id>.events.jsonl).
append_audit_events(path, [{
    "ts": datetime.now(timezone.utc).isoformat(),
    "event_type": "preflight",
    "step": "cowork_preflight",
    "status": "ok",
    "notes": "engine green; bankroll confirmed at $1000",
    "trace_ids": []
}])
```

Required sidecar fields (all must be present before session close):
`session_id`, `opened_at`, `closed_at`, `model_version`, `purpose`, `bankroll`, `bankroll_confirmed`, `exec_stats`, `agent_notes`, `audit_events`

**Never hand-edit the sidecar JSON.** Use `create_sidecar` to open it and
`append_audit_events` to add events (both atomic). A sibling `<session_id>.events.jsonl`
mirror is written for recovery; if the summary JSON is ever truncated, quarantine it
with `validate_session_sidecars.py --quarantine` and recover events from the mirror.
Never put engine-owned values in audit event fields — `ProtectedValueError` will reject the append.

---

## Step 5 — Calibration Health (when data exists)

```bash
omega-report-calibration --db /tmp/omega_traces.db --league NBA --window-days 30
```

Read the "Evidence signal performance" section. Weight evidence: `predictive` → trust; `noise` → discount; `insufficient_n` → treat as unproven.

---

## Step 6 — Confirm Bankroll

Default $1000 unless user specifies otherwise. Record in sidecar `bankroll` and set `bankroll_confirmed: true`.

---

## Session Close Checklist

```bash
omega-ingest-traces --db "$OMEGA_TRACE_DB" --verbose
omega-validate-session-sidecars
omega-render-session-audits --session-id <session_id>
omega-fetch-closing-lines --dry-run   # if bets placed
```

If `OMEGA_TRACE_DB` is unset, omit `--db`. If using a redirected/local DB copy,
sync back through `tools/windows/sync_to_mount.ps1` per `OMEGA_RUNTIME.md`.

---

## References

- `OMEGA_RUNTIME.md` — authoritative runtime instructions
- `omega-trace-qa` skill — trace completeness checklist
