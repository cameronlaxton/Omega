---
name: omega-session-bootstrap
description: Run the mandatory Omega session startup sequence — interpreter check, dependency install, preflight gate, SQLite workaround, session ID and sidecar creation, calibration health check. Use at the start of every Omega Cowork session before any analysis or engine invocation.
---

# Omega Session Bootstrap

Run this sequence at the start of every Cowork session. Do not invoke
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
python scripts/cowork_preflight.py
python scripts/cowork_preflight.py --formal-output-gate
```

| Preflight output | Meaning | Next action |
|---|---|---|
| `cowork_preflight_ready` | All gates pass | Proceed to Step 3 |
| `cowork_preflight_core_ready` | Core OK, non-critical divergence | Qualitative/debug only |
| `cowork_preflight_failed` | Hard failure | Fix before proceeding |
| `Source diverges from git HEAD: <file>` | Mount corruption (Pattern C) | See corruption repair below |

### Corruption Repair (Pattern C)

**Do not use `--repair-from-git`** — it is a no-op on FUSE mounts (BUG-REPAIR-FROM-GIT-001).

Manual repair for each corrupt file:
```bash
git show HEAD:"omega/core/contracts/service.py" > /tmp/_service.py \
  && cp /tmp/_service.py omega/core/contracts/service.py
```

**Never use the Write tool to repair source files** — Windows-side writes do not invalidate the Linux mount cache.

---

## Step 3 — SQLite / TraceStore Workaround

The repo DB at `omega_traces.db` is on a FUSE mount. SQLite WAL mode fails (BUG-SQLITE-WAL-FUSE-001).

```bash
cp /sessions/<sandbox>/mnt/Omega/omega_traces.db /tmp/omega_traces.db
export OMEGA_TRACE_DB=/tmp/omega_traces.db
```

Use `--db /tmp/omega_traces.db` on all TraceStore scripts:
```bash
python scripts/report_calibration.py --db /tmp/omega_traces.db --league NBA --window-days 30
python scripts/ingest_traces.py --db /tmp/omega_traces.db --verbose
```

**Warning:** writes to `/tmp/omega_traces.db` are NOT auto-persisted. Sync manually at session close.

---

## Step 4 — Mint Session ID and Open Sidecar

Format: `sess-YYYYMMDD-XXXX` (4-char alphanumeric suffix). Mint once, reuse for all traces.

```python
from omega.trace.session_sidecar import append_audit_events
from datetime import datetime, timezone
from pathlib import Path

session_id = "sess-20260528-a1b2"
opened_at = datetime.now(timezone.utc).isoformat()

append_audit_events(Path(f"inbox/sessions/{session_id}.json"), [{
    "ts": opened_at,
    "event_type": "preflight",
    "step": "cowork_preflight",
    "status": "ok",
    "notes": "engine green; bankroll confirmed at $1000",
    "trace_ids": []
}])
```

Required sidecar fields (all must be present before session close):
`session_id`, `opened_at`, `closed_at`, `model_version`, `purpose`, `bankroll`, `bankroll_confirmed`, `exec_stats`, `agent_notes`, `audit_events`

**Never hand-edit the sidecar JSON.** Always use `append_audit_events`.
Never put engine-owned values in audit event fields — `ProtectedValueError` will reject the append.

---

## Step 5 — Calibration Health (when data exists)

```bash
python scripts/report_calibration.py --db /tmp/omega_traces.db --league NBA --window-days 30
```

Read the "Evidence signal performance" section. Weight evidence: `predictive` → trust; `noise` → discount; `insufficient_n` → treat as unproven.

---

## Step 6 — Bug Sentinel

```bash
python scripts/bug_sentinel.py --session-id sess-YYYYMMDD-XXXX
```

This runs automatically via preflight (unless `--skip-bug-sentinel` is passed). Read the gate_summary. Any gate marked `suppressed` blocks Bet Cards for that sport/kind for the session.

**Active bugs as of 2026-05-28:**

| Bug | Status | Impact |
|---|---|---|
| MLB def_rating inverted | FIXED | None |
| MLB draw_prob leak | FIXED | None |
| input_snapshot identity (props) | FIXED | None |
| Evidence policy shadow mode | Present (design) | Signals recorded, not applied |
| SQLite WAL FUSE failure | Present (HIGH) | Use /tmp workaround |
| --repair-from-git no-op | Unknown (manual) | Use git show workaround |

---

## Step 7 — Confirm Bankroll

Default $1000 unless user specifies otherwise. Record in sidecar `bankroll` and set `bankroll_confirmed: true`.

---

## Session Close Checklist

```bash
python scripts/ingest_traces.py --db /tmp/omega_traces.db --verbose
python scripts/validate_session_sidecars.py
python scripts/render_session_audits.py --session-id <session_id>
python scripts/fetch_closing_lines.py --dry-run   # if bets placed
```

Sync DB back to mount manually (host PowerShell `sync_to_mount.ps1`).

---

## References

- `OMEGA_COWORK.md` — authoritative runtime instructions
- `omega-known-bug-sentinel` skill — bug status and gate enforcement
- `omega-trace-qa` skill — trace completeness checklist
