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

Use `--repair-from-git` when preflight recommends mount-corruption repair. The
live Cowork contract restores syntax-corrupt tracked files through git, writes a
repair-taint lockfile, and requires the follow-up formal-output gate cycle before
formal output is authorized.

```bash
python scripts/cowork_preflight.py --repair-from-git
python scripts/cowork_preflight.py --formal-output-gate
python scripts/cowork_preflight.py --formal-output-gate
```

If `--repair-from-git` fails to change the file or verification still reports the
same corruption, fall back to manual repair for each corrupt file:

```bash
git show HEAD:"omega/core/contracts/service.py" > /tmp/_service.py \
  && cp /tmp/_service.py omega/core/contracts/service.py
```

**Never use the Write tool to repair source files** — Windows-side writes do not invalidate the Linux mount cache.

---

## Step 3 — SQLite / TraceStore Workaround

Use the repo default DB path for normal local runs. If running in Cowork,
FUSE/SMB/CIFS/NFS, or if TraceStore reports a local-copy redirect, keep all
TraceStore scripts pointed at the same working DB path and sync back through the
repo-owned sync path at session close.

Manual Linux-sandbox fallback:

```bash
cp omega_traces.db /tmp/omega_traces.db
export OMEGA_TRACE_DB=/tmp/omega_traces.db
```

Use the configured DB path on all TraceStore scripts. If `OMEGA_TRACE_DB` is not
set, omit `--db` and let `TraceStore` resolve the repo default.

```bash
python scripts/report_calibration.py --db "$OMEGA_TRACE_DB" --league NBA --window-days 30
python scripts/ingest_traces.py --db "$OMEGA_TRACE_DB" --verbose
```

**Warning:** writes to a redirected/local copy are NOT auto-persisted. Sync with
`scripts/sync_to_mount.ps1` when the Cowork contract requires it.

---

## Step 4 — Mint Session ID and Open Sidecar

Format: `sess-YYYYMMDD-XXXX` (4-char alphanumeric suffix). Mint once, reuse for all traces.

```python
from omega.trace.session_sidecar import create_sidecar, bootstrap_payload, append_audit_events
from datetime import datetime, timezone
from pathlib import Path

session_id = "sess-20260528-a1b2"
path = Path(f"inbox/sessions/{session_id}.json")

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
python scripts/report_calibration.py --db /tmp/omega_traces.db --league NBA --window-days 30
```

Read the "Evidence signal performance" section. Weight evidence: `predictive` → trust; `noise` → discount; `insufficient_n` → treat as unproven.

---

## Step 6 — Bug Sentinel

```bash
python scripts/bug_sentinel.py --session-id sess-YYYYMMDD-XXXX
```

This runs automatically via preflight (unless `--skip-bug-sentinel` is passed). Read the gate_summary. Any gate marked `suppressed` blocks Bet Cards for that sport/kind for the session.

**Known-bug snapshot as of 2026-05-28. Live sentinel output is authoritative:**

| Bug | Status | Impact |
|---|---|---|
| MLB def_rating inverted | FIXED | None |
| MLB draw_prob leak | FIXED | None |
| input_snapshot identity (props) | FIXED | None |
| Evidence policy shadow mode | Present (design) | Signals recorded, not applied |
| SQLite WAL FUSE failure | Check live sentinel | Use TraceStore redirect or local DB fallback if present |
| --repair-from-git no-op | Unknown (manual) | Use manual git-show repair only if repair verification fails |

---

## Step 7 — Confirm Bankroll

Default $1000 unless user specifies otherwise. Record in sidecar `bankroll` and set `bankroll_confirmed: true`.

---

## Session Close Checklist

```bash
python scripts/ingest_traces.py --db "$OMEGA_TRACE_DB" --verbose
python scripts/validate_session_sidecars.py
python scripts/render_session_audits.py --session-id <session_id>
python scripts/fetch_closing_lines.py --dry-run   # if bets placed
```

If `OMEGA_TRACE_DB` is unset, omit `--db`. If using a redirected/local DB copy,
sync back through `scripts/sync_to_mount.ps1` per `OMEGA_COWORK.md`.

---

## References

- `OMEGA_COWORK.md` — authoritative runtime instructions
- `omega-known-bug-sentinel` skill — bug status and gate enforcement
- `omega-trace-qa` skill — trace completeness checklist
