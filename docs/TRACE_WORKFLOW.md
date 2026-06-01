# Omega Trace Workflow & Loss Prevention

## Overview

Traces are **runtime artifacts**, not source code. They flow from analysis sessions → export → ingest → database, and should never be committed to git.

## Proper Trace Flow

```
Session (analyze)
    ↓
Trace export → var/inbox/traces/*.json
    ↓
Ingest (python -m omega.ops.ingest_traces)
    ↓
var/omega_traces.db (database)
    ↓
[Traces processed → var/inbox/traces/processed/]
```

## Files & Directories

| Path | Purpose | Committed? |
|------|---------|-----------|
| `var/inbox/traces/*.json` | **Staging area** — exported traces awaiting ingest | ❌ No |
| `var/inbox/traces/processed/` | Archive of successfully ingested traces | ❌ No |
| `var/inbox/traces/failed/` | Archive of traces that failed validation | ❌ No |
| `var/omega_traces.db` | **Source of truth** — all graded traces | ❌ No |
| `inbox/` (git-tracked) | Old/legacy inbox (deprecated) | ✓ Yes |

## What Happened: Trace Loss Root Cause

On 2026-05-30 through 2026-05-31, the system accumulated **61 traces** in `inbox/traces/processed/` because:

1. **Dual inbox directories**: Code exported traces to both `inbox/` and `var/inbox/` inconsistently
2. **Ingest non-recursive**: `ingest_traces.py` only scans root level (`var/inbox/traces/*.json`), **skips subdirectories**
3. **No automation**: No scheduled job to recover stuck traces from `processed/`
4. **Result**: 40 MLB traces + others accumulated, invisible to calibration pipeline

## Prevention: 4-Layer Defense

### 1. Pre-Commit Hook (`.git/hooks/pre-commit`)
Blocks any attempt to commit trace files or databases:
```bash
git add var/inbox/traces/  # ❌ BLOCKED by hook
git add var/omega_traces.db  # ❌ BLOCKED
```

### 2. Audit Script (`scripts/audit_trace_inbox.py`)
Detects and recovers stuck traces:
```bash
# Check status
python scripts/audit_trace_inbox.py --report

# Recover stuck traces
python scripts/audit_trace_inbox.py --recover

# Dry-run to see what would happen
python scripts/audit_trace_inbox.py --dry-run
```

### 3. Automated Ingest (Post-Session)
After each analysis session, run:
```bash
python -m omega.ops.ingest_traces --db var/omega_traces.db --inbox var/inbox/traces
```

This:
- Scans root-level `*.json` files
- Validates each trace
- Persists to database
- Moves processed files to `processed/`

### 4. Periodic Recovery (Cron or Hook)
Add to session startup or CI/CD:
```bash
python scripts/audit_trace_inbox.py --recover
python -m omega.ops.ingest_traces --db var/omega_traces.db --inbox var/inbox/traces
```

## Checklist for Session Closure

Before closing an Omega session:

- [ ] Run `python scripts/audit_trace_inbox.py --report` to check status
- [ ] If `processed/` has traces: `python scripts/audit_trace_inbox.py --recover`
- [ ] Re-run ingest: `python -m omega.ops.ingest_traces`
- [ ] Verify: `python scripts/audit_trace_inbox.py --report` (should show 0 in processed/)

## Monitoring

Check for stuck traces daily:

```bash
# Quick check
python scripts/audit_trace_inbox.py --report | grep "AT RISK\|OK"

# With league breakdown
python scripts/audit_trace_inbox.py --report
```

## Configuration

**Export path:** Ensure all exports go to `var/inbox/traces/` (not `inbox/traces/`)

**Ingest path:** Default is `var/inbox/traces` — use `--inbox` to override only if necessary

**Database path:** Default is `var/omega_traces.db` — use `--db` to override only if necessary

## Troubleshooting

### Traces in `failed/` directory
These are traces that failed validation. Check `.error.txt` files for details:
```bash
ls -1 var/inbox/traces/failed/
head var/inbox/traces/failed/sandbox-*.error.txt
```

### `.json.HASH` files (duplicates)
Sometimes duplicate files with hash suffixes appear during recovery. Safe to delete:
```bash
rm var/inbox/traces/processed/*.*.json  # Remove hash-suffixed duplicates
```

### Database is "empty" but sidecars show traces
Database may be out of sync. Run recovery:
```bash
python scripts/audit_trace_inbox.py --recover
python -m omega.ops.ingest_traces --db var/omega_traces.db --inbox var/inbox/traces
```

## References

- `src/omega/ops/ingest_traces.py` — Full ingest implementation
- `src/omega/trace/export_validator.py` — Validation rules
- `scripts/audit_trace_inbox.py` — Recovery script (this repo)
