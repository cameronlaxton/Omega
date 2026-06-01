# Trace Recovery Summary — June 1, 2026

## Incident

Between May 29–31, the calibration pipeline accumulated **61 hidden traces** that were invisible to the database and calibration workflow, including **40 MLB traces** that were urgently needed for the 100-sample fit threshold.

### Root Cause Analysis

1. **Dual inbox directories** — Code exported traces to both `inbox/` and `var/inbox/` inconsistently
2. **Non-recursive ingest** — `ingest_traces.py` only scans root level (`var/inbox/traces/*.json`), skips subdirectories
3. **Manual ingest only** — No scheduled recovery of traces stuck in `processed/` subdirectory
4. **No validation** — No tooling to detect hidden traces before database state became out of sync

### Impact

- **MLB calibration delayed** by ~72 traces (70% of 100-sample threshold lost)
- Run audits showed 219 traces (May 30) and 247 traces (May 31) in DB at generation time, but only 9 remained on June 1
- **Root cause:** Traces were persisted to disk but never moved to database

## Recovery Actions Taken

### 1. Trace Recovery
- Located **61 archived traces** in `inbox/traces/processed/` and `var/inbox/traces/processed/`
- Moved all 61 back to root inbox: `var/inbox/traces/*.json`
- Verified against run_audit ledger (18 MLB from sess-20260530-mlb2, 10 MLB from sess-20260531-mlb1)

### 2. Database Ingest
- Ran full ingest: `python -m omega.ops.ingest_traces`
- **61 traces successfully persisted** (3 with bet records attached)
- **No duplicates** — INSERT OR IGNORE handled idempotency

### 3. Final State
```
MLB:           40 total, 28 eligible (28% of 100-sample threshold)
WNBA:          10 total, 10 eligible
MLS:            4 total,  4 eligible
WORLD_CUP:      4 total,  4 eligible
CHAMPIONS_LEAGUE: 3 total, 3 eligible
─────────────────────────────────────
TOTAL:         61 total, 49 eligible
```

## Prevention: 4-Layer Defense Implemented

### 1. Pre-Commit Hook (`.git/hooks/pre-commit`)
- **Blocks** any attempt to commit trace files or databases to git
- Prevents accidental source-control pollution
- Runs automatically on `git commit`

### 2. Audit Script (`scripts/audit_trace_inbox.py`)
Detects and recovers stuck traces:
```bash
# Report status
python scripts/audit_trace_inbox.py --report

# Recover stuck traces from processed/
python scripts/audit_trace_inbox.py --recover

# Dry-run (no changes)
python scripts/audit_trace_inbox.py --dry-run
```

**Output example:**
```
[OK] No traces stuck in subdirectories
Root inbox traces:       61
  CHAMPIONS_LEAGUE: 3
  MLB: 40
  MLS: 4
  WNBA: 10
  WORLD_CUP: 4
Processed (stuck):       0
```

### 3. Documentation (`docs/TRACE_WORKFLOW.md`)
- Explains proper trace flow: export → ingest → database
- Documents file locations and commit policy
- Provides troubleshooting guide

### 4. Integration Points
- **Post-session cleanup** — Run `audit_trace_inbox.py --recover` after closing session sidecars
- **Pre-ingest validation** — Audit before running `omega-ingest-traces`
- **CI/CD integration** — Add to pre-push hooks or nightly jobs

## Calibration Status After Recovery

| Metric | Value |
|--------|-------|
| MLB eligible | 28 / 100 |
| MLB progress | 28% |
| Samples needed | 72 |
| ETA (20/day) | 4 days |
| ETA (10/day) | 7 days |

**Next step:** Generate 20 MLB analyses/day with `context_source="provided"` to reach 100 eligible by ~June 5.

## Lessons Learned

1. **Dual inbox directories are a liability** — Consider consolidating to single `var/inbox/` path in config
2. **Ingest should be idempotent AND comprehensive** — Maybe add `--recursive` flag to scan subdirectories
3. **Monitoring gap** — No alerting when trace DB count diverges from run_audits
4. **Session closure checklist missing** — Should enforce recovery audit before session_id closes

## Files Modified

- `scripts/audit_trace_inbox.py` — NEW (trace recovery script)
- `.git/hooks/pre-commit` — NEW (prevents trace commits)
- `docs/TRACE_WORKFLOW.md` — NEW (trace workflow documentation)
- `TRACE_RECOVERY_2026_06_01.md` — NEW (this file)

## Going Forward

Before ending any Omega session:

```bash
# 1. Audit for stuck traces
python scripts/audit_trace_inbox.py --report

# 2. Recover if needed
python scripts/audit_trace_inbox.py --recover

# 3. Ingest all traces
python -m omega.ops.ingest_traces --db var/omega_traces.db --inbox var/inbox/traces

# 4. Final check
python scripts/audit_trace_inbox.py --report  # Should show 0 in processed/
```

This prevents trace loss and keeps the calibration pipeline synchronized with reality.
