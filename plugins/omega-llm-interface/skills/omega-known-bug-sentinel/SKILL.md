---
name: omega-known-bug-sentinel
description: Run the Omega known-bug sentinel before any formal analysis. Checks live engine state against the bug catalog, reports which bugs are present or fixed, enforces Bet Card suppression for active critical bugs, and writes results to the session sidecar. Use at session start, when switching sports, or whenever a suspected regression occurs.
---

# Omega Known-Bug Sentinel

## When to Invoke

- At session start, after preflight (runs automatically via preflight)
- Before the first formal analyze() call of a new sport/kind
- When a result looks wrong (unexpected edge direction, draw_prob > 0 on MLB)
- When the bug catalog has been recently updated

---

## Invocation

```bash
# Human-readable summary (default)
python scripts/bug_sentinel.py

# Write results to session sidecar as an audit event
python scripts/bug_sentinel.py --session-id sess-YYYYMMDD-XXXX

# Structured JSON
python scripts/bug_sentinel.py --json

# CI mode — exits 1 if any critical bug is present or a regression detected
python scripts/bug_sentinel.py --ci
```

All checks are read-only. No engine writes, no network.

---

## Output Interpretation

| Status | Meaning |
|---|---|
| `fixed` | Bug not present in current code |
| `present` | Bug is active |
| `shadow_mode` | Design-state limiter (not a crash) |
| `unknown` | Manual check required |
| `check_error` | Check itself failed — treat as unknown |

**Gate Summary:** `clear` = formal analysis permitted; `suppressed` = do not emit Bet Cards for that sport+kind.

---

## Enforcement Rules

**If any gate is `suppressed`:**
- Do not emit Bet Cards for that sport+kind
- Qualitative research permitted
- State suppression in `reasoning_downgrade_rationale`

**If `regression_count > 0`:**
- A previously-fixed bug is present again
- Halt formal analysis for affected gates
- Add `bug` audit event with `status: "fail"`

**If any bug is `unknown` (manual):**
- Do not auto-suppress; note in sidecar `agent_notes`
- User must verify manually

---

## Known Bugs Summary (2026-05-28)

| Bug ID | Severity | Status | Suppresses | Workaround |
|---|---|---|---|---|
| BUG-MLB-DEF-RATING-001 | CRITICAL | Fixed | MLB/NHL game | None needed |
| BUG-MLB-DRAW-PROB-001 | CRITICAL | Fixed | MLB game | None needed |
| BUG-INPUT-SNAPSHOT-001 | CRITICAL | Fixed | prop ingest | None needed |
| BUG-EVIDENCE-SHADOW-001 | LOW | Present (design) | None | Note in downgrade_rationale |
| BUG-SQLITE-WAL-FUSE-001 | HIGH | Present | None | cp to /tmp; use --db |
| BUG-REPAIR-FROM-GIT-001 | MEDIUM | Unknown | None | Manual git show workaround |

---

## Catalog Maintenance

Catalog: `omega/qa/bug_catalog.json`

When a bug is fixed: update `status_at_last_audit` to `"fixed"` and `last_audited`.
When a new bug is discovered: add entry with `bug_id`, `title`, `severity`, `check_type`, `check`, `status_at_last_audit`, `last_audited`.

Check types: `grep` (regex on source), `import_test` (behavioral check via analyze()), `db_query` (SQLite probe), `manual` (always unknown).

Set `suppresses_bet_card: true` only for bugs that directly corrupt simulation outputs.

---

## Sidecar Integration

Running with `--session-id` appends a `bug` audit event:

```json
{
  "event_type": "bug",
  "step": "bug_sentinel",
  "status": "ok",
  "notes": "Sentinel ran: No critical/high bugs or regressions.",
  "bugs": []
}
```

`status`: `ok` = all_clear; `warn` = non-critical bugs only; `fail` = critical open or regression.

---

## References

- `omega/qa/bug_catalog.json` — authoritative bug catalog
- `scripts/bug_sentinel.py` — check runner
- `tests/qa/test_bug_sentinel.py` — 25 unit tests
- `omega-session-bootstrap` skill — calls sentinel at Step 6
