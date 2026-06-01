---
name: omega-failure-budget
description: Enforces bounded recovery and stop conditions for Omega sessions. Apply after omega-session-bootstrap completes. Covers daily sessions, shadow runs, ingest/outcome loops, calibration/report runs, and debugging sessions.
---

# Omega Failure Budget

## 1. Precondition

This skill assumes `omega-session-bootstrap` has already run. Do not repeat its setup steps.

Use this skill for stop conditions, failure classification, bounded recovery, and token discipline only.

---

## 2. Pre-run Diagnostic Commands

For daily league sessions:

```bash
omega-db-status
omega-cowork-preflight --formal-output-gate
omega-validate-session-sidecars
```

For ingest/outcome/calibration sessions:

```bash
omega-db-status
omega-ingest-traces --dry-run --explain
omega-validate-session-sidecars
```

For trace export validation (only if `omega-validate-trace-export` exists):

```bash
omega-validate-trace-export var/inbox/traces --strict   # fresh/formal exports
omega-validate-trace-export var/inbox/traces            # legacy/backfill inspection
```

---

## 3. Failure Budget Rules

- Maximum 1 repair attempt per failure category.
- Maximum 2 total setup-recovery attempts per session.
- Do not run the same failing command more than twice unless the second run uses a materially different, documented fix.
- If a schema/API mismatch is discovered, stop immediately and produce the failure report.
- If a sub-agent or prior note claims an API/file exists, verify directly from source before using or editing it.
- **Hard interrupt:** if you are about to attempt a third fix for any single failure, stop now and produce the failure report. Do not count - interrupt.

---

## 4. Stop-and-report Conditions

Stop and produce the failure report if any of the following are true:

- `cowork_preflight_ready` not reached after one repair attempt.
- `db_status.py` cannot identify a valid effective DB path.
- `TraceStore` is unexpectedly in `EMPTY_HISTORY_MODE`.
- Source DB is malformed.
- Runtime DB is empty when non-empty history is expected.
- Required scripts disagree about effective DB path.
- Trace export validation fails and cannot be fixed by re-wrapping.
- Outcome scripts process 0 and dry-run shows no eligible traces in effective DB.
- Tests reveal the prompt's assumptions conflict with repo behavior.
- Agent is about to rerun `analyze()` only to fix export packaging.
- Agent is relying on sub-agent claims not yet verified from source.

---

## 5. Failure Report Format

When stopping, emit exactly this structure:

```text
FAILURE_TYPE:
COMMAND_FAILED:
ROOT_CAUSE:
DB_PATH_USED:
TRACE_COUNT_VISIBLE:
ATTEMPTED_FIXES:
NEXT_REQUIRED_HUMAN_DECISION:
```

---

## 6. Failure Classification Table

| Failure | First response | Stop if |
|---|---|---|
| Missing Python package | `pip install -e .[mcp]` | Still missing after install |
| DB redirect / empty runtime DB | Check `OMEGA_TRACE_DB`, use `--db` flag | Scripts disagree on path |
| Malformed source DB | Inspect with `db_status.py` | Cannot identify valid path |
| Unexpected `EMPTY_HISTORY_MODE` | Check DB path and env vars | Persists after one fix |
| Pattern C / source truncation | `cowork_preflight.py --repair-from-git`; fall back to manual `git show` | Repair no-op (see bootstrap skill) |
| Malformed sidecar | `validate_session_sidecars.py --quarantine`, recover from `.events.jsonl` mirror | Cannot recover events |
| Bad trace export wrapper | Re-wrap only; do NOT rerun `analyze()` | Validation still fails after re-wrap |
| Outcome fetch returning 0 | `ingest_traces.py --dry-run --explain` to check eligibility | Dry-run confirms 0 eligible |
| Test failure from repo contract | Read test to understand intended contract; do not patch to pass | Assumption conflicts with repo |
| Unknown API/file assumption | Verify from source before proceeding | Cannot locate from source |

---

## 7. Token Discipline

- Prefer diagnostic commands over exploratory command chains.
- Prefer `--dry-run --explain` before real writes.
- Prefer validators over rerunning simulations.
- Do not rerun `analyze()` unless model inputs changed.
- Do not repeat web searches for facts already resolved from trusted sources.
- Do not produce long narrative progress updates during setup loops.
- Summarize only after the gate passes or the failure report is produced.

---

## 8. Scope Exclusions

This skill does not authorize:

- Broad observability framework changes
- TraceStore replacement
- JSONL promotion to canonical storage
- Calibration/bet-record recoupling
- CLV/closing_line requirement for ordinary grading
- Full props-domain unification
- Large prompt/doc rewrites unless directly required by a stop condition
