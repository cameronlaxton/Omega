# Trace Integrity Bugs — Audit Report 2026-05-23

**Scope:** Two systemic defects discovered via DB-wide audit of 139 traces.  
**Priority:** Fix before Phase 6i calibration work begins — these corrupt the calibration training set.

---

## Bug 1 — `input_snapshot` Missing Top-Level Request Fields

### What is broken

Every prop trace in the DB should have `input_snapshot` contain the full serialized `PlayerPropRequest` — including `player_name`, `line`, `home_team`, `away_team`, `game_date`. These fields are **required** on the Pydantic model and are critical for: outcome attachment, grading, calibration slice fitting, and automated backtest replay.

In practice, many traces have a stripped `input_snapshot` containing only `player_context` and `game_context`:

```json
"input_snapshot": {
  "player_context": {"reb_mean": 17.5, "reb_std": 5.0},
  "game_context": {"is_playoff": true, "rest_days": 2}
}
```

The missing fields (`player_name`, `home_team`, etc.) are present in `result` instead — a separate output block that is not the canonical input record.

### DB-wide impact

| Field | Present | Missing |
|---|---|---|
| `player_name` | 86 / 139 | 53 traces (38%) |
| `home_team` | 82 / 139 | 57 traces (41%) |
| `game_date` | 78 / 139 | 61 traces (44%) |
| `line` | 64 / 139 | 75 traces (54%) |
| `session_id` | 101 / 139 | 38 traces (27%) — pre-session era |

### Why it went undetected

The ingest validator (`omega-ingest-traces`, function `_enforce_prop_game_identity`) only rejects a trace if it has **both** a missing field **and** an attached `bet_record`. Traces without `bet_record` pass ingest silently regardless of missing fields. This means the majority of affected traces — all those where the user did not confirm a bet — have been accumulating with no warning.

### Root cause (requires investigation)

**The path that should produce correct traces:**

`service.analyze()` → `"input_snapshot": _safe_dump(typed_req)` → `typed_req.model_dump(mode="json")` → full `PlayerPropRequest` dict including all required fields.

**Code:** `omega/core/contracts/service.py` line 223.

This path is correct. `PlayerPropRequest` defines `player_name`, `home_team`, `away_team`, `game_date`, `line` as required fields (`omega/core/contracts/schemas.py` lines 173, 221–225). If `analyze()` is called with a properly constructed request, `input_snapshot` will be complete.

**The actual problem:** The stripped traces were produced at a single timestamp (`2026-05-22T16:23:55Z` for all 6 May 22 traces), suggesting they were either:
1. Constructed by the agent manually rather than via `analyze()`, or
2. Passed through a MCP layer that stripped the request before calling `analyze()`, or
3. Called with a raw dict missing top-level fields (only `player_context`/`game_context` supplied), which would cause `PlayerPropRequest` validation to fail — meaning `analyze()` was never actually called and the trace was written without engine output.

**Investigate:** Check `omega/mcp/server.py` — does the MCP tool accept partial dicts and construct traces without calling `analyze()`? Also check whether any session agent code builds trace JSON files directly without going through the engine.

### Fix

**Step 1 — Ingest validation (fast, defensive):**

In `omega-ingest-traces`, `_enforce_prop_game_identity()` — extend to warn (not just reject) for prop traces **without** a bet_record that are also missing identity fields. This surfaces the gap on every ingest, not just when bets are attached.

**Step 2 — Engine correctness:**

Confirm that every `analyze()` call site passes a fully populated request dict with all `PlayerPropRequest` top-level fields. The agent should never omit `player_name`, `home_team`, `away_team`, `game_date`, `line` from the request.

If the MCP layer is constructing partial traces, fix it to always call `analyze()` with the complete request.

**Step 3 — Historical backfill:**

Run a one-time backfill to copy identity fields from `full_trace.result` → `full_trace.input_snapshot` for the 61–75 affected traces. Update the `traces.full_trace` JSON in place (use a migration script, not raw SQL on individual rows). Add a `schema_version` bump.

**Test:** After fix, assert that `input_snapshot.player_name == result.player` and `input_snapshot.home_team == result.home_team` for all prop traces.

---

## Bug 2 — LLM Reasoning Not Captured in Traces

### What is broken

The trace schema has fields designed to capture the reasoning chain: `context_labels`, `evidence_application`, `calibration_audit`, `predictions`, `recommendations`, `odds_snapshot`, `aggregate_quality`, `simulation_seed`. Most are empty or null across the full trace history.

### DB-wide impact

| Field | Populated | Coverage |
|---|---|---|
| `simulation_seed` | 83 / 139 | 60% |
| `odds_snapshot` | 55 / 139 | 40% |
| `aggregate_quality` | 64 / 139 | 46% |
| `predictions` | 46 / 139 | 33% |
| `recommendations` | 40 / 139 | 29% |
| `context_labels` | 0 / 139 | 0% (5 have `{}` empty dict) |
| `evidence_application` | 0 / 139 | 0% |
| `calibration_audit` | 0 / 139 | 0% |
| `prompt` | 139 / 139 | 100% — but contains only a brief label, not reasoning |

Sample `prompt` values actually stored:
```
"prop: Oklahoma City Thunder @ San Antonio Spurs"
"Who wins?"
"NBA game: Lakers vs Celtics"
```

No LLM reasoning, evidence evaluation, narrative, or downgrade rationale is preserved anywhere in the trace.

### Consequence

Traces cannot support:
- Replay audit (what evidence was used to reach this recommendation?)
- Evidence signal scoring (0/139 `evidence_application` → `score_evidence_signals.py` has nothing to score)
- Calibration slice fitting via `context_labels` (0/139 populated → fitter always falls back to base profile)
- Quality gate introspection (`aggregate_quality` null on 46%)

### Root cause — two separate gaps

**Gap A: `quality_gate` not returned by `service.analyze()`**

`service.py` returns this dict (lines ~217–229):
```python
return {
    "trace_id": ...,
    "input_snapshot": _safe_dump(typed_req),
    "result": _safe_dump(result),
    "context_labels": context_labels,
    "evidence_mode": evidence_mode,
    "evidence_application": evidence_application,
    # NOTE: quality_gate is NOT in this return dict
}
```

`persistable.py` `from_analyze_output()` reads `aggregate_quality` from `analyze_out.get("quality_gate")` (line ~67). Since `quality_gate` is absent from the `analyze()` return, `aggregate_quality` is always `None` and `downgrades` may also be incomplete.

**Fix:** Add `"quality_gate"` to the `service.analyze()` return dict. Construct it from the gating logic already present in the function. Minimum shape:
```python
"quality_gate": {
    "aggregate_quality": <float 0-1>,
    "downgrades": [...],
    "passed": <bool>,
}
```

**Gap B: LLM reasoning has no serialization hook**

The agent's orchestrator runs before `analyze()` is called. It gathers evidence, evaluates source quality, writes narrative, and decides downgrades — but none of this is passed into `analyze()` or written to the trace file. The trace schema has an `evidence_application` field that expects this data, but the orchestrator never populates it.

This is an architectural gap, not a simple field addition. The fix requires the orchestrator to:
1. Collect structured `EvidenceSignal` objects during reasoning (already defined in `omega/core/contracts/evidence.py`)
2. Pass them into the `analyze()` request via the `evidence` field on `PlayerPropRequest`
3. Optionally serialize a `reasoning_narrative` string to a field not yet in the schema

The `evidence_application` field will then be populated automatically by the engine (it already has handler logic in `omega/core/simulation/evidence_handlers.py`). The calibration audit populates automatically once calibration is called with `apply_calibration_audited()` — which is already wired in `service.py` but its output is being lost (see Gap A).

**Fix sequence:**
1. Add `quality_gate` to `service.analyze()` return — unblocks `aggregate_quality` and `calibration_audit` immediately
2. Add a `reasoning_narrative: str | None` field to `PersistableTrace` in `omega/trace/persistable.py` — gives the agent a place to write its reasoning
3. Update the agent orchestrator to populate `evidence` signals on the request and write `reasoning_narrative` to the export block before filing the trace
4. Confirm `evidence_application` flows through to `evidence_signals` table (TraceStore `persist()` is supposed to explode it — verify it does)

---

## Files to Change

| File | Change |
|---|---|
| `omega/core/contracts/service.py` | Add `quality_gate` dict to `analyze()` return |
| `omega/trace/persistable.py` | Add `reasoning_narrative: str \| None` field; verify `from_analyze_output` handles it |
| `omega/trace/store.py` | Verify `evidence_application` is exploded into `evidence_signals` table on persist |
| `omega-ingest-traces` | Extend `_enforce_prop_game_identity()` to warn on identity-missing traces without bet_records |
| `omega/mcp/server.py` | Audit whether MCP layer bypasses `analyze()` or strips request fields |
| `omega-backfill-input-snapshot` | **New script** — one-time fix for the 61–75 historically broken traces |

---

## Verification Plan

```bash
# After fix: all prop traces should have identity in input_snapshot
python - << 'EOF'
import sqlite3, json
conn = sqlite3.connect('var/omega_traces.db')
cur = conn.cursor()
cur.execute("""
  SELECT COUNT(*) FROM traces
  WHERE kind = 'prop'
  AND (json_extract(full_trace, '$.input_snapshot.player_name') IS NULL
    OR json_extract(full_trace, '$.input_snapshot.home_team') IS NULL
    OR json_extract(full_trace, '$.input_snapshot.game_date') IS NULL)
""")
broken = cur.fetchone()[0]
assert broken == 0, f"{broken} prop traces still missing identity in input_snapshot"
print("PASS: all prop traces have game identity in input_snapshot")
EOF

# After fix: quality_gate should be populated
python - << 'EOF'
import sqlite3
conn = sqlite3.connect('var/omega_traces.db')
cur = conn.cursor()
cur.execute("""
  SELECT COUNT(*) FROM traces
  WHERE json_extract(full_trace, '$.aggregate_quality') IS NULL
""")
null_q = cur.fetchone()[0]
print(f"Traces with null aggregate_quality: {null_q} (target: 0 for new traces)")
EOF

# Smoke test: run analyze() and verify output has quality_gate
omega-cowork-preflight
```

---

## Session Notes

- **Workaround applied 2026-05-23:** `prepare_inbox_traces.py` script copied `result` → `input_snapshot` for 6 May 22 inbox traces before ingest. This is a session hotfix only; the upstream engine is still broken.
- **Reference:** BUG-2 and BUG-4 in `docs/session_bugs_20260519.md` document the same `input_snapshot` gap from May 19. The fix was not applied then. This is at minimum the third occurrence.
