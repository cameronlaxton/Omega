# Known Bug Sentinel — Design Plan

**Status:** Planned (not yet implemented)
**Phase:** 6h
**Owner:** Agent Systems / QA

---

## Problem

Five or more engine bugs are open simultaneously. Currently there is no automated check before analysis runs. An agent can call `analyze()` on MLB game lines and produce corrupted Bet Cards without knowing the def_rating inversion is still present. The bugs are documented in memory and OMEGA_COWORK.md but are never checked programmatically at session time.

---

## Goal

A skill (and optionally a script) that runs a fast, read-only diagnostic against the live repo state, compares against the known bug catalog, and emits a structured report: which bugs are confirmed present, which are fixed, and whether formal analysis is gated.

The sentinel **does not fix bugs**. It surfaces status and enforces suppression rules.

---

## Architecture

### Two-layer design

**Layer 1: Script** — `scripts/bug_sentinel.py`

A standalone Python script that performs mechanical, deterministic checks. No LLM involved. Returns structured JSON. Fast (<5s). Called by the skill and optionally by the preflight gate.

**Layer 2: Skill** — `omega-known-bug-sentinel/SKILL.md`

A Cowork skill that tells the agent how to invoke the script, interpret results, and enforce suppression rules. The skill wraps the script output into session audit events and communicates the gate status to the user.

---

## Bug Catalog Schema

Each entry in the catalog:

```json
{
  "bug_id": "BUG-MLB-DEF-RATING-001",
  "title": "MLB/NHL def_rating inverted in _sim_baseball/_sim_hockey",
  "severity": "critical",
  "sport_gates": ["MLB", "NHL"],
  "analysis_kind_gates": ["game"],
  "suppresses_bet_card": true,
  "check_type": "grep",
  "check": {
    "file": "omega/core/simulation/engine.py",
    "pattern": "league_avg_rpg / away_def",
    "line_hint": "648-649",
    "present_means": "bug_present"
  },
  "fix_direction": "Change divisor to multiplicand: home_off * (away_def / league_avg_rpg)",
  "workaround": "Suppress MLB/NHL game-line Bet Cards; qualitative only"
}
```

Catalog stored at: `omega/qa/bug_catalog.json` (versioned, schema-checked).

---

## Check Types

| type | mechanism | speed |
|---|---|---|
| `grep` | `re.search` on file contents | <1ms |
| `ast_check` | parse file AST, inspect specific node | <50ms |
| `import_test` | import module, call check function | <2s |
| `db_query` | read-only SQL against TraceStore | <500ms |
| `manual` | cannot be automated; always reports as `unknown` | N/A |

All checks are read-only. No writes, no network, no analyze() calls.

---

## Bugs to Cover in v1

| Bug ID | Check Type | Signal |
|---|---|---|
| `BUG-MLB-DEF-RATING-001` | grep | `league_avg_rpg / away_def` in engine.py |
| `BUG-MLB-DRAW-PROB-001` | ast_check | `draw_prob` in `_sim_baseball` return dict |
| `BUG-INPUT-SNAPSHOT-001` | import_test | Call `PersistableTrace.from_analyze_output()` with mock; verify identity fields appear in `input_snapshot` |
| `BUG-REPAIR-FROM-GIT-001` | manual | Always `unknown`; suppress auto-repair, note workaround |
| `BUG-SQLITE-WAL-FUSE-001` | db_query | Attempt WAL PRAGMA on the FUSE-mounted DB path; if it raises `OperationalError`, bug confirmed |
| `BUG-EVIDENCE-REGISTRY-001` | import_test | Load AdjustmentPolicy; check `win_streak`, `series_lead`, `starter_era`, `season_record` have no coefficients |

---

## Output Schema

```json
{
  "sentinel_version": "1.0",
  "ran_at": "<ISO-8601>",
  "repo_path": "C:/repos/Omega",
  "results": [
    {
      "bug_id": "BUG-MLB-DEF-RATING-001",
      "status": "present",
      "suppresses_bet_card": true,
      "sport_gates": ["MLB", "NHL"],
      "analysis_kind_gates": ["game"],
      "evidence": "engine.py:648 matched pattern 'league_avg_rpg / away_def'"
    }
  ],
  "gate_summary": {
    "MLB_game": "suppressed",
    "NHL_game": "suppressed",
    "NBA_game": "clear",
    "NBA_prop": "clear",
    "MLB_prop": "clear"
  },
  "open_critical": 2,
  "open_non_critical": 4,
  "all_clear": false
}
```

---

## Skill Behavior (Layer 2)

When the skill is invoked:

1. Run `python scripts/bug_sentinel.py --json` and parse output.
2. If `all_clear: true` → report green, proceed normally.
3. If any `suppresses_bet_card: true` bug is `present`:
   - Emit a formatted warning table listing affected sport/kind gates.
   - Write a `bug` audit event to the session sidecar.
   - Enforce suppression for those gates for the rest of the session.
4. If any bug is `unknown` (manual check):
   - Warn user; do not auto-suppress but note in sidecar.
5. Display the `gate_summary` table so the user sees at a glance what analysis
   is safe to run.

The skill does not attempt to fix bugs. It surfaces status and enforces rules.

---

## Integration Points

### Option A: Manual invocation
Agent invokes the skill when starting a session or before switching sports.
No changes to existing scripts.

### Option B: Preflight integration (recommended)
`cowork_preflight.py` optionally calls `bug_sentinel.py` and includes the
gate_summary in its output. Preflight already runs at session start; this
adds one more structured check without a new entry point.

Add to `cowork_preflight.py`:
```python
if not args.skip_bug_sentinel:
    sentinel = run_bug_sentinel()
    if sentinel["open_critical"] > 0:
        print(f"[bug_sentinel] {sentinel['open_critical']} critical bugs open")
        print(f"[bug_sentinel] gate_summary: {sentinel['gate_summary']}")
```

### Option C: CI guard
Add `bug_sentinel.py --ci` to the test suite. In CI mode, any `present` +
`severity=critical` bug causes a non-zero exit code. This would have caught
the def_rating inversion before it contaminated a session.

**Recommended: implement B and C together.**

---

## Implementation Steps

1. Create `omega/qa/bug_catalog.json` with the 6 v1 entries above.
2. Create `scripts/bug_sentinel.py`:
   - Load catalog from `omega/qa/bug_catalog.json`
   - Dispatch each check by `check_type`
   - Return structured JSON to stdout with `--json` flag
   - Human-readable summary by default
3. Add `--skip-bug-sentinel` flag to `cowork_preflight.py` (off by default).
   Call sentinel from preflight; append gate_summary to preflight output.
4. Write `omega-known-bug-sentinel/SKILL.md` with invocation instructions,
   output interpretation, and suppression rules.
5. Add `test_bug_sentinel.py`:
   - Mock each check type
   - Verify `gate_summary` is correct for each combination of bug states
   - Verify that a critical present bug produces `all_clear: false`
6. Add `bug_sentinel.py --ci` to CI (`.github/workflows` or equivalent).

**Estimated scope:** ~300 lines of Python, 1 JSON catalog, 1 SKILL.md.
No new packages required. No changes to engine contracts.

---

## Risks and Failure Modes

| Risk | Mitigation |
|---|---|
| grep check produces false positive after refactor | Use `ast_check` for the def_rating bug; grep is a fast pre-filter only |
| `import_test` has side effects | Use `unittest.mock.patch` to isolate |
| Catalog goes stale after a bug is fixed | Add `status: "fixed_in"` field with commit hash; sentinel skips confirmed-fixed entries |
| Agent ignores sentinel output | Preflight integration (Option B) makes it unavoidable |
| New bug introduced without catalog entry | CI mode catches regressions only for cataloged bugs; new bugs still require manual discovery |

---

## Files to Create / Modify

| File | Action |
|---|---|
| `omega/qa/bug_catalog.json` | Create (new) |
| `omega/qa/__init__.py` | Create (empty) |
| `scripts/bug_sentinel.py` | Create (new) |
| `scripts/cowork_preflight.py` | Modify — add optional sentinel call |
| `tests/qa/test_bug_sentinel.py` | Create (new) |
| `.claude/skills/omega-known-bug-sentinel/SKILL.md` | Create after script exists |
