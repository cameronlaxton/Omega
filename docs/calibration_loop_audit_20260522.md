# Omega Calibration Loop Audit — 2026-05-22

**Prepared by:** Cowork agent (claude-sonnet-4-6), sess-20260521-c001  
**For:** Claude Code — design and implementation of fixes  
**Repo:** `C:\repos\Omega`  
**Phase:** 6h  
**Status of calibration loop:** Architecturally complete, broken in practice

---

## 1. Executive Summary

Omega's calibration learning loop — designed to improve probability estimates over time as outcomes accumulate — is not functioning. The pipeline exists from engine through fitter through registry, but four independent failure modes prevent it from producing a single fitted profile. Separately, LLM reasoning quality has no improvement mechanism at all; that is a missing feature requiring new design.

This document covers:
- What the loop is supposed to do (design intent)
- What is actually happening at each stage (grounded in DB and source code)
- Root cause analysis for each failure mode
- A prioritized fix plan with files, risks, and verification steps

---

## 2. Design Intent (What Should Happen)

```
analyze() called
    → engine runs simulation → produces home_win_prob / over_prob / under_prob
    → trace written with predictions + game_context → ingested to DB
    → outcome attached (fetch_outcomes_*.py)
    → fit_calibration.py extracts (prediction, outcome) pairs
    → CalibrationFitter fits isotonic/shrinkage profile
    → profile registered as CANDIDATE
    → human promotes → PRODUCTION
    → next analyze() call uses calibrated probability instead of static prior
    → repeat → ECE decreases over time
```

Context slice design: `game_context.is_playoff` and `game_context.rest_days` feed `_build_context_labels()` → stored in trace → fitter groups traces by slice → separate `playoff` profile fitted alongside base profile → `CalibrationRegistry.get_production(league, context_slice="playoff")` used when `is_playoff=True`.

---

## 3. Current DB State (Evidence)

Queried from `omega_traces.db` on 2026-05-22.

### 3a. Trace distribution by execution mode and result status

| execution_mode | result_status | count | calibration usable? |
|---|---|---|---|
| `sandbox_prop` | `success` | 35 | YES — has `over_prob`/`under_prob` |
| `sandbox_prop` | `null` (pre-status field) | 21 | YES — has `predictions` column populated |
| `sandbox_prop` | `skipped` | 13 | NO — `manual:no_engine_run` |
| `sandbox_parlay` | `skipped` | 21 | NO — parlay traces, no model prob |
| `research` | `null` | 17 | NO — old orchestrator schema |
| `native_sim` | `null` | 10 | NO — old schema, no `result` key |
| `mixed` | `null` | 5 | NO — old schema |
| `narrative` | `null` | 4 | NO — old schema |
| `sandbox_game` | `success` | 4 | YES — but needs `home_context`/`away_context` |
| `sandbox_game` | `null` | 1 | PARTIAL |

Total traces: 133 | Calibration-usable: ~60 | Actually producing calibration pairs: 38 prop, 2 game

### 3b. Calibration fitter output (actual run)

```
python scripts/fit_calibration.py --league NBA --dry-run
→ ERROR: only 2 graded home_win_prob/outcome pairs, minimum 100 required

python scripts/fit_calibration.py --league NBA --plane prop --dry-run
→ ERROR: only 38 graded prop probability/outcome pairs, minimum 100 required
```

No profile has ever been fitted. `omega/core/calibration/profiles.json` does not exist. All analyses run against the static fallback policy.

### 3c. Calibration metrics (static fallback performance)

From `reports/latest.md` generated 2026-05-22:

| Plane | n | Brier | ECE (10-bin) | Log Loss |
|---|---|---|---|---|
| Game | 2 | 0.1175 | 0.3095 | 0.3937 |
| Prop | 38 | 0.3116 | 0.2607 | 0.8556 |

ECE threshold for flag: >0.05. Both planes are flagged. The static prior is materially miscalibrated.

### 3d. Context coverage

```python
# game_context on all graded traces:
sandbox-3d5592d9-70b5 | game_context: {} | context_labels: MISSING
sandbox-4dbb52a4-3494 | game_context: {} | context_labels: MISSING
sandbox-bef88920-cc0e | game_context: {} | context_labels: MISSING
sandbox-7ff33f2a-111c | game_context: {} | context_labels: MISSING
# ... same for all 57 graded traces
```

`game_context` is `{}` on every trace in the DB. `_build_context_labels()` returns `{}` for all. Context-slice profiles cannot be fitted even when volume arrives.

### 3e. Closing lines

Zero closing lines captured. `closing_lines` table is empty. CLV measurement is impossible.

---

## 4. Root Cause Analysis

### FAIL-1: Game plane has 2 calibration pairs (critical)

**What happens:** `analyze_game()` calls `_engine.run_fast_game_simulation()` which requires `home_context` and `away_context` (team offensive/defensive ratings, pace). When these are absent the engine returns `success: False` with `missing_requirements: [...]`, producing a `status: "skipped"` response with no `home_win_prob`. Only 4 game traces have `sandbox_game` execution mode; only 2 produced usable predictions.

**Why context is absent:** The agent never fetches team-level ratings before calling `analyze()`. The odds resolver (`resolve_odds.py`) returns market lines but not team context. There is no automated ESPN/stats integration for team ratings — the agent would need to web-fetch and parse these manually per analysis.

**Effect:** Game calibration is permanently blocked. The 0.31 ECE on the game plane cannot improve until `home_context` / `away_context` are populated.

**Files involved:**
- `omega/core/simulation/engine.py` — `run_fast_game_simulation()` skip path
- `omega/core/contracts/schemas.py` — `GameAnalysisRequest.home_context` / `away_context`
- `scripts/resolve_odds.py` — returns lines, not team context

---

### FAIL-2: `manual:no_engine_run` traces pollute the graded set (high)

**What happens:** 13 prop traces and 21 parlay traces in the DB have `skip_reason: "manual:no_engine_run"`. These were written by the agent constructing trace JSON by hand (old workflow) rather than calling `analyze()`. They get ingested and outcomes get attached, so they appear as "graded" in coverage stats but carry no model probability — the fitter skips them entirely.

**Why this happens:** The pre-6h workflow had the agent compose trace dicts manually and drop them to `inbox/traces/`. The 6h `analyze()` contract deprecates this but old sessions still used it. Additionally, when the agent records a parlay bet (which the engine doesn't model natively), it writes a manual trace with execution_mode `sandbox_parlay`.

**Effect:** Inflates "graded" count (57 shows as healthy; real usable pairs: 38). Misleads calibration health reports.

**Files involved:**
- `scripts/ingest_traces.py` — accepts manual traces without rejecting `manual:no_engine_run`
- `omega/trace/store.py` — `get_graded_traces()` returns these without filtering

---

### FAIL-3: `game_context` never populated (high — blocks context slicing forever)

**What happens:** `OMEGA_COWORK.md §6b` declares `is_playoff` and `rest_days` mandatory on every `analyze()` call. Neither field appears in any trace in the DB. The agent never populates `game_context` before calling `analyze()`.

**Why this happens:** There is no enforcement mechanism. The `PlayerPropRequest` and `GameAnalysisRequest` schemas declare `game_context` as `Optional[dict]`. `analyze()` calls `_build_context_labels()` which silently returns `{}` when `game_context` is absent. Nothing warns or rejects.

**Effect:**
1. `_apply_game_context()` never fires — playoff suppression factors, B2B fatigue, and pace adjustments are never applied to player means. Prop predictions are unconditionally using raw seasonal averages against playoff lines.
2. `context_labels` is `{}` in every trace → fitter has no slice data → `playoff` and `regular` profiles can never be independently fitted.
3. A player like Wembanyama facing playoff defensive intensity at 3 rest days looks identical to the engine as a regular-season B2B.

**Files involved:**
- `omega/core/contracts/schemas.py` — `game_context: Optional[dict] = None` has no validator
- `omega/core/contracts/service.py` — `_build_context_labels()` silently returns `{}`
- `prompts/system_prompt.txt` — agent instruction to populate game_context

---

### FAIL-4: Volume shortfall — prop plane needs 62 more pairs (medium)

**What happens:** `fit_calibration.py` requires `--min-samples 100` (default). The prop plane has 38 usable pairs. At a rate of ~6-10 graded props per session, volume will be sufficient in roughly 8-12 more sessions — but only if FAIL-2 and FAIL-3 are fixed so that new traces are usable.

**Why the rate is low:** Each session analyzes 5-15 props. Outcomes attach the following session. Many props are parlays (no individual trace). At current rate, without fixing the other failures, volume alone will not produce a fittable dataset even by end of Phase 6.

**Files involved:**
- `scripts/fit_calibration.py` — `_DEFAULT_MIN_SAMPLES = 100`

---

### FAIL-5: Closing lines never captured (medium — blocks CLV)

**What happens:** `fetch_closing_lines.py` exists and is documented in `OMEGA_COWORK.md §7`. It has never been run. The `closing_lines` table is empty. CLV (closing line value) — the gold-standard measure of whether the agent is finding genuine edges before the market corrects — is unmeasurable.

**Why:** No scheduled task. No session startup hook. The script requires `OMEGA_ODDS_API_KEY` (paid API) and must run before game time. The window is typically 15 minutes before tipoff.

**Files involved:**
- `scripts/fetch_closing_lines.py`
- `mcp__scheduled-tasks__create_scheduled_task` — could automate pre-game capture

---

### FAIL-6: LLM reasoning has no improvement mechanism (missing feature)

**What happens:** Nothing. The reasoning layer (evidence gathering, source arbitration, matchup framing, downgrade decisions) produces no structured output that is stored and fed back as a learning signal. Session notes exist in sidecars as free text. The Replay plane (designed to audit routing quality) is implemented architecturally but never run.

**What would be needed:**
1. Per-analysis evidence quality scoring (which sources were available, used, trusted)
2. Outcome-retrospective: after grading, did the evidence direction match the result?
3. Persistent evidence source performance tracking (e.g., "injury reports from ESPN beat context from theScore on Wembanyama blk props")
4. Replay plane activation on sampled historical sessions

This is the hardest problem and the longest-horizon fix. It is not a bug — it is genuinely unimplemented.

---

## 5. Fix Plan (Prioritized)

### Priority 1 — Enforce `game_context` at schema level

**Why first:** Fixes FAIL-3 at zero cost. No new infrastructure. Every future trace will carry context labels. Context-slice fitting becomes possible the moment volume arrives.

**What to do:**

`omega/core/contracts/schemas.py` — add a validator to `PlayerPropRequest` and `GameAnalysisRequest`:

```python
from pydantic import model_validator

class PlayerPropRequest(BaseModel):
    # ... existing fields ...
    game_context: dict | None = None

    @model_validator(mode="after")
    def _require_game_context_keys(self) -> "PlayerPropRequest":
        gc = self.game_context or {}
        missing = []
        if "is_playoff" not in gc:
            missing.append("game_context.is_playoff")
        if "rest_days" not in gc:
            missing.append("game_context.rest_days")
        if missing:
            import warnings
            warnings.warn(
                f"analyze() called without required game_context keys: {missing}. "
                "Calibration slice fitting will be degraded.",
                stacklevel=3,
            )
        return self
```

Consider making it a hard error after a deprecation window (2 sessions). Start with a warning so existing callers don't break.

**Also:** Update `prompts/system_prompt.txt` to make the two keys explicit in the pre-flight checklist for every `analyze()` call.

**Verification:** After fix, run preflight smoke test and confirm `context_labels` is non-empty in returned trace. Then run `report_calibration.py` and confirm context_labels are visible.

---

### Priority 2 — Filter unusable traces from calibration queries

**Why second:** Fixes FAIL-2. Prevents inflated "graded" count from masking real calibration progress. Keeps metrics honest.

**What to do:**

`omega/trace/store.py` — `get_graded_traces()` should filter out traces with `manual:no_engine_run` downgrade or `execution_mode IN ('sandbox_parlay', 'research', 'native_sim', 'mixed', 'narrative')`.

```python
def get_graded_traces(self, league: str | None = None, limit: int = 500) -> list[dict]:
    # existing query...
    # Add to WHERE clause:
    #   AND (t.predictions IS NOT NULL)
    #   AND (json_extract(t.full_trace, '$.result.status') != 'skipped'
    #        OR json_extract(t.full_trace, '$.result.status') IS NULL)
    # This ensures old-format traces with predictions but no result.status still pass
```

`scripts/report_calibration.py` — update coverage section to distinguish:
- `traces_total` (all traces)
- `traces_with_predictions` (calibration-eligible)
- `graded_with_predictions` (actually usable pairs)

**Verification:** After fix, `fit_calibration.py --dry-run` error message should show accurate pair count.

---

### Priority 3 — Populate `home_context` / `away_context` for game analyses

**Why third:** Fixes FAIL-1. Required for game-plane calibration to ever function. More effort than priorities 1-2.

**What to do:**

The engine skips when `home_context` and `away_context` are absent because it has no baseline team ratings. Two approaches:

**Option A (recommended): Static league-average fallback in engine**
When `home_context` / `away_context` are `None`, use league-average archetype values (already encoded in `omega/core/simulation/archetypes.py`) instead of returning `missing_requirements`. This allows the engine to produce a calibration-eligible prediction even without team-specific data, at the cost of accuracy. The agent can still supply context when available to improve the estimate.

Files: `omega/core/simulation/engine.py` — `run_fast_game_simulation()` skip path; add fallback to archetype defaults.

**Option B: ESPN team ratings integration**
Add `omega/integrations/espn_teams.py` that fetches current offensive/defensive ratings from ESPN's team stats endpoint. Call before `analyze_game()`. Higher accuracy, adds network dependency and latency.

Option A is lower risk and unblocks calibration immediately. Option B can be layered on top later.

**Verification:** After Option A, a game analyze call with no `home_context` should return `status: "success"` with a valid `home_win_prob`. Confirm in `fit_calibration.py --dry-run` that game-plane pair count increases.

---

### Priority 4 — Schedule closing line capture

**Why:** Fixes FAIL-5. CLV is the primary market-quality signal. Without it the system cannot distinguish luck from skill in bet selection.

**What to do:**

1. Run `fetch_closing_lines.py` manually once to confirm API key works and schema is correct.
2. Create a scheduled task (via `mcp__scheduled-tasks__create_scheduled_task`) to run `fetch_closing_lines.py` daily at game-time windows (e.g., 7:45 PM ET for NBA).
3. Add a `fetch_closing_lines` action type to `run_action_plan.py` so it can be triggered from session action plans.

**Verification:** After one day, `closing_lines` table should have rows. `report_calibration.py` §4 CLV section should be non-empty.

---

### Priority 5 — Fix `ingest_traces.py` to reject `manual:no_engine_run`

**Why:** Prevents future pollution of the calibration dataset from manually-constructed traces.

**What to do:**

`scripts/ingest_traces.py` — add a validation rule: if `downgrades` contains `"manual:no_engine_run"` and `predictions` is null, log a warning and skip ingestion to the `traces` table. Route to `inbox/traces/failed/` with `.error.txt` sidecar explaining that manual traces without predictions cannot feed calibration.

Exception: parlay traces are explicitly unsupported by the engine — they should still be ingested for bet record purposes but flagged as `calibration_ineligible: true` in a new column.

**Verification:** Drop a manually-constructed trace into `inbox/traces/` and confirm it is rejected with a `.error.txt` sidecar.

---

### Priority 6 — LLM reasoning feedback loop (design + implement)

**Why last:** Requires design before implementation. No existing hook to extend. Longest horizon.

**Proposed design:**

**Phase A — Evidence source tracking (low lift)**
Extend the trace schema with an `evidence_sources` array:
```json
{
  "evidence_sources": [
    {"source": "espn_boxscore", "field": "pts_last_5", "value": 24.3, "confidence": 0.9},
    {"source": "injury_report", "field": "status", "value": "probable", "confidence": 0.7},
    {"source": "agent_imputation", "field": "pts_std", "value": 6.1, "confidence": 0.4}
  ]
}
```
After outcome attaches, a retrospective script scores each source: did the direction of its signal match the result? Accumulate source accuracy scores in a new `source_performance` table.

**Phase B — Source weighting in agent prompt**
At session start, `report_calibration.py` (or a new `report_sources.py`) emits a source performance summary. The agent instruction is updated to weight high-performing sources (e.g., ESPN box-score recency) more heavily than low-performing ones (e.g., agent imputation for unknown players).

**Phase C — Replay plane activation**
Run `omega/reasoning/` replay on 5-10% of historical sessions using frozen evidence bundles. Score routing quality, downgrade discipline, and evidence selection. Surface patterns (e.g., "agent consistently over-trusted injury reports for B2B games") as structured feedback.

**New files needed:**
- `omega/trace/evidence.py` — EvidenceSource schema
- `scripts/score_evidence_sources.py` — retrospective scoring
- `omega/strategy/source_performance.py` — accumulates source accuracy
- `scripts/report_sources.py` — feeds source summary into session context

---

## 6. Known Bugs (from `docs/session_bugs_20260522.md`)

These should be fixed alongside or before the above priorities:

| Bug ID | Severity | Description | Fix |
|---|---|---|---|
| BUG-SS-1 | High | Historical session sidecars failed schema validation (migrated 2026-05-22) | ✅ Migrated this session. Add migration step to `validate_session_sidecars.py` for future schema drifts. |
| BUG-DRY-1 | Low | `fetch_outcomes_nba.py --dry-run` reports DRY-attached traces also as unmatched | Fix dry-run unmatched accumulator to exclude already-matched trace_ids |
| BUG-PROP-1 | Low | `fetch_outcomes_props.py` emits duplicate `missing_fields` warnings per bet_record | Deduplicate warnings by `trace_id` before printing |
| BUG-PROP-2 | Low | `first_basket` prop type misclassified as `missing game_date/home/away` | Check `prop_type` against supported list first; log as `unsupported_prop_type` |
| BUG-OUTCOME-1 | Medium | `sandbox-fe2718ac-28d4` (Jalen Duren reb) permanently unresolvable — pre-6b trace lacks game identity | Manual outcome attach required. Prevention: §6b enforcement (Priority 1) |
| BUG-SS-2 (existing) | High | `bet_records` missing `session_id` column — all session-scoped bet queries join through `traces` | `ALTER TABLE bet_records ADD COLUMN session_id TEXT`; backfill via `trace_id` join |
| BUG-TRACE-1 (existing) | High | Double trace minting on bet confirmation — analysis trace and confirmation trace get separate `trace_id`s, breaking grading | Single-trace policy (OMEGA_COWORK.md §6a) must be enforced in `ingest_traces.py` — reject `bet_record` payloads referencing a `trace_id` that already has a `bet_record` |

---

## 7. Quick Wins (can be done in one session)

These are self-contained, low-risk, and unblock downstream work:

1. Add `game_context` warning validator to `PlayerPropRequest` and `GameAnalysisRequest` (Priority 1 — 30 min)
2. Fix `get_graded_traces()` to exclude null-prediction traces (Priority 2 — 20 min)
3. Fix BUG-DRY-1, BUG-PROP-1, BUG-PROP-2 (30 min total)
4. Fix BUG-SS-2 (`bet_records.session_id` migration) (20 min)
5. Update `report_calibration.py` coverage section to show `traces_with_predictions` separately (20 min)

---

## 8. Verification Plan (for the full fix set)

After implementing all priorities, run this sequence:

```bash
# 1. Preflight
python scripts/cowork_preflight.py

# 2. Smoke test — confirm context_labels non-empty
python -c "
from omega.core.contracts.service import analyze
import hashlib
prompt, date = 'Test NBA prop', '2026-05-22'
seed = int.from_bytes(hashlib.sha256(f'{prompt}|{date}'.encode()).digest()[:4], 'big')
result = analyze({
    'player_name': 'Test Player', 'league': 'NBA', 'prop_type': 'pts',
    'line': 20.0, 'home_team': 'Test Home', 'away_team': 'Test Away',
    'game_date': date, 'odds_over': -110, 'odds_under': -110,
    'player_context': {'pts_mean': 20.0, 'pts_std': 5.0},
    'game_context': {'is_playoff': True, 'rest_days': 2},
    'n_iterations': 1000, 'seed': seed,
}, session_id='sess-test', bankroll=1000.0)
assert result['context_labels'] != {}, 'context_labels empty'
assert result['result']['status'] == 'success', 'engine skipped'
print('PASS context_labels:', result['context_labels'])
"

# 3. Outcome fetch
python scripts/fetch_outcomes_all.py --verbose

# 4. Calibration fit attempt (will fail until 100 pairs, but should show correct count)
python scripts/fit_calibration.py --league NBA --plane prop --dry-run

# 5. Calibration report
python scripts/report_calibration.py --league NBA --window-days 30

# 6. Sidecar validation
python scripts/validate_session_sidecars.py
```

---

## 9. Files Changed Summary

| File | Change | Priority |
|---|---|---|
| `omega/core/contracts/schemas.py` | Add `game_context` warning validator | P1 |
| `prompts/system_prompt.txt` | Explicit `game_context` keys in pre-analysis checklist | P1 |
| `omega/trace/store.py` | Filter null-prediction traces from `get_graded_traces()` | P2 |
| `scripts/report_calibration.py` | Add `traces_with_predictions` to coverage section | P2 |
| `omega/core/simulation/engine.py` | Fallback to archetype defaults when `home_context` absent | P3 |
| `scripts/fetch_closing_lines.py` | No code change — schedule via scheduled tasks | P4 |
| `scripts/ingest_traces.py` | Reject `manual:no_engine_run` traces without predictions | P5 |
| `scripts/fetch_outcomes_nba.py` | Fix dry-run unmatched accumulator | BUG-DRY-1 |
| `scripts/fetch_outcomes_props.py` | Deduplicate warnings; fix unsupported prop type classification | BUG-PROP-1/2 |
| DB migration | `ALTER TABLE bet_records ADD COLUMN session_id TEXT` | BUG-SS-2 |
| `omega/trace/evidence.py` (new) | EvidenceSource schema | P6-A |
| `scripts/score_evidence_sources.py` (new) | Retrospective source scoring | P6-A |

---

*Generated: 2026-05-22 | Session: sess-20260521-c001 | Engine: omega-core-phase6h*
