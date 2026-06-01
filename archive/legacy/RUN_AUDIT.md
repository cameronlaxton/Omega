# RUN_AUDIT.md — Daily Outcome Evidence Loop

---

## Run: 2026-05-29T09:07:12Z

**Session:** `daily-outcome-evidence-loop-20260529`
**Plan:** `daily_outcome_evidence_loop`
**Triggered by:** scheduled task
**Final status:** `ok_with_warnings`

---

### Pipeline Steps

| # | Action | Status | Duration | Notes |
|---|--------|--------|----------|-------|
| 0 | env_bootstrap | ⚠️ recovered | ~12s overhead | pydantic missing; recovered via pip install |
| 1 | ingest_traces | ✅ ok | <1s | 0 new traces |
| 2 | fetch_outcomes | ✅ ok | ~1s | 0 outcomes attached (nba/mlb/props) |
| 3 | score_evidence_signals | ✅ ok | <1s | 0 scoreable signals |
| 4 | report_calibration | ✅ ok | <1s | 0 traces/sessions/candidates; 3 sidecars skipped |

---

### Count Summary

| Metric | Count |
|--------|-------|
| Traces ingested | 0 |
| Closing lines captured | 0 |
| Outcomes attached (NBA) | 0 |
| Outcomes attached (MLB) | 0 |
| Outcomes attached (props) | 0 |
| Signal rows processed | 0 |
| Calibration-eligible traces | 0 |
| Session sidecars skipped (malformed) | 3 |
| Env failures recovered | 1 |

---

### Data Provenance

**Outcome source — NBA/MLB:** ESPN scoreboard API for 2026-05-28.
**Outcome source — props:** Internal props outcome script; no external fetch required.
**TraceStore DB:** Redirected from `var/omega_traces.db` (FUSE mount) → `/sessions/affectionate-peaceful-maxwell/.omega/runtime/omega_traces.db`. Reads and writes against the runtime DB, not the repo-local file. This is expected behavior per OMEGA_COWORK.md §2c.
**Calibration report output:** `var/reports/latest.md` — 0 traces, static fallback profile, NBA 30-day window.

---

### Degradation Decisions

1. **pydantic not installed (first attempt):** All 4 pipeline steps failed with `ModuleNotFoundError: No module named 'pydantic'`. Remediation: installed pydantic v2.13.4 via pip. Second run succeeded. No data lost; pipeline fully recovered.

2. **0 traces in DB:** fetch_outcomes and score_evidence_signals both short-circuit cleanly when no open traces exist. This is correct behavior — not a bug. No outcomes can be attached without matching trace_ids in the DB.

3. **3 malformed session sidecars skipped:** `report_calibration` logged warnings and skipped all three. The sidecar files are truncated (write was interrupted mid-JSON-array). Content that was written before truncation is readable and informative (see below), but the files cannot be parsed as valid JSON.

---

### Validation Outcomes

- ingest_traces: PASS — correctly reports 0 new files; no silent failures.
- fetch_outcomes nba: PASS — ESPN API reachable; 0 attachments expected given empty DB.
- fetch_outcomes mlb: PASS — ESPN API reachable; 0 attachments expected.
- fetch_outcomes props: PASS — 0 attachments, no unsupported/missing_fields errors.
- score_evidence_signals: PASS — correctly exits with no-op when 0 graded traces.
- report_calibration: PASS (with warnings) — wrote valid report; correctly skipped 3 malformed sidecars with logged warnings rather than crashing.

---

### Malformed Sidecar Detail

All three files were produced during the 2026-05-28 session and share the same failure mode: the JSON array is truncated at the end of a `null_data_audit` step entry. The closing `]` and `}` of the steps array and session object are missing. Cause is likely an interrupted write (session ended or process killed before flush).

**`sess-20260528-mlb1.json`** (127 lines, 5757 chars)
- Truncated inside `null_data_audit` step
- Last readable content notes: `def_rating INVERSION BUG` at `engine.py:648-649` (all MLB game-line edges corrupt); `draw_prob LEAK` (~13% draw_prob deflating win probs); MLB game-line Bet Cards from this session must NOT be acted on
- **Action required:** MLB engine bugs from this session are confirmed in the sidecar. Do not use game-line or run-line outputs from 2026-05-28 MLB session.

**`sess-20260528-prp1.json`** (98 lines, 4550 chars)
- Truncated inside `null_data_audit` step
- Last readable content notes: NBA props (SGA assists / Wemby pts / J.Williams pts) running on static calibration; no matchup-specific defensive assignment; narrow edges near tier boundary unreliable
- **Action required:** Props from this session carry elevated uncertainty; verify before acting.

**`sess-20260528-wnb1.json`** (161 lines, 7903 chars)
- Truncated inside `null_data_audit` step
- Last readable content notes: Caitlin Clark line (21.5) was estimated from ESPN milestone threshold — no standard O/U line available; WNBA totals suppressed; both game props on static calibration
- **Action required:** Clark prop line is estimated — must verify against live book before acting. Do not treat as engine-confirmed.

---

### LLM Orchestration Notes

- No LLM inference was invoked during this pipeline run. All steps are deterministic Python scripts.
- The LLM (this session) identified the missing pydantic dependency, installed it, and re-executed the action plan. This is an environment remediation action, not a data inference action.
- No Bet Cards, edge%, Kelly fractions, probability estimates, or confidence tiers were generated or estimated by the LLM. All counts above come from deterministic script output.

---

### Rejected Candidates

None. No candidates were surfaced — the DB is empty of open traces for the relevant date window.

---

### Calibration Eligibility

- Calibration profile: static fallback (no fitted profile)
- Calibration-eligible traces in 30-day window: 0
- Condition for next calibration fit: requires graded traces with model predictions in the trace store

---

### Failure Modes / Operational Risks

| Risk | Severity | Status |
|------|----------|--------|
| pydantic absent from sandbox Python env | High | Mitigated (installed); will recur on fresh session unless added to requirements.txt or auto-installed |
| 3 truncated session sidecars from 2026-05-28 | Medium | Known; sidecars skipped cleanly; content partially readable |
| MLB game-line outputs from 2026-05-28 corrupt (engine bugs) | High | Noted in sidecar; not acted on |
| TraceStore FUSE redirect | Low | Expected behavior; runtime DB path is stable per session |
| DB has 0 traces — no outcome attachment possible | Informational | Expected given current pipeline state |

---

### Assumptions

1. ESPN scoreboard API returning 0 results for 2026-05-28 means no games played or no matching traces — both are consistent with an empty DB.
2. The `var/inbox/traces/backfill_20260528/` directory contains traces that were already processed in a prior session (hence not re-ingested).
3. Sidecar truncation is a write-interrupt issue, not a serialization bug in the sidecar writer itself.

---

### Suggested Actions

1. **Add pydantic to requirements.txt** or ensure sandbox env has it pre-installed; the missing-module failure will recur on any fresh sandbox boot.
2. **Recover or discard malformed sidecars:** Attempt to close the JSON manually (add `]`, `}`), or flag them as `invalid/` and move to a quarantine directory so they don't appear in future calibration report warnings.
3. **Do not act on 2026-05-28 MLB game-line or run-line outputs.** The def_rating inversion and draw_prob leak bugs are confirmed in the sidecar.
4. **Verify Clark prop line (21.5) against live book** before any WNBA action — the line was estimated, not engine-confirmed.
5. **Monitor DB population:** Until traces are present in the runtime DB, outcome attachment and signal scoring will continue to produce 0-row results. Confirm that the trace export pipeline is writing to the correct DB path.

---

*Generated: 2026-05-29T09:07:26Z | Sidecar: RUN_TRACE.jsonl (18 events)*
