---
canonical: false
generated_at: '2026-05-30T20:02:49.970843+00:00'
source_db_path: 'C:\\repos\\Omega\\omega_traces.db'
db_path_source: 'default'
trace_count_at_generation: 226
source_artifacts:
  - omega_traces.db
  - inbox/sessions/*.json (sidecars)
  - calibration registry
---
# Omega Calibration Report — WNBA

Generated: `2026-05-30T20:02:49+00:00` | Window: last 30 days

## Agent Directive — Output Mode

**`RESEARCH_CANDIDATE`** — formal output (Bet Cards, edge%, Kelly, confidence tiers) is **not authorized** for this league in this window.

**Reason(s):**
- No fitted calibration profile — static fallback is active.

**Permitted:** qualitative matchup narrative, news synthesis, recent form, listed sportsbook lines from a cited source.
**Forbidden language:** “best bet”, “Tier A”, “Tier B”, “engine-confirmed”, “actionable bet”. Stake cap: ≤ 1u.

## 1. Coverage

| Metric | Count |
|---|---|
| Traces (all) | 7 |
| Traces with model predictions (calibration-eligible) | 7 |
| Graded (any outcome) | 0 |
| &nbsp;&nbsp;of which game-graded | 0 |
| &nbsp;&nbsp;of which prop-graded | 0 |
| **Graded + calibration-eligible (usable pairs)** | **0** |
| With bet_record _(wager tracking only — not used for calibration)_ | 0 |
| With closing_line _(CLV only — not required for grading)_ | 0 |

## 2. Production calibration profile

**None** — calibration is using the static fallback policy.

## 3. Realized metrics — game plane (graded game traces in window)

_Fewer than 10 game-graded traces in window — metrics suppressed (noise dominates)._

## 3B. Realized metrics — prop plane (graded prop traces in window)

_Fewer than 10 prop (prediction, outcome) pairs in window — metrics suppressed._

## 3C. Distribution CRPS â€” prop projection curves

_No V10 distribution rows with realized prop outcomes in window â€” CRPS suppressed._

## 4. CLV (bets with attached closing lines)

_No CLV-resolvable bets in window._

## 5. Sessions (most recent)

| session_id | traces | graded | model | pipeline | next_action | closes | webfetch_fail | notes |
|---|---|---|---|---|---|---|---|---|
| `sess-20260530-wnb1` | 7 | 0 | gemini-3.5-flash | ? | ? | ? | ? | WNBA regular season slate - 3 games + 3 player props analyzed. Matchups: Storm @ |

## 6. Pending CANDIDATE profiles

_No pending candidates._

## 6B. Evidence signal performance (retrospective)

_No scored evidence signals yet — run `scripts/score_evidence_signals.py` after outcomes attach._

## 7. Suggested actions

_This section is intentionally empty. The LLM consumes the data above and emits an action plan per system_prompt.txt §13._
