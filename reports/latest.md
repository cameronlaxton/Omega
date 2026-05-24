# Omega Calibration Report — NBA

Generated: `2026-05-24T20:13:02+00:00` | Window: last 30 days

## 1. Coverage

| Metric | Count |
|---|---|
| Traces (all) | 71 |
| Traces with model predictions (calibration-eligible) | 0 |
| Graded (any outcome) | 65 |
| &nbsp;&nbsp;of which game-graded | 9 |
| &nbsp;&nbsp;of which prop-graded | 56 |
| **Graded + calibration-eligible (usable pairs)** | **0** |
| With bet_record | 28 |
| With closing_line | 0 |

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

| session_id | traces | graded | model | closes | webfetch_fail | notes |
|---|---|---|---|---|---|---|
| `sess-20260523-nba1` | 8 | 8 | claude-sonnet-4-6 | ? | 0 | Preflight: cowork_preflight.py truncated 1 line (recurring Pattern C); restored  |
| `sess-20260521-nyk2` | 6 | 6 | ? | ? | ? |  |
| `sess-20260520-g001` | 7 | 7 | claude-sonnet-4-6 | ? | 0 | [migrated from pre-schema sidecar] \| archived_keys: date="2026-05-20"; traces=[ |
| `sess-20260519-nyk1` | 15 | 14 | claude-sonnet-4-6 | ? | 0 | [migrated from pre-schema sidecar] |
| `sess-20260519-nba1` | 3 | 3 | claude-sonnet-4-6 | ? | 0 | [migrated from pre-schema sidecar] \| archived_keys: date="2026-05-19"; traces=[ |
| `sess-20260518-wcf1` | 4 | 4 | claude-sonnet-4-6 | ? | 0 | [migrated from pre-schema sidecar] \| archived_keys: date="2026-05-18"; traces=[ |
| `sess-20260517-p3k8` | 12 | 12 | ? | ? | ? |  |
| `sess-20260515-g7d1` | 16 | 11 | claude-sonnet-4-6 | ? | 0 | User requested 4-6 player prop bets for Game 7 Pistons vs Cavaliers (Sunday May  |

## 6. Pending CANDIDATE profiles

_No pending candidates._

## 6B. Evidence signal performance (retrospective)

_No scored evidence signals yet — run `scripts/score_evidence_signals.py` after outcomes attach._

## 7. Suggested actions

_This section is intentionally empty. The LLM consumes the data above and emits an action plan per system_prompt.txt §13._
