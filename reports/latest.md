# Omega Calibration Report — NBA

Generated: `2026-05-22T03:29:36+00:00` | Window: last 30 days

## 1. Coverage

| Metric | Count |
|---|---|
| Traces | 63 |
| Graded (any outcome) | 57 |
| &nbsp;&nbsp;of which game-graded | 9 |
| &nbsp;&nbsp;of which prop-graded | 48 |
| With bet_record | 28 |
| With closing_line | 0 |

## 2. Production calibration profile

**None** — calibration is using the static fallback policy.

## 3. Realized metrics — game plane (graded game traces in window)

- n: 2
- Brier: 0.1175
- ECE (10-bin): 0.3095
- Log loss: 0.3937

> **FLAG — ECE > 0.05 on game plane.** Investigate which probability quintile is miscalibrated.

## 3B. Realized metrics — prop plane (graded prop traces in window)

- n: 38
- Brier: 0.3116
- ECE (10-bin): 0.2607
- Log loss: 0.8556

> **FLAG — ECE > 0.05 on prop plane.** Prop calibration is separately tunable; consider a prop-specific shrinkage profile.

## 4. CLV (bets with attached closing lines)

_No CLV-resolvable bets in window._

## 5. Sessions (most recent)

| session_id | traces | graded | model | closes | webfetch_fail | notes |
|---|---|---|---|---|---|---|
| `sess-20260521-nyk2` | 6 | 6 | ? | ? | ? |  |
| `sess-20260520-g001` | 7 | 7 | claude-sonnet-4-6 | ? | 0 | [migrated from pre-schema sidecar] \| archived_keys: date="2026-05-20"; traces=[ |
| `sess-20260519-nyk1` | 15 | 14 | claude-sonnet-4-6 | ? | 0 | [migrated from pre-schema sidecar] |
| `sess-20260519-nba1` | 3 | 3 | claude-sonnet-4-6 | ? | 0 | [migrated from pre-schema sidecar] \| archived_keys: date="2026-05-19"; traces=[ |
| `sess-20260518-wcf1` | 4 | 4 | claude-sonnet-4-6 | ? | 0 | [migrated from pre-schema sidecar] \| archived_keys: date="2026-05-18"; traces=[ |
| `sess-20260517-p3k8` | 12 | 12 | ? | ? | ? |  |
| `sess-20260515-g7d1` | 16 | 11 | claude-sonnet-4-6 | ? | 0 | User requested 4-6 player prop bets for Game 7 Pistons vs Cavaliers (Sunday May  |

## 6. Pending CANDIDATE profiles

_No pending candidates._

## 7. Suggested actions

_This section is intentionally empty. The LLM consumes the data above and emits an action plan per system_prompt.txt §13._
