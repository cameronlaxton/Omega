# Omega Calibration Report — NBA

Generated: `2026-05-20T20:46:52+00:00` | Window: last 30 days

## 1. Coverage

| Metric | Count |
|---|---|
| Traces | 36 |
| Graded (any outcome) | 35 |
| &nbsp;&nbsp;of which game-graded | 1 |
| &nbsp;&nbsp;of which prop-graded | 34 |
| With bet_record | 11 |
| With closing_line | 0 |

## 2. Production calibration profile

**None** — calibration is using the static fallback policy.

## 3. Realized metrics — game plane (graded game traces in window)

- n: 1
- Brier: 0.2088
- ECE (10-bin): 0.4570
- Log loss: 0.6106

> **FLAG — ECE > 0.05 on game plane.** Investigate which probability quintile is miscalibrated.

## 3B. Realized metrics — prop plane (graded prop traces in window)

- n: 27
- Brier: 0.3257
- ECE (10-bin): 0.2519
- Log loss: 0.8976

> **FLAG — ECE > 0.05 on prop plane.** Prop calibration is separately tunable; consider a prop-specific shrinkage profile.

## 4. CLV (bets with attached closing lines)

_No CLV-resolvable bets in window._

## 5. Sessions (most recent)

| session_id | traces | graded | model | closes | webfetch_fail | notes |
|---|---|---|---|---|---|---|
| `sess-20260519-nyk1` | 12 | 12 | ? | ? | ? |  |
| `sess-20260519-nba1` | 3 | 3 | ? | ? | ? |  |
| `sess-20260517-p3k8` | 12 | 12 | ? | ? | ? |  |
| `sess-20260515-g7d1` | 9 | 8 | claude-sonnet-4-6 | ? | 0 | User requested 4-6 player prop bets for Game 7 Pistons vs Cavaliers (Sunday May  |

## 6. Pending CANDIDATE profiles

_No pending candidates._

## 7. Suggested actions

_This section is intentionally empty. The LLM consumes the data above and emits an action plan per system_prompt.txt §13._
