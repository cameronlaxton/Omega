# Omega Calibration Report — NBA

Generated: `2026-05-27T10:08:29+00:00` | Window: last 30 days

## 1. Coverage

| Metric | Count |
|---|---|
| Traces (all) | 13 |
| Traces with model predictions (calibration-eligible) | 10 |
| Graded (any outcome) | 13 |
| &nbsp;&nbsp;of which game-graded | 4 |
| &nbsp;&nbsp;of which prop-graded | 9 |
| **Graded + calibration-eligible (usable pairs)** | **10** |
| With bet_record | 0 |
| With closing_line | 0 |

## 2. Production calibration profile

**None** — calibration is using the static fallback policy.

## 3. Realized metrics — game plane (graded game traces in window)

- n: 3
- Brier: 0.4160
- ECE (10-bin): 0.6450
- Log loss: 1.0357

> **FLAG — ECE > 0.05 on game plane.** Investigate which probability quintile is miscalibrated.

## 3B. Realized metrics — prop plane (graded prop traces in window)

_Fewer than 10 prop (prediction, outcome) pairs in window — metrics suppressed._

## 3C. Distribution CRPS â€” prop projection curves

- metric_version: `distribution_metrics_v1`
- n: 7
- Mean CRPS: 4.4267

| stat_key | n | mean_crps |
|---|---:|---:|
| blk | 1 | 0.4048 |
| pra | 1 | 12.6424 |
| pts | 4 | 3.4424 |
| reb | 1 | 4.1702 |

## 4. CLV (bets with attached closing lines)

_No CLV-resolvable bets in window._

## 5. Sessions (most recent)

| session_id | traces | graded | model | closes | webfetch_fail | notes |
|---|---|---|---|---|---|---|
| `sess-20260526-nba1` | 13 | 13 | claude-sonnet-4-6 | ? | 0 | 10 candidates analyzed (3 game-level Markov, 7 props). 7 positive-edge bets surf |

## 6. Pending CANDIDATE profiles

_No pending candidates._

## 6B. Evidence signal performance (retrospective)

| signal_type | source | window | n | dir_acc | mean_conf | cal_gap | brier | verdict |
|---|---|---|---|---|---|---|---|---|
| def_matchup_strong | nba.com/stats | season | 6 | 0.50 | 0.80 | +0.30 | 0.292 | insufficient_n |
| series_avg | nba.com | series | 5 | 0.20 | 0.85 | +0.65 | 0.576 | insufficient_n |
| usage_role_change | injury_report | matchup | 3 | 1.00 | 0.72 | -0.28 | 0.078 | insufficient_n |
| opponent_stat_rank | nba.com/stats | season | 2 | 0.00 | 0.78 | +0.78 | 0.617 | insufficient_n |
| def_matchup_strong | basketball-reference.com | series | 1 | 1.00 | 0.80 | -0.20 | 0.040 | insufficient_n |
| home_away_split | nba.com/stats | season | 1 | 1.00 | 0.55 | -0.45 | 0.202 | insufficient_n |
| usage_role_change | official.nba.com | matchup | 1 | 1.00 | 0.75 | -0.25 | 0.062 | insufficient_n |

> Weight evidence by empirical accuracy: trust `predictive` signal types/sources, discount `noise`, treat `insufficient_n` as unproven. A positive `cal_gap` means the agent was overconfident in that signal.

## 7. Suggested actions

_This section is intentionally empty. The LLM consumes the data above and emits an action plan per system_prompt.txt §13._
