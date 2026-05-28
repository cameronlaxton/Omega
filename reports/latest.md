# Omega Calibration Report — NBA

Generated: `2026-05-28T20:30:47+00:00` | Window: last 30 days

## Agent Directive — Output Mode

**`RESEARCH_CANDIDATE`** — formal output (Bet Cards, edge%, Kelly, confidence tiers) is **not authorized** for this league in this window.

**Reason(s):**
- No fitted calibration profile — static fallback is active.
- 0 calibration-eligible traces in window.

**Permitted:** qualitative matchup narrative, news synthesis, recent form, listed sportsbook lines from a cited source.
**Forbidden language:** “best bet”, “Tier A”, “Tier B”, “engine-confirmed”, “actionable bet”. Stake cap: ≤ 1u.

## 1. Coverage

| Metric | Count |
|---|---|
| Traces (all) | 84 |
| Traces with model predictions (calibration-eligible) | 0 |
| Graded (any outcome) | 75 |
| &nbsp;&nbsp;of which game-graded | 14 |
| &nbsp;&nbsp;of which prop-graded | 61 |
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
| `sess-20260528-nba1` | 7 | 0 | claude-sonnet-4-6 | ? | 0 | WCF Game 4 SA vs OKC. 6 bets: UNDER 218.5, Wemby UNDER pts 27.5 reb 12.5, Chet O |
| `sess-20260526-nba1` | 3 | 3 | claude-sonnet-4-6 | ? | 0 | 10 candidates analyzed (3 game-level Markov, 7 props). 7 positive-edge bets surf |
| `sess-20260524-nba1` | 3 | 3 | claude-sonnet-4-6 | ? | 0 | Final: SA 103 OKC 82. Series tied 2-2. 2W/1L on session props. SGA held to 19 pt |
| `sess-20260523-nba1` | 8 | 8 | claude-sonnet-4-6 | ? | 0 | Preflight: cowork_preflight.py truncated 1 line (recurring Pattern C); restored  |
| `sess-20260521-nyk2` | 6 | 6 | ? | ? | ? |  |
| `sess-20260520-g001` | 7 | 7 | claude-sonnet-4-6 | ? | 0 | [migrated from pre-schema sidecar] \| archived_keys: date="2026-05-20"; traces=[ |
| `sess-20260519-nyk1` | 15 | 14 | claude-sonnet-4-6 | ? | 0 | [migrated from pre-schema sidecar] |
| `sess-20260519-nba1` | 3 | 3 | claude-sonnet-4-6 | ? | 0 | [migrated from pre-schema sidecar] \| archived_keys: date="2026-05-19"; traces=[ |
| `sess-20260518-wcf1` | 4 | 4 | claude-sonnet-4-6 | ? | 0 | [migrated from pre-schema sidecar] \| archived_keys: date="2026-05-18"; traces=[ |
| `sess-20260517-p3k8` | 12 | 12 | ? | ? | ? |  |

## 6. Pending CANDIDATE profiles

_No pending candidates._

## 6B. Evidence signal performance (retrospective)

| signal_type | source | window | n | dir_acc | mean_conf | cal_gap | brier | verdict |
|---|---|---|---|---|---|---|---|---|
| series_avg | nba.com | series | 4 | 0.50 | 0.88 | +0.38 | 0.391 | insufficient_n |
| home_away_split | agent_reasoning | series | 2 | 0.00 | 0.57 | +0.57 | 0.331 | insufficient_n |
| last_game_outlier | nba.com | last_1 | 2 | 0.50 | 0.78 | +0.28 | 0.301 | insufficient_n |
| def_matchup_strong | basketball-reference.com | series | 1 | 1.00 | 0.80 | -0.20 | 0.040 | insufficient_n |
| overtime_adjustment | agent_reasoning | last_1 | 1 | 1.00 | 0.70 | -0.30 | 0.090 | insufficient_n |
| recent_form | nba.com | last_3 | 1 | 0.00 | 0.80 | +0.80 | 0.640 | insufficient_n |
| recent_form | sportsbettingdime.com | last_5 | 1 | 1.00 | 0.80 | -0.20 | 0.040 | insufficient_n |
| season_baseline | statmuse.com | season | 1 | 1.00 | 0.85 | -0.15 | 0.023 | insufficient_n |
| series_avg | statmuse.com | series | 1 | 0.00 | 0.90 | +0.90 | 0.810 | insufficient_n |
| usage_role_change | official.nba.com | matchup | 1 | 1.00 | 0.75 | -0.25 | 0.062 | insufficient_n |

> Weight evidence by empirical accuracy: trust `predictive` signal types/sources, discount `noise`, treat `insufficient_n` as unproven. A positive `cal_gap` means the agent was overconfident in that signal.

## 7. Suggested actions

_This section is intentionally empty. The LLM consumes the data above and emits an action plan per system_prompt.txt §13._
