---
canonical: false
generated_at: '2026-05-29T20:34:57.589405+00:00'
source_db_path: '/sessions/vibrant-ecstatic-clarke/.omega/runtime/omega_traces.db'
db_path_source: 'env_override'
trace_count_at_generation: 190
source_artifacts:
  - omega_traces.db
  - inbox/sessions/*.json (sidecars)
  - calibration registry
---
# Omega Calibration Report — MLB

Generated: `2026-05-29T20:34:57+00:00` | Window: last 30 days

## Agent Directive — Output Mode

**`RESEARCH_CANDIDATE`** — formal output (Bet Cards, edge%, Kelly, confidence tiers) is **not authorized** for this league in this window.

**Reason(s):**
- No fitted calibration profile — static fallback is active.

**Permitted:** qualitative matchup narrative, news synthesis, recent form, listed sportsbook lines from a cited source.
**Forbidden language:** “best bet”, “Tier A”, “Tier B”, “engine-confirmed”, “actionable bet”. Stake cap: ≤ 1u.

## 1. Coverage

| Metric | Count |
|---|---|
| Traces (all) | 51 |
| Traces with model predictions (calibration-eligible) | 25 |
| Graded (any outcome) | 38 |
| &nbsp;&nbsp;of which game-graded | 24 |
| &nbsp;&nbsp;of which prop-graded | 14 |
| **Graded + calibration-eligible (usable pairs)** | **24** |
| With bet_record _(wager tracking only — not used for calibration)_ | 16 |
| With closing_line _(CLV only — not required for grading)_ | 2 |

## 2. Production calibration profile

**None** — calibration is using the static fallback policy.

## 3. Realized metrics — game plane (graded game traces in window)

- n: 20
- Brier: 0.1968
- ECE (10-bin): 0.2435
- Log loss: 0.5832

> **FLAG — ECE > 0.05 on game plane.** Investigate which probability quintile is miscalibrated.

## 3B. Realized metrics — prop plane (graded prop traces in window)

_Fewer than 10 prop (prediction, outcome) pairs in window — metrics suppressed._

## 3C. Distribution CRPS â€” prop projection curves

- metric_version: `distribution_metrics_v1`
- n: 4
- Mean CRPS: 1.1210

| stat_key | n | mean_crps |
|---|---:|---:|
| strikeouts_pitched | 4 | 1.1210 |

## 4. CLV (bets with attached closing lines)

- n: 2
- Mean CLV: +0.00 cents
- Beat-close rate: 0.0%

## 5. Sessions (most recent)

| session_id | traces | graded | model | pipeline | next_action | closes | webfetch_fail | notes |
|---|---|---|---|---|---|---|---|---|
| `sess-20260529-mlb1` | 5 | 0 | claude-sonnet-4-6 | ? | ? | ? | ? | Calibration accumulation session 2026-05-29 evening MLB slate. 5 complete-contex |
| `sess-20260527-mlb1` | 13 | 13 | claude-sonnet-4-6 | ? | ? | ? | 2 | Session opened for MLB early-slate prop research May 27 2026. Preflight failed:  |
| `sess-20260526-auto` | 12 | 11 | claude-sonnet-4-6 | ? | ? | ? | 0 | Automated run. 7 game analyses (CIN@NYM, ATL@BOS, MIN@CWS, PHI@SD, HOU@TEX, TAM@ |
| `sess-20260519-mlb1` | 1 | 0 | ? | ? | ? | ? | ? |  |
| `sess-20260518-mlb1` | 5 | 4 | claude-sonnet-4-6 | ? | ? | ? | 0 | [migrated from pre-schema sidecar] \| archived_keys: date="2026-05-18"; traces=[ |
| `sess-20260517-k9m2` | 5 | 3 | ? | ? | ? | ? | ? |  |
| `sess-20260516-mlb1` | 1 | 1 | ? | ? | ? | ? | ? |  |
| `sess-20260515-mlb1` | 6 | 3 | ? | ? | ? | ? | ? |  |
| `sess-20260514-mlb1` | 3 | 3 | ? | ? | ? | ? | ? |  |

## 6. Pending CANDIDATE profiles

_No pending candidates._

## 6B. Evidence signal performance (retrospective)

| signal_type | source | window | n | dir_acc | mean_conf | cal_gap | brier | verdict |
|---|---|---|---|---|---|---|---|---|
| starter_era | mlb.com | season | 3 | 0.67 | 0.87 | +0.20 | 0.272 | insufficient_n |
| motivation_edge | agent_reasoning | matchup | 1 | 0.00 | 0.60 | +0.60 | 0.360 | insufficient_n |
| starter_era | baseball-reference.com | season | 1 | 1.00 | 0.87 | -0.13 | 0.017 | insufficient_n |
| win_streak | espn.com | last_10 | 1 | 0.00 | 0.70 | +0.70 | 0.490 | insufficient_n |

> Weight evidence by empirical accuracy: trust `predictive` signal types/sources, discount `noise`, treat `insufficient_n` as unproven. A positive `cal_gap` means the agent was overconfident in that signal.

## 7. Suggested actions

_This section is intentionally empty. The LLM consumes the data above and emits an action plan per system_prompt.txt §13._
