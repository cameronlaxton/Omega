# OMEGA_HANDBOOK

The single source of truth about what Omega is, what it produces, and when it refuses to produce things. Intended as Claude.ai Project knowledge — keep concise; every section maps to behavior an LLM consumer needs to imitate or describe faithfully.

## What Omega is

Omega is a quantitative sports analytics agent. The end-state is a **reasoning-led quality agent sitting on top of a deterministic quantitative engine**. The LLM owns intent classification, evidence judgement, refusal logic, and explanation; the deterministic engine owns simulation, calibration, edge calculation, and staking. Omega is *not* "a sports model app with LLM features bolted on" and it is *not* a free-form chatbot that invents numbers.

## The 5 architectural layers

1. **Conversation** (`omega/api/`) — FastAPI surface, sessions, SSE streaming.
2. **Reasoning** (`omega/reasoning/`) — intent classification, answer-strategy routing, gather-slot planning, quality gate, orchestrator, LLM client.
3. **Evidence** (`omega/evidence/`) — collectors, entity resolution, validators (freshness, sanity, stats), fusion across sources.
4. **Execution** (`omega/core/`) — Monte Carlo simulation, sport archetypes, probability calibration, strict request/response contracts, Kelly staking.
5. **Synthesis** (`omega/synthesis/`) — response composition into one or more `OutputPackage` types.

When you (a downstream LLM) act on Omega's behalf, you are *Layer 5 only*. You do not own Layers 1–4.

## The 8 user-intent subjects

Used by `QueryUnderstanding.subjects`: `game`, `player_prop`, `slate`, `comparison`, `bankroll`, `news_context`, `unsupported_sport`, `general_sports`.

## The 8 user goals

Used by `QueryUnderstanding.goal`: `decide`, `analyze`, `compare`, `explain`, `discuss`, `summarize`, `learn`, `monitor`. The goal shapes which output packages the strategist selects.

## The 5 execution modes

Defined as `ExecutionMode`:

- **NATIVE_SIM** — full Monte Carlo + calibration + edges. Requires critical inputs filled.
- **RESEARCH** — fact gathering + LLM synthesis. No formal edges.
- **BANKROLL_CALC** — Kelly/staking math only. No simulation. Requires user-supplied bankroll.
- **MIXED** — simulation for directional context, narrative for the rest. Used when data is mid-quality.
- **NARRATIVE** — pure explanation/discussion. No numbers claimed as model output.

## The 11 output packages

Defined as `OutputPackage`:

- **BET_CARD** — edges, staking, confidence tiers. Gated by data quality.
- **GAME_BREAKDOWN** — simulation results + narrative analysis.
- **SCENARIO_ANALYSIS** — branching what-ifs over simulation results.
- **KEY_FACTORS** — top drivers, matchup advantages.
- **ALTERNATIVE_BETS** — secondary edges if BET_CARD is present.
- **BANKROLL_GUIDANCE** — Kelly fraction, unit sizing.
- **NEWS_DIGEST** — recent news signals affecting the matchup.
- **RESEARCH_REPORT** — structured analysis without formal edges.
- **PLAIN_EXPLANATION** — answer to a "why / how" question.
- **COMPACT_SUMMARY** — one-paragraph answer.
- **LIMITED_CONTEXT_ANSWER** — ultra-low-data fallback. No bet recommendations.

## The 3 input importance tiers

Defined as `InputImportance`. Drives the quality gate.

- **CRITICAL** — missing → no formal edges, no BET_CARD.
- **IMPORTANT** — missing → confidence tier capped at C.
- **OPTIONAL** — missing → answer still valid, just less nuanced.

## Quality gate behavior

Runs after evidence gathering, before execution (`omega/reasoning/evaluator.py`). Concrete rules:

1. **BET_CARD gate** — if any CRITICAL input is missing or aggregate data-quality < 0.7 (default threshold), drop BET_CARD and ALTERNATIVE_BETS, add RESEARCH_REPORT. Downgrade reason logged as `dropped_bet_card`.
2. **GAME_BREAKDOWN gate** — if any CRITICAL missing or quality < 0.5, keep the package but force narrative-only output (no sim numbers). Reason: `game_breakdown_narrative_only`.
3. **NATIVE_SIM feasibility** — if no CRITICAL filled:
   - Fill rate ≥ 0.5 across CRITICAL+IMPORTANT slots → downgrade NATIVE_SIM → MIXED. Reason: `native_sim_to_mixed`.
   - Fill rate < 0.5 → downgrade NATIVE_SIM → RESEARCH, add RESEARCH_REPORT. Reason: `native_sim_to_research`.
4. **Ultra-low data** — if fewer than 3 filled facts AND quality < 0.3, force `output_packages = [LIMITED_CONTEXT_ANSWER]` and switch any simulation mode to RESEARCH. Reason: `ultra_low_data`.

If Omega refuses a BET_CARD, a downstream LLM consumer must also refuse. Do not "rescue" the response by inventing edges or sims.

## The canonical Bet Card shape

Composer output for the `bet_card` section is:
```json
{
  "edges": [ /* List[EdgeDetail] */ ],
  "best_bet": { /* BetSlip */ },
  "data_completeness": { /* slot.key → "real" | "defaulted" | "missing" */ }
}
```

`BetSlip` fields (every recommended bet must conform): `selection`, `odds`, `edge_pct`, `ev_pct`, `confidence_tier` (A | B | C | Pass), `recommended_units`, `kelly_fraction`.

`EdgeDetail` fields: `side`, `team`, `true_prob`, `calibrated_prob`, `market_implied`, `edge_pct`, `ev_pct`, `market_odds`, `confidence_tier`.

`SimulationResult` fields: `iterations`, `home_win_prob`, `away_win_prob`, optional `draw_prob`, `predicted_spread`, `predicted_total`, `predicted_home_score`, `predicted_away_score`.

When you render a Bet Card from a real Omega run, mirror these fields exactly. Do not invent additional fields.

## Provenance — `ExecutionTrace`

Every run produces an `ExecutionTrace` with `trace_id`, `run_id`, `model_version`, per-stage timings, downgrades, gathered-facts summary, and the final predictions and recommendations. Cite the `trace_id` when rendering a real Omega run; if absent, the response is not a "real Omega run."

## What Omega does NOT do

- It does not auto-execute bets.
- It does not promise calibration across all sports — calibration is per-archetype, and archetypes for some sports may not exist.
- It does not scrape sportsbooks. Market odds come from The Odds API (with an API key) or from user-supplied input.
- It does not run inside Claude.ai's analysis tool. Anything claiming "I ran Omega in the sandbox" without a `trace_id` is hallucinated.

## Source code anchors

- `omega/core/models.py` — `Subject`, `UserGoal`, `ExecutionMode`, `OutputPackage`, `InputImportance`, `ExecutionTrace`.
- `omega/core/contracts/schemas.py` — Pydantic request/response models.
- `omega/reasoning/evaluator.py` — quality-gate logic.
- `omega/synthesis/composer.py` — `compose_response` and bet-card shape.
