# OMEGA_HANDBOOK

Omega is a reasoning-led sports analytics system on top of a deterministic quantitative engine. The LLM owns intent, evidence judgment, downgrade/refusal decisions, and explanation. The Python engine owns simulation, calibration, edge, EV, Kelly, staking, confidence tiers, trace IDs, backtesting, and grading.

## Runtime

Phase 6h standardizes on local VM execution:

1. Use MCP tools from `python -m omega.mcp.server`.
2. If MCP is unavailable, use `omega.core.contracts.service.analyze`.
3. If deterministic execution is unavailable, produce qualitative research only.

## Execution Modes

- `NATIVE_SIM`: full Monte Carlo + calibration + edges. Requires critical inputs.
- `RESEARCH`: fact gathering + synthesis only. No protected Omega numbers.
- `BANKROLL_CALC`: Python-only staking math from user-supplied bankroll.
- `MIXED`: directional context plus narrative, no formal BetSlip.
- `NARRATIVE`: pure explanation/discussion.

## Downgrade Discipline

The agent must enforce these checks before rendering a formal Bet Card:

- Critical inputs present.
- Aggregate input quality at least `0.7`.
- Engine status is not `skipped` or `error`.
- Trace ID was minted by Python execution.

If fewer than 3 real facts are available and quality is below `0.3`, return a limited-context narrative. Research-only leans may include sourced sportsbook lines and qualitative rationale, but no edge, EV, Kelly, units, confidence tier, or invented trace ID.

## Bet Card Shape

Composer output for the `bet_card` section is:

```json
{
  "edges": [],
  "best_bet": {},
  "data_completeness": {}
}
```

`BetSlip` fields: `selection`, `odds`, `edge_pct`, `ev_pct`, `confidence_tier`, `recommended_units`, `kelly_fraction`.

`EdgeDetail` fields: `side`, `team`, `true_prob`, `calibrated_prob`, `market_implied`, `edge_pct`, `ev_pct`, `market_odds`, `confidence_tier`.

Render these fields only from a real Python trace.

## Trace Provenance

Every formal run produces a trace envelope with `trace_id`, `model_version`, `ran_at`, `kind`, `session_id`, `bankroll`, `input_snapshot`, `result`, and `downgrades`. Cite the `trace_id` when rendering a real Omega run. If there is no trace ID, it is not a formal Omega run.

## Source Anchors

- `omega/core/contracts/service.py`: canonical analyze wrapper and trace envelope.
- `omega/core/contracts/schemas.py`: request/response models.
- `omega/mcp/server.py`: MCP tool layer.
- `omega/synthesis/composer.py`: response composition.
- `omega/trace/store.py`: trace persistence.
