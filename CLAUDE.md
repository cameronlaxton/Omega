# Omega — Project Guidelines

## End-State Vision

Keep Omega's end-state in mind as you design and review the system:

Omega is not meant to become just a sports model app with some LLM features bolted on. The target is a reasoning-led quality agent sitting on top of a deterministic quantitative engine. Its job is to understand the user's real intent, judge whether the available evidence is sufficient and trustworthy, choose the right analysis path, decide when simulation is appropriate, downgrade or refuse when quality is weak, and explain outputs clearly and honestly. The LLM should grow primarily in the control, planning, arbitration, and explanation layers; not replace the deterministic engine of record for simulation, calibration, backtesting, and staking. Build toward an Omega that feels adaptive and intelligent at the decision-quality layer, while remaining disciplined, auditable, and reproducible at the execution layer.

## Anti-Overengineering Constraint

Before adding anything, ask: "Does this directly prevent bad sim inputs, bad recommendations, or bad backtests?" If the answer is no, it should probably be deferred.

## Architecture (5 layers)

1. **Conversation** (`omega/api/`) — FastAPI, sessions, SSE streaming
2. **Reasoning** (`omega/reasoning/`) — intent, routing, planning, quality gate, orchestration, LLM client
3. **Evidence** (`omega/evidence/`) — collectors, entity resolution, validation, fusion, pipeline
4. **Execution** (`omega/core/`) — Monte Carlo simulation, archetypes, calibration, contracts, staking
5. **Synthesis** (`omega/synthesis/`) — response composition across 11 output package types

## Testing

```bash
python -m pytest tests/ -v
```

273 tests, ~3s. All tests run without API keys (heuristic/deterministic paths).
