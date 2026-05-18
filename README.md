# Omega

Omega is a reasoning-led sports analytics system: an LLM/operator layer gathers auditable inputs, and a deterministic Python engine performs simulation, calibration, edge, EV, Kelly staking, confidence tiering, backtesting, grading, and trace ID generation.

The canonical runtime is now local VM / MCP-first. Stateless sandbox bridging has been retired.

## How Omega Runs

Use the local MCP server for agent operation:

```bash
python -m omega.mcp.server
```

MCP analyze tools delegate to `omega.core.contracts.service.analyze()`. Direct Python imports are acceptable for scripts and smoke tests when no MCP client is present:

```python
from omega.core.contracts.service import analyze
```

Formal numeric outputs require Python execution and a `sandbox-` trace ID returned by the engine. If the deterministic path cannot run, produce qualitative research only.

## What Lives In `omega/`

- `omega/core/`: simulation, sport archetypes, calibration, request/response contracts, Kelly staking, and parlay math.
- `omega/reasoning/`: quality and evidence policy helpers.
- `omega/integrations/odds_api.py`: The Odds API client for BetMGM-first current odds, closing lines, and historical market snapshots.
- `omega/trace/`: trace persistence, bet records, closing lines, outcomes, CLV, and market snapshots.
- `omega/strategy/`: frozen artifacts, backtest engine, anchor-parlay scanner, and strategy versioning.
- `omega/mcp/`: typed local MCP tools over the deterministic contracts.
- `omega/synthesis/`: response composition for narrative and fallback surfaces.

## Quick Start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e .
python -m pytest tests/ -v
```

Install optional MCP dependencies when running an MCP client:

```bash
pip install -e .[mcp]
python -m omega.mcp.server
```

## Local Odds Resolution

Normal local odds resolution uses BetMGM:

```bash
python scripts/resolve_odds.py --kind game --league NBA --home-team "Los Angeles Lakers" --away-team "Boston Celtics"
```

Use multi-book mode only when explicitly shopping lines or auditing consensus:

```bash
python scripts/resolve_odds.py --kind game --league NBA --event-id evt-id --line-shopping
```

The resolver prepares market inputs and provenance only. It does not compute probabilities, edge, EV, Kelly, staking, confidence tiers, or trace IDs.

## Operating Documents

- [OMEGA_COWORK.md](OMEGA_COWORK.md): local VM / Cowork runtime instruction.
- [docs/LLM_MCP_INTERFACE.md](docs/LLM_MCP_INTERFACE.md): MCP tool contract and replay boundary.
- [OMEGA_DATA_SOURCES.md](OMEGA_DATA_SOURCES.md): data sourcing and freshness rules.
- [OMEGA_RUN_RECIPE.md](OMEGA_RUN_RECIPE.md): local run recipe.
- [OMEGA_HANDBOOK.md](OMEGA_HANDBOOK.md): response and downgrade policy reference.
- [OMEGA_STRATEGY.md](OMEGA_STRATEGY.md): anchor parlay playbook.

## Testing

```bash
python -m pytest tests/ -v
```

Tests are network-free and do not spend Odds API quota.
