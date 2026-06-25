# Omega

Omega is a reasoning-led sports analytics system: an LLM/operator layer gathers auditable inputs, and a deterministic Python engine performs simulation, calibration, edge, EV, Kelly staking, confidence tiering, backtesting, grading, and trace ID generation.

The canonical runtime is now local VM / MCP-first. Stateless sandbox bridging has been retired.

Please see [AGENTS.md](AGENTS.md) for the canonical product doctrine, phase guidelines, and artifact authority rules.

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

## What Lives In `src/omega/`

- `src/omega/core/`: simulation, sport archetypes, calibration, request/response contracts, Kelly staking, and parlay math.
- `src/omega/integrations/`: external providers (Odds API, ESPN, WeHoop, ETL guards).
- `src/omega/trace/`: trace persistence, bet records, closing lines, outcomes, CLV, and market snapshots.
- `src/omega/strategy/`: frozen artifacts, backtest engine, anchor-parlay scanner, and strategy versioning.
- `src/omega/mcp/`: typed local MCP tools over the deterministic contracts.
- `src/omega/ops/`: operational entrypoints (ingest, grading, calibration/reporting, validation, action-plan dispatch).

Runtime state now belongs under `var/` (`var/omega_traces.db`, transient var/inbox/report artifacts) and is intentionally git-ignored.

## Quick Start

Omega requires Python 3.10+.

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
omega-preflight
python -m omega.mcp.server
```

## Local Odds Resolution

Normal local odds resolution uses BetMGM:

```bash
omega-resolve-odds --kind game --league NBA --home-team "Los Angeles Lakers" --away-team "Boston Celtics"
```

Use multi-book mode only when explicitly shopping lines or auditing consensus:

```bash
omega-resolve-odds --kind game --league NBA --event-id evt-id --line-shopping
```

The resolver prepares market inputs and provenance only. It does not compute probabilities, edge, EV, Kelly, staking, confidence tiers, or trace IDs.

## Operating Documents

- [AGENTS.md](AGENTS.md): **cross-agent entrypoint â€” start here.** (`CLAUDE.md` is a shim to it.)
- [prompts/reference/output_modes.md](prompts/reference/output_modes.md): canonical output-mode + engine-execution rules.
- [OMEGA_RUNTIME.md](OMEGA_RUNTIME.md): local VM / MCP runtime instruction.
- [docs/LLM_MCP_INTERFACE.md](docs/LLM_MCP_INTERFACE.md): MCP tool contract and replay boundary.
- [docs/data_sources.md](docs/data_sources.md): data sourcing and freshness rules.

Retired to `archive/historical/` (non-authoritative): `OMEGA_HANDBOOK.md`, `OMEGA_RUN_RECIPE.md`.

## Testing

```bash
python -m pytest tests/ -v
```

Tests are network-free and do not spend Odds API quota.

