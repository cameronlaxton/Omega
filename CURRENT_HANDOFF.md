# Current Handoff - Phase 6h

Omega now runs through the local MCP server or the canonical core service.

## Runtime

- MCP entry point: `python -m omega.mcp.server`
- Direct Python entry point: `omega.core.contracts.service.analyze`
- Trace envelope model version: `omega-core-phase6h`
- Analyze calls require explicit `session_id` and `bankroll`.
- Live analyze calls should receive a deterministic integer seed derived from `sha256(prompt + date)`.

## Retired

The previous standalone bridge has been removed. Do not rebuild or route through it.

## Key Files

| Path | Purpose |
|---|---|
| `omega/core/contracts/service.py` | Canonical analyze wrapper and deterministic service |
| `omega/mcp/server.py` | Typed MCP tools |
| `scripts/resolve_odds.py` | BetMGM-first current odds resolver |
| `scripts/ingest_traces.py` | Trace export ingestion |
| `scripts/fetch_closing_lines.py` | Closing-line capture |
| `scripts/fetch_outcomes_props.py` | Player-prop outcome attachment |
| `OMEGA_COWORK.md` | Local VM runtime protocol |

## Next Pickup

Continue hardening the MCP-native workflow: improve seed provenance, add richer trace-quality metadata when needed, and keep replay/benchmark boundaries separate.
