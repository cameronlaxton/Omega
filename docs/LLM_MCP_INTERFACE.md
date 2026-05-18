# Omega LLM/MCP Interface

Omega's standard LLM interface is a local MCP server that exposes typed tools over the canonical deterministic contracts. The MCP layer is not a second pipeline: it validates inputs, calls existing Omega modules, and returns versioned JSON-friendly envelopes.

## Ownership Boundary

The LLM may plan, route, gather evidence, arbitrate sources, explain results, and decide when to downgrade or refuse.

The deterministic Omega engine owns simulation, probability calibration, fair-price conversion, edge, EV, Kelly staking, confidence tiers, backtesting, grading, and calibration fitting math.

## Tool Surface

- `omega_analyze_game`: validates a single-game request and delegates to `omega.core.contracts.service.analyze`.
- `omega_analyze_prop`: validates a player-prop request and delegates to `omega.core.contracts.service.analyze`.
- `omega_analyze_slate`: validates a slate request and delegates to `omega.core.contracts.service.analyze`.
- `omega_chat_orchestrate`: returns an explicit unsupported response until a real chat orchestrator exists.
- `omega_replay_bundle`: performs replay-plane audit over a frozen `ReplayBundle`; live fetching is disabled.
- `omega_trace_get`: retrieves a persisted trace through `TraceStore`.
- `omega_trace_query`: queries persisted traces through `TraceStore`.
- `omega_trace_attach_outcome`: attaches outcomes after initial trace persistence.
- `omega_calibration_fit_preview`: previews calibration fitting without writing profiles or promoting candidates.
- `omega_evidence_retrieve`: returns a no-live-fetch skipped response in this adapter.
- `omega_resolve_odds`: resolves current Odds API markets into engine-ready input fields and provenance.

Analyze tools require explicit `session_id` and `bankroll`. Callers must not rely on default bankroll values for formal recommendations.

## Runtime Order

1. Use Omega MCP tools when available.
2. If MCP is unavailable but the local repo can execute Python, import `omega.core.contracts.service.analyze` directly.
3. If no deterministic path can run, produce qualitative text only. Do not emit Bet Cards, probabilities, fair prices, edge, EV, Kelly, units, confidence tiers, or fabricated trace IDs.

## Replay And Evaluation Rules

Replay is sampled audit for routing, evidence selection, downgrade discipline, trace completeness, and refusal discipline. Replay is not the default benchmark path and must not grade strategy quality.

Quant benchmark evaluation belongs to frozen artifacts, frozen odds snapshots, normalized decision-time contexts, known actual outcomes, exact seeds, and the shared calibration/staking/grading policy path.

## Local Server

For an MCP client that supports command-based local servers:

```json
{
  "mcpServers": {
    "omega": {
      "command": "python",
      "args": ["-m", "omega.mcp.server"],
      "cwd": "C:\\Users\\camer\\OneDrive\\Documents\\GitHub\\Omega"
    }
  }
}
```

The optional MCP SDK is loaded only by `build_server()`. Direct imports and unit tests for the domain tool functions do not require the optional dependency.
