# Omega LLM/MCP Interface

Omega's standard LLM interface is a local MCP server that exposes typed tools over the canonical deterministic contracts. The MCP layer is not a second pipeline: it validates inputs, calls existing Omega modules, and returns versioned JSON-friendly envelopes.

## Ownership Boundary

The LLM may plan, route, gather evidence, arbitrate sources, explain results, and decide when to downgrade or refuse.

The deterministic Omega engine owns simulation, probability calibration, fair-price conversion, edge, EV, Kelly staking, confidence tiers, backtesting, grading, and calibration fitting math.

## Tool Surface

- `omega_analyze_game`: validates a single-game request and delegates to `omega.core.contracts.service.analyze`.
- `omega_analyze_prop`: validates a player-prop request and delegates to `omega.core.contracts.service.analyze`.
- `omega_analyze_slate`: validates a slate request and delegates to `omega.core.contracts.service.analyze`.
- `omega_run_batch`: accepts a list of `BatchAnalysisEntry` dicts (mixed game/prop), resolves odds per-entry via `omega_resolve_odds` (with prop_type fallback chain if a list is supplied), calls `analyze()` per entry, and writes each export block to `var/inbox/traces/<trace_id>.json`. Use for sessions producing more than 3 analyses — replaces the need for manual looping or scratch scripts. Returns a summary with per-entry status (ok/skipped/error), all trace_ids, and export paths.
- `omega_chat_orchestrate`: returns an explicit unsupported response until a real chat orchestrator exists.
- `omega_replay_bundle`: performs replay-plane audit over a frozen `ReplayBundle`; live fetching is disabled.
- `omega_trace_get`: retrieves a persisted trace through `TraceStore`.
- `omega_trace_query`: queries persisted traces through `TraceStore`.
- `omega_trace_attach_outcome`: attaches game outcomes after initial trace persistence.
- `omega_trace_void_prop`: records a DNP / no-action void for a player prop absent from the box score (player did not play), so settlement returns VOID (stake returned) instead of leaving the bet pending or grading it as a loss.
- `omega_fetch_outcomes`: batch-gathers outcomes across leagues (wraps `omega.ops.fetch_outcomes_all`). Defaults to all leagues; pass `leagues` without `"soccer"` to exclude future-dated fixtures. `dry_run=True` reports what would run. Returns per-league status — use this instead of shelling out to `omega-fetch-outcomes`.
- `omega_settle_bets`: settles pending `bet_ledger` rows with attached outcomes (wraps `omega.ops.settle_bets`). `apply=False` (default) is a dry run. Returns settled counts, staked, net PnL, ROI — use this instead of shelling out to `settle_bets`.
- `omega_calibration_fit_preview`: previews calibration fitting without writing profiles or promoting candidates.
- `omega_evidence_retrieve`: returns a no-live-fetch skipped response in this adapter.
- `omega_resolve_odds`: resolves current Odds API markets into engine-ready input fields and provenance. For terminal slate discovery before a specific matchup is known, use `omega-resolve-odds --list-events --league <LEAGUE>`; repeated identical event-list calls are locally cached for 5 minutes, and Odds API budget exhaustion is a hard-stop CLI error.

Analyze tools require explicit `session_id` and `bankroll`. Callers must not rely on default bankroll values for formal recommendations.

## Runtime Order

1. Use Omega MCP tools when available.
2. If MCP is unavailable but the local repo can execute Python, import `omega.core.contracts.service.analyze` directly.
3. If no deterministic path can run, produce qualitative text only. Do not emit Bet Cards, probabilities, fair prices, edge, EV, Kelly, units, confidence tiers, or fabricated trace IDs.

## Replay And Evaluation Rules

Replay is sampled audit for routing, evidence selection, downgrade discipline, trace completeness, and refusal discipline. Replay is not the default benchmark path and must not grade strategy quality.

Quant benchmark evaluation belongs to frozen artifacts, frozen odds snapshots, normalized decision-time contexts, known actual outcomes, exact seeds, and the shared calibration/staking/grading policy path.

## Local Server

Before starting a local MCP server, the interpreter must be Python 3.10+ and
the project must be installed with the MCP extra:

```bash
python -m pip install -e .[mcp]
omega-cowork-preflight
```

For an MCP client that supports command-based local servers:

```json
{
  "mcpServers": {
    "omega": {
      "command": "python",
      "args": ["-m", "omega.mcp.server"],
      "cwd": "C:\\repos\\Omega"
    }
  }
}
```

The optional MCP SDK is loaded only by `build_server()`. Direct imports and unit tests for the domain tool functions do not require the optional dependency.
