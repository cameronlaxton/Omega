# Omega LLM/MCP Interface

Omega's preferred local LLM interface is a repo-local MCP server that exposes
typed tools over the existing deterministic contracts. The phrase to preserve:
MCP layer is not a second pipeline. It validates input, calls existing Omega modules, and returns
versioned JSON-friendly envelopes for agents and automation.

## Ownership Boundary

The LLM may plan, route, gather evidence, arbitrate sources, explain tool
results, and decide when to downgrade or refuse.

The deterministic Omega engine owns simulation, probability calibration, fair
price conversion, edge, EV, Kelly staking, confidence tiers, backtesting,
grading, and calibration fitting math. MCP tools must delegate those
responsibilities to existing Omega modules.

## Tool Surface

- `omega_analyze_game`: validates a single-game request and delegates to
  `omega_lite.run.analyze`.
- `omega_analyze_prop`: validates a player-prop request and delegates to
  `omega_lite.run.analyze`.
- `omega_analyze_slate`: validates a slate request and delegates to
  `omega_lite.run.analyze`.
- `omega_chat_orchestrate`: currently returns an explicit unsupported response
  because the current source-of-truth repo has no chat orchestrator module.
- `omega_replay_bundle`: performs replay-plane audit over a frozen
  `ReplayBundle`; live fetching is disabled and post-outcome facts are rejected.
- `omega_trace_get`: retrieves a persisted trace through `TraceStore`.
- `omega_trace_query`: queries persisted traces through `TraceStore`.
- `omega_trace_attach_outcome`: attaches outcomes after initial trace
  persistence through `TraceStore.attach_outcome`.
- `omega_calibration_fit_preview`: previews calibration fitting from graded
  traces without writing profiles or promoting candidates.
- `omega_evidence_retrieve`: returns a no-live-fetch skipped response in this
  adapter; use approved evidence channels outside replay when inputs are needed.

## Resources And Prompts

Resources:

- `omega://docs/llm-mcp-interface`
- `omega://schemas/contracts`
- `omega://calibration/universal-latest`

Prompts:

- `omega_runtime_prompt`
- `omega_missing_input_repair`
- `omega_trace_audit`
- `omega_replay_review`

## Runtime Order

1. Use Omega MCP tools when available.
2. If MCP is unavailable but the local repo can execute Python, use the repo
   engine path documented in `OMEGA_COWORK.md`.
3. If only a no-local-access project sandbox is available, use
   `omega_lite_standalone.py` as documented in `prompts/system_prompt.txt`.
4. If no deterministic path can run, produce Standard Text only. Do not emit
   Bet Cards, probabilities, fair prices, edge, EV, Kelly, units, confidence
   tiers, or fabricated trace IDs.

## Replay And Evaluation Rules

Replay is sampled audit for routing, evidence selection, downgrade discipline,
trace completeness, and refusal discipline. Replay is not the default benchmark
path and must not grade strategy quality.

Replay is not the default benchmark path.

Quant benchmark evaluation belongs to frozen artifacts, frozen odds snapshots,
normalized decision-time contexts, known actual outcomes, exact seeds, and the
shared calibration/staking/grading policy path.

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

The optional MCP SDK is loaded only by `build_server()`. Direct imports and unit
tests for the domain tool functions do not require the optional dependency.
