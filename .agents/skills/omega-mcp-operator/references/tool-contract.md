# Omega MCP Tool Contract

The MCP server lives at `omega/mcp/server.py`. Its domain functions remain
importable even when the optional MCP package is not installed.

## Tools

- `omega_analyze_game(request, bankroll=1000.0)`
- `omega_analyze_prop(request, bankroll=1000.0)`
- `omega_analyze_slate(request)`
- `omega_chat_orchestrate(prompt, context=None)`
- `omega_replay_bundle(bundle, strict=False)`
- `omega_trace_get(trace_id, db_path=None)`
- `omega_trace_query(...)`
- `omega_trace_attach_outcome(trace_id, home_score, away_score, source="mcp", db_path=None)`
- `omega_calibration_fit_preview(db_path=None, league=None, method="isotonic", limit=1000)`
- `omega_evidence_retrieve(slots)`

## Resources

- `omega://docs/llm-mcp-interface`
- `omega://schemas/contracts`
- `omega://calibration/universal-latest`

## Prompts

- `omega_runtime_prompt`
- `omega_missing_input_repair`
- `omega_trace_audit`
- `omega_replay_review`

## Safety Notes

- Do not compute edge, EV, Kelly, calibration, staking, or grading in the LLM.
- Do not make replay the quant benchmark path.
- Do not promote calibration profiles from the preview tool.
- Do not fetch live evidence in replay.
