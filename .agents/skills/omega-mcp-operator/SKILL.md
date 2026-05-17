---
name: omega-mcp-operator
description: Operate the Omega sports analytics repo through its local MCP server and typed tools. Use when Codex, Claude Code, or another LLM agent needs to run Omega analysis, call Omega MCP tools, inspect the MCP server, repair missing inputs, or preserve the boundary between LLM reasoning and deterministic simulation/calibration/edge/staking logic.
---

# Omega MCP Operator

## Core Rule

Use typed Omega tools before shell scripts or prose math. The LLM may plan,
route, gather missing evidence, explain, and downgrade. Omega's deterministic
engine owns simulation, calibration, edge, EV, Kelly, staking, backtesting, and
grading.

## Tool Order

1. Prefer the local MCP server:
   `python -m omega.mcp.server`
2. If MCP is unavailable, use the local repo engine path documented in
   `OMEGA_COWORK.md`.
3. Use `omega_lite_standalone.py` only for no-local-access project sandboxes.
4. If no deterministic path can run, produce qualitative Standard Text only.

## MCP Tools

- `omega_analyze_game`: deterministic single-game analysis.
- `omega_analyze_prop`: deterministic player-prop analysis.
- `omega_analyze_slate`: deterministic slate analysis.
- `omega_chat_orchestrate`: explicit unsupported response until the current
  repo has a real orchestrator.
- `omega_replay_bundle`: replay-plane audit from frozen facts with live
  fetching disabled.
- `omega_trace_get`, `omega_trace_query`, `omega_trace_attach_outcome`: trace
  persistence and post-decision outcome attachment.
- `omega_calibration_fit_preview`: dry-run calibration fitting only.
- `omega_evidence_retrieve`: no-live-fetch placeholder for explicit slots.

## Workflow

1. Read `docs/LLM_MCP_INTERFACE.md` when you need the full contract.
2. Validate the request shape against `omega/core/contracts/schemas.py` or the
   mirrored `omega_lite` schemas used by `omega_lite.run`.
3. Call the narrowest Omega MCP tool that satisfies the task.
4. If a result is skipped or gated, inspect `missing_requirements`,
   `skip_reason`, and quality-gate downgrades.
5. Retrieve only missing pre-decision inputs; preserve source notes.
6. Rerun the same deterministic tool.
7. Render Bet Cards only from successful deterministic output.

## Input Mapping

For package prop analysis, player context uses sport-specific keys:

- `stat_mean` maps to `{prop_type}_mean`
- `stat_std_dev` maps to `{prop_type}_std`

For standalone fallback only, use the generic `stat_mean` / `stat_std_dev`
schema described in `OMEGA_RUN_RECIPE.md`.

## References

- Read `references/tool-contract.md` for exact tools, resources, prompts, and
  cross-client setup notes.
