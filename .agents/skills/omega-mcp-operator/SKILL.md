---
name: omega-mcp-operator
description: Operate Omega through repo-local MCP tools while preserving deterministic engine boundaries.
---

# Omega MCP Operator

Use `python -m omega.mcp.server` when MCP is available. For exact tools and
ownership rules, read `docs/LLM_MCP_INTERFACE.md` and the repo skill at
`.agents/skills/omega-mcp-operator/SKILL.md`.

## Tool discovery (deferred tools)

The `omega_*` tools are **not** in the base tool list — they are **deferred** and the
stdio server **boots slowly on the FUSE/Windows mount**. Load them with `ToolSearch`
(keyword `"omega"`, or `"select:omega_trace_query,omega_get_portfolio_summary,..."`)
before concluding they are unavailable; if the server shows "still connecting," wait and
re-run `ToolSearch`. Seeing no `omega_*` tools at t=0 means *not yet loaded*, not *not
declared*. See the omega-session-bootstrap skill (Step 2.5) for the full sequence.

Prefer typed tools over hand-rolled DB access: `omega_trace_query` (the `traces` table has
no `kind` column) and `omega_get_portfolio_summary` (its result key is `active_ledgers`,
not `bets`).
