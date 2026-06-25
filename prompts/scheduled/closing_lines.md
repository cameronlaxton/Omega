# Closing-Line Gather — Scheduled Operator Prompt

**Cadence:** daily
**Purpose:** attach closing-line values to pending CLV traces/bets.

Copy the block below into the scheduled task. This repo copy is the source of truth — keep the
scheduler in sync with this file.

---

Use `prompts/system_prompt.txt`, the **omega-session-bootstrap** skill, and the omega MCP tools.

Daily task: gather closing-line values for all pending CLV bets/traces. Attach each closing line to
its trace/bet and report coverage (how many pending, how many resolved, how many still open).

Follow the continuous CLV loop in `docs/issue28_clv_loop.md`. This is a data task — no betting
recommendations and no protected values are required.
