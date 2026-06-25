# Pending-Outcome Fetch — Scheduled Operator Prompt

**Cadence:** daily
**Purpose:** fetch and grade outcomes for pending traces/bets, excluding tennis.

Copy the block below into the scheduled task. This repo copy is the source of truth — keep the
scheduler in sync with this file.

---

Use `prompts/system_prompt.txt`, the **omega-session-bootstrap** skill, and the omega MCP tools.

Daily task: fetch outcomes for all pending traces/bets **except tennis** (tennis is handled weekly
to preserve OddsAPI credits). Attach outcomes, grade graded markets, and provide a concise updated
overview.

Follow the outcome loop in `prompts/ops/fetch_outcomes.md`. This is a data task — no betting
recommendations and no protected values are required.
