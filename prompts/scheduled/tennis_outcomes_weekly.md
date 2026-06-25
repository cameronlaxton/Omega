# Tennis Outcome Fetch — Scheduled Operator Prompt

**Cadence:** weekly (tennis only — repo results report weekly, and this preserves OddsAPI credits)
**Purpose:** fetch and grade outcomes for pending tennis traces/bets.

Copy the block below into the scheduled task. This repo copy is the source of truth — keep the
scheduler in sync with this file.

---

Use `prompts/system_prompt.txt`, the **omega-session-bootstrap** skill, and the omega MCP tools.

Weekly task (tennis only): fetch outcomes for all pending tennis traces and bets in the DB. Attach
outcomes, grade, and provide a concise updated overview.

Follow the outcome loop in `prompts/ops/fetch_outcomes.md`. This is a data task — no betting
recommendations and no protected values are required.
