# Daily Slate Gathering — Scheduled Operator Prompt

**Cadence:** daily
**Purpose:** build today's full slate (game + prop) across active leagues and close with a
narrative-first session report.

Copy the block below into the scheduled task. This repo copy is the source of truth — keep the
scheduler in sync with this file. It is an LLM-reasoning task (not a cron script): the reasoning,
context-gathering, and downgrade decisions are the point.

---

Use the **omega-session-bootstrap** and **omega-mcp-operator** skills first, then run today's full
slate end-to-end.

Fetch all of **today's** scheduled matchups via the league daily prompts — MLB, FIFA, WNBA; golf on
weekends; tennis for the day's matches. Today only; do not analyze future days.

For every game and player prop:
- gather real context, news, and reasoning to influence the call;
- file typed `evidence` signals (not free text);
- run `analyze()` so traces persist — research-only markets still run, because that feeds the
  calibration loop.

Full-slate volume is roughly ~20 MLB, 4–6 FIFA, ~10–20 WNBA game+prop, weekend golf, plus the day's
tennis. Treat these as expectations for a full slate, not quotas.

Close with a **narrative-first** session report following
`prompts/reference/presentation_contract.md`, in both `ACTIONABLE` and `RESEARCH_CANDIDATE` modes:
- slate snapshot;
- ranked recommendations / watchlist;
- per-matchup narrative (match context, market read, Omega read, risks, verdict);
- an honesty block for every recommendation/watchlist item;
- explicit pass rationale for major markets scanned but not recommended.

Authorize formal Bet Cards per `output_modes.<market>`, using engine-trace values only. For
research-only markets, present research leans / watchlist language.

**Never fabricate actionable edges to hit a count.** A low or zero actionable slate is acceptable
when honestly warranted — do not promote a research-only market just to avoid a zero count.

Finally, run the post-session action plans:

```bash
omega-run-action-plan fixtures/action_plans/daily_trace_intake.json
omega-run-action-plan fixtures/action_plans/render_session_audits.json
```
