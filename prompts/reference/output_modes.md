# Output Modes & Engine Execution — Canonical Reference

**This file is the single canonical operational source for output-mode semantics.**
Every other document (`AGENTS.md`, `OMEGA_COWORK.md`, `prompts/system_prompt.txt`, the daily
prompts) must **link to this file** rather than restate the rules below. If those docs disagree
with this file, this file wins.

---

## The core rule: output authorization ⊥ engine execution

`RESEARCH_CANDIDATE` is an **output-authorization** mode, **not** an execution mode.

When the engine is available:

- **`analyze()` still runs.**
- A **`sandbox-` trace_id is still minted** by the engine.
- The **trace is still persisted** to `var/omega_traces.db` (export → `ingest_traces.py` → store).
- **User-facing betting numbers are withheld or downgraded** (no Bet Card, edge%, EV%, Kelly,
  units, confidence tier, or trace_id in the reply).
- **Database trace generation is never withheld.** Withholding output from the user does **not**
  withhold the trace from the calibration loop.

This is how the cold-start calibration loop escapes: traces accumulate (with predictions) even
while output is research-only, get graded against outcomes, and eventually fit a profile that
unlocks `ACTIONABLE` output. **Skipping `analyze()` in `RESEARCH_CANDIDATE` mode starves that loop
and is a defect.**

---

## Two distinct scenarios (do not conflate)

| Scenario | Engine runs? | Trace persisted? | User-facing output |
|---|---|---|---|
| **Engine unavailable** — no Python/MCP path, preflight failed, import error | No | No | Qualitative-only; never a Bet Card; no fabricated trace_id |
| **Engine available + `RESEARCH_CANDIDATE`** — no fitted profile / 0 eligible traces / invalid sidecar | **Yes** | **Yes** | Qualitative-only to the user; **full engine output retained in the persisted trace** |

The first is an availability failure. The second is a presentation downgrade. Only the first
skips the engine.

---

## How `output_mode` is determined

Computed by `omega.ops.output_modes.classify_output_mode(calibration_profile, trace_count,
sidecar_valid)`. It returns `RESEARCH_CANDIDATE` if **any** of:

- no fitted production calibration profile (static fallback active), or
- 0 calibration-eligible traces in window, or
- the session sidecar is invalid/corrupt.

Otherwise it returns `ACTIONABLE`. Bet records (logged wagers) are **never** a factor — a Bet Card
is emitted before any wager exists.

### Machine-readable source of truth

`omega-report-calibration` writes the current mode into the **frontmatter** of
`var/reports/latest.md`:

```yaml
output_mode: research_candidate        # or: actionable
output_mode_reasons:
  - No fitted calibration profile — static fallback is active.
```

**Read `output_mode` from the frontmatter** as the authoritative machine-readable flag. The prose
"Agent Directive — Output Mode" block in the report body says the same thing in human form and is
kept for backward compatibility.

---

## `RESEARCH_CANDIDATE` — permitted vs forbidden

**Permitted in the user-facing reply:**

- Qualitative matchup narrative, news synthesis, recent form.
- Listed sportsbook lines from a cited source.
- Research-only lean / missing-data watchlist labels (no protected numbers).
- Stake guidance capped at **≤ 1u**.

**Forbidden in the user-facing reply** (these stay in the DB trace, not the response):

- Bet Cards / BetSlips / EdgeDetail rows.
- edge%, EV%, Kelly fraction, recommended units, confidence tier.
- model/calibrated probability, fair price / no-vig price.
- the `trace_id`.

**Forbidden language:** "best bet", "Tier A", "Tier B", "engine-confirmed", "actionable bet".

---

## Downgrade discipline (applies in any mode)

Before rendering a formal Bet Card, confirm all of:

- critical inputs present;
- aggregate input quality ≥ `0.7`;
- engine status is not `skipped` or `error`;
- the `trace_id` was minted by Python execution.

If quality is below `0.7`, or the profile is unfitted (`RESEARCH_CANDIDATE`), or the sidecar is
invalid: **the engine still runs** (if available) and the trace still persists — only the
user-facing output is downgraded. If fewer than 3 real facts are available and quality is below
`0.3`, return a limited-context narrative.

---

## Book provenance & line shopping in the Bet Card

Every actionable Bet Card must name the **source sportsbook** for the price it
quotes. This is the book the engine's `market_odds` came from — in the default
flow that is BetMGM (`omega-resolve-odds` is BetMGM-first). The book is recorded
on the persisted bet (`bet_ledger.bookmaker`); when it is genuinely unknown the
ledger stores `consensus`, and the Bet Card should say so rather than guess.

When odds were resolved with `--line-shopping` / `--all-books`, the resolver
payload carries a **`best_prices`** block: the best available price per
selection, each tagged with the real book that offers it. Surface it as an
advisory line under the Bet Card, e.g.:

> **Bet:** Celtics ML −130 (BetMGM) · *Best available: −115 (DraftKings)*

Rules for this advisory:

- `best_prices` is a **listed sportsbook line from a cited source**, so it is
  permitted even in `RESEARCH_CANDIDATE` mode (unlike edge%/EV%/Kelly/tier).
- It is **advisory only**. The engine's edge/EV/Kelly stay computed against the
  single anchored book in `market_odds` — never recompute them against a
  best-shopped price, and never present a synthetic cross-book line (best-over
  from one book + best-under from another is two separate quotes, not one bet).
- If you did not line-shop (no `best_prices` block), omit the advisory line.

---

## The LLM ⊥ engine ownership boundary (hard rule)

The LLM may control: reasoning, planning, routing, evidence arbitration, explanation, and
downgrade decisions. The deterministic engine owns: simulation, probability calibration, edge
calculation, staking, backtesting, and grading.

The LLM is **forbidden from generating any of the following via text** — they must come from Python
execution through the local MCP server or `omega.core.contracts.service.analyze`:

- Bet Cards / BetSlips / EdgeDetail rows;
- model probabilities, calibrated probabilities, fair-price / no-vig price;
- EV% / edge% / expected-value calculations;
- Kelly fractions, recommended units, staking sizes;
- confidence tiers (A / B / C / Pass);
- `trace_id`s (always begin with `sandbox-`, minted by the engine).

There is no "estimated", "rough", "ballpark", `[LLM-ESTIMATED]`, or "estimated lean" mode for these
fields. If the engine is unavailable, the response is qualitative-only.
