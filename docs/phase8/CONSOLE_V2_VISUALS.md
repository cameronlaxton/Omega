# Operator Console V2 — Data Visualizations (first wave + deferred backlog)

**Phase:** 8 (operator console)
**Status:** first wave landed; backlog deferred (this document is the authoritative plan for what is *not* yet built)
**Scope:** read-only operator console (`src/omega/ui/`, served by `omega-console`)

---

## Context

The console had exactly one chart (the calibration model-vs-market time series); every
other page was an HTML table. The data layer was far richer than the UI exposed — per-bet
CLV, evidence signals, signal-performance scoring, simulation distributions, bet-ledger
PnL, and QA verdicts were all readable but never *seen*.

This change adds a **focused first wave** of high-value, data-ready visualizations that make
the operator's core questions answerable at a glance — *did we beat the close? where is
Omega over/under-confident? where is data coverage thin? is good process being rewarded?* —
while staying inside the console's hard doctrine.

### Doctrine (unchanged, enforced by tests)

- **Read-only.** No mutation path; `ConsoleService` refuses a writable store; every route is GET.
- **Server-computed geometry.** Pure helpers scale already-computed values into pixels
  (`_strip_x`, `_scatter_geometry`, `_calibration_geometry`); templates only drop coordinates.
- **No client charting library.** Inline SVG + CSS; `app.js` does display-only tooltips.
- **One unit per chart**, declared explicitly; never mix units on one axis.
- **Honest empty / low-n states.** Missing data renders a muted dash or `panel-empty`; thin
  reliability buckets are *suppressed*, never drawn.
- **Provenance-labeled** and **forbidden terminology** respected (no "BET" verb / "Best
  Price" / "Value Score").

---

## First wave — shipped

| Visual | Where | Backing data |
|---|---|---|
| **Per-bet comparison strip (dumbbell)** — Omega P(selection) vs market implied probability; gap = edge | Edge Scanner column, Trace Detail primary card, Bet Detail | normalized recommendations (`calibrated`/`raw` vs `implied`); probability-space so the unit is identical for every market |
| **Market-movement ribbon** — taken → closing implied, outcome end-cap | CLV page rows | `ClvRow` (`taken_implied` → `closing_implied`, `status`) |
| **Calibration reliability diagram** — model-probability bucket vs realized hit rate, with the y=x diagonal | Calibration page | graded bets joined to the model probability that produced them; buckets `< min_n` suppressed |
| **CLV scatter** — closing-line value (x) vs net result (y), quadrant guides (process vs luck) | CLV page | `clv_report` join + `bet_ledger.net_pnl` |
| **Data-quality heatmap** — per-league coverage (evidence / closing line / outcome), R/A/G cells | new `/data-quality` page + nav | `get_session_trace_facts_batch` + `get_closing_lines_batch` (two bounded batch reads) |

**Shared primitives:** `_strip_x` / `_prob_strip` back the dumbbell *and* ribbon;
`_scatter_geometry` backs the reliability diagram *and* the CLV scatter — mirroring the
existing `_calibration_geometry` precedent.

**Key files:** `schemas.py` (new view models + `strip`/`net_pnl`/`primary_strip`/`linked_strip`
fields), `service.py` (geometry helpers + `reliability_diagram()` / `clv_scatter()` /
`data_quality()`), `templates/base.html` (`comp_strip` / `reliability_chart` / `clv_scatter` /
`quality_heatmap` macros), `templates/data_quality.html` (new), `static/styles.css`,
`static/app.js`, `ops/console_server.py` (route + nav), `api.py` (3 GET endpoints).

This wave also lands the **carried-over Console V2 re-skin** working-tree changes (hero
headers and table restructures on `bets.html` / `traces.html`, dash normalization, and the
shared `styles.css` / `base.html` polish they depend on).

---

## Deferred backlog (NOT built — authoritative list)

Each item below is specced but intentionally out of the first wave. Several are *reframes* of
originally-proposed ideas, changed to stay inside the honesty doctrine.

| # | Visual | Status / reframe | Why deferred |
|---|---|---|---|
| B1 | **Bankroll / PnL equity curve** (cumulative `net_pnl` over `graded_at`) | new; cheapest follow-up (reuses `_calibration_geometry` line path) | top of backlog — promote first |
| B2 | **Trace lifecycle coverage funnel** (traces → reviewed → ledger → graded → closing-line → calibration-eligible, with drop-off counts) | *reframed from "Bet Lifecycle Sankey"* | a full multi-path Sankey implies per-flow attribution we can't all source; the funnel is the honest form |
| B3 | **Signal coverage-vs-usefulness scatter** (x=coverage, y=realized usefulness/CLV-alignment, bubble=n) | *reframed from "Signal Radar"* | radar overstates weak axes and reads poorly; low-n must stay visibly faint |
| B4 | **Evidence coverage bar** (segmented by signal type, applied vs shadow) | *reframed from "Evidence Waterfall"* | a true contribution *waterfall* implies per-signal numeric deltas we do not have — false precision |
| B5 | **Review-queue kanban** (triage columns over existing buckets) | layout-only over existing `review_queue()` data | UX-only; no new data |
| B6 | **Per-trace lifecycle timeline** | *largely already exists* as the Decision Replay timeline in `trace_detail.html` | enhance pass/warn/fail states rather than rebuild |
| B7 | **League/market coverage treemap** | defer / merge | overlaps the data-quality heatmap; squarify geometry is heavy — only build if a distinct volume-weighted view is wanted |

### Explicitly out of scope (data not present, would be fabricated)
- **Odds-freshness** column on the heatmap — no multi-timestamp odds snapshot exists in the
  schema, so it is omitted rather than faked.
- **Live multi-book quote spread** — Omega records a single decision-time price; there is no
  "best price" to show.
- **Native-unit (points/goals) spread/total/prop dumbbell** — the first-wave dumbbell is
  probability-space only (universally comparable); native-unit framing needs sign/units
  verification before shipping.

---

## Verification

- Pure geometry helpers unit-tested for deterministic pixel coords, incl. a low-n
  suppression case (`tests/ui/test_comparison_strip.py`, `test_reliability_diagram.py`).
- Per-page route + HTML-assertion tests: SVG/heatmap elements present, honest empty states,
  XSS escaping, GET-only (`POST → 405`) on the new endpoints.
- `tests/ui` green (215 passing); `ruff` clean on changed files.
- Read-only audit: no new mutation path; `test_console_no_mutation_imports` still passes.
