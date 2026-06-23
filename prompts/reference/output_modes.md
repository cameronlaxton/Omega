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

## How `output_mode` is determined — per market

Output authorization is resolved **independently per market** (`game` and `prop`). Each market is
classified by `omega.ops.output_modes.classify_market_output_mode(...)` from that market's **own**
production calibration profile — there is no prop→game fallback for *authorization*, even though
the runtime calibration path does fall back when applying a profile.

A market is `RESEARCH_CANDIDATE` if **any** of:

- no fitted production profile **for that market** (static fallback active), or
- 0 calibration-eligible traces **for that market** in window, or
- the session sidecar is invalid/corrupt, or
- the market's production profile fails the **calibration-quality floor**: `sample_size ≥ 100`
  **and** `calibration_error (ECE) ≤ 0.05`, read from the profile's recorded fit metrics. A
  force-promoted or under-sampled profile therefore does **not** unlock formal output.

Otherwise that market is `ACTIONABLE`. Bet records (logged wagers) are **never** a factor — a Bet
Card is emitted before any wager exists.

A trustworthy prop market can be `ACTIONABLE` while the game market is `RESEARCH_CANDIDATE`, and
vice versa. **Apply suppression per market**: a game Bet Card and a prop Bet Card are authorized
separately, off their own market's mode.

### Machine-readable source of truth

`omega-report-calibration` writes the per-market map into the **frontmatter** of
`var/reports/latest.md`:

```yaml
output_modes:
  game: 'research_candidate'
  prop: 'actionable'
output_mode_reasons:
  game:
    - 'No fitted calibration profile for this market - static fallback active.'
  prop: []
output_mode: 'research_candidate'   # backward-compat scalar — see below
```

**Read `output_modes.<market>` from the frontmatter** as the authoritative machine-readable flag
for that market. The scalar `output_mode` is a conservative aggregate (`actionable` only when
*every* market is `actionable`) kept for un-updated consumers — never use it to authorize one
market when the map says that market is `actionable`. The prose "Agent Directive — Output Mode"
block in the report body restates the per-market modes in human form.

---

## Graduated evidence application modes

Structured evidence no longer flows through a binary shadow/live switch. The active
`AdjustmentPolicy.mode` is one rung of a graduated ladder (`OMEGA_EVIDENCE_MODE` overrides it):

| Mode | Computed? | Recorded? | Feeds learning? | Moves prediction math? |
|---|---|---|---|---|
| `disabled` | no | no | no | no |
| `observe` | yes | yes | no | no |
| `score_only` *(default)* | yes | yes | yes | no |
| `bounded_live` | yes | yes | yes | **yes, under hard caps** |
| `live` | yes | yes | yes | yes (policy caps) |

- Legacy `shadow` normalizes to `score_only` on load.
- `bounded_live` enforces hard per-signal / family / plane caps and **cannot lift a recommendation
  to `A`** unless the policy's evidence metrics have passed promotion gates.
- The seed policy ships at `score_only` (records, never moves predictions) until an operator flips
  it with an explicit, audited `set_mode`.

## Calibration profile maturity

Profiles are no longer all-or-nothing. A profile carries a `maturity` orthogonal to its lifecycle
`status`: `none | provisional | probation | production | retired`. Selection walks a hierarchical
fallback — **league → sport/archetype → league-market → global → static** — and a lower rung lowers
trust. `provisional`/`probation` profiles apply **small, capped** probability corrections and **cap
confidence below `A`**, so a sparse market can leave permanent research mode with a thin profile
instead of waiting forever for a full `production` fit.

## Per-recommendation honesty fields (always shown)

Every recommendation — in **either** output mode — must surface an honesty block. These are
truth-in-labeling, **not** protected betting numbers, so they are permitted even in
`RESEARCH_CANDIDATE`:

- confidence tier **and** the cap reason (why it is not higher);
- trace quality score (`aggregate_quality`, 0–100) and band;
- evidence mode, evidence status, number of evidence signals, applied factor;
- calibration path, profile id, profile status/maturity, profile sample size, profile ECE/Brier;
- whether the `static_identity` fallback was used (no real calibration applied).

`A` confidence is reserved: it requires a real `production`-maturity profile with a passing ECE and
enough samples, a strong trace (`aggregate_quality ≥ 75`), provided context, complete identity, and
sufficient iterations. **Iteration count alone never earns an `A`.** Zero evidence with empty/baseline
context can never produce actionable `A/B/C` output — it is forced to `Pass`.

## Zero-evidence is not harmless — the session blocker

A trace with **no structured evidence AND no provided context** (`zero_evidence_empty_context`)
cannot calibrate, cannot learn, and must not produce actionable output. The run/report layer
surfaces these explicitly: when a session contains **more than 10**, the run summary is **failed**
with a diagnostic (the session-report renderer shows a `⛔ BLOCKER` section and
`render_session_report` exits non-zero). Do not paper over a blind session as "no edges found".

---

## `RESEARCH_CANDIDATE` — permitted vs forbidden

These rules apply **per market**: when a market (game or prop) is `RESEARCH_CANDIDATE`, its
user-facing output is restricted as below, independently of the other market's mode.

**Permitted in the user-facing reply:**

- Qualitative matchup narrative, news synthesis, recent form.
- Listed sportsbook lines from a cited source.
- Research-only lean / missing-data watchlist labels (no protected numbers).
- The **honesty block** (trace quality score, confidence cap reason, evidence mode/status,
  calibration path + profile maturity, `static_identity` flag) — truth-in-labeling, not protected
  numbers.
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
