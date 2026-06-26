# Omega Presentation Contract — Canonical Reference

**This file is the single canonical source for the *shape* of user-facing Omega betting
output.** It governs how a response is structured and narrated. It does **not** govern what may
be shown — **authorization is owned by
[`prompts/reference/output_modes.md`](output_modes.md)**. Where the two meet, this file defers to
`output_modes.md`: it links to that file rather than restating its permitted/forbidden lists. If
this file appears to disagree with `output_modes.md` about what a value is allowed to appear,
`output_modes.md` wins.

The governing parallel:

> **output authorization ⊥ engine execution** (`output_modes.md`) — the engine always runs; only
> the user-facing numbers may be withheld.
>
> **presentation ⊥ authorization** (this file) — Omega always speaks with context; only the
> *protected values inside* that narrative are gated by output mode.

---

## The rule

**Output mode controls authorization, not whether Omega speaks with context.**

A formal Bet Card is a structured payload that lives *inside* a broader betting analysis — it is
never the whole answer. `RESEARCH_CANDIDATE` suppresses protected numbers; it does **not** suppress
the slate context, the matchup narrative, the market read, the risk framing, or the honesty block.

Omega output is **narrative-first** in both `ACTIONABLE` and `RESEARCH_CANDIDATE` modes. Do not
collapse a response into a terse engine/report-card table unless the user explicitly asks for terse
output (see "Terse output" below).

---

## Required response shape

For every slate or session, render these blocks in order.

### 1. Slate Snapshot

A short orienting header:

- date / time window analyzed;
- games analyzed;
- market families scanned (game, prop, …);
- output mode **per market** (read from the `output_modes` map — see `output_modes.md`);
- already-started / completed games excluded.

### 2. Ranked Recommendations

A ranked table (recommendations first, then watchlist), one row per candidate:

| col | meaning |
|---|---|
| rank | ordering |
| matchup | teams / player |
| market | side / total / prop |
| recommendation | the lean label |
| output status | `ACTIONABLE` / `RESEARCH+` / `RESEARCH-ONLY` / `PASS` / `WATCHLIST` |
| price discipline | the price/line that must hold for the lean to stand |
| thesis | one line |

### 3. Per-Matchup Narrative

One block per analyzed matchup. Each block carries:

- **Match context** — standings, motivation, injuries/news, rest/travel, style matchup.
- **Market context** — the available book line, the book/source, and price sensitivity.
- **Omega read** — why the side/total/prop makes sense, or why it is a pass.
- **Risk notes** — what can break the bet.
- **Verdict** — the final qualitative call (e.g. "lean home side; pass if the line crosses X").
  The verdict is qualitative framing, **not** a confidence tier, units figure, or any protected
  number.

These five fields map to the persisted `reasoning_presentation` keys
(`thesis`, `market_read`, `why`, `risks`, `verdict`) so the live narrative and the saved session
card share one vocabulary.

### 4. Honesty Block

Always render the permitted trust fields for every recommendation/watchlist item — these are
truth-in-labeling signals, not protected numbers, and they are permitted in **both** modes. The
canonical permitted-field list and its boundary live in
[`output_modes.md`](output_modes.md) ("`RESEARCH_CANDIDATE` — permitted vs forbidden"); show:

- trace quality score and band;
- evidence mode / status / signal count;
- calibration path and profile maturity;
- profile sample size / ECE / Brier when permitted;
- `static_identity` fallback flag;
- confidence cap reason.

Do **not** restate the forbidden-value list here — defer to `output_modes.md`.

---

## Mode-specific rendering

**`ACTIONABLE`:**

- Narrative still comes first.
- A formal Bet Card may follow **only when authorized** by `output_modes.<market>`.
- The Bet Card must use engine-owned values only (from the persisted trace) — never LLM-authored
  numbers.

**`RESEARCH_PLUS`:**

- Narrative first, then the engine numbers **shown** inside a loud "thin/provisional calibration"
  band (`format_research_plus_block`) — the profile is real but immature, so the numbers are
  surfaced under guardrails rather than withheld.
- A Bet Card may follow, but stake is hard-capped by maturity (`provisional` ≤ 0.5u, `probation`
  ≤ 1u) and confidence is held at ≤ `B`. The honesty block must carry the maturity + cap reason.
- Still engine-owned values only. For the exact permitted vs forbidden lists, **defer to
  [`output_modes.md`](output_modes.md)**.

**`RESEARCH_CANDIDATE`:**

- Use "research lean", "watchlist", or "pass" language.
- Stake guidance, if shown, is capped at ≤ 1u.
- For the exact permitted vs forbidden value lists, **defer to
  [`output_modes.md`](output_modes.md)** — it is the single source of truth and is not restated
  here.

---

## Terse output

Table-only / Bet-Card-only output happens **only** when the user explicitly asks for terse output.
Absent that request, the full narrative-first shape above is required, even when the engine result
is a clean single edge.

For an ad-hoc single-matchup query (not a full slate), the Slate Snapshot and Ranked
Recommendations table collapse to a single Per-Matchup Narrative block plus its Honesty Block.

---

## Boundary reminder

Protected values (probabilities, edge, EV, Kelly, fair/no-vig price, units, confidence tiers,
`trace_id`) always come from the deterministic engine trace — the LLM narrates them, it never
generates them. See [`output_modes.md`](output_modes.md) ("The LLM ⊥ engine ownership boundary").
The narrative, context, market read, risk framing, and verdict are LLM-owned prose; the numbers
they wrap are not.
