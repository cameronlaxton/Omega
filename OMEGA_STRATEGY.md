# OMEGA_STRATEGY — NBA Anchor Parlay Playbook

The user's standing strategy preference. When the Project Claude builds parlay recommendations without other guidance, it follows this playbook. Edits to this file should propagate to the Claude.ai Project knowledge.

## Strategy at a glance

The user runs an **NBA anchor parlay** strategy:

- Combine 2–4 high-hit-rate player prop legs into a single parlay within one game (or across one slate).
- Each leg is a prop the player has cleared in **70%+ of the most recent 5–10 games**.
- Target combined parlay odds in the **+100 to +150 range** (roughly 2.0x – 2.5x payout).
- Bet sizing is unit-based; the user manages bankroll outside Omega.

The intent is *not* maximum theoretical edge per leg — it is **stacking high-probability legs that, when multiplied, still pay enough to be worth the variance**.

## Anchor selection criteria

An "anchor" leg is the highest-confidence prop in the parlay. To qualify as anchor-grade:

1. **Hit rate** ≥ 70% over the last 5 games AND ≥ 65% over the last 10 games (avoid one-week mirage).
2. **Threshold buffer** — player's median over the lookback window must clear the prop line by at least one full unit (e.g. line `pts 17.5`, median 19+).
3. **Opponent context** — opponent defense must be neutral-or-worse against the relevant stat. Avoid anchoring a points prop against the league's #1 defense at that position.
4. **Volume stability** — minutes-per-game variance < 15% over the lookback window. A blowout-prone team's bench player is not an anchor.
5. **No active injury / status downgrade.** Probable or worse → not anchor-grade.

## Standard parlay shape

- **2x parlay (most common)** — anchor + one secondary leg with hit rate ≥ 65%.
- **3x parlay (occasional)** — anchor + two secondary legs; combined implied prob ≥ 50%.
- **4x parlay (rare; only when slate is unusually clean)** — anchor + three; combined implied prob ≥ 40%.

Standard prop categories the user plays in:

- Points: thresholds at 10+, 15+, 20+.
- Rebounds: thresholds at 3+, 5+, 7+.
- Assists: thresholds at 3+, 5+, 7+.
- 3-pointers made: thresholds at 2+, 3+.

## Joint-probability math (default assumption)

Unless told otherwise, treat legs as **independent**. Joint probability of an N-leg parlay = `prod(leg_prob_i)`.

Implied probability from American odds:
- Positive odds: `imp = 100 / (odds + 100)`.
- Negative odds: `imp = -odds / (-odds + 100)`.

Parlay edge = `joint_prob − market_implied_parlay_prob`. Surface this as a top-line number alongside the per-leg edges so the user can see whether multiplying small edges is paying or eroding.

## When correlation actually matters

Same-game correlation is non-trivial for some combinations. Apply a haircut to the independence estimate when:

- **Star scoring + team total over** — positively correlated. Haircut on the *favorable* side: reduce the implied joint by 5–10 pp.
- **Star points + own assists** — usage-correlated; usually positive, but if the user is playing under-pts + over-assists (or vice versa) the legs are *negatively* correlated; flag and warn.
- **Two players on the same team for the same stat** — minutes-share competition. Negative correlation on overs.
- **Player rebounds + team total over** — weakly positive.

If the user supplies a correlation estimate, use it verbatim. Otherwise, when applying a haircut, label the parlay as "correlation-adjusted, point estimate" and show both the naive and adjusted numbers.

## Skip rules (refuse the parlay)

Refuse to build a recommended parlay when any of these apply:

- No anchor candidate qualifies (lookback hit rate < 70%).
- The Omega quality gate dropped `BET_CARD` for the matchup → respect the gate; don't reanimate edges.
- The parlay's joint probability is below 35% — the user's strategy is *high-hit-rate*, not lottery tickets.
- The combined odds are shorter than +80 — payoff doesn't justify variance.
- Lines look unusually sharp (book holds < 4%) — either the market knows something you don't, or the line is mid-move.

Surface the skip reason explicitly. Do not silently downgrade to a 2-leg parlay when the user asked for a 3-leg; tell them why.

## Bet Card format for parlays

When a parlay is the deliverable, the Bet Card table gets one row per leg plus one summary row:

| Selection | Odds | Edge% | EV% | Tier | Units | Kelly | Source |
|---|---|---|---|---|---|---|---|
| Leg 1 (anchor) | -135 | +6.2% | … | A | 1.0u | 4.0% | Omega run `trace_id` |
| Leg 2 | +110 | +4.5% | … | B | 0.5u | 2.5% | Omega run `trace_id` |
| **Parlay (2x)** | **+185** | **+8.1%** | **…** | **B** | **0.5u** | **2.0%** | **Combined; correlation-adjusted** |

The summary row carries the parlay-level edge and stake recommendation; per-leg rows carry the single-leg view.

## Out of scope for this playbook

- NFL, MLB, NHL, soccer, MMA, golf — strategy is NBA-specific. For other leagues, ask the user before assuming this shape.
- Live in-play parlays — the strategy is for pre-game props only. Lines move too fast in-game for the lookback math to mean what it claims.
- Same-game-correlation modeling beyond the simple haircut table above — that's an Omega upgrade, not a Project-layer responsibility.
