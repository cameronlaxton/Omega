# Anchor Parlay Strategy — Knowledgebase

## What It Is

An "anchor bet" is a player prop with a very high empirical probability of hitting — typically 70%+ based on the player's recent game performance. Sportsbooks price these legs at low payouts (~1.2–1.5x) precisely because they're likely to win. Individually, they're not worth betting.

The **anchor parlay** strategy combines 2–4 anchor bets from the same game into a single same-game parlay (SGP). The combined odds reach ~2.0–2.5x, while the combined probability stays meaningfully above the breakeven threshold, creating a positive expected value bet.

## Why It Works

### The Book's Pricing Model vs. Recent Form

Sportsbooks set player prop lines based on season-long averages, matchup models, and market balance. These lines are slow to adjust when a player enters a hot streak or changes role mid-season. By focusing on the **last 5–10 games**, the bettor exploits the lag between a player's current production level and the book's line.

### The Math

| Metric | Single Anchor | 3-Leg Parlay |
|--------|--------------|--------------|
| Win probability | ~75% | ~42% (0.75^3) |
| Decimal odds offered | ~1.30 | ~2.20 (1.30^3) |
| Implied probability (from odds) | ~77% | ~45% |
| Edge | ~-2% (negative individually) | ~-3% to +5% (depends on correlation) |

Wait — so individual legs are slightly *negative* EV? Yes. The magic is in how sportsbooks price parlays:

1. **Vig stacking**: Books apply vig to each leg independently, then multiply. But correlation between legs means the actual combined probability can differ from the naive product.
2. **Correlation mispricing in SGPs**: Same-game parlays are priced using models that often underestimate the *independence* of certain leg combinations (e.g., Player A's rebounds have near-zero correlation with Player B's assists). The book's SGP pricing sometimes gives better combined odds than the true independent product warrants.
3. **Threshold selection**: Picking the *lowest reasonable threshold* (where the player hits 70–90% of the time) maximizes hit rate while still getting offered as a priced leg.

### When the Edge is Real

The edge is real when:
- The player's recent form genuinely reflects their current production level (not a 2-game outlier)
- The legs are genuinely independent (different players, different stats)
- The book's SGP model overestimates correlation (gives better odds than warranted)
- The hit rate sample is meaningful (minimum 5+ games, ideally 10+)

The edge is illusory when:
- Legs are correlated (same player's points + PRA, same team's assists driven by pace)
- Recent form is driven by a blowout or OT game that inflated stats
- The player's minutes are about to change (rotation change, blowout risk, injury)
- Sample size is too small to distinguish signal from noise

## Stat Categories and Thresholds

### Points
| Threshold | Typical Player Profile |
|-----------|----------------------|
| 10+ | Starter-level scorer, role players who consistently score |
| 15+ | Secondary scorers, consistent starters |
| 20+ | Primary/secondary scorers, All-Star level |
| 25+ | Star scorers in good matchups |
| 30+ | Elite scorers on hot streaks — riskier, smaller sample |

### Rebounds
| Threshold | Typical Player Profile |
|-----------|----------------------|
| 3+ | Guards who crash boards, small forwards |
| 5+ | Power forwards, active centers, rebounding wings |
| 7+ | Starting bigs, double-double candidates |
| 10+ | Elite rebounders (rare anchor — high variance) |

### Assists
| Threshold | Typical Player Profile |
|-----------|----------------------|
| 3+ | Secondary playmakers, wings who facilitate |
| 5+ | Point guards, primary ball handlers |
| 7+ | High-usage playmakers, pass-first guards |
| 10+ | Elite facilitators on good passing teams |

### Three-Pointers Made
| Threshold | Typical Player Profile |
|-----------|----------------------|
| 1+ | Any decent 3PT shooter — very high hit rate |
| 2+ | Volume shooters, 6+ attempts per game |
| 3+ | Elite shooters on hot streaks |
| 4+ | Rare — only during heater stretches |

### Other Props
| Stat | Thresholds | Notes |
|------|-----------|-------|
| Steals | 1+, 2+ | High variance, harder to predict |
| Blocks | 1+, 2+ | Matchup dependent, moderate variance |
| PRA (Pts+Reb+Ast) | 15+, 20+, 25+, 30+ | Correlated with individual stats — avoid combining with pts/reb/ast legs |

## Parlay Construction Rules

### Leg Count
- **2 legs**: Safest, lowest payout (~2.0x). Good for conservative bankroll management.
- **3 legs**: Sweet spot. Combined odds ~2.1–2.5x with ~40–50% combined hit rate.
- **4 legs**: Higher payout (~2.5–3.5x) but combined probability drops below 35%. Use sparingly.
- **5+ legs**: Not recommended. Combined probability drops too low even with strong anchors.

### Same-Game Requirement
All legs must come from the same game. This is required by sportsbook SGP rules, and also ensures all legs share the same game state (no early cancellation risk across games).

### Correlation Awareness

**Safe combinations** (low correlation):
- Player A's points + Player B's rebounds (different players, different stats)
- Player A's assists + Player B's points (complementary — A assists on B's baskets)
- Guard's assists + Center's rebounds (positionally independent)

**Risky combinations** (correlated — avoid):
- Same player's points + PRA (PRA includes points — near-guaranteed correlation)
- Same player's points + assists (high-usage players do both)
- Two players on same team, same stat (team pace affects both)
- Opposite team's stats in blowout-sensitive stats (if one team dominates, other team's stats suffer)

### Target Odds Range
- Minimum: 1.80x decimal (below this, risk/reward isn't worth the parlay structure)
- Sweet spot: 2.00–2.50x decimal
- Maximum: 3.00x decimal (above this, combined probability is usually too low)

## Risk Factors

### Blowout Risk
If a game becomes a blowout, starters get pulled in the 4th quarter. This is the #1 killer of anchor parlays. A player on pace for 25 points sits with 18 because the game was decided by halftime.

**Mitigation**: Prefer games projected to be competitive (spread < 7 points). Avoid stars on heavy favorites.

### Minutes Uncertainty
Back-to-backs, playoff rest, minor injuries, coach decisions. A player who averaged 34 minutes but only plays 24 will likely miss stat thresholds.

**Mitigation**: Check injury reports, rest day patterns, and coach tendencies before scanning.

### Pace and Game Script
A fast-paced game inflates all counting stats. A slow, grinding game suppresses them. The last 10 games may have included varied paces.

**Mitigation**: Weight recent games against similar opponents/pace. Flag games with extreme pace outliers in the lookback window.

### Sample Size
5–10 games is a very small sample. A player who went 8/10 at a threshold has a 95% confidence interval of roughly [0.49, 0.96]. The "true" hit rate could be anywhere in that range.

**Mitigation**: Prefer 10-game samples over 5. Weight more recent games slightly higher. Be skeptical of players who *just barely* clear 70% (7/10 is borderline; 9/10 is strong).

## Bankroll Management

### Stake Sizing
- Flat stake: simplest approach. Bet the same amount on every parlay.
- Fractional Kelly: bet a fraction (usually 25–50%) of the Kelly-optimal stake to reduce variance.
- The user's actual approach: stakes of $25–$234, suggesting confidence-weighted sizing.

### Expected Performance
With a disciplined approach (70%+ hit rate anchors, 2–3 leg parlays, ~2.0–2.5x odds):
- Expected parlay win rate: 35–50%
- Expected ROI per bet: 5–15% (theoretical, before vig adjustments)
- Variance is high — expect losing streaks of 3–5 even with a genuine edge
- Long-term edge comes from volume and discipline, not any single bet

## Worked Example (from user's real bets)

### 4/8/2026 — OKC @ LAC, 4-leg parlay

| Leg | Threshold | Odds (implied) | Hit? |
|-----|-----------|----------------|------|
| Chet Holmgren 10+ pts | ~85% hit rate | ~1.25x | Yes |
| SGA 5+ ast | ~80% hit rate | ~1.30x | Yes |
| Kris Dunn 3+ ast | ~75% hit rate | ~1.35x | Yes |
| Brook Lopez 3+ reb | ~85% hit rate | ~1.25x | Yes |

Combined decimal odds: 2.20x (SGP pricing, not straight multiplication)
Stake: $75 → Paid: $165 (2.20x return)
Combined empirical probability: ~0.85 × 0.80 × 0.75 × 0.85 ≈ 43%
Breakeven probability at 2.20x: 45%
Edge: ~-2% (marginal) to slight positive (if book overpriced correlation)

### 4/5/2026 — HOU @ GSW, 3-leg parlay

| Leg | Threshold | Odds (implied) | Hit? |
|-----|-----------|----------------|------|
| Kevin Durant 20+ pts | ~80% hit rate | ~1.40x | Yes |
| Alperen Sengun 15+ pts | ~70% hit rate | ~1.45x | Yes |
| Kevin Durant 5+ ast | ~70% hit rate | ~1.40x | Yes |

Combined decimal odds: 2.55x
Stake: $234 → Paid: $596.70
Combined empirical probability: ~0.80 × 0.70 × 0.70 ≈ 39%
Breakeven probability at 2.55x: 39%

**Note**: Two KD legs are correlated (his scoring and playmaking both depend on usage/minutes). This is a risk the user accepted — the payout reflected it.

## Glossary

- **Anchor bet**: A single player prop leg with ≥70% empirical hit rate over last 5–10 games
- **SGP**: Same-game parlay — all legs from a single game
- **Hit rate**: Fraction of recent games where player exceeded the threshold
- **Implied probability**: 1 / decimal_odds — what the book "thinks" the probability is
- **Edge**: empirical_probability - implied_probability
- **EV (Expected Value)**: (win_prob × payout) - (loss_prob × stake)
- **Vig/Juice**: The book's margin built into the odds
- **CLV (Closing Line Value)**: Whether the line moved in your favor after bet placement — a marker of sharp betting
- **Kelly criterion**: Optimal stake sizing formula: f = (bp - q) / b, where b = decimal_odds - 1, p = win_prob, q = 1 - p
