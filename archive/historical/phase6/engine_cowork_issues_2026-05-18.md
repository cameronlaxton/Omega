> [!NOTE]
> This document is from a legacy phase that has been implemented and merged to `main`. It is retained here for historical reference.

# Engine / Cowork Session Issues — 2026-05-18

**Session:** sess-20260518-mlb1  
**Recorded by:** Omega Cowork agent  
**Scope:** Issues encountered during live MLB analysis run. All findings are against the repo state as of this date.

---

## 1. Python Version Gate Blocks Sandbox Execution

**What happened:** The Linux sandbox (Cowork shell) has Python 3.10. `pyproject.toml` declares `requires-python = ">=3.12"`. Running `pip install -e .` (or any package-level install) fails with:

```
ERROR: Package 'omega' requires a different Python: 3.10.12 not in '>=3.12'
```

**Root cause:** The version gate is enforced at install time by setuptools. Python 3.12 is the intended Omega runtime floor, so a Python 3.10 Cowork shell is the wrong interpreter for formal engine execution.

**Workaround used in the affected session:** Import via `sys.path.insert(0, repo_root)` and install only the two real dependencies (`pydantic`, `numpy`) directly. This bypasses setup.py and can make a single smoke test pass, but it is no longer an accepted Cowork runtime path.

**Resolution:** `OMEGA_COWORK.md` now requires a Python 3.12+ preflight before MCP or direct engine execution:

```bash
python -m pip install -e .[mcp]
omega-cowork-preflight
```

If preflight reports a lower Python version or missing `pydantic`, `numpy`, `mcp`, or Omega package metadata, repair setup first. Do not emit formal Omega numeric outputs until preflight passes.

---

## 2. `resolve_odds.py` — Prop Type Keys Are Undocumented

**What happened:** Calling `resolve_odds.py --prop-type strikeouts` returns:

```json
{ "status": "unavailable", "skipped_reasons": ["no provider market mapping for MLB prop_type='strikeouts'"] }
```

The correct key is `strikeouts_pitched`. There is also `batter_strikeouts` for the opposing side. Neither is documented in `OMEGA_COWORK.md`, `OMEGA_HANDBOOK.md`, or the script's `--help` output.

**Risk:** An agent or operator will silently get `unavailable` and either skip the prop or escalate incorrectly. The failure message is clear but the valid key list is buried in the script source.

**Recommendation:** Add a `--list-prop-types --league MLB` flag to `resolve_odds.py`, or document the full stat key map in `OMEGA_COWORK.md` Section 4. Minimum: add valid examples to `--help`.

---

## 3. `best_bet` in Game Analysis Selects the Negative-EV Side

**What happened:** Every game analysis result has a `best_bet` field. In all runs during this session it pointed to the **losing** side:

- TB/BAL: `best_bet = {"selection": "Baltimore Orioles away", "edge_pct": -15.27, "ev_pct": -33.29}`  
- HOU/MIN: `best_bet` pointed to Minnesota Twins (edge -31.24%)

The `edges` array correctly identifies the positive-edge side in both cases. The `best_bet` field appears to be selecting `edges[0]` (the home team) unconditionally rather than the max-EV element.

**Impact:** Any downstream consumer reading `best_bet` for automated bet placement or card rendering would act on the wrong side. The `edges` array is reliable; `best_bet` is not.

**Reproduction:** Pass any game request where the away team has the edge — the home team will be returned as `best_bet` regardless of EV sign.

**Recommendation:** Fix `best_bet` selection in `service.py` `analyze_game()` to select `max(edges, key=lambda e: e["ev_pct"])` where `ev_pct > 0`, else `None`. Add a unit test: assert `best_bet.team == argmax(edge_pct)` for a known asymmetric case.

---

## 4. `recommended_units` Missing from `EdgeDetail` Objects

**What happened:** The `EdgeDetail` Pydantic schema has a `recommended_units` field (visible in the schema definition). When iterating over `result["edges"]` in Python, `edge["recommended_units"]` raises `KeyError`.

**Observed:** `best_bet` does contain `recommended_units: 0.0` (always zero), but individual `EdgeDetail` rows in the `edges` list do not serialize this field when it is zero or `None`.

**Impact:** Can't read per-edge stake sizing from game results without catching `KeyError` or using `.get()`. The `best_bet` path always returns 0.0, making it useless for stake sizing regardless.

**Recommendation:** Either (a) always serialize `recommended_units` in `EdgeDetail` (default `0.0`), or (b) populate it from `recommend_stake()` in `analyze_game()` the same way `analyze_player_prop()` does. The prop path returns `edge_over` and `edge_under` directly, which is cleaner — consider unifying the two result shapes.

---

## 5. Game Analysis Maps Outright Win Probability Against Run Line Price

**What happened:** When `spread_home_price` is provided (e.g., TB -1.5 at +140), the engine's `edges` array for the home side uses:

- `true_prob` = outright home win probability (from Poisson simulation, e.g., 55.3%)
- `market_implied` = implied probability from `spread_home_price` (+140 → 41.67%)
- `edge_pct` = difference = +13.63%

This is methodologically incorrect. The true probability of **covering -1.5** (winning by 2+ runs) is materially lower than the outright win probability. For a Poisson game with λ_home=4.2 and λ_away=3.3, P(home wins) ≈ 55% but P(home wins by 2+) ≈ 35–40%. Mapping 55% against a run-line implied of 41.67% overstates the edge by roughly 13–18 percentage points.

**Impact:** Run line edge figures from game analysis are unreliable and should not drive Bet Cards until this is fixed. The moneyline edge figures are correct (win prob vs ML implied prob).

**Recommendation:** In `analyze_game()`, compute a separate run-line coverage probability from the simulated score distribution: `P(home_scores - away_scores > spread_home)`. This is already available from the `home_scores` and `away_scores` arrays produced by the Poisson simulation. Add `spread_coverage_prob` to `EdgeDetail` and use it (not `home_win_prob`) when calculating run-line edge.

---

## 6. `kind` Field Not Auto-Detected in Trace Export

**Minor:** When building the trace payload for export, the `kind` field (`"prop"` vs `"game"`) cannot be reliably inferred from the result dict because neither the prop result nor the game result exposes the original request type at the top level of the trace return value. The `input_snapshot` field was often empty (`{}`). Had to infer from presence of `prop_type` key in the result — fragile.

**Recommendation:** Have `analyze()` return a `kind` key at the top level of the trace dict alongside `trace_id`, `session_id`, etc.

---

## Summary Table

| # | Severity | Component | Issue |
|---|----------|-----------|-------|
| 1 | Low | pyproject.toml | Python 3.11 gate blocks sandbox; runtime is 3.10-compatible |
| 2 | Medium | resolve_odds.py | Prop stat keys undocumented; silent unavailable on wrong key |
| 3 | **High** | service.py `analyze_game` | `best_bet` always selects home team regardless of EV sign |
| 4 | Medium | service.py `analyze_game` | `recommended_units` missing / always 0.0 in EdgeDetail |
| 5 | **High** | service.py `analyze_game` | Run-line edge uses outright win prob, not spread coverage prob |
| 6 | Low | service.py `analyze` | `kind` not returned in trace top-level; input_snapshot often empty |

