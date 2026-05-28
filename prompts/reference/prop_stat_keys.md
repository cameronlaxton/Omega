# Prop Type → Stat Key Reference

Use this when calling `scripts/resolve_odds.py --kind prop` or supplying `prop_type` in an `analyze()` request. This documents the correct key values so you don't hit the silent `status: "unavailable"` from a wrong key (Issue #2 from `docs/phase6/engine_cowork_issues_2026-05-18.md`).

Source of truth: `omega/integrations/espn_boxscore.py` (NBA_STAT_KEYS, MLB_BATTING_KEYS, MLB_PITCHING_KEYS).

---

## NBA Prop Types

| prop_type (use this in analyze + resolve_odds) | ESPN stat key | Notes |
|---|---|---|
| `pts` | PTS | Points; preferred alias |
| `points` | PTS | Alias for `pts` |
| `reb` | REB | Rebounds; preferred alias |
| `rebounds` | REB | Alias for `reb` |
| `ast` | AST | Assists; preferred alias |
| `assists` | AST | Alias for `ast` |
| `stl` | STL | Steals |
| `steals` | STL | Alias for `stl` |
| `blk` | BLK | Blocks |
| `blocks` | BLK | Alias for `blk` |
| `3pm` | 3PTM | Three-pointers made |
| `threes` | 3PTM | Alias for `3pm` |
| `pra` | computed | Points + rebounds + assists (derived; graded by summing pts+reb+ast) |

**Context fields by NBA prop_type:**

| prop_type | required context fields |
|---|---|
| `pts` | `pts_mean`, `pts_std` |
| `reb` | `reb_mean`, `reb_std` |
| `ast` | `ast_mean`, `ast_std` |
| `blk` | `blk_mean`, `blk_std` |
| `stl` | `stl_mean`, `stl_std` |
| `3pm` / `threes` | `threes_mean`, `threes_std` |
| `pra` | `pts_mean`, `pts_std`, `reb_mean`, `reb_std`, `ast_mean`, `ast_std` |

---

## MLB Prop Types

### Batting

| prop_type | ESPN stat key | Notes |
|---|---|---|
| `hits` | H | Hits |
| `runs` | R | Runs scored |
| `rbi` | RBI | Runs batted in; preferred alias |
| `rbis` | RBI | Alias for `rbi` |
| `hr` | HR | Home runs |
| `home_runs` | HR | Alias for `hr` |
| `sb` | SB | Stolen bases |
| `stolen_bases` | SB | Alias for `sb` |
| `bb` | BB | Walks (batter) |
| `walks` | BB | Alias for `bb` |

### Pitching

| prop_type | ESPN stat key | Notes |
|---|---|---|
| `strikeouts_pitched` | K | ✅ Correct key for pitcher strikeouts |
| `strikeouts` | K | Alias — works too, but `strikeouts_pitched` is canonical |
| `k` | K | Short alias |
| `pitching_outs` | IP | Innings pitched (fractional; ESPN IP format) |
| `outs_recorded` | IP | Alias for `pitching_outs` |
| `earned_runs` | ER | Earned runs allowed |
| `er` | ER | Alias for `earned_runs` |
| `hits_allowed` | H | Hits allowed by pitcher |
| `walks_allowed` | BB | Walks issued by pitcher |

> **Common mistake:** Using `strikeouts` or `k` when you mean the pitcher's strikeout total is fine — both keys are supported. The bug in the 2026-05-18 session was using `strikeouts` with a wrong expectation that it maps to `batter_strikeouts`. There is no `batter_strikeouts` key — use `strikeouts` or `strikeouts_pitched` for the **pitcher** prop only.

**Context fields by MLB prop_type:**

| prop_type | required context fields |
|---|---|
| `hits` | `hits_mean`, `hits_std` |
| `hr` | `hr_mean`, `hr_std` |
| `rbi` | `rbi_mean`, `rbi_std` |
| `strikeouts_pitched` | `strikeouts_pitched_mean`, `strikeouts_pitched_std` |
| `earned_runs` | `er_mean`, `er_std` |

---

## Free tier vs. paid tier (The Odds API)

`scripts/resolve_odds.py` uses `OMEGA_ODDS_API_KEY`. Some prop markets are only available on paid tiers:

| Market | Free tier | Paid tier |
|---|---|---|
| Game moneylines | ✅ | ✅ |
| Game spreads | ✅ | ✅ |
| Game totals | ✅ | ✅ |
| Player props (NBA pts, reb, ast) | ❌ | ✅ |
| Player props (MLB hitting/pitching) | ❌ | ✅ |
| Historical closing lines | ❌ | ✅ |

If `resolve_odds.py` returns `status: "unavailable"` for a prop, the cause is either (a) wrong `prop_type` key (check table above) or (b) the free-tier key doesn't include player prop markets.

---

## `check supported_prop_type` (programmatic)

```python
from omega.integrations.espn_boxscore import supported_prop_type

supported_prop_type("NBA", "pts")               # True
supported_prop_type("NBA", "batter_strikeouts") # False — does not exist
supported_prop_type("MLB", "strikeouts_pitched") # True
```
