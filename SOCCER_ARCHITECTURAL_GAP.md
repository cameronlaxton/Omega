# Soccer Architectural Gap — Readiness Assessment

**Status:** Audit only. No simulation, schema, or grading code was changed to produce
this document. All claims below are cited to source at `file:line`.

## Verdict

Soccer is **not structurally blocked**. The brief's stated blockers — "no draw
support," "primitives optimized exclusively for basketball/baseball," "needs a
bivariate-Poisson rewrite," "needs new 3-way moneyline schema" — **do not match the
code**. Soccer is already wired end-to-end through the `fast_score` backend: a Poisson
soccer archetype, a registered `_sim_soccer` simulator, soccer league configs, 3-way
outcome math, `moneyline_draw`/`draw_prob` schema fields, draw edge analysis, and
`result="draw"` outcome grading all exist today.

The genuine remaining work is **two narrow correctness gaps** (Draw price ingestion;
draw-as-winnable grading) plus optional refinements (Dixon-Coles draw correction;
calibration cold-start; exotic markets). None require new top-level architecture.

### Premise correction

| Brief premise | Repo reality |
|---|---|
| Primitives optimized exclusively for basketball/baseball | Archetype framework spans basketball, baseball, hockey, **soccer**, tennis, golf, fighting, esports, american_football — `omega/core/simulation/archetypes.py:313` |
| Soccer can't map to score structures / no draw support | `SOCCER` archetype: `score_distribution="poisson"`, `supports_draw=True`, 3-way `supported_markets` — `archetypes.py:313-365` |
| Must transition to a bivariate Poisson model | A working soccer scorer already exists: `_sim_soccer` — `omega/core/simulation/engine.py:775`; registered as `_ARCHETYPE_SIMULATORS["soccer"]` — `engine.py:1081`. It is *independent* Poisson, which is a refinement target (§Gap 3), not a blocker |
| Needs DB/schema updates for Home/Away/Draw moneyline | `OddsInput.moneyline_draw` — `omega/core/contracts/schemas.py:73`; `SimulationResult.draw_prob` — `schemas.py:253`; engine emits `draw_prob` — `engine.py:306`; outcomes store `result="draw"` — `omega/trace/store.py:978` |

---

## Verified-ready layers

| Layer | Evidence |
|---|---|
| **Archetype** | `SOCCER = SportArchetype(score_distribution="poisson", supports_draw=True, result_type="team_score", required_team_keys=("off_rating","def_rating"), supported_markets=("moneyline_3way","spread","total","double_chance","draw_no_bet","both_teams_to_score",...), avg_total=2.5, default_std=1.3)` — `archetypes.py:313-365` |
| **Simulator (registered)** | `_sim_soccer(home_ctx, away_ctx, league, n_iter, config)` derives `home/away` λ from `off_rating`/`def_rating` (preferring `xg_for`/`xg_against`), applies a home-advantage split, and draws independent Poisson goals — `engine.py:775-805`. Dispatched via `_ARCHETYPE_SIMULATORS["soccer"]` — `engine.py:1081`; the `fast_score` backend looks this up at `engine.py:1169` |
| **League configs** | `MLS`, `EPL`, and further soccer leagues with `sport/archetype="soccer"`, `distribution="poisson"`, `home_advantage`, `std` — `omega/core/config/leagues.py:185+` |
| **3-way outcome math** | `supports_draw` honored at `engine.py:283`; non-draw sports redistribute ties (`_allocate_ties`) and report `draw_prob=0` (`engine.py:288-292`); soccer reports real `draw_prob` at `engine.py:306`. Backend contract *requires* `draw_prob` in every success result — `omega/core/simulation/backends.py:23` |
| **3-way odds schema** | `OddsInput.moneyline_draw` — `schemas.py:73`; `SimulationResult.draw_prob` — `schemas.py:253` |
| **Draw edge analysis** | 3-way probabilities renormalized to sum to 1 (`service.py:1002-1006`); a Draw edge is built (calibrated, with Kelly/EV) when `moneyline_draw` is present — `omega/core/contracts/service.py:1158-1168` |
| **Draw outcome grading** | Tied final score records `result="draw"` in the `outcomes` table — `store.py:973-978` |
| **No league gating** | No `SUPPORTED_LEAGUES` allowlist rejects soccer at the analyze/contract boundary (verified absent in `service.py`) |

**Net:** given valid `off_rating`/`def_rating` (or xG) team context and a `moneyline_draw`
price, a soccer game already flows: simulate → `draw_prob` → renormalize → draw edge →
Bet Card; and a tied result already grades as `draw`.

---

## Genuine remaining gaps

Ranked. Gaps 1–2 block a *correct* first soccer bet; 3–5 are refinements/cold-start.

### Gap 1 — 3-way `h2h` Draw price is dropped on ingestion (BLOCKER for soccer ML)
The provider's soccer `h2h` returns three outcomes (Home/Draw/Away).
`normalize_book_odds` maps `h2h → "moneyline"` and **preserves the selection name**
(so a `{market_type:"moneyline", selection:"Draw"}` quote survives normalization) —
`omega/integrations/odds_resolver.py:95-96, 123`. But `_select_game_input` only has
branches for `moneyline`+home, `moneyline`+away, `spread`+home, and `total`+over —
`odds_resolver.py:228-237`. **The Draw quote is silently discarded** and never reaches
`OddsInput.moneyline_draw`, so the draw edge at `service.py:1158` never fires from real
market data.
**Fix scope:** add a `selection == _norm("Draw")` branch in `_select_game_input` →
`selected["moneyline_draw"] = quote["price"]`. Localized, ~2 lines.

### Gap 2 — Bet grading treats a draw as a push, not a winnable selection (BLOCKER)
The benchmark grader scores moneyline by score comparison and classifies a tie as a
**push** — `omega/strategy/backtest/engine.py:350-357` (`push = home_score == away_score`).
Evidence scoring explicitly **skips draws** — `omega/strategy/signal_performance.py:133`.
The `outcomes` table *can* store `result="draw"` (`store.py:978`), but no grading path
treats a Draw moneyline selection as a winner. A Draw bet would never be graded a win
and would silently corrupt CLV/calibration if logged.
**Fix scope:** make moneyline grading selection-aware (a `draw` selection wins on
`home_score == away_score`); stop blanket-skipping draws where a draw is the bet. Confined
to the backtest grader and signal-performance scorer; no schema change required.

### Gap 3 — Independent vs. bivariate Poisson (refinement, not a blocker)
`_sim_soccer` samples home and away goals **independently** (`engine.py:803-804`). This
slightly understates draw mass and ignores low-score correlation. A Dixon-Coles low-score
correction (τ adjustment on the 0-0/1-0/0-1/1-1 cells) or a bivariate-Poisson shared
component would improve draw calibration. It changes *accuracy*, not *function* — the
model already produces valid, normalized 3-way probabilities. Deterministic-engine work;
must stay inside `omega/core/simulation/*` and remain seed-reproducible.

### Gap 4 — No soccer calibration profiles / graded traces (ordinary cold-start)
No soccer calibration profiles exist because no graded soccer traces exist yet. This is
the same cold-start every league passes through; isotonic/shrinkage fits become eligible
once graded soccer traces accumulate (standard `fit_calibration.py` flow). Not structural.

### Gap 5 — Exotic soccer markets unimplemented beyond 3-way moneyline
`double_chance`, `draw_no_bet`, `both_teams_to_score`, `correct_score`, `1h_*` are
declared in the archetype's `supported_markets` (`archetypes.py:347-358`) but have no
edge/grading path. The `outcomes` table carries a single `result` field plus
`home_score`/`away_score`, which is sufficient to *derive* these post hoc, but no code
maps a bet on them to a graded result today. Out of scope for a first soccer cut.

---

## Minimum path to a correct first soccer bet

1. Close **Gap 1** (Draw price → `moneyline_draw` in `_select_game_input`).
2. Close **Gap 2** (selection-aware draw grading in the backtest grader / signal scorer).
3. Accumulate graded soccer traces, then fit calibration (Gap 4, normal flow).

Gaps 3 and 5 are deferrable quality/coverage work, not prerequisites. No new packages,
services, or schema migrations are required for a first 3-way moneyline soccer bet.

---

## Resolution — implemented 2026-05-29

All five gaps are now closed. Branch: `soccer-gap-closure`. Full suite green (921 passed).

| Gap | Status | Key changes |
|---|---|---|
| 1 — Draw price ingestion | ✅ Closed | Draw branch in `_select_game_input` → `OddsInput.moneyline_draw` (`omega/integrations/odds_resolver.py`) |
| 2 — Draw-as-winnable grading | ✅ Closed | Draw side in backtest grader + selection-aware `_grade_selection` (`omega/strategy/backtest/engine.py`); `realized_game_direction` returns `"draw"`, `"draw"` added to `_GAME_DIRECTIONS` (`omega/strategy/signal_performance.py`) |
| 3 — Dixon-Coles correction | ✅ Closed | Seed-stable `_dixon_coles_scores` joint-pmf sampler in `_sim_soccer`, opt-in per league via `dixon_coles`/`rho` config (`omega/core/simulation/engine.py`, `omega/core/config/leagues.py`) |
| 4 — Draw calibration plane | ✅ Closed | `market` dimension on `CalibrationProfile` + registry selection (game-profile fallback); `extract_draw_pairs`; `fit_calibration.py --plane draw` (`omega/core/calibration/*`, `scripts/fit_calibration.py`) |
| 5 — Exotic markets | ✅ Closed | double_chance / draw_no_bet / both_teams_to_score / correct_score: probabilities in `_build_team_score_result` (gated on `supports_draw`), additive `OddsInput` fields, provider ingestion in `_select_exotic_quote`, edge construction in `service.analyze`, grading in `_grade_selection` |

Cold-start note: no soccer calibration profiles exist until graded soccer traces accumulate; draw and exotic probabilities reuse the league `game` calibration profile until then (Gap 4 fallback). This is expected, not a regression.
