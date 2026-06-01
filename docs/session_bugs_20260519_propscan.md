# Session Bug Report — 2026-05-19 (Cowork prop scan)

Session: `sess-20260519-nyk1` — NYK vs CLE ECF Game 1 prop scan (Harden / KAT / Hart), BetMGM, 12 traces ingested.

---

## BUG-A: Cowork VM ships Python 3.10; preflight requires 3.12+

**Severity:** High (hard wall — blocks all engine execution out-of-the-box)
**Where:** `omega-cowork-preflight` enforces `>=3.12`; the Cowork sandbox `/usr/bin/python` is 3.10.12.
**Symptom:**
```
cowork_preflight_failed:
- Python 3.12+ is required. Current interpreter is 3.10.12 at /usr/bin/python.
```
**Workaround used this session:** installed via `uv python install 3.12`, created a venv outside the OneDrive-mounted repo (`/tmp/venv-omega`), invoked engine with `PYTHONPATH=$PWD`.
**Why outside repo:** `uv pip install -e .[mcp]` inside `.venv-omega` placed in the OneDrive mount failed with `Operation not permitted` deleting `__editable__.omega-0.2.0.pth` — the OneDrive bridge does not support the file-replace semantics uv needs. Reproducible.
**Fix options:**
1. Ship Python 3.12 in the Cowork sandbox image (preferred).
2. Add a `scripts/bootstrap_cowork.sh` that runs `uv python install 3.12 && uv venv /tmp/venv-omega && uv pip install -e .[mcp]` so the agent doesn't have to rediscover this each session.
3. Loosen `cowork_preflight.py` to 3.11+ if engine actually works on 3.11 (untested).

---

## BUG-B: `.env` has CRLF line endings; sourcing fails silently with `$'\r': command not found`

**Severity:** Low (cosmetic) but blocks naive `source .env`
**Symptom:** `bash: .env: line 3: $'\r': command not found`
**Workaround:** `tr -d '\r' < .env > /tmp/env.clean && source /tmp/env.clean`
**Fix:** Normalize `.env` to LF on commit (`.gitattributes`: `.env text eol=lf`) or have `cowork_preflight.py` detect and warn.

---

## BUG-C: `PlayerPropResponse` omits projected mean

**Severity:** Medium (audit / explainability gap)
**Where:** `omega/core/contracts/schemas.py::PlayerPropResponse` and `omega/core/contracts/service.py::analyze_player_prop`.
**Symptom:** Response carries `over_prob`, `under_prob`, `edge_over`, `edge_under`, `recommendation`, `confidence_tier` but **no `projection_mean`** or `projection_std`. The simulated distribution's central tendency is recoverable only by reading back `input_snapshot.player_context.{stat}_mean`, which is the *input*, not the engine's modeled output.
**Impact:** Anyone consuming the trace for a "projected vs line" view has to assume input mean ≈ projected mean (true for the current passthrough model, but not future-proof and not explicit). Boxscore renders are forced to read inputs.
**Fix:** Add `projection_mean: float | None`, `projection_std: float | None`, and `projection_p50: float | None` to `PlayerPropResponse`; populate from `sim_result`. Persist in trace.

---

## BUG-D: Props pipeline has no Kelly / recommended_units output

**Severity:** Medium-High (violates ownership boundary in `CLAUDE.md`)
**Where:** `analyze_player_prop` returns no `kelly_fraction` or `recommended_units`. `BetSlip` is built only by `_pick_best_bet` for game lines.
**Impact:** Per `CLAUDE.md` hard rule, Kelly fractions and recommended units are deterministic-engine responsibilities. For props, the engine declines that responsibility entirely, forcing the LLM to either skip staking or compute it externally — both bad.
**Fix:** Mirror game's `_pick_best_bet` / `recommend_stake` for prop responses: when `confidence_tier in {A,B}` and `recommendation != "pass"`, compute `kelly_fraction` and `recommended_units` on the chosen side and attach to `PlayerPropResponse`. Extend trace schema accordingly.

---

## BUG-E: Confidence tier inflation — every prop in this scan returned tier A

**Severity:** Medium (calibration / discipline risk)
**Where:** `_compute_edge_detail` uses `tier = "A" if n_iterations >= 1000 else "B"` and only downgrades to "Pass" when `abs(edge) < 3.0`. Same logic effectively reaches the prop tier through `run_player_simulation`.
**Symptom:** 12/12 props in this scan returned tier A on shallow inputs (season-average mean only, default std = max(1.0, mean*0.25), no opponent / minutes / DvP / lineup adjustments).
**Impact:** "A" loses signal. A user following tier blindly stakes the same on Harden AST O6.5 (proj 7.7 vs 6.5, +17.86% edge) as on Hart PTS U12.5 (proj 12.0 vs 12.5, -11.36% edge with only 4.43% under edge). Real discipline requires gating tier A on **input-quality**, not just iteration count + raw edge magnitude.
**Fix:** Tie tier ceiling to `imputed_fraction`, presence of opponent context (`game_context`), minutes projection, and std source. Without contextual covariates, cap at B. The schema already carries `imputed_fraction` and notes hooks (`tier_capped_imputation`) — wire them in.

---

## BUG-F: `player_context` is single-stat scoped per call; no game-wide projection object

**Severity:** Low (design)
**Symptom:** To analyze pts + reb + ast + 3pm for one player I had to call `analyze` 4× with separate `player_context` dicts because the engine only reads `{prop_type}_mean`. There's no `analyze_player_boxscore(player, stats=[pts,reb,ast,3pm])` shortcut.
**Impact:** 4× trace IDs per player for what is conceptually one projection. Inflates trace counts and prevents joint distributions (e.g., conditional `reb | pts > 25`).
**Fix:** Add a `analyze_player_boxscore` entry point or accept a list of `(prop_type, line)` in a single request and return one response with per-prop sub-results sharing one trace_id.

---

## Notes (not bugs)

- `resolve_odds.py` BetMGM path worked first try once `.env` was sourced cleanly; Odds API quota at 19,890 remaining.
- All 12 traces ingested cleanly via `omega-ingest-traces`; no validation rejections.
- Bankroll defaulted to $1000 (demo) because OMEGA_COWORK §2 requires asking before producing a Bet Card and the user hasn't supplied one for this session. **Action required before any real stake:** user confirms bankroll, rerun or scale units.

