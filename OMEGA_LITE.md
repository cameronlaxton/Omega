# OMEGA_LITE — Sandbox-Runnable Omega Core

`omega_lite` is a deterministic-engine slice of Omega designed to run inside an LLM analysis tool (Claude.ai analysis tool, ChatGPT code interpreter) when no local Omega server is available. The simulation, calibration, edge calculation, and Kelly staking math is **the same code** as the canonical Omega package — verified bit-identical with a fixed seed (see `tests/omega_lite/test_parity.py`).

The package is what powers **Mode A — sandbox** in the Project instructions. It's the honesty bridge between "real local Omega run" (best) and "LLM-improvised methodology emulation" (worst).

## What's in it (~3,300 lines)

| File | Purpose |
|---|---|
| `omega_lite/odds.py` | American↔decimal, implied probability, edge%, EV% |
| `omega_lite/kelly.py` | Fractional Kelly with confidence-tier scaling |
| `omega_lite/calibration.py` | Shrinkage, cap, isotonic, combined calibration |
| `omega_lite/archetypes.py` | All 9 sport archetypes + league mapping |
| `omega_lite/leagues.py` | Per-league config (pace, totals, home edge, etc.) |
| `omega_lite/validation.py` | Sim-input sanity bounds and coercion |
| `omega_lite/schemas.py` | Pydantic v2 request/response models |
| `omega_lite/engine.py` | Fast-path Monte Carlo for all archetypes |
| `omega_lite/service.py` | `analyze_game`, `analyze_player_prop`, `analyze_slate` |
| `omega_lite/models.py` | Subset of canonical models — what the quality gate needs |
| `omega_lite/_quality_helpers.py` | Aggregate quality + critical-input checks |
| `omega_lite/quality_gate.py` | Plan-level downgrade rules (same as canonical) |
| `omega_lite/run.py` | Sandbox wrapper — `analyze(...)` entry point with trace_id |
| `omega_lite/__init__.py` | Public API |

## What it is NOT

- **No collectors.** Network is unavailable inside omega_lite itself. The caller must supply all inputs in the request payload. If the surrounding LLM has web browsing, it should actively gather public raw inputs when the user asks for an analysis, then run omega_lite only for candidates with sufficient normalized inputs. Candidates that cannot be normalized should still be surfaced as research-only leans or missing-data watchlist items.
- **No FastAPI / SSE / sessions.** This is the math layer, not the conversation layer.
- **No markov_engine.** The canonical repo plans a possession-level Markov simulator; until it lands, `omega_lite` uses the same archetype-aware Poisson/Normal sampler the canonical service uses for player props.
- **Not a long-term replacement for a hosted Omega API.** When you're ready, deploy the FastAPI service and have the Project call it via MCP / custom action.

## Building and refreshing

The package is built by extracting from the canonical `omega/` source tree:

```bash
python scripts/build_omega_lite.py                 # rebuild omega_lite/ in place (idempotent)
python scripts/build_omega_lite.py --single-file   # ALSO emit omega_lite_standalone.py (the sandbox artifact)
python scripts/build_omega_lite.py --zip           # legacy: produce omega_lite-v1.zip (deprecated)
```

Run it whenever `omega/core/` or `omega/reasoning/evaluator.py` changes. The script rewrites all imports to `omega_lite.*` and truncates `engine.py` to drop the two markov-dependent methods.

The **single-file** mode is the supported sandbox distribution. It concatenates `odds.py`, `kelly.py`, `archetypes.py`, `calibration.py`, `engine.py`, `service.py`, `schemas.py`, `quality_gate.py`, `_quality_helpers.py`, `validation.py`, `models.py`, `run.py` into one self-contained `omega_lite_standalone.py` at the repo root. No internal imports; numpy + stdlib only. Upload this file to the Claude.ai Project knowledge panel alongside [`prompts/system_prompt.txt`](prompts/system_prompt.txt). The agent reads it from project knowledge and writes it byte-for-byte to its sandbox cwd at session start — no zip extraction required.

The legacy `--zip` mode is retained only for back-compat with old project knowledge bundles; new sessions should use `--single-file`.

## Parity guarantee

`tests/omega_lite/test_parity.py` asserts that for the same inputs and seed:

- `analyze_game` produces identical win probabilities, predicted spread, predicted total, and per-edge `edge_pct` / `ev_pct` / `confidence_tier` between canonical and lite.
- `analyze_player_prop` produces identical `over_prob`, `under_prob`, `recommendation`, and edges.
- Both refuse on the same missing-critical-input cases with identical `missing_requirements` lists.

If parity breaks, the rebuild has drifted — rerun `build_omega_lite.py` and re-run the tests.

## Sandbox workflow

The intended user flow inside a Claude.ai Project or ChatGPT Project:

1. **One-time setup.** Upload `omega_lite_standalone.py` AND [`prompts/system_prompt.txt`](prompts/system_prompt.txt) to the Project's knowledge panel along with the other docs (`CLAUDE.md`, `OMEGA_HANDBOOK.md`, etc.). No zip.
2. **At session start**, the Project LLM reads `omega_lite_standalone.py` from project knowledge, writes it byte-for-byte to its sandbox cwd, and imports it. The system prompt §3 dictates this exact sequence and runs a smoke test that asserts `trace_id.startswith("sandbox-")`.
3. **For a game or prop analysis**, the LLM normalizes user input into a `GameAnalysisRequest` / `PlayerPropRequest` dict, runs `analyze(...)`, and renders the result. If the engine returns `status: "skipped"` or `quality_gate.downgrades` contains `"dropped_bet_card"`, the LLM enters the self-heal loop (system prompt §6): WebSearch the missing slots, inject them back into the request, and re-run, up to 3 retries.
4. **For broad slate questions**, the LLM should not wait for the user to enumerate every game. It should:
   - infer a reasonable default scope from the user's standing preferences,
   - browse for schedules, odds, team/player context, and injury/news context,
   - normalize any complete candidates into omega_lite_standalone requests,
   - label incomplete candidates as research-only or missing-data watchlist,
   - and explain which markets were excluded due to missing inputs.
5. **The LLM runs**:
   ```python
   from omega_lite_standalone import analyze
   result = analyze({
       "home_team": "Boston Celtics",
       "away_team": "Indiana Pacers",
       "league": "NBA",
       "n_iterations": 5000,
       "seed": 42,
       "home_context": {"off_rating": 118.0, "def_rating": 108.0, "pace": 100.0},
       "away_context": {"off_rating": 115.0, "def_rating": 110.0, "pace": 98.0},
       "odds": {"moneyline_home": -160, "moneyline_away": 140, "over_under": 226.5},
   })
   ```
6. **The result** is a dict with `trace_id="sandbox-XXXX"`, `model_version="omega-lite-v1"`, `input_snapshot`, `result` (the full `GameAnalysisResponse` / `PlayerPropResponse` shape), and `quality_gate` (the plan-level downgrade summary).
7. **The Project Claude renders** the Bet Card in Mode A-sandbox citing the `sandbox-` `trace_id` and echoing `Inputs used` with source URLs/timestamps for every external value (system prompt §8).

## Player props — per-sport gallery (omega_lite is NOT NBA-only)

The same `PlayerPropRequest` shape covers all 9 sport archetypes. Only the `prop_type` and the `player_context["{prop}_mean"]` / `player_context["{prop}_std"]` keys vary. The full list of supported `prop_type` keys per archetype is in [prompts/system_prompt.txt](prompts/system_prompt.txt) §5.

NBA — points:
```python
analyze({"player_name": "Jayson Tatum", "league": "NBA", "prop_type": "pts",
         "line": 27.5, "odds_over": -115, "odds_under": -105,
         "player_context": {"pts_mean": 28.4, "pts_std": 6.2},
         "n_iterations": 5000, "seed": 42})
```

MLB — pitcher strikeouts:
```python
analyze({"player_name": "Gerrit Cole", "league": "MLB", "prop_type": "strikeouts_pitched",
         "line": 7.5, "odds_over": -120, "odds_under": +100,
         "player_context": {"strikeouts_pitched_mean": 8.1, "strikeouts_pitched_std": 2.3},
         "n_iterations": 5000, "seed": 42})
```

NHL — shots on goal:
```python
analyze({"player_name": "Nathan MacKinnon", "league": "NHL", "prop_type": "shots_on_goal",
         "line": 3.5, "odds_over": -135, "odds_under": +110,
         "player_context": {"shots_on_goal_mean": 4.2, "shots_on_goal_std": 1.6},
         "n_iterations": 5000, "seed": 42})
```

NFL — receiving yards:
```python
analyze({"player_name": "CeeDee Lamb", "league": "NFL", "prop_type": "rec_yds",
         "line": 78.5, "odds_over": -110, "odds_under": -110,
         "player_context": {"rec_yds_mean": 85.3, "rec_yds_std": 28.7},
         "n_iterations": 5000, "seed": 42})
```

EPL — shots on target:
```python
analyze({"player_name": "Erling Haaland", "league": "EPL", "prop_type": "shots_on_target",
         "line": 1.5, "odds_over": -120, "odds_under": +100,
         "player_context": {"shots_on_target_mean": 2.1, "shots_on_target_std": 1.2},
         "n_iterations": 5000, "seed": 42})
```

ATP — total games:
```python
analyze({"player_name": "Carlos Alcaraz", "league": "ATP", "prop_type": "total_games",
         "line": 22.5, "odds_over": -110, "odds_under": -110,
         "player_context": {"total_games_mean": 23.4, "total_games_std": 3.1},
         "n_iterations": 5000, "seed": 42})
```

PGA — top 10 finish:
```python
analyze({"player_name": "Scottie Scheffler", "league": "PGA", "prop_type": "top_10",
         "line": 0.5, "odds_over": -160, "odds_under": +135,
         "player_context": {"top_10_mean": 0.42, "top_10_std": 0.18},
         "n_iterations": 5000, "seed": 42})
```

UFC — significant strikes landed:
```python
analyze({"player_name": "Max Holloway", "league": "UFC", "prop_type": "sig_strikes",
         "line": 99.5, "odds_over": -130, "odds_under": +105,
         "player_context": {"sig_strikes_mean": 112.0, "sig_strikes_std": 38.0},
         "n_iterations": 5000, "seed": 42})
```

CS2 — total kills:
```python
analyze({"player_name": "s1mple", "league": "CS2", "prop_type": "kills",
         "line": 19.5, "odds_over": -110, "odds_under": -110,
         "player_context": {"kills_mean": 21.2, "kills_std": 5.8},
         "n_iterations": 5000, "seed": 42})
```

If you're unsure which `prop_type` keys a league supports, ask the engine directly:
```python
from omega_lite_standalone import get_prop_stat_keys
print(get_prop_stat_keys("MLB"))   # → ['hits', 'total_bases', 'runs', 'rbis', 'hrs', ...]
```

For a 2-leg anchor parlay, call `analyze(...)` once per leg with the same seed for each leg's individual sim, then multiply the per-leg probabilities for the joint (per `OMEGA_STRATEGY.md`).

## Limitations to disclose in every Mode A-sandbox response

1. **No native live collectors.** omega_lite itself does not fetch live data. Inputs may come from the user, public web browsing, screenshots, or pasted sportsbook lines. Flag the source and freshness for each numeric input. Lack of native live data should not block an exploratory answer; it only limits whether a formal omega_lite Bet Card can be produced.
2. **No `ExecutionTrace` ledger persistence.** The `trace_id` is per-call and not stored anywhere. If you want backtest reproducibility, run the canonical pipeline locally and feed Mode A-local instead.
3. **Player prop simulator is archetype-default Poisson/Normal.** When canonical `markov_engine` ships, sandbox numbers may diverge from canonical numbers for props until omega_lite is rebuilt.

## Honesty contract for the Project Claude

When responding with Mode A-sandbox:

- Say "I ran `omega_lite_standalone` in the sandbox" — never "I ran Omega".
- Cite the `sandbox-` `trace_id` in the response.
- Echo the inputs you ran on under an `Inputs used` section, with source URLs and timestamps for every external value (system prompt §8).
- If `quality_gate.applied=True` and `downgrades` is non-empty, surface the downgrade reasons in the response — don't paper over the gate.
- If the engine returned `status: "skipped"` or `dropped_bet_card`, you MUST first run the self-heal loop (system prompt §6) — WebSearch the missing slots, re-run — before falling back to a Research Report. Never fabricate.

When the user wants the canonical pipeline instead, send them to `OMEGA_RUN_RECIPE.md`.
