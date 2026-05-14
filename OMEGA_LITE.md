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
python scripts/build_omega_lite.py            # rebuild in place (idempotent)
python scripts/build_omega_lite.py --zip      # also produce omega_lite-v1.zip
```

Run it whenever `omega/core/` or `omega/reasoning/evaluator.py` changes. The script rewrites all imports to `omega_lite.*`, truncates `engine.py` to drop the two markov-dependent methods, and emits a 36 KB zip ready for upload.

## Parity guarantee

`tests/omega_lite/test_parity.py` asserts that for the same inputs and seed:

- `analyze_game` produces identical win probabilities, predicted spread, predicted total, and per-edge `edge_pct` / `ev_pct` / `confidence_tier` between canonical and lite.
- `analyze_player_prop` produces identical `over_prob`, `under_prob`, `recommendation`, and edges.
- Both refuse on the same missing-critical-input cases with identical `missing_requirements` lists.

If parity breaks, the rebuild has drifted — rerun `build_omega_lite.py` and re-run the tests.

## Sandbox workflow

The intended user flow inside a Claude.ai Project or ChatGPT Project:

1. **One-time setup.** Upload `omega_lite-v1.zip` to the Project's knowledge panel along with the other docs (`CLAUDE.md`, `OMEGA_HANDBOOK.md`, etc.).
2. **At session start**, the Project LLM extracts the zip in its analysis tool and imports `omega_lite`. The Project instructions explicitly direct it to do this.
3. **For a game analysis**, the user pastes raw stats + odds, or asks the LLM to fetch them via its web browsing tool. The LLM normalizes into a `GameAnalysisRequest` dict.
 3a. For broad slate questions, the LLM should not wait for the user to enumerate every game unless scope is truly impossible. It should:
   - infer a reasonable default scope from the user’s standing preferences,
   - browse for schedules, odds, team/player context, and injury/news context,
   - normalize any complete candidates into omega_lite requests,
   - label incomplete candidates as research-only,
   - and explain which markets were excluded due to missing inputs.
4. **The LLM runs**:
   ```python
   from omega_lite import analyze
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
5. **The result** is a dict with `trace_id="sandbox-XXXX"`, `model_version="omega-lite-v1"`, `input_snapshot`, `result` (the full `GameAnalysisResponse` shape), and `quality_gate` (the plan-level downgrade summary).
6. **The Project Claude renders** the Bet Card in Mode A-sandbox using the existing instructions. The `sandbox-` trace_id prefix tells the user this came from omega_lite, not a canonical local run.

## Player props — the user's anchor parlay use case

```python
from omega_lite import analyze

result = analyze({
    "player_name": "Jayson Tatum",
    "league": "NBA",
    "prop_type": "pts",
    "line": 27.5,
    "odds_over": -115,
    "odds_under": -105,
    "player_context": {
        "pts_mean": 28.4,   # required: rolling mean over lookback window
        "pts_std": 6.2,     # optional but recommended; defaults to 25% of mean
    },
    "n_iterations": 5000,
    "seed": 42,
})
print(result["result"]["over_prob"], result["result"]["recommendation"])
```

For a 2-leg anchor parlay, the LLM calls `analyze(...)` once per leg with the same seed for each leg's individual sim, then multiplies the per-leg probabilities for the joint (per `OMEGA_STRATEGY.md`).

## Limitations to disclose in every Mode A-sandbox response

1. **No native live collectors.** omega_lite itself does not fetch live data. Inputs may come from the user, public web browsing, screenshots, or pasted sportsbook lines. Flag the source and freshness for each numeric input. Lack of native live data should not block an exploratory answer; it only limits whether a formal omega_lite Bet Card can be produced.
2. **No `ExecutionTrace` ledger persistence.** The `trace_id` is per-call and not stored anywhere. If you want backtest reproducibility, run the canonical pipeline locally and feed Mode A-local instead.
3. **Player prop simulator is archetype-default Poisson/Normal.** When canonical `markov_engine` ships, sandbox numbers may diverge from canonical numbers for props until omega_lite is rebuilt.

## Honesty contract for the Project Claude

When responding with Mode A-sandbox:

- Say "I ran `omega_lite` in the sandbox" — never "I ran Omega".
- Cite the `sandbox-` trace_id in the response.
- Echo the inputs you ran on under a `Inputs used` section so the user can spot bad data fast.
- If `quality_gate.applied=True` and `downgrades` is non-empty, surface the downgrade reasons in the response — don't paper over the gate.

When the user wants the canonical pipeline instead, send them to `OMEGA_RUN_RECIPE.md`.
