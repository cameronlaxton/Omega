# OMEGA_RUN_RECIPE

How to actually run Omega and feed its JSON output back into a Claude.ai Project. There are now **two real paths** — pick whichever has lower friction for you on a given day:

- **Path A (default, recommended for most queries):** Run `omega_lite` directly in the Claude.ai Project's analysis tool. No local server. No env vars. Same deterministic math as canonical. Trace ids carry a `sandbox-` prefix. See [OMEGA_LITE.md](OMEGA_LITE.md).
- **Path B (best for backtests and reproducibility ledgers):** Run the canonical `omega/` FastAPI service locally, capture the JSON, and paste it into the Project. Steps below.

If you are a Claude.ai Project user reading this with neither path available (e.g. Perplexity Spaces, no sandbox + no local server), your default response mode is `[CLAUDE-ESTIMATED]` and you must say so up front.

## Path A — `omega_lite` in the sandbox (no local install)

In a Claude.ai Project chat with `omega_lite-v1.zip` attached to the Project knowledge:

```python
# In the analysis tool, at session start:
import zipfile
zipfile.ZipFile("/mnt/user-data/uploads/omega_lite-v1.zip").extractall(".")

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
# result["trace_id"] is "sandbox-XXXX"
# result["result"] is the full GameAnalysisResponse shape
```

For player props (anchor parlay use case):

```python
result = analyze({
    "player_name": "Jayson Tatum",
    "league": "NBA",
    "prop_type": "pts",
    "line": 27.5,
    "odds_over": -115,
    "odds_under": -105,
    "player_context": {"pts_mean": 28.4, "pts_std": 6.2},
    "n_iterations": 5000,
    "seed": 42,
})
```

Read [OMEGA_LITE.md](OMEGA_LITE.md) for the full contract — including the parity guarantee with canonical Omega.

## Path B — local FastAPI server (full pipeline, real `trace_id`)

### Prerequisites

- Python 3.11+ in a virtualenv with repo dependencies installed.
- Recommended environment variables in `.env` or shell:
  - `ANTHROPIC_API_KEY` — drives the reasoning-layer LLM client and the Anthropic web_search fallback.
  - `PERPLEXITY_API_KEY` — primary structured-search backend (preferred over Anthropic web_search).
  - `ODDS_API_KEY` — required for live market odds via The Odds API.
  - `REDIS_URL` — optional; for session persistence. Without it, sessions are in-memory.
  - `CORS_ORIGINS` — defaults to `http://localhost:3000,http://localhost:8000`.

## Option 1 — JSON-in / JSON-out analysis endpoints (no LLM, no streaming)

Best path when you want a clean Omega artifact to paste into Claude.ai. Caller supplies all context; engine does no live fetching.

Start the server:
```bash
uvicorn omega.api.app:app --reload
```

Single game:
```bash
curl -s -X POST http://localhost:8000/api/v1/analyze/game \
  -H "Content-Type: application/json" \
  -d '{
    "home_team": "Boston Celtics",
    "away_team": "Los Angeles Lakers",
    "league": "NBA",
    "n_iterations": 5000,
    "seed": 42,
    "odds": {
      "spread_home": -3.5,
      "spread_home_price": -110,
      "moneyline_home": -165,
      "moneyline_away": 140,
      "over_under": 224.5
    },
    "home_context": { /* pre-fetched home stats */ },
    "away_context": { /* pre-fetched away stats */ }
  }'
```

Slate (provide pre-fetched games list):
```bash
curl -s -X POST http://localhost:8000/api/v1/analyze/slate \
  -H "Content-Type: application/json" \
  -d '{
    "league": "NBA",
    "date": "2026-05-14",
    "bankroll": 1000,
    "edge_threshold": 0.03,
    "games": [ /* list of {home_team, away_team, odds} */ ]
  }'
```

Player prop:
```bash
curl -s -X POST http://localhost:8000/api/v1/analyze/prop \
  -H "Content-Type: application/json" \
  -d '{
    "player_name": "Jayson Tatum",
    "league": "NBA",
    "prop_type": "pts",
    "line": 27.5,
    "odds_over": -110,
    "odds_under": -110,
    "player_context": { /* recent rolling stats */ },
    "game_context":   { /* opponent def rating, pace, etc */ }
  }'
```

The response is a `GameAnalysisResponse` / `SlateAnalysisResponse` / `PlayerPropResponse` JSON object — paste the full body into your Claude.ai Project to render in Mode A.

## Option 2 — Conversational endpoint with full pipeline + streaming

Runs the whole reasoning loop including intent classification, gather planning, evidence collection, quality gate, simulation, and synthesis. Streams Server-Sent Events.

```bash
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Analyze tonight'\''s Celtics vs Lakers for me, including any bet edges"}'
```

The stream emits events: `stage_update`, `partial_text`, `structured_data`, `done`, `error`. The `done` event carries the final composed response including `sections.bet_card` (if not gated out) and the `metadata.execution_mode`. For a single-shot run, capture the `done` payload and paste it into the Project.

## Option 3 — Drive from Claude Code locally

If you don't want to manage the server, ask Claude Code to call `omega.core.contracts.service.analyze_game` (or `analyze_slate`, `analyze_player_prop`) directly with the request payload, dump the response to JSON, and surface it.

Example prompt to Claude Code:
> Run a single-game Omega analysis for Celtics vs Lakers tonight. Use `omega.core.contracts.service.analyze_game` with `n_iterations=5000`, `seed=42`. Use today's consensus odds from The Odds API (call `omega.evidence.collectors.odds_api.get_upcoming_odds("NBA")` and find the matching game). For team stats, pull standings via `omega.evidence.collectors.espn.get_standings("NBA")` and produce a minimal `home_context` / `away_context`. Print the full `GameAnalysisResponse` as JSON.

## Copying Omega output into a Claude.ai Project

1. Open the Project chat.
2. Start a new message with `MODE A — Omega run` and the league + matchup.
3. Paste the full JSON body of the `GameAnalysisResponse`.
4. Ask for the synthesis you want: Bet Card render, parlay construction, scenario walk-through, narrative explanation.
5. The Project Claude will render against the contract, cite `metadata.trace_id` if present, and not invent new numbers.

## When you have NO Omega run available

The Project Claude defaults to Mode B (`[CLAUDE-ESTIMATED]`). It will:

- Tag every numeric value as estimated, cap confidence at tier C.
- Show its working: rate stats used, expected scores, market-implied probability, edge math.
- End with a `verify_with_omega` block telling you the exact local command to run.

If you see a Project response that has confident numbers but no `MODE A` declaration and no `trace_id`, that's a hallucination — refuse it and run Omega.
