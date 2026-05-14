# Omega

A reasoning-led, quantitative sports analytics agent. Omega sits an LLM control layer (intent, planning, quality judgement, explanation) on top of a deterministic engine (Monte Carlo simulation, calibration, contracts, staking) so betting recommendations are reproducible, auditable, and refuse cleanly when evidence is thin.

This is the canonical repository — `cameronlaxton/Omega`. The older repo `cameronlaxton/OmegaSports` is not the same project; do not use it as a reference.

## Architecture (5 layers)

1. **Conversation** (`omega/api/`) — FastAPI surface, sessions, Server-Sent Events streaming.
2. **Reasoning** (`omega/reasoning/`) — intent classification, answer-strategy routing, gather-slot planning, quality gate, orchestrator, LLM client.
3. **Evidence** (`omega/evidence/`) — collectors (ESPN, The Odds API, web search), entity resolution, validators (freshness/sanity/stats), fusion.
4. **Execution** (`omega/core/`) — Monte Carlo simulation, sport archetypes, probability calibration, strict request/response contracts, Kelly staking.
5. **Synthesis** (`omega/synthesis/`) — response composition into one or more of 11 `OutputPackage` types (Bet Card, Game Breakdown, Limited Context Answer, etc.).

See [CLAUDE.md](CLAUDE.md) for the project's architectural rules and design constraints.

## Quick start

```bash
# Install
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Environment (a subset are required depending on which path you run)
export ANTHROPIC_API_KEY=...
export PERPLEXITY_API_KEY=...    # primary structured search
export ODDS_API_KEY=...          # live market odds via The Odds API
export REDIS_URL=...             # optional, for session persistence

# Run the API
uvicorn omega.api.app:app --reload

# Tests
python -m pytest tests/ -v
```

## Public endpoints

- `POST /api/v1/analyze/game` — single-game JSON-in / JSON-out analysis.
- `POST /api/v1/analyze/slate` — full slate analysis.
- `POST /api/v1/analyze/prop` — single player prop analysis.
- `POST /chat` — conversational endpoint with SSE streaming through the full reasoning loop.
- `GET /sessions/{session_id}` — session history.
- `GET /health` — health check.

Request and response shapes are defined as Pydantic models in `omega/core/contracts/schemas.py`. The canonical Bet Card structure is documented in [OMEGA_HANDBOOK.md](OMEGA_HANDBOOK.md).

## Working with Omega from an LLM (Claude.ai Projects, etc.)

Omega is designed to be paired with an LLM Synthesis layer (Claude.ai Project, Claude Code, your own agent). The LLM is *not* the pipeline — it renders Omega outputs, composes narrative, builds parlays, and explains the math. Anything claiming to "run Omega in a sandbox" without an `ExecutionTrace.trace_id` is hallucinated.

For LLM consumers, the following docs are the operating contract:

- [OMEGA_HANDBOOK.md](OMEGA_HANDBOOK.md) — what Omega is, the 5 layers, 11 output packages, 5 execution modes, 3 input importance tiers, and the quality-gate refusal logic.
- [OMEGA_DATA_SOURCES.md](OMEGA_DATA_SOURCES.md) — the 7 evidence collectors that actually exist, freshness windows, and the explicit "NOT INTEGRATED" list (covers.com, dimers.com, rotowire, betmgm as analytic sources).
- [OMEGA_RUN_RECIPE.md](OMEGA_RUN_RECIPE.md) — exact commands to run a single-game, slate, or player-prop analysis locally and feed the JSON back into a Project.
- [OMEGA_STRATEGY.md](OMEGA_STRATEGY.md) — the user's NBA anchor parlay playbook: anchor selection criteria, parlay shape, joint-probability math, correlation haircuts, skip rules.

## Testing

```bash
python -m pytest tests/ -v
```

Tests run without API keys via heuristic/deterministic code paths.

## License

See repository.
