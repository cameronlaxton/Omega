"""
omega_lite — sandbox-runnable deterministic core of Omega.

Use from an LLM analysis tool (Claude.ai analysis tool, ChatGPT code
interpreter) when no local Omega server is available. The package is
pure-Python with optional numpy and Pydantic v2 dependencies.

Quick start:

    from omega_lite import analyze

    result = analyze({
        "home_team": "Boston Celtics",
        "away_team": "Indiana Pacers",
        "league": "NBA",
        "n_iterations": 5000,
        "seed": 42,
        "home_context": {"off_rating": 118.0, "def_rating": 108.0, "pace": 100.0},
        "away_context": {"off_rating": 115.0, "def_rating": 110.0, "pace": 98.0},
        "odds": {
            "moneyline_home": -160,
            "moneyline_away": 140,
            "spread_home": -4.5,
            "spread_home_price": -110,
            "over_under": 226.5,
        },
    })
    print(result["trace_id"])         # sandbox-XXXX
    print(result["result"]["best_bet"])  # BetSlip dict or None

The trace_id is always prefixed `sandbox-` to make this output
distinguishable from a canonical local Omega run.
"""

from omega_lite.run import MODEL_VERSION, analyze
from omega_lite.service import (
    analyze_game,
    analyze_player_prop,
    analyze_slate,
)
from omega_lite.schemas import (
    GameAnalysisRequest,
    GameAnalysisResponse,
    OddsInput,
    PlayerPropRequest,
    PlayerPropResponse,
    SlateAnalysisRequest,
    SlateAnalysisResponse,
)

__version__ = "1.0.0"

__all__ = [
    "MODEL_VERSION",
    "analyze",
    "analyze_game",
    "analyze_player_prop",
    "analyze_slate",
    "GameAnalysisRequest",
    "GameAnalysisResponse",
    "OddsInput",
    "PlayerPropRequest",
    "PlayerPropResponse",
    "SlateAnalysisRequest",
    "SlateAnalysisResponse",
]
