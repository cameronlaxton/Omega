"""
Analysis endpoints — game, slate, and player prop.

These are the JSON-in/JSON-out deterministic endpoints.
No LLM, no streaming. Caller supplies all context.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from omega.core.contracts.schemas import (
    GameAnalysisRequest,
    GameAnalysisResponse,
    PlayerPropRequest,
    PlayerPropResponse,
    SlateAnalysisRequest,
    SlateAnalysisResponse,
)
from omega.core.contracts.service import (
    analyze_game,
    analyze_player_prop,
    analyze_slate,
)

router = APIRouter(prefix="/api/v1", tags=["analysis"])


@router.post("/analyze/game", response_model=GameAnalysisResponse)
async def analyze_game_endpoint(req: GameAnalysisRequest):
    """Analyze a single game matchup. Caller supplies all context."""
    try:
        return analyze_game(req)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.post("/analyze/slate", response_model=SlateAnalysisResponse)
async def analyze_slate_endpoint(req: SlateAnalysisRequest):
    """Analyze a full slate of games."""
    try:
        return analyze_slate(req)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.post("/analyze/prop", response_model=PlayerPropResponse)
async def analyze_prop_endpoint(req: PlayerPropRequest):
    """Analyze a single player prop."""
    try:
        return analyze_player_prop(req)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
