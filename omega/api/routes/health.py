"""Health and readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from omega.api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        version="0.1.0",
        services={"core": "ready"},
    )
