"""GET-only JSON API for the read-only operator console (Milestone A).

Every route here is a read. There are deliberately no POST/PUT/PATCH/DELETE
handlers — ``tests/ui/test_console_routes_read_only.py`` fails the build if one
is introduced. The service dependency opens a fresh ``TraceStore(read_only=True)``
per request and closes it afterwards.
"""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from omega.ui.schemas import (
    BetDetail,
    BetListResponse,
    HealthResponse,
    SessionDetail,
    SessionListResponse,
    TraceDetail,
    TraceListResponse,
)
from omega.ui.service import ConsoleService, open_service


def get_service(request: Request) -> Iterator[ConsoleService]:
    """Per-request read-only service. Config is read from ``app.state``."""
    state = request.app.state
    service = open_service(
        db_path=getattr(state, "console_db_path", None),
        sessions_dir=getattr(state, "console_sessions_dir", None),
        max_scan=getattr(state, "console_max_scan", None),
    )
    try:
        yield service
    finally:
        service.close()


router = APIRouter(prefix="/api", tags=["console"])


@router.get("/healthz", response_model=HealthResponse)
def healthz(service: ConsoleService = Depends(get_service)) -> HealthResponse:
    return service.health()


@router.get("/traces", response_model=TraceListResponse)
def list_traces(
    service: ConsoleService = Depends(get_service),
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
    league: str | None = Query(None),
    sport: str | None = Query(None),
    kind: str | None = Query(None),
    market: str | None = Query(None),
    confidence: str | None = Query(None),
    session_id: str | None = Query(None),
) -> TraceListResponse:
    return service.list_traces(
        page=page,
        page_size=page_size,
        date_from=date_from,
        date_to=date_to,
        league=league,
        sport=sport,
        kind=kind,
        market=market,
        confidence=confidence,
        session_id=session_id,
    )


@router.get("/traces/{trace_id}", response_model=TraceDetail)
def get_trace(
    trace_id: str, service: ConsoleService = Depends(get_service)
) -> TraceDetail:
    detail = service.get_trace_detail(trace_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"trace {trace_id!r} not found")
    return detail


@router.get("/bets", response_model=BetListResponse)
def list_bets(
    service: ConsoleService = Depends(get_service),
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
    league: str | None = Query(None),
    sport: str | None = Query(None),
    status: str | None = Query(None),
    bookmaker: str | None = Query(None),
    provenance: str | None = Query(None),
) -> BetListResponse:
    return service.list_bets(
        page=page,
        page_size=page_size,
        date_from=date_from,
        date_to=date_to,
        league=league,
        sport=sport,
        status=status,
        bookmaker=bookmaker,
        provenance=provenance,
    )


@router.get("/bets/{ledger_id}", response_model=BetDetail)
def get_bet(
    ledger_id: str, service: ConsoleService = Depends(get_service)
) -> BetDetail:
    detail = service.get_bet_detail(ledger_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"bet {ledger_id!r} not found")
    return detail


@router.get("/sessions", response_model=SessionListResponse)
def list_sessions(
    service: ConsoleService = Depends(get_service),
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
) -> SessionListResponse:
    return service.list_sessions(page=page, page_size=page_size)


@router.get("/sessions/{session_id}", response_model=SessionDetail)
def get_session(
    session_id: str, service: ConsoleService = Depends(get_service)
) -> SessionDetail:
    detail = service.get_session_detail(session_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"session {session_id!r} not found")
    return detail
