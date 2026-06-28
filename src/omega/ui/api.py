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
    CalibrationChart,
    CalibrationStatusView,
    ClvScatter,
    ClvView,
    DiagnosticsView,
    EdgeScannerView,
    HealthResponse,
    QualityHeatmap,
    ReliabilityDiagram,
    ReviewQueueView,
    SessionDetail,
    SessionListResponse,
    SignalPerformanceView,
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
        calibration_registry_path=getattr(state, "console_calibration_registry", None),
    )
    try:
        yield service
    finally:
        service.close()


router = APIRouter(prefix="/api", tags=["console"])


@router.get("/healthz", response_model=HealthResponse)
def healthz(service: ConsoleService = Depends(get_service)) -> HealthResponse:
    return service.health()


@router.get("/diagnostics", response_model=DiagnosticsView)
def diagnostics(service: ConsoleService = Depends(get_service)) -> DiagnosticsView:
    return service.diagnostics()


@router.get("/calibration", response_model=CalibrationStatusView)
def calibration(
    service: ConsoleService = Depends(get_service),
    league: str | None = Query(None),
    status: str | None = Query(None),
) -> CalibrationStatusView:
    return service.calibration_status(league=league, status=status)


@router.get("/calibration-chart", response_model=CalibrationChart)
def calibration_chart(
    service: ConsoleService = Depends(get_service),
    league: str | None = Query(None),
) -> CalibrationChart:
    return service.calibration_chart(league=league)


@router.get("/signals", response_model=SignalPerformanceView)
def signals(
    service: ConsoleService = Depends(get_service),
    league: str | None = Query(None),
) -> SignalPerformanceView:
    return service.signal_performance(league=league)


@router.get("/review", response_model=ReviewQueueView)
def review(service: ConsoleService = Depends(get_service)) -> ReviewQueueView:
    return service.review_queue()


@router.get("/clv", response_model=ClvView)
def clv(
    service: ConsoleService = Depends(get_service),
    league: str | None = Query(None),
) -> ClvView:
    return service.clv_report(league=league)


@router.get("/clv-scatter", response_model=ClvScatter)
def clv_scatter(
    service: ConsoleService = Depends(get_service),
    league: str | None = Query(None),
) -> ClvScatter:
    return service.clv_scatter(league=league)


@router.get("/reliability", response_model=ReliabilityDiagram)
def reliability(
    service: ConsoleService = Depends(get_service),
    league: str | None = Query(None),
) -> ReliabilityDiagram:
    return service.reliability_diagram(league=league)


@router.get("/data-quality", response_model=QualityHeatmap)
def data_quality(
    service: ConsoleService = Depends(get_service),
    league: str | None = Query(None),
) -> QualityHeatmap:
    return service.data_quality(league=league)


@router.get("/scanner", response_model=EdgeScannerView)
def scanner(
    service: ConsoleService = Depends(get_service),
    league: str | None = Query(None),
    limit: int | None = Query(None, ge=1, le=200),
) -> EdgeScannerView:
    return service.edge_scanner(limit=limit, league=league)


@router.get("/traces", response_model=TraceListResponse)
def list_traces(
    service: ConsoleService = Depends(get_service),
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
    limit: int | None = Query(None, ge=1, le=200),
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
    league: str | None = Query(None),
    sport: str | None = Query(None),
    kind: str | None = Query(None),
    market: str | None = Query(None),
    confidence: str | None = Query(None),
    session_id: str | None = Query(None),
) -> TraceListResponse:
    effective_page_size = min(page_size, limit) if limit is not None else page_size
    return service.list_traces(
        page=page,
        page_size=effective_page_size,
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
def get_trace(trace_id: str, service: ConsoleService = Depends(get_service)) -> TraceDetail:
    detail = service.get_trace_detail(trace_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"trace {trace_id!r} not found")
    return detail


@router.get("/bets", response_model=BetListResponse)
def list_bets(
    service: ConsoleService = Depends(get_service),
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
    limit: int | None = Query(None, ge=1, le=200),
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
    league: str | None = Query(None),
    sport: str | None = Query(None),
    status: str | None = Query(None),
    bookmaker: str | None = Query(None),
    provenance: str | None = Query(None),
) -> BetListResponse:
    effective_page_size = min(page_size, limit) if limit is not None else page_size
    return service.list_bets(
        page=page,
        page_size=effective_page_size,
        date_from=date_from,
        date_to=date_to,
        league=league,
        sport=sport,
        status=status,
        bookmaker=bookmaker,
        provenance=provenance,
    )


@router.get("/bets/{ledger_id}", response_model=BetDetail)
def get_bet(ledger_id: str, service: ConsoleService = Depends(get_service)) -> BetDetail:
    detail = service.get_bet_detail(ledger_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"bet {ledger_id!r} not found")
    return detail


@router.get("/sessions", response_model=SessionListResponse)
def list_sessions(
    service: ConsoleService = Depends(get_service),
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
    limit: int | None = Query(None, ge=1, le=200),
) -> SessionListResponse:
    effective_page_size = min(page_size, limit) if limit is not None else page_size
    return service.list_sessions(page=page, page_size=effective_page_size)


@router.get("/sessions/{session_id}", response_model=SessionDetail)
def get_session(session_id: str, service: ConsoleService = Depends(get_service)) -> SessionDetail:
    detail = service.get_session_detail(session_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"session {session_id!r} not found")
    return detail
