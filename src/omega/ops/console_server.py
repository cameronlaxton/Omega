"""omega-console — standalone, local-loopback, read-only operator console.

Phase 8 Milestone A. A separate FastAPI app (NOT mounted into ``omega-mcp-http``
and sharing none of its state-mutating transports) that serves server-rendered
HTML plus a GET-only JSON API for exploring traces, bets, and session QA.

Access posture (mirrors the MCP transport's fail-closed bind policy, but
reimplemented locally so the console never imports the MCP server):

* default bind is ``127.0.0.1``;
* a non-loopback bind is refused unless ``OMEGA_CONSOLE_ALLOW_REMOTE`` is set
  (or ``--allow-remote``) **and** ``OMEGA_CONSOLE_TOKEN`` provides a bearer
  secret; in that mode a bearer-token gate is installed in front of every route
  except health;
* loopback stays zero-config (no token required);
* no CORS middleware is installed — cross-origin browser access is not enabled.

Everything served is read-only: the service opens ``TraceStore(read_only=True)``
and only ever reads.
"""

from __future__ import annotations

import argparse
import hmac
import ipaddress
import logging
import os
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import omega.ui as _ui_pkg
from omega.ui.api import get_service
from omega.ui.api import router as api_router

logger = logging.getLogger("omega.ops.console_server")

_UI_ROOT = Path(_ui_pkg.__file__).resolve().parent
TEMPLATES_DIR = _UI_ROOT / "templates"
STATIC_DIR = _UI_ROOT / "static"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787

# Paths reachable without a bearer token even on a non-loopback bind.
_OPEN_PATHS = ("/healthz",)

# Navigation: Milestone-A pages are enabled; later-milestone destinations are
# rendered as disabled placeholders (no working endpoint behind them).
NAV_ENABLED = (
    {"key": "scanner", "label": "Edge Scanner", "href": "/scanner"},
    {"key": "traces", "label": "Trace Ledger", "href": "/traces"},
    {"key": "bets", "label": "Bet Ledger", "href": "/bets"},
    {"key": "sessions", "label": "Session Review", "href": "/sessions"},
    {"key": "diagnostics", "label": "Diagnostics", "href": "/diagnostics"},
    {"key": "calibration", "label": "Calibration Status", "href": "/calibration"},
    {"key": "data_quality", "label": "Data Quality", "href": "/data-quality"},
    {"key": "signals", "label": "Signal Performance", "href": "/signals"},
    {"key": "review", "label": "Review Queue", "href": "/review"},
    {"key": "clv", "label": "Market Movement / CLV", "href": "/clv"},
)
# All placeholder pages are implemented as of Milestone B.3.
NAV_PLACEHOLDERS: tuple[dict[str, str], ...] = ()


# ---------------------------------------------------------------------------
# Bind policy (pure, testable — no socket side effects)
# ---------------------------------------------------------------------------


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _configured_token() -> str | None:
    return (os.environ.get("OMEGA_CONSOLE_TOKEN") or "").strip() or None


def is_loopback_host(host: str) -> bool:
    """True only when *host* binds the loopback interface.

    ``localhost`` and 127.0.0.0/8 / ``::1`` are loopback. Wildcards (``0.0.0.0``,
    ``::``) and routable addresses are not. An unresolvable host is treated as
    non-loopback so the policy fails closed.
    """
    normalized = (host or "").strip().strip("[]").lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def resolve_bind_policy(host: str, *, allow_remote: bool, token: str | None) -> str | None:
    """Validate a bind host; return the bearer token to enforce (or None).

    Loopback binds are zero-config: a configured token is honored but not
    required, so this returns ``token`` (possibly None). A non-loopback bind is
    refused unless ``allow_remote`` is set AND a non-empty ``token`` is provided.
    Raises ``RuntimeError`` on any fail-closed condition.
    """
    if is_loopback_host(host):
        return token
    if not allow_remote:
        raise RuntimeError(
            f"Refusing to bind the Omega console to non-loopback host {host!r}: this "
            f"is a local read-only tool and is not exposed on the LAN by default. Bind "
            f"127.0.0.1, or set OMEGA_CONSOLE_ALLOW_REMOTE=1 and "
            f"OMEGA_CONSOLE_TOKEN=<shared-secret> to opt in."
        )
    if not token:
        raise RuntimeError(
            f"OMEGA_CONSOLE_ALLOW_REMOTE is set but OMEGA_CONSOLE_TOKEN is empty. A "
            f"bearer secret is required before binding the console to non-loopback host "
            f"{host!r}; refusing to start an unauthenticated remote listener."
        )
    logger.warning(
        "Omega console binding to NON-LOOPBACK host %s. Bearer-token auth is REQUIRED "
        "on every route except %s. The console is READ-ONLY (no mutation routes).",
        host,
        " and ".join(_OPEN_PATHS),
    )
    return token


# ---------------------------------------------------------------------------
# Bearer auth middleware (pure ASGI; installed only when a token is enforced)
# ---------------------------------------------------------------------------


class _BearerAuthMiddleware:
    """Require ``Authorization: Bearer <token>`` on all but the open paths."""

    def __init__(self, app, *, token: str, open_paths: tuple[str, ...]) -> None:
        self.app = app
        self._expected = f"Bearer {token}"
        self._open = tuple(open_paths)

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        if self._is_open(path) or self._authorized(scope):
            await self.app(scope, receive, send)
            return
        await self._reject(send)

    def _is_open(self, path: str) -> bool:
        return any(path == p for p in self._open)

    def _authorized(self, scope) -> bool:
        for name, value in scope.get("headers", []):
            if name == b"authorization":
                try:
                    provided = value.decode("latin-1")
                except UnicodeDecodeError:
                    return False
                return hmac.compare_digest(provided, self._expected)
        return False

    async def _reject(self, send) -> None:
        body = b'{"error":"unauthorized","detail":"missing or invalid bearer token"}'
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"www-authenticate", b"Bearer"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def build_console_app(
    *,
    db_path: str | None = None,
    sessions_dir: str | Path | None = None,
    max_scan: int | None = None,
    calibration_registry: str | Path | None = None,
    auth_token: str | None = None,
    enrichment_enabled: bool = False,
):
    """Build the read-only console FastAPI app.

    ``auth_token`` installs the bearer gate (used for non-loopback binds). On a
    loopback bind pass ``None`` to keep local usage zero-config. No CORS
    middleware is added — cross-origin access is intentionally not enabled.
    """
    app = FastAPI(
        title="Omega Operator Console",
        version="1",
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    app.state.console_db_path = db_path
    app.state.console_sessions_dir = str(sessions_dir) if sessions_dir else None
    app.state.console_max_scan = max_scan
    app.state.console_calibration_registry = (
        str(calibration_registry) if calibration_registry else None
    )

    if auth_token:
        app.add_middleware(_BearerAuthMiddleware, token=auth_token, open_paths=_OPEN_PATHS)

    app.include_router(api_router)

    @app.get("/healthz")
    def liveness() -> dict[str, str]:
        # DB-independent liveness probe (stays 200 even if the DB is down). The
        # DB-aware readiness probe with counts is /api/healthz.
        return {"status": "ok"}

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    def _ctx(request: Request, **extra: Any) -> dict[str, Any]:
        base = {
            "request": request,
            "nav_enabled": NAV_ENABLED,
            "nav_placeholders": NAV_PLACEHOLDERS,
            "enrichment_enabled": enrichment_enabled,
        }
        base.update(extra)
        return base

    # -- HTML pages (GET only) ------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def page_index(request: Request, service=Depends(get_service)):
        cc = service.command_center()
        return templates.TemplateResponse(
            request,
            "index.html",
            _ctx(
                request,
                cc=cc.model_dump(),
                health=(cc.health.model_dump() if cc.health else None),
                review_count=cc.review_count,
                active="home",
            ),
        )

    @app.get("/scanner", response_class=HTMLResponse)
    def page_scanner(
        request: Request,
        service=Depends(get_service),
        league: str | None = Query(None),
        limit: int | None = Query(None, ge=1, le=200),
    ):
        data = service.edge_scanner(limit=limit, league=league)
        return templates.TemplateResponse(
            request, "scanner.html", _ctx(request, data=data.model_dump(), active="scanner")
        )

    @app.get("/traces", response_class=HTMLResponse)
    def page_traces(
        request: Request,
        service=Depends(get_service),
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
    ):
        effective_page_size = min(page_size, limit) if limit is not None else page_size
        data = service.list_traces(
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
        return templates.TemplateResponse(
            request, "traces.html", _ctx(request, data=data.model_dump(), active="traces")
        )

    @app.get("/traces/{trace_id}", response_class=HTMLResponse)
    def page_trace_detail(request: Request, trace_id: str, service=Depends(get_service)):
        detail = service.get_trace_detail(trace_id)
        if detail is None:
            return templates.TemplateResponse(
                request,
                "trace_detail.html",
                _ctx(request, detail=None, not_found_id=trace_id, active="traces"),
                status_code=404,
            )
        return templates.TemplateResponse(
            request, "trace_detail.html", _ctx(request, detail=detail.model_dump(), active="traces")
        )

    @app.get("/traces/{trace_id}/similar", response_class=HTMLResponse)
    def page_trace_similar(request: Request, trace_id: str, service=Depends(get_service)):
        view = service.similar_spots(trace_id)
        if view is None:
            return templates.TemplateResponse(
                request,
                "similar.html",
                _ctx(request, view=None, not_found_id=trace_id, active="traces"),
                status_code=404,
            )
        return templates.TemplateResponse(
            request, "similar.html", _ctx(request, view=view.model_dump(), active="traces")
        )

    @app.get("/bets", response_class=HTMLResponse)
    def page_bets(
        request: Request,
        service=Depends(get_service),
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
    ):
        effective_page_size = min(page_size, limit) if limit is not None else page_size
        data = service.list_bets(
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
        return templates.TemplateResponse(
            request, "bets.html", _ctx(request, data=data.model_dump(), active="bets")
        )

    @app.get("/bets/{ledger_id}", response_class=HTMLResponse)
    def page_bet_detail(request: Request, ledger_id: str, service=Depends(get_service)):
        detail = service.get_bet_detail(ledger_id)
        if detail is None:
            return templates.TemplateResponse(
                request,
                "bet_detail.html",
                _ctx(request, detail=None, not_found_id=ledger_id, active="bets"),
                status_code=404,
            )
        return templates.TemplateResponse(
            request, "bet_detail.html", _ctx(request, detail=detail.model_dump(), active="bets")
        )

    @app.get("/sessions", response_class=HTMLResponse)
    def page_sessions(
        request: Request,
        service=Depends(get_service),
        page: int = Query(1, ge=1),
        page_size: int = Query(25, ge=1, le=200),
        limit: int | None = Query(None, ge=1, le=200),
    ):
        effective_page_size = min(page_size, limit) if limit is not None else page_size
        data = service.list_sessions(page=page, page_size=effective_page_size)
        return templates.TemplateResponse(
            request, "sessions.html", _ctx(request, data=data.model_dump(), active="sessions")
        )

    @app.get("/sessions/{session_id}", response_class=HTMLResponse)
    def page_session_detail(request: Request, session_id: str, service=Depends(get_service)):
        detail = service.get_session_detail(session_id)
        if detail is None:
            return templates.TemplateResponse(
                request,
                "session_detail.html",
                _ctx(request, detail=None, not_found_id=session_id, active="sessions"),
                status_code=404,
            )
        return templates.TemplateResponse(
            request,
            "session_detail.html",
            _ctx(request, detail=detail.model_dump(), active="sessions"),
        )

    @app.get("/diagnostics", response_class=HTMLResponse)
    def page_diagnostics(request: Request, service=Depends(get_service)):
        data = service.diagnostics()
        return templates.TemplateResponse(
            request,
            "diagnostics.html",
            _ctx(request, data=data.model_dump(), active="diagnostics"),
        )

    @app.get("/calibration", response_class=HTMLResponse)
    def page_calibration(
        request: Request,
        service=Depends(get_service),
        league: str | None = Query(None),
        status: str | None = Query(None),
    ):
        data = service.calibration_status(league=league, status=status)
        chart = service.calibration_chart(league=league)
        reliability = service.reliability_diagram(league=league)
        return templates.TemplateResponse(
            request,
            "calibration.html",
            _ctx(
                request,
                data=data.model_dump(),
                chart=chart.model_dump(),
                reliability=reliability.model_dump(),
                active="calibration",
            ),
        )

    @app.get("/data-quality", response_class=HTMLResponse)
    def page_data_quality(
        request: Request,
        service=Depends(get_service),
        league: str | None = Query(None),
    ):
        data = service.data_quality(league=league)
        return templates.TemplateResponse(
            request,
            "data_quality.html",
            _ctx(request, data=data.model_dump(), active="data_quality"),
        )

    @app.get("/signals", response_class=HTMLResponse)
    def page_signals(
        request: Request,
        service=Depends(get_service),
        league: str | None = Query(None),
    ):
        data = service.signal_performance(league=league)
        return templates.TemplateResponse(
            request,
            "signals.html",
            _ctx(request, data=data.model_dump(), active="signals"),
        )

    @app.get("/review", response_class=HTMLResponse)
    def page_review(
        request: Request,
        service=Depends(get_service),
        severity: str | None = Query(None),
    ):
        data = service.review_queue().model_dump()
        severity = (severity or "").strip().lower() or None
        if severity in {"info", "warn", "fail"}:
            data["buckets"] = [b for b in data["buckets"] if b["severity"] == severity]
        return templates.TemplateResponse(
            request,
            "review.html",
            _ctx(request, data=data, active="review", severity=severity),
        )

    @app.get("/clv", response_class=HTMLResponse)
    def page_clv(
        request: Request,
        service=Depends(get_service),
        league: str | None = Query(None),
    ):
        data = service.clv_report(league=league)
        scatter = service.clv_scatter(league=league, clv=data)
        return templates.TemplateResponse(
            request,
            "clv.html",
            _ctx(request, data=data.model_dump(), scatter=scatter.model_dump(), active="clv"),
        )

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="omega-console",
        description="Read-only Omega operator console (loopback by default).",
    )
    p.add_argument("--host", default=None, help="Bind host (default 127.0.0.1).")
    p.add_argument("--port", type=int, default=None, help="Bind port (default 8787).")
    p.add_argument("--db", default=None, help="Trace DB path override (else default).")
    p.add_argument("--sessions-dir", default=None, help="Session sidecar dir override.")
    p.add_argument("--max-scan", type=int, default=None, help="Max read-scan rows.")
    p.add_argument(
        "--calibration-registry",
        default=None,
        help="Calibration profiles.json override (else the registry default).",
    )
    p.add_argument(
        "--allow-remote",
        action="store_true",
        help="Opt in to a non-loopback bind (requires OMEGA_CONSOLE_TOKEN).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    """Run the console with uvicorn on the loopback interface by default."""
    logging.basicConfig(level=logging.INFO)
    args = _build_arg_parser().parse_args(argv)

    host = args.host or os.environ.get("OMEGA_CONSOLE_HOST") or DEFAULT_HOST
    port_env = os.environ.get("OMEGA_CONSOLE_PORT")
    port = args.port or (int(port_env) if port_env else DEFAULT_PORT)
    allow_remote = args.allow_remote or _truthy(os.environ.get("OMEGA_CONSOLE_ALLOW_REMOTE"))
    token = _configured_token()

    enforced_token = resolve_bind_policy(host, allow_remote=allow_remote, token=token)

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "omega-console requires uvicorn: python -m pip install -e .[console]"
        ) from exc

    app = build_console_app(
        db_path=args.db,
        sessions_dir=args.sessions_dir,
        max_scan=args.max_scan,
        calibration_registry=args.calibration_registry
        or os.environ.get("OMEGA_CONSOLE_CALIBRATION_REGISTRY"),
        auth_token=enforced_token,
    )
    logger.info(
        "Omega console starting on http://%s:%s (read-only; token=%s)",
        host,
        port,
        "required" if enforced_token else "not required (loopback)",
    )
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
