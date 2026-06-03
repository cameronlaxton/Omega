"""HTTP transport wrapper for the Omega FastMCP server.

The stdio server remains the canonical local-agent entry point. This module only
mounts official FastMCP ASGI helpers so browser-facing clients can reach the same
tool registry over HTTP transports.
"""

from __future__ import annotations

import os
from typing import Any

from omega.mcp.server import build_server

DEFAULT_CORS_ORIGINS = (
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
)


def _parse_cors_origins(raw: str | None = None) -> list[str]:
    value = raw if raw is not None else os.environ.get("OMEGA_CORS_ORIGINS")
    if not value:
        return list(DEFAULT_CORS_ORIGINS)
    origins = [part.strip() for part in value.split(",") if part.strip()]
    if not origins:
        return list(DEFAULT_CORS_ORIGINS)
    if "*" in origins:
        raise RuntimeError(
            "OMEGA_CORS_ORIGINS must not include '*' while credentialed CORS is enabled"
        )
    return origins


def _require_fastmcp_http_helpers(mcp: Any) -> None:
    missing = [
        name for name in ("sse_app", "streamable_http_app") if not callable(getattr(mcp, name, None))
    ]
    if missing:
        raise RuntimeError(
            "Omega MCP HTTP transport requires mcp[cli]>=1.27 with "
            "FastMCP.sse_app() and FastMCP.streamable_http_app(); missing: "
            + ", ".join(missing)
        )


def build_http_app():
    """Build the browser-reachable MCP FastAPI app.

    Both transports come from a single FastMCP instance and share the same tool
    registry. Two wiring details are load-bearing and easy to get wrong:

    1. **Path prefixes.** FastMCP's ``streamable_http_app()``/``sse_app()`` each
       carry an internal route at ``settings.streamable_http_path`` / ``sse_path``.
       If those keep their ``/mcp`` / ``/sse`` defaults *and* we mount the apps at
       ``/mcp`` / ``/sse``, the real endpoint doubles to ``/mcp/mcp``. We set the
       internal paths to ``/`` so the mount prefix alone defines the public path.
       ``sse_app(mount_path="/sse")`` makes the advertised SSE message endpoint
       resolve to ``/sse/messages/`` rather than a bare ``/messages/`` the client
       can't reach.
    2. **Lifespan.** ``streamable_http_app()`` starts its session manager only
       inside its own Starlette lifespan, and Starlette does **not** run a mounted
       sub-app's lifespan. We must run ``mcp.session_manager.run()`` in the parent
       app's lifespan or every ``/mcp`` request fails with
       "Task group is not initialized".
    """
    try:
        from contextlib import asynccontextmanager

        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError as exc:
        raise RuntimeError(
            "Omega MCP HTTP app requires FastAPI and uvicorn: "
            "python -m pip install -e .[mcp]"
        ) from exc

    mcp = build_server()
    _require_fastmcp_http_helpers(mcp)

    # The mount prefix is the public path; keep the inner routes at root so the
    # effective paths are exactly /mcp and /sse (not /mcp/mcp, /sse/sse).
    mcp.settings.streamable_http_path = "/"
    mcp.settings.sse_path = "/"

    # Build the transport apps up front. streamable_http_app() lazily creates the
    # session manager, which the parent lifespan below needs to start.
    streamable_app = mcp.streamable_http_app()
    sse_app = mcp.sse_app(mount_path="/sse")

    @asynccontextmanager
    async def lifespan(_app):
        async with mcp.session_manager.run():
            yield

    app = FastAPI(title="Omega MCP", version="1", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_parse_cors_origins(),
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=[
            "authorization",
            "content-type",
            "mcp-protocol-version",
            "mcp-session-id",
        ],
    )

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    app.mount("/sse", sse_app)
    app.mount("/mcp", streamable_app)
    return app


def run_http() -> None:
    """Run the HTTP MCP server on localhost by default."""
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "Omega MCP HTTP app requires uvicorn: python -m pip install -e .[mcp]"
        ) from exc

    host = os.environ.get("OMEGA_MCP_HOST") or "127.0.0.1"
    port = int(os.environ.get("OMEGA_MCP_PORT") or "8000")
    uvicorn.run(build_http_app(), host=host, port=port)
