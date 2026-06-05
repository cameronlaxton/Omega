"""HTTP transport wrapper for the Omega FastMCP server.

The stdio server remains the canonical local-agent entry point. This module only
mounts official FastMCP ASGI helpers so browser-facing clients can reach the same
tool registry over HTTP transports.
"""

from __future__ import annotations

import hmac
import ipaddress
import logging
import os
from typing import Any

from omega.mcp.server import build_server

logger = logging.getLogger("omega.mcp.http")

DEFAULT_CORS_ORIGINS = (
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
)

# Tools that mutate persisted state (bets, ledger settlement, trace outcomes,
# batch runs). These are the surface that an unauthenticated, non-loopback bind
# would expose to any client able to reach the port — named loudly in the bind
# warning and the refusal error so operators understand the blast radius.
STATE_MUTATING_TOOLS = (
    "omega_record_flat_bet",
    "omega_settle_bets",
    "omega_trace_attach_outcome",
    "omega_trace_void_prop",
    "omega_run_batch",
)

# Top-level path prefixes that carry the tool-invoking transports and therefore
# require the bearer token when one is configured. /healthz stays open.
_PROTECTED_PREFIXES = ("/mcp", "/sse")


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _configured_token() -> str | None:
    """Return the configured shared secret, or None when unset/blank."""
    return (os.environ.get("OMEGA_MCP_TOKEN") or "").strip() or None


def _is_loopback_host(host: str) -> bool:
    """True when *host* binds only the loopback interface.

    Accepts the ``localhost`` hostname and any address in 127.0.0.0/8 or ``::1``.
    Wildcard binds (``0.0.0.0``, ``::``) and routable addresses are NOT loopback.
    Unresolvable hostnames are treated as non-loopback so the bind policy fails
    closed rather than open.
    """
    normalized = host.strip().strip("[]").lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _enforce_bind_policy(host: str) -> str | None:
    """Validate a bind host and return the bearer token to enforce (or None).

    Loopback binds are zero-config: any configured token is still honored but
    none is required. A non-loopback bind is refused unless the operator has
    explicitly opted in via ``OMEGA_MCP_ALLOW_REMOTE`` *and* configured a shared
    secret via ``OMEGA_MCP_TOKEN``; in that case the exposed state-mutating tools
    are logged at WARNING level before the listener starts.
    """
    token = _configured_token()
    if _is_loopback_host(host):
        return token

    if not _truthy(os.environ.get("OMEGA_MCP_ALLOW_REMOTE")):
        raise RuntimeError(
            f"Refusing to bind the Omega MCP HTTP transport to non-loopback host "
            f"{host!r}: the transport is unauthenticated by default and would expose "
            f"state-mutating tools ({', '.join(STATE_MUTATING_TOOLS)}) to any client "
            f"that can reach the port. Bind 127.0.0.1, or set OMEGA_MCP_ALLOW_REMOTE=1 "
            f"and OMEGA_MCP_TOKEN=<shared-secret> to opt in."
        )
    if not token:
        raise RuntimeError(
            f"OMEGA_MCP_ALLOW_REMOTE is set but OMEGA_MCP_TOKEN is empty. A shared "
            f"secret is required before binding the Omega MCP HTTP transport to "
            f"non-loopback host {host!r}; refusing to start an unauthenticated remote "
            f"listener."
        )
    logger.warning(
        "Omega MCP HTTP transport binding to NON-LOOPBACK host %s:%s. Bearer-token "
        "auth is REQUIRED on %s. State-mutating tools now reachable off-host: %s",
        host,
        os.environ.get("OMEGA_MCP_PORT") or "8000",
        " and ".join(_PROTECTED_PREFIXES),
        ", ".join(STATE_MUTATING_TOOLS),
    )
    return token


class _BearerAuthMiddleware:
    """Pure-ASGI bearer-token gate for the mounted /mcp and /sse transports.

    Implemented at the ASGI layer (not Starlette's ``BaseHTTPMiddleware``) so it
    never buffers the streaming SSE response — it either rejects with a 401 or
    forwards the original ``receive``/``send`` untouched. Loopback-only
    deployments never install it, keeping local usage zero-config.
    """

    def __init__(self, app, *, token: str, protected_prefixes: tuple[str, ...]):
        self.app = app
        self._expected = f"Bearer {token}"
        self._prefixes = protected_prefixes

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        # Let CORS preflight through unauthenticated; it carries no credentials
        # and no request body.
        if scope.get("method") == "OPTIONS" or not self._is_protected(scope.get("path", "")):
            await self.app(scope, receive, send)
            return
        if self._authorized(scope):
            await self.app(scope, receive, send)
            return
        await self._reject(send)

    def _is_protected(self, path: str) -> bool:
        return any(path == prefix or path.startswith(prefix + "/") for prefix in self._prefixes)

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


def build_http_app(auth_token: str | None = None):
    """Build the browser-reachable MCP FastAPI app.

    When ``auth_token`` is supplied (or ``OMEGA_MCP_TOKEN`` is set in the
    environment), a bearer-token gate is installed in front of the ``/mcp`` and
    ``/sse`` transports. Pass nothing on a loopback deployment to keep local
    usage zero-config (no token, no middleware).

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

    if auth_token is None:
        auth_token = _configured_token()

    app = FastAPI(title="Omega MCP", version="1", lifespan=lifespan)

    # Bearer auth (when configured) is added before CORS so that CORS ends up the
    # outermost middleware: preflight OPTIONS is answered by CORS, and the auth
    # 401 still carries CORS headers for browser clients.
    if auth_token:
        app.add_middleware(
            _BearerAuthMiddleware,
            token=auth_token,
            protected_prefixes=_PROTECTED_PREFIXES,
        )

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
    """Run the HTTP MCP server on the loopback interface by default.

    Binding a non-loopback host is refused unless ``OMEGA_MCP_ALLOW_REMOTE=1``
    and ``OMEGA_MCP_TOKEN`` are both set (see ``_enforce_bind_policy``).
    """
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "Omega MCP HTTP app requires uvicorn: python -m pip install -e .[mcp]"
        ) from exc

    host = os.environ.get("OMEGA_MCP_HOST") or "127.0.0.1"
    port = int(os.environ.get("OMEGA_MCP_PORT") or "8000")
    token = _enforce_bind_policy(host)
    uvicorn.run(build_http_app(auth_token=token), host=host, port=port)
