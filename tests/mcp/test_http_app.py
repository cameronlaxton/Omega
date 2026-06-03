from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from omega.mcp.http_app import _parse_cors_origins, build_http_app
from omega.mcp.server import build_server

# FastMCP's transport-security layer validates the Host header against
# localhost by default, so functional requests must use an allowed host.
LOCAL_BASE_URL = "http://127.0.0.1:8000"

_INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-03-26",
        "capabilities": {},
        "clientInfo": {"name": "omega-tests", "version": "1"},
    },
}


def _parse_mcp_response(response) -> dict:
    """Streamable HTTP may answer as JSON or as a single SSE ``data:`` frame."""
    text = response.text
    if "application/json" in response.headers.get("content-type", ""):
        return response.json()
    for line in text.splitlines():
        if line.startswith("data: "):
            return json.loads(line[len("data: ") :])
    raise AssertionError(f"Unparseable MCP response: {text!r}")


def test_build_server_smoke():
    assert build_server() is not None


def test_http_app_health_and_transports():
    app = build_http_app()
    client = TestClient(app)

    assert client.get("/healthz").json() == {"status": "ok"}
    assert {route.path for route in app.routes} >= {"/healthz", "/sse", "/mcp"}


def test_http_app_streamable_initialize_handshake():
    """A real MCP initialize must succeed over /mcp (regression: the session
    manager lifespan was never started and the route was doubled to /mcp/mcp)."""
    app = build_http_app()
    with TestClient(app, base_url=LOCAL_BASE_URL) as client:
        response = client.post(
            "/mcp",
            json=_INITIALIZE,
            headers={
                "content-type": "application/json",
                "accept": "application/json, text/event-stream",
            },
        )

    assert response.status_code == 200
    assert response.headers.get("mcp-session-id")
    payload = _parse_mcp_response(response)
    result = payload["result"]
    assert result["serverInfo"]["name"] == "Omega"
    assert result["protocolVersion"]


def test_http_app_transports_are_not_double_mounted():
    """Mounting must not duplicate the transport path (was /mcp/mcp, /sse/sse)."""
    app = build_http_app()
    with TestClient(app, base_url=LOCAL_BASE_URL) as client:
        # /sse resolves to the SSE handler (307 to /sse/), not a 404.
        assert client.get(
            "/sse",
            headers={"accept": "text/event-stream"},
            follow_redirects=False,
        ).status_code == 307
        # The old doubled paths must no longer exist.
        assert client.post("/mcp/mcp", json={}, follow_redirects=False).status_code == 404
        assert client.get("/sse/sse", follow_redirects=False).status_code == 404


def test_http_app_allowed_cors_origin():
    client = TestClient(build_http_app())

    response = client.options(
        "/mcp",
        headers={
            "origin": "http://localhost:3000",
            "access-control-request-method": "POST",
            "access-control-request-headers": "content-type",
        },
    )

    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_http_app_disallowed_cors_origin_gets_no_allow_origin_header():
    client = TestClient(build_http_app())

    response = client.options(
        "/mcp",
        headers={
            "origin": "https://example.com",
            "access-control-request-method": "POST",
            "access-control-request-headers": "content-type",
        },
    )

    assert "access-control-allow-origin" not in response.headers


def test_wildcard_cors_with_credentials_is_rejected():
    with pytest.raises(RuntimeError, match="must not include"):
        _parse_cors_origins("*")
