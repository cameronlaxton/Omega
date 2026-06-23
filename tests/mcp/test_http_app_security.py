"""Bind-policy and bearer-auth tests for the Omega MCP HTTP transport.

These guard the hardening that keeps the HTTP transport loopback-only by
default: a non-loopback bind must refuse without an explicit opt-in + token, a
loopback bind must stay zero-config, and the bearer gate must protect the
state-mutating transports without breaking local usage.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from omega.mcp.http_app import (
    _enforce_bind_policy,
    _is_loopback_host,
    build_http_app,
    run_http,
)

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


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Each test controls the transport env explicitly; start from a clean slate."""
    for name in (
        "OMEGA_MCP_HOST",
        "OMEGA_MCP_PORT",
        "OMEGA_MCP_ALLOW_REMOTE",
        "OMEGA_MCP_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------- #
# Loopback classification
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "host",
    ["127.0.0.1", "127.0.0.5", "::1", "[::1]", "localhost", "LOCALHOST", " 127.0.0.1 "],
)
def test_is_loopback_host_true(host):
    assert _is_loopback_host(host) is True


@pytest.mark.parametrize(
    "host",
    ["0.0.0.0", "::", "192.168.1.10", "10.0.0.4", "example.com", ""],
)
def test_is_loopback_host_false(host):
    assert _is_loopback_host(host) is False


# --------------------------------------------------------------------------- #
# Bind policy (_enforce_bind_policy / run_http)
# --------------------------------------------------------------------------- #
def test_loopback_starts_with_no_token():
    # (b) loopback host starts with no token configured.
    assert _enforce_bind_policy("127.0.0.1") is None
    assert _enforce_bind_policy("localhost") is None
    assert _enforce_bind_policy("::1") is None


def test_non_loopback_without_opt_in_refuses():
    # (a) non-loopback host without OMEGA_MCP_ALLOW_REMOTE refuses.
    with pytest.raises(RuntimeError, match="non-loopback"):
        _enforce_bind_policy("0.0.0.0")


def test_non_loopback_opt_in_without_token_refuses(monkeypatch):
    monkeypatch.setenv("OMEGA_MCP_ALLOW_REMOTE", "1")
    with pytest.raises(RuntimeError, match="OMEGA_MCP_TOKEN"):
        _enforce_bind_policy("0.0.0.0")


def test_non_loopback_opt_in_with_token_allows(monkeypatch, caplog):
    monkeypatch.setenv("OMEGA_MCP_ALLOW_REMOTE", "1")
    monkeypatch.setenv("OMEGA_MCP_TOKEN", "s3cret")
    with caplog.at_level("WARNING", logger="omega.mcp.http"):
        assert _enforce_bind_policy("0.0.0.0") == "s3cret"
    # Loud warning naming the state-mutating tools.
    assert any("NON-LOOPBACK" in r.message for r in caplog.records)
    assert any("omega_run_batch" in r.message for r in caplog.records)


def test_run_http_loopback_starts_without_token(monkeypatch):
    """run_http on the default loopback host builds an app and starts uvicorn,
    with no token required (regression guard for zero-config local usage)."""
    captured = {}

    def fake_run(app, **kwargs):
        captured["app"] = app
        captured["kwargs"] = kwargs

    monkeypatch.setattr("uvicorn.run", fake_run)
    run_http()

    assert captured["kwargs"]["host"] == "127.0.0.1"
    assert captured["app"] is not None


def test_run_http_non_loopback_refuses_and_never_binds(monkeypatch):
    monkeypatch.setenv("OMEGA_MCP_HOST", "0.0.0.0")

    called = {"run": False}

    def fake_run(app, **kwargs):  # pragma: no cover - must not be reached
        called["run"] = True

    monkeypatch.setattr("uvicorn.run", fake_run)
    with pytest.raises(RuntimeError, match="non-loopback"):
        run_http()
    assert called["run"] is False


# --------------------------------------------------------------------------- #
# Bearer-auth middleware
# --------------------------------------------------------------------------- #
def _parse_mcp_response(response) -> dict:
    if "application/json" in response.headers.get("content-type", ""):
        return response.json()
    for line in response.text.splitlines():
        if line.startswith("data: "):
            return json.loads(line[len("data: ") :])
    raise AssertionError(f"Unparseable MCP response: {response.text!r}")


def test_no_middleware_when_no_token():
    """Loopback default: no token => /mcp reachable without auth."""
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


def test_auth_required_rejects_missing_token():
    app = build_http_app(auth_token="s3cret")
    with TestClient(app, base_url=LOCAL_BASE_URL) as client:
        response = client.post(
            "/mcp",
            json=_INITIALIZE,
            headers={
                "content-type": "application/json",
                "accept": "application/json, text/event-stream",
            },
        )
    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"


def test_auth_required_rejects_wrong_token():
    app = build_http_app(auth_token="s3cret")
    with TestClient(app, base_url=LOCAL_BASE_URL) as client:
        response = client.post(
            "/mcp",
            json=_INITIALIZE,
            headers={
                "content-type": "application/json",
                "accept": "application/json, text/event-stream",
                "authorization": "Bearer wrong",
            },
        )
    assert response.status_code == 401


def test_auth_required_accepts_valid_token():
    app = build_http_app(auth_token="s3cret")
    with TestClient(app, base_url=LOCAL_BASE_URL) as client:
        response = client.post(
            "/mcp",
            json=_INITIALIZE,
            headers={
                "content-type": "application/json",
                "accept": "application/json, text/event-stream",
                "authorization": "Bearer s3cret",
            },
        )
    assert response.status_code == 200
    payload = _parse_mcp_response(response)
    assert payload["result"]["serverInfo"]["name"] == "Omega"


def test_auth_token_read_from_env(monkeypatch):
    monkeypatch.setenv("OMEGA_MCP_TOKEN", "envtok")
    app = build_http_app()  # no explicit token => read from env
    with TestClient(app, base_url=LOCAL_BASE_URL) as client:
        unauthed = client.post(
            "/mcp",
            json=_INITIALIZE,
            headers={
                "content-type": "application/json",
                "accept": "application/json, text/event-stream",
            },
        )
    assert unauthed.status_code == 401


def test_healthz_open_without_token():
    """/healthz stays open so liveness probes work even with auth enabled."""
    app = build_http_app(auth_token="s3cret")
    client = TestClient(app)
    assert client.get("/healthz").json() == {"status": "ok"}


def test_cors_preflight_allowed_without_token():
    """OPTIONS preflight must pass the auth gate so browsers can negotiate CORS."""
    app = build_http_app(auth_token="s3cret")
    client = TestClient(app)
    response = client.options(
        "/mcp",
        headers={
            "origin": "http://localhost:3000",
            "access-control-request-method": "POST",
            "access-control-request-headers": "authorization,content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
