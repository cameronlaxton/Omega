"""Bind policy + bearer-token enforcement for the console (Milestone A).

Loopback is zero-config; any non-loopback bind must fail closed unless it is
explicitly opted into AND token-protected.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from omega.ops import console_server
from omega.ops.console_server import (
    DEFAULT_HOST,
    build_console_app,
    is_loopback_host,
    resolve_bind_policy,
)


def test_default_host_is_loopback():
    assert DEFAULT_HOST == "127.0.0.1"
    assert is_loopback_host("127.0.0.1") is True
    assert is_loopback_host("localhost") is True
    assert is_loopback_host("::1") is True


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10", "10.0.0.5", "example.com"])
def test_non_loopback_hosts_detected(host):
    assert is_loopback_host(host) is False


def test_loopback_requires_no_token():
    # No token configured, no opt-in needed: returns None (no gate).
    assert resolve_bind_policy("127.0.0.1", allow_remote=False, token=None) is None
    # A token on loopback is honored but not required.
    assert resolve_bind_policy("127.0.0.1", allow_remote=False, token="t") == "t"


def test_non_loopback_without_optin_fails_closed():
    with pytest.raises(RuntimeError, match="non-loopback"):
        resolve_bind_policy("0.0.0.0", allow_remote=False, token="t")


def test_non_loopback_without_token_fails_closed():
    with pytest.raises(RuntimeError, match="OMEGA_CONSOLE_TOKEN"):
        resolve_bind_policy("0.0.0.0", allow_remote=True, token=None)


def test_non_loopback_with_optin_and_token_returns_token():
    assert resolve_bind_policy("0.0.0.0", allow_remote=True, token="secret") == "secret"


def test_main_refuses_non_loopback_without_optin(monkeypatch):
    # main() must raise before any server starts.
    monkeypatch.delenv("OMEGA_CONSOLE_ALLOW_REMOTE", raising=False)
    monkeypatch.delenv("OMEGA_CONSOLE_TOKEN", raising=False)

    def _fail_run(*a, **k):  # pragma: no cover - must never be reached
        raise AssertionError("uvicorn.run must not be called when bind is refused")

    monkeypatch.setattr(console_server, "uvicorn", type("U", (), {"run": staticmethod(_fail_run)}), raising=False)
    with pytest.raises(RuntimeError):
        console_server.main(["--host", "0.0.0.0"])


# --- middleware behavior on a token-protected (non-loopback-style) app ---


@pytest.fixture
def token_client(seeded) -> TestClient:
    app = build_console_app(
        db_path=seeded["db_path"],
        sessions_dir=str(seeded["sessions_dir"]),
        auth_token="secret-token",
    )
    return TestClient(app)


def test_protected_route_401_without_token(token_client):
    assert token_client.get("/api/traces").status_code == 401


def test_protected_route_401_with_wrong_token(token_client):
    resp = token_client.get("/api/traces", headers={"Authorization": "Bearer nope"})
    assert resp.status_code == 401


def test_protected_route_200_with_token(token_client):
    resp = token_client.get("/api/traces", headers={"Authorization": "Bearer secret-token"})
    assert resp.status_code == 200


def test_healthz_open_even_when_token_required(token_client):
    assert token_client.get("/healthz").status_code == 200
    assert token_client.get("/api/healthz").status_code == 401
    assert token_client.get(
        "/api/healthz",
        headers={"Authorization": "Bearer secret-token"},
    ).status_code == 200


def test_html_pages_gated_when_token_required(token_client):
    assert token_client.get("/traces").status_code == 401
    assert token_client.get("/traces", headers={"Authorization": "Bearer secret-token"}).status_code == 200


def test_loopback_app_has_no_auth_middleware(app):
    # The default (loopback) app installs no bearer middleware.
    names = [m.cls.__name__ for m in app.user_middleware]
    assert "_BearerAuthMiddleware" not in names
