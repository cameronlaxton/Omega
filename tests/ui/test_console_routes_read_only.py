"""The console must expose only GET routes — no mutation surface (Milestone A)."""

from __future__ import annotations

from starlette.routing import Mount

MUTATING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


def _api_routes(app):
    for route in app.routes:
        methods = getattr(route, "methods", None)
        if methods is not None and not isinstance(route, Mount):
            yield route, set(methods)


def test_no_mutating_methods_registered(app):
    offenders = [
        (route.path, methods & MUTATING_METHODS)
        for route, methods in _api_routes(app)
        if methods & MUTATING_METHODS
    ]
    assert offenders == [], f"console exposes mutating routes: {offenders}"


def test_every_route_is_get(app):
    # Every concrete (non-mount) route is GET-only (HEAD is a harmless read alias).
    for route, methods in _api_routes(app):
        assert methods <= {"GET", "HEAD"}, f"{route.path} allows {methods}"


def test_healthz_returns_200(client):
    resp = client.get("/api/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["read_only"] is True


def test_post_to_existing_resource_is_405(client):
    # The routers define GET handlers only, so a POST must be rejected by the
    # router (405), never silently accepted.
    for path in ("/api/traces", "/api/bets", "/api/sessions"):
        resp = client.post(path, json={})
        assert resp.status_code == 405, f"POST {path} -> {resp.status_code}"
    for method in ("put", "patch", "delete"):
        resp = getattr(client, method)("/api/traces/sandbox-aaa")
        assert resp.status_code == 405, f"{method.upper()} -> {resp.status_code}"


def test_future_milestone_endpoints_absent(client):
    # Placeholder nav (Diagnostics, Calibration, CLV, Review Queue, Signal
    # Performance) must NOT be backed by working endpoints in Milestone A.
    for path in (
        "/api/diagnostics",
        "/api/calibration",
        "/api/calibration/status",
        "/api/clv",
        "/api/market-movement",
        "/api/review-queue",
        "/api/signals",
        "/api/signal-performance",
    ):
        assert client.get(path).status_code == 404, f"{path} unexpectedly exists"


def test_no_permissive_cors(app):
    # No CORS middleware is installed at all (no cross-origin enablement).
    names = [m.cls.__name__ for m in app.user_middleware]
    assert "CORSMiddleware" not in names


def test_open_paths_are_registered_routes(app):
    # Every path the bearer gate treats as open must be a real route, so the
    # operator-facing "open" advertisement is honest.
    from omega.ops.console_server import _OPEN_PATHS

    registered = {getattr(r, "path", None) for r in app.routes}
    missing = [p for p in _OPEN_PATHS if p not in registered]
    assert missing == [], f"_OPEN_PATHS references unregistered routes: {missing}"
