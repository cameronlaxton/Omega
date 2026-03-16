"""
Tests for the Omega API layer.

Tests session management, health endpoint, and analysis endpoints.
Uses FastAPI TestClient (no real server needed).
"""

import pytest


class TestSessionManager:
    """Test in-memory session management."""

    def test_create_session(self):
        from omega.api.session.manager import SessionManager

        sm = SessionManager()
        session = sm.create_session()
        assert session.session_id
        assert session.messages == []

    def test_get_or_create_new(self):
        from omega.api.session.manager import SessionManager

        sm = SessionManager()
        session = sm.get_or_create()
        assert session.session_id

    def test_get_or_create_existing(self):
        from omega.api.session.manager import SessionManager

        sm = SessionManager()
        s1 = sm.create_session("test-123")
        s2 = sm.get_or_create("test-123")
        assert s1.session_id == s2.session_id

    def test_add_message(self):
        from omega.api.session.manager import SessionManager

        sm = SessionManager()
        session = sm.create_session()
        msg = session.add_message("user", "Hello")
        assert msg["role"] == "user"
        assert msg["content"] == "Hello"
        assert msg["timestamp"]
        assert len(session.messages) == 1

    def test_session_persistence(self):
        from omega.api.session.manager import SessionManager

        sm = SessionManager()
        session = sm.create_session("persist-test")
        session.add_message("user", "test message")
        sm.save_session(session)

        retrieved = sm.get_session("persist-test")
        assert retrieved is not None
        assert len(retrieved.messages) == 1

    def test_delete_session(self):
        from omega.api.session.manager import SessionManager

        sm = SessionManager()
        sm.create_session("delete-me")
        assert sm.delete_session("delete-me") is True
        assert sm.get_session("delete-me") is None

    def test_session_serialization(self):
        from omega.api.session.manager import Session

        session = Session("ser-test")
        session.add_message("user", "hello")
        session.add_message("assistant", "hi there")

        data = session.to_dict()
        restored = Session.from_dict(data)
        assert restored.session_id == "ser-test"
        assert len(restored.messages) == 2


class TestHealthEndpoint:
    """Test health endpoint via TestClient."""

    def test_health_returns_ok(self):
        from fastapi.testclient import TestClient
        from omega.api.app import app

        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["version"] == "0.1.0"


class TestAnalysisEndpoints:
    """Test deterministic analysis endpoints."""

    def test_analyze_game(self):
        from fastapi.testclient import TestClient
        from omega.api.app import app

        client = TestClient(app)
        resp = client.post("/api/v1/analyze/game", json={
            "home_team": "Boston Celtics",
            "away_team": "Indiana Pacers",
            "league": "NBA",
            "n_iterations": 100,
            "home_context": {"off_rating": 118.0, "def_rating": 108.0, "pace": 100.0},
            "away_context": {"off_rating": 115.0, "def_rating": 110.0, "pace": 98.0},
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert "simulation" in body
        assert body["simulation"]["iterations"] == 100

    def test_analyze_game_missing_context(self):
        from fastapi.testclient import TestClient
        from omega.api.app import app

        client = TestClient(app)
        resp = client.post("/api/v1/analyze/game", json={
            "home_team": "Team A",
            "away_team": "Team B",
            "league": "NBA",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "skipped"


class TestChatEndpoint:
    """Test chat SSE endpoint."""

    def test_chat_returns_sse(self):
        from fastapi.testclient import TestClient
        from omega.api.app import app

        client = TestClient(app)
        resp = client.post("/chat", json={
            "message": "Who should I bet on in Lakers vs Celtics?",
        })
        # SSE returns 200 with text/event-stream
        assert resp.status_code == 200


class TestAPISchemas:
    """Test API-layer schemas."""

    def test_chat_request_validation(self):
        from omega.api.schemas import ChatRequest

        req = ChatRequest(message="test")
        assert req.session_id is None
        assert req.message == "test"

    def test_chat_request_rejects_empty(self):
        from omega.api.schemas import ChatRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ChatRequest(message="")

    def test_health_response(self):
        from omega.api.schemas import HealthResponse

        resp = HealthResponse()
        assert resp.status == "ok"
