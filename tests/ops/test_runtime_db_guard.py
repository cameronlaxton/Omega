from __future__ import annotations

import logging

import pytest

from omega.ops import runtime_db_guard


def _status(**overrides):
    base = {
        "requested": None,
        "env_override": None,
        "repo_default_path": "C:/repos/Omega/var/omega_traces.db",
        "effective_path": "C:/Users/test/.omega/workspace/Omega/var/omega_traces.db",
        "source": "default",
        "would_be_runtime_path": "C:/Users/test/AppData/Local/omega/runtime/var/omega_traces.db",
        "default_exists": True,
        "effective_exists": True,
        "runtime_exists": False,
        "default_integrity_ok": True,
        "effective_integrity_ok": True,
        "default_trace_count": 12,
        "effective_trace_count": 12,
        "latest_session_ids": [],
        "schema_version": None,
        "divergence": None,
        "empty_history_mode": False,
        "recommended_action": "ok",
    }
    base.update(overrides)
    return base


def test_assert_safe_runtime_db_passes_safe_status(monkeypatch):
    status = _status(source="env_override")
    monkeypatch.setattr(runtime_db_guard, "db_status", lambda requested=None: status)

    assert runtime_db_guard.assert_safe_runtime_db("x.db") is status


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"source": "auto_redirect_network_fs"}, "auto-redirected"),
        ({"divergence": {"source_trace_count": 4, "runtime_trace_count": 2}}, "divergence"),
        ({"effective_exists": False, "effective_trace_count": None}, "missing"),
        ({"effective_integrity_ok": False}, "integrity"),
        ({"effective_trace_count": None}, "trace count is unavailable"),
        ({"effective_trace_count": 0}, "zero traces"),
    ],
)
def test_assert_safe_runtime_db_raises_for_unsafe_status(monkeypatch, overrides, expected):
    monkeypatch.setattr(runtime_db_guard, "db_status", lambda requested=None: _status(**overrides))

    with pytest.raises(runtime_db_guard.UnsafeRuntimeDbError, match=expected):
        runtime_db_guard.assert_safe_runtime_db()


def test_assert_safe_runtime_db_dry_run_warns_without_raising(monkeypatch, caplog):
    status = _status(source="auto_redirect_network_fs")
    monkeypatch.setattr(runtime_db_guard, "db_status", lambda requested=None: status)

    with caplog.at_level(logging.WARNING, logger="omega.ops.runtime_db_guard"):
        assert runtime_db_guard.assert_safe_runtime_db(dry_run=True) is status

    assert "DRY-RUN runtime DB warning" in caplog.text
    assert "auto-redirected" in caplog.text


@pytest.mark.parametrize(
    "overrides",
    [
        {"effective_exists": False, "effective_integrity_ok": None, "effective_trace_count": None},
        {"effective_trace_count": 0},
    ],
)
def test_empty_history_mode_allows_intentional_empty_db(monkeypatch, caplog, overrides):
    status = _status(**overrides)
    monkeypatch.setenv("OMEGA_ALLOW_EMPTY_DB", "1")
    monkeypatch.setattr(runtime_db_guard, "db_status", lambda requested=None: status)

    with caplog.at_level(logging.WARNING, logger="omega.ops.runtime_db_guard"):
        assert runtime_db_guard.assert_safe_runtime_db() is status

    assert "EMPTY_HISTORY_MODE=true" in caplog.text
    assert status["effective_path"] in caplog.text
    assert f"source={status['source']}" in caplog.text


def test_empty_history_mode_does_not_allow_auto_redirect(monkeypatch):
    monkeypatch.setenv("OMEGA_ALLOW_EMPTY_DB", "1")
    monkeypatch.setattr(
        runtime_db_guard,
        "db_status",
        lambda requested=None: _status(source="auto_redirect_network_fs", effective_trace_count=0),
    )

    with pytest.raises(runtime_db_guard.UnsafeRuntimeDbError, match="auto-redirected"):
        runtime_db_guard.assert_safe_runtime_db()
