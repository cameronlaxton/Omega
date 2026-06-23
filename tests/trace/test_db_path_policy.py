"""
Tests for the DB-path policy: read-only db_status, the FUSE-redirect empty-history
guard (fail-loud, no auto-mutation), and explicit seed_runtime_db.

These pin the fix for the 0-trace-vs-7-trace discrepancy: a redirected runtime DB
must never silently become believable-empty history.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from omega.ops import db_status as db_status_cli
from omega.trace import store as store_mod
from omega.trace.store import TraceStore, _raw_trace_count, db_status, seed_runtime_db


def _trace(trace_id: str) -> dict[str, Any]:
    return {
        "trace_id": trace_id,
        "run_id": "r",
        "timestamp": "2026-05-28T00:00:00Z",
        "prompt": "p",
        "league": "NBA",
        "matchup": "A @ B",
        "execution_mode": "native_sim",
        "simulation_seed": 1,
        "aggregate_quality": 0.8,
        "predictions": {"home_win_prob": 0.6, "away_win_prob": 0.4},
        "recommendations": [],
        "odds_snapshot": {},
        "downgrades": [],
        "kind": "game",
        "result": {"status": "success"},
        "trace_quality": {"calibration_eligible": True},
    }


def _make_db(path: Path, n: int) -> None:
    s = TraceStore(db_path=str(path))
    for i in range(n):
        s.persist(_trace(f"sandbox-seed-{i}"))
    s.close()


def _force_redirect(monkeypatch, source: Path, runtime: Path) -> None:
    monkeypatch.delenv("OMEGA_TRACE_DB", raising=False)
    monkeypatch.setattr(store_mod, "_repo_default_db_path", lambda: str(source))
    monkeypatch.setattr(store_mod, "_local_runtime_db_path", lambda: runtime)
    monkeypatch.setattr(store_mod, "_is_network_filesystem", lambda _p: True)


class TestRedirectGuard:
    def test_valid_nonempty_source_absent_runtime_raises(self, tmp_path, monkeypatch):
        source = tmp_path / "mount" / "var/omega_traces.db"
        source.parent.mkdir(parents=True)
        _make_db(source, 3)
        runtime = tmp_path / "rt" / "var/omega_traces.db"
        _force_redirect(monkeypatch, source, runtime)
        monkeypatch.delenv("OMEGA_ALLOW_EMPTY_DB", raising=False)

        with pytest.raises(RuntimeError, match="omega-db-status --seed"):
            TraceStore(db_path=None)
        assert not runtime.exists(), "guard must not create an empty runtime DB"

    def test_missing_source_raises(self, tmp_path, monkeypatch):
        source = tmp_path / "mount" / "var/omega_traces.db"  # never created
        runtime = tmp_path / "rt" / "var/omega_traces.db"
        _force_redirect(monkeypatch, source, runtime)
        monkeypatch.delenv("OMEGA_ALLOW_EMPTY_DB", raising=False)

        with pytest.raises(RuntimeError, match="missing"):
            TraceStore(db_path=None)
        assert not runtime.exists()

    def test_malformed_source_raises(self, tmp_path, monkeypatch):
        source = tmp_path / "mount" / "var/omega_traces.db"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"this is not a sqlite database")
        runtime = tmp_path / "rt" / "var/omega_traces.db"
        _force_redirect(monkeypatch, source, runtime)
        monkeypatch.delenv("OMEGA_ALLOW_EMPTY_DB", raising=False)

        with pytest.raises(RuntimeError, match="integrity_check"):
            TraceStore(db_path=None)
        assert not runtime.exists()

    def test_allow_empty_db_opens_empty_runtime(self, tmp_path, monkeypatch):
        source = tmp_path / "mount" / "var/omega_traces.db"  # missing
        runtime = tmp_path / "rt" / "var/omega_traces.db"
        _force_redirect(monkeypatch, source, runtime)
        monkeypatch.setenv("OMEGA_ALLOW_EMPTY_DB", "1")

        store = TraceStore(db_path=None)
        try:
            assert store.empty_history_mode is True
            assert store.count() == 0
        finally:
            store.close()
        assert runtime.exists()


class TestSeedRuntimeDb:
    def test_seed_copies_trace_count(self, tmp_path):
        source = tmp_path / "src.db"
        _make_db(source, 4)
        runtime = tmp_path / "rt" / "var/omega_traces.db"
        result = seed_runtime_db(str(source), str(runtime))
        assert result["trace_count"] == 4
        assert runtime.exists()
        assert _raw_trace_count(str(runtime)) == 4

    def test_seed_refuses_to_overwrite_existing_runtime(self, tmp_path):
        source = tmp_path / "src.db"
        _make_db(source, 3)
        runtime = tmp_path / "rt.db"
        _make_db(runtime, 1)
        with pytest.raises(RuntimeError, match="overwrite"):
            seed_runtime_db(str(source), str(runtime))
        assert _raw_trace_count(str(runtime)) == 1  # untouched

    def test_seed_refuses_empty_source(self, tmp_path):
        source = tmp_path / "empty.db"
        TraceStore(db_path=str(source)).close()  # schema only, 0 traces
        with pytest.raises(RuntimeError, match="empty"):
            seed_runtime_db(str(source), str(tmp_path / "rt.db"))


class TestDbStatus:
    def test_reports_fields_and_is_read_only(self, tmp_path, monkeypatch):
        source = tmp_path / "src.db"
        _make_db(source, 3)
        st = db_status(str(source))
        assert st["effective_path"] == str(source)
        assert st["effective_trace_count"] == 3
        assert st["effective_integrity_ok"] is True
        assert st["source"] == "requested"

    def test_does_not_create_runtime_and_recommends_seed(self, tmp_path, monkeypatch):
        source = tmp_path / "mount" / "var/omega_traces.db"
        source.parent.mkdir(parents=True)
        _make_db(source, 3)
        runtime = tmp_path / "rt" / "var/omega_traces.db"
        _force_redirect(monkeypatch, source, runtime)

        st = db_status(None)
        assert st["source"] == "auto_redirect_network_fs"
        assert not runtime.exists(), "db_status must be read-only"
        assert "seed" in st["recommended_action"]

    def test_divergence_reported_no_merge(self, tmp_path, monkeypatch):
        source = tmp_path / "mount" / "var/omega_traces.db"
        source.parent.mkdir(parents=True)
        _make_db(source, 5)
        runtime = tmp_path / "rt" / "var/omega_traces.db"
        runtime.parent.mkdir(parents=True)
        _make_db(runtime, 2)
        _force_redirect(monkeypatch, source, runtime)

        st = db_status(None)
        assert st["divergence"] is not None
        assert st["divergence"]["source_trace_count"] == 5
        assert st["divergence"]["runtime_trace_count"] == 2

    def test_query_traces_prints_identity_header(self, tmp_path, capsys):
        source = tmp_path / "traces.db"
        _make_db(source, 2)

        code = db_status_cli.main(["--db", str(source), "--query-traces", "--limit", "5"])

        assert code == 0
        first = capsys.readouterr().out.splitlines()[0]
        assert first.startswith("TraceStore DB Path: ")
        assert str(source.resolve()) in first

    def test_query_traces_json_has_identity_metadata(self, tmp_path, capsys):
        source = tmp_path / "traces.db"
        _make_db(source, 1)

        code = db_status_cli.main(
            ["--db", str(source), "--query-traces", "--limit", "5", "--format", "json"]
        )

        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["trace_store_db_path"] == str(source.resolve())
        assert payload["trace_store_db_source"] == "requested"
        assert payload["workspace_identity"]
        assert len(payload["traces"]) == 1

    def test_view_ledger_reads_existing_rows(self, tmp_path, capsys):
        source = tmp_path / "ledger.db"
        _make_db(source, 1)
        store = TraceStore(db_path=str(source))
        try:
            store.conn.execute(
                """INSERT INTO bet_ledger (
                    ledger_id, trace_id, bet_date, league, sport, matchup,
                    market, bookmaker, selection, selection_descriptor, line,
                    odds, stake_amount, status, provenance, decision_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    "ledger-1",
                    "sandbox-seed-0",
                    "2026-06-02",
                    "NBA",
                    "basketball",
                    "A @ B",
                    "moneyline",
                    "betmgm",
                    "home",
                    "home",
                    None,
                    -110,
                    25.0,
                    "pending",
                    "user_confirmed",
                    "2026-06-02T00:00:00Z",
                ),
            )
            store.conn.commit()
        finally:
            store.close()

        code = db_status_cli.main(
            ["--db", str(source), "--view-ledger", "--limit", "5", "--format", "json"]
        )

        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["query"] == "ledger"
        assert payload["count"] == 1
        assert payload["ledger"][0]["ledger_id"] == "ledger-1"


class TestRuntimeDirPrecedence:
    def test_default_db_path_respects_omega_runtime_dir(self, tmp_path, monkeypatch):
        from omega.trace import store as store_mod

        runtime = tmp_path / "runtime"
        monkeypatch.setenv("OMEGA_RUNTIME_DIR", str(runtime))
        monkeypatch.delenv("OMEGA_TRACE_DB", raising=False)
        monkeypatch.setattr(store_mod, "_is_network_filesystem", lambda _p: False)

        store = TraceStore(db_path=None)
        try:
            assert store.db_path_source == "default"
            assert Path(store.db_path) == runtime / "omega_traces.db"
        finally:
            store.close()

    def test_omega_trace_db_wins_over_omega_runtime_dir(self, tmp_path, monkeypatch):
        runtime = tmp_path / "runtime"
        env_db = tmp_path / "env" / "override.db"
        monkeypatch.setenv("OMEGA_RUNTIME_DIR", str(runtime))
        monkeypatch.setenv("OMEGA_TRACE_DB", str(env_db))

        store = TraceStore(db_path=None)
        try:
            assert store.db_path_source == "env_override"
            assert Path(store.db_path) == env_db
        finally:
            store.close()

    def test_explicit_db_wins_over_omega_runtime_dir(self, tmp_path, monkeypatch):
        from omega.trace import store as store_mod

        runtime = tmp_path / "runtime"
        explicit_db = tmp_path / "explicit.db"
        monkeypatch.setenv("OMEGA_RUNTIME_DIR", str(runtime))
        monkeypatch.delenv("OMEGA_TRACE_DB", raising=False)
        monkeypatch.setattr(store_mod, "_is_network_filesystem", lambda _p: False)

        store = TraceStore(db_path=str(explicit_db))
        try:
            assert store.db_path_source == "requested"
            assert Path(store.db_path) == explicit_db
        finally:
            store.close()

    def test_default_db_path_still_uses_omega_traces_filename(self, tmp_path, monkeypatch):
        from omega.paths import trace_db_path

        runtime = tmp_path / "runtime"
        monkeypatch.setenv("OMEGA_RUNTIME_DIR", str(runtime))

        assert trace_db_path() == runtime / "omega_traces.db"
