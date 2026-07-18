"""Phase 0 ledger policy: engine_auto autolog is disabled by default.

Verification-plan coverage (design §17, ledger tests):
- disabled mode produces no engine_auto rows;
- environment variables alone cannot enable shadow writes;
- explicit shadow mode requires BOTH the per-run mode and the env gate;
- the legacy OMEGA_BET_LEDGER_AUTOLOG variable is a kill switch only;
- scoped suppression still wins over everything;
- SQLite and Postgres share one decision function (parity by construction);
- user_confirmed recording is unaffected;
- calibration eligibility never depends on a ledger row.
"""

from __future__ import annotations

import ast
import tempfile
from pathlib import Path
from typing import Any

import pytest

from omega.trace.autolog_policy import (
    ENV_ENGINE_SHADOW,
    ENV_LEGACY_AUTOLOG_KILL_SWITCH,
    engine_auto_autolog_decision,
)
from omega.trace.ledger_bet import BetProvenance, LedgerBet, LedgerStatus
from omega.trace.store import TraceStore

_SRC = Path(__file__).resolve().parents[2] / "src" / "omega" / "trace"


def _shadow_trace(trace_id: str = "t-shadow", **overrides: Any) -> dict[str, Any]:
    """A trace whose recommendation is extractable into an engine_auto bet."""
    base: dict[str, Any] = {
        "trace_id": trace_id,
        "run_id": f"r-{trace_id}",
        "timestamp": "2026-07-16T00:00:00Z",
        "kind": "prop",
        "prompt": "MLB prop",
        "league": "MLB",
        "matchup": "Mets @ Mariners",
        "execution_mode": "sandbox_prop",
        "simulation_seed": 1,
        "aggregate_quality": 0.9,
        "downgrades": [],
        "input_snapshot": {
            "league": "MLB",
            "player_name": "Julio Rodriguez",
            "prop_type": "hits",
            "line": 1.5,
            "odds_over": -110,
            "odds_under": -110,
            "home_team": "Seattle Mariners",
            "away_team": "New York Mets",
            "game_date": "2026-07-16",
        },
        "predictions": {"over_prob": 0.6, "under_prob": 0.4},
        "result": {
            "status": "success",
            "recommendation": "over",
            "confidence_tier": "B",
            "bet_side_odds": -110,
        },
        "trace_quality": {"aggregate_quality": 0.9},
    }
    base.update(overrides)
    return base


def _tmp_store() -> TraceStore:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return TraceStore(db_path=tmp.name)


def _engine_auto_rows(store: TraceStore, trace_id: str) -> list[Any]:
    return store.conn.execute(
        "SELECT * FROM bet_ledger WHERE trace_id = ? AND provenance = 'engine_auto'",
        (trace_id,),
    ).fetchall()


class TestDecisionFunction:
    def test_default_is_disabled(self, monkeypatch):
        monkeypatch.delenv(ENV_ENGINE_SHADOW, raising=False)
        monkeypatch.delenv(ENV_LEGACY_AUTOLOG_KILL_SWITCH, raising=False)
        allowed, reason = engine_auto_autolog_decision({})
        assert not allowed
        assert reason == "engine_auto_ledger_mode_disabled"

    def test_legacy_env_alone_cannot_enable(self, monkeypatch):
        # The historical default-on enabler must be inert as an enabler.
        monkeypatch.setenv(ENV_LEGACY_AUTOLOG_KILL_SWITCH, "1")
        monkeypatch.delenv(ENV_ENGINE_SHADOW, raising=False)
        allowed, reason = engine_auto_autolog_decision({})
        assert not allowed
        assert reason == "engine_auto_ledger_mode_disabled"

    def test_shadow_env_alone_cannot_enable(self, monkeypatch):
        monkeypatch.setenv(ENV_ENGINE_SHADOW, "1")
        allowed, reason = engine_auto_autolog_decision({})
        assert not allowed
        assert reason == "engine_auto_ledger_mode_disabled"

    def test_shadow_mode_without_env_gate_is_denied(self, monkeypatch):
        monkeypatch.delenv(ENV_ENGINE_SHADOW, raising=False)
        allowed, reason = engine_auto_autolog_decision(
            {"engine_auto_ledger_mode": "shadow"}
        )
        assert not allowed
        assert reason == "shadow_env_not_enabled"

    def test_shadow_mode_plus_env_gate_is_allowed(self, monkeypatch):
        monkeypatch.setenv(ENV_ENGINE_SHADOW, "1")
        allowed, reason = engine_auto_autolog_decision(
            {"engine_auto_ledger_mode": "shadow"}
        )
        assert allowed
        assert reason == "shadow_enabled"

    def test_legacy_env_is_kill_switch(self, monkeypatch):
        monkeypatch.setenv(ENV_ENGINE_SHADOW, "1")
        monkeypatch.setenv(ENV_LEGACY_AUTOLOG_KILL_SWITCH, "0")
        allowed, reason = engine_auto_autolog_decision(
            {"engine_auto_ledger_mode": "shadow"}
        )
        assert not allowed
        assert reason == "legacy_kill_switch"

    def test_scoped_suppression_wins_over_everything(self, monkeypatch):
        monkeypatch.setenv(ENV_ENGINE_SHADOW, "1")
        allowed, reason = engine_auto_autolog_decision(
            {"engine_auto_ledger_mode": "shadow"}, suppressed=True
        )
        assert not allowed
        assert reason == "scoped_suppression"

    def test_unrecognized_mode_fails_closed(self, monkeypatch):
        monkeypatch.setenv(ENV_ENGINE_SHADOW, "1")
        allowed, reason = engine_auto_autolog_decision(
            {"engine_auto_ledger_mode": "SHADOW-ish"}
        )
        assert not allowed
        assert reason == "engine_auto_ledger_mode_disabled"


class TestSqlitePersistBehavior:
    def test_disabled_mode_writes_zero_engine_auto_rows(self, monkeypatch):
        # Even with the legacy enabler env present — it can no longer enable.
        monkeypatch.setenv(ENV_LEGACY_AUTOLOG_KILL_SWITCH, "1")
        monkeypatch.delenv(ENV_ENGINE_SHADOW, raising=False)
        store = _tmp_store()
        store.persist(_shadow_trace("t-disabled"))
        assert _engine_auto_rows(store, "t-disabled") == []
        store.close()

    def test_shadow_requires_both_mode_and_env(self, monkeypatch):
        store = _tmp_store()
        # mode without env
        monkeypatch.delenv(ENV_ENGINE_SHADOW, raising=False)
        store.persist(_shadow_trace("t-mode-only", engine_auto_ledger_mode="shadow"))
        assert _engine_auto_rows(store, "t-mode-only") == []
        # env without mode
        monkeypatch.setenv(ENV_ENGINE_SHADOW, "1")
        store.persist(_shadow_trace("t-env-only"))
        assert _engine_auto_rows(store, "t-env-only") == []
        # both
        store.persist(_shadow_trace("t-both", engine_auto_ledger_mode="shadow"))
        rows = _engine_auto_rows(store, "t-both")
        assert len(rows) == 1
        store.close()

    def test_shadow_persist_is_idempotent(self, monkeypatch):
        monkeypatch.setenv(ENV_ENGINE_SHADOW, "1")
        store = _tmp_store()
        trace = _shadow_trace("t-idem", engine_auto_ledger_mode="shadow")
        store.persist(trace)
        store.persist(trace)  # re-persist: rowcount guard must not duplicate
        assert len(_engine_auto_rows(store, "t-idem")) == 1
        store.close()

    def test_scoped_suppression_blocks_shadow(self, monkeypatch):
        monkeypatch.setenv(ENV_ENGINE_SHADOW, "1")
        store = _tmp_store()
        with store.autolog_suppressed():
            store.persist(_shadow_trace("t-suppressed", engine_auto_ledger_mode="shadow"))
        assert _engine_auto_rows(store, "t-suppressed") == []
        store.close()

    def test_user_confirmed_recording_unaffected(self, monkeypatch):
        monkeypatch.delenv(ENV_ENGINE_SHADOW, raising=False)
        store = _tmp_store()
        store.persist(_shadow_trace("t-user"))
        bet = LedgerBet(
            ledger_id="led-user-1",
            trace_id="t-user",
            bet_date="2026-07-16",
            league="MLB",
            sport="baseball",
            matchup="Mets @ Mariners",
            market="player_prop:hits",
            bookmaker="betmgm",
            selection="Over 1.5",
            selection_descriptor="over_1.5",
            odds=-110.0,
            stake_amount=10.0,
            status=LedgerStatus.PENDING,
            provenance=BetProvenance.USER_CONFIRMED,
            decision_timestamp="2026-07-16T00:00:00Z",
        )
        store.record_ledger_bet(bet)
        rows = store.conn.execute(
            "SELECT provenance FROM bet_ledger WHERE trace_id = 't-user'"
        ).fetchall()
        assert [r["provenance"] for r in rows] == ["user_confirmed"]
        store.close()

    def test_calibration_eligibility_needs_no_ledger_row(self, monkeypatch):
        monkeypatch.delenv(ENV_ENGINE_SHADOW, raising=False)
        store = _tmp_store()
        store.persist(
            _shadow_trace(
                "t-elig",
                trace_quality={
                    "aggregate_quality": 0.9,
                    "calibration_eligible": True,
                    "identity_status": "complete",
                    "context_source": "provided",
                },
            )
        )
        assert _engine_auto_rows(store, "t-elig") == []
        eligible = store.query_traces(calibration_eligible_only=True, limit=10)
        assert [t["trace_id"] for t in eligible] == ["t-elig"]
        store.close()


def test_both_backends_route_through_shared_policy():
    """SQLite/Postgres parity by construction: both _maybe_autolog_ledger_bet
    implementations must call engine_auto_autolog_decision (no local env logic)."""
    for module in ("store.py", "repository.py"):
        tree = ast.parse((_SRC / module).read_text(encoding="utf-8"))
        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_maybe_autolog_ledger_bet":
                calls = {
                    n.func.id
                    for n in ast.walk(node)
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                }
                assert "engine_auto_autolog_decision" in calls, (
                    f"{module}: _maybe_autolog_ledger_bet must delegate to the "
                    "shared autolog policy"
                )
                # No direct env reads inside the decision site.
                env_reads = [
                    n
                    for n in ast.walk(node)
                    if isinstance(n, ast.Attribute) and n.attr in ("environ", "getenv")
                ]
                assert env_reads == [], f"{module}: env logic belongs in autolog_policy"
                found = True
        assert found, f"{module}: _maybe_autolog_ledger_bet not found"


def test_legacy_kill_switch_blocks_end_to_end(monkeypatch):
    monkeypatch.setenv(ENV_ENGINE_SHADOW, "1")
    monkeypatch.setenv(ENV_LEGACY_AUTOLOG_KILL_SWITCH, "0")
    store = _tmp_store()
    store.persist(_shadow_trace("t-killed", engine_auto_ledger_mode="shadow"))
    assert _engine_auto_rows(store, "t-killed") == []
    store.close()


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
