"""End-to-end + decouple-regression tests for backfill_bets."""

from __future__ import annotations

import tempfile

import pytest

from omega.ops.backfill_bets import run_backfill
from omega.trace.store import TraceStore


@pytest.fixture
def store(monkeypatch):
    # Disable the persist() autolog so the ledger starts empty and we isolate
    # backfill behavior.
    monkeypatch.setenv("OMEGA_BET_LEDGER_AUTOLOG", "0")
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    s = TraceStore(db_path=tmp.name)
    yield s
    s.close()


def _persist_game(store, trace_id, *, with_outcome=True):
    store.persist({
        "trace_id": trace_id,
        "run_id": "r",
        "timestamp": "2026-05-01T00:00:00Z",
        "prompt": "p",
        "league": "NBA",
        "matchup": "A @ B",
        "execution_mode": "native_sim",
        "kind": "game",
        "result": {
            "status": "success",
            "edges": [
                {"side": "home", "team": "B", "market": "moneyline",
                 "ev_pct": 5.0, "market_odds": -120, "confidence_tier": "A"},
            ],
            "best_bet": {"selection": "B ML", "odds": -120, "confidence_tier": "A"},
        },
    })
    if with_outcome:
        store.attach_outcome(trace_id=trace_id, home_score=110, away_score=104)


def _persist_prop(store, trace_id, *, with_outcome=True):
    store.persist({
        "trace_id": trace_id,
        "run_id": "r",
        "timestamp": "2026-05-01T00:00:00Z",
        "prompt": "p",
        "league": "NBA",
        "matchup": "Tatum points",
        "execution_mode": "native_sim",
        "kind": "prop",
        "input_snapshot": {
            "league": "NBA", "player_name": "Jayson Tatum",
            "prop_type": "points", "line": 27.5, "odds_over": -115,
        },
        "result": {"status": "success", "recommendation": "over", "bet_side_odds": -115},
    })
    if with_outcome:
        store.attach_prop_outcome(
            trace_id=trace_id, player_name="Jayson Tatum", stat_type="points",
            stat_value=31, line=27.5, side="over",
        )


def _persist_pass(store, trace_id):
    store.persist({
        "trace_id": trace_id,
        "run_id": "r",
        "timestamp": "2026-05-01T00:00:00Z",
        "prompt": "p",
        "league": "NBA",
        "matchup": "C @ D",
        "execution_mode": "native_sim",
        "kind": "prop",
        "result": {"status": "success", "recommendation": "pass"},
    })


class TestBackfill:
    def test_dry_run_writes_nothing(self, store):
        _persist_game(store, "g1")
        summary = run_backfill(store, apply=False)
        assert summary.eligible == 1
        assert store.get_ledger_bets("g1") == []  # dry-run: no write

    def test_apply_logs_and_grades(self, store):
        _persist_game(store, "g1")           # won moneyline
        _persist_prop(store, "p1")           # won over
        _persist_game(store, "g2", with_outcome=False)  # pending
        _persist_pass(store, "x1")           # skipped

        summary = run_backfill(store, apply=True)

        assert summary.eligible == 3
        assert summary.graded["won"] == 2
        assert summary.pending == 1
        assert summary.skipped.get("skip_pass", 0) == 1

        g1 = store.get_ledger_bets("g1")[0]
        assert g1["status"] == "won"
        assert g1["provenance"] == "backfill"
        # -120 win on $25: decimal 1.8333 -> payout 45.83, net 20.83
        assert g1["net_pnl"] == pytest.approx(20.83, abs=0.01)

        p1 = store.get_ledger_bets("p1")[0]
        assert p1["status"] == "won"
        assert p1["market"] == "player_prop:points"

    def test_regrades_previously_pending_after_outcome_attached(self, store):
        # Log first with NO outcome -> pending.
        _persist_game(store, "g1", with_outcome=False)
        run_backfill(store, apply=True)
        assert store.get_ledger_bets("g1")[0]["status"] == "pending"

        # Outcome arrives later; re-run picks up the already-logged bet.
        store.attach_outcome(trace_id="g1", home_score=110, away_score=104)
        summary = run_backfill(store, apply=True)
        assert summary.eligible == 0          # nothing new to insert
        assert summary.regraded["won"] == 1   # the prior pending row got settled
        row = store.get_ledger_bets("g1")[0]
        assert row["status"] == "won"
        assert row["net_pnl"] is not None
        assert row["graded_at"] is not None

    def test_idempotent_rerun(self, store):
        _persist_game(store, "g1")
        run_backfill(store, apply=True)
        again = run_backfill(store, apply=True)
        assert again.eligible == 0
        assert again.already_present == 1
        assert len(store.get_ledger_bets("g1")) == 1


class TestDecoupleRegression:
    def test_backfill_writes_no_user_confirmed_rows(self, store):
        _persist_game(store, "g1")
        _persist_prop(store, "p1")
        run_backfill(store, apply=True)
        # Backfill must only create provenance='backfill' rows — it must never
        # mint user_confirmed wagers (the CLV / closing-line population).
        uc = store.conn.execute(
            "SELECT COUNT(*) FROM bet_ledger WHERE provenance = 'user_confirmed'"
        ).fetchone()[0]
        assert uc == 0
        # And the legacy bet_records table is gone (consolidated at V14).
        tbl = store.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='bet_records'"
        ).fetchone()
        assert tbl is None
