from __future__ import annotations

from omega.trace.portfolio import summarize_ledger


def test_portfolio_artifact_fields_for_pending_and_graded_rows():
    summary = summarize_ledger(
        [
            {
                "ledger_id": "open-1",
                "league": "NBA",
                "market": "moneyline",
                "status": "pending",
                "stake_amount": 25,
                "odds": -110,
                "decision_timestamp": "2026-06-01T12:00:00Z",
            },
            {
                "ledger_id": "won-1",
                "league": "NBA",
                "market": "spread",
                "status": "won",
                "stake_amount": 50,
                "odds": 120,
                "net_pnl": 60,
            },
        ]
    )

    assert summary["realized_pnl"] == summary["net_pnl"] == 60.0
    assert summary["unrealized_pnl"] == 0.0
    assert summary["pending_exposure"] == summary["pending_stake"] == 25.0
    assert summary["open_positions_count"] == summary["pending_count"] == 1

    active = summary["active_ledgers"]
    assert len(active) == 1
    assert active[0] == {
        "ledger_id": "open-1",
        "league": "NBA",
        "market_type": "moneyline",
        "status": "pending",
        "stake": 25.0,
        "potential_payout": 47.73,
        "pnl": 0.0,
        "last_updated": "2026-06-01T12:00:00Z",
    }
    assert isinstance(active[0]["stake"], float)
    assert isinstance(active[0]["potential_payout"], float)
    assert isinstance(active[0]["pnl"], float)


def test_portfolio_artifact_empty_state_is_stable():
    summary = summarize_ledger([])

    assert summary["realized_pnl"] == 0.0
    assert summary["unrealized_pnl"] == 0.0
    assert summary["pending_exposure"] == 0.0
    assert summary["open_positions_count"] == 0
    assert summary["active_ledgers"] == []
