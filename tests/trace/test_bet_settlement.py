"""Unit tests for the pure bet-ledger settlement helpers."""

from __future__ import annotations

import pytest

from omega.trace.bet_settlement import (
    REASON_OK,
    REASON_SKIP_BAD_ODDS,
    REASON_SKIP_PASS,
    REASON_SKIP_SLATE,
    coerce_american_odds,
    compute_pnl,
    extract_recommended_bet,
    settle_game_bet,
    settle_prop_bet,
)
from omega.trace.ledger_bet import BetProvenance, LedgerStatus


class TestCoerceOdds:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (-110, -110.0),
            (150, 150.0),
            (100, 100.0),
            (-100, -100.0),
        ],
    )
    def test_valid_american(self, value, expected):
        assert coerce_american_odds(value) == expected

    @pytest.mark.parametrize("value", [None, 0, 1.91, 2.3, 50, -50, "abc"])
    def test_rejects_invalid_or_decimal(self, value):
        # Decimal-looking prices inside (-100, 100) are rejected, not mis-read.
        assert coerce_american_odds(value) is None


class TestSettleGameBet:
    def test_moneyline(self):
        assert settle_game_bet("moneyline", "home", None, 110, 104) == LedgerStatus.WON
        assert settle_game_bet("moneyline", "away", None, 110, 104) == LedgerStatus.LOST
        assert settle_game_bet("moneyline", "home", None, 100, 100) == LedgerStatus.PUSH

    def test_total(self):
        assert settle_game_bet("total", "over", 224.5, 120, 110) == LedgerStatus.WON
        assert settle_game_bet("total", "under", 224.5, 120, 110) == LedgerStatus.LOST
        assert settle_game_bet("total", "over", 230, 120, 110) == LedgerStatus.PUSH

    def test_spread(self):
        # home -3.5, win by 6 -> cover
        assert settle_game_bet("spread", "home", -3.5, 110, 104) == LedgerStatus.WON
        # home -7.5, win by only 6 -> no cover
        assert settle_game_bet("spread", "home", -7.5, 110, 104) == LedgerStatus.LOST
        # home -6, win by exactly 6 -> push
        assert settle_game_bet("spread", "home", -6, 110, 104) == LedgerStatus.PUSH
        # away +3.5, loses by 6 -> away margin -6 + 3.5 = -2.5 -> lost
        assert settle_game_bet("spread", "away", 3.5, 110, 104) == LedgerStatus.LOST


class TestSettlePropBet:
    def test_same_side(self):
        assert settle_prop_bet("over", "win", "over") == LedgerStatus.WON
        assert settle_prop_bet("over", "loss", "over") == LedgerStatus.LOST
        assert settle_prop_bet("over", "push", "over") == LedgerStatus.PUSH

    def test_opposite_side_inverts(self):
        # graded the under side; a recommended over wins when under lost
        assert settle_prop_bet("over", "loss", "under") == LedgerStatus.WON
        assert settle_prop_bet("over", "win", "under") == LedgerStatus.LOST

    def test_void_dnp_is_void_not_loss(self):
        # A DNP / no-action void must map to VOID regardless of side, never fall
        # through the win/loss branch (which would mis-grade it as a LOSS).
        assert settle_prop_bet("over", "void", "over") == LedgerStatus.VOID
        assert settle_prop_bet("over", "void", "under") == LedgerStatus.VOID
        assert settle_prop_bet("under", "void", "over") == LedgerStatus.VOID


class TestComputePnl:
    def test_won_minus_110(self):
        payout, net = compute_pnl(LedgerStatus.WON, -110, 25.0)
        assert payout == 47.73
        assert net == 22.73

    def test_won_plus_150(self):
        payout, net = compute_pnl(LedgerStatus.WON, 150, 25.0)
        assert payout == 62.5
        assert net == 37.5

    def test_lost(self):
        assert compute_pnl(LedgerStatus.LOST, -110, 25.0) == (0.0, -25.0)

    def test_push_returns_stake(self):
        assert compute_pnl(LedgerStatus.PUSH, -110, 25.0) == (25.0, 0.0)

    def test_pending_is_none(self):
        assert compute_pnl(LedgerStatus.PENDING, -110, 25.0) == (None, None)


def _game_trace() -> dict:
    return {
        "trace_id": "t-game",
        "run_id": "r",
        "timestamp": "2026-05-01T00:00:00Z",
        "kind": "game",
        "league": "NBA",
        "matchup": "A @ B",
        "input_snapshot": {"league": "NBA", "home_team": "B", "away_team": "A"},
        "result": {
            "edges": [
                {
                    "side": "home",
                    "team": "B",
                    "market": "spread",
                    "line": -3.5,
                    "ev_pct": 4.2,
                    "market_odds": -110,
                    "confidence_tier": "B",
                },
                {
                    "side": "away",
                    "team": "A",
                    "market": "moneyline",
                    "ev_pct": 1.0,
                    "market_odds": 120,
                    "confidence_tier": "Pass",
                },
            ],
            "best_bet": {"selection": "B -3.5", "odds": -110, "confidence_tier": "B"},
        },
    }


def _prop_trace() -> dict:
    return {
        "trace_id": "t-prop",
        "run_id": "r",
        "timestamp": "2026-05-01T00:00:00Z",
        "kind": "prop",
        "league": "NBA",
        "input_snapshot": {
            "league": "NBA",
            "player_name": "Jayson Tatum",
            "prop_type": "points",
            "line": 27.5,
            "odds_over": -115,
            "odds_under": -105,
        },
        "result": {"recommendation": "over", "confidence_tier": "A", "bet_side_odds": -115},
    }


class TestExtractRecommendedBet:
    def test_game_picks_best_actionable_edge(self):
        res = extract_recommended_bet(_game_trace(), provenance=BetProvenance.BACKFILL)
        assert res.reason == REASON_OK
        bet = res.bet
        assert bet.market == "spread"
        assert bet.selection_descriptor == "home_spread_-3.5"
        assert bet.line == -3.5
        assert bet.odds == -110
        assert bet.sport == "basketball"
        assert bet.provenance == BetProvenance.BACKFILL

    def test_prop_over(self):
        res = extract_recommended_bet(_prop_trace(), provenance=BetProvenance.ENGINE_AUTO)
        assert res.reason == REASON_OK
        bet = res.bet
        assert bet.market == "player_prop:points"
        assert "over" in bet.selection_descriptor
        assert bet.odds == -115
        assert bet.line == 27.5

    def test_pass_prop_skipped(self):
        trace = _prop_trace()
        trace["result"]["recommendation"] = "pass"
        res = extract_recommended_bet(trace, provenance=BetProvenance.BACKFILL)
        assert res.bet is None
        assert res.reason == REASON_SKIP_PASS

    def test_bad_odds_skipped(self):
        trace = _game_trace()
        trace["result"]["edges"][0]["market_odds"] = 1.91  # decimal, not American
        trace["result"].pop("best_bet")
        res = extract_recommended_bet(trace, provenance=BetProvenance.BACKFILL)
        assert res.bet is None
        assert res.reason == REASON_SKIP_BAD_ODDS

    def test_slate_skipped(self):
        res = extract_recommended_bet(
            {"trace_id": "t", "kind": "slate", "result": {"analyses": []}},
            provenance=BetProvenance.BACKFILL,
        )
        assert res.reason == REASON_SKIP_SLATE


class TestBookProvenance:
    def test_game_book_from_matching_market_quote(self):
        trace = _game_trace()
        trace["input_snapshot"]["odds"] = {
            "markets": [
                {
                    "market_type": "spread",
                    "selection": "B",
                    "line": -3.5,
                    "price": -110,
                    "bookmaker": "draftkings",
                },
                {
                    "market_type": "moneyline",
                    "selection": "A",
                    "price": 120,
                    "bookmaker": "fanduel",
                },
            ]
        }
        bet = extract_recommended_bet(trace, provenance=BetProvenance.BACKFILL).bet
        assert bet.bookmaker == "draftkings"  # matched the chosen spread selection

    def test_game_book_uniform_fallback_when_no_exact_match(self):
        trace = _game_trace()
        # No spread quote for the chosen selection, but every quote is one book.
        trace["input_snapshot"]["odds"] = {
            "markets": [
                {
                    "market_type": "moneyline",
                    "selection": "A",
                    "price": 120,
                    "bookmaker": "caesars",
                },
                {
                    "market_type": "moneyline",
                    "selection": "B",
                    "price": -140,
                    "bookmaker": "caesars",
                },
            ]
        }
        bet = extract_recommended_bet(trace, provenance=BetProvenance.BACKFILL).bet
        assert bet.bookmaker == "caesars"

    def test_game_book_consensus_when_books_mixed_and_no_match(self):
        trace = _game_trace()
        trace["input_snapshot"]["odds"] = {
            "markets": [
                {
                    "market_type": "moneyline",
                    "selection": "A",
                    "price": 120,
                    "bookmaker": "caesars",
                },
                {
                    "market_type": "moneyline",
                    "selection": "B",
                    "price": -140,
                    "bookmaker": "fanduel",
                },
            ]
        }
        bet = extract_recommended_bet(trace, provenance=BetProvenance.BACKFILL).bet
        assert bet.bookmaker == "consensus"

    def test_game_book_consensus_when_no_markets(self):
        # The default _game_trace carries no odds.markets — unknown book.
        bet = extract_recommended_bet(_game_trace(), provenance=BetProvenance.BACKFILL).bet
        assert bet.bookmaker == "consensus"

    def test_game_book_best_bet_fallback_does_not_use_moneyline_quote_for_spread_label(self):
        # No actionable structured edge, but an actionable best_bet whose label
        # ("B -3.5") prefixes a moneyline quote selection ("B"). That quote is
        # not the spread bet being logged, so mixed-book snapshots must fall
        # back to consensus rather than borrowing the moneyline book.
        trace = _game_trace()
        trace["result"]["edges"] = [
            {
                "side": "home",
                "team": "B",
                "market": "spread",
                "line": -3.5,
                "ev_pct": 1.0,
                "market_odds": -110,
                "confidence_tier": "Pass",
            },
        ]
        trace["result"]["best_bet"] = {
            "selection": "B -3.5",
            "odds": -110,
            "confidence_tier": "B",
        }
        trace["input_snapshot"]["odds"] = {
            "markets": [
                {
                    "market_type": "moneyline",
                    "selection": "B",
                    "price": -140,
                    "bookmaker": "betmgm",
                },
                {
                    "market_type": "spread",
                    "selection": "B",
                    "line": -4.5,
                    "price": -110,
                    "bookmaker": "draftkings",
                },
            ]
        }
        bet = extract_recommended_bet(trace, provenance=BetProvenance.BACKFILL).bet
        assert bet.bookmaker == "consensus"

    def test_prop_book_from_request_field(self):
        trace = _prop_trace()
        trace["input_snapshot"]["bookmaker"] = "betmgm"
        bet = extract_recommended_bet(trace, provenance=BetProvenance.ENGINE_AUTO).bet
        assert bet.bookmaker == "betmgm"

    def test_prop_book_consensus_when_absent(self):
        bet = extract_recommended_bet(_prop_trace(), provenance=BetProvenance.ENGINE_AUTO).bet
        assert bet.bookmaker == "consensus"
