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
    engine_auto_stake_constraint,
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

    @pytest.mark.parametrize("value", [None, 0, -50, 1.0, 0.5, "abc"])
    def test_rejects_invalid(self, value):
        # Junk, zero, negative sub-American, and decimal <= 1.0 are rejected.
        assert coerce_american_odds(value) is None

    @pytest.mark.parametrize(
        "value,expected",
        [
            (1.91, -110),
            (2.3, 130),
            (50, 4900),
        ],
    )
    def test_converts_decimal_to_american(self, value, expected):
        # Decimal prices inside (-100, 100) are converted, not dropped.
        assert coerce_american_odds(value) == expected


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

    def test_decimal_odds_converted(self):
        trace = _game_trace()
        trace["result"]["edges"][0]["market_odds"] = 1.91  # decimal, not American
        trace["result"].pop("best_bet")
        res = extract_recommended_bet(trace, provenance=BetProvenance.BACKFILL)
        assert res.bet is not None
        assert res.bet.odds == -110
        assert res.reason == REASON_OK

    def test_unusable_odds_skipped(self):
        trace = _game_trace()
        trace["result"]["edges"][0]["market_odds"] = 0  # neither American nor decimal
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


def _audit(
    maturity: str | None = "production",
    sample_size: int | None = 130,
    ece: float | None = 0.049,
    profile_id: str | None = "iso_x_v1",
) -> dict:
    return {
        "profile_id": profile_id,
        "profile_maturity": maturity,
        "sample_size": sample_size,
        "ece": ece,
    }


class TestEngineAutoStakeCap:
    """RESEARCH_PLUS / RESEARCH_CANDIDATE stake ceilings on the autolog seam.

    1 unit == 1% of bankroll, so on the $1000 default bankroll the flat $25
    default stake is 2.5u and the ceilings are provisional $5 (0.5u),
    probation $10 (1u), research-candidate $10 (1u).
    """

    def test_constraint_none_for_actionable_profile(self):
        assert engine_auto_stake_constraint(_audit()) is None

    def test_constraint_provisional_is_half_unit(self):
        units, label = engine_auto_stake_constraint(_audit(maturity="provisional", ece=0.0746))
        assert units == 0.5
        assert label == "research_plus:provisional"

    def test_constraint_probation_is_one_unit(self):
        units, label = engine_auto_stake_constraint(_audit(maturity="probation"))
        assert units == 1.0
        assert label == "research_plus:probation"

    def test_constraint_below_floor_production_is_research_plus(self):
        # A production profile that misses the ECE floor is RESEARCH_PLUS.
        units, label = engine_auto_stake_constraint(_audit(ece=0.075))
        assert units == 0.5
        assert label == "research_plus:production"

    def test_constraint_missing_audit_is_research_candidate(self):
        units, label = engine_auto_stake_constraint(None)
        assert units == 1.0
        assert label == "research_candidate"

    def test_prop_research_plus_capped_before_persistable_row(self):
        trace = _prop_trace()
        trace["result"]["over_calibration_audit"] = _audit(maturity="provisional", ece=0.0746)
        res = extract_recommended_bet(trace, provenance=BetProvenance.ENGINE_AUTO)
        assert res.reason == REASON_OK
        assert res.bet.stake_amount == 5.0
        assert res.bet.sizing_reasons == ["research_stake_cap:research_plus:provisional:0.5u"]

    def test_prop_actionable_not_capped(self):
        trace = _prop_trace()
        trace["result"]["over_calibration_audit"] = _audit()
        res = extract_recommended_bet(trace, provenance=BetProvenance.ENGINE_AUTO)
        assert res.bet.stake_amount == 25.0
        assert res.bet.sizing_reasons is None

    def test_prop_without_audit_gets_research_candidate_cap(self):
        res = extract_recommended_bet(_prop_trace(), provenance=BetProvenance.ENGINE_AUTO)
        assert res.bet.stake_amount == 10.0
        assert res.bet.sizing_reasons == ["research_stake_cap:research_candidate:1u"]

    def test_game_edge_audit_drives_cap(self):
        trace = _game_trace()
        trace["result"]["edges"][0]["calibration_audit"] = _audit(
            maturity="probation", ece=0.06, profile_id="shrink_x_v2"
        )
        res = extract_recommended_bet(trace, provenance=BetProvenance.ENGINE_AUTO)
        assert res.bet.stake_amount == 10.0
        assert res.bet.sizing_reasons == ["research_stake_cap:research_plus:probation:1u"]

    def test_backfill_provenance_never_capped(self):
        trace = _prop_trace()
        trace["result"]["over_calibration_audit"] = _audit(maturity="provisional", ece=0.0746)
        res = extract_recommended_bet(trace, provenance=BetProvenance.BACKFILL)
        assert res.bet.stake_amount == 25.0
        assert res.bet.sizing_reasons is None

    def test_stake_already_below_ceiling_unchanged(self):
        trace = _prop_trace()
        trace["result"]["over_calibration_audit"] = _audit(maturity="provisional", ece=0.0746)
        res = extract_recommended_bet(
            trace, provenance=BetProvenance.ENGINE_AUTO, stake_amount=3.0
        )
        assert res.bet.stake_amount == 3.0
        assert res.bet.sizing_reasons is None

    def test_cap_scales_with_bankroll(self):
        trace = _prop_trace()
        trace["result"]["over_calibration_audit"] = _audit(maturity="provisional", ece=0.0746)
        res = extract_recommended_bet(
            trace, provenance=BetProvenance.ENGINE_AUTO, bankroll=2000.0
        )
        assert res.bet.stake_amount == 10.0  # 0.5u of $2000


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
