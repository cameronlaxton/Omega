"""Tests for NFL Wong-teaser leg evaluation (Phase 7 M4).

Hand-computable margin/total pmf fixtures verify the teased-line cover buckets,
push semantics on integer crossings, and the EV bridge into the binary edge
framework. The leg math reuses the sport-neutral threshold evaluator, so these
mirror tests/core/test_soccer_derivatives.py.
"""

from __future__ import annotations

import pytest

from omega.core.edge.nfl_teasers import (
    WONG_TEASER_POINTS,
    evaluate_teaser_spread_leg,
    evaluate_teaser_total_leg,
    tease_spread,
    tease_total,
)

# home-minus-away margins; n=100. Clusters on the key numbers 3 and 7.
_MARGINS = {"-3": 20, "0": 20, "3": 30, "7": 30}
_TOTALS = {"38": 30, "45": 40, "52": 30}  # n=100


def test_tease_spread_moves_line_in_bettor_favor():
    assert tease_spread(-8.5) == pytest.approx(-2.5)  # favorite teased down
    assert tease_spread(1.5) == pytest.approx(7.5)  # underdog teased up
    assert tease_spread(-1.5, points=10.0) == pytest.approx(8.5)


def test_tease_total_directional():
    assert tease_total(45.5, "over") == pytest.approx(45.5 - WONG_TEASER_POINTS)
    assert tease_total(45.5, "under") == pytest.approx(45.5 + WONG_TEASER_POINTS)
    with pytest.raises(ValueError):
        tease_total(45.5, "middle")


def test_home_favorite_leg_half_line_no_push():
    # Home teased to -2.5: covers when margin > 2.5 → margins 3 and 7.
    ev = evaluate_teaser_spread_leg(_MARGINS, -2.5, "home")
    assert ev.p_win == pytest.approx(0.6)
    assert ev.p_loss == pytest.approx(0.4)
    assert ev.p_push == ev.p_half_win == ev.p_half_loss == 0.0


def test_home_leg_integer_line_pushes_on_key_number():
    # Home teased to -3.0: win margin>3 (only 7), push margin==3, loss otherwise.
    ev = evaluate_teaser_spread_leg(_MARGINS, -3.0, "home")
    assert ev.p_win == pytest.approx(0.3)
    assert ev.p_push == pytest.approx(0.3)
    assert ev.p_loss == pytest.approx(0.4)


def test_away_leg_mirrors_margin_under():
    # Away teased to +2.5: covers when -margin + 2.5 > 0 → margin < 2.5 → -3, 0.
    ev = evaluate_teaser_spread_leg(_MARGINS, 2.5, "away")
    assert ev.p_win == pytest.approx(0.4)
    assert ev.p_loss == pytest.approx(0.6)


def test_total_legs_over_under():
    over = evaluate_teaser_total_leg(_TOTALS, 39.5, "over")  # win when total > 39.5
    assert over.p_win == pytest.approx(0.7)  # 45 and 52
    assert over.p_loss == pytest.approx(0.3)

    under = evaluate_teaser_total_leg(_TOTALS, 51.5, "under")  # win when total < 51.5
    assert under.p_win == pytest.approx(0.7)  # 38 and 45
    assert under.p_loss == pytest.approx(0.3)


def test_ev_bridge_matches_hand_value():
    ev = evaluate_teaser_spread_leg(_MARGINS, -2.5, "home")  # p_win 0.6 / p_loss 0.4
    assert ev.ev_per_unit(+100) == pytest.approx(0.6 - 0.4)
    assert ev.equivalent_win_prob(+100) == pytest.approx(0.6)


def test_invalid_side_fails_loud():
    with pytest.raises(ValueError):
        evaluate_teaser_spread_leg(_MARGINS, -2.5, "draw")
    with pytest.raises(ValueError):
        evaluate_teaser_total_leg(_TOTALS, 39.5, "push")
