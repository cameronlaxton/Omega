"""Tests for parlay math utilities."""

import pytest

from omega.core.betting.parlay import (
    ParlayLeg,
    ParlaySlip,
    build_parlay,
    check_correlation,
    compute_parlay_odds,
    compute_parlay_probability,
)


# ---------------------------------------------------------------------------
# Fixtures — modeled after user's real bets
# ---------------------------------------------------------------------------

def _leg(selection, decimal_odds, win_prob, player="", stat_key="", team=""):
    return ParlayLeg(
        selection=selection,
        decimal_odds=decimal_odds,
        win_probability=win_prob,
        player=player,
        stat_key=stat_key,
        team=team,
    )


class TestComputeParlayOdds:
    def test_single_leg(self):
        legs = [_leg("A", 1.30, 0.80)]
        assert compute_parlay_odds(legs) == pytest.approx(1.30)

    def test_two_legs(self):
        legs = [_leg("A", 1.30, 0.80), _leg("B", 1.50, 0.70)]
        assert compute_parlay_odds(legs) == pytest.approx(1.95)

    def test_three_legs_real_bet(self):
        """KD 20+ pts (1.40x) + Sengun 15+ pts (1.45x) + KD 5+ ast (1.40x) ≈ 2.83x.
        User got 2.55x from SGP pricing (book applies correlation discount)."""
        legs = [
            _leg("KD 20+ pts", 1.40, 0.80),
            _leg("Sengun 15+ pts", 1.45, 0.70),
            _leg("KD 5+ ast", 1.40, 0.70),
        ]
        assert compute_parlay_odds(legs) == pytest.approx(2.842, rel=0.01)

    def test_four_legs(self):
        legs = [
            _leg("A", 1.25, 0.85),
            _leg("B", 1.30, 0.80),
            _leg("C", 1.35, 0.75),
            _leg("D", 1.25, 0.85),
        ]
        expected = 1.25 * 1.30 * 1.35 * 1.25
        assert compute_parlay_odds(legs) == pytest.approx(expected)

    def test_empty_legs(self):
        assert compute_parlay_odds([]) == pytest.approx(1.0)


class TestComputeParlayProbability:
    def test_independent_legs(self):
        legs = [_leg("A", 1.30, 0.80), _leg("B", 1.50, 0.70)]
        assert compute_parlay_probability(legs) == pytest.approx(0.56)

    def test_three_legs(self):
        legs = [_leg("A", 1.3, 0.80), _leg("B", 1.4, 0.75), _leg("C", 1.3, 0.85)]
        assert compute_parlay_probability(legs) == pytest.approx(0.51, rel=0.01)

    def test_independence_discount(self):
        legs = [_leg("A", 1.30, 0.80), _leg("B", 1.50, 0.70)]
        discounted = compute_parlay_probability(legs, independence_discount=0.95)
        assert discounted == pytest.approx(0.56 * 0.95, rel=0.01)

    def test_empty_legs(self):
        assert compute_parlay_probability([]) == pytest.approx(1.0)


class TestCheckCorrelation:
    def test_no_correlation_different_players(self):
        legs = [
            _leg("A pts", 1.3, 0.8, player="SGA", stat_key="pts", team="OKC"),
            _leg("B reb", 1.4, 0.7, player="Lopez", stat_key="reb", team="LAC"),
        ]
        assert check_correlation(legs) == []

    def test_same_player_pts_pra(self):
        legs = [
            _leg("KD pts", 1.4, 0.8, player="Kevin Durant", stat_key="pts", team="GSW"),
            _leg("KD pra", 1.3, 0.7, player="Kevin Durant", stat_key="pra", team="GSW"),
        ]
        warnings = check_correlation(legs)
        assert len(warnings) == 1
        assert "Correlated" in warnings[0]

    def test_same_player_different_uncorrelated_stats(self):
        """pts + ast for same player — not in the correlated pairs list, so no warning."""
        legs = [
            _leg("KD pts", 1.4, 0.8, player="Kevin Durant", stat_key="pts", team="GSW"),
            _leg("KD ast", 1.4, 0.7, player="Kevin Durant", stat_key="ast", team="GSW"),
        ]
        warnings = check_correlation(legs)
        assert len(warnings) == 0

    def test_same_player_same_stat_redundant(self):
        legs = [
            _leg("KD 20+ pts", 1.4, 0.8, player="Kevin Durant", stat_key="pts", team="GSW"),
            _leg("KD 25+ pts", 1.8, 0.5, player="Kevin Durant", stat_key="pts", team="GSW"),
        ]
        warnings = check_correlation(legs)
        assert len(warnings) == 1
        assert "Redundant" in warnings[0]

    def test_same_team_same_stat(self):
        legs = [
            _leg("SGA pts", 1.4, 0.8, player="SGA", stat_key="pts", team="OKC"),
            _leg("Chet pts", 1.3, 0.75, player="Chet Holmgren", stat_key="pts", team="OKC"),
        ]
        warnings = check_correlation(legs)
        assert len(warnings) == 1
        assert "Team correlation" in warnings[0]

    def test_same_team_different_stats_no_warning(self):
        legs = [
            _leg("SGA pts", 1.4, 0.8, player="SGA", stat_key="pts", team="OKC"),
            _leg("Chet reb", 1.3, 0.75, player="Chet Holmgren", stat_key="reb", team="OKC"),
        ]
        assert check_correlation(legs) == []


class TestBuildParlay:
    def test_full_slip_construction(self):
        """Matches user's 4/7 bet: Davion Mitchell 5+ ast + Wiggins 2+ 3pt @ 2.00x."""
        legs = [
            _leg("Mitchell 5+ ast", 1.40, 0.75, player="Davion Mitchell", stat_key="ast", team="TOR"),
            _leg("Wiggins 2+ 3pt", 1.45, 0.72, player="Andrew Wiggins", stat_key="3pm", team="TOR"),
        ]
        slip = build_parlay(legs)

        assert slip.combined_decimal_odds == pytest.approx(2.03, rel=0.01)
        assert slip.combined_win_probability == pytest.approx(0.54, rel=0.01)
        assert slip.implied_probability == pytest.approx(1.0 / 2.03, rel=0.01)
        assert slip.ev_pct > 0  # positive EV expected
        assert slip.correlation_warnings == []

    def test_slip_with_correlation(self):
        legs = [
            _leg("KD 20+ pts", 1.40, 0.80, player="Kevin Durant", stat_key="pts", team="GSW"),
            _leg("KD 25+ pra", 1.35, 0.75, player="Kevin Durant", stat_key="pra", team="GSW"),
        ]
        slip = build_parlay(legs)
        assert len(slip.correlation_warnings) == 1

    def test_from_american_odds(self):
        leg = ParlayLeg.from_american(
            selection="SGA Over 5 Assists",
            american_odds=-200,
            win_probability=0.80,
            player="SGA",
            stat_key="ast",
            team="OKC",
        )
        assert leg.decimal_odds == pytest.approx(1.50)
        assert leg.win_probability == 0.80
