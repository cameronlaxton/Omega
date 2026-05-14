"""Tests for anchor bet detection engine."""

import pytest

from omega.strategy.anchor.detector import (
    AnchorLeg,
    AnchorThreshold,
    compute_hit_rate,
    detect_anchors,
    match_anchor_to_odds,
    scan_player,
)


# ---------------------------------------------------------------------------
# Test data — inspired by real NBA player performances
# ---------------------------------------------------------------------------

# Donovan Mitchell's last 10 games (points) — consistent 20+ scorer
MITCHELL_PTS = [28, 25, 32, 22, 18, 26, 30, 24, 21, 27]

# Jarrett Allen's last 10 games (rebounds) — consistent 7+ rebounder
ALLEN_REB = [9, 11, 8, 7, 10, 6, 12, 8, 9, 7]

# SGA's last 10 games (assists) — consistent 5+ facilitator
SGA_AST = [7, 5, 8, 6, 4, 7, 6, 5, 8, 5]

# Sparse data — only 3 games
SHORT_SAMPLE = [25, 20, 30]


class TestComputeHitRate:
    def test_all_hit(self):
        checked, hit, rate = compute_hit_rate([25, 30, 20], 20)
        assert checked == 3
        assert hit == 3
        assert rate == pytest.approx(1.0)

    def test_none_hit(self):
        checked, hit, rate = compute_hit_rate([5, 8, 3], 20)
        assert checked == 3
        assert hit == 0
        assert rate == pytest.approx(0.0)

    def test_partial_hit(self):
        checked, hit, rate = compute_hit_rate(MITCHELL_PTS, 25)
        # 28, 25, 32, 26, 30, 27 >= 25 → 6/10
        assert checked == 10
        assert hit == 6
        assert rate == pytest.approx(0.6)

    def test_mitchell_20_plus(self):
        checked, hit, rate = compute_hit_rate(MITCHELL_PTS, 20)
        # All 10 are >= 20 except 18 → 9/10
        assert checked == 10
        assert hit == 9
        assert rate == pytest.approx(0.9)

    def test_allen_7_plus_reb(self):
        checked, hit, rate = compute_hit_rate(ALLEN_REB, 7)
        # 9,11,8,7,10,12,8,9,7 >= 7 → 9/10 (only 6 misses)
        assert checked == 10
        assert hit == 9
        assert rate == pytest.approx(0.9)

    def test_empty_values(self):
        checked, hit, rate = compute_hit_rate([], 10)
        assert checked == 0
        assert hit == 0
        assert rate == pytest.approx(0.0)

    def test_exact_threshold(self):
        """Values exactly at threshold should count as hits."""
        checked, hit, rate = compute_hit_rate([20, 20, 20], 20)
        assert hit == 3


class TestDetectAnchors:
    def test_mitchell_anchors(self):
        anchors = detect_anchors("Donovan Mitchell", "pts", MITCHELL_PTS)
        # Should detect 10+ (10/10=100%) and 20+ (9/10=90%) as anchors
        # 15+ should also qualify: all >= 15 except 18 is >= 15, so 10/10=100%? No wait:
        # 28,25,32,22,18,26,30,24,21,27 — all >= 15? Yes, all 10.
        # 25+ → 6/10 = 60% — does NOT qualify
        thresholds_found = {a.threshold for a in anchors}
        assert 10 in thresholds_found
        assert 15 in thresholds_found
        assert 20 in thresholds_found
        assert 25 not in thresholds_found  # 60% < 70%

    def test_mitchell_anchor_hit_rates(self):
        anchors = detect_anchors("Donovan Mitchell", "pts", MITCHELL_PTS)
        anchor_map = {a.threshold: a for a in anchors}
        assert anchor_map[20].hit_rate == pytest.approx(0.9)
        assert anchor_map[10].hit_rate == pytest.approx(1.0)

    def test_sorted_by_threshold_descending(self):
        anchors = detect_anchors("Donovan Mitchell", "pts", MITCHELL_PTS)
        thresholds = [a.threshold for a in anchors]
        assert thresholds == sorted(thresholds, reverse=True)

    def test_too_few_games(self):
        anchors = detect_anchors("Player", "pts", SHORT_SAMPLE, min_games=5)
        assert anchors == []

    def test_custom_min_hit_rate(self):
        # At 60% threshold, 25+ should now qualify (6/10 = 60%)
        anchors = detect_anchors(
            "Donovan Mitchell", "pts", MITCHELL_PTS, min_hit_rate=0.60
        )
        thresholds_found = {a.threshold for a in anchors}
        assert 25 in thresholds_found

    def test_custom_thresholds(self):
        anchors = detect_anchors(
            "Test", "pts", [15, 16, 14, 17, 15],
            thresholds=[14, 15, 16],
        )
        anchor_map = {a.threshold: a for a in anchors}
        assert 14 in anchor_map  # 5/5 = 100%
        assert 15 in anchor_map  # 4/5 = 80%
        assert 16 not in anchor_map  # 2/5 = 40%

    def test_sga_assists(self):
        anchors = detect_anchors("SGA", "ast", SGA_AST)
        anchor_map = {a.threshold: a for a in anchors}
        # 3+: all 10 hit → 100%
        assert 3 in anchor_map
        assert anchor_map[3].hit_rate == pytest.approx(1.0)
        # 5+: 7,5,8,6,7,6,5,8,5 → 9/10 (only 4 misses)
        assert 5 in anchor_map
        assert anchor_map[5].hit_rate == pytest.approx(0.9)


class TestMatchAnchorToOdds:
    def test_positive_edge(self):
        anchor = AnchorThreshold(
            player_name="Mitchell",
            stat_key="pts",
            threshold=20,
            games_checked=10,
            games_hit=9,
            hit_rate=0.9,
        )
        leg = match_anchor_to_odds(anchor, team="CLE", odds_over=-300)
        # implied_prob at -300 = 300/400 = 0.75
        assert leg.implied_prob == pytest.approx(0.75)
        assert leg.empirical_prob == pytest.approx(0.9)
        assert leg.edge_pct == pytest.approx(15.0)  # (0.9 - 0.75) * 100

    def test_negative_edge(self):
        anchor = AnchorThreshold(
            player_name="Player",
            stat_key="pts",
            threshold=20,
            games_checked=10,
            games_hit=7,
            hit_rate=0.7,
        )
        leg = match_anchor_to_odds(anchor, team="X", odds_over=-250)
        # implied at -250 = 250/350 ≈ 0.714
        assert leg.edge_pct < 0  # 0.7 < 0.714


class TestScanPlayer:
    def test_scan_with_odds(self):
        game_logs = {
            "pts": MITCHELL_PTS,
            "ast": [5, 6, 4, 5, 7, 3, 5, 6, 4, 5],
        }
        prop_odds = {
            "pts": {20: -300, 25: +110},
            "ast": {3: -400, 5: -150},
        }
        legs = scan_player(
            "Donovan Mitchell", "CLE", game_logs, prop_odds
        )
        # Should find pts anchors at 10, 15, 20 (but only 20 has odds)
        # And ast anchors at 3 (if hit rate >= 70%)
        assert len(legs) > 0
        players = {leg.player_name for leg in legs}
        assert players == {"Donovan Mitchell"}

    def test_scan_without_odds(self):
        game_logs = {"pts": MITCHELL_PTS}
        legs = scan_player("Mitchell", "CLE", game_logs)
        # Should still return anchors, just without odds/edge
        assert len(legs) > 0
        assert all(leg.odds_over is None for leg in legs)

    def test_scan_insufficient_games(self):
        game_logs = {"pts": [25, 30]}
        legs = scan_player("Player", "X", game_logs)
        assert legs == []
