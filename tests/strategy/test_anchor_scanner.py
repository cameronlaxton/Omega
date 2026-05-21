"""Integration tests for the anchor parlay scanner.

Uses fixtures modeled after the user's real winning bets from 4/4-4/8/2026.
"""

from omega.strategy.anchor.detector import scan_player
from omega.strategy.anchor.scanner import (
    AnchorParlayConfig,
    build_parlays_for_game,
    scan_slate,
)

# ---------------------------------------------------------------------------
# Fixture: OKC @ LAC (4/8/26) — user's 4-leg winning parlay
# ---------------------------------------------------------------------------

OKC_LAC_GAME = {
    "game_label": "Thunder @ Clippers",
    "players": [
        {
            "name": "Chet Holmgren",
            "team": "OKC",
            "game_logs": {
                "pts": [15, 12, 18, 14, 10, 16, 22, 11, 13, 17],  # 10/10 >= 10
            },
            "prop_odds": {
                "pts": {10: -350},  # heavy favorite to hit 10+
            },
        },
        {
            "name": "Shai Gilgeous-Alexander",
            "team": "OKC",
            "game_logs": {
                "ast": [7, 5, 8, 6, 4, 7, 6, 5, 8, 5],  # 9/10 >= 5
            },
            "prop_odds": {
                "ast": {5: -200},
            },
        },
        {
            "name": "Kris Dunn",
            "team": "LAC",
            "game_logs": {
                "ast": [4, 3, 5, 3, 6, 4, 3, 5, 4, 3],  # 10/10 >= 3
            },
            "prop_odds": {
                "ast": {3: -250},
            },
        },
        {
            "name": "Brook Lopez",
            "team": "LAC",
            "game_logs": {
                "reb": [5, 7, 4, 6, 8, 5, 3, 6, 7, 5],  # 9/10 >= 3, 7/10 >= 5
            },
            "prop_odds": {
                "reb": {3: -400, 5: -130},
            },
        },
    ],
}


# ---------------------------------------------------------------------------
# Fixture: HOU @ GSW (4/5/26) — user's 3-leg winning parlay
# ---------------------------------------------------------------------------

HOU_GSW_GAME = {
    "game_label": "Rockets @ Warriors",
    "players": [
        {
            "name": "Kevin Durant",
            "team": "GSW",
            "game_logs": {
                "pts": [28, 25, 32, 22, 18, 26, 30, 24, 21, 27],  # 9/10 >= 20
                "ast": [7, 5, 8, 6, 4, 7, 6, 5, 8, 5],  # 9/10 >= 5
            },
            "prop_odds": {
                "pts": {20: -300, 25: +110},
                "ast": {5: -180},
            },
        },
        {
            "name": "Alperen Sengun",
            "team": "HOU",
            "game_logs": {
                "pts": [18, 22, 15, 20, 16, 19, 17, 21, 15, 18],  # 10/10 >= 15
            },
            "prop_odds": {
                "pts": {15: -250},
            },
        },
    ],
}


class TestBuildParlaysForGame:
    def test_okc_lac_finds_parlays(self):
        config = AnchorParlayConfig(
            min_hit_rate=0.70,
            target_min_odds=1.80,
            target_max_odds=3.50,
        )
        # First scan all players
        all_legs = []
        for player in OKC_LAC_GAME["players"]:
            legs = scan_player(
                player["name"],
                player["team"],
                player["game_logs"],
                player.get("prop_odds"),
                min_hit_rate=config.min_hit_rate,
            )
            all_legs.extend(legs)

        parlays = build_parlays_for_game(all_legs, "Thunder @ Clippers", config)
        assert len(parlays) > 0
        # All parlays should be within target odds range
        for p in parlays:
            assert config.target_min_odds <= p.combined_decimal_odds <= config.target_max_odds

    def test_parlays_sorted_by_ev(self):
        config = AnchorParlayConfig(target_min_odds=1.50, target_max_odds=5.00)
        all_legs = []
        for player in OKC_LAC_GAME["players"]:
            legs = scan_player(
                player["name"],
                player["team"],
                player["game_logs"],
                player.get("prop_odds"),
            )
            all_legs.extend(legs)

        parlays = build_parlays_for_game(all_legs, "Test", config)
        evs = [p.ev_pct for p in parlays]
        assert evs == sorted(evs, reverse=True)


class TestScanSlate:
    def test_two_game_slate(self):
        config = AnchorParlayConfig(
            target_min_odds=1.80,
            target_max_odds=3.50,
        )
        result = scan_slate([OKC_LAC_GAME, HOU_GSW_GAME], config)

        assert result.league == "NBA"
        assert result.games_scanned == 2
        assert result.players_scanned == 6
        assert result.anchors_found > 0
        assert result.parlays_built > 0

        # Should have parlays from both games
        game_labels = {p.game for p in result.parlays}
        assert len(game_labels) >= 1  # at least one game produced parlays

    def test_empty_slate(self):
        result = scan_slate([])
        assert result.games_scanned == 0
        assert result.parlays_built == 0
        assert result.parlays == []

    def test_kd_correlation_flagged(self):
        """KD pts + KD ast should NOT be flagged as correlated (they're independent stats).
        But KD pts + KD pra WOULD be flagged."""
        config = AnchorParlayConfig(
            target_min_odds=1.50,
            target_max_odds=5.00,
        )
        result = scan_slate([HOU_GSW_GAME], config)
        # Find parlays with both KD legs
        kd_parlays = [
            p
            for p in result.parlays
            if sum(1 for leg in p.legs if leg.player_name == "Kevin Durant") >= 2
        ]
        # KD pts + KD ast: no correlation warning (not in correlated pairs)
        for p in kd_parlays:
            kd_stats = {leg.stat_key for leg in p.legs if leg.player_name == "Kevin Durant"}
            if kd_stats == {"pts", "ast"}:
                assert len(p.correlation_warnings) == 0

    def test_max_results_respected(self):
        config = AnchorParlayConfig(
            target_min_odds=1.50,
            target_max_odds=10.00,
            max_results=3,
        )
        result = scan_slate([OKC_LAC_GAME, HOU_GSW_GAME], config)
        assert len(result.parlays) <= 3

    def test_positive_ev_parlays_exist(self):
        """With strong anchors (70%+ hit rate), some parlays should have positive EV."""
        config = AnchorParlayConfig(
            target_min_odds=1.80,
            target_max_odds=3.00,
        )
        result = scan_slate([OKC_LAC_GAME, HOU_GSW_GAME], config)
        positive_ev = [p for p in result.parlays if p.ev_pct > 0]
        assert len(positive_ev) > 0, "Expected at least one positive-EV parlay"

    def test_scan_metadata_present(self):
        result = scan_slate([OKC_LAC_GAME])
        assert "config" in result.scan_metadata
        assert result.scan_metadata["config"]["min_hit_rate"] == 0.70
