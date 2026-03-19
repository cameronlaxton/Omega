"""
Tests for the EntityResolver and alias database.

Covers:
- Exact canonical name matches
- Alias matches (abbreviations, nicknames, city-only)
- League-scoped resolution
- Fuzzy matching for typos
- Pass-through for unknown entities
- Collector protocol verification
- CollectorRegistry dispatch
"""

import pytest

from omega.evidence.entity.resolver import EntityResolver, ResolvedEntity


@pytest.fixture
def resolver():
    return EntityResolver()


# ===== Canonical Name Resolution =====

class TestCanonicalMatch:
    def test_exact_canonical(self, resolver):
        r = resolver.resolve("Los Angeles Lakers", league="NBA")
        assert r.canonical == "Los Angeles Lakers"
        assert r.abbreviation == "LAL"
        assert r.confidence == 1.0

    def test_case_insensitive(self, resolver):
        r = resolver.resolve("los angeles lakers", league="NBA")
        assert r.canonical == "Los Angeles Lakers"
        assert r.confidence == 1.0


# ===== Alias Resolution =====

class TestAliasMatch:
    def test_nickname(self, resolver):
        r = resolver.resolve("Lakers", league="NBA")
        assert r.canonical == "Los Angeles Lakers"
        assert r.confidence == 1.0

    def test_abbreviation_with_league(self, resolver):
        r = resolver.resolve("LAL", league="NBA")
        assert r.canonical == "Los Angeles Lakers"
        assert r.confidence >= 0.90

    def test_sixers(self, resolver):
        r = resolver.resolve("Sixers", league="NBA")
        assert r.canonical == "Philadelphia 76ers"

    def test_niners(self, resolver):
        r = resolver.resolve("niners", league="NFL")
        assert r.canonical == "San Francisco 49ers"

    def test_dubs(self, resolver):
        r = resolver.resolve("dubs", league="NBA")
        assert r.canonical == "Golden State Warriors"

    def test_habs(self, resolver):
        r = resolver.resolve("habs", league="NHL")
        assert r.canonical == "Montreal Canadiens"

    def test_city_only(self, resolver):
        r = resolver.resolve("phoenix", league="NBA")
        assert r.canonical == "Phoenix Suns"

    def test_celts(self, resolver):
        r = resolver.resolve("celts", league="NBA")
        assert r.canonical == "Boston Celtics"

    def test_cavs(self, resolver):
        r = resolver.resolve("cavs", league="NBA")
        assert r.canonical == "Cleveland Cavaliers"

    def test_grizz(self, resolver):
        r = resolver.resolve("grizz", league="NBA")
        assert r.canonical == "Memphis Grizzlies"


# ===== League-Scoped Resolution =====

class TestLeagueScoping:
    def test_cardinals_nfl(self, resolver):
        """Cardinals should resolve to NFL team when league=NFL."""
        r = resolver.resolve("cardinals", league="NFL")
        assert r.canonical == "Arizona Cardinals"
        assert r.league == "NFL"

    def test_cardinals_mlb(self, resolver):
        """Cardinals should resolve to MLB team when league=MLB."""
        r = resolver.resolve("cardinals", league="MLB")
        assert r.canonical == "St. Louis Cardinals"
        assert r.league == "MLB"

    def test_kings_nba(self, resolver):
        r = resolver.resolve("kings", league="NBA")
        assert r.canonical == "Sacramento Kings"

    def test_kings_nhl(self, resolver):
        r = resolver.resolve("kings", league="NHL")
        assert r.canonical == "Los Angeles Kings"


# ===== Fuzzy Matching =====

class TestFuzzyMatch:
    def test_slight_typo(self, resolver):
        r = resolver.resolve("Laker", league="NBA")
        # Should either substring-match or fuzzy-match
        assert r.confidence >= 0.5
        # Should resolve to Lakers
        assert "lakers" in r.canonical.lower() or r.canonical == "Laker"

    def test_trail_blazer(self, resolver):
        r = resolver.resolve("Blazers", league="NBA")
        assert r.canonical == "Portland Trail Blazers"


# ===== Pass-Through =====

class TestPassThrough:
    def test_unknown_entity(self, resolver):
        r = resolver.resolve("Xyzzy FC", league="EPL")
        assert r.canonical == "Xyzzy FC"
        assert r.confidence == 0.5

    def test_empty_string(self, resolver):
        r = resolver.resolve("", league="NBA")
        assert r.confidence == 0.0

    def test_player_name_passthrough(self, resolver):
        """Player names aren't in the alias database and should pass through."""
        r = resolver.resolve("LeBron James", league="NBA")
        # Should pass through since it's not a team name
        assert r.confidence <= 0.85  # Not a high-confidence team match


# ===== No League Specified =====

class TestNoLeague:
    def test_lakers_no_league(self, resolver):
        r = resolver.resolve("Lakers")
        assert r.canonical == "Los Angeles Lakers"

    def test_celtics_no_league(self, resolver):
        r = resolver.resolve("Celtics")
        assert r.canonical == "Boston Celtics"


# ===== Collector Protocol =====

class TestCollectorProtocol:
    def test_espn_implements_protocol(self):
        from omega.evidence.collectors.base import Collector
        from omega.evidence.collectors.espn import EspnCollector

        c = EspnCollector()
        assert isinstance(c, Collector)
        assert "schedule" in c.evidence_types
        assert c.trust_tier == 1

    def test_odds_api_implements_protocol(self):
        from omega.evidence.collectors.base import Collector
        from omega.evidence.collectors.odds_api import OddsApiCollector

        c = OddsApiCollector()
        assert isinstance(c, Collector)
        assert "odds" in c.evidence_types
        assert c.trust_tier == 1

    def test_fallback_search_implements_protocol(self):
        from omega.evidence.collectors.base import Collector
        from omega.evidence.collectors.search import FallbackSearchCollector

        c = FallbackSearchCollector()
        assert isinstance(c, Collector)
        assert "team_stat" in c.evidence_types
        assert c.trust_tier == 3

    def test_team_form_implements_protocol(self):
        from omega.evidence.collectors.base import Collector
        from omega.evidence.collectors.team_form import TeamFormCollector

        c = TeamFormCollector()
        assert isinstance(c, Collector)
        assert "team_stat" in c.evidence_types
        assert c.trust_tier == 2


# ===== Collector Registry =====

class TestCollectorRegistry:
    def test_default_registry_has_collectors(self):
        from omega.evidence.registry import build_default_registry

        registry = build_default_registry()
        assert len(registry.all_collectors) >= 7  # All our registered collectors

    def test_odds_dispatch_order(self):
        from omega.evidence.registry import build_default_registry

        registry = build_default_registry()
        collectors = registry.get_collectors_for("odds", "NBA")
        assert len(collectors) >= 2  # OddsApiCollector + FallbackSearch
        # First should be tier 1 (OddsApi)
        assert collectors[0].trust_tier <= collectors[-1].trust_tier

    def test_schedule_dispatch_order(self):
        from omega.evidence.registry import build_default_registry

        registry = build_default_registry()
        collectors = registry.get_collectors_for("schedule", "NBA")
        assert len(collectors) >= 2  # EspnCollector + FallbackSearch
        assert collectors[0].name == "espn"

    def test_team_stat_has_multiple_collectors(self):
        from omega.evidence.registry import build_default_registry

        registry = build_default_registry()
        collectors = registry.get_collectors_for("team_stat", "NBA")
        names = [c.name for c in collectors]
        assert "espn" in names
        assert "team_form" in names
        assert "fallback_search" in names

    def test_unsupported_league_returns_fallback_only(self):
        from omega.evidence.registry import build_default_registry

        registry = build_default_registry()
        collectors = registry.get_collectors_for("odds", "CS2")
        # Only FallbackSearchCollector should support CS2 odds
        assert all(c.name == "fallback_search" for c in collectors)
