"""
Tests for the data pipeline (Phase 5).

Covers:
- Data models (SourceAttribution, SportsFact, FactBundle, SearchResult)
- Source trust config (trust tiers, domain mapping)
- Stat normalizer
- Sanity validator
- Freshness validator
- Fusion (fuse_facts, score_confidence)
- Odds API client (unit tests with mocked HTTP)
- ESPN client (unit tests with mocked HTTP)
- Web search client (unit tests with mocked HTTP)
- Retrieval orchestrator (integration with mocks)
"""

import json
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest


# ===== Data Models =====

class TestDataModels:
    def test_source_attribution_defaults(self):
        from omega.research.data.models import SourceAttribution
        attr = SourceAttribution(source_name="test")
        assert attr.source_name == "test"
        assert attr.trust_tier == 4
        assert attr.confidence == 0.5
        assert attr.fetched_at is not None

    def test_sports_fact_creation(self):
        from omega.research.data.models import SourceAttribution, SportsFact
        attr = SourceAttribution(source_name="espn", trust_tier=1, confidence=0.95)
        fact = SportsFact(
            key="pts_per_game", value=115.2, data_type="team_stat",
            entity="Lakers", league="NBA", attribution=attr,
        )
        assert fact.key == "pts_per_game"
        assert fact.value == 115.2
        assert fact.normalized is False
        assert fact.validated is False

    def test_fact_bundle_empty(self):
        from omega.research.data.models import FactBundle
        bundle = FactBundle(slot_key="test", data_type="odds", entity="Lakers", league="NBA")
        assert bundle.facts == []
        assert bundle.fused_value is None
        assert bundle.fused_confidence == 0.0

    def test_search_result(self):
        from omega.research.data.models import SearchResult
        sr = SearchResult(url="https://espn.com/nba", title="NBA", snippet="data", domain="espn.com")
        assert sr.domain == "espn.com"


# ===== Source Config =====

class TestSourceConfig:
    def test_direct_api_tier_1(self):
        from omega.research.data.sources.config import get_trust_tier
        assert get_trust_tier("espn") == 1
        assert get_trust_tier("odds_api") == 1

    def test_reference_site_tier_2(self):
        from omega.research.data.sources.config import get_trust_tier
        assert get_trust_tier("basketball-reference.com") == 2
        assert get_trust_tier("pro-football-reference.com") == 2

    def test_major_sports_site_tier_3(self):
        from omega.research.data.sources.config import get_trust_tier
        assert get_trust_tier("espn.com") == 3
        assert get_trust_tier("covers.com") == 3

    def test_unknown_source_tier_4(self):
        from omega.research.data.sources.config import get_trust_tier
        assert get_trust_tier("randomsite.xyz") == 4

    def test_url_parsing(self):
        from omega.research.data.sources.config import get_trust_tier
        assert get_trust_tier("https://www.espn.com/nba/scores") == 3

    def test_subdomain_matching(self):
        from omega.research.data.sources.config import get_trust_tier
        assert get_trust_tier("stats.espn.com") == 3

    def test_confidence_for_tier(self):
        from omega.research.data.sources.config import get_confidence_for_tier
        assert get_confidence_for_tier(1) == 0.95
        assert get_confidence_for_tier(2) == 0.90
        assert get_confidence_for_tier(4) == 0.50
        assert get_confidence_for_tier(99) == 0.50


# ===== Stat Normalizer =====

class TestStatNormalizer:
    def test_percentage_normalization(self):
        from omega.research.data.normalizers.stats import normalize_stat_value
        assert normalize_stat_value("fg_pct", 48.5) == pytest.approx(0.485)
        assert normalize_stat_value("fg_pct", 0.485) == pytest.approx(0.485)

    def test_string_to_number(self):
        from omega.research.data.normalizers.stats import normalize_stat_value
        assert normalize_stat_value("pts_per_game", "115.2") == pytest.approx(115.2)

    def test_percentage_string_strip(self):
        from omega.research.data.normalizers.stats import normalize_stat_value
        assert normalize_stat_value("fg_pct", "48.5%") == pytest.approx(0.485)

    def test_none_passthrough(self):
        from omega.research.data.normalizers.stats import normalize_stat_value
        assert normalize_stat_value("anything", None) is None

    def test_non_pct_key_unchanged(self):
        from omega.research.data.normalizers.stats import normalize_stat_value
        assert normalize_stat_value("pts_per_game", 115.2) == pytest.approx(115.2)


# ===== Sanity Validator =====

class TestSanityValidator:
    def _make_fact(self, key, value):
        from omega.research.data.models import SourceAttribution, SportsFact
        attr = SourceAttribution(source_name="test")
        return SportsFact(
            key=key, value=value, data_type="team_stat",
            entity="Lakers", league="NBA", attribution=attr,
        )

    def test_valid_off_rating(self):
        from omega.research.data.validators.sanity import validate_sanity
        facts = [self._make_fact("off_rating", 112.5)]
        result = validate_sanity(facts)
        assert len(result) == 1

    def test_insane_off_rating(self):
        from omega.research.data.validators.sanity import validate_sanity
        facts = [self._make_fact("off_rating", 500.0)]
        result = validate_sanity(facts)
        assert len(result) == 0

    def test_none_value_passes(self):
        from omega.research.data.validators.sanity import validate_sanity
        facts = [self._make_fact("off_rating", None)]
        result = validate_sanity(facts)
        assert len(result) == 1

    def test_unknown_key_passes(self):
        from omega.research.data.validators.sanity import validate_sanity
        facts = [self._make_fact("custom_metric", 999999)]
        result = validate_sanity(facts)
        assert len(result) == 1

    def test_odds_in_range(self):
        from omega.research.data.validators.sanity import validate_sanity
        facts = [
            self._make_fact("moneyline_home", -150),
            self._make_fact("spread_home", -3.5),
            self._make_fact("total", 224.5),
        ]
        result = validate_sanity(facts)
        assert len(result) == 3

    def test_spread_out_of_range(self):
        from omega.research.data.validators.sanity import validate_sanity
        facts = [self._make_fact("spread_home", -100.0)]
        result = validate_sanity(facts)
        assert len(result) == 0


# ===== Freshness Validator =====

class TestFreshnessValidator:
    def test_fresh_fact_passes(self):
        from omega.research.data.models import SourceAttribution, SportsFact
        from omega.research.data.validators.freshness import validate_freshness

        attr = SourceAttribution(
            source_name="test",
            fetched_at=datetime.now(timezone.utc),
        )
        fact = SportsFact(
            key="pts", value=25, data_type="odds",
            entity="Lakers", league="NBA", attribution=attr,
        )
        result = validate_freshness([fact], "odds")
        assert len(result) == 1
        assert result[0].validated is True

    def test_stale_fact_filtered(self):
        from omega.research.data.models import SourceAttribution, SportsFact
        from omega.research.data.validators.freshness import validate_freshness

        attr = SourceAttribution(
            source_name="test",
            fetched_at=datetime.now(timezone.utc) - timedelta(days=30),
        )
        fact = SportsFact(
            key="pts", value=25, data_type="odds",
            entity="Lakers", league="NBA", attribution=attr,
        )
        result = validate_freshness([fact], "odds")
        assert len(result) == 0


# ===== Fusion =====

class TestFusion:
    def _make_fact(self, key, value, trust_tier=3, source="test"):
        from omega.research.data.models import SourceAttribution, SportsFact
        attr = SourceAttribution(source_name=source, trust_tier=trust_tier, confidence=0.8)
        return SportsFact(
            key=key, value=value, data_type="team_stat",
            entity="Lakers", league="NBA", attribution=attr,
        )

    def test_fuse_single_source(self):
        from omega.research.data.models import FactBundle
        from omega.research.data.fusion.fuser import fuse_facts

        bundle = FactBundle(
            slot_key="test", data_type="team_stat", entity="Lakers", league="NBA",
            facts=[self._make_fact("off_rating", 115.2)],
        )
        result = fuse_facts(bundle)
        assert result["off_rating"] == 115.2

    def test_fuse_multi_source_picks_best_trust(self):
        from omega.research.data.models import FactBundle
        from omega.research.data.fusion.fuser import fuse_facts

        bundle = FactBundle(
            slot_key="test", data_type="team_stat", entity="Lakers", league="NBA",
            facts=[
                self._make_fact("off_rating", 115.2, trust_tier=3, source="espn"),
                self._make_fact("off_rating", 114.8, trust_tier=2, source="bball-ref"),
            ],
        )
        result = fuse_facts(bundle)
        # Should pick trust_tier 2 (better)
        assert result["off_rating"] == 114.8

    def test_fuse_empty_bundle(self):
        from omega.research.data.models import FactBundle
        from omega.research.data.fusion.fuser import fuse_facts

        bundle = FactBundle(
            slot_key="test", data_type="team_stat", entity="Lakers", league="NBA",
        )
        result = fuse_facts(bundle)
        assert result == {}

    def test_score_confidence(self):
        from omega.research.data.models import FactBundle
        from omega.research.data.fusion.fuser import score_confidence

        bundle = FactBundle(
            slot_key="test", data_type="team_stat", entity="Lakers", league="NBA",
            facts=[
                self._make_fact("off_rating", 115.2, trust_tier=2, source="src1"),
                self._make_fact("off_rating", 115.0, trust_tier=2, source="src2"),
            ],
        )
        score = score_confidence(bundle)
        assert 0.0 < score <= 1.0

    def test_score_confidence_empty(self):
        from omega.research.data.models import FactBundle
        from omega.research.data.fusion.fuser import score_confidence

        bundle = FactBundle(
            slot_key="test", data_type="team_stat", entity="Lakers", league="NBA",
        )
        assert score_confidence(bundle) == 0.0


# ===== Odds API Client =====

class TestOddsApiClient:
    def test_league_sport_mapping(self):
        from omega.research.data.acquisition.odds_api import LEAGUE_SPORT_MAPPING
        assert LEAGUE_SPORT_MAPPING["NBA"] == "basketball_nba"
        assert LEAGUE_SPORT_MAPPING["NFL"] == "americanfootball_nfl"
        assert "MLB" in LEAGUE_SPORT_MAPPING

    def test_get_sport_key_unsupported(self):
        from omega.research.data.acquisition.odds_api import _get_sport_key
        assert _get_sport_key("CURLING") is None

    def test_no_api_key_returns_empty(self):
        from omega.research.data.acquisition.odds_api import get_upcoming_odds
        with patch.dict("os.environ", {}, clear=True):
            result = get_upcoming_odds("NBA")
            assert result == []

    def test_extract_consensus_odds(self):
        from omega.research.data.acquisition.odds_api import extract_consensus_odds
        games = [{
            "game_id": "abc123",
            "league": "NBA",
            "home_team": "Lakers",
            "away_team": "Celtics",
            "commence_time": "2026-03-18T00:00:00Z",
            "bookmakers": [
                {
                    "name": "DraftKings",
                    "markets": {
                        "h2h": [
                            {"name": "Lakers", "price": -150},
                            {"name": "Celtics", "price": 130},
                        ],
                        "spreads": [
                            {"name": "Lakers", "price": -110, "point": -3.5},
                        ],
                        "totals": [
                            {"name": "Over", "price": -110, "point": 224.5},
                        ],
                    },
                },
            ],
        }]
        consensus = extract_consensus_odds(games)
        assert len(consensus) == 1
        assert consensus[0]["moneyline_home"] == -150
        assert consensus[0]["spread_home"] == -3.5
        assert consensus[0]["total"] == 224.5

    def test_check_api_status_no_key(self):
        from omega.research.data.acquisition.odds_api import check_api_status
        with patch.dict("os.environ", {}, clear=True):
            status = check_api_status()
            assert status["status"] == "no_key"


# ===== ESPN Client =====

class TestEspnClient:
    def test_league_paths(self):
        from omega.research.data.acquisition.espn import LEAGUE_PATHS
        assert "NBA" in LEAGUE_PATHS
        assert "NFL" in LEAGUE_PATHS
        assert LEAGUE_PATHS["NBA"] == "basketball/nba"

    def test_unsupported_league(self):
        from omega.research.data.acquisition.espn import get_todays_games
        result = get_todays_games("CURLING")
        assert result == []

    def test_parse_competitors(self):
        from omega.research.data.acquisition.espn import _parse_competitors
        competition = {
            "competitors": [
                {
                    "id": "1",
                    "homeAway": "home",
                    "team": {"displayName": "Lakers", "abbreviation": "LAL"},
                    "score": "110",
                    "records": [{"summary": "30-20"}],
                },
                {
                    "id": "2",
                    "homeAway": "away",
                    "team": {"displayName": "Celtics", "abbreviation": "BOS"},
                    "score": "105",
                    "records": [{"summary": "35-15"}],
                },
            ],
        }
        home, away = _parse_competitors(competition)
        assert home["name"] == "Lakers"
        assert away["name"] == "Celtics"
        assert home["abbreviation"] == "LAL"

    def test_parse_odds(self):
        from omega.research.data.acquisition.espn import _parse_odds
        competition = {
            "odds": [
                {
                    "details": "LAL -3.5",
                    "overUnder": 224.5,
                    "spread": -3.5,
                    "provider": {"name": "ESPN BET"},
                },
            ],
        }
        odds = _parse_odds(competition)
        assert odds["spread"] == "LAL -3.5"
        assert odds["over_under"] == 224.5


# ===== Web Search Client =====

class TestSearchClient:
    def test_auto_generate_queries_odds(self):
        from omega.core.models import GatherSlot
        from omega.research.data.acquisition.search import _auto_generate_queries

        slot = GatherSlot(key="odds_lakers", data_type="odds", entity="Lakers", league="NBA")
        queries = _auto_generate_queries(slot)
        assert len(queries) == 1
        assert "odds" in queries[0].lower()
        assert "lakers" in queries[0].lower()

    def test_auto_generate_queries_schedule(self):
        from omega.core.models import GatherSlot
        from omega.research.data.acquisition.search import _auto_generate_queries

        slot = GatherSlot(key="sched_lakers", data_type="schedule", entity="Lakers", league="NBA")
        queries = _auto_generate_queries(slot)
        assert "schedule" in queries[0].lower()

    def test_try_parse_json_valid(self):
        from omega.research.data.acquisition.search import _try_parse_json
        result = _try_parse_json('{"off_rating": 115.2, "def_rating": 108.5}')
        assert result == {"off_rating": 115.2, "def_rating": 108.5}

    def test_try_parse_json_markdown_wrapped(self):
        from omega.research.data.acquisition.search import _try_parse_json
        result = _try_parse_json('```json\n{"pts": 25}\n```')
        assert result == {"pts": 25}

    def test_try_parse_json_invalid(self):
        from omega.research.data.acquisition.search import _try_parse_json
        result = _try_parse_json("This is not JSON at all")
        assert result is None

    def test_extract_domain(self):
        from omega.research.data.acquisition.search import _extract_domain
        assert _extract_domain("https://www.espn.com/nba/scores") == "espn.com"
        assert _extract_domain("https://basketball-reference.com/teams") == "basketball-reference.com"

    def test_no_backends_returns_empty(self):
        from omega.core.models import GatherSlot
        from omega.research.data.acquisition.search import _execute_search

        slot = GatherSlot(key="test", data_type="odds", entity="Lakers", league="NBA")
        with patch.dict("os.environ", {}, clear=True):
            result = _execute_search("test query", slot)
            assert result == []


# ===== Retrieval Orchestrator =====

class TestRetrievalOrchestrator:
    def test_session_cache(self):
        from omega.research.data.orchestration.retrieval import _SessionCache
        cache = _SessionCache()
        cache.put("odds", "Lakers", "NBA", {"spread": -3.5})
        result = cache.get("odds", "Lakers", "NBA")
        assert result == {"spread": -3.5}

    def test_session_cache_miss(self):
        from omega.research.data.orchestration.retrieval import _SessionCache
        cache = _SessionCache()
        assert cache.get("odds", "Lakers", "NBA") is None

    def test_session_cache_lru_eviction(self):
        from omega.research.data.orchestration.retrieval import _SessionCache
        cache = _SessionCache(max_size=2)
        cache.put("a", "e1", "NBA", {"v": 1})
        cache.put("b", "e2", "NBA", {"v": 2})
        cache.put("c", "e3", "NBA", {"v": 3})  # should evict first
        assert cache.get("a", "e1", "NBA") is None
        assert cache.get("b", "e2", "NBA") is not None

    def test_retrieve_facts_all_unfilled_no_apis(self):
        """Without API keys or mocks, all slots should come back unfilled but not crash."""
        from omega.core.models import GatherSlot
        from omega.research.data.orchestration.retrieval import retrieve_facts

        slots = [
            GatherSlot(key="odds_lakers", data_type="odds", entity="Lakers", league="NBA"),
            GatherSlot(key="sched_lakers", data_type="schedule", entity="Lakers", league="NBA"),
        ]
        with patch.dict("os.environ", {}, clear=True):
            # Mock out the ESPN call so it doesn't hit the network
            with patch("omega.research.data.acquisition.espn.get_todays_games", return_value=[]):
                with patch("omega.research.data.acquisition.odds_api.get_upcoming_odds", return_value=[]):
                    facts = retrieve_facts(slots)
        assert len(facts) == 2

    def test_retrieve_facts_with_mocked_espn(self):
        """ESPN direct API returns schedule data."""
        from omega.core.models import GatherSlot
        from omega.research.data.orchestration.retrieval import retrieve_facts

        mock_games = [{
            "game_id": "401",
            "league": "NBA",
            "name": "Lakers vs Celtics",
            "home_team": {"name": "Los Angeles Lakers"},
            "away_team": {"name": "Boston Celtics"},
        }]

        slot = GatherSlot(key="sched_lakers", data_type="schedule", entity="Lakers", league="NBA")

        with patch("omega.research.data.acquisition.espn.get_todays_games", return_value=mock_games):
            facts = retrieve_facts([slot])

        assert len(facts) == 1
        assert facts[0].filled is True
        assert facts[0].result is not None
        assert facts[0].result.source == "espn"

    def test_retrieve_facts_with_mocked_odds(self):
        """Odds API direct returns odds data."""
        from omega.core.models import GatherSlot
        from omega.research.data.orchestration.retrieval import retrieve_facts

        mock_games = [{
            "game_id": "abc",
            "league": "NBA",
            "home_team": "Lakers",
            "away_team": "Celtics",
            "commence_time": "2026-03-18T00:00:00Z",
            "bookmakers": [{
                "name": "DK",
                "markets": {
                    "h2h": [
                        {"name": "Lakers", "price": -150},
                        {"name": "Celtics", "price": 130},
                    ],
                    "spreads": [{"name": "Lakers", "price": -110, "point": -3.5}],
                    "totals": [{"name": "Over", "price": -110, "point": 224.5}],
                },
            }],
        }]

        slot = GatherSlot(key="odds_lakers", data_type="odds", entity="Lakers", league="NBA")

        with patch("omega.research.data.acquisition.odds_api.get_upcoming_odds", return_value=mock_games):
            facts = retrieve_facts([slot])

        assert len(facts) == 1
        assert facts[0].filled is True
        assert facts[0].result.source == "odds_api"
