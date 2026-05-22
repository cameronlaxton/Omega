"""
Tests for structured evidence signals (Phase A — capture).

Covers the typed model, the sport-tagged taxonomy registry, and the
zero-behavior-change wiring into the request schemas.
"""

import warnings

import pytest
from pydantic import ValidationError

from omega.core.contracts.evidence import (
    SIGNAL_REGISTRY,
    EvidenceSignal,
    resolve_archetype,
    signal_applies,
    signal_applies_to_league,
)


def _known_signal(**overrides):
    """A valid EvidenceSignal matching a registry entry, with overrides."""
    base = dict(
        signal_type="recent_form",
        category="player_form",
        plane="player",
        value=[27.0, 31.0, 24.0],
        source="boxscore_derived",
        confidence=0.8,
        window="last_5",
        stat_key="pts",
    )
    base.update(overrides)
    return base


class TestEvidenceSignalValidation:
    """EvidenceSignal field-level validation."""

    def test_valid_signal_constructs(self):
        sig = EvidenceSignal(**_known_signal())
        assert sig.signal_type == "recent_form"
        assert sig.confidence == 0.8
        assert sig.direction is None

    def test_confidence_lower_bound(self):
        with pytest.raises(ValidationError):
            EvidenceSignal(**_known_signal(confidence=-0.1))

    def test_confidence_upper_bound(self):
        with pytest.raises(ValidationError):
            EvidenceSignal(**_known_signal(confidence=1.5))

    def test_extra_keys_forbidden(self):
        with pytest.raises(ValidationError):
            EvidenceSignal(**_known_signal(bogus_key="x"))

    def test_invalid_window_rejected(self):
        with pytest.raises(ValidationError):
            EvidenceSignal(**_known_signal(window="last_42"))

    def test_invalid_category_rejected(self):
        with pytest.raises(ValidationError):
            EvidenceSignal(**_known_signal(category="vibes"))

    def test_value_accepts_bool_and_scalar_and_series(self):
        EvidenceSignal(
            **_known_signal(
                signal_type="last_game_outlier", category="player_form", value=True
            )
        )
        EvidenceSignal(
            **_known_signal(
                signal_type="opponent_stat_rank", category="matchup", value=3
            )
        )
        EvidenceSignal(**_known_signal(value=[10.0, 12.5]))


class TestRegistryWarnings:
    """The validator warns (never raises) on registry disagreement."""

    def test_unknown_signal_type_warns_not_raises(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sig = EvidenceSignal(**_known_signal(signal_type="totally_made_up"))
        assert sig.signal_type == "totally_made_up"  # constructed, not rejected
        assert any("unknown signal_type" in str(w.message) for w in caught)

    def test_plane_mismatch_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # usage_spike is a player-plane signal
            EvidenceSignal(
                signal_type="usage_spike",
                category="player_form",
                plane="game",
                value=0.1,
                source="agent_reasoning",
                confidence=0.6,
                window="matchup",
            )
        assert any("plane" in str(w.message) for w in caught)

    def test_missing_required_stat_key_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            EvidenceSignal(
                signal_type="recent_form",
                category="player_form",
                plane="player",
                value=[20.0],
                source="boxscore_derived",
                confidence=0.7,
                window="last_5",
                stat_key=None,
            )
        assert any("stat_key" in str(w.message) for w in caught)

    def test_known_signal_no_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            EvidenceSignal(**_known_signal())
        assert [w for w in caught if issubclass(w.category, UserWarning)] == []


class TestTaxonomyRegistry:
    """Sport-tagged signal taxonomy."""

    def test_registry_is_populated(self):
        assert len(SIGNAL_REGISTRY) >= 15

    def test_registry_keys_match_signal_type(self):
        for key, spec in SIGNAL_REGISTRY.items():
            assert key == spec.signal_type

    def test_resolve_archetype_reuses_engine_map(self):
        assert resolve_archetype("NBA") == "basketball"
        assert resolve_archetype("nhl") == "hockey"
        assert resolve_archetype("MLB") == "baseball"
        assert resolve_archetype("NOT_A_LEAGUE") is None

    def test_universal_signal_applies_everywhere(self):
        # recent_form is universal
        for archetype in ("basketball", "baseball", "tennis", "esports"):
            assert signal_applies("recent_form", archetype) is True

    def test_sport_specific_signal_gating(self):
        # usage_spike applies to basketball/hockey only
        assert signal_applies("usage_spike", "basketball") is True
        assert signal_applies("usage_spike", "hockey") is True
        assert signal_applies("usage_spike", "baseball") is False
        # park_factor_evidence is baseball-only
        assert signal_applies("park_factor_evidence", "baseball") is True
        assert signal_applies("park_factor_evidence", "basketball") is False

    def test_unknown_signal_never_applies(self):
        assert signal_applies("totally_made_up", "basketball") is False

    def test_unmapped_archetype_never_applies(self):
        assert signal_applies("recent_form", None) is False

    def test_signal_applies_to_league(self):
        assert signal_applies_to_league("usage_spike", "NBA") is True
        assert signal_applies_to_league("usage_spike", "MLB") is False
        assert signal_applies_to_league("course_fit", "PGA") is True


class TestSchemaWiring:
    """evidence field on both request models — zero behavior change."""

    def test_player_prop_request_defaults_empty_evidence(self):
        from omega.core.contracts.schemas import PlayerPropRequest

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            req = PlayerPropRequest(
                player_name="Test Player",
                league="NBA",
                prop_type="pts",
                line=20.5,
                home_team="Home",
                away_team="Away",
                game_date="2026-05-22",
            )
        assert req.evidence == []

    def test_game_analysis_request_defaults_empty_evidence(self):
        from omega.core.contracts.schemas import GameAnalysisRequest

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            req = GameAnalysisRequest(home_team="H", away_team="A", league="NBA")
        assert req.evidence == []

    def test_request_accepts_evidence_list(self):
        from omega.core.contracts.schemas import PlayerPropRequest

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            req = PlayerPropRequest(
                player_name="Test Player",
                league="NBA",
                prop_type="pts",
                line=20.5,
                home_team="Home",
                away_team="Away",
                game_date="2026-05-22",
                evidence=[_known_signal()],
            )
        assert len(req.evidence) == 1
        assert isinstance(req.evidence[0], EvidenceSignal)
