"""Calibration-league bucket resolution (audit remediation Phase C2).

A profile fitted under a canonical bucket code (e.g. ``FIFA_INTL``, ``EPL``) must
be selected for every runtime league that maps to that bucket, instead of missing
on an exact-string mismatch (the bug that left the World Cup draw profile keyed
``WORLD_CUP`` unreachable by live ``FIFA_WORLD_CUP_2026`` traces).
"""

from omega.core.calibration.league_buckets import (
    CALIBRATION_LEAGUE_BUCKETS,
    resolve_calibration_bucket,
)
from omega.core.calibration.registry import CalibrationRegistry


def test_resolve_bucket_naming_aliases():
    # Pure naming aliases collapse to the canonical league code.
    assert resolve_calibration_bucket("PREMIER_LEAGUE") == "EPL"
    assert resolve_calibration_bucket("premier_league") == "EPL"  # case-insensitive
    assert resolve_calibration_bucket("LALIGA") == "LA_LIGA"


def test_resolve_bucket_competitive_international():
    # The live bivariate-DC World Cup maps to the FIFA_INTL calibration bucket.
    assert resolve_calibration_bucket("FIFA_WORLD_CUP_2026") == "FIFA_INTL"


def test_resolve_bucket_unmapped_passthrough():
    # An unmapped league is returned uppercased and unchanged.
    assert resolve_calibration_bucket("NBA") == "NBA"
    assert resolve_calibration_bucket("mlb") == "MLB"


def test_friendlies_not_bucketed_with_intl():
    # FIFA_FRIENDLY must stay its own bucket (different draw base-rate, excluded
    # from the fifa_intl fit dataset) — never pooled into FIFA_INTL.
    assert resolve_calibration_bucket("FIFA_FRIENDLY") == "FIFA_FRIENDLY"
    assert CALIBRATION_LEAGUE_BUCKETS.get("FIFA_FRIENDLY") != "FIFA_INTL"


def test_get_production_resolves_through_bucket(monkeypatch):
    """A FIFA_INTL-keyed draw profile is selected for a FIFA_WORLD_CUP_2026 request."""
    registry = CalibrationRegistry()
    profiles_data = {
        "profiles": [
            {
                "profile_id": "iso_fifa_intl_draw_v1",
                "method": "isotonic",
                "league": "FIFA_INTL",
                "market": "draw",
                "status": "production",
                "context_slice": None,
                "version": 1,
                "dataset_hash": "deadbeef",
                "params": {"calibration_map": {"0.25": 0.25, "0.75": 0.75}},
                "metrics": {},
                "training_window": "fitted",
                "sample_size": 300,
            },
            {
                "profile_id": "iso_epl_game_v1",
                "method": "isotonic",
                "league": "EPL",
                "market": "game",
                "status": "production",
                "context_slice": None,
                "version": 1,
                "dataset_hash": "cafef00d",
                "params": {"calibration_map": {"0.25": 0.25, "0.75": 0.75}},
                "metrics": {},
                "training_window": "fitted",
                "sample_size": 300,
            },
        ]
    }
    monkeypatch.setattr(registry, "_load", lambda: profiles_data)

    # FIFA_WORLD_CUP_2026 -> FIFA_INTL bucket -> the draw profile resolves.
    prof = registry.get_production("FIFA_WORLD_CUP_2026", market="draw")
    assert prof is not None
    assert prof.profile_id == "iso_fifa_intl_draw_v1"

    # PREMIER_LEAGUE alias -> EPL bucket -> the EPL game profile resolves.
    prof2 = registry.get_production("PREMIER_LEAGUE", market="game")
    assert prof2 is not None
    assert prof2.profile_id == "iso_epl_game_v1"

    # A league with no bucket + no profile still returns None (static fallback).
    assert registry.get_production("WORLD_CUP", market="draw") is None
