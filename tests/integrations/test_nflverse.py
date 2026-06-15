"""Tests for the nflverse NFL weekly-stats dispersion adapter (Phase 7 M4).

Covers the transform (position-eligibility + entity resolution), ETL fail-loud on
schema drift, NaN-value tolerance (data, not drift), alias-based exclusion,
cache-served fetch (no network), replay-mode guarding, and the end-to-end load →
fit_dispersions wiring that omega-fit-nfl-dispersion uses.

References:
  omega/integrations/nflverse.py
  omega/integrations/_etl.py
  omega/ops/fit_nfl_dispersion.py
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from omega.integrations import nflverse
from omega.integrations._etl import SourceSchemaDriftError
from omega.integrations._guards import OmegaReplayModeError
from omega.ops.fit_nfl_dispersion import fit_dispersions


def _row(player, pos, *, season=2025, week=1, rush=None, rec=None, pas=None, pid=None):
    return {
        "player_id": pid or f"id-{player}",
        "player_display_name": player,
        "position_group": pos,
        "season": season,
        "week": week,
        "rushing_yards": rush,
        "receiving_yards": rec,
        "passing_yards": pas,
    }


def test_transform_emits_position_eligible_observations():
    rows = [
        _row("Saquon Barkley", "RB", rush=95.0, rec=30.0),  # both RB-eligible
        _row("Justin Jefferson", "WR", rec=120.0, rush=0.0),  # only receiving
        _row("Patrick Mahomes", "QB", pas=300.0, rush=20.0),  # passing + rushing
        _row("Anonymous Lineman", None, rush=0.0),  # no position group -> skipped
    ]
    obs, unresolved = nflverse.build_dispersion_observations(rows)
    assert unresolved == []
    got = {(o.entity, o.stat_type, o.position_group, o.value) for o in obs}
    assert ("Saquon Barkley", "rushing_yards", "RB", 95.0) in got
    assert ("Saquon Barkley", "receiving_yards", "RB", 30.0) in got
    assert ("Justin Jefferson", "receiving_yards", "WR", 120.0) in got
    assert ("Patrick Mahomes", "passing_yards", "QB", 300.0) in got
    assert ("Patrick Mahomes", "rushing_yards", "QB", 20.0) in got
    # WR rushing is not an eligible (position, stat) pair; lineman emits nothing.
    assert not any(o.stat_type == "rushing_yards" and o.position_group == "WR" for o in obs)
    assert "Anonymous Lineman" not in {o.entity for o in obs}


def test_schema_drift_fails_loud():
    rows = [_row("Saquon Barkley", "RB", rush=95.0, rec=30.0)]
    del rows[0]["passing_yards"]  # upstream renamed/dropped a required column
    with pytest.raises(SourceSchemaDriftError) as exc:
        nflverse.build_dispersion_observations(rows)
    assert exc.value.source == "nflverse"


def test_nan_values_are_data_not_drift():
    rows = [
        _row("Justin Jefferson", float("nan"), rec=float("nan")),  # missing pos + stat
        _row("Justin Jefferson", "WR", rec=120.0),
    ]
    obs, _ = nflverse.build_dispersion_observations(rows)  # must not raise
    # The NaN-position row is skipped; only the valid receiving obs survives.
    assert [(o.stat_type, o.value) for o in obs] == [("receiving_yards", 120.0)]


def test_unresolved_player_excluded_when_alias_table_present():
    rows = [
        _row("Pat Mahomes", "QB", pas=305.0),  # alias of canonical
        _row("Mystery Rookie", "WR", rec=40.0),  # not in alias table
    ]
    alias_table = {"canonical": ["Patrick Mahomes"], "aliases": {"Pat Mahomes": "Patrick Mahomes"}}
    obs, unresolved = nflverse.build_dispersion_observations(rows, alias_table=alias_table)
    assert {o.entity for o in obs} == {"Patrick Mahomes"}  # alias resolved
    assert unresolved == ["Mystery Rookie"]


def test_fetch_served_from_cache_without_network(tmp_path):
    cache_dir = tmp_path / "nflverse"
    cache_dir.mkdir(parents=True)
    df = pd.DataFrame([_row("Saquon Barkley", "RB", rush=95.0, rec=30.0)])
    df.to_parquet(cache_dir / "nfl_player_stats_2025.parquet", index=False)

    def _never(*_a, **_kw):
        raise AssertionError("network fetch should not happen on a cache hit")

    out = nflverse.fetch_player_stats(2025, cache_root=str(tmp_path), url_opener=_never)
    assert len(out) == 1


def test_cold_fetch_blocked_in_replay_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("OMEGA_REPLAY_MODE", "1")

    def _never(*_a, **_kw):
        raise AssertionError("guard should fire before any network call")

    with pytest.raises(OmegaReplayModeError, match="OMEGA_REPLAY_MODE=1"):
        nflverse.fetch_player_stats(2025, cache_root=str(tmp_path), url_opener=_never)


def test_load_observations_end_to_end_feeds_fitter(tmp_path):
    # A WR with enough over-dispersed weekly receiving yards to fit a dispersion.
    rows = [
        _row("Justin Jefferson", "WR", week=w, rec=v)
        for w, v in enumerate([20.0, 140.0] * 9, start=1)  # n=18, var >> mean
    ]
    cache_dir = tmp_path / "nflverse"
    cache_dir.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(cache_dir / "nfl_player_stats_2025.parquet", index=False)

    def _never(*_a, **_kw):
        raise AssertionError("cache hit expected")

    observations = nflverse.load_dispersion_observations(
        2025,
        cache_root=str(tmp_path),
        alias_table={"canonical": [], "aliases": {}},  # pass-through
        url_opener=_never,
    )
    assert all(o.stat_type == "receiving_yards" for o in observations)
    assert len(observations) == 18

    fit_rows = fit_dispersions(observations, season="2025", as_of_date="2026-06-15")
    jj = next(r for r in fit_rows if r.entity == "Justin Jefferson")
    assert jj.stat_type == "receiving_yards"
    assert jj.position_group == "WR"
    assert math.isfinite(jj.nb_dispersion_k)
