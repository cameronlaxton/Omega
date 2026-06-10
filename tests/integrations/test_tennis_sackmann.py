"""Tests for the Sackmann tennis adapter (Phase 7 M3 PR-T2).

Hand-checked SPW/RPW math, half-life weighting, walkover tolerance, fail-loud
column drift, local-first reads with zero network, alias exclusion, and the
real on-disk data/tennis ATP files.
"""

from __future__ import annotations

import pytest

from omega.integrations import tennis_sackmann as ts
from omega.integrations._etl import SourceSchemaDriftError
from omega.integrations._guards import OmegaReplayModeError

_HEADER = (
    "tourney_id,tourney_name,surface,draw_size,tourney_level,tourney_date,"
    "match_num,winner_id,winner_seed,winner_entry,winner_name,winner_hand,"
    "winner_ht,winner_ioc,winner_age,loser_id,loser_seed,loser_entry,"
    "loser_name,loser_hand,loser_ht,loser_ioc,loser_age,score,best_of,round,"
    "minutes,w_ace,w_df,w_svpt,w_1stIn,w_1stWon,w_2ndWon,w_SvGms,w_bpSaved,"
    "w_bpFaced,l_ace,l_df,l_svpt,l_1stIn,l_1stWon,l_2ndWon,l_SvGms,l_bpSaved,"
    "l_bpFaced,winner_rank,winner_rank_points,loser_rank,loser_rank_points"
)


def _row(
    *,
    date="20260601",
    surface="Grass",
    winner="Jannik Sinner",
    loser="Novak Djokovic",
    w_svpt=74,
    w_1stWon=40,
    w_2ndWon=13,
    l_svpt=95,
    l_1stWon=44,
    l_2ndWon=16,
):
    return (
        f"2026-001,Test,{surface},32,A,{date},300,1,,,{winner},R,191,ITA,24.0,"
        f"2,,,{loser},R,188,SRB,39.0,7-6(5) 6-4,3,F,136,8,2,{w_svpt},52,"
        f"{w_1stWon},{w_2ndWon},11,3,3,9,3,{l_svpt},58,{l_1stWon},{l_2ndWon},"
        f"11,8,9,14,2570,8,3660"
    )


def _csv(*rows) -> str:
    return "\n".join([_HEADER, *rows]) + "\n"


def test_spw_rpw_hand_computed():
    rows = ts.parse_matches(_csv(_row()))
    rates = ts.compute_rolling_rates(rows, as_of_date="2026-06-10")
    sinner = rates[("Jannik Sinner", "grass")]
    assert sinner["spw_won"] / sinner["spw_pts"] == pytest.approx(53 / 74)
    assert sinner["rpw_won"] / sinner["rpw_pts"] == pytest.approx(35 / 95)
    djokovic = rates[("Novak Djokovic", "grass")]
    assert djokovic["spw_won"] / djokovic["spw_pts"] == pytest.approx(60 / 95)
    assert djokovic["rpw_won"] / djokovic["rpw_pts"] == pytest.approx(21 / 74)


def test_half_life_downweights_old_matches():
    """A match ~12 months old carries half the weight of a fresh one."""
    fresh_great = _row(date="20260601", w_svpt=100, w_1stWon=90, w_2ndWon=0)
    old_poor = _row(date="20250601", w_svpt=100, w_1stWon=40, w_2ndWon=0)
    rows = ts.parse_matches(_csv(fresh_great, old_poor))
    rates = ts.compute_rolling_rates(rows, as_of_date="2026-06-10")
    spw = rates[("Jannik Sinner", "grass")]
    blended = spw["spw_won"] / spw["spw_pts"]
    # Unweighted mean would be 0.65; half-life pulls toward the fresh 0.90.
    assert blended > 0.72
    assert spw["matches"] == 2  # match count stays unweighted


def test_walkover_rows_are_tolerated_but_excluded():
    walkover = _row(w_svpt="", w_1stWon="", w_2ndWon="", l_svpt="", l_1stWon="", l_2ndWon="")
    rows = ts.parse_matches(_csv(walkover))
    assert len(rows) == 1  # validation tolerates blank stats
    rates = ts.compute_rolling_rates(rows, as_of_date="2026-06-10")
    assert rates == {}  # but the match contributes nothing


def test_renamed_column_fails_loud():
    broken = _csv(_row()).replace("w_svpt", "w_serve_points")
    with pytest.raises(SourceSchemaDriftError) as exc:
        ts.parse_matches(broken)
    assert exc.value.source == "sackmann"


def test_priors_built_with_min_matches_gate():
    rows = ts.parse_matches(_csv(_row(), _row(date="20260520"), _row(date="20260510")))
    priors, unresolved = ts.build_tennis_priors(
        rows, tour="atp", as_of_date="2026-06-10", min_matches=3
    )
    assert unresolved == []
    by_player = {p.player: p for p in priors}
    assert by_player["Jannik Sinner"].n_matches == 3
    assert by_player["Jannik Sinner"].surface == "grass"
    assert by_player["Jannik Sinner"].tour == "ATP"
    assert by_player["Jannik Sinner"].spw_pct == pytest.approx(53 / 74, abs=1e-3)


def test_alias_exclusion_for_unknown_players():
    rows = ts.parse_matches(_csv(_row(winner="Mystery Qualifier")))
    alias_table = {"canonical": ["Novak Djokovic"], "aliases": {}}
    priors, unresolved = ts.build_tennis_priors(
        rows, tour="atp", as_of_date="2026-06-10", alias_table=alias_table, min_matches=1
    )
    assert unresolved == ["Mystery Qualifier"]
    assert [p.player for p in priors] == ["Novak Djokovic"]


def test_local_first_read_makes_no_network_call(tmp_path):
    local = tmp_path / "atp_matches_2026.csv"
    local.write_text(_csv(_row()), encoding="utf-8")

    def _never(*_a, **_kw):
        raise AssertionError("local file present; network must not be touched")

    text = ts.fetch_matches_csv(
        "atp", 2026, local_root=tmp_path, cache_root=str(tmp_path / "cache"), url_opener=_never
    )
    assert "Jannik Sinner" in text


def test_cold_remote_fetch_blocked_in_replay_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("OMEGA_REPLAY_MODE", "1")
    with pytest.raises(OmegaReplayModeError):
        ts.fetch_matches_csv(
            "wta", 2026, local_root=tmp_path, cache_root=str(tmp_path / "cache")
        )


def test_invalid_tour_rejected():
    with pytest.raises(ValueError, match="tour must be one of"):
        ts.fetch_matches_csv("itf", 2026)


def test_real_local_atp_files_produce_priors():
    """The repo's own data/tennis ATP CSVs parse and yield plausible rates."""
    priors, _ = ts.load_tennis_priors(
        "atp",
        [2025, 2026],
        as_of_date="2026-06-10",
        min_matches=5,
        url_opener=lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("local files must satisfy the read")
        ),
    )
    assert len(priors) > 50
    for prior in priors:
        assert 0.45 < prior.spw_pct < 0.85
        assert 0.15 < prior.rpw_pct < 0.55
