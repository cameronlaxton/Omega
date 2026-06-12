"""Tests for omega-fit-tennis-pressure-coefficients (Phase 7 M3 PR-T4).

State classification is exclusive (one fit population per point) and the
set/match-point deltas are residuals against their flat serving-for-set/-match
parent, so the additive application in tennis_markov never double-counts a
depression that was fit twice (review finding, PR #12).
"""

from __future__ import annotations

import pytest

from omega.integrations.tennis_sackmann import ChartingMatchRow, ChartingPointRow
from omega.ops.fit_tennis_pressure import (
    accumulate_pressure_stats,
    build_pressure_deltas,
    classify_point_state,
)
from omega.trace.priors import (
    PRESSURE_GROUP_PLAYER_KEY,
    PRESSURE_SOURCE_GROUP,
    PRESSURE_SOURCE_PLAYER,
)


def _point(match_id="m1", set1=0, set2=0, gm1=2, gm2=2, pts="15-0", svr=1, winner=1):
    return ChartingPointRow(
        match_id=match_id, Set1=set1, Set2=set2, Gm1=gm1, Gm2=gm2,
        Pts=pts, Svr=svr, PtWinner=winner,
    )


def _match(match_id="m1", p1="Player A", p2="Opponent", surface="Hard", best_of=3):
    return ChartingMatchRow(
        match_id=match_id, player_1=p1, player_2=p2, surface=surface, best_of=best_of
    )


def _classify(point, server_sets=0, sets_to_win=2):
    return classify_point_state(point, server_sets=server_sets, sets_to_win=sets_to_win)


# ---------------------------------------------------------------------------
# Exclusive state classification
# ---------------------------------------------------------------------------


def test_break_point_states_in_plain_games():
    for pts in ("0-40", "15-40", "30-40", "40-AD"):
        assert _classify(_point(pts=pts)) == "break_point_against"
    assert _classify(_point(pts="40-30")) is None  # plain-game GP carries no delta
    assert _classify(_point(pts="15-0")) is None


def test_tiebreak_state_is_exclusive():
    assert _classify(_point(gm1=6, gm2=6, pts="3-2")) == "tiebreak"


def test_serving_for_set_populations_are_disjoint():
    # Game point in a serving-for-set game -> residual set-point state only.
    assert _classify(_point(gm1=5, gm2=4, pts="40-30")) == "set_point_serving"
    # Non-node point in the same game -> flat serving_for_set only.
    assert _classify(_point(gm1=5, gm2=4, pts="15-0")) == "serving_for_set"
    assert _classify(_point(gm1=6, gm2=5, pts="0-0")) == "serving_for_set"
    # Break point inside a serving-for-set game is the overlap node the
    # additive model predicts — it belongs to NO fit population.
    assert _classify(_point(gm1=5, gm2=4, pts="30-40")) is None


def test_clinch_set_upgrades_to_match_states():
    assert (
        _classify(_point(set1=1, gm1=5, gm2=3, pts="40-15"), server_sets=1)
        == "match_point_serving"
    )
    assert (
        _classify(_point(set1=1, gm1=5, gm2=3, pts="0-15"), server_sets=1)
        == "serving_for_match"
    )


def test_server_two_perspective():
    """When player 2 serves, game/set counts swap perspective."""
    point = _point(gm1=3, gm2=5, pts="40-0", svr=2, winner=2)
    assert _classify(point) == "set_point_serving"


# ---------------------------------------------------------------------------
# Aggregation, residual deltas, fallback
# ---------------------------------------------------------------------------


def _synthetic_dataset():
    """Player A: 1000 charted points (800 plain @70%, 200 plain BP @60%).
    Player B: 200 plain points @65% — below the N=500 threshold."""
    points = []
    for i in range(800):
        points.append(_point(pts="15-0", winner=1 if i < 560 else 2))
    for i in range(200):
        points.append(_point(pts="30-40", winner=1 if i < 120 else 2))
    for i in range(200):
        points.append(_point(match_id="m2", pts="15-0", winner=1 if i < 130 else 2))
    matches = [_match(), _match(match_id="m2", p1="Player B")]
    return points, matches


def test_player_deltas_hand_computed():
    points, matches = _synthetic_dataset()
    acc = accumulate_pressure_stats(points, matches)
    rows = build_pressure_deltas(acc, tour="ATP", as_of_date="2026-06-10")

    a_bp = next(
        r for r in rows
        if r.player == "Player A" and r.state == "break_point_against"
    )
    # Baseline 680/1000 = .68; BP SPW 120/200 = .60 -> delta -0.08.
    assert a_bp.delta == pytest.approx(-0.08, abs=1e-4)
    assert a_bp.source == PRESSURE_SOURCE_PLAYER
    assert a_bp.n_points == 200
    assert a_bp.surface == "hard"


def test_set_point_delta_is_residual_over_serving_for_set():
    """sp_delta = SPW(GP in SFS) − (baseline + sfs_delta), not vs baseline.

    600 plain @65% + 200 SFS non-node @62% + 200 GP-in-SFS @60%:
      baseline  = (390+124+120)/1000 = .634
      sfs_delta = .62 − .634          = −.014
      sp_delta  = .60 − (.634 − .014) = −.02   (naive vs-baseline = −.034)
    """
    points = []
    for i in range(600):
        points.append(_point(pts="15-0", winner=1 if i < 390 else 2))
    for i in range(200):
        points.append(_point(gm1=5, gm2=3, pts="15-0", winner=1 if i < 124 else 2))
    for i in range(200):
        points.append(_point(gm1=5, gm2=3, pts="40-30", winner=1 if i < 120 else 2))

    acc = accumulate_pressure_stats(points, [_match()])
    rows = build_pressure_deltas(
        acc, tour="ATP", as_of_date="2026-06-10", min_points=500
    )
    by_state = {r.state: r for r in rows if r.player == "Player A"}
    assert by_state["serving_for_set"].delta == pytest.approx(-0.014, abs=1e-4)
    assert by_state["set_point_serving"].delta == pytest.approx(-0.02, abs=1e-4)
    # The additive application reproduces the observed node SPW exactly:
    # base(.634) + sfs(−.014) + sp(−.02) = .60.
    applied = 0.634 + by_state["serving_for_set"].delta + by_state["set_point_serving"].delta
    assert applied == pytest.approx(0.60, abs=1e-4)


def test_overlap_break_points_are_excluded_from_both_fits():
    """A BP inside a serving-for-set game must not contaminate either fit."""
    points = []
    for i in range(600):
        points.append(_point(pts="15-0", winner=1 if i < 390 else 2))
    # Heavily depressed BPs inside SFS games (SPW .10) — excluded entirely.
    for i in range(100):
        points.append(_point(gm1=5, gm2=3, pts="30-40", winner=1 if i < 10 else 2))

    acc = accumulate_pressure_stats(points, [_match()])
    bucket = acc[("Player A", "hard")]
    assert "break_point_against" not in bucket.state_pts
    assert "serving_for_set" not in bucket.state_pts
    # They still count toward the baseline volume (they are service points).
    assert bucket.total_pts == 700


def test_sub_threshold_player_gets_group_means_not_zeros():
    points, matches = _synthetic_dataset()
    acc = accumulate_pressure_stats(points, matches)
    rows = build_pressure_deltas(acc, tour="ATP", as_of_date="2026-06-10")

    b_rows = [r for r in rows if r.player == "Player B"]
    assert b_rows, "sub-threshold player must still get rows (group means)"
    assert all(r.source == PRESSURE_SOURCE_GROUP for r in b_rows)
    group_bp = next(
        r for r in rows
        if r.player == PRESSURE_GROUP_PLAYER_KEY and r.state == "break_point_against"
    )
    b_bp = next(r for r in b_rows if r.state == "break_point_against")
    assert b_bp.delta == group_bp.delta
    assert b_bp.delta != 0.0  # never silent zeros


def test_group_rows_written_under_reserved_key():
    points, matches = _synthetic_dataset()
    acc = accumulate_pressure_stats(points, matches)
    rows = build_pressure_deltas(acc, tour="ATP", as_of_date="2026-06-10")
    group_rows = [r for r in rows if r.player == PRESSURE_GROUP_PLAYER_KEY]
    assert group_rows
    assert all(r.source == PRESSURE_SOURCE_GROUP for r in group_rows)


def test_unwinnered_or_unmatched_points_are_ignored():
    points = [
        _point(match_id="ghost"),  # no match metadata
        ChartingPointRow(
            match_id="m1", Set1=0, Set2=0, Gm1=2, Gm2=2, Pts="15-0", Svr=1, PtWinner=None
        ),
    ]
    acc = accumulate_pressure_stats(points, [_match()])
    assert acc == {}
