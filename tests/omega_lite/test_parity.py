"""
omega_lite ↔ canonical omega parity tests.

With a fixed seed, the sandbox port must produce numerically identical
simulation output to the canonical service for the same inputs. This is the
guarantee that lets us tell users "Mode A-sandbox is honest" — the math
isn't drifting.
"""

import pytest


SEED = 42
ITERS = 1000


# ---------------------------------------------------------------------------
# Game parity
# ---------------------------------------------------------------------------

def test_game_parity_nba():
    from omega.core.contracts.schemas import GameAnalysisRequest as CanonReq, OddsInput as CanonOdds
    from omega.core.contracts.service import analyze_game as canon_analyze

    from omega_lite.schemas import GameAnalysisRequest as LiteReq, OddsInput as LiteOdds
    from omega_lite.service import analyze_game as lite_analyze

    payload = dict(
        home_team="Boston Celtics",
        away_team="Indiana Pacers",
        league="NBA",
        n_iterations=ITERS,
        seed=SEED,
        home_context={"off_rating": 118.0, "def_rating": 108.0, "pace": 100.0},
        away_context={"off_rating": 115.0, "def_rating": 110.0, "pace": 98.0},
    )
    odds = dict(moneyline_home=-160, moneyline_away=140, spread_home=-4.5, over_under=226.5)

    canon = canon_analyze(CanonReq(**{**payload, "odds": CanonOdds(**odds)}))
    lite = lite_analyze(LiteReq(**{**payload, "odds": LiteOdds(**odds)}))

    assert canon.status == lite.status == "success"
    assert canon.simulation.home_win_prob == lite.simulation.home_win_prob
    assert canon.simulation.away_win_prob == lite.simulation.away_win_prob
    assert canon.simulation.predicted_spread == lite.simulation.predicted_spread
    assert canon.simulation.predicted_total == lite.simulation.predicted_total
    assert len(canon.edges) == len(lite.edges)
    for c_edge, l_edge in zip(canon.edges, lite.edges):
        assert c_edge.edge_pct == l_edge.edge_pct
        assert c_edge.ev_pct == l_edge.ev_pct
        assert c_edge.confidence_tier == l_edge.confidence_tier


def test_game_missing_context_skips_in_both():
    from omega.core.contracts.schemas import GameAnalysisRequest as CanonReq
    from omega.core.contracts.service import analyze_game as canon_analyze
    from omega_lite.schemas import GameAnalysisRequest as LiteReq
    from omega_lite.service import analyze_game as lite_analyze

    payload = dict(home_team="A", away_team="B", league="NBA", n_iterations=100)
    canon = canon_analyze(CanonReq(**payload))
    lite = lite_analyze(LiteReq(**payload))

    assert canon.status == lite.status == "skipped"
    assert canon.missing_requirements == lite.missing_requirements


# ---------------------------------------------------------------------------
# Prop parity (uses run_player_simulation under the hood in both)
# ---------------------------------------------------------------------------

def test_prop_parity_nba_points():
    from omega.core.contracts.schemas import PlayerPropRequest as CanonReq
    from omega.core.contracts.service import analyze_player_prop as canon_analyze
    from omega_lite.schemas import PlayerPropRequest as LiteReq
    from omega_lite.service import analyze_player_prop as lite_analyze

    payload = dict(
        player_name="Jayson Tatum",
        league="NBA",
        prop_type="pts",
        line=27.5,
        home_team="Boston Celtics",
        away_team="Indiana Pacers",
        game_date="2026-05-14",
        odds_over=-115,
        odds_under=-105,
        player_context={"pts_mean": 28.4, "pts_std": 6.2},
        n_iterations=ITERS,
        seed=SEED,
    )
    canon = canon_analyze(CanonReq(**payload))
    lite = lite_analyze(LiteReq(**payload))

    assert canon.status == lite.status == "success"
    assert canon.over_prob == lite.over_prob
    assert canon.under_prob == lite.under_prob
    assert canon.recommendation == lite.recommendation
    assert canon.edge_over == lite.edge_over
    assert canon.edge_under == lite.edge_under


def test_prop_missing_mean_skips_in_both():
    from omega.core.contracts.schemas import PlayerPropRequest as CanonReq
    from omega.core.contracts.service import analyze_player_prop as canon_analyze
    from omega_lite.schemas import PlayerPropRequest as LiteReq
    from omega_lite.service import analyze_player_prop as lite_analyze

    payload = dict(
        player_name="Some Player",
        league="NBA",
        prop_type="pts",
        line=22.5,
        home_team="Boston Celtics",
        away_team="Indiana Pacers",
        game_date="2026-05-14",
        player_context={},
        n_iterations=ITERS,
    )
    canon = canon_analyze(CanonReq(**payload))
    lite = lite_analyze(LiteReq(**payload))

    assert canon.status == lite.status == "skipped"
    assert canon.missing_requirements == lite.missing_requirements


# ---------------------------------------------------------------------------
# Slate parity
# ---------------------------------------------------------------------------

def test_slate_parity_two_games():
    from omega.core.contracts.schemas import SlateAnalysisRequest as CanonReq
    from omega.core.contracts.service import analyze_slate as canon_analyze
    from omega_lite.schemas import SlateAnalysisRequest as LiteReq
    from omega_lite.service import analyze_slate as lite_analyze

    games = [
        {
            "home_team": "Boston Celtics",
            "away_team": "Indiana Pacers",
            "home_context": {"off_rating": 118.0, "def_rating": 108.0, "pace": 100.0},
            "away_context": {"off_rating": 115.0, "def_rating": 110.0, "pace": 98.0},
            "odds": {"moneyline_home": -160, "moneyline_away": 140, "over_under": 226.5},
        },
        {
            "home_team": "Los Angeles Lakers",
            "away_team": "Golden State Warriors",
            "home_context": {"off_rating": 114.0, "def_rating": 112.0, "pace": 99.0},
            "away_context": {"off_rating": 117.0, "def_rating": 113.0, "pace": 102.0},
            "odds": {"moneyline_home": 115, "moneyline_away": -135, "over_under": 230.5},
        },
    ]

    canon = canon_analyze(CanonReq(league="NBA", date="2026-05-14", bankroll=1000, games=games))
    lite = lite_analyze(LiteReq(league="NBA", date="2026-05-14", bankroll=1000, games=games))

    assert canon.total_games == lite.total_games == 2
    assert canon.games_analyzed == lite.games_analyzed
    # Note: per-game numbers won't match because slate analyze doesn't pin a seed
    # (PR opportunity). We assert structural parity instead.
    assert len(canon.analyses) == len(lite.analyses)


# ---------------------------------------------------------------------------
# Sandbox wrapper smoke
# ---------------------------------------------------------------------------

def test_sandbox_analyze_wraps_game_result():
    from omega_lite import analyze

    out = analyze({
        "home_team": "Boston Celtics",
        "away_team": "Indiana Pacers",
        "league": "NBA",
        "n_iterations": 500,
        "seed": SEED,
        "home_context": {"off_rating": 118.0, "def_rating": 108.0, "pace": 100.0},
        "away_context": {"off_rating": 115.0, "def_rating": 110.0, "pace": 98.0},
        "odds": {"moneyline_home": -160, "moneyline_away": 140},
    }, bankroll=2500.0, session_id="sess-20260518-test")

    assert out["trace_id"].startswith("sandbox-")
    assert out["model_version"] == "omega-lite-v1"
    assert out["kind"] == "game"
    assert out["session_id"] == "sess-20260518-test"
    assert out["bankroll"] == 2500.0
    assert out["result"]["status"] == "success"
    assert "quality_gate" in out
    assert out["quality_gate"]["applied"] is True


def test_game_trace_hash_excludes_nested_odds_object():
    from omega_lite.run import _input_hash
    from omega_lite.schemas import GameAnalysisRequest

    base = {
        "home_team": "Boston Celtics",
        "away_team": "Indiana Pacers",
        "league": "NBA",
        "n_iterations": 500,
        "seed": SEED,
        "home_context": {"off_rating": 118.0, "def_rating": 108.0, "pace": 100.0},
        "away_context": {"off_rating": 115.0, "def_rating": 110.0, "pace": 98.0},
    }
    early = GameAnalysisRequest(**{
        **base,
        "odds": {"spread_home": -3.5, "spread_home_price": -110},
    })
    moved = GameAnalysisRequest(**{
        **base,
        "odds": {"spread_home": -4.0, "spread_home_price": -105},
    })

    assert _input_hash(early) == _input_hash(moved)


def test_sandbox_refuses_when_critical_missing():
    """If critical inputs are missing, the wrapped result should still skip
    AND the quality_gate should reflect the data gap."""
    from omega_lite import analyze

    out = analyze({
        "home_team": "A",
        "away_team": "B",
        "league": "NBA",
        "n_iterations": 100,
    })

    assert out["trace_id"].startswith("sandbox-")
    assert out["result"]["status"] == "skipped"
    assert out["quality_gate"]["applied"] is True
    assert out["quality_gate"]["aggregate_quality"] < 0.5
