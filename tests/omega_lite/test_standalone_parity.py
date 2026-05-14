"""
omega_lite ↔ omega_lite_standalone parity tests.

With a fixed seed, the single-file standalone module emitted by
`python scripts/build_omega_lite.py --single-file` must produce numerically
identical output to the omega_lite package for the same inputs. This locks
in the contract that the sandbox-uploaded `omega_lite_standalone.py` is
bit-identical math to canonical omega — so an agent running it inside an
LLM analysis tool produces the same numbers as the local FastAPI service.

The standalone file is generated, not hand-written. If these tests fail
after a build, rerun the build script and check that --single-file inlines
every required module without dropping symbols.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
STANDALONE = REPO_ROOT / "omega_lite_standalone.py"


@pytest.fixture(scope="module")
def standalone():
    """Import the standalone module, regenerating it if missing."""
    if not STANDALONE.is_file():
        pytest.skip(
            "omega_lite_standalone.py not present — run "
            "`python scripts/build_omega_lite.py --single-file` to generate it."
        )
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import importlib
    # Force reimport so we always test the current build artifact.
    if "omega_lite_standalone" in sys.modules:
        del sys.modules["omega_lite_standalone"]
    return importlib.import_module("omega_lite_standalone")


SEED = 42
ITERS = 1000


def _prop_payload(league: str, prop_type: str, line: float, mean: float, std: float) -> dict:
    return {
        "player_name": "Test Player",
        "league": league,
        "prop_type": prop_type,
        "line": line,
        "odds_over": -110,
        "odds_under": -110,
        "player_context": {f"{prop_type}_mean": mean, f"{prop_type}_std": std},
        "n_iterations": ITERS,
        "seed": SEED,
    }


# ---------------------------------------------------------------------------
# Per-archetype prop parity
# ---------------------------------------------------------------------------

PROP_CASES = [
    # archetype hint, league, prop_type, line, mean, std
    ("basketball",        "NBA", "pts",                27.5, 28.4, 6.2),
    ("american_football", "NFL", "rec_yds",            78.5, 85.3, 28.7),
    ("baseball",          "MLB", "strikeouts_pitched",  7.5,  8.1,  2.3),
    ("hockey",            "NHL", "shots_on_goal",       3.5,  4.2,  1.6),
    ("soccer",            "EPL", "shots_on_target",     1.5,  2.1,  1.2),
    ("tennis",            "ATP", "total_games",        22.5, 23.4,  3.1),
    ("esports",           "CS2", "kills",              19.5, 21.2,  5.8),
]


@pytest.mark.parametrize("archetype,league,prop_type,line,mean,std", PROP_CASES)
def test_prop_parity_per_archetype(standalone, archetype, league, prop_type, line, mean, std):
    """Standalone single-file ≡ omega_lite package for one prop per archetype."""
    from omega_lite.schemas import PlayerPropRequest as LiteReq
    from omega_lite.service import analyze_player_prop as lite_analyze

    payload = _prop_payload(league, prop_type, line, mean, std)

    lite = lite_analyze(LiteReq(**payload))
    std_out = standalone.analyze(dict(payload))["result"]

    assert lite.status == std_out["status"] == "success", (
        f"{archetype}/{league}: lite={lite.status} std={std_out['status']}"
    )
    assert lite.over_prob == std_out["over_prob"], f"{archetype}/{league} over_prob drift"
    assert lite.under_prob == std_out["under_prob"], f"{archetype}/{league} under_prob drift"
    assert lite.recommendation == std_out["recommendation"], (
        f"{archetype}/{league} recommendation drift"
    )
    assert lite.edge_over == std_out["edge_over"], f"{archetype}/{league} edge_over drift"
    assert lite.edge_under == std_out["edge_under"], f"{archetype}/{league} edge_under drift"


# ---------------------------------------------------------------------------
# Game parity
# ---------------------------------------------------------------------------

def test_standalone_game_parity_nba(standalone):
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

    lite = lite_analyze(LiteReq(**{**payload, "odds": LiteOdds(**odds)}))
    std = standalone.analyze({**payload, "odds": odds})["result"]

    assert lite.status == std["status"] == "success"
    assert lite.simulation.home_win_prob == std["simulation"]["home_win_prob"]
    assert lite.simulation.away_win_prob == std["simulation"]["away_win_prob"]
    assert lite.simulation.predicted_spread == std["simulation"]["predicted_spread"]
    assert lite.simulation.predicted_total == std["simulation"]["predicted_total"]
    assert len(lite.edges) == len(std["edges"])
    for l_edge, s_edge in zip(lite.edges, std["edges"]):
        assert l_edge.edge_pct == s_edge["edge_pct"]
        assert l_edge.ev_pct == s_edge["ev_pct"]
        assert l_edge.confidence_tier == s_edge["confidence_tier"]


# ---------------------------------------------------------------------------
# Quality-gate / self-heal behavior surface
# ---------------------------------------------------------------------------

def test_standalone_emits_sandbox_trace_id(standalone):
    out = standalone.analyze({
        "player_name": "X",
        "league": "NBA",
        "prop_type": "pts",
        "line": 20.0,
        "odds_over": -110,
        "odds_under": -110,
        "player_context": {"pts_mean": 20.0, "pts_std": 5.0},
        "n_iterations": 500,
        "seed": 1,
    })
    assert out["trace_id"].startswith("sandbox-")
    assert out["model_version"] == "omega-lite-v1"
    assert out["kind"] == "prop"
    assert "quality_gate" in out


def test_standalone_skips_with_missing_requirements(standalone):
    """When critical inputs are missing, status=skipped and missing_requirements
    is populated — the self-heal loop in system_prompt.txt §6 keys off this."""
    out = standalone.analyze({
        "player_name": "X",
        "league": "NBA",
        "prop_type": "pts",
        "line": 20.0,
        "player_context": {},
        "n_iterations": 100,
    })
    assert out["result"]["status"] == "skipped"
    assert out["result"]["missing_requirements"]
    assert any("pts_mean" in slot for slot in out["result"]["missing_requirements"])


def test_standalone_prop_stat_keys_cover_all_archetypes(standalone):
    """The sport-coverage table in prompts/system_prompt.txt §5 lists prop_type
    keys per archetype. The standalone exposes get_prop_stat_keys() so an agent
    can confirm support before refusing — kills the NBA-only confusion."""
    coverage = {
        "NBA": "pts",
        "NFL": "rec_yds",
        "MLB": "strikeouts_pitched",
        "NHL": "shots_on_goal",
        "EPL": "shots_on_target",
        "ATP": "aces",
        "PGA": "top_10",
        "UFC": "sig_strikes",
        "CS2": "kills",
    }
    for league, must_have in coverage.items():
        keys = standalone.get_prop_stat_keys(league)
        assert keys, f"{league}: no prop stat keys returned"
        assert must_have in keys, (
            f"{league}: expected {must_have!r} in prop_stat_keys, got {keys!r}"
        )
