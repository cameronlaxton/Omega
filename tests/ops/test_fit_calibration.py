from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_SCRIPTS = _REPO_ROOT / "src" / "omega" / "ops"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import fit_calibration  # type: ignore  # noqa: E402

from omega.core.calibration.fitter import CalibrationFitter  # noqa: E402


def test_extract_plane_pairs_routes_game_and_prop_separately():
    graded = [
        {
            "predictions": {"home_win_prob": 0.62, "over_prob": 0.57},
            "_outcome": {"result": "home_win"},
            "_prop_outcomes": [{"side": "over", "result": "win"}],
        }
    ]
    fitter = CalibrationFitter()

    game_preds, game_outcomes, game_label = fit_calibration._extract_plane_pairs(
        fitter, graded, "game"
    )
    prop_preds, prop_outcomes, prop_label = fit_calibration._extract_plane_pairs(
        fitter, graded, "prop"
    )

    assert game_preds == [0.62]
    assert game_outcomes == [1]
    assert game_label == "home_win_prob/outcome"
    assert prop_preds == [0.57]
    assert prop_outcomes == [1]
    assert prop_label == "prop probability/outcome"


def test_extract_plane_pairs_routes_draw():
    graded = [
        {"predictions": {"draw_prob": 0.28}, "_outcome": {"result": "draw"}},
        {"predictions": {"draw_prob": 0.22}, "_outcome": {"result": "home_win"}},
    ]
    fitter = CalibrationFitter()

    preds, outcomes, label = fit_calibration._extract_plane_pairs(fitter, graded, "draw")

    assert preds == [0.28, 0.22]
    assert outcomes == [1, 0]
    assert label == "draw_prob/outcome"


def test_plane_market_mapping():
    # Each plane maps to its own market so a profile is only applied to the
    # plane it was fit on (prop no longer collapses onto the game market).
    assert fit_calibration._plane_market("draw") == "draw"
    assert fit_calibration._plane_market("game") == "game"
    assert fit_calibration._plane_market("prop") == "prop"


def test_fit_and_register_prop_candidate_carries_prop_market(tmp_path):
    # A prop-plane fit must produce a market="prop" candidate (and a prop-tagged
    # id) so it never competes with / overwrites the game-market profile.
    from omega.core.calibration.registry import CalibrationRegistry

    fitter = CalibrationFitter()
    registry = CalibrationRegistry(path=str(tmp_path / "profiles.json"))
    # >= _MIN_SAMPLES (30) train pairs where higher prob correlates with a hit.
    train_p = [round(0.2 + (i % 7) * 0.1, 2) for i in range(40)]
    train_o = [1 if p >= 0.5 else 0 for p in train_p]
    hold_p = [0.35, 0.6, 0.5, 0.7, 0.45]
    hold_o = [0, 1, 1, 1, 0]

    profile = fit_calibration.fit_and_register(
        fitter, registry, "MLB", "isotonic",
        train_p, train_o, hold_p, hold_o, dry_run=False, market="prop",
    )

    assert profile.market == "prop"
    assert "prop_" in profile.profile_id
    # It registered under the prop market, leaving the game slot untouched.
    assert registry.get_production("MLB", market="game") is None

