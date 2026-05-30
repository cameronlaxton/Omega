from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_SCRIPTS = _REPO_ROOT / "scripts"
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
    assert fit_calibration._plane_market("draw") == "draw"
    assert fit_calibration._plane_market("game") == "game"
    assert fit_calibration._plane_market("prop") == "game"
