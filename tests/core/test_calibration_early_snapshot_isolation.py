"""Milestone 1 — calibration ignores early-market traces by default.

Red-team finding 4: a synthetic dataset with intentionally inflated/divergent
early-line traces must NOT drift the production calibration fit. Opting in
(extract_pairs_with_early_slice) quarantines those traces into a dedicated
context_slice instead of the production base slice.

References:
  omega/core/calibration/fitter.py (EARLY_MARKET_SLICE, include_early_snapshots)
  docs/phase7/MULTI_SPORT_EXPANSION.md (Milestone 1 acceptance; red-team finding 4)
"""

from __future__ import annotations

from omega.core.calibration.fitter import EARLY_MARKET_SLICE, CalibrationFitter


def _game_trace(home_win_prob: float, result: str, early: bool = False) -> dict:
    trace = {
        "predictions": {"home_win_prob": home_win_prob},
        "_outcome": {"result": result},
    }
    if early:
        trace["liquidity_profile"] = EARLY_MARKET_SLICE
    return trace


def _clean_dataset(n: int = 60) -> list[dict]:
    # Well-behaved: high prob -> mostly home wins, low prob -> mostly away wins.
    traces = []
    for i in range(n):
        if i % 2 == 0:
            traces.append(_game_trace(80.0, "home_win"))
        else:
            traces.append(_game_trace(30.0, "away_win"))
    return traces


def _inflated_early(n: int = 60) -> list[dict]:
    # Phantom edge: confidently wrong early-line traces that would wreck a fit.
    return [_game_trace(90.0, "away_win", early=True) for _ in range(n)]


def test_default_extract_excludes_early_traces():
    clean = _clean_dataset()
    mixed = clean + _inflated_early()

    preds_clean, out_clean = CalibrationFitter.extract_pairs(clean)
    preds_mixed, out_mixed = CalibrationFitter.extract_pairs(mixed)

    # The inflated early traces are dropped -> pairs identical to the clean set.
    assert preds_mixed == preds_clean
    assert out_mixed == out_clean


def test_fit_does_not_drift_with_early_traces():
    fitter = CalibrationFitter()
    clean = _clean_dataset()
    mixed = clean + _inflated_early()

    p_clean, o_clean = CalibrationFitter.extract_pairs(clean)
    p_mixed, o_mixed = CalibrationFitter.extract_pairs(mixed)

    prof_clean = fitter.fit_shrinkage(p_clean, o_clean, "WNBA", eligible_sample_size=len(p_clean))
    prof_mixed = fitter.fit_shrinkage(p_mixed, o_mixed, "WNBA", eligible_sample_size=len(p_mixed))

    # Same production profile despite 60 phantom early traces in the input.
    assert prof_mixed.params == prof_clean.params
    assert prof_mixed.dataset_hash == prof_clean.dataset_hash


def test_opt_in_routes_early_traces_to_dedicated_slice():
    mixed = _clean_dataset() + _inflated_early()
    slices = CalibrationFitter.extract_pairs_with_early_slice(mixed)

    assert None in slices  # production base slice
    assert EARLY_MARKET_SLICE in slices

    base_preds, _ = slices[None]
    early_preds, _ = slices[EARLY_MARKET_SLICE]

    # Base slice has only the clean traces; early slice holds the quarantined ones.
    assert len(base_preds) == 60
    assert len(early_preds) == 60
    assert all(p == 0.9 for p in early_preds)
