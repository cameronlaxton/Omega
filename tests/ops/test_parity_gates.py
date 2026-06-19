"""Phase 7: distribution PSI, historical-vs-live parity (3-state), backtest parity."""

from __future__ import annotations

from omega.core.calibration.fitter import CalibrationFitter
from omega.historical.distribution import category_psi, psi
from omega.ops.backtest_parity import evaluate_backtest_parity
from omega.ops.historical_live_parity import evaluate_parity


def _trace(home_prob, context_source="provided", book="dk"):
    return {
        "predictions": {"home_win_prob": home_prob},
        "trace_quality": {"context_source": context_source},
        "odds_snapshot": {"book": book},
    }


# --- distribution helpers ---------------------------------------------------


def test_psi_zero_for_identical_and_large_for_shifted():
    a = [0.5] * 100
    assert psi(a, a) < 1e-6
    low = [0.2] * 100
    high = [0.9] * 100
    assert psi(low, high) > 0.25
    assert category_psi(["x"] * 50, ["x"] * 50) < 1e-6


# --- historical-vs-live parity (3-state) ------------------------------------


def test_parity_pass_when_distributions_match():
    hist = [_trace(0.6) for _ in range(250)]
    live = [_trace(0.6) for _ in range(250)]
    r = evaluate_parity(hist, live, min_live_n=200)
    assert r["state"] == "PASS"
    assert r["promotable_historical_only"] is True


def test_parity_fail_on_shift():
    hist = [_trace(0.9) for _ in range(250)]
    live = [_trace(0.2) for _ in range(250)]
    r = evaluate_parity(hist, live, min_live_n=200)
    assert r["state"] == "FAIL"
    assert r["promotable_historical_only"] is False


def test_parity_inconclusive_on_small_live_n():
    hist = [_trace(0.6) for _ in range(250)]
    live = [_trace(0.6) for _ in range(10)]
    r = evaluate_parity(hist, live, min_live_n=200)
    assert r["state"] == "INCONCLUSIVE"


def test_parity_ignores_advisory_book_mix():
    # Only the (advisory) book differs; gated distributions are identical -> PASS.
    hist = [_trace(0.6, book="dk") for _ in range(250)]
    live = [_trace(0.6, book="fanduel") for _ in range(250)]
    r = evaluate_parity(hist, live, min_live_n=200)
    assert r["state"] == "PASS"
    assert r["advisory"]["book"] > 0.25  # the book mix really did shift


# --- backtest parity (candidate vs incumbent) -------------------------------


def _graded(n=40):
    graded = []
    for i in range(n):
        p = 0.3 + 0.4 * ((i % 5) / 4)
        graded.append({
            "predictions": {"home_win_prob": p},
            "_outcome": {"result": "home_win" if i % 2 == 0 else "away_win"},
        })
    return graded


def test_backtest_parity_refuses_without_incumbent():
    graded = _graded()
    fitter = CalibrationFitter()
    preds, outs = fitter.extract_pairs(graded)
    cand = fitter.fit_isotonic(preds, outs, league="TEST", market="game")
    r = evaluate_backtest_parity(graded, cand, None, plane="game")
    assert r["n_eval"] == len(preds) > 0
    assert r["recommend_promotion"] is False
    assert "no_incumbent_baseline" in r["reasons"]
    assert r["candidate"]["brier_score"] is not None


def test_backtest_parity_refuses_when_no_brier_improvement():
    graded = _graded()
    fitter = CalibrationFitter()
    preds, outs = fitter.extract_pairs(graded)
    prof = fitter.fit_isotonic(preds, outs, league="TEST", market="game")
    # Same profile as candidate and incumbent -> no Brier improvement -> refuse.
    r = evaluate_backtest_parity(graded, prof, prof, plane="game")
    assert r["recommend_promotion"] is False
    assert "brier_not_improved" in r["reasons"]
