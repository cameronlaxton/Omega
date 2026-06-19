"""Holdout sealing: single winner, holdout touched exactly once, winner's-curse."""

from __future__ import annotations

import pytest

from omega.historical.lab.grid import run_grid
from omega.historical.lab.schemas import AttemptedVariant, AttemptedVariantLedger, Window
from omega.historical.lab.seal import seal_winner, select_winner

TRAIN = Window(start="2023-01-01", end="2023-03-31")
VALID = Window(start="2023-04-01", end="2023-05-31")
HOLD = Window(start="2023-06-01", end="2023-07-31")

_PROBS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.35]


def _trace(date: str, prob: float, home_win: bool) -> dict:
    return {
        "decision_time": f"{date}T18:00:00+00:00",
        "predictions": {"home_win_prob": prob},
        "_outcome": {"result": "home_win" if home_win else "away_win"},
        "context_labels": {},
    }


def _graded(n: int, months: list[int], off: int = 0) -> list[dict]:
    out = []
    for i in range(n):
        p = _PROBS[(i + off) % len(_PROBS)]
        home_win = ((i * 13 + off * 5 + 3) % 100) < int(p * 100)
        m = months[i % len(months)]
        d = 1 + (i % 26)
        out.append(_trace(f"2023-{m:02d}-{d:02d}", p, home_win))
    return out


def _full_graded() -> list[dict]:
    return _graded(90, [1, 2, 3]) + _graded(36, [4, 5], off=1) + _graded(40, [6, 7], off=2)


def _grid(graded):
    return run_grid(
        graded,
        lab_run_id="lab_001",
        league="FIFA_INTL",
        plane="game",
        train_window=TRAIN,
        validation_window=VALID,
        holdout_window=HOLD,
    )


def _seal(ledger, graded):
    return seal_winner(
        ledger,
        graded,
        league="FIFA_INTL",
        plane="game",
        train_window=TRAIN,
        validation_window=VALID,
        holdout_window=HOLD,
    )


def test_seal_selects_one_winner_and_touches_holdout_once():
    graded = _full_graded()
    result = _seal(_grid(graded), graded)
    assert result.winner is not None
    assert result.holdout_sealed is True
    assert result.holdout_access_count == 1
    selected = [v for v in result.ledger.variants if v.status == "selected"]
    assert len(selected) == 1
    assert selected[0].touched_holdout
    # No other variant carries holdout metrics.
    assert all(not v.touched_holdout for v in result.ledger.variants if v.status != "selected")


def test_seal_produces_promotion_ready_profile():
    graded = _full_graded()
    result = _seal(_grid(graded), graded)
    prof = result.winner_profile
    assert prof is not None
    assert prof.method == result.winner.profile_family
    assert "calibration_error" in prof.metrics
    assert "cv_calibration_error" in prof.metrics  # CV engages at this sample size


def test_seal_winners_curse_counts_all_attempts():
    graded = _full_graded()
    ledger = _grid(graded)
    result = _seal(ledger, graded)
    assert result.winners_curse.n_variants == len(ledger.variants)
    assert result.winners_curse.risk in {"low", "elevated", "high"}
    assert result.winners_curse.val_to_holdout_ece_delta is not None


def test_select_winner_prefers_lowest_calibration_error():
    def _v(vid, ece, cv_ece, brier):
        return AttemptedVariant(
            variant_id=vid,
            profile_family="isotonic",
            train_window=TRAIN,
            validation_window=VALID,
            holdout_window=HOLD,
            ece=ece,
            cv_ece=cv_ece,
            brier=brier,
            status="rejected",
        )

    ledger = AttemptedVariantLedger(
        lab_run_id="lab_001",
        variants=[_v("a", 0.04, 0.05, 0.22), _v("b", 0.02, 0.03, 0.23)],
    )
    assert select_winner(ledger).variant_id == "b"


def test_seal_no_eligible_variants_returns_no_winner():
    # Too few traces → every variant skipped.
    graded = _graded(10, [1, 2]) + _graded(8, [4]) + _graded(8, [6])
    result = _seal(_grid(graded), graded)
    assert result.winner is None
    assert result.winner_profile is None
    assert result.holdout_sealed is False
    assert result.holdout_access_count == 0


def test_seal_rejects_overlapping_holdout():
    ledger = AttemptedVariantLedger(lab_run_id="lab_001", variants=[])
    with pytest.raises(ValueError, match="overlaps"):
        seal_winner(
            ledger,
            [],
            league="FIFA_INTL",
            plane="game",
            train_window=TRAIN,
            validation_window=VALID,
            holdout_window=Window(start="2023-05-15", end="2023-07-31"),
        )
