"""Variant grid: fits the sweep, evaluates on validation, never touches the registry."""

from __future__ import annotations

from omega.historical.lab.grid import profile_grid_hash, run_grid
from omega.historical.lab.schemas import Window

TRAIN = Window(start="2023-01-01", end="2023-03-31")
VALID = Window(start="2023-04-01", end="2023-05-31")
HOLD = Window(start="2023-06-01", end="2023-07-31")

_PROBS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.35]


def _trace(date: str, prob: float, home_win: bool, labels=None) -> dict:
    return {
        "decision_time": f"{date}T18:00:00+00:00",
        "predictions": {"home_win_prob": prob},
        "_outcome": {"result": "home_win" if home_win else "away_win"},
        "context_labels": labels or {},
    }


def _graded(n: int, months: list[int], off: int = 0, labels=None) -> list[dict]:
    out = []
    for i in range(n):
        p = _PROBS[(i + off) % len(_PROBS)]
        home_win = ((i * 13 + off * 5 + 3) % 100) < int(p * 100)
        m = months[i % len(months)]
        d = 1 + (i % 26)
        out.append(_trace(f"2023-{m:02d}-{d:02d}", p, home_win, labels))
    return out


def _run(graded, **kw):
    return run_grid(
        graded,
        lab_run_id="lab_001",
        league="FIFA_INTL",
        plane="game",
        train_window=TRAIN,
        validation_window=VALID,
        holdout_window=HOLD,
        **kw,
    )


def test_grid_fits_both_methods_on_base():
    ledger = _run(_graded(90, [1, 2, 3]) + _graded(36, [4, 5], off=1))
    base = [v for v in ledger.variants if v.context_slice is None]
    families = {v.profile_family for v in base}
    assert families == {"isotonic", "shrinkage"}
    for v in base:
        assert v.status == "rejected"  # provisional; seal selects later
        assert v.brier is not None and v.ece is not None
        assert v.n_validation == 36
        assert v.sample_size == 90
        assert not v.touched_holdout  # grid never reads holdout


def test_grid_marks_no_variant_selected():
    ledger = _run(_graded(90, [1, 2, 3]) + _graded(36, [4, 5], off=1))
    assert ledger.selected is None
    assert ledger.holdout_access_count == 0


def test_grid_cv_ece_populated_with_enough_data():
    ledger = _run(_graded(90, [1, 2, 3]) + _graded(36, [4, 5], off=1))
    iso = next(v for v in ledger.variants if v.profile_family == "isotonic")
    assert iso.cv_ece is not None  # 90 train pairs, both classes → CV engages


def test_grid_skips_slice_with_no_traces():
    ledger = _run(_graded(90, [1, 2, 3]) + _graded(36, [4, 5], off=1), slices=["playoff"])
    playoff = [v for v in ledger.variants if v.context_slice == "playoff"]
    assert playoff and all(v.status == "skipped" for v in playoff)


def test_grid_hash_is_stable_and_input_sensitive():
    h1 = profile_grid_hash("FIFA_INTL", "game", ["isotonic"], [], TRAIN, VALID, HOLD)
    h2 = profile_grid_hash("FIFA_INTL", "game", ["isotonic"], [], TRAIN, VALID, HOLD)
    h3 = profile_grid_hash("FIFA_INTL", "game", ["isotonic", "shrinkage"], [], TRAIN, VALID, HOLD)
    assert h1 == h2 != h3


def test_grid_does_not_write_production_registry():
    from omega.core.calibration.registry import _DEFAULT_PATH

    before = _DEFAULT_PATH.read_bytes() if _DEFAULT_PATH.exists() else None
    ledger = _run(_graded(90, [1, 2, 3]) + _graded(36, [4, 5], off=1))
    after = _DEFAULT_PATH.read_bytes() if _DEFAULT_PATH.exists() else None
    assert before == after  # grid must never register candidates
    assert ledger.profile_grid_hash  # sanity: it did run
