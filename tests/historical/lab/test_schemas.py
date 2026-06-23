"""Lab artifact schemas: validators, seal invariants, and cross-artifact consistency."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omega.historical.lab.schemas import (
    AttemptedVariant,
    AttemptedVariantLedger,
    HistoricalLabRun,
    Window,
    assert_consistent,
    windows_overlap,
)
from omega.paths import default_trace_db_path

TRAIN = Window(start="2023-01-01", end="2023-03-31")
VALID = Window(start="2023-04-01", end="2023-05-31")
HOLD = Window(start="2023-06-01", end="2023-07-31")


def _lab_run(tmp_path, **overrides):
    base = dict(
        lab_run_id="lab_001",
        dataset_manifest_id="m1",
        league="FIFA_INTL",
        plane="draw",
        replay_id="replay_m1",
        replay_db_path=str(tmp_path / "replay.db"),
        train_window=TRAIN,
        validation_window=VALID,
        holdout_window=HOLD,
    )
    base.update(overrides)
    return HistoricalLabRun(**base)


def _variant(status="rejected", *, vid="v1", holdout=False, slice_=None):
    kw = dict(
        variant_id=vid,
        profile_family="isotonic",
        plane="draw",
        context_slice=slice_,
        train_window=TRAIN,
        validation_window=VALID,
        holdout_window=HOLD,
        sample_size=200,
        n_validation=60,
        brier=0.21,
        ece=0.03,
        status=status,
    )
    if holdout:
        kw["holdout_brier"] = 0.22
        kw["holdout_ece"] = 0.035
        kw["n_holdout"] = 50
    return AttemptedVariant(**kw)


# --- Window ---------------------------------------------------------------


def test_window_rejects_reversed_range():
    with pytest.raises(ValidationError):
        Window(start="2023-05-01", end="2023-04-01")


def test_windows_overlap_adjacent_is_not_overlap():
    assert not windows_overlap(TRAIN, VALID)
    assert windows_overlap(Window(start="2023-01-01", end="2023-04-15"), VALID)


# --- HistoricalLabRun -----------------------------------------------------


def test_lab_run_market_auto_derived_from_plane(tmp_path):
    run = _lab_run(tmp_path)  # plane=draw, market omitted
    assert run.market == "draw"


def test_lab_run_market_mismatch_rejected(tmp_path):
    with pytest.raises(ValidationError):
        _lab_run(tmp_path, plane="game", market="draw")


def test_lab_run_rejects_production_db(tmp_path):
    with pytest.raises(ValidationError):
        _lab_run(tmp_path, replay_db_path=str(default_trace_db_path()))


def test_lab_run_requires_chronological_windows(tmp_path):
    with pytest.raises(ValidationError):
        _lab_run(
            tmp_path,
            validation_window=Window(start="2023-03-01", end="2023-05-31"),
        )


def test_lab_run_sealed_holdout_access_bound(tmp_path):
    with pytest.raises(ValidationError):
        _lab_run(tmp_path, holdout_sealed=True, holdout_access_count=2)


def test_lab_run_roundtrip(tmp_path):
    run = _lab_run(tmp_path, git_commit="abc", attempted_variant_count=3)
    restored = HistoricalLabRun.model_validate(run.model_dump(mode="json"))
    assert restored == run


def test_lab_run_rejects_unknown_field(tmp_path):
    with pytest.raises(ValidationError):
        _lab_run(tmp_path, bogus=1)


# --- Ledger seal invariants ----------------------------------------------


def test_ledger_at_most_one_selected():
    with pytest.raises(ValidationError):
        AttemptedVariantLedger(
            lab_run_id="lab_001",
            variants=[_variant("selected", vid="a"), _variant("selected", vid="b")],
        )


def test_ledger_rejects_holdout_metrics_on_non_winner():
    with pytest.raises(ValidationError):
        AttemptedVariantLedger(
            lab_run_id="lab_001",
            variants=[_variant("rejected", vid="a", holdout=True)],
        )


def test_ledger_holdout_access_count_and_selected():
    ledger = AttemptedVariantLedger(
        lab_run_id="lab_001",
        variants=[
            _variant("rejected", vid="a"),
            _variant("selected", vid="b", holdout=True),
        ],
    )
    assert ledger.holdout_access_count == 1
    assert ledger.selected is not None and ledger.selected.variant_id == "b"


# --- Cross-artifact consistency ------------------------------------------


def test_assert_consistent_passes(tmp_path):
    run = _lab_run(
        tmp_path, attempted_variant_count=2, holdout_access_count=1, profile_grid_hash="g"
    )
    ledger = AttemptedVariantLedger(
        lab_run_id="lab_001",
        profile_grid_hash="g",
        variants=[_variant("rejected", vid="a"), _variant("selected", vid="b", holdout=True)],
    )
    assert_consistent(run, ledger)  # no raise


def test_assert_consistent_count_mismatch(tmp_path):
    run = _lab_run(tmp_path, attempted_variant_count=5, profile_grid_hash="g")
    ledger = AttemptedVariantLedger(
        lab_run_id="lab_001", profile_grid_hash="g", variants=[_variant()]
    )
    with pytest.raises(ValueError, match="attempted_variant_count"):
        assert_consistent(run, ledger)


def test_assert_consistent_grid_hash_mismatch(tmp_path):
    run = _lab_run(tmp_path, attempted_variant_count=1, profile_grid_hash="g")
    ledger = AttemptedVariantLedger(
        lab_run_id="lab_001", profile_grid_hash="OTHER", variants=[_variant()]
    )
    with pytest.raises(ValueError, match="profile_grid_hash"):
        assert_consistent(run, ledger)
