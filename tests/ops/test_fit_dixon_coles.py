"""Tests for omega-fit-dixon-coles (Phase 7 M2).

Covers synthetic-rho recovery, the promote flow, the frozen-production refit
refusal, and the minimum-matches gate.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from omega.core.simulation.engine import _dixon_coles_scores
from omega.ops.fit_dixon_coles import (
    FrozenProductionFitError,
    fit_rho,
    run_fit,
)
from omega.trace.priors import get_production_dc_profile
from omega.trace.store import TraceStore


def _store() -> TraceStore:
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    f.close()
    return TraceStore(db_path=f.name)


def _synthetic_pairs(rho: float, n: int = 30_000, seed: int = 7) -> list[tuple[int, int]]:
    rng = np.random.default_rng(seed)
    home, away = _dixon_coles_scores(1.45, 1.15, rho, n, rng=rng)
    return list(zip(home, away))


def test_fit_recovers_negative_rho():
    fit = fit_rho(_synthetic_pairs(-0.12))
    assert fit.rho == pytest.approx(-0.12, abs=0.04)
    assert fit.n_matches == 30_000


def test_fit_recovers_zero_rho():
    fit = fit_rho(_synthetic_pairs(0.0))
    assert fit.rho == pytest.approx(0.0, abs=0.04)


def test_fit_loss_prefers_true_rho_direction():
    """On draw-heavy data the fitted rho must come out negative."""
    fit = fit_rho(_synthetic_pairs(-0.2))
    assert fit.rho < -0.05


def test_min_matches_gate():
    with pytest.raises(ValueError, match="need at least"):
        fit_rho(_synthetic_pairs(-0.1, n=50))


def test_run_fit_writes_candidate_then_promotes():
    store = _store()
    try:
        pairs = _synthetic_pairs(-0.12, n=5_000)
        run_fit(
            store,
            "fifa_intl_v1",
            pairs,
            as_of_date="2026-06-10",
            min_matches=1_000,
        )
        # Candidate only — production lookup still fails closed.
        assert get_production_dc_profile(store, "fifa_intl_v1") is None

        run_fit(
            store,
            "fifa_intl_v1",
            pairs,
            as_of_date="2026-06-11",
            promote=True,
            min_matches=1_000,
        )
        prod = get_production_dc_profile(store, "fifa_intl_v1")
        assert prod is not None
        assert prod.as_of_date == "2026-06-11"
        assert prod.source == "statsbomb_open_data"
    finally:
        store.close()


def test_refit_over_frozen_production_row_is_refused():
    store = _store()
    try:
        pairs = _synthetic_pairs(-0.12, n=5_000)
        run_fit(
            store,
            "fifa_intl_v1",
            pairs,
            as_of_date="2026-06-10",
            promote=True,
            min_matches=1_000,
        )
        with pytest.raises(FrozenProductionFitError, match="frozen production row"):
            run_fit(
                store,
                "fifa_intl_v1",
                pairs,
                as_of_date="2026-06-10",
                min_matches=1_000,
            )
        # A new as_of is the sanctioned path; the frozen row stays production
        # until the new fit is explicitly promoted.
        run_fit(store, "fifa_intl_v1", pairs, as_of_date="2026-09-01", min_matches=1_000)
        prod = get_production_dc_profile(store, "fifa_intl_v1")
        assert prod is not None and prod.as_of_date == "2026-06-10"
    finally:
        store.close()
