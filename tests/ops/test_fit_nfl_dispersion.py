"""Tests for omega-fit-nfl-dispersion (Phase 7 M4).

Verifies the hierarchical-shrinkage classification (small-sample players shrink
toward the position-group posterior; high-sample players keep their own signal),
the persist/read round-trip, and the red-team property the shrinkage exists to
enforce: a small-sample, fat-tailed player's NB tail is pulled toward the group
baseline rather than producing a runaway tail edge.
"""

from __future__ import annotations

import tempfile

import pytest

from omega.core.simulation.backends import PropSimulationInput
from omega.core.simulation.prop_neg_binom import NegBinomPropBackend
from omega.ops.fit_nfl_dispersion import (
    NB_K_SOURCE_GROUP,
    NB_K_SOURCE_PLAYER,
    DispersionObservation,
    _mom_k,
    fit_dispersions,
    run_fit,
    shrink_entity_k,
)
from omega.trace.priors import get_nfl_dispersion
from omega.trace.store import TraceStore

_OVERDISPERSED_30 = [10.0, 30.0] * 15  # n=30, var >> mean
_OVERDISPERSED_200 = [10.0, 30.0] * 100  # n=200


def _tmp_store() -> TraceStore:
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    f.close()
    return TraceStore(db_path=f.name)


def test_small_sample_player_shrinks_to_group():
    fit = shrink_entity_k(_OVERDISPERSED_30, group_k=6.0)
    assert fit.weight == pytest.approx(30 / 80)  # 0.375
    assert fit.weight < 0.6
    assert fit.source == NB_K_SOURCE_GROUP
    # Shrunk value lies between the raw player k and the group k.
    raw = _mom_k(_OVERDISPERSED_30)
    assert min(raw, 6.0) <= fit.k <= max(raw, 6.0)


def test_high_sample_player_keeps_signal():
    fit = shrink_entity_k(_OVERDISPERSED_200, group_k=6.0)
    assert fit.weight == pytest.approx(200 / 250)  # 0.8
    assert fit.weight >= 0.6
    assert fit.source == NB_K_SOURCE_PLAYER


def test_no_player_signal_falls_back_to_group():
    fit = shrink_entity_k([12.0, 12.0, 12.0], group_k=5.0)  # zero variance → no k
    assert fit.k == pytest.approx(5.0)
    assert fit.weight == 0.0
    assert fit.source == NB_K_SOURCE_GROUP


def test_fit_dispersions_classifies_by_sample_size():
    observations = []
    for v in _OVERDISPERSED_30:
        observations.append(DispersionObservation("Rookie WR", "receiving_yards", "WR", v))
    for v in _OVERDISPERSED_200:
        observations.append(DispersionObservation("Veteran WR", "receiving_yards", "WR", v))
    rows = fit_dispersions(observations, season="2025", as_of_date="2026-06-15")
    by_entity = {r.entity: r for r in rows}
    assert by_entity["Rookie WR"].nb_k_source == NB_K_SOURCE_GROUP
    assert by_entity["Rookie WR"].nb_k_shrinkage_weight < 0.6
    assert by_entity["Veteran WR"].nb_k_source == NB_K_SOURCE_PLAYER
    assert by_entity["Veteran WR"].nb_k_shrinkage_weight >= 0.6


def test_run_fit_persists_and_reads_back():
    store = _tmp_store()
    try:
        observations = [
            DispersionObservation("Rookie WR", "receiving_yards", "WR", v)
            for v in _OVERDISPERSED_30
        ] + [
            DispersionObservation("Veteran WR", "receiving_yards", "WR", v)
            for v in _OVERDISPERSED_200
        ]
        rows = run_fit(store, observations, season="2025", as_of_date="2026-06-15")
        assert rows
        loaded = get_nfl_dispersion(store, "Rookie WR", "receiving_yards", season="2025")
        assert loaded is not None
        assert loaded.nb_k_source == NB_K_SOURCE_GROUP
        assert loaded.position_group == "WR"
    finally:
        store.close()


def test_shrinkage_pulls_fat_tail_toward_group_baseline():
    """A 30-sample fat-tailed rookie's longest-reception tail must track the
    group baseline, not the player's noisy raw k (red-team finding 2)."""
    rookie_values = [5.0] * 25 + [120.0] * 5  # n=30, very fat tail → tiny raw k
    group_k = 6.0
    fit = shrink_entity_k(rookie_values, group_k=group_k)
    raw_k = _mom_k(rookie_values)
    assert fit.source == NB_K_SOURCE_GROUP
    assert raw_k is not None and raw_k < 1.0  # raw signal is wildly over-dispersed

    backend = NegBinomPropBackend()

    def _over(k: float) -> float:
        return backend.run(
            PropSimulationInput(
                player_name="Rookie WR",
                league="NFL",
                stat_type="longest_reception",
                line=40.5,
                projection_mean=24.0,
                n_iter=1,
                prior_payload={"nb_dispersion_k": k},
                exact=True,
            )
        )["over_prob"]

    over_shrunk = _over(fit.k)
    over_group = _over(group_k)
    over_raw = _over(raw_k)
    # Shrinkage moves the tail probability toward the group baseline.
    assert abs(over_shrunk - over_group) < abs(over_raw - over_group)
