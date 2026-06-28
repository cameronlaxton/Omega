"""
Tests for the Phase C auto-fit loop:
- reliability_weight damping in the evidence handler evaluator
- fit_adjustment_policy: signal_performance -> CANDIDATE AdjustmentPolicy
- promote_adjustment_policy: gated promotion + --go-live mode flip
"""

from __future__ import annotations

import shutil
import tempfile
import warnings
from pathlib import Path

import pytest

import omega.ops.fit_adjustment_policy as fit_mod
import omega.ops.promote_adjustment_policy as promote_mod
from omega.core.calibration.adjustment_policy import (
    AdjustmentPolicy,
    AdjustmentPolicyRegistry,
)
from omega.core.calibration.profiles import ProfileStatus
from omega.core.contracts.evidence import EvidenceSignal
from omega.core.simulation.evidence_handlers import compute_player_adjustment
from omega.strategy.signal_performance import SignalPerformanceRow
from omega.trace.store import TraceStore

_SEED_POLICY = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "omega"
    / "core"
    / "calibration"
    / "adjustment_policies.json"
)


def _tmp_path(suffix: str) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.close()
    return tmp.name


def _tmp_policy_registry() -> str:
    """A temp adjustment_policies.json seeded from the real production seed."""
    path = _tmp_path(".json")
    shutil.copy(_SEED_POLICY, path)
    return path


def _usage_signal() -> EvidenceSignal:
    return EvidenceSignal(
        signal_type="usage_spike",
        category="player_form",
        plane="player",
        value=0.20,
        source="agent_reasoning",
        confidence=0.7,
        window="matchup",
    )


# A whitelisted feature-combo proposal spec (predicate grammar) + a matching signal,
# used to prove a graduated proposal applies at full weight end-to-end after promotion.
_PRED = {
    "kind": "predicate",
    "when": {
        "op": "AND",
        "terms": [
            {"feature": "usage", "op": ">", "value": 0.30},
            {"feature": "teammate_injured", "op": "==", "value": True},
        ],
    },
    "true_factor": 1.06,
}


def _proposal_signal() -> EvidenceSignal:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # unknown signal_type warns by design
        return EvidenceSignal(
            signal_type="usage_when_star_out",
            category="player_form",
            plane="player",
            value={"usage": 0.35, "teammate_injured": True},
            source="llm",
            confidence=0.8,
            window="matchup",
            direction="over",
        )


class TestReliabilityWeightDamping:
    def _policy(self, reliability: float) -> AdjustmentPolicy:
        return AdjustmentPolicy(
            policy_id="p",
            version=1,
            coefficients={
                "usage_spike": {"scale": 1.0, "cap": 0.5, "reliability_weight": reliability}
            },
        )

    def test_weight_zero_damps_to_identity(self):
        adj = compute_player_adjustment(
            player_context={"pts_mean": 25.0},
            evidence=[_usage_signal()],
            league="NBA",
            prop_type="pts",
            policy=self._policy(0.0),
            evidence_mode="live",
        )
        # reliability 0 -> factor fully damped to 1.0
        assert adj.records[0].factor == 1.0
        assert adj.mean_factor == 1.0

    def test_weight_one_is_full_strength(self):
        adj = compute_player_adjustment(
            player_context={"pts_mean": 25.0},
            evidence=[_usage_signal()],
            league="NBA",
            prop_type="pts",
            policy=self._policy(1.0),
            evidence_mode="live",
        )
        assert abs(adj.records[0].factor - 1.2) < 1e-9

    def test_weight_half_is_halfway(self):
        adj = compute_player_adjustment(
            player_context={"pts_mean": 25.0},
            evidence=[_usage_signal()],
            league="NBA",
            prop_type="pts",
            policy=self._policy(0.5),
            evidence_mode="live",
        )
        # raw factor 1.2; half-damped -> 1.0 + 0.5*(0.2) = 1.10
        assert abs(adj.records[0].factor - 1.10) < 1e-9


class TestReliabilityRule:
    """The reliability weight SHAPE (issue #22) is now ``_clamp_weight``."""

    def test_coin_flip_maps_to_zero(self):
        assert fit_mod._clamp_weight(0.50) == 0.0

    def test_below_random_clamps_to_zero(self):
        assert fit_mod._clamp_weight(0.30) == 0.0

    def test_perfect_maps_to_one(self):
        assert fit_mod._clamp_weight(1.0) == 1.0

    def test_midpoint(self):
        assert abs(fit_mod._clamp_weight(0.75) - 0.5) < 1e-9

    def test_aggregate_collapses_by_signal_type(self):
        rows = [
            {"signal_type": "recent_form", "sample_size": 20, "direction_correct": 14},
            {"signal_type": "recent_form", "sample_size": 10, "direction_correct": 4},
        ]
        agg = fit_mod._aggregate_by_signal_type(rows)["recent_form"]
        assert agg.dir_n == 30
        assert abs(agg.direction_accuracy - 18 / 30) < 1e-9
        assert agg.clv_n == 0  # no CLV in these rows


class TestClvPrimaryReliabilityWeight:
    """Issue #28: CLV-primary blend with graceful direction fallback."""

    def test_no_clv_degrades_to_direction(self):
        agg = fit_mod._SignalAgg(dir_n=40, dir_correct=30)  # acc 0.75
        w = fit_mod._reliability_weight(agg, graduated=False, min_samples=30)
        assert abs(w - 0.5) < 1e-9  # clamp(2*(0.75-0.5))

    def test_thin_everything_returns_none(self):
        agg = fit_mod._SignalAgg(dir_n=5, dir_correct=4)  # below min_samples, no CLV
        assert fit_mod._reliability_weight(agg, graduated=False, min_samples=30) is None

    def test_graduated_is_clv_primary(self):
        # clv_aligned 0.70; thin direction -> pure CLV weight clamp(2*(0.7-0.5))=0.4
        agg = fit_mod._SignalAgg(dir_n=5, dir_correct=3, clv_n=4000, clv_aligned_weighted=0.70 * 4000)
        w = fit_mod._reliability_weight(agg, graduated=True, min_samples=30)
        assert abs(w - 0.4) < 1e-9

    def test_graduated_blends_direction_at_large_n(self):
        # clv_aligned 0.70 -> 0.4; dir acc 0.60 -> 0.2; blend 0.75*0.4 + 0.25*0.2 = 0.35
        agg = fit_mod._SignalAgg(
            dir_n=40, dir_correct=24, clv_n=4000, clv_aligned_weighted=0.70 * 4000
        )
        w = fit_mod._reliability_weight(agg, graduated=True, min_samples=30)
        assert abs(w - 0.35) < 1e-9

    def test_ungraduated_with_clv_is_capped(self):
        # Aligned but not graduated -> fail-closed cap at the unproven ceiling.
        agg = fit_mod._SignalAgg(dir_n=50, dir_correct=30, clv_n=50, clv_aligned_weighted=0.70 * 50)
        w = fit_mod._reliability_weight(agg, graduated=False, min_samples=30)
        assert abs(w - fit_mod._UNPROVEN_CEILING) < 1e-9

    def test_ungraduated_misaligned_damps_to_zero(self):
        # The recent_form case: lots of data, CLV not aligned -> 0.
        agg = fit_mod._SignalAgg(dir_n=80, dir_correct=40, clv_n=200, clv_aligned_weighted=0.42 * 200)
        w = fit_mod._reliability_weight(agg, graduated=False, min_samples=30)
        assert w == 0.0


class TestProbationPipeline:
    """End-to-end statistical bar: aggregate -> probation stats -> graduation."""

    def test_strong_clv_graduates_dead_signal_does_not(self):
        n_min = fit_mod.min_samples_for_power()  # ~3863
        rows = [
            {
                "signal_type": "recent_form_residual",
                "sample_size": 100,
                "direction_correct": 55,
                "clv_sample": n_min + 200,  # clears N_min
                "clv_aligned": 0.65,
                "clv_cents_when_followed": 3.0,
                "clv_cents_std": 10.0,
            },
            {
                "signal_type": "recent_form",  # the dead public-stat signal
                "sample_size": 100,
                "direction_correct": 42,
                "clv_sample": 200,  # below N_min and misaligned
                "clv_aligned": 0.42,
                "clv_cents_when_followed": -1.0,
                "clv_cents_std": 8.0,
            },
        ]
        aggs = fit_mod._aggregate_by_signal_type(rows)
        stats = fit_mod._probation_stats(aggs, n_min)
        grad = fit_mod.graduation_mask(stats)
        assert grad.get("recent_form_residual") is True
        assert grad.get("recent_form", False) is False


def _seed_signal_performance(db_path: str) -> None:
    store = TraceStore(db_path=db_path)
    rows = [
        # noise signal â€” accuracy 0.50 -> reliability 0
        SignalPerformanceRow(
            "last_game_outlier",
            "agent_reasoning",
            "last_3",
            "NBA",
            40,
            20,
            0.50,
            0.70,
            0.50,
            0.20,
            0.30,
        ),
        # predictive signal â€” accuracy 0.75 -> reliability 0.5
        SignalPerformanceRow(
            "opponent_stat_rank",
            "boxscore_derived",
            "season",
            "NBA",
            40,
            30,
            0.75,
            0.70,
            0.75,
            -0.05,
            0.18,
        ),
    ]
    store.upsert_signal_performance(rows, dataset_hash="testhash123456")
    store.close()


class TestFitAndPromoteLifecycle:
    def test_fit_creates_candidate_with_reliability_weights(self):
        db = _tmp_path(".db")
        _seed_signal_performance(db)
        policy_path = _tmp_policy_registry()

        rc = fit_mod.main(["--db", db, "--policy-path", policy_path, "--min-samples", "30"])
        assert rc == 0

        registry = AdjustmentPolicyRegistry(path=policy_path)
        candidates = registry.list_policies(status=ProfileStatus.CANDIDATE.value)
        assert len(candidates) == 1
        cand = candidates[0]
        # The seed registry now ships adj_v1_seed (v1, archived) + adj_v2_seed
        # (v2, production), so _next_version yields 3 for the first fitted candidate.
        assert cand.version == 3
        # noise signal damped to 0, predictive signal to 0.5
        assert cand.coefficients["last_game_outlier"]["reliability_weight"] == 0.0
        assert cand.coefficients["opponent_stat_rank"]["reliability_weight"] == 0.5

    def test_fit_dry_run_writes_nothing(self):
        db = _tmp_path(".db")
        _seed_signal_performance(db)
        policy_path = _tmp_policy_registry()
        rc = fit_mod.main(
            ["--db", db, "--policy-path", policy_path, "--min-samples", "30", "--dry-run"]
        )
        assert rc == 0
        registry = AdjustmentPolicyRegistry(path=policy_path)
        assert registry.list_policies(status=ProfileStatus.CANDIDATE.value) == []

    def test_promote_gated_then_go_live(self):
        db = _tmp_path(".db")
        _seed_signal_performance(db)
        policy_path = _tmp_policy_registry()
        fit_mod.main(["--db", db, "--policy-path", policy_path, "--min-samples", "30"])
        registry = AdjustmentPolicyRegistry(path=policy_path)
        cand_id = registry.list_policies(status=ProfileStatus.CANDIDATE.value)[0].policy_id

        # Auto-promote without backtest confirmation must fail the gate.
        rc = promote_mod.main(
            [
                "--candidate-id",
                cand_id,
                "--policy-path",
                policy_path,
                "--auto",
                "--min-samples",
                "1",
            ]
        )
        assert rc == 1  # BACKTEST_IMPROVES gate fails

        # With confirmation + go-live it promotes and flips to live.
        rc = promote_mod.main(
            [
                "--candidate-id",
                cand_id,
                "--policy-path",
                policy_path,
                "--auto",
                "--min-samples",
                "1",
                "--confirm-backtest",
                "--go-live",
            ]
        )
        assert rc == 0
        prod = AdjustmentPolicyRegistry(path=policy_path).get_production_policy()
        assert prod.policy_id == cand_id
        assert prod.mode == "live"

    def test_fit_fails_without_signal_performance(self):
        db = _tmp_path(".db")
        TraceStore(db_path=db).close()  # empty DB, no signal_performance rows
        policy_path = _tmp_policy_registry()
        rc = fit_mod.main(["--db", db, "--policy-path", policy_path])
        assert rc == 1


class TestLifecycleRecommendation:
    """Issue #28 WS3: the fit's advisory transition logic (fail-closed)."""

    def _agg(self, **kw):
        return fit_mod._SignalAgg(**kw)

    def test_probation_graduates_to_active(self):
        agg = self._agg(clv_n=5000, clv_aligned_weighted=0.65 * 5000)
        rec = fit_mod._lifecycle_recommendation(
            agg, current="probation", graduated=True, n_min=3863, deprecate_floor=100
        )
        assert rec == "active"

    def test_probation_rejected_when_budget_spent(self):
        agg = self._agg(clv_n=4000, clv_aligned_weighted=0.50 * 4000)
        rec = fit_mod._lifecycle_recommendation(
            agg, current="probation", graduated=False, n_min=3863, deprecate_floor=100
        )
        assert rec == "rejected"

    def test_probation_none_while_gathering(self):
        agg = self._agg(clv_n=50, clv_aligned_weighted=0.55 * 50)
        rec = fit_mod._lifecycle_recommendation(
            agg, current="probation", graduated=False, n_min=3863, deprecate_floor=100
        )
        assert rec is None

    def test_active_deprecated_when_clv_misaligned(self):
        agg = self._agg(clv_n=200, clv_aligned_weighted=0.45 * 200)
        rec = fit_mod._lifecycle_recommendation(
            agg, current="active", graduated=False, n_min=3863, deprecate_floor=100
        )
        assert rec == "deprecated"

    def test_active_kept_when_clv_aligned(self):
        agg = self._agg(clv_n=200, clv_aligned_weighted=0.60 * 200)
        rec = fit_mod._lifecycle_recommendation(
            agg, current="active", graduated=False, n_min=3863, deprecate_floor=100
        )
        assert rec is None

    def test_active_deprecated_direction_fallback(self):
        # Thin CLV; clearly sub-coin-flip realized direction over a big sample.
        agg = self._agg(dir_n=300, dir_correct=120, clv_n=0)  # acc 0.40
        rec = fit_mod._lifecycle_recommendation(
            agg, current="active", graduated=False, n_min=3863, deprecate_floor=100
        )
        assert rec == "deprecated"

    def test_terminal_states_yield_no_recommendation(self):
        agg = self._agg(clv_n=5000, clv_aligned_weighted=0.65 * 5000)
        assert (
            fit_mod._lifecycle_recommendation(
                agg, current="deprecated", graduated=True, n_min=3863, deprecate_floor=100
            )
            is None
        )


class TestSetSignalLifecycle:
    def test_set_and_read_back(self):
        path = _tmp_policy_registry()
        reg = AdjustmentPolicyRegistry(path=path)
        base = reg.get_production_policy()
        reg.set_signal_lifecycle(base.policy_id, {"recent_form": "deprecated"})
        assert reg.get_policy(base.policy_id).signal_lifecycle["recent_form"] == "deprecated"

    def test_invalid_lifecycle_value_rejected(self):
        path = _tmp_policy_registry()
        reg = AdjustmentPolicyRegistry(path=path)
        base = reg.get_production_policy()
        with pytest.raises(ValueError):
            reg.set_signal_lifecycle(base.policy_id, {"recent_form": "bogus"})


class TestApplyLifecycleRecommendations:
    def _register_candidate(self, path: str, recs: dict[str, str]) -> str:
        reg = AdjustmentPolicyRegistry(path=path)
        base = reg.get_production_policy()
        cand = AdjustmentPolicy(
            policy_id="adj_test_lifecycle",
            version=base.version + 1,
            status=ProfileStatus.CANDIDATE,
            mode="shadow",
            coefficients=dict(base.coefficients),
            sample_size=10,
            lifecycle_recommendations=recs,
            incumbent_id=base.policy_id,
        )
        reg.register(cand)
        return cand.policy_id

    def test_flag_binds_recommendations_into_overrides(self):
        path = _tmp_policy_registry()
        cid = self._register_candidate(
            path, {"recent_form": "deprecated", "usage_spike": "probation"}
        )
        rc = promote_mod.main(
            [
                "--candidate-id", cid, "--policy-path", path, "--auto",
                "--min-samples", "1", "--confirm-backtest",
                "--apply-lifecycle-recommendations",
            ]
        )
        assert rc == 0
        prod = AdjustmentPolicyRegistry(path=path).get_production_policy()
        assert prod.policy_id == cid
        assert prod.signal_lifecycle["recent_form"] == "deprecated"
        assert prod.signal_lifecycle["usage_spike"] == "probation"

    def test_without_flag_recommendations_stay_advisory(self):
        path = _tmp_policy_registry()
        cid = self._register_candidate(path, {"recent_form": "deprecated"})
        rc = promote_mod.main(
            [
                "--candidate-id", cid, "--policy-path", path, "--auto",
                "--min-samples", "1", "--confirm-backtest",
            ]
        )
        assert rc == 0
        prod = AdjustmentPolicyRegistry(path=path).get_production_policy()
        assert prod.signal_lifecycle == {}  # not bound without the explicit flag

    def test_active_proposal_binds_feature_combo_from_trace_store(self):
        path = _tmp_policy_registry()
        db = _tmp_path(".db")
        store = TraceStore(db_path=db)
        store.upsert_signal_proposal(
            name="usage_when_star_out",
            feature_combo={"op": "and", "items": [{"feature": "usage_rate", "gt": 0.3}]},
            plane="player",
        )
        store.close()
        cid = self._register_candidate(path, {"usage_when_star_out": "active"})

        rc = promote_mod.main(
            [
                "--candidate-id", cid, "--policy-path", path, "--db", db, "--auto",
                "--min-samples", "1", "--confirm-backtest",
                "--apply-lifecycle-recommendations",
            ]
        )

        assert rc == 0
        prod = AdjustmentPolicyRegistry(path=path).get_production_policy()
        assert prod.signal_lifecycle["usage_when_star_out"] == "active"
        coeff = prod.coefficients["usage_when_star_out"]
        assert coeff["feature_combo"]["op"] == "and"
        # It cleared the bar but carries no scored clv_aligned on the proposal row,
        # so it is trusted in full (1.0) rather than the unfitted-prior sliver (0.25).
        assert coeff["reliability_weight"] == 1.0

    def test_active_proposal_binds_measured_reliability_weight(self):
        # A scored proposal (clv_aligned on record) binds its MEASURED weight,
        # mirroring the fit: _clamp_weight(0.70) = 2*(0.70-0.5) = 0.40.
        path = _tmp_policy_registry()
        db = _tmp_path(".db")
        store = TraceStore(db_path=db)
        store.upsert_signal_proposal(
            name="usage_when_star_out",
            feature_combo={"op": "and", "items": [{"feature": "usage_rate", "gt": 0.3}]},
            plane="player",
        )
        store.conn.execute(
            "UPDATE signal_proposals SET clv_aligned = 0.70, graduates = 1 WHERE name = ?",
            ("usage_when_star_out",),
        )
        store.conn.commit()
        store.close()
        cid = self._register_candidate(path, {"usage_when_star_out": "active"})

        rc = promote_mod.main(
            [
                "--candidate-id", cid, "--policy-path", path, "--db", db, "--auto",
                "--min-samples", "1", "--confirm-backtest",
                "--apply-lifecycle-recommendations",
            ]
        )

        assert rc == 0
        prod = AdjustmentPolicyRegistry(path=path).get_production_policy()
        assert prod.coefficients["usage_when_star_out"]["reliability_weight"] == 0.4

    def test_graduated_proposal_does_not_apply_at_unfitted_sliver(self):
        # End-to-end: after promotion the graduated proposal moves a live prediction
        # at full weight, NOT the 0.25 unfitted prior. _PRED's true_factor is 1.06;
        # the seed prior would yield only 1 + 0.25*0.06 = 1.015 (the sliver).
        path = _tmp_policy_registry()
        db = _tmp_path(".db")
        store = TraceStore(db_path=db)
        store.upsert_signal_proposal(
            name="usage_when_star_out", feature_combo=_PRED, plane="player"
        )
        store.close()
        cid = self._register_candidate(path, {"usage_when_star_out": "active"})
        rc = promote_mod.main(
            [
                "--candidate-id", cid, "--policy-path", path, "--db", db, "--auto",
                "--min-samples", "1", "--confirm-backtest", "--go-live",
                "--apply-lifecycle-recommendations",
            ]
        )
        assert rc == 0
        prod = AdjustmentPolicyRegistry(path=path).get_production_policy()
        adj = compute_player_adjustment(
            player_context={"pts_mean": 25.0, "pts_std": 6.0},
            evidence=[_proposal_signal()],
            league="NBA",
            prop_type="pts",
            policy=prod,
            evidence_mode="bounded_live",
        )
        assert adj.records[0].applied is True
        assert adj.mean_factor == pytest.approx(1.06)
        assert adj.mean_factor != pytest.approx(1.015)


class TestReliabilityWeightFromClvAlignment:
    """The shared derivation promote uses to stamp a graduated proposal's weight."""

    def test_none_when_no_scored_clv(self):
        assert fit_mod.reliability_weight_from_clv_alignment(None, graduated=True) is None

    def test_graduated_is_clv_primary_clamp(self):
        # 2*(0.70-0.5) = 0.40 — the measured magnitude, well above the 0.25 prior.
        assert fit_mod.reliability_weight_from_clv_alignment(0.70, graduated=True) == 0.4

    def test_ungraduated_capped_at_unproven_ceiling(self):
        w = fit_mod.reliability_weight_from_clv_alignment(0.95, graduated=False)
        assert w == fit_mod._UNPROVEN_CEILING

    def test_misaligned_damps_to_zero(self):
        assert fit_mod.reliability_weight_from_clv_alignment(0.40, graduated=True) == 0.0
