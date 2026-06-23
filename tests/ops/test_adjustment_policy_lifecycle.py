"""
Tests for the Phase C auto-fit loop:
- reliability_weight damping in the evidence handler evaluator
- fit_adjustment_policy: signal_performance -> CANDIDATE AdjustmentPolicy
- promote_adjustment_policy: gated promotion + --go-live mode flip
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

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
    def test_coin_flip_maps_to_zero(self):
        assert fit_mod._reliability_weight(0.50) == 0.0

    def test_below_random_clamps_to_zero(self):
        assert fit_mod._reliability_weight(0.30) == 0.0

    def test_perfect_maps_to_one(self):
        assert fit_mod._reliability_weight(1.0) == 1.0

    def test_midpoint(self):
        assert abs(fit_mod._reliability_weight(0.75) - 0.5) < 1e-9

    def test_aggregate_collapses_by_signal_type(self):
        rows = [
            {"signal_type": "recent_form", "sample_size": 20, "direction_correct": 14},
            {"signal_type": "recent_form", "sample_size": 10, "direction_correct": 4},
        ]
        agg = fit_mod._aggregate_by_signal_type(rows)
        n, acc = agg["recent_form"]
        assert n == 30
        assert abs(acc - 18 / 30) < 1e-9


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
        assert cand.version == 2
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
