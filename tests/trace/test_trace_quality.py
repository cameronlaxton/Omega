"""Graded trace quality (Issue 3 + remediation proofs).

Pins the honesty contract: zero evidence is not harmless, aggregate_quality is
never None, and a failed QA verdict zeroes every learning weight.
"""

from __future__ import annotations

import itertools

from omega.trace.quality import aggregate_quality


def _q(**over):
    base = dict(
        evidence_status="present",
        evidence_count=2,
        context_source="provided",
        baseline_used=False,
        identity_status="complete",
        calibration_eligible=True,
        calibration_path="profile",
        qa_verdict=None,
        imputed_fraction=None,
        evidence_mode="score_only",
    )
    base.update(over)
    return aggregate_quality(**base)


class TestZeroEvidenceEmptyContext:
    def test_zero_evidence_empty_context_floors_and_blocks(self):
        q = _q(
            evidence_status="empty",
            evidence_count=0,
            context_source=None,
            baseline_used=False,
            calibration_eligible=False,
            calibration_path="static_identity",
        )
        assert q.aggregate_quality <= 20
        assert q.probability_calibration_weight == 0
        assert q.evidence_learning_weight == 0
        assert q.confidence_cap == "Pass"
        assert "zero_evidence_empty_context" in q.quality_reasons

    def test_baseline_context_no_evidence_is_blocked(self):
        q = _q(
            evidence_status="empty",
            evidence_count=0,
            context_source="baseline_default_context",
            baseline_used=True,
            calibration_eligible=False,
            calibration_path="static_identity",
        )
        assert q.confidence_cap == "Pass"
        assert "zero_evidence_empty_context" in q.quality_reasons


class TestEmptyEvidenceProvidedContext:
    def test_empty_evidence_provided_caps_c_and_zeroes_learning(self):
        q = _q(evidence_status="empty", evidence_count=0)
        assert q.evidence_learning_weight == 0
        assert q.confidence_cap == "C"
        # Probability calibration is evidence-agnostic — still positive.
        assert q.probability_calibration_weight > 0
        assert "empty_evidence_provided_context" in q.quality_reasons


class TestStaticIdentityCap:
    def test_static_identity_caps_b_when_usable(self):
        q = _q(calibration_path="static_identity")  # aggregate ~85 (strong)
        assert q.confidence_cap == "B"
        assert "static_identity_calibration" in q.quality_reasons

    def test_static_identity_caps_c_when_weak(self):
        q = _q(
            calibration_path="static_identity",
            identity_status="missing",
            evidence_status="empty",
            evidence_count=0,
            context_source="provided",
            imputed_fraction=0.3,
        )
        # Weak aggregate -> static_identity caps at C (or lower via other rules).
        assert q.confidence_cap in ("C", "Pass")


class TestQaFailure:
    def test_failed_qa_zeroes_all_weights(self):
        q = _q(qa_verdict="fail")
        assert q.trace_weight == 0
        assert q.probability_calibration_weight == 0
        assert q.evidence_learning_weight == 0
        assert q.quality_band == "invalid"
        assert q.confidence_cap == "Pass"


class TestLearningModes:
    def test_observe_mode_zeroes_evidence_learning(self):
        q = _q(evidence_mode="observe")
        assert q.evidence_learning_weight == 0

    def test_score_only_keeps_evidence_learning(self):
        q = _q(evidence_mode="score_only")
        assert q.evidence_learning_weight > 0


class TestAggregateNeverNone:
    def test_aggregate_quality_is_never_none(self):
        statuses = ["present", "empty", "recovered_predecision", None]
        contexts = ["provided", "baseline_default_context", None, "legacy_missing_context_source"]
        identities = ["complete", "missing", None]
        paths = ["profile", "base_profile_fallback", "static_calibrated", "static_identity", None]
        qas = [None, "pass", "fail"]
        counts = [0, 1, 5]
        for status, ctx, ident, path, qa, n in itertools.product(
            statuses, contexts, identities, paths, qas, counts
        ):
            q = aggregate_quality(
                evidence_status=status,
                evidence_count=n,
                context_source=ctx,
                baseline_used=ctx == "baseline_default_context",
                identity_status=ident,
                calibration_eligible=ctx == "provided" and ident == "complete",
                calibration_path=path,
                qa_verdict=qa,
                imputed_fraction=None,
                evidence_mode="score_only",
            )
            assert q.aggregate_quality is not None
            assert isinstance(q.aggregate_quality, int)
            assert 0 <= q.aggregate_quality <= 100
            assert q.quality_band in ("strong", "usable", "weak", "invalid")


class TestEndToEndNoActionable:
    def test_uncalibrated_evidence_free_game_cannot_produce_a(self):
        # Full envelope: provided context but NO evidence and NO calibration
        # profile (static_identity). 1000 iterations must NOT manufacture an A,
        # and no edge may surface as a headline best_bet.
        from omega.core.contracts.service import analyze

        out = analyze(
            {
                "kind": "game",
                "league": "NBA",
                "home_team": "Lakers",
                "away_team": "Celtics",
                "game_date": "2026-01-15",
                "n_iterations": 1000,
                "simulation_seed": 7,
                "game_context": {"is_playoff": False, "rest_days": 2},
                "home_context": {"off_rating": 118, "def_rating": 108, "pace": 100},
                "away_context": {"off_rating": 110, "def_rating": 112, "pace": 99},
                "odds": {"moneyline_home": -130, "moneyline_away": 110},
            },
            session_id="ze-test",
            bankroll=1000.0,
        )
        tq = out["trace_quality"]
        assert isinstance(tq["aggregate_quality"], int)
        # empty evidence + provided context -> cap C; static_identity present.
        assert tq["confidence_cap"] in ("C", "Pass")
        assert tq["calibration_path"] == "static_identity"
        tiers = {e["confidence_tier"] for e in out["result"]["edges"]}
        assert "A" not in tiers
        assert out["result"]["best_bet"] is None
