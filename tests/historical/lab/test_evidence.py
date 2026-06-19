"""Promotion decision matrix + single fail-closed gate (auto-promote)."""

from __future__ import annotations

from omega.core.calibration.profiles import CalibrationProfile, ProfileStatus
from omega.core.calibration.registry import CalibrationRegistry
from omega.historical.lab.evidence import (
    EvidenceContext,
    _verdict,
    evaluate_decision,
    resolve,
)
from omega.historical.lab.schemas import WinnersCurse

_BT_PASS = {"recommend_promotion": True, "reasons": ["no_incumbent_baseline"], "n_eval": 100}
_BT_FAIL = {"recommend_promotion": False, "reasons": ["brier_not_improved"], "n_eval": 100}
_CLV_PASS = {"verdict": "PASS", "avg_clv": 0.01, "n_bets": 12}
_CLV_INCONCLUSIVE = {"verdict": "INCONCLUSIVE", "n_bets": 0}
_LIVE_PASS = {"state": "PASS", "live_n": 300}
_LIVE_INCONCLUSIVE = {"state": "INCONCLUSIVE", "live_n": 0}
_LIVE_FAIL = {"state": "FAIL", "live_n": 300}


def _profile(*, ece: float = 0.02, brier: float = 0.20, n: int = 200) -> CalibrationProfile:
    return CalibrationProfile(
        profile_id="iso_seed_v1",
        version=1,
        method="isotonic",
        league="FIFA_INTL",
        market="draw",
        training_window="2023-01..2023-05",
        sample_size=n,
        dataset_hash="seedhash",
        metrics={
            "brier_score": brier,
            "calibration_error": ece,
            "log_loss": 0.5,
            "cv_calibration_error": ece,
            "cv_n_folds": 20,
            "n_eval": 50,
        },
    )


def _ctx(**over) -> EvidenceContext:
    base = dict(
        lab_run_id="lab_001",
        league="FIFA_INTL",
        plane="draw",
        market="draw",
        winner_profile=_profile(),
        winners_curse=WinnersCurse(n_variants=4, val_to_holdout_ece_delta=0.005, risk="low"),
        holdout_sealed=True,
        attempted_variant_count=4,
        working_tree_dirty=False,
        armed=False,
        backtest_parity=_BT_PASS,
        clv_walk_forward=_CLV_PASS,
        live_parity=_LIVE_PASS,
    )
    base.update(over)
    return EvidenceContext(**base)


# --- pure decision matrix -------------------------------------------------


def test_decision_evidence_ready_when_all_green():
    assert evaluate_decision(_ctx()) == ("evidence_ready", True)


def test_decision_not_recommended_without_winner():
    assert evaluate_decision(_ctx(winner_profile=None)) == ("not_recommended", False)


def test_decision_blocked_when_holdout_unsealed():
    assert evaluate_decision(_ctx(holdout_sealed=False)) == ("blocked", False)


def test_decision_blocked_on_backtest_parity_fail():
    assert evaluate_decision(_ctx(backtest_parity=_BT_FAIL)) == ("blocked", False)


def test_decision_blocked_on_clv_not_pass():
    assert evaluate_decision(_ctx(clv_walk_forward=_CLV_INCONCLUSIVE)) == ("blocked", False)


def test_decision_blocked_on_live_fail():
    assert evaluate_decision(_ctx(live_parity=_LIVE_FAIL)) == ("blocked", False)


def test_decision_shadow_only_on_inconclusive_live():
    assert evaluate_decision(_ctx(live_parity=_LIVE_INCONCLUSIVE)) == ("shadow_only", False)


def test_decision_blocked_on_dirty_tree_even_if_green():
    assert evaluate_decision(_ctx(working_tree_dirty=True)) == ("blocked", False)


# --- resolve() through the single gate ------------------------------------


def test_resolve_not_armed_never_writes_registry(tmp_path):
    reg = CalibrationRegistry(path=str(tmp_path / "profiles.json"))
    bundle = resolve(reg, _ctx(armed=False))
    assert bundle.decision == "evidence_ready"
    assert bundle.candidate_id is None
    assert reg.list_profiles() == []  # no side effects without arming


def test_resolve_armed_green_promotes_through_gate(tmp_path):
    reg = CalibrationRegistry(path=str(tmp_path / "profiles.json"))
    bundle = resolve(reg, _ctx(armed=True))
    assert bundle.decision == "promoted"
    assert bundle.candidate_id is not None
    promoted = reg.get_profile(bundle.candidate_id)
    assert promoted is not None and promoted.status == ProfileStatus.PRODUCTION
    assert bundle.gate_report is not None and bundle.gate_report["passed"] is True


def test_resolve_armed_but_gate_rejects_is_blocked(tmp_path):
    # ECE above the floor → the single gate fails closed; outcome is blocked.
    reg = CalibrationRegistry(path=str(tmp_path / "profiles.json"))
    bundle = resolve(reg, _ctx(armed=True, winner_profile=_profile(ece=0.20)))
    assert bundle.decision == "blocked"
    # Candidate stays registered (inspectable) but never reaches production.
    cand = reg.get_profile(bundle.candidate_id)
    assert cand is not None and cand.status == ProfileStatus.CANDIDATE
    assert bundle.gate_report is not None and bundle.gate_report["passed"] is False


def test_resolve_armed_inconclusive_live_stays_shadow_only(tmp_path):
    reg = CalibrationRegistry(path=str(tmp_path / "profiles.json"))
    bundle = resolve(reg, _ctx(armed=True, live_parity=_LIVE_INCONCLUSIVE))
    assert bundle.decision == "shadow_only"
    assert reg.list_profiles() == []  # never attempted promotion


# --- verdict display preserves three states -------------------------------


def test_verdict_preserves_inconclusive_clv():
    # An INCONCLUSIVE CLV artifact must display as INCONCLUSIVE, not collapse to FAIL.
    assert _verdict(_CLV_INCONCLUSIVE) == "INCONCLUSIVE"
    assert _verdict(_CLV_PASS) == "PASS"
    assert _verdict({"verdict": "FAIL", "n_bets": 3}) == "FAIL"


def test_verdict_backtest_no_incumbent_and_live():
    assert _verdict(_BT_PASS) == "no_incumbent"  # recommend + no_incumbent_baseline reason
    assert _verdict(_BT_FAIL) == "FAIL"
    assert _verdict(_LIVE_INCONCLUSIVE) == "INCONCLUSIVE"
    assert _verdict(None) is None


def test_inconclusive_clv_still_blocks_promotion():
    # Display fix must not change gating: INCONCLUSIVE CLV is not a pass.
    assert evaluate_decision(_ctx(clv_walk_forward=_CLV_INCONCLUSIVE)) == ("blocked", False)
