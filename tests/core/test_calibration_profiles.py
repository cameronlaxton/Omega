"""
Tests for Phase 6c: Calibration profiles, registry, fitter, and selection.

All tests are deterministic — no network calls, no LLM.
"""

import os
import tempfile

import pytest

# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


def _make_profile(**overrides):
    """Create a CalibrationProfile with sensible defaults."""
    from omega.core.calibration.profiles import CalibrationProfile

    defaults = {
        "profile_id": "test_nba_v1",
        "version": 1,
        "method": "shrinkage",
        "league": "NBA",
        "params": {"shrink_factor": 0.6},
        "training_window": "2025-01-01/2025-06-30",
        "sample_size": 200,
        "dataset_hash": "abc123def456",
        "metrics": {"brier_score": 0.22, "calibration_error": 0.04, "log_loss": 0.65},
    }
    defaults.update(overrides)
    return CalibrationProfile(**defaults)


def _make_graded_traces(n=50, bias=0.0):
    """Synthetic graded traces matching TraceStore.get_graded_traces() shape.

    Creates traces where home_win_prob correlates with actual outcomes,
    with optional bias to create overconfidence.
    """
    import random

    random.seed(42)
    traces = []
    for i in range(n):
        raw_prob = random.uniform(0.3, 0.8) + bias
        raw_prob = max(0.1, min(0.95, raw_prob))
        # Outcome correlates with probability
        actual_home_win = random.random() < (raw_prob * 0.9)  # slightly less than predicted
        traces.append(
            {
                "trace_id": f"trace-{i:04d}",
                "league": "NBA",
                "predictions": {"home_win_prob": raw_prob * 100},  # stored as percentage
                "_outcome": {
                    "outcome_id": f"out-{i:04d}",
                    "home_score": 110 if actual_home_win else 95,
                    "away_score": 95 if actual_home_win else 110,
                    "result": "home_win" if actual_home_win else "away_win",
                },
            }
        )
    return traces


# -----------------------------------------------------------------------
# CalibrationProfile model tests
# -----------------------------------------------------------------------


class TestCalibrationProfile:
    def test_create_valid(self):
        profile = _make_profile()
        assert profile.profile_id == "test_nba_v1"
        assert profile.method == "shrinkage"
        assert profile.league == "NBA"
        assert profile.schema_version == 1

    def test_round_trip(self):
        from omega.core.calibration.profiles import CalibrationProfile

        profile = _make_profile(method="isotonic", params={"calibration_map": {0.5: 0.48}})
        dumped = profile.model_dump()
        restored = CalibrationProfile(**dumped)
        assert restored == profile

    def test_json_round_trip(self):
        from omega.core.calibration.profiles import CalibrationProfile

        profile = _make_profile()
        json_str = profile.model_dump_json()
        restored = CalibrationProfile.model_validate_json(json_str)
        assert restored == profile

    def test_default_status_is_candidate(self):
        from omega.core.calibration.profiles import ProfileStatus

        profile = _make_profile()
        assert profile.status == ProfileStatus.CANDIDATE


# -----------------------------------------------------------------------
# CalibrationRegistry tests
# -----------------------------------------------------------------------


class TestCalibrationRegistry:
    def test_register_and_list(self):
        from omega.core.calibration.registry import CalibrationRegistry

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            os.unlink(path)  # start with no file
            reg = CalibrationRegistry(path=path)
            profile = _make_profile()
            reg.register(profile)
            listed = reg.list_profiles()
            assert len(listed) == 1
            assert listed[0].profile_id == "test_nba_v1"
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_get_production_none(self):
        from omega.core.calibration.registry import CalibrationRegistry

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            os.unlink(path)
            reg = CalibrationRegistry(path=path)
            profile = _make_profile()
            reg.register(profile)
            assert reg.get_production("NBA") is None  # still candidate
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_promote_workflow(self):
        from omega.core.calibration.profiles import ProfileStatus
        from omega.core.calibration.registry import CalibrationRegistry

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            os.unlink(path)
            reg = CalibrationRegistry(path=path)

            p1 = _make_profile(profile_id="nba_v1")
            p2 = _make_profile(profile_id="nba_v2", version=2)
            reg.register(p1)
            reg.register(p2)

            # Promote p1
            reg._apply_promotion("nba_v1")
            prod = reg.get_production("NBA")
            assert prod is not None
            assert prod.profile_id == "nba_v1"

            # Promote p2 — p1 should be archived
            reg._apply_promotion("nba_v2")
            prod = reg.get_production("NBA")
            assert prod.profile_id == "nba_v2"

            archived = reg.get_profile("nba_v1")
            assert archived.status == ProfileStatus.ARCHIVED
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_game_and_prop_production_coexist(self):
        """A prop profile and a game profile can both be PRODUCTION for one
        league at once -- promoting one market must not archive the other."""
        from omega.core.calibration.profiles import ProfileStatus
        from omega.core.calibration.registry import CalibrationRegistry

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            os.unlink(path)
            reg = CalibrationRegistry(path=path)
            reg.register(_make_profile(profile_id="nba_game_v1", market="game"))
            reg.register(_make_profile(profile_id="nba_prop_v1", market="prop"))
            reg._apply_promotion("nba_game_v1")
            reg._apply_promotion("nba_prop_v1")

            assert reg.get_production("NBA", market="game").profile_id == "nba_game_v1"
            assert reg.get_production("NBA", market="prop").profile_id == "nba_prop_v1"
            assert reg.get_profile("nba_game_v1").status == ProfileStatus.PRODUCTION
            assert reg.get_profile("nba_prop_v1").status == ProfileStatus.PRODUCTION
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_prop_lookup_falls_back_to_game_when_no_prop_profile(self):
        """With no prop profile, a prop lookup falls back to the game profile
        (then static) -- this is the safe transitional state."""
        from omega.core.calibration.registry import CalibrationRegistry

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            os.unlink(path)
            reg = CalibrationRegistry(path=path)
            reg.register(_make_profile(profile_id="nba_game_v1", market="game"))
            reg._apply_promotion("nba_game_v1")
            resolved = reg.get_production("NBA", market="prop")
            assert resolved is not None and resolved.profile_id == "nba_game_v1"
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_reject(self):
        from omega.core.calibration.profiles import ProfileStatus
        from omega.core.calibration.registry import CalibrationRegistry

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            os.unlink(path)
            reg = CalibrationRegistry(path=path)
            reg.register(_make_profile())
            reg.reject("test_nba_v1", "Brier score degraded")
            p = reg.get_profile("test_nba_v1")
            assert p.status == ProfileStatus.REJECTED
            assert p.reject_reason == "Brier score degraded"
            assert reg.get_production("NBA") is None
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_empty_file(self):
        from omega.core.calibration.registry import CalibrationRegistry

        reg = CalibrationRegistry(path="/tmp/nonexistent_profiles_12345.json")
        assert reg.get_production("NBA") is None
        assert reg.list_profiles() == []

    def test_duplicate_id_raises(self):
        from omega.core.calibration.registry import CalibrationRegistry

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            os.unlink(path)
            reg = CalibrationRegistry(path=path)
            reg.register(_make_profile())
            with pytest.raises(ValueError, match="already exists"):
                reg.register(_make_profile())
        finally:
            if os.path.exists(path):
                os.unlink(path)


# -----------------------------------------------------------------------
# CalibrationFitter tests
# -----------------------------------------------------------------------


class TestCalibrationFitter:
    def test_extract_pairs(self):
        from omega.core.calibration.fitter import CalibrationFitter

        traces = _make_graded_traces(n=50)
        fitter = CalibrationFitter()
        preds, outs = fitter.extract_pairs(traces)
        assert len(preds) == 50
        assert len(outs) == 50
        assert all(0.0 <= p <= 1.0 for p in preds)
        assert all(o in (0, 1) for o in outs)

    def test_extract_pairs_skips_incomplete(self):
        from omega.core.calibration.fitter import CalibrationFitter

        traces = [
            {"predictions": None, "_outcome": {"result": "home_win"}},
            {"predictions": {"home_win_prob": 60}, "_outcome": None},
            {"predictions": {"home_win_prob": 65}, "_outcome": {"result": "home_win"}},
        ]
        fitter = CalibrationFitter()
        preds, outs = fitter.extract_pairs(traces)
        assert len(preds) == 1
        assert preds[0] == pytest.approx(0.65, abs=0.01)
        assert outs[0] == 1

    def test_extract_pairs_ignores_prop_only_traces(self):
        """Prop traces have no _outcome — extract_pairs must skip them silently."""
        from omega.core.calibration.fitter import CalibrationFitter

        traces = [
            {
                "predictions": {"over_prob": 0.6, "under_prob": 0.4},
                "_prop_outcomes": [{"side": "over", "result": "win"}],
                # No _outcome — this is a prop-only trace
            },
            {"predictions": {"home_win_prob": 0.65}, "_outcome": {"result": "home_win"}},
        ]
        preds, outs = CalibrationFitter.extract_pairs(traces)
        assert len(preds) == 1
        assert preds[0] == pytest.approx(0.65, abs=0.01)

    def test_extract_prop_pairs_over_win(self):
        from omega.core.calibration.fitter import CalibrationFitter

        traces = [
            {
                "predictions": {"over_prob": 0.62, "under_prob": 0.38},
                "_prop_outcomes": [{"side": "over", "result": "win"}],
            }
        ]
        preds, outs = CalibrationFitter.extract_prop_pairs(traces)
        assert preds == [pytest.approx(0.62, abs=1e-6)]
        assert outs == [1]

    def test_extract_prop_pairs_under_loss(self):
        from omega.core.calibration.fitter import CalibrationFitter

        traces = [
            {
                "predictions": {"over_prob": 0.45, "under_prob": 0.55},
                "_prop_outcomes": [{"side": "under", "result": "loss"}],
            }
        ]
        preds, outs = CalibrationFitter.extract_prop_pairs(traces)
        assert preds == [pytest.approx(0.55, abs=1e-6)]
        assert outs == [0]

    def test_extract_prop_pairs_skips_push(self):
        from omega.core.calibration.fitter import CalibrationFitter

        traces = [
            {
                "predictions": {"over_prob": 0.5, "under_prob": 0.5},
                "_prop_outcomes": [{"side": "over", "result": "push"}],
            }
        ]
        preds, outs = CalibrationFitter.extract_prop_pairs(traces)
        assert preds == []
        assert outs == []

    def test_extract_prop_pairs_skips_void(self):
        from omega.core.calibration.fitter import CalibrationFitter

        # A DNP / no-action void carries no calibration signal and must not be
        # counted (it would otherwise fall through and be scored as a loss).
        traces = [
            {
                "predictions": {"over_prob": 0.5, "under_prob": 0.5},
                "_prop_outcomes": [{"side": "over", "result": "void"}],
            }
        ]
        preds, outs = CalibrationFitter.extract_prop_pairs(traces)
        assert preds == []
        assert outs == []

    def test_extract_prop_pairs_multiple_props_per_trace(self):
        """One trace can have many prop outcomes (e.g. pts, reb, ast)."""
        from omega.core.calibration.fitter import CalibrationFitter

        traces = [
            {
                "predictions": {"over_prob": 0.58, "under_prob": 0.42},
                "_prop_outcomes": [
                    {"side": "over", "result": "win"},
                    {"side": "over", "result": "loss"},
                    {"side": "under", "result": "win"},
                ],
            }
        ]
        preds, outs = CalibrationFitter.extract_prop_pairs(traces)
        assert len(preds) == 3
        assert outs == [1, 0, 1]

    def test_extract_prop_pairs_normalizes_percentage_form(self):
        """If over_prob comes in as 62 (percent) it must be coerced to 0.62."""
        from omega.core.calibration.fitter import CalibrationFitter

        traces = [
            {
                "predictions": {"over_prob": 62, "under_prob": 38},
                "_prop_outcomes": [{"side": "over", "result": "win"}],
            }
        ]
        preds, _ = CalibrationFitter.extract_prop_pairs(traces)
        assert preds[0] == pytest.approx(0.62, abs=1e-6)

    def test_extract_prop_pairs_skips_game_only_traces(self):
        from omega.core.calibration.fitter import CalibrationFitter

        traces = [
            {"predictions": {"home_win_prob": 0.6}, "_outcome": {"result": "home_win"}},
        ]
        preds, outs = CalibrationFitter.extract_prop_pairs(traces)
        assert preds == []
        assert outs == []

    def test_fit_isotonic_reproducible(self):
        from omega.core.calibration.fitter import CalibrationFitter

        traces = _make_graded_traces(n=100)
        fitter = CalibrationFitter()
        preds, outs = fitter.extract_pairs(traces)

        p1 = fitter.fit_isotonic(preds, outs, "NBA")
        p2 = fitter.fit_isotonic(preds, outs, "NBA")

        assert p1.params == p2.params
        assert p1.dataset_hash == p2.dataset_hash

    def test_fit_isotonic_monotonic(self):
        from omega.core.calibration.fitter import CalibrationFitter

        traces = _make_graded_traces(n=100)
        fitter = CalibrationFitter()
        preds, outs = fitter.extract_pairs(traces)
        profile = fitter.fit_isotonic(preds, outs, "NBA")

        cal_map = profile.params["calibration_map"]
        sorted_keys = sorted(cal_map.keys())
        values = [cal_map[k] for k in sorted_keys]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1] + 1e-9, (
                f"Monotonicity violation at index {i}: {values[i]} > {values[i + 1]}"
            )

    def test_fit_shrinkage(self):
        from omega.core.calibration.fitter import CalibrationFitter

        # Create overconfident traces (bias predictions upward)
        traces = _make_graded_traces(n=100, bias=0.1)
        fitter = CalibrationFitter()
        preds, outs = fitter.extract_pairs(traces)
        profile = fitter.fit_shrinkage(preds, outs, "NBA", eligible_sample_size=len(preds))

        assert profile.method == "shrinkage"
        assert 0.3 <= profile.params["shrink_factor"] <= 1.0

    def test_fit_too_few_samples(self):
        from omega.core.calibration.fitter import CalibrationFitter

        fitter = CalibrationFitter()
        with pytest.raises(ValueError, match="at least"):
            fitter.fit_isotonic([0.5] * 10, [1] * 10, "NBA")

    def test_evaluate_metrics(self):
        from omega.core.calibration.fitter import CalibrationFitter

        traces = _make_graded_traces(n=100)
        fitter = CalibrationFitter()
        preds, outs = fitter.extract_pairs(traces)
        profile = fitter.fit_isotonic(preds, outs, "NBA")
        metrics = fitter.evaluate(profile, preds, outs)

        assert "brier_score" in metrics
        assert "calibration_error" in metrics
        assert "log_loss" in metrics
        assert metrics["n_eval"] == 100
        assert 0.0 <= metrics["brier_score"] <= 1.0
        assert 0.0 <= metrics["calibration_error"] <= 1.0
        assert metrics["log_loss"] >= 0.0

    def test_compare_recommend(self):
        from omega.core.calibration.fitter import CalibrationFitter

        traces = _make_graded_traces(n=100)
        fitter = CalibrationFitter()
        preds, outs = fitter.extract_pairs(traces)

        candidate = fitter.fit_isotonic(preds, outs, "NBA")

        # Make a deliberately worse incumbent (extreme shrinkage)
        incumbent = _make_profile(
            profile_id="incumbent",
            method="shrinkage",
            params={"shrink_factor": 0.3},  # Over-shrinks toward 0.5
        )

        result = fitter.compare(candidate, incumbent, preds, outs)
        assert "candidate_metrics" in result
        assert "incumbent_metrics" in result
        assert "recommend_promote" in result
        assert isinstance(result["recommend_promote"], bool)


# -----------------------------------------------------------------------
# apply_calibration backward compatibility + profile selection
# -----------------------------------------------------------------------


class TestApplyCalibration:
    def test_backward_compat_no_league(self):
        """apply_calibration(p) without league must match pre-Phase-6c behavior."""
        from omega.core.calibration.probability import apply_calibration

        # Mild probs pass through (gate check)
        assert apply_calibration(0.65) == 0.65
        assert apply_calibration(0.50) == 0.50

        # Extreme probs get calibrated
        cal_95 = apply_calibration(0.95)
        assert cal_95 < 0.95  # shrunk
        assert cal_95 > 0.50  # still above midpoint

    def test_with_league_no_profile_uses_static(self):
        """apply_calibration(p, league='XYZ') with no profile falls back to static."""
        from omega.core.calibration.probability import apply_calibration

        # No production profile for a made-up league
        static = apply_calibration(0.95)
        with_league = apply_calibration(0.95, league="NONEXISTENT_LEAGUE")
        assert static == with_league

    def test_with_league_and_profile(self):
        """When a production profile exists, apply_calibration uses it."""
        from omega.core.calibration.probability import apply_calibration
        from omega.core.calibration.profiles import CalibrationProfile
        from omega.core.calibration.registry import CalibrationRegistry

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            os.unlink(path)
            reg = CalibrationRegistry(path=path)

            # Register and promote an isotonic profile for NBA
            profile = CalibrationProfile(
                profile_id="test_iso_nba",
                version=1,
                method="isotonic",
                league="NBA",
                params={
                    "calibration_map": {
                        0.0: 0.10,
                        0.2: 0.18,
                        0.4: 0.38,
                        0.5: 0.48,
                        0.6: 0.55,
                        0.8: 0.72,
                        1.0: 0.85,
                    }
                },
                training_window="2025-01-01/2025-06-30",
                sample_size=200,
                dataset_hash="testhash",
                metrics={"brier_score": 0.20},
            )
            reg.register(profile)
            reg._apply_promotion("test_iso_nba")

            # Monkeypatch _get_active_profile to use our temp registry
            import omega.core.calibration.probability as prob_mod

            original = prob_mod._get_active_profile

            def _patched_get(league, context_slice=None, market="game"):
                r = CalibrationRegistry(path=path)
                return r.get_production(league, context_slice=context_slice, market=market)

            prob_mod._get_active_profile = _patched_get
            try:
                # With profile: should use isotonic map (0.65 → interpolated)
                result_nba = apply_calibration(0.65, league="NBA")
                # Without profile: static fallback
                result_static = apply_calibration(0.65)

                # The isotonic profile maps 0.65 differently than static
                # Static: 0.65 passes through gate (returns 0.65)
                # Isotonic: 0.65 interpolates between 0.6→0.55 and 0.8→0.72
                assert result_static == 0.65  # gate check passes through
                assert result_nba != 0.65  # profile changes it
                assert 0.0 < result_nba < 1.0  # sanity
            finally:
                prob_mod._get_active_profile = original
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_parity_service_and_backtest_with_league(self):
        """_calibrate(p, league) must equal apply_calibration(p, league)."""
        from omega.core.calibration.probability import apply_calibration
        from omega.core.contracts.service import _calibrate

        test_probs = [0.05, 0.15, 0.50, 0.65, 0.85, 0.92, 0.99]
        for p in test_probs:
            # Without league (existing parity)
            assert apply_calibration(p) == _calibrate(p), f"Parity fail at {p}"
            # With league (new parity)
            assert apply_calibration(p, league="NBA") == _calibrate(p, league="NBA"), (
                f"League parity fail at {p}"
            )


# ---------------------------------------------------------------------------
# context_slice on CalibrationProfile
# ---------------------------------------------------------------------------


class TestContextSliceProfile:
    def test_base_profile_has_none_slice(self):
        from omega.core.calibration.profiles import CalibrationProfile

        p = CalibrationProfile(
            profile_id="iso_nba_v1",
            version=1,
            method="shrinkage",
            league="NBA",
            params={"shrink_factor": 0.7},
            training_window="2025-01-01/2025-12-31",
            sample_size=100,
            dataset_hash="aaa",
        )
        assert p.context_slice is None

    def test_playoff_profile_stores_slice(self):
        from omega.core.calibration.profiles import CalibrationProfile

        p = CalibrationProfile(
            profile_id="iso_nba_playoff_v1",
            version=1,
            method="shrinkage",
            league="NBA",
            context_slice="playoff",
            params={"shrink_factor": 0.65},
            training_window="2025-01-01/2025-12-31",
            sample_size=50,
            dataset_hash="bbb",
        )
        assert p.context_slice == "playoff"
        # Round-trip
        dumped = p.model_dump()
        p2 = CalibrationProfile(**dumped)
        assert p2.context_slice == "playoff"

    def test_mlb_regular_profile(self):
        from omega.core.calibration.profiles import CalibrationProfile

        p = CalibrationProfile(
            profile_id="iso_mlb_regular_v1",
            version=1,
            method="isotonic",
            league="MLB",
            context_slice="regular",
            params={"calibration_map": {"0.5": 0.5}},
            training_window="2025-01-01/2025-12-31",
            sample_size=80,
            dataset_hash="ccc",
        )
        assert p.context_slice == "regular"
        assert p.league == "MLB"


# ---------------------------------------------------------------------------
# Registry context_slice lookup and promotion
# ---------------------------------------------------------------------------


class TestRegistryContextSlice:
    def test_get_production_returns_base_when_no_slice_profile(self):
        """If no slice-specific profile exists, falls back to base."""
        import os
        import tempfile

        from omega.core.calibration.registry import CalibrationRegistry

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            reg = CalibrationRegistry(path=path)
            base = _make_profile(profile_id="iso_nba_base", version=1, league="NBA")
            reg.register(base)
            reg._apply_promotion("iso_nba_base")

            # Base lookup
            assert reg.get_production("NBA") is not None
            # Slice lookup falls back to base
            result = reg.get_production("NBA", context_slice="playoff")
            assert result is not None
            assert result.profile_id == "iso_nba_base"
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_get_production_returns_slice_when_available(self):
        """Slice-specific profile is preferred over base."""
        import os
        import tempfile

        from omega.core.calibration.registry import CalibrationRegistry

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            reg = CalibrationRegistry(path=path)
            base = _make_profile(profile_id="iso_nba_base", version=1, league="NBA")
            playoff = _make_profile(
                profile_id="iso_nba_playoff_v1",
                version=1,
                league="NBA",
                context_slice="playoff",
            )
            reg.register(base)
            reg._apply_promotion("iso_nba_base")
            reg.register(playoff)
            reg._apply_promotion("iso_nba_playoff_v1")

            # Exact slice match
            result = reg.get_production("NBA", context_slice="playoff")
            assert result is not None
            assert result.profile_id == "iso_nba_playoff_v1"
            # Base still accessible
            base_result = reg.get_production("NBA")
            assert base_result is not None
            assert base_result.profile_id == "iso_nba_base"
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_promote_slice_does_not_archive_base(self):
        """Promoting a playoff profile must not archive the base profile."""
        import os
        import tempfile

        from omega.core.calibration.profiles import ProfileStatus
        from omega.core.calibration.registry import CalibrationRegistry

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            reg = CalibrationRegistry(path=path)
            base = _make_profile(profile_id="iso_nba_base", version=1, league="NBA")
            playoff = _make_profile(
                profile_id="iso_nba_playoff_v1",
                version=1,
                league="NBA",
                context_slice="playoff",
            )
            reg.register(base)
            reg._apply_promotion("iso_nba_base")
            reg.register(playoff)
            reg._apply_promotion("iso_nba_playoff_v1")

            # Base must still be PRODUCTION
            base_result = reg.get_production("NBA")
            assert base_result is not None
            assert base_result.status == ProfileStatus.PRODUCTION

            # Playoff must be PRODUCTION
            playoff_result = reg.get_production("NBA", context_slice="playoff")
            assert playoff_result is not None
            assert playoff_result.status == ProfileStatus.PRODUCTION
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_list_profiles_filters_by_context_slice(self):
        """list_profiles(context_slice='playoff') returns only playoff profiles."""
        import os
        import tempfile

        from omega.core.calibration.registry import CalibrationRegistry

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            reg = CalibrationRegistry(path=path)
            base = _make_profile(profile_id="base_v1", version=1, league="NBA")
            playoff = _make_profile(
                profile_id="playoff_v1", version=1, league="NBA", context_slice="playoff"
            )
            b2b = _make_profile(
                profile_id="b2b_v1", version=1, league="NBA", context_slice="back_to_back"
            )
            for p in [base, playoff, b2b]:
                reg.register(p)

            playoff_profiles = reg.list_profiles(league="NBA", context_slice="playoff")
            assert len(playoff_profiles) == 1
            assert playoff_profiles[0].profile_id == "playoff_v1"

            base_profiles = reg.list_profiles(league="NBA", context_slice=None)
            assert len(base_profiles) == 1
            assert base_profiles[0].profile_id == "base_v1"

            all_profiles = reg.list_profiles(league="NBA")
            assert len(all_profiles) == 3
        finally:
            if os.path.exists(path):
                os.unlink(path)


# ---------------------------------------------------------------------------
# Fitter: extract_pairs_by_context
# ---------------------------------------------------------------------------


class TestExtractPairsByContext:
    def _make_game_trace(self, is_playoff: bool, home_win_prob: float, result: str):
        return {
            "context_labels": {"is_playoff": is_playoff},
            "predictions": {"home_win_prob": home_win_prob},
            "_outcome": {"result": result},
        }

    def test_partitions_traces_by_context(self):
        from omega.core.calibration.fitter import CalibrationFitter

        traces = [
            self._make_game_trace(True, 0.7, "home_win"),
            self._make_game_trace(True, 0.6, "home_win"),
            self._make_game_trace(False, 0.55, "away_win"),
            self._make_game_trace(False, 0.52, "home_win"),
        ]

        def fn(trace):
            return "playoff" if trace.get("context_labels", {}).get("is_playoff") else "regular"

        result = CalibrationFitter.extract_pairs_by_context(traces, fn)

        assert "playoff" in result
        assert "regular" in result
        assert len(result["playoff"][0]) == 2
        assert len(result["regular"][0]) == 2

    def test_empty_partition_returns_empty_lists(self):
        from omega.core.calibration.fitter import CalibrationFitter

        traces = [self._make_game_trace(False, 0.6, "home_win")]

        def fn(trace):
            return "playoff" if trace.get("context_labels", {}).get("is_playoff") else "regular"

        result = CalibrationFitter.extract_pairs_by_context(traces, fn)

        assert "regular" in result
        assert "playoff" not in result

    def test_all_traces_to_same_slice(self):
        from omega.core.calibration.fitter import CalibrationFitter

        traces = [
            self._make_game_trace(True, 0.65, "home_win"),
            self._make_game_trace(True, 0.55, "away_win"),
        ]

        def fn(_trace):
            return None

        result = CalibrationFitter.extract_pairs_by_context(traces, fn)
        assert None in result
        assert len(result[None][0]) == 2


# ---------------------------------------------------------------------------
# _derive_context_slice
# ---------------------------------------------------------------------------


class TestDeriveContextSlice:
    def test_none_hints_returns_none(self):
        from omega.core.calibration.probability import _derive_context_slice

        assert _derive_context_slice(None) is None

    def test_empty_hints_returns_none(self):
        from omega.core.calibration.probability import _derive_context_slice

        assert _derive_context_slice({}) is None

    def test_playoff_true_returns_playoff(self):
        from omega.core.calibration.probability import _derive_context_slice

        assert _derive_context_slice({"is_playoff": True}) == "playoff"

    def test_playoff_false_returns_none(self):
        from omega.core.calibration.probability import _derive_context_slice

        assert _derive_context_slice({"is_playoff": False}) is None

    def test_b2b_returns_back_to_back(self):
        from omega.core.calibration.probability import _derive_context_slice

        assert _derive_context_slice({"rest_days": 0}) == "back_to_back"

    def test_playoff_takes_precedence_over_b2b(self):
        from omega.core.calibration.probability import _derive_context_slice

        result = _derive_context_slice({"is_playoff": True, "rest_days": 0})
        assert result == "playoff"  # playoff wins


class TestDrawMarketSelection:
    """Gap 4: market-aware profile selection for the 3-way draw plane."""

    def _registry_with(self, *profiles):
        from omega.core.calibration.registry import CalibrationRegistry

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        os.unlink(path)
        reg = CalibrationRegistry(path=path)
        for p in profiles:
            reg.register(p)
            reg._apply_promotion(p.profile_id)
        return reg, path

    def test_draw_profile_selected_for_draw_market(self):
        from omega.core.calibration.profiles import ProfileStatus

        game = _make_profile(
            profile_id="g_epl",
            league="EPL",
            market="game",
            params={"shrink_factor": 0.5},
            status=ProfileStatus.CANDIDATE,
        )
        draw = _make_profile(
            profile_id="d_epl",
            league="EPL",
            market="draw",
            params={"shrink_factor": 0.9},
            status=ProfileStatus.CANDIDATE,
        )
        reg, path = self._registry_with(game, draw)
        try:
            assert reg.get_production("EPL", market="draw").profile_id == "d_epl"
            assert reg.get_production("EPL", market="game").profile_id == "g_epl"
            # Default market is game.
            assert reg.get_production("EPL").profile_id == "g_epl"
        finally:
            os.unlink(path)

    def test_draw_market_falls_back_to_game_profile(self):
        from omega.core.calibration.profiles import ProfileStatus

        game = _make_profile(
            profile_id="g_epl",
            league="EPL",
            market="game",
            params={"shrink_factor": 0.5},
            status=ProfileStatus.CANDIDATE,
        )
        reg, path = self._registry_with(game)
        try:
            # No draw profile registered → draw lookup falls back to game.
            assert reg.get_production("EPL", market="draw").profile_id == "g_epl"
        finally:
            os.unlink(path)

    def test_legacy_profile_without_market_treated_as_game(self):
        # A profile dict stored before the market field existed defaults to game.
        from omega.core.calibration.profiles import CalibrationProfile

        p = _make_profile(profile_id="legacy_epl", league="EPL")
        dumped = p.model_dump()
        dumped.pop("market", None)
        assert CalibrationProfile(**dumped).market == "game"

    def test_apply_calibration_draw_uses_draw_profile(self, monkeypatch):
        import omega.core.calibration.probability as prob_mod
        from omega.core.calibration.profiles import ProfileStatus

        game = _make_profile(
            profile_id="g_epl",
            league="EPL",
            market="game",
            params={"shrink_factor": 0.2},
            status=ProfileStatus.CANDIDATE,
        )
        draw = _make_profile(
            profile_id="d_epl",
            league="EPL",
            market="draw",
            params={"shrink_factor": 1.0},
            status=ProfileStatus.CANDIDATE,
        )
        reg, path = self._registry_with(game, draw)
        try:
            monkeypatch.setattr(
                prob_mod,
                "_get_active_profile",
                lambda league, context_slice=None, market="game": reg.get_production(
                    league, context_slice=context_slice, market=market
                ),
            )
            # shrink_factor 1.0 (draw profile) is identity; 0.2 (game) shrinks hard.
            draw_cal = prob_mod.apply_calibration(0.30, league="EPL", market="draw")
            game_cal = prob_mod.apply_calibration(0.30, league="EPL", market="game")
            assert abs(draw_cal - 0.30) < 1e-9
            assert game_cal != draw_cal
        finally:
            os.unlink(path)


class TestCoverTotalExtractors:
    """Point-spread (cover) and total (over/under) calibration-plane extraction.

    Grading must match the simulation engine: home covers when
    ``(home - away) + spread_home > 0``; over wins when ``home + away > line``;
    exact pushes carry no calibration signal and are excluded.
    """

    @staticmethod
    def _trace(*, preds, spread_home=None, over_under=None, hs, as_):
        return {
            "predictions": preds,
            "odds_snapshot": {"spread_home": spread_home, "over_under": over_under},
            "_outcome": {"home_score": hs, "away_score": as_},
        }

    def test_cover_home_covers(self):
        from omega.core.calibration.fitter import CalibrationFitter

        # home -3.5, wins by 7 → (24-17) + (-3.5) = 3.5 > 0 → covered.
        t = self._trace(preds={"home_cover_prob": 60.0}, spread_home=-3.5, hs=24, as_=17)
        preds, outs = CalibrationFitter.extract_cover_pairs([t])
        assert preds == [pytest.approx(0.60, abs=1e-6)]
        assert outs == [1]

    def test_cover_does_not_cover(self):
        from omega.core.calibration.fitter import CalibrationFitter

        # home -7.5, wins by 7 → 7 + (-7.5) = -0.5 < 0 → did not cover.
        t = self._trace(preds={"home_cover_prob": 55.0}, spread_home=-7.5, hs=24, as_=17)
        _preds, outs = CalibrationFitter.extract_cover_pairs([t])
        assert outs == [0]

    def test_cover_push_excluded(self):
        from omega.core.calibration.fitter import CalibrationFitter

        # home -7, wins by exactly 7 → value 0 → push, no signal.
        t = self._trace(preds={"home_cover_prob": 50.0}, spread_home=-7.0, hs=24, as_=17)
        preds, outs = CalibrationFitter.extract_cover_pairs([t])
        assert preds == [] and outs == []

    def test_cover_skips_without_line(self):
        from omega.core.calibration.fitter import CalibrationFitter

        t = self._trace(preds={"home_cover_prob": 50.0}, spread_home=None, hs=24, as_=17)
        preds, _outs = CalibrationFitter.extract_cover_pairs([t])
        assert preds == []

    def test_total_over_win_and_under_win(self):
        from omega.core.calibration.fitter import CalibrationFitter

        # total 41. line 38.5 → over wins; line 45.5 → under wins.
        over_t = self._trace(
            preds={"over_prob": 58.0, "under_prob": 42.0}, over_under=38.5, hs=24, as_=17
        )
        op, oo = CalibrationFitter.extract_total_pairs([over_t], "over")
        assert op == [pytest.approx(0.58, abs=1e-6)] and oo == [1]

        under_t = self._trace(
            preds={"over_prob": 0.40, "under_prob": 0.60}, over_under=45.5, hs=24, as_=17
        )
        up, uo = CalibrationFitter.extract_total_pairs([under_t], "under")
        assert up == [pytest.approx(0.60, abs=1e-6)] and uo == [1]

    def test_total_push_excluded(self):
        from omega.core.calibration.fitter import CalibrationFitter

        t = self._trace(preds={"over_prob": 0.5, "under_prob": 0.5}, over_under=41.0, hs=24, as_=17)
        preds, outs = CalibrationFitter.extract_total_pairs([t], "over")
        assert preds == [] and outs == []

    def test_total_reads_line_from_input_snapshot_odds(self):
        from omega.core.calibration.fitter import CalibrationFitter

        # Line under input_snapshot.odds (the mirror) rather than odds_snapshot.
        t = {
            "predictions": {"over_prob": 0.58},
            "input_snapshot": {"odds": {"over_under": 38.5}},
            "_outcome": {"home_score": 24, "away_score": 17},
        }
        _preds, outs = CalibrationFitter.extract_total_pairs([t], "over")
        assert outs == [1]


class TestCalibrationMarketRouting:
    def test_routes_new_planes_to_their_markets(self):
        from omega.core.calibration.market import calibration_market_for_plane

        assert calibration_market_for_plane("cover") == "cover"
        assert calibration_market_for_plane("over") == "over"
        assert calibration_market_for_plane("under") == "under"
        # Existing planes unchanged.
        assert calibration_market_for_plane("game") == "game"
        assert calibration_market_for_plane("draw") == "draw"
        assert calibration_market_for_plane("prop") == "prop"
        assert calibration_market_for_plane("unknown") == "game"
        # Legacy market="draw" override still wins.
        assert calibration_market_for_plane("game", market="draw") == "draw"
