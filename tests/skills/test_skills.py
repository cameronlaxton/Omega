"""
Tests for Omega operational skills.

Covers:
- SkillBase / SkillObservation contract
- trace-recorder: happy path + missing fields + write failure
- evidence-validator: happy path + null result + low confidence + empty data + no source
- data-quality-grader: happy path + under-downgrade + sanity bounds
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from omega.core.models import (
    AnswerPlan,
    ExecutionMode,
    GatheredFact,
    GatherSlot,
    InputImportance,
    OutputPackage,
    ProviderResult,
)
from omega.skills.base import SkillBase, SkillObservation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_slot(
    key: str = "home.off_rating",
    importance: InputImportance = InputImportance.CRITICAL,
    data_type: str = "team_stat",
    entity: str = "Lakers",
    league: str = "NBA",
) -> GatherSlot:
    return GatherSlot(
        key=key,
        data_type=data_type,
        entity=entity,
        league=league,
        importance=importance,
    )


def _make_result(
    confidence: float = 0.9,
    source: str = "espn",
    method: str = "structured_api",
    data: Dict[str, Any] | None = None,
) -> ProviderResult:
    return ProviderResult(
        data=data if data is not None else {"off_rating": 115.2},
        source=source,
        confidence=confidence,
        method=method,
    )


def _make_fact(
    key: str = "home.off_rating",
    importance: InputImportance = InputImportance.CRITICAL,
    filled: bool = True,
    result: ProviderResult | None = None,
    quality_score: float = 0.9,
) -> GatheredFact:
    slot = _make_slot(key=key, importance=importance)
    if filled and result is None:
        result = _make_result()
    return GatheredFact(slot=slot, result=result, filled=filled, quality_score=quality_score)


def _make_plan(
    packages: List[OutputPackage] | None = None,
    modes: List[ExecutionMode] | None = None,
    simulation_required: bool = True,
    betting_included: bool = True,
    downgrades: List[str] | None = None,
    thresholds: Dict[str, float] | None = None,
) -> AnswerPlan:
    return AnswerPlan(
        execution_modes=modes or [ExecutionMode.NATIVE_SIM],
        output_packages=packages or [OutputPackage.BET_CARD, OutputPackage.GAME_BREAKDOWN],
        simulation_required=simulation_required,
        betting_recommendations_included=betting_included,
        quality_thresholds=thresholds or {"bet_card": 0.7},
        downgrades=downgrades or [],
    )


# ---------------------------------------------------------------------------
# SkillBase / SkillObservation contract
# ---------------------------------------------------------------------------

class TestSkillBase:
    def test_observe_wraps_exception(self):
        """observe() must never raise even if _run() raises."""
        class BrokenSkill(SkillBase):
            name = "broken"
            stage = "test"
            def _run(self, **kwargs):
                raise RuntimeError("unexpected crash")

        obs = BrokenSkill().observe()
        assert isinstance(obs, SkillObservation)
        assert obs.ok is False
        assert obs.error is not None
        assert "crash" in obs.error

    def test_observation_to_dict(self):
        obs = SkillObservation(
            skill="test", stage="s", ok=True, findings=["a"], error=None
        )
        d = obs.to_dict()
        assert d["skill"] == "test"
        assert d["ok"] is True
        assert d["findings"] == ["a"]
        assert d["error"] is None


class TestOrchestratorSkillIntegration:
    """Prove that _emit_skill never breaks the orchestrator pipeline."""

    def test_emit_skill_continues_on_skill_crash(self):
        """If get_skill returns a skill that crashes, orchestrator logs and continues."""
        from omega.reasoning.orchestrator import Orchestrator, OrchestratorConfig

        orch = Orchestrator(OrchestratorConfig())

        class ExplodingSkill(SkillBase):
            name = "exploding"
            stage = "test"
            def _run(self, **kwargs):
                raise Exception("BOOM")

        # Patch get_skill to return our exploding skill
        with patch("omega.reasoning.orchestrator.get_skill", return_value=ExplodingSkill()):
            # Must not raise
            orch._emit_skill("exploding", trace={"test": True})

    def test_emit_skill_continues_when_disabled(self):
        """If the skill is disabled (get_skill returns None), no error."""
        from omega.reasoning.orchestrator import Orchestrator, OrchestratorConfig

        orch = Orchestrator(OrchestratorConfig())

        with patch("omega.reasoning.orchestrator.get_skill", return_value=None):
            # Must not raise
            orch._emit_skill("disabled-skill", trace={"test": True})

    def test_emit_skill_continues_on_logger_crash(self):
        """Even if the event logger crashes, orchestrator continues."""
        from omega.reasoning.orchestrator import Orchestrator, OrchestratorConfig

        orch = Orchestrator(OrchestratorConfig())

        class OkSkill(SkillBase):
            name = "ok-skill"
            stage = "test"
            def _run(self, **kwargs):
                return SkillObservation(skill=self.name, stage=self.stage, ok=True)

        with patch("omega.reasoning.orchestrator.get_skill", return_value=OkSkill()):
            with patch("omega.skills.logger.write_event", side_effect=IOError("disk full")):
                # Must not raise even if write_event crashes
                orch._emit_skill("ok-skill", trace={"test": True})


# ---------------------------------------------------------------------------
# trace-recorder
# ---------------------------------------------------------------------------

class TestTraceRecorder:
    def _skill(self):
        from omega.skills.trace_recorder import TraceRecorder
        return TraceRecorder()

    def _valid_trace(self) -> Dict[str, Any]:
        return {
            "trace_id": "abc-123",
            "run_id": "r-001",
            "timestamp": "2026-03-21T00:00:00Z",
            "prompt": "Who wins?",
            "aggregate_quality": 0.85,
        }

    def test_happy_path_sqlite(self):
        """Primary path: trace persists to SQLite via TraceStore."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            from omega.trace.store import TraceStore as RealStore
            with patch("omega.trace.store.TraceStore", lambda db_path=None: RealStore(db_path=db_path or str(Path(tmpdir) / "test.db"))):
                obs = self._skill().observe(trace=self._valid_trace())
            assert obs.ok is True
            assert obs.findings == []

    def test_jsonl_fallback_on_sqlite_failure(self):
        """When SQLite fails, trace falls back to JSONL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "traces.jsonl"
            with patch("omega.trace.store.TraceStore", side_effect=Exception("db error")), \
                 patch("omega.skills.trace_recorder._resolve_log_path", return_value=path):
                obs = self._skill().observe(trace=self._valid_trace())
            assert obs.ok is True
            lines = path.read_text().strip().splitlines()
            assert len(lines) == 1
            record = json.loads(lines[0])
            assert record["schema_version"] == 1
            assert record["trace_id"] == "abc-123"

    def test_missing_required_fields(self):
        trace = {"prompt": "test"}  # missing trace_id, run_id, timestamp
        obs = self._skill().observe(trace=trace)
        assert obs.ok is False
        finding_keys = {f.split(":")[0] for f in obs.findings}
        assert "missing_field" in finding_keys

    def test_write_failure_captured_as_finding(self):
        """When both SQLite and JSONL fail, finding is emitted."""
        with patch("omega.trace.store.TraceStore", side_effect=Exception("db error")), \
             patch("omega.skills.trace_recorder._resolve_log_path") as mock_path:
            mock_path.return_value = Path("/nonexistent/deep/path/traces.jsonl")
            with patch("pathlib.Path.mkdir", side_effect=PermissionError("no access")):
                obs = self._skill().observe(trace=self._valid_trace())
        assert obs.ok is False
        assert any("write_failed" in f for f in obs.findings)

    def test_schema_version_injected(self):
        """schema_version is injected into the trace record before persistence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "traces.jsonl"
            with patch("omega.trace.store.TraceStore", side_effect=Exception("db error")), \
                 patch("omega.skills.trace_recorder._resolve_log_path", return_value=path):
                self._skill().observe(trace=self._valid_trace())
            record = json.loads(path.read_text().strip())
            assert "schema_version" in record
            assert record["schema_version"] == 1

    def test_multiple_writes_append(self):
        """Multiple traces append to JSONL fallback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "traces.jsonl"
            with patch("omega.trace.store.TraceStore", side_effect=Exception("db error")), \
                 patch("omega.skills.trace_recorder._resolve_log_path", return_value=path):
                skill = self._skill()
                t = self._valid_trace()
                skill.observe(trace=t)
                t2 = {**t, "trace_id": "abc-456"}
                skill.observe(trace=t2)
            lines = path.read_text().strip().splitlines()
            assert len(lines) == 2


# ---------------------------------------------------------------------------
# evidence-validator
# ---------------------------------------------------------------------------

class TestEvidenceValidator:
    def _skill(self):
        from omega.skills.evidence_validator import EvidenceValidator
        return EvidenceValidator()

    def test_happy_path_all_good(self):
        facts = [_make_fact("home.off_rating"), _make_fact("away.off_rating")]
        obs = self._skill().observe(facts=facts)
        assert obs.ok is True
        assert obs.findings == []

    def test_filled_with_null_result(self):
        fact = GatheredFact(
            slot=_make_slot(importance=InputImportance.CRITICAL),
            result=None,
            filled=True,
            quality_score=0.0,
        )
        obs = self._skill().observe(facts=[fact])
        assert obs.ok is False
        assert any("null_result" in f for f in obs.findings)

    def test_unfilled_fact_is_ignored(self):
        """Unfilled facts should not generate findings."""
        fact = GatheredFact(
            slot=_make_slot(importance=InputImportance.CRITICAL),
            result=None,
            filled=False,
            quality_score=0.0,
        )
        obs = self._skill().observe(facts=[fact])
        assert obs.ok is True

    def test_low_confidence_critical(self):
        result = _make_result(confidence=0.2)
        fact = _make_fact(importance=InputImportance.CRITICAL, result=result)
        obs = self._skill().observe(facts=[fact])
        assert obs.ok is False
        assert any("low_confidence" in f for f in obs.findings)

    def test_low_confidence_optional_ignored(self):
        """Low confidence on OPTIONAL slots should not fire."""
        result = _make_result(confidence=0.1)
        fact = _make_fact(importance=InputImportance.OPTIONAL, result=result)
        obs = self._skill().observe(facts=[fact])
        assert obs.ok is True

    def test_empty_data_critical(self):
        result = _make_result(data={})
        fact = _make_fact(importance=InputImportance.CRITICAL, result=result)
        obs = self._skill().observe(facts=[fact])
        assert obs.ok is False
        assert any("empty_data" in f for f in obs.findings)

    def test_unknown_source_critical(self):
        result = _make_result(source="unknown", method="structured_api")
        fact = _make_fact(importance=InputImportance.CRITICAL, result=result)
        obs = self._skill().observe(facts=[fact])
        assert obs.ok is False
        assert any("no_source" in f for f in obs.findings)

    def test_unknown_method_critical(self):
        result = _make_result(source="espn", method="unknown")
        fact = _make_fact(importance=InputImportance.CRITICAL, result=result)
        obs = self._skill().observe(facts=[fact])
        assert obs.ok is False
        assert any("unknown_method" in f for f in obs.findings)

    def test_empty_facts_list(self):
        obs = self._skill().observe(facts=[])
        assert obs.ok is True


# ---------------------------------------------------------------------------
# data-quality-grader
# ---------------------------------------------------------------------------

class TestDataQualityGrader:
    def _skill(self):
        from omega.skills.data_quality_grader import DataQualityGrader
        return DataQualityGrader()

    def _good_facts(self) -> List[GatheredFact]:
        return [_make_fact("home.off_rating"), _make_fact("away.off_rating")]

    def test_happy_path_good_quality(self):
        facts = self._good_facts()
        plan = _make_plan(
            packages=[OutputPackage.GAME_BREAKDOWN],  # no bet_card
            betting_included=False,
            downgrades=[],
        )
        obs = self._skill().observe(facts=facts, quality=0.85, revised_plan=plan)
        assert obs.ok is True

    def test_quality_out_of_range_high(self):
        plan = _make_plan(packages=[OutputPackage.GAME_BREAKDOWN], betting_included=False)
        obs = self._skill().observe(facts=self._good_facts(), quality=1.5, revised_plan=plan)
        assert obs.ok is False
        assert any("quality_out_of_range" in f for f in obs.findings)

    def test_quality_out_of_range_low(self):
        plan = _make_plan(packages=[OutputPackage.GAME_BREAKDOWN], betting_included=False)
        obs = self._skill().observe(facts=self._good_facts(), quality=-0.1, revised_plan=plan)
        assert obs.ok is False
        assert any("quality_out_of_range" in f for f in obs.findings)

    def test_under_downgrade_bet_card_without_critical(self):
        """bet_card in revised plan but critical inputs not all filled."""
        facts = [
            _make_fact("home.off_rating", filled=False),  # CRITICAL, not filled
            _make_fact("away.off_rating"),
        ]
        plan = _make_plan(
            packages=[OutputPackage.BET_CARD],
            betting_included=True,
            downgrades=[],
        )
        obs = self._skill().observe(facts=facts, quality=0.5, revised_plan=plan)
        assert obs.ok is False
        assert any("bet_card_without_critical_inputs" in f for f in obs.findings)

    def test_under_downgrade_no_downgrades_on_low_quality(self):
        """Very low quality but gate applied no downgrades."""
        facts = self._good_facts()
        plan = _make_plan(
            packages=[OutputPackage.GAME_BREAKDOWN],
            betting_included=False,
            downgrades=[],
        )
        obs = self._skill().observe(facts=facts, quality=0.1, revised_plan=plan)
        assert obs.ok is False
        assert any("no_downgrades_on_low_quality" in f for f in obs.findings)

    def test_no_finding_when_downgrades_present_on_low_quality(self):
        """Low quality + downgrades present = expected behavior, no finding."""
        facts = self._good_facts()
        plan = _make_plan(
            packages=[OutputPackage.RESEARCH_REPORT],
            betting_included=False,
            downgrades=["dropped_bet_card", "native_sim_to_research"],
        )
        obs = self._skill().observe(facts=facts, quality=0.1, revised_plan=plan)
        # Should not fire the under_downgrade.no_downgrades_on_low_quality check
        low_quality_findings = [f for f in obs.findings if "no_downgrades_on_low_quality" in f]
        assert low_quality_findings == []
