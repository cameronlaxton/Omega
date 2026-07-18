"""Phase 1: decision-support enrichment focus values.

Verification-plan coverage (design §11):
- supported focus values validate at the API; unknown focus is rejected;
- the matchup pack is the safe brief only (no raw trace payload, no
  recommendation view, no quarantined similar-spots cohorts, no denied keys);
- stress_test completes deterministically with an explicit
  "not available for this analysis" artifact (no provider call);
- other focus values run against the decision-support pack and stay
  pick-free/rank-free;
- the legacy no-focus Deep Dive path is unchanged.
"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from omega.enrichment.api import build_enrichment_app
from omega.enrichment.context_pack import build_matchup_context_pack
from omega.enrichment.providers import StubProvider
from omega.enrichment.schemas import MATCHUP_FOCUS_VALUES
from omega.enrichment.store import EnrichmentStore
from omega.enrichment.worker import STRESS_TEST_UNAVAILABLE, run_enrichment
from omega.ui.service import open_service
from tests.enrichment.conftest import seed_traces_db

_HEADERS = {"X-Omega-Enrich-Intent": "operator-console"}


def _wait_for_terminal(enrich_db: str, eid: str, timeout: float = 10.0) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        store = EnrichmentStore(enrich_db, read_only=True)
        try:
            record = store.get(eid)
        finally:
            store.close()
        if record is not None and record.status in ("completed", "failed"):
            return record
        time.sleep(0.05)
    raise AssertionError(f"enrichment {eid} did not reach a terminal status")


class TestFocusApi:
    def test_unknown_focus_is_rejected(self, traces_db, enrich_db, sessions_dir=None):
        app = build_enrichment_app(enrich_db=enrich_db, console_db=traces_db)
        client = TestClient(app)
        resp = client.post("/traces/enr-1?focus=tell_me_the_pick", headers=_HEADERS)
        assert resp.status_code == 422
        assert "focus" in resp.json()["detail"]

    @pytest.mark.parametrize("focus", MATCHUP_FOCUS_VALUES)
    def test_supported_focus_values_enqueue_and_complete(self, traces_db, enrich_db, focus):
        app = build_enrichment_app(enrich_db=enrich_db, console_db=traces_db)
        client = TestClient(app)
        resp = client.post(f"/traces/enr-1?focus={focus}", headers=_HEADERS)
        assert resp.status_code == 200
        eid = resp.json()["enrichment_id"]
        record = _wait_for_terminal(enrich_db, eid)
        assert record.status == "completed", record.error
        assert record.context_pack["mode"] == "decision_support"
        assert record.context_pack["focus"] == focus


class TestMatchupPack:
    def test_pack_is_safe_brief_only(self, traces_db):
        service = open_service(db_path=traces_db)
        try:
            pack = build_matchup_context_pack("enr-1", service, "breakdown")
        finally:
            service.close()
        assert pack["mode"] == "decision_support"
        assert pack["brief"]["markets"][0]["trace_id"] == "enr-1"
        # No recommendation view, no quarantined cohorts, no raw trace payload.
        dumped = json.dumps(pack)
        for key in (
            "recommendation_view",
            "historical_support",
            "historical_cohorts",
            "edge_pct",
            "ev_pct",
            "kelly_fraction",
            "confidence_tier",
            "recommended_units",
            "best_bet",
        ):
            assert f'"{key}"' not in dumped, key

    def test_unknown_focus_raises(self, traces_db):
        service = open_service(db_path=traces_db)
        try:
            with pytest.raises(ValueError, match="focus"):
                build_matchup_context_pack("enr-1", service, "give_me_a_pick")
        finally:
            service.close()

    def test_missing_trace_raises(self, traces_db):
        service = open_service(db_path=traces_db)
        try:
            with pytest.raises(ValueError, match="not found"):
                build_matchup_context_pack("nope", service, "breakdown")
        finally:
            service.close()


class TestStressTest:
    def test_stress_test_completes_unavailable_without_provider(
        self, traces_db, enrich_db, monkeypatch
    ):
        # A provider call would be a violation — make it explode if reached.
        def _boom(*a: Any, **k: Any):
            raise AssertionError("provider must not be called for unavailable stress_test")

        monkeypatch.setattr("omega.enrichment.worker.get_provider", _boom)
        store = EnrichmentStore(enrich_db)
        eid = store.create(
            trace_id="enr-1", trace_type=None, league=None, market=None,
            depth="focus:stress_test",
        )
        store.close()
        run_enrichment(
            enrichment_id=eid, enrich_db=enrich_db, console_db=traces_db,
            focus="stress_test",
        )
        record = _wait_for_terminal(enrich_db, eid)
        assert record.status == "completed"
        assert record.provider == "deterministic"
        assert STRESS_TEST_UNAVAILABLE in record.result.summary
        assert STRESS_TEST_UNAVAILABLE in record.result.headline.lower()


class TestStubDecisionSupport:
    def test_stub_answers_symmetrically_without_picks(self, traces_db):
        service = open_service(db_path=traces_db)
        try:
            pack = build_matchup_context_pack("enr-1", service, "counter_case")
        finally:
            service.close()
        result = StubProvider().generate_enrichment(pack)
        assert result.recommendation_type == "monitor"
        dumped = json.dumps(result.model_dump())
        from omega.core.contracts.language import blocked_language

        assert blocked_language(dumped) == []
        assert "no picks" in " ".join(result.operator_notes).lower()

    def test_legacy_no_focus_path_unchanged(self, traces_db, enrich_db):
        store = EnrichmentStore(enrich_db)
        eid = store.create(
            trace_id="enr-1", trace_type=None, league=None, market=None, depth="deep"
        )
        store.close()
        run_enrichment(enrichment_id=eid, enrich_db=enrich_db, console_db=traces_db)
        record = _wait_for_terminal(enrich_db, eid)
        assert record.status == "completed"
        # Legacy pack shape (recommendation-era deep dive), not the brief.
        assert "trust" in record.context_pack
        assert record.context_pack.get("mode") != "decision_support"
