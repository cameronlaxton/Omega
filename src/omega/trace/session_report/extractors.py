"""DB and sidecar extractors for derived session reports."""

from __future__ import annotations

import hashlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omega.paths import session_inbox_dir
from omega.trace.session_report.context_bundle import ReportContextBundle, ReportContextEntry
from omega.trace.session_report.models import (
    ContextBullet,
    CoverageRow,
    EngineView,
    IgnoredContextEntry,
    IntakeReportData,
    LedgerView,
    TraceReportCard,
)
from omega.trace.session_sidecar import load_sidecar_safe
from omega.trace.store import TraceStore

UTC = timezone.utc


def _db_fingerprint(path: str) -> str:
    db = Path(path)
    if not db.exists():
        return "missing"
    h = hashlib.sha256()
    with db.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _cell(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _format_number(value: Any, suffix: str = "") -> str | None:
    if value is None:
        return None
    try:
        if isinstance(value, str):
            return value
        return f"{float(value):.2f}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def _first_dict(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, dict):
            return value
    return {}


def _extract_engine_view(trace: dict[str, Any]) -> EngineView:
    result = _first_dict(trace.get("result"))
    best = _first_dict(result.get("best_bet"))
    recommendation = _first_dict(result.get("recommendation"))
    recs = trace.get("recommendations")
    rec = recs[0] if isinstance(recs, list) and recs and isinstance(recs[0], dict) else {}
    edge = _first_dict(best, recommendation, rec, result)
    tq = _first_dict(trace.get("trace_quality"))
    return EngineView(
        model_probability=_format_number(
            result.get("model_prob")
            or result.get("over_prob")
            or result.get("under_prob")
            or edge.get("model_prob")
            or edge.get("probability"),
            "%",
        ),
        edge=_format_number(edge.get("edge_pct") or edge.get("edge"), "%"),
        units=_format_number(edge.get("recommended_units") or edge.get("units")),
        tier=_cell(edge.get("confidence_tier")),
        calibration_status=_cell(tq.get("calibration_status") or tq.get("status")),
    )


def _decision_fields(trace: dict[str, Any], ledger_rows: list[dict[str, Any]]) -> dict[str, str | None]:
    first_ledger = ledger_rows[0] if ledger_rows else {}
    result = _first_dict(trace.get("result"))
    best = _first_dict(result.get("best_bet"))
    recs = trace.get("recommendations")
    rec = recs[0] if isinstance(recs, list) and recs and isinstance(recs[0], dict) else {}
    return {
        "selection": _cell(first_ledger.get("selection") or best.get("selection") or rec.get("selection")),
        "market": _cell(first_ledger.get("market") or rec.get("market") or trace.get("kind")),
        "book": _cell(first_ledger.get("bookmaker") or trace.get("input_snapshot", {}).get("bookmaker")),
        "stake_status": _cell(first_ledger.get("status") or "no ledger row"),
    }


def _ledger_view(
    trace: dict[str, Any],
    ledger_rows: list[dict[str, Any]],
    closing_rows: list[dict[str, Any]],
) -> LedgerView:
    outcome_attached = bool(trace.get("_outcome") or trace.get("_prop_outcomes"))
    if not ledger_rows:
        return LedgerView(status="no ledger row", outcome_attached=outcome_attached)
    first = ledger_rows[0]
    close_attached = bool(closing_rows)
    return LedgerView(
        status=str(first.get("status") or "pending"),
        provenance=_cell(first.get("provenance")),
        close_attached=close_attached,
        clv_available=close_attached,
        outcome_attached=outcome_attached,
        ledger_id=_cell(first.get("ledger_id")),
    )


def _sidecar_events(sidecar: dict[str, Any] | None, trace_id: str) -> list[dict[str, Any]]:
    if not sidecar:
        return []
    out = []
    for event in sidecar.get("audit_events") or []:
        if trace_id in (event.get("trace_ids") or []):
            out.append(event)
    return out


def _bundle_bucket(entry: ReportContextEntry) -> str:
    if entry.category == "data_gap":
        return "missing"
    if entry.category in {"injury_role", "weather", "lineup"}:
        return "concern"
    return "support"


def _context_for_trace(
    trace: dict[str, Any],
    *,
    evidence_rows: list[dict[str, Any]],
    sidecar: dict[str, Any] | None,
    bundle_entries: list[ReportContextEntry],
) -> list[ContextBullet]:
    bullets: list[ContextBullet] = []
    for ev in evidence_rows[:3]:
        sig = ev.get("signal_type") or "evidence"
        source = ev.get("source") or "unknown source"
        window = ev.get("obs_window") or ev.get("window") or "window unknown"
        bullets.append(
            ContextBullet(
                bucket="support",
                source_type="trace_evidence",
                text=f"{sig} captured from {source} ({window}).",
                source_title=str(source),
            )
        )
    input_snapshot = trace.get("input_snapshot") if isinstance(trace.get("input_snapshot"), dict) else {}
    captured = [
        key
        for key in ("home_context", "away_context", "player_context", "game_context", "evidence")
        if input_snapshot.get(key)
    ]
    if captured:
        bullets.append(
            ContextBullet(
                bucket="support",
                source_type="input_context",
                text=f"input context captured: {', '.join(captured)}.",
            )
        )
    for event in _sidecar_events(sidecar, str(trace.get("trace_id", "")))[:2]:
        notes = (event.get("notes") or "").replace("\n", " ").strip()
        if notes:
            bullets.append(
                ContextBullet(
                    bucket="concern" if event.get("status") in {"warn", "fail"} else "support",
                    source_type="sidecar_note",
                    text=notes[:160],
                )
            )
    for entry in bundle_entries:
        bullets.append(
            ContextBullet(
                bucket=_bundle_bucket(entry),  # type: ignore[arg-type]
                source_type="context_bundle",
                text=entry.claim,
                source_title=entry.source_title,
                source_url=entry.source_url,
            )
        )
    if not any(b.bucket == "missing" for b in bullets):
        tq = _first_dict(trace.get("trace_quality"))
        if tq.get("evidence_status") == "empty" or not evidence_rows:
            bullets.append(
                ContextBullet(
                    bucket="missing",
                    source_type="missing",
                    text="not captured: no evidence rows were available for this trace.",
                )
            )
    return bullets


def _bundle_entries_for_trace(
    *,
    bundle: ReportContextBundle | None,
    trace: dict[str, Any],
    session_id: str | None,
    ignored: list[IgnoredContextEntry],
) -> list[ReportContextEntry]:
    if bundle is None:
        return []
    trace_id = str(trace.get("trace_id") or "")
    if session_id and bundle.session_id != session_id:
        for entry in bundle.entries:
            ignored.append(
                IgnoredContextEntry(entry_id=entry.entry_id, reason="bundle session_id mismatch")
            )
        return []
    out: list[ReportContextEntry] = []
    trace_league = (trace.get("league") or "").upper()
    trace_matchup = str(trace.get("matchup") or "")
    for entry in bundle.entries:
        if entry.trace_id and entry.trace_id != trace_id:
            ignored.append(IgnoredContextEntry(entry_id=entry.entry_id, reason="trace_id mismatch"))
            continue
        if entry.league and trace_league and entry.league.upper() != trace_league:
            ignored.append(IgnoredContextEntry(entry_id=entry.entry_id, reason="league mismatch"))
            continue
        if entry.matchup and trace_matchup and entry.matchup != trace_matchup:
            ignored.append(IgnoredContextEntry(entry_id=entry.entry_id, reason="matchup mismatch"))
            continue
        if not entry.trace_id and not entry.matchup and not entry.player:
            ignored.append(IgnoredContextEntry(entry_id=entry.entry_id, reason="no match key"))
            continue
        out.append(entry)
    return out


def _sidecar_status(session_id: str | None, sidecar: dict[str, Any] | None) -> str:
    if not session_id:
        return "not scoped"
    return "loaded" if sidecar is not None else "missing_or_invalid"


def _trace_quality_status(trace: dict[str, Any]) -> str:
    tq = _first_dict(trace.get("trace_quality"))
    if not tq:
        return "missing"
    if tq.get("calibration_eligible") is True:
        return "calibration_eligible"
    return str(tq.get("status") or "ineligible")


def _evidence_learning_eligible(trace: dict[str, Any]) -> bool | None:
    tq = _first_dict(trace.get("trace_quality"))
    if not tq:
        return None
    return tq.get("evidence_status") in {"present", "recovered_predecision"}


def _coverage_rows(traces: list[dict[str, Any]]) -> list[CoverageRow]:
    counts: Counter[str] = Counter()
    for trace in traces:
        league = trace.get("league") or "unknown"
        kind = trace.get("kind") or "unknown"
        tq = _first_dict(trace.get("trace_quality"))
        status = "eligible" if tq.get("calibration_eligible") is True else "not_eligible"
        counts[f"{league} / {kind} / {status}"] += 1
    return [CoverageRow(label=k, count=v) for k, v in sorted(counts.items())]


def extract_intake_report(
    store: TraceStore,
    *,
    session_id: str | None = None,
    league: str | None = None,
    since: str | None = None,
    until: str | None = None,
    context_mode: str = "persisted",
    context_bundle: ReportContextBundle | None = None,
) -> IntakeReportData:
    if context_mode == "persisted+cited" and context_bundle is None:
        raise ValueError("persisted+cited context mode requires a context bundle")

    if session_id:
        traces = store.query_by_session(session_id)
    else:
        traces = store.query_traces(league=league, start=since, end=until, limit=100_000)
    if league:
        traces = [t for t in traces if (t.get("league") or "").upper() == league.upper()]

    sidecar_obj = None
    if session_id:
        sidecar_obj = load_sidecar_safe(session_inbox_dir() / f"{session_id}.json")
    sidecar = sidecar_obj.to_report_dict() if sidecar_obj is not None else None

    trace_ids = {str(t.get("trace_id")) for t in traces}
    ledger_rows = store.query_ledger(league=league.upper() if league else None, start=since, end=until, limit=100_000)
    ledger_by_trace: dict[str, list[dict[str, Any]]] = {}
    for row in ledger_rows:
        ledger_by_trace.setdefault(str(row.get("trace_id")), []).append(row)

    ignored_context_entries: list[IgnoredContextEntry] = []
    cards: list[TraceReportCard] = []
    linkage_counter: Counter[str] = Counter()
    provenance_counter: Counter[str] = Counter()

    for trace in sorted(traces, key=lambda t: str(t.get("timestamp") or ""), reverse=True):
        trace_id = str(trace.get("trace_id"))
        ledgers = ledger_by_trace.get(trace_id, [])
        closing = store.get_closing_lines(trace_id)
        evidence = store.get_evidence_signals(trace_id)
        if not ledgers:
            linkage_counter["no ledger row"] += 1
        else:
            provenances = sorted({str(r.get("provenance") or "unknown") for r in ledgers})
            for provenance in provenances:
                linkage_counter[provenance] += 1
                provenance_counter[provenance] += 1
        bundle_entries = _bundle_entries_for_trace(
            bundle=context_bundle,
            trace=trace,
            session_id=session_id,
            ignored=ignored_context_entries,
        )
        decision = _decision_fields(trace, ledgers)
        tq = _first_dict(trace.get("trace_quality"))
        cards.append(
            TraceReportCard(
                trace_id=trace_id,
                session_id=_cell(trace.get("session_id") or session_id),
                timestamp=_cell(trace.get("timestamp")),
                league=_cell(trace.get("league")),
                market=decision["market"],
                matchup=_cell(trace.get("matchup")),
                selection=decision["selection"],
                book=decision["book"],
                stake_status=decision["stake_status"],
                engine_view=_extract_engine_view(trace),
                ledger_view=_ledger_view(trace, ledgers, closing),
                context=_context_for_trace(
                    trace,
                    evidence_rows=evidence,
                    sidecar=sidecar,
                    bundle_entries=bundle_entries,
                ),
                trace_quality_status=_trace_quality_status(trace),
                sidecar_status=_sidecar_status(session_id, sidecar),
                evidence_status=str(tq.get("evidence_status") or "unknown"),
                calibration_eligible=tq.get("calibration_eligible"),
                evidence_learning_eligible=_evidence_learning_eligible(trace),
            )
        )

    unmatched = sorted(tid for tid in ledger_by_trace if tid not in trace_ids)
    return IntakeReportData(
        generated_at=datetime.now(UTC).isoformat(),
        source_db_path=store.db_path,
        source_db_fingerprint=_db_fingerprint(store.db_path),
        source_session_id=session_id,
        context_mode=context_mode,  # type: ignore[arg-type]
        context_bundle_id=context_bundle.bundle_id if context_bundle else None,
        trace_count=len(traces),
        ledger_count=sum(len(rows) for rows in ledger_by_trace.values()),
        sidecar_status=_sidecar_status(session_id, sidecar),
        coverage=_coverage_rows(traces),
        ledger_linkage=[CoverageRow(label=k, count=v) for k, v in sorted(linkage_counter.items())],
        provenance_split=[CoverageRow(label=k, count=v) for k, v in sorted(provenance_counter.items())],
        cards=cards,
        unmatched_ledger_rows=unmatched,
        ignored_context_entries=ignored_context_entries,
    )
