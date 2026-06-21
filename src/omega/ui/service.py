"""Backend-neutral, read-only data service for the operator console.

``ConsoleService`` is the only data path for the console. It reads through
``TraceStore(read_only=True)`` (which delegates to the Postgres repository when
``DATABASE_URL`` routes there, so SQLite/Postgres parity is preserved) and reads
validated session sidecars from the filesystem via the existing
``load_sidecar_safe`` contract.

Read-only by construction:

* the constructor refuses any store that is not ``read_only``;
* only read methods on ``TraceStore`` are ever called (no raw connection use,
  no write/mutation helpers);
* numbers come exclusively from DB rows — sidecar prose is surfaced as
  process/narrative and never parsed for protected quant fields.

Pagination/secondary filtering note: ``TraceStore.query_traces`` /
``query_ledger`` expose ``limit`` but not SQL ``OFFSET``. For Milestone A the
service performs a bounded read scan (``max_scan``, default 2000) with the
DB-native filters applied, then filters/paginates the candidate set in memory.
``Pagination.scan_capped`` is surfaced honestly when the scan window is full so
the operator knows older rows may exist beyond it. SQL-native offset paging is a
Milestone-B sharpening point and is intentionally not faked here.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any

from omega.paths import session_inbox_dir
from omega.trace.session_sidecar import (
    load_sidecar_safe,
    quality_gate_status,
)
from omega.trace.store import TraceStore
from omega.ui.schemas import (
    AuditEventView,
    BetDetail,
    BetListResponse,
    BetRow,
    HealthResponse,
    Pagination,
    SessionDetail,
    SessionListResponse,
    SessionSummary,
    Source,
    TraceDetail,
    TraceListResponse,
    TraceRow,
)

DEFAULT_PAGE_SIZE = 25
MAX_PAGE_SIZE = 200
DEFAULT_MAX_SCAN = 2000

# Only filenames matching this are accepted as a session_id when resolving a
# sidecar path — blocks path traversal on the filesystem-backed session reads.
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _clean(value: str | None) -> str | None:
    """Normalize a query-string filter: blank/whitespace -> None."""
    if value is None:
        return None
    value = value.strip()
    return value or None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _date10(value: Any) -> str:
    """First 10 chars of an ISO timestamp (the calendar date), or ''."""
    return str(value)[:10] if value else ""


def _iter_recommendations(recs: Any) -> Iterable[dict[str, Any]]:
    """Yield recommendation dicts whether the payload is a list or a single dict."""
    if isinstance(recs, list):
        for item in recs:
            if isinstance(item, dict):
                yield item
    elif isinstance(recs, dict):
        yield recs


@lru_cache(maxsize=512)
def _sport_for_league(league: str | None) -> str | None:
    """Best-effort sport label for a league code (read-only config lookup)."""
    if not league:
        return None
    try:
        from omega.core.config.leagues import get_league_config

        return get_league_config(league).get("sport")
    except Exception:  # noqa: BLE001 — sport is a display nicety, never fatal
        return None


def _paginate(
    items: list[Any], page: int, page_size: int, *, scan_capped: bool = False
) -> tuple[list[Any], Pagination]:
    page_size = max(1, min(int(page_size or DEFAULT_PAGE_SIZE), MAX_PAGE_SIZE))
    total = len(items)
    total_pages = max(1, math.ceil(total / page_size)) if total else 1
    page = max(1, min(int(page or 1), total_pages))
    start = (page - 1) * page_size
    window = items[start : start + page_size]
    pagination = Pagination(
        page=page,
        page_size=page_size,
        total=total,
        total_pages=total_pages,
        has_prev=page > 1,
        has_next=page < total_pages,
        scan_capped=scan_capped,
    )
    return window, pagination


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class ConsoleService:
    """Read-only read model for traces, bets, and session sidecars."""

    def __init__(
        self,
        store: TraceStore,
        sessions_dir: str | Path | None = None,
        *,
        max_scan: int | None = None,
    ) -> None:
        if not getattr(store, "read_only", False):
            raise ValueError(
                "ConsoleService requires a read-only TraceStore (read_only=True); "
                "the console must never hold a writable store."
            )
        self.store = store
        self.sessions_dir = Path(sessions_dir) if sessions_dir else session_inbox_dir()
        if max_scan is None:
            try:
                max_scan = int(os.environ.get("OMEGA_CONSOLE_MAX_SCAN", DEFAULT_MAX_SCAN))
            except ValueError:
                max_scan = DEFAULT_MAX_SCAN
        self.max_scan = max(1, max_scan)

    def close(self) -> None:
        self.store.close()

    # -- health ----------------------------------------------------------

    def health(self) -> HealthResponse:
        # Readiness probe: degrade gracefully rather than 500 when the DB is
        # missing/unopenable (a misconfigured --db is exactly what an operator
        # hits health to discover). The read-only open never creates the file.
        status = "ok"
        try:
            trace_count = self.store.count()
            schema_version = self.store.schema_version()
        except Exception:  # noqa: BLE001 — health must never raise
            status = "degraded"
            trace_count = -1
            schema_version = -1
        return HealthResponse(
            status=status,
            read_only=bool(self.store.read_only),
            db_path=str(self.store.db_path),
            db_source=str(self.store.db_path_source),
            trace_count=trace_count,
            schema_version=schema_version,
            sessions_dir=str(self.sessions_dir),
        )

    # -- traces ----------------------------------------------------------

    def list_traces(
        self,
        *,
        page: int = 1,
        page_size: int = DEFAULT_PAGE_SIZE,
        date_from: str | None = None,
        date_to: str | None = None,
        league: str | None = None,
        sport: str | None = None,
        kind: str | None = None,
        market: str | None = None,
        confidence: str | None = None,
        session_id: str | None = None,
    ) -> TraceListResponse:
        date_from = _clean(date_from)
        date_to = _clean(date_to)
        league = _clean(league)
        sport = _clean(sport)
        kind = _clean(kind)
        market = _clean(market)
        confidence = _clean(confidence)
        session_id = _clean(session_id)

        if session_id:
            # Direct, uncapped fetch of one session's traces (the session may be
            # older than the recency scan window).
            raw = self.store.query_by_session(session_id)
            scan_capped = False
        else:
            # DB-native filters: league + an inclusive lower date bound. The upper
            # date bound and all secondary filters are applied in memory below.
            raw = self.store.query_traces(
                league=league, start=date_from, limit=self.max_scan
            )
            scan_capped = len(raw) >= self.max_scan

        rows: list[TraceRow] = []
        for trace in raw:
            row = self._trace_row(trace)
            if league and (row.league or "") != league:
                continue
            cal = _date10(row.timestamp)
            if date_from and cal and cal < date_from:
                continue
            if date_to and cal and cal > date_to:
                continue
            if sport and (row.sport or "") != sport:
                continue
            if kind and (row.kind or "") != kind:
                continue
            if confidence and confidence not in row.confidence_tiers:
                continue
            if market and market not in row.markets:
                continue
            rows.append(row)

        window, pagination = _paginate(rows, page, page_size, scan_capped=scan_capped)
        filters = {
            "date_from": date_from,
            "date_to": date_to,
            "league": league,
            "sport": sport,
            "kind": kind,
            "market": market,
            "confidence": confidence,
            "session_id": session_id,
        }
        return TraceListResponse(rows=window, pagination=pagination, filters=filters)

    def get_trace_detail(self, trace_id: str) -> TraceDetail | None:
        trace = self.store.get_trace(trace_id)
        if trace is None:
            return None
        evidence = self.store.get_evidence_signals(trace_id)
        distributions = self.store.get_simulation_distributions(trace_id)
        outcome = self.store.get_outcome(trace_id)
        prop_outcomes = self.store.get_prop_outcomes(trace_id)
        bets = self.store.get_ledger_bets(trace_id)
        closing_lines = self.store.get_closing_lines(trace_id)
        qa_verdict = self.store.get_qa_verdict(trace_id)
        return TraceDetail(
            trace_id=trace_id,
            timestamp=trace.get("timestamp"),
            league=trace.get("league"),
            sport=_sport_for_league(trace.get("league")),
            kind=trace.get("kind"),
            matchup=trace.get("matchup"),
            session_id=trace.get("session_id"),
            execution_mode=trace.get("execution_mode"),
            aggregate_quality=_as_float(trace.get("aggregate_quality")),
            payload=trace,
            recommendations=trace.get("recommendations"),
            predictions=trace.get("predictions"),
            evidence_signals=evidence,
            simulation_distributions=distributions,
            outcome=outcome,
            prop_outcomes=prop_outcomes,
            bets=bets,
            closing_lines=closing_lines,
            qa_verdict=qa_verdict,
            field_sources={
                "recommendations": Source.DB_TRACE_PAYLOAD,
                "predictions": Source.DB_TRACE_PAYLOAD,
                "aggregate_quality": Source.DB_TRACE_PAYLOAD,
                "payload": Source.DB_TRACE_PAYLOAD,
                "evidence_signals": Source.EVIDENCE_SIGNALS,
                "simulation_distributions": Source.SIMULATION_DISTRIBUTIONS,
                "outcome": Source.OUTCOMES,
                "prop_outcomes": Source.PROP_OUTCOMES,
                "bets": Source.BET_LEDGER,
                "closing_lines": Source.CLOSING_LINES,
                "qa_verdict": Source.TRACE_QA_VERDICTS,
            },
        )

    def _trace_row(self, trace: dict[str, Any]) -> TraceRow:
        recs = trace.get("recommendations")
        tiers = sorted(
            {
                str(r.get("confidence_tier"))
                for r in _iter_recommendations(recs)
                if r.get("confidence_tier")
            }
        )
        markets = sorted(
            {str(r.get("market")) for r in _iter_recommendations(recs) if r.get("market")}
        )
        league = trace.get("league")
        has_outcome = bool(trace.get("_outcome") or trace.get("_prop_outcomes"))
        return TraceRow(
            trace_id=str(trace.get("trace_id", "")),
            timestamp=trace.get("timestamp"),
            league=league,
            sport=_sport_for_league(league),
            kind=trace.get("kind"),
            matchup=trace.get("matchup"),
            session_id=trace.get("session_id"),
            execution_mode=trace.get("execution_mode"),
            aggregate_quality=_as_float(trace.get("aggregate_quality")),
            confidence_tiers=tiers,
            markets=markets,
            has_outcome=has_outcome,
            field_sources={
                "aggregate_quality": Source.DB_TRACE_PAYLOAD,
                "confidence_tiers": Source.DB_TRACE_PAYLOAD,
                "markets": Source.DB_TRACE_PAYLOAD,
                "has_outcome": Source.OUTCOMES,
            },
        )

    # -- bets ------------------------------------------------------------

    def list_bets(
        self,
        *,
        page: int = 1,
        page_size: int = DEFAULT_PAGE_SIZE,
        date_from: str | None = None,
        date_to: str | None = None,
        league: str | None = None,
        sport: str | None = None,
        status: str | None = None,
        bookmaker: str | None = None,
        provenance: str | None = None,
    ) -> BetListResponse:
        date_from = _clean(date_from)
        date_to = _clean(date_to)
        league = _clean(league)
        sport = _clean(sport)
        status = _clean(status)
        bookmaker = _clean(bookmaker)
        provenance = _clean(provenance)

        raw = self.store.query_ledger(
            league=league,
            sport=sport,
            status=status,
            provenance=provenance,
            start=date_from,
            limit=self.max_scan,
        )
        scan_capped = len(raw) >= self.max_scan

        rows: list[BetRow] = []
        for bet in raw:
            cal = bet.get("bet_date") or _date10(bet.get("decision_timestamp"))
            if date_from and cal and cal < date_from:
                continue
            if date_to and cal and cal > date_to:
                continue
            if bookmaker and (bet.get("bookmaker") or "") != bookmaker:
                continue
            rows.append(self._bet_row(bet))

        window, pagination = _paginate(rows, page, page_size, scan_capped=scan_capped)
        filters = {
            "date_from": date_from,
            "date_to": date_to,
            "league": league,
            "sport": sport,
            "status": status,
            "bookmaker": bookmaker,
            "provenance": provenance,
        }
        return BetListResponse(rows=window, pagination=pagination, filters=filters)

    def get_bet_detail(self, ledger_id: str) -> BetDetail | None:
        row = self.store.get_ledger_bet(ledger_id)
        if row is None:
            return None
        staking = {
            "staking_policy_id": row.get("staking_policy_id"),
            "staking_policy_version": row.get("staking_policy_version"),
            "exposure_limits_version": row.get("exposure_limits_version"),
            "sizing_reasons": row.get("sizing_reasons"),
        }
        linked_trace_id = row.get("trace_id")
        linked_recs: Any | None = None
        if linked_trace_id:
            linked = self.store.get_trace(linked_trace_id)
            if linked is not None:
                # Recommendation values (edge%, kelly, units, …) come ONLY from
                # the linked trace's DB payload — never assumed to live in
                # bet_ledger.
                linked_recs = linked.get("recommendations")
        return BetDetail(
            ledger_id=ledger_id,
            trace_id=linked_trace_id,
            ledger=row,
            staking=staking,
            correlation_group=row.get("correlation_group"),
            linked_trace_id=linked_trace_id if linked_recs is not None else None,
            linked_trace_recommendations=linked_recs,
            field_sources={
                "ledger": Source.BET_LEDGER,
                "settlement": Source.BET_LEDGER,
                "staking": Source.BET_LEDGER,
                "correlation_group": Source.BET_LEDGER,
                "linked_trace_recommendations": Source.DB_TRACE_PAYLOAD,
            },
        )

    def _bet_row(self, bet: dict[str, Any]) -> BetRow:
        return BetRow(
            ledger_id=str(bet.get("ledger_id", "")),
            trace_id=bet.get("trace_id"),
            bet_date=bet.get("bet_date"),
            league=bet.get("league"),
            sport=bet.get("sport"),
            matchup=bet.get("matchup"),
            market=bet.get("market"),
            bookmaker=bet.get("bookmaker"),
            selection=bet.get("selection"),
            line=_as_float(bet.get("line")),
            odds=_as_float(bet.get("odds")),
            stake_amount=_as_float(bet.get("stake_amount")),
            net_pnl=_as_float(bet.get("net_pnl")),
            status=bet.get("status"),
            provenance=bet.get("provenance"),
            field_sources={
                "line": Source.BET_LEDGER,
                "odds": Source.BET_LEDGER,
                "stake_amount": Source.BET_LEDGER,
                "net_pnl": Source.BET_LEDGER,
            },
        )

    # -- sessions --------------------------------------------------------

    def list_sessions(
        self, *, page: int = 1, page_size: int = DEFAULT_PAGE_SIZE
    ) -> SessionListResponse:
        files = self._session_files()
        files.sort(key=lambda p: p.name, reverse=True)
        summaries = [self._session_summary(p) for p in files]
        window, pagination = _paginate(summaries, page, page_size)
        return SessionListResponse(rows=window, pagination=pagination)

    def get_session_detail(self, session_id: str) -> SessionDetail | None:
        session_id = (session_id or "").strip()
        if not _SESSION_ID_RE.match(session_id):
            return None
        path = self.sessions_dir / f"{session_id}.json"
        # Defense in depth: never read outside the sessions directory.
        try:
            if path.resolve().parent != self.sessions_dir.resolve():
                return None
        except OSError:
            return None
        if not path.is_file():
            return None

        sidecar = load_sidecar_safe(path)
        canonical_sid = sidecar.session_id if sidecar is not None else session_id
        db_traces = [self._trace_row(t) for t in self.store.query_by_session(canonical_sid)]
        field_sources = self._session_field_sources()

        if sidecar is None:
            return SessionDetail(
                session_id=session_id,
                file_name=path.name,
                sidecar_valid=False,
                sidecar_error=(
                    "sidecar unreadable or failed validation; quality-gate history "
                    "is UNKNOWN. Numbers below come only from DB traces."
                ),
                quality_gate_status="unknown",
                db_traces=db_traces,
                field_sources=field_sources,
            )

        metadata = {
            "opened_at": sidecar.opened_at,
            "closed_at": sidecar.closed_at,
            "model_version": sidecar.model_version,
            "purpose": sidecar.purpose,
            "league": sidecar.league,
            "window": sidecar.window,
            "effective_db_path": sidecar.effective_db_path,
            "runtime_db_status": sidecar.runtime_db_status,
            "bankroll": sidecar.bankroll,
            "bankroll_confirmed": sidecar.bankroll_confirmed,
        }
        audit_events = [
            AuditEventView(
                ts=e.ts,
                event_type=e.event_type,
                step=e.step,
                status=e.status,
                notes=e.notes,
                assumptions=list(e.assumptions),
                bugs=list(e.bugs),
                trace_ids=list(e.trace_ids),
                inputs=e.inputs,
                outputs=e.outputs,
            )
            for e in sidecar.audit_events
        ]
        assumptions = [a for e in sidecar.audit_events for a in e.assumptions]
        bugs = [b for e in sidecar.audit_events for b in e.bugs]
        return SessionDetail(
            session_id=canonical_sid,
            file_name=path.name,
            sidecar_valid=True,
            metadata=metadata,
            pipeline_status=dict(sidecar.pipeline_status),
            next_required_action=sidecar.next_required_action,
            exec_stats=dict(sidecar.exec_stats),
            agent_notes=sidecar.agent_notes,
            assumptions=assumptions,
            bugs=bugs,
            audit_events=audit_events,
            quality_gate_status=quality_gate_status(sidecar),
            db_traces=db_traces,
            field_sources=field_sources,
        )

    def _session_files(self) -> list[Path]:
        if not self.sessions_dir.exists():
            return []
        out: list[Path] = []
        # Non-recursive glob: the invalid/ quarantine subdir is excluded, and
        # .events.jsonl mirrors never match *.json.
        for path in self.sessions_dir.glob("*.json"):
            if not path.is_file():
                continue
            if path.name.endswith(".legacy.json"):
                continue
            out.append(path)
        return out

    def _session_summary(self, path: Path) -> SessionSummary:
        sidecar = load_sidecar_safe(path)
        if sidecar is not None:
            session_id = sidecar.session_id or path.stem
            return SessionSummary(
                session_id=session_id,
                file_name=path.name,
                sidecar_valid=True,
                opened_at=sidecar.opened_at,
                closed_at=sidecar.closed_at,
                purpose=sidecar.purpose,
                league=sidecar.league,
                model_version=sidecar.model_version,
                quality_gate_status=quality_gate_status(sidecar),
                db_trace_count=len(self.store.query_by_session(session_id)),
                event_count=len(sidecar.audit_events),
            )
        session_id = path.stem
        return SessionSummary(
            session_id=session_id,
            file_name=path.name,
            sidecar_valid=False,
            quality_gate_status="unknown",
            db_trace_count=len(self.store.query_by_session(session_id)),
            event_count=0,
        )

    @staticmethod
    def _session_field_sources() -> dict[str, str]:
        # Everything sidecar-derived is process/narrative and NON-canonical; only
        # db_traces carry authoritative numbers.
        return {
            "metadata": Source.SIDECAR_PROCESS,
            "pipeline_status": Source.SIDECAR_PROCESS,
            "next_required_action": Source.SIDECAR_PROCESS,
            "exec_stats": Source.SIDECAR_PROCESS,
            "agent_notes": Source.SIDECAR_PROCESS,
            "assumptions": Source.SIDECAR_PROCESS,
            "bugs": Source.SIDECAR_PROCESS,
            "audit_events": Source.SIDECAR_PROCESS,
            "db_traces": Source.DB_TRACE_PAYLOAD,
        }


def open_service(
    db_path: str | None = None,
    sessions_dir: str | Path | None = None,
    *,
    max_scan: int | None = None,
) -> ConsoleService:
    """Open a read-only ConsoleService (fresh read-only store per call).

    A fresh ``TraceStore(read_only=True)`` per call keeps each request on its own
    SQLite connection (sqlite3 connections are not safe to share across threads),
    and the read-only open never creates schema or touches journal mode.
    """
    store = TraceStore(db_path=db_path, read_only=True)
    return ConsoleService(store, sessions_dir=sessions_dir, max_scan=max_scan)
