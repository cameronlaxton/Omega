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

import dataclasses
import math
import os
import re
from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from omega.core.calibration.registry import CalibrationRegistry
from omega.paths import session_inbox_dir
from omega.trace.session_sidecar import (
    load_sidecar_safe,
    quality_gate_status,
)
from omega.trace.store import TraceStore
from omega.ui.clv import closing_line_value
from omega.ui.normalizers import (
    SessionTraceFacts,
    build_evidence_coverage,
    build_session_health_view,
    build_trace_recommendation_view,
)
from omega.ui.schemas import (
    AuditEventView,
    BetDetail,
    BetListResponse,
    BetRow,
    CalibrationProfileRow,
    CalibrationStatusView,
    CalibrationSummary,
    ClvRow,
    ClvSummary,
    ClvView,
    DiagnosticsView,
    EvidenceCoverageSummary,
    HealthResponse,
    OperatorWarningModel,
    Pagination,
    ReviewBucket,
    ReviewItem,
    ReviewQueueView,
    SessionDetail,
    SessionHealthViewModel,
    SessionListResponse,
    SessionSummary,
    SignalPerformanceRow,
    SignalPerformanceView,
    SignalScoringSummary,
    Source,
    TraceDetail,
    TraceListResponse,
    TraceRecommendationViewModel,
    TraceRow,
)

# Max sample items shown per Review Queue bucket (counts are full; lists are bounded).
_REVIEW_SAMPLE = 10

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


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _profile_status_str(status: Any) -> str:
    """Normalize a CalibrationProfile.status (ProfileStatus str-enum) to its value."""
    return str(getattr(status, "value", status))


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
        calibration_registry_path: str | Path | None = None,
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
        # Calibration registry is read-only here: only list_profiles/get_production
        # are ever called (never register/promote/_save). None -> registry default.
        self.calibration_registry_path = (
            str(calibration_registry_path) if calibration_registry_path else None
        )
        self._calreg: CalibrationRegistry | None = None

    def _calibration_registry(self) -> CalibrationRegistry:
        if self._calreg is None:
            self._calreg = CalibrationRegistry(self.calibration_registry_path)
        return self._calreg

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

    # -- diagnostics (Milestone B.2) -------------------------------------

    def diagnostics(self) -> DiagnosticsView:
        """System-health snapshot: runtime DB + calibration registry + signal
        scoring. Degrades gracefully (never raises) so a misconfigured DB or an
        absent registry surfaces as warnings rather than a 500."""
        warnings: list[OperatorWarningModel] = []
        status = "ok"
        try:
            trace_count = self.store.count()
            schema_version = self.store.schema_version()
        except Exception:  # noqa: BLE001 — diagnostics must never raise
            status = "degraded"
            trace_count = -1
            schema_version = -1
            warnings.append(
                OperatorWarningModel(
                    code="db_unavailable",
                    severity="fail",
                    message="trace DB unavailable or unreadable",
                )
            )

        latest_trace_ts: str | None = None
        try:
            recent = self.store.query_traces(limit=1)  # ORDER BY timestamp DESC
            if recent:
                latest_trace_ts = recent[0].get("timestamp")
        except Exception:  # noqa: BLE001
            pass

        # Bet count via a bounded read scan (no exact COUNT helper on the store);
        # surfaced honestly as a lower bound when the scan window is full.
        bet_count = 0
        bet_count_capped = False
        try:
            bet_count = len(self.store.query_ledger(limit=self.max_scan))
            bet_count_capped = bet_count >= self.max_scan
            if bet_count_capped:
                warnings.append(
                    OperatorWarningModel(
                        code="bet_count_capped",
                        severity="info",
                        message=f"bet count is a lower bound (scan-capped at {self.max_scan})",
                    )
                )
        except Exception:  # noqa: BLE001
            pass

        session_count = len(self._session_files())
        calibration = self._calibration_summary(warnings)
        signal_scoring = self._signal_scoring_summary()

        return DiagnosticsView(
            status=status,
            db_path=str(self.store.db_path),
            db_source=str(self.store.db_path_source),
            schema_version=schema_version,
            trace_count=trace_count,
            session_count=session_count,
            bet_count=bet_count,
            bet_count_capped=bet_count_capped,
            latest_trace_ts=latest_trace_ts,
            calibration=calibration,
            signal_scoring=signal_scoring,
            generated_at=datetime.now(timezone.utc).isoformat(),
            warnings=warnings,
            field_sources={
                "trace_count": Source.RUNTIME,
                "schema_version": Source.RUNTIME,
                "db_path": Source.RUNTIME,
                "session_count": Source.RUNTIME,
                "bet_count": Source.BET_LEDGER,
                "latest_trace_ts": Source.DB_TRACE_PAYLOAD,
                "calibration": Source.CALIBRATION_REGISTRY,
                "signal_scoring": Source.SIGNAL_PERFORMANCE,
            },
        )

    def _calibration_summary(
        self, warnings: list[OperatorWarningModel]
    ) -> CalibrationSummary:
        try:
            profiles = self._calibration_registry().list_profiles()
        except Exception:  # noqa: BLE001 — registry is optional/external
            warnings.append(
                OperatorWarningModel(
                    code="registry_unavailable",
                    severity="warn",
                    message="calibration registry unavailable or unreadable",
                )
            )
            return CalibrationSummary(registry_available=False)
        counts = Counter(_profile_status_str(p.status) for p in profiles)
        leagues_with_production = sorted(
            {
                str(p.league).upper()
                for p in profiles
                if _profile_status_str(p.status) == "production"
            }
        )
        if not profiles:
            warnings.append(
                OperatorWarningModel(
                    code="registry_empty",
                    severity="warn",
                    message="calibration registry has no profiles",
                )
            )
        return CalibrationSummary(
            registry_available=True,
            total_profiles=len(profiles),
            production=counts.get("production", 0),
            candidate=counts.get("candidate", 0),
            archived=counts.get("archived", 0),
            rejected=counts.get("rejected", 0),
            leagues_with_production=leagues_with_production,
        )

    def _signal_scoring_summary(self) -> SignalScoringSummary:
        try:
            rows = self.store.get_signal_performance(limit=100_000)
        except Exception:  # noqa: BLE001
            return SignalScoringSummary()
        if not rows:
            return SignalScoringSummary()
        leagues = {r.get("league") for r in rows if r.get("league")}
        return SignalScoringSummary(
            last_scored_at=rows[0].get("scored_at"),
            rows_in_latest_run=len(rows),
            league_count=len(leagues),
        )

    # -- calibration status (Milestone B.2) ------------------------------

    def calibration_status(
        self, *, league: str | None = None, status: str | None = None
    ) -> CalibrationStatusView:
        """All calibration profiles (optionally filtered) with the live
        production profile for each (league, slice, market) key marked active."""
        league = _clean(league)
        status = _clean(status)
        filters = {"league": league, "status": status}
        warnings: list[OperatorWarningModel] = []

        try:
            profiles = self._calibration_registry().list_profiles(
                league=league, status=status
            )
        except Exception:  # noqa: BLE001
            warnings.append(
                OperatorWarningModel(
                    code="registry_unavailable",
                    severity="warn",
                    message="calibration registry unavailable or unreadable",
                )
            )
            return CalibrationStatusView(
                registry_available=False, filters=filters, warnings=warnings
            )

        active_ids = self._active_profile_ids(profiles)
        rows = [self._calibration_row(p, active_ids) for p in profiles]

        if not profiles and not (league or status):
            warnings.append(
                OperatorWarningModel(
                    code="registry_empty",
                    severity="warn",
                    message="calibration registry has no profiles",
                )
            )
        # Leagues that have profiles but no active production profile.
        leagues_present = {str(p.league).upper() for p in profiles}
        leagues_active = {r.league.upper() for r in rows if r.is_active}
        missing = sorted(leagues_present - leagues_active)
        if missing:
            warnings.append(
                OperatorWarningModel(
                    code="no_production_profile",
                    severity="warn",
                    message=f"no production profile for: {', '.join(missing)}",
                )
            )
        stale = sorted(
            r.profile_id
            for r in rows
            if r.is_active and r.brier is None and r.calibration_error is None
        )
        if stale:
            warnings.append(
                OperatorWarningModel(
                    code="stale_metrics",
                    severity="info",
                    message=f"active profile(s) without eval metrics: {', '.join(stale)}",
                )
            )

        return CalibrationStatusView(
            registry_available=True, rows=rows, filters=filters, warnings=warnings
        )

    def _active_profile_ids(self, profiles: list[Any]) -> set[str]:
        """profile_ids the engine would actually use, via the registry's own
        ``get_production`` resolution (single source of truth) — computed once per
        distinct (league, slice, market) key rather than per row."""
        active: set[str] = set()
        seen: set[tuple[Any, Any, Any]] = set()
        registry = self._calibration_registry()
        for p in profiles:
            key = (p.league, p.context_slice, p.market)
            if key in seen:
                continue
            seen.add(key)
            try:
                prod = registry.get_production(p.league, p.context_slice, p.market)
            except Exception:  # noqa: BLE001
                prod = None
            if prod is not None:
                active.add(prod.profile_id)
        return active

    def _calibration_row(
        self, profile: Any, active_ids: set[str]
    ) -> CalibrationProfileRow:
        metrics = profile.metrics or {}
        n_eval = metrics.get("n_eval")
        return CalibrationProfileRow(
            profile_id=profile.profile_id,
            league=str(profile.league),
            version=profile.version,
            method=profile.method,
            market=profile.market,
            context_slice=profile.context_slice,
            status=_profile_status_str(profile.status),
            is_active=profile.profile_id in active_ids,
            sample_size=profile.sample_size,
            brier=_as_float(metrics.get("brier_score")),
            calibration_error=_as_float(metrics.get("calibration_error")),
            log_loss=_as_float(metrics.get("log_loss")),
            n_eval=int(n_eval) if n_eval is not None else None,
            training_window=profile.training_window,
            created_at=profile.created_at,
            promoted_at=profile.promoted_at,
            field_sources={"profile": Source.CALIBRATION_REGISTRY},
        )

    # -- signal performance (Milestone B.3) ------------------------------

    def signal_performance(
        self, *, league: str | None = None
    ) -> SignalPerformanceView:
        """The most recent signal-performance scoring run (optionally per league)."""
        league = _clean(league)
        filters = {"league": league}
        warnings: list[OperatorWarningModel] = []
        try:
            raw = self.store.get_signal_performance(league=league, limit=self.max_scan)
        except Exception:  # noqa: BLE001 — signal scoring is optional
            warnings.append(
                OperatorWarningModel(
                    code="signal_perf_unavailable",
                    severity="warn",
                    message="signal_performance table unavailable or unreadable",
                )
            )
            return SignalPerformanceView(filters=filters, warnings=warnings)

        rows = [
            SignalPerformanceRow(
                signal_type=r.get("signal_type"),
                source=r.get("source"),
                obs_window=r.get("obs_window"),
                league=r.get("league"),
                sample_size=_as_int(r.get("sample_size")),
                direction_correct=_as_int(r.get("direction_correct")),
                direction_accuracy=_as_float(r.get("direction_accuracy")),
                mean_confidence=_as_float(r.get("mean_confidence")),
                realized_hit_rate=_as_float(r.get("realized_hit_rate")),
                calibration_gap=_as_float(r.get("calibration_gap")),
                brier=_as_float(r.get("brier")),
                scored_at=r.get("scored_at"),
            )
            for r in raw
        ]
        if not rows:
            warnings.append(
                OperatorWarningModel(
                    code="no_scoring_run",
                    severity="info",
                    message="no signal-performance scoring run recorded yet",
                )
            )
        return SignalPerformanceView(
            rows=rows,
            last_scored_at=raw[0].get("scored_at") if raw else None,
            filters=filters,
            warnings=warnings,
        )

    # -- review queue (Milestone B.3) ------------------------------------

    def review_queue(self) -> ReviewQueueView:
        """Operator work buckets aggregated from existing reads — each a
        bounded-scan count plus a small linkable sample. Read-only."""
        buckets: list[ReviewBucket] = []

        # Ungraded traces (no outcome attached). Single DB-native query.
        ungraded = self.store.query_traces(has_outcome=False, limit=self.max_scan)
        buckets.append(
            ReviewBucket(
                code="ungraded_traces",
                title="Ungraded traces",
                severity="info",
                count=len(ungraded),
                scan_capped=len(ungraded) >= self.max_scan,
                source=Source.DB_TRACE_PAYLOAD,
                items=[
                    ReviewItem(
                        kind="trace",
                        id=str(t.get("trace_id") or ""),
                        label=t.get("matchup"),
                        detail=" ".join(
                            x for x in (t.get("league"), t.get("kind")) if x
                        )
                        or None,
                        href=f"/traces/{t.get('trace_id')}",
                    )
                    for t in ungraded[:_REVIEW_SAMPLE]
                ],
            )
        )

        # Pending bets (need settlement). Single DB-native query.
        pending = self.store.query_ledger(status="pending", limit=self.max_scan)
        buckets.append(
            ReviewBucket(
                code="pending_bets",
                title="Pending bets (need settlement)",
                severity="warn",
                count=len(pending),
                scan_capped=len(pending) >= self.max_scan,
                source=Source.BET_LEDGER,
                items=[
                    ReviewItem(
                        kind="bet",
                        id=str(b.get("ledger_id") or ""),
                        label=b.get("selection"),
                        detail=" · ".join(
                            x for x in (b.get("market"), b.get("matchup")) if x
                        )
                        or None,
                        href=f"/bets/{b.get('ledger_id')}",
                    )
                    for b in pending[:_REVIEW_SAMPLE]
                ],
            )
        )

        # Problem sessions: invalid sidecar OR a failed quality gate. Bounded by
        # the session inbox size (a directory listing + per-file validate).
        problems: list[tuple[str, str]] = []
        for path in self._session_files():
            sidecar = load_sidecar_safe(path)
            if sidecar is None:
                problems.append((path.stem, "sidecar invalid / unreadable"))
            elif quality_gate_status(sidecar) == "fail":
                problems.append((sidecar.session_id or path.stem, "quality gate FAIL"))
        buckets.append(
            ReviewBucket(
                code="problem_sessions",
                title="Problem sessions (invalid sidecar / gate fail)",
                severity="warn",
                count=len(problems),
                source=Source.SIDECAR_PROCESS,
                items=[
                    ReviewItem(
                        kind="session", id=sid, label=sid, detail=reason,
                        href=f"/sessions/{sid}",
                    )
                    for sid, reason in problems[:_REVIEW_SAMPLE]
                ],
            )
        )
        return ReviewQueueView(buckets=buckets)

    # -- closing-line value / market movement (Milestone B.3) ------------

    def clv_report(self, *, league: str | None = None) -> ClvView:
        """Closing-line value per bet: taken price vs the attached closing line.

        Joins ``bet_ledger`` to ``closing_lines`` on (trace_id, selection
        descriptor / market). Only bets with a matched closing line and computable
        CLV appear in ``rows``; coverage is reported in ``summary``. Read-only;
        CLV math lives in :mod:`omega.ui.clv`."""
        league = _clean(league)
        bets = self.store.query_ledger(league=league, limit=self.max_scan)
        scan_capped = len(bets) >= self.max_scan

        rows: list[ClvRow] = []
        points: list[float] = []
        beat = 0
        for bet in bets:
            trace_id = bet.get("trace_id")
            taken_odds = _as_float(bet.get("odds"))
            match = (
                self._match_closing_line(
                    self.store.get_closing_lines(trace_id),
                    bet.get("selection_descriptor"),
                    bet.get("market"),
                )
                if trace_id
                else None
            )
            if match is None:
                continue
            closing_odds = _as_float(match.get("closing_odds"))
            clv = closing_line_value(taken_odds, closing_odds)
            if clv.clv_points is None:
                continue
            points.append(clv.clv_points)
            if clv.beat_close:
                beat += 1
            rows.append(
                ClvRow(
                    ledger_id=str(bet.get("ledger_id") or ""),
                    trace_id=trace_id,
                    bet_date=bet.get("bet_date"),
                    league=bet.get("league"),
                    matchup=bet.get("matchup"),
                    market=bet.get("market"),
                    selection=bet.get("selection"),
                    status=bet.get("status"),
                    taken_odds=taken_odds,
                    closing_odds=closing_odds,
                    taken_implied=clv.taken_implied,
                    closing_implied=clv.closing_implied,
                    clv_points=clv.clv_points,
                    beat_close=clv.beat_close,
                    closing_source=match.get("source"),
                    field_sources={
                        "taken_odds": Source.BET_LEDGER,
                        "closing_odds": Source.CLOSING_LINES,
                        "clv_points": "computed:closing_minus_taken_implied",
                    },
                )
            )

        warnings: list[OperatorWarningModel] = []
        if scan_capped:
            warnings.append(
                OperatorWarningModel(
                    code="scan_capped",
                    severity="info",
                    message=f"bet scan capped at {self.max_scan}; older bets not included",
                )
            )
        summary = ClvSummary(
            bets_scanned=len(bets),
            with_closing_line=len(rows),
            beat_close=beat,
            avg_clv_points=(sum(points) / len(points)) if points else None,
        )
        return ClvView(
            rows=rows,
            summary=summary,
            filters={"league": league},
            scan_capped=scan_capped,
            warnings=warnings,
        )

    @staticmethod
    def _match_closing_line(
        closes: list[dict[str, Any]],
        selection_descriptor: str | None,
        market: str | None,
    ) -> dict[str, Any] | None:
        """Pick the closing line matching a bet: descriptor first, then market."""
        if not closes:
            return None
        if selection_descriptor:
            for c in closes:
                if c.get("selection_descriptor") == selection_descriptor:
                    return c
        if market:
            for c in closes:
                if c.get("market") == market:
                    return c
        return closes[0]

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
        # Evidence coverage counts for the visible window only (B.1). Bounded to
        # the page so the list view stays cheap; older off-page rows are not read.
        counts = self.get_trace_evidence_counts([r.trace_id for r in window])
        for row in window:
            row.evidence_coverage = counts.get(row.trace_id)
            row.field_sources["evidence_coverage"] = Source.EVIDENCE_SIGNALS
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
        # Normalized read view (B.1). All interpretation/derivation lives in the
        # read-only normalizer; here we only convert its dataclass output into the
        # JSON-safe Pydantic mirror (dataclass -> asdict -> model_validate).
        view = build_trace_recommendation_view(trace, evidence_signals=evidence)
        recommendation_view = TraceRecommendationViewModel.model_validate(
            dataclasses.asdict(view)
        )
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
            recommendation_view=recommendation_view,
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
        outcome_source = Source.OUTCOMES if trace.get("_outcome") else Source.PROP_OUTCOMES
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
                "has_outcome": outcome_source,
            },
        )

    def get_trace_evidence_counts(
        self, trace_ids: list[str]
    ) -> dict[str, EvidenceCoverageSummary]:
        """Compact evidence counts for a small set of trace_ids (the visible row
        window). Backend-neutral: one ``get_evidence_signals`` read per id, scoped
        to the page so it never scans the whole table. Counts only — no score."""
        out: dict[str, EvidenceCoverageSummary] = {}
        for tid in trace_ids:
            cov = build_evidence_coverage(self.store.get_evidence_signals(tid))
            out[tid] = EvidenceCoverageSummary(
                total_signals=cov.total_signals,
                applied_signals=cov.applied_signals,
                shadow_signals=cov.shadow_signals,
            )
        return out

    def _session_trace_facts(self, trace_ids: list[str]) -> list[SessionTraceFacts]:
        """Per-trace DB facts for session health (evidence count + outcome/bet
        presence). Backend-neutral reads; never sidecar-derived."""
        facts: list[SessionTraceFacts] = []
        for tid in trace_ids:
            evidence = self.store.get_evidence_signals(tid)
            has_outcome = bool(self.store.get_outcome(tid)) or bool(
                self.store.get_prop_outcomes(tid)
            )
            has_bet = bool(self.store.get_ledger_bets(tid))
            facts.append(
                SessionTraceFacts(
                    trace_id=tid,
                    evidence_signal_count=len(evidence),
                    has_outcome=has_outcome,
                    has_bet=has_bet,
                )
            )
        return facts

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
            # Health is still computable from DB traces alone (sidecar flags only).
            health = self._build_session_health(
                session_id=session_id,
                quality_gate="unknown",
                db_traces=db_traces,
                sidecar_valid=False,
                assumption_count=0,
                bug_count=0,
                audit_events=[],
                pipeline_status=None,
            )
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
                health=health,
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
        gate_status = quality_gate_status(sidecar)
        health = self._build_session_health(
            session_id=canonical_sid,
            quality_gate=gate_status,
            db_traces=db_traces,
            sidecar_valid=True,
            assumption_count=len(assumptions),
            bug_count=len(bugs),
            audit_events=audit_events,
            pipeline_status=dict(sidecar.pipeline_status),
        )
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
            quality_gate_status=gate_status,
            db_traces=db_traces,
            health=health,
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
            if path.is_symlink():
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

    def _build_session_health(
        self,
        *,
        session_id: str,
        quality_gate: str,
        db_traces: list[TraceRow],
        sidecar_valid: bool,
        assumption_count: int,
        bug_count: int,
        audit_events: list[AuditEventView],
        pipeline_status: dict[str, Any] | None,
    ) -> SessionHealthViewModel:
        """Build the JSON-safe session health view (B.1).

        DB-backed per-trace facts drive every count; sidecar contributes only
        counts/flags (assumption/bug counts, audit status/step, validity) — never
        narrative numbers. Conversion is dataclass -> asdict -> model_validate.
        """
        facts = self._session_trace_facts([t.trace_id for t in db_traces])
        health = build_session_health_view(
            session_id=session_id,
            quality_gate_status=quality_gate,
            trace_facts=facts,
            sidecar_valid=sidecar_valid,
            assumption_count=assumption_count,
            bug_count=bug_count,
            audit_events=audit_events,
            pipeline_status=pipeline_status,
        )
        return SessionHealthViewModel.model_validate(dataclasses.asdict(health))

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
            # Counts come from DB-backed per-trace facts (evidence/outcome/bet),
            # never from sidecar prose; only flags/counts of sidecar items.
            "health": Source.DB_TRACE_PAYLOAD,
        }


def open_service(
    db_path: str | None = None,
    sessions_dir: str | Path | None = None,
    *,
    max_scan: int | None = None,
    calibration_registry_path: str | Path | None = None,
) -> ConsoleService:
    """Open a read-only ConsoleService (fresh read-only store per call).

    A fresh ``TraceStore(read_only=True)`` per call keeps each request on its own
    SQLite connection (sqlite3 connections are not safe to share across threads),
    and the read-only open never creates schema or touches journal mode.
    """
    store = TraceStore(db_path=db_path, read_only=True)
    return ConsoleService(
        store,
        sessions_dir=sessions_dir,
        max_scan=max_scan,
        calibration_registry_path=calibration_registry_path,
    )
