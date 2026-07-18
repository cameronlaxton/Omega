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
import json
import logging
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
from omega.ui.insights import (
    build_evidence_audit,
    build_market_movement,
    build_signal_conflict,
    build_trace_guardrails,
    build_trust_breakdown,
    clv_interpretation,
)
from omega.ui.normalizers import (
    SessionTraceFacts,
    build_evidence_coverage,
    build_session_health_view,
    build_trace_recommendation_view,
)
from omega.ui.schemas import (
    AuditEventView,
    BacktestListView,
    BacktestMarginalRowView,
    BacktestRunDetail,
    BacktestRunSummary,
    BacktestScorecardRowView,
    BetDetail,
    BetListResponse,
    BetRow,
    CalibrationChart,
    CalibrationChartDot,
    CalibrationChartPoint,
    CalibrationProfileRow,
    CalibrationStatusView,
    CalibrationSummary,
    ClvRow,
    ClvScatter,
    ClvScatterPoint,
    ClvSummary,
    ClvView,
    CommandCenterView,
    ComparisonStrip,
    ComparisonStripDot,
    DiagnosticsView,
    EdgeScannerRow,
    EdgeScannerView,
    EvidenceAuditViewModel,
    EvidenceCoverageSummary,
    GuardrailsViewModel,
    HealthResponse,
    MarketMovementViewModel,
    OperatorWarningModel,
    Pagination,
    PanelState,
    QualityCell,
    QualityHeatmap,
    QualityRow,
    ReliabilityBin,
    ReliabilityDiagram,
    ReviewBucket,
    ReviewItem,
    ReviewQueueView,
    SessionDetail,
    SessionHealthViewModel,
    SessionListResponse,
    SessionSummary,
    SignalConflictViewModel,
    SignalPerformanceRow,
    SignalPerformanceView,
    SignalScoringSummary,
    SimilarCohort,
    SimilarSpotsView,
    Source,
    TraceDetail,
    TraceListResponse,
    TraceRecommendationViewModel,
    TraceRow,
    TrustBreakdownViewModel,
)

# Max sample items shown per Review Queue bucket (counts are full; lists are bounded).
_REVIEW_SAMPLE = 10

DEFAULT_PAGE_SIZE = 25
MAX_PAGE_SIZE = 200
DEFAULT_MAX_SCAN = 2000

# Edge Scanner: how many recent traces to normalize for the full page vs. the
# Command Center snapshot. Bounded so the scanner is never an unbounded scan.
DEFAULT_SCANNER_LIMIT = 50
SCANNER_SNAPSHOT_LIMIT = 8

# Similar Historical Spots (B.4): a settled-bet cohort needs this many decided
# (win+loss) results before it yields a verdict; below it the cohort is flagged
# thin (no verdict). Distinct candidate trace reads are bounded for a local budget.
SIMILAR_MIN_SAMPLE = 25
SIMILAR_MAX_TRACES = 400

# Calibration chart SVG canvas (server-computed geometry; no client math).
CHART_W = 680
CHART_H = 220

# Only filenames matching this are accepted as a session_id when resolving a
# sidecar path — blocks path traversal on the filesystem-backed session reads.
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
# Same posture for lab_run_id → directory name: block path traversal on the
# filesystem-backed backtest-artifact reads.
_LAB_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


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
# Edge Scanner helpers (pure; honest column derivation)
# ---------------------------------------------------------------------------


def _market_family(market: Any) -> str:
    """Coarse market family for market-aware labels. Never raises."""
    if not market:
        return "unknown"
    s = str(market).lower()
    if "spread" in s or "handicap" in s:
        return "spread"
    if "total" in s or "over_under" in s or "o/u" in s or s in {"over", "under"}:
        return "total"
    if "prop" in s:
        return "prop"
    if "moneyline" in s or "money_line" in s or s in {"ml", "h2h"}:
        return "moneyline"
    return "unknown"

def _edge_bucket(value: Any) -> str | None:
    """Coarse edge bucket for similar-spot cohorting. Unit-normalized like
    :func:`_edge_display`: |v|<=1 is treated as a fraction (x100)."""
    v = _as_float(value)
    if v is None:
        return None
    pct = abs(v * 100 if abs(v) <= 1 else v)
    if pct < 3.0:
        return "0-3%"
    if pct < 6.0:
        return "3-6%"
    return "6%+"


def _prob_bucket(p: float) -> str:
    """Coarse confidence letter from a calibrated probability (0-1).

    A clearly-labeled, computed derivation — surfaced only as a fallback when the
    engine did not stamp a confidence tier (see ``_confidence_grade``)."""
    if p >= 0.65:
        return "A"
    if p >= 0.58:
        return "B"
    if p >= 0.53:
        return "C"
    return "D"

def _model_output_field(rec) -> tuple[Any, str, bool]:
    """Pick the market-aware model output (field, label, is_pct).

    Spread/Total/Prop use the model line/projection; Moneyline uses the model
    probability directly (no prob→odds conversion). Returns the existing
    ExtractedFieldModel so provenance/computed flags are preserved.
    """
    fam = _market_family(rec.market.value)
    prob = rec.calibrated_probability if rec.calibrated_probability.value is not None else rec.raw_probability
    if fam == "spread":
        return rec.line, "Recorded Spread", False
    if fam == "total":
        return rec.line, "Recorded Total", False
    if fam == "prop":
        return rec.line, "Recorded Line", False
    if fam == "moneyline":
        return prob, "Model Probability", True
    # unknown — prefer a line if present, else the model probability.
    if rec.line.value is not None:
        return rec.line, "Model Output", False
    if prob.value is not None:
        return prob, "Model Probability", True
    return rec.line, "Model Output", False


def _confidence_grade(rec) -> tuple[str | None, str, bool]:
    """Confidence only when stamped by the engine."""
    tier = rec.raw_confidence_tier.value
    if tier is not None and str(tier).strip():
        return str(tier).strip().upper(), "model_confidence_tier", False
    return None, "unavailable", False


def _normalize_edge_pct(value: Any, source_path: str | None = None) -> float | None:
    """Normalize known engine edge fields to percentage points."""
    v = _as_float(value)
    if v is None:
        return None
    source = (source_path or "").lower()
    if "edge_pct" in source or "ev_pct" in source:
        return v
    if source.endswith(".edge") or source == "edge":
        return v * 100
    return v * 100 if abs(v) <= 1 else v


def _edge_display(value: Any, source_path: str | None = None) -> tuple[str | None, bool | None]:
    """Format an engine edge as a percent string from its source contract."""
    pct = _normalize_edge_pct(value, source_path)
    if pct is None:
        return None, None
    return f"{pct:+.2f}%", pct > 0


def _data_quality(trace: dict[str, Any], evidence_count: int) -> tuple[str, str | None]:
    """Data-quality verdict (state, detail) from trace_quality + evidence count.

    Handles both textual verdicts (pass/warn/fail) and numeric aggregate_quality.
    Zero evidence never reads as 'clean'."""
    tq = trace.get("trace_quality") if isinstance(trace.get("trace_quality"), dict) else {}
    agg = tq.get("aggregate_quality")
    state = "unknown"
    if isinstance(agg, str):
        a = agg.strip().lower()
        if a in {"pass", "ok", "clean", "complete"}:
            state = "clean"
        elif a == "fail":
            state = "fail"
        elif a in {"warn", "partial", "warning"}:
            state = "warn"
    elif isinstance(agg, (int, float)):
        state = "clean" if agg >= 0.8 else "warn" if agg >= 0.5 else "fail"

    detail: str | None = None
    if evidence_count == 0:
        if state in {"clean", "unknown"}:
            state = "warn"
        detail = "zero evidence"
    elif state == "fail":
        detail = "QA fail"
    elif state == "clean":
        detail = "clean"
    return state, detail


def _odds_age_seconds(trace: dict[str, Any]) -> float | None:
    """Best-effort recorded-price age (seconds) from wherever it was persisted."""
    for container in (
        trace,
        trace.get("result"),
        trace.get("trace_quality"),
        trace.get("odds_snapshot"),
    ):
        if isinstance(container, dict) and container.get("odds_age_seconds") is not None:
            return _as_float(container.get("odds_age_seconds"))
    return None


# Reason tokens that warrant a list-row caution (subset of quality.py tokens).
_ROW_WARN_REASONS = frozenset(
    {
        "high_imputation",
        "static_identity_calibration",
        "not_calibration_eligible",
        "missing_identity",
        "baseline_context",
        "empty_evidence_provided_context",
    }
)


def _row_guardrail(trace: dict[str, Any]) -> str:
    """Light worst-severity guardrail (ok|info|warn|fail) from trace_quality only.

    A cheap pre-drill-down risk signal for list rows — the full
    :func:`omega.ui.insights.build_trace_guardrails` runs on trace detail."""
    tq = trace.get("trace_quality") if isinstance(trace.get("trace_quality"), dict) else {}
    reasons = tq.get("quality_reasons") if isinstance(tq.get("quality_reasons"), list) else []
    band = tq.get("quality_band")
    if (
        "qa_failed" in reasons
        or "zero_evidence_empty_context" in reasons
        or tq.get("confidence_cap") == "Pass"
        or band == "invalid"
    ):
        return "fail"
    if band == "weak" or any(r in _ROW_WARN_REASONS for r in reasons):
        return "warn"
    if band == "usable":
        return "info"
    if band == "strong":
        return "ok"
    agg = tq.get("aggregate_quality")
    if isinstance(agg, str):
        token = agg.strip().lower()
        if token in {"fail", "invalid"}:
            return "fail"
        if token in {"warn", "weak"}:
            return "warn"
        if token in {"info", "usable"}:
            return "info"
    if isinstance(agg, (int, float)) and not isinstance(agg, bool):
        norm = agg / 100.0 if agg > 1.0 else agg
        if norm < 0.20:
            return "fail"
        if norm < 0.50:
            return "warn"
        if norm < 0.75:
            return "info"
    return "ok"


def _calibration_geometry(
    points: list[CalibrationChartPoint], view_w: int, view_h: int
) -> tuple[str, str, list[CalibrationChartDot], float, float]:
    """Lay out a single-unit two-line series into SVG coords (pure; tested).

    This is presentation geometry only — it scales already-computed values into
    pixels. It derives no betting quantity. Returns
    (model_polyline, market_polyline, dots, y_min, y_max).
    """
    pad_l, pad_r, pad_t, pad_b = 44, 14, 14, 26
    plot_w = view_w - pad_l - pad_r
    plot_h = view_h - pad_t - pad_b

    vals: list[float] = []
    for p in points:
        if p.model_value is not None:
            vals.append(p.model_value)
        if p.market_value is not None:
            vals.append(p.market_value)
    if vals:
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-9:
            lo -= 0.05
            hi += 0.05
        pad = (hi - lo) * 0.12
        y_min, y_max = lo - pad, hi + pad
    else:
        y_min, y_max = 0.0, 1.0
    span = (y_max - y_min) or 1.0
    n = len(points)

    def xcoord(i: int) -> float:
        if n <= 1:
            return pad_l + plot_w / 2
        return pad_l + plot_w * i / (n - 1)

    def ycoord(v: float) -> float:
        return pad_t + plot_h * (1 - (v - y_min) / span)

    model_pts: list[str] = []
    market_pts: list[str] = []
    dots: list[CalibrationChartDot] = []
    for i, p in enumerate(points):
        x = round(xcoord(i), 1)
        if p.model_value is not None:
            my = round(ycoord(p.model_value), 1)
            model_pts.append(f"{x},{my}")
            dots.append(
                CalibrationChartDot(
                    cx=x,
                    cy=my,
                    label=p.label,
                    model_value=p.model_value,
                    market_value=p.market_value,
                )
            )
        if p.market_value is not None:
            market_pts.append(f"{x},{round(ycoord(p.market_value), 1)}")
    return " ".join(model_pts), " ".join(market_pts), dots, round(y_min, 4), round(y_max, 4)


# ---------------------------------------------------------------------------
# V2 visual geometry (first wave) — pure, server-side, tested.
#
# These scale already-computed values into pixels (strips, scatter) or pick a
# colour tone from a ratio (heatmap). They derive NO betting quantity.
# ---------------------------------------------------------------------------

# Compact comparison strip (dumbbell / ribbon) canvas.
STRIP_W = 200
STRIP_H = 24


def _strip_x(value: float, lo: float, hi: float, view_w: int, pad: float = 8.0) -> float:
    """Map a value on a fixed [lo, hi] axis to a pixel x on the strip (clamped)."""
    plot_w = view_w - 2 * pad
    span = (hi - lo) or 1.0
    frac = (value - lo) / span
    frac = 0.0 if frac < 0.0 else 1.0 if frac > 1.0 else frac
    return round(pad + plot_w * frac, 1)


def _prob01(value: Any) -> float | None:
    """Coerce a probability to the [0, 1] scale (a percent like 58.0 -> 0.58).

    Returns None for missing/invalid/negative input. Defensive only: the
    normalizer already yields 0-1 probabilities, but a percent-scaled source must
    never clamp to 1.0 on the strip."""
    f = _as_float(value)
    if f is None or f < 0.0:
        return None
    return f / 100.0 if f > 1.0 else f


def _prob_strip(
    *,
    mode: str,
    unit: str,
    primary: tuple[float | None, str, str, str],  # (value, key, label, tone)
    secondary: tuple[float | None, str, str, str],
    gap: float | None,
    outcome: str | None = None,
    missing_note: str | None = None,
) -> ComparisonStrip | None:
    """Build a fixed-domain [0, 1] probability strip (dumbbell or ribbon).

    Returns ``None`` when both sides are missing. When only one side is present it
    renders that single dot plus ``missing_note`` — it never invents the other
    side. ``gap`` is the signed delta (model − market, or closing − taken); a
    positive gap is the favourable direction in both modes, so it tints the bar.
    """
    cy = round(STRIP_H / 2, 1)
    dots: list[ComparisonStripDot] = []
    xs: list[float] = []
    for value, key, label, tone in (primary, secondary):
        if value is None:
            continue
        v = float(value)
        cx = _strip_x(v, 0.0, 1.0, STRIP_W)
        xs.append(cx)
        dots.append(
            ComparisonStripDot(
                key=key, label=label, cx=cx, value=round(v, 6), display=f"{v * 100:.1f}%", tone=tone
            )
        )
    if not dots:
        return None
    both = len(dots) == 2
    gap_display: str | None = None
    gap_positive: bool | None = None
    seg_tone = "neutral"
    if both and gap is not None:
        gap_positive = gap > 0
        gap_display = f"{gap * 100:+.2f}%"
        if gap > 0:
            seg_tone = "pos"
        elif gap < 0:
            seg_tone = "neg"
    return ComparisonStrip(
        mode=mode,
        unit=unit,
        view_w=STRIP_W,
        view_h=STRIP_H,
        cy=cy,
        seg_x1=(min(xs) if both else None),
        seg_x2=(max(xs) if both else None),
        seg_tone=seg_tone,
        dots=dots,
        gap_display=gap_display,
        gap_positive=gap_positive,
        outcome=outcome,
        note=(missing_note if not both else None),
    )


def _model_vs_market_strip(rec: Any) -> ComparisonStrip | None:
    """Per-bet dumbbell from a normalized recommendation: Omega P(selection) vs
    the market's implied probability. Probability-space so the unit is the same
    for every market and the gap is exactly the computed edge."""
    raw = (
        rec.calibrated_probability.value
        if rec.calibrated_probability.value is not None
        else rec.raw_probability.value
    )
    model_p = _prob01(raw)
    market_p = _prob01(rec.implied_probability.value)
    if model_p is None and market_p is None:
        return None
    gap = (model_p - market_p) if (model_p is not None and market_p is not None) else None
    return _prob_strip(
        mode="model_vs_market",
        unit="win probability",
        primary=(model_p, "model", "Omega", "model"),
        secondary=(market_p, "market", "Market", "market"),
        gap=gap,
        missing_note=("market price only" if model_p is None else "model only"),
    )


def _match_rec_prob(recs: list[Any], bet: dict[str, Any]) -> float | None:
    """Model probability (calibrated, else raw) for the recommendation that a
    graded bet realized — for the reliability diagram. Matches by market family
    then exact selection, falling back to the sole recommendation. Returns None
    (skip the pair) when no confident match exists — never guesses."""
    if not recs:
        return None
    bet_fam = _market_family(bet.get("market"))
    candidates = [r for r in recs if _market_family(r.market.value) == bet_fam] or list(recs)
    chosen = None
    bet_sel = str(bet.get("selection") or "").strip().lower()
    if bet_sel:
        for r in candidates:
            if str(r.selection.value or "").strip().lower() == bet_sel:
                chosen = r
                break
    if chosen is None and len(candidates) == 1:
        chosen = candidates[0]
    if chosen is None and len(recs) == 1:
        chosen = recs[0]
    if chosen is None:
        return None
    raw = (
        chosen.calibrated_probability.value
        if chosen.calibrated_probability.value is not None
        else chosen.raw_probability.value
    )
    return _prob01(raw)


def _scatter_geometry(
    points: list[tuple[float, float]],
    *,
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
    view_w: int,
    view_h: int,
    pad_l: int = 48,
    pad_r: int = 16,
    pad_t: int = 16,
    pad_b: int = 30,
) -> tuple[list[tuple[float, float]], float | None, float | None]:
    """Lay out (x, y) values into pixel coords; also return the pixel of x=0 and
    y=0 (for quadrant guides), or None when 0 is outside the domain. Pure."""
    plot_w = view_w - pad_l - pad_r
    plot_h = view_h - pad_t - pad_b
    xspan = (x_hi - x_lo) or 1.0
    yspan = (y_hi - y_lo) or 1.0

    def xc(x: float) -> float:
        f = (x - x_lo) / xspan
        f = 0.0 if f < 0.0 else 1.0 if f > 1.0 else f
        return round(pad_l + plot_w * f, 1)

    def yc(y: float) -> float:
        f = (y - y_lo) / yspan
        f = 0.0 if f < 0.0 else 1.0 if f > 1.0 else f
        return round(pad_t + plot_h * (1 - f), 1)

    out = [(xc(px), yc(py)) for px, py in points]
    x0 = xc(0.0) if x_lo <= 0.0 <= x_hi else None
    y0 = yc(0.0) if y_lo <= 0.0 <= y_hi else None
    return out, x0, y0


def _quality_tone(ratio: float | None) -> str:
    """Green/amber/red/muted tone for a coverage ratio (0-1). None -> muted."""
    if ratio is None:
        return "muted"
    if ratio >= 0.8:
        return "good"
    if ratio >= 0.5:
        return "warn"
    return "bad"


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
        backtests_dir: str | Path | None = None,
    ) -> None:
        if not getattr(store, "read_only", False):
            raise ValueError(
                "ConsoleService requires a read-only TraceStore (read_only=True); "
                "the console must never hold a writable store."
            )
        self.store = store
        self.sessions_dir = Path(sessions_dir) if sessions_dir else session_inbox_dir()
        # Lab-run artifacts root (read-only JSON). Defaults to the orchestrator's
        # var/historical/lab_runs (see omega.historical.lab.orchestrator.lab_dir).
        self.backtests_dir = (
            Path(backtests_dir) if backtests_dir else Path("var/historical/lab_runs")
        )
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

    # -- backtest scorecard (read-only lab-run artifacts) ----------------

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        """Read a JSON object from disk, or None if absent/unreadable (read-only)."""
        try:
            if not path.is_file():
                return None
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except (OSError, ValueError):
            return None

    def list_backtest_runs(self) -> BacktestListView:
        """Summaries of every historical-validation lab run under the artifacts root.

        Reads each run's LAB_RUN.json (+ PROMOTION_EVIDENCE.json for the
        incremental-edge coherence flag). Pure filesystem reads — no DB, no mutation;
        the numbers shown on the detail page come from the engine-produced
        backtest_report.json, never recomputed here.
        """
        root = self.backtests_dir
        warnings: list[str] = []
        if not root.is_dir():
            warnings.append(
                f"No lab-run artifacts directory at {root} (run omega-historical-lab-run)."
            )
            return BacktestListView(runs=[], backtests_dir=str(root), warnings=warnings)
        runs: list[BacktestRunSummary] = []
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            lab_run = self._read_json(d / "LAB_RUN.json")
            if lab_run is None:
                continue
            evidence = self._read_json(d / "PROMOTION_EVIDENCE.json") or {}
            runs.append(
                BacktestRunSummary(
                    lab_run_id=str(lab_run.get("lab_run_id") or d.name),
                    league=lab_run.get("league"),
                    plane=lab_run.get("plane"),
                    promotion_status=lab_run.get("promotion_status"),
                    created_at=lab_run.get("created_at"),
                    holdout_sealed=bool(lab_run.get("holdout_sealed")),
                    clv_coherent=evidence.get("clv_coherent"),
                    has_report=(d / "backtest_report.json").is_file(),
                )
            )
        runs.sort(key=lambda r: r.created_at or "", reverse=True)
        return BacktestListView(runs=runs, backtests_dir=str(root), warnings=warnings)

    def backtest_run_detail(self, lab_run_id: str) -> BacktestRunDetail | None:
        """Full scorecard for one lab run, or None when missing / id is unsafe."""
        if not _LAB_RUN_ID_RE.match(lab_run_id or ""):
            return None
        d = self.backtests_dir / lab_run_id
        lab_run = self._read_json(d / "LAB_RUN.json")
        if lab_run is None:
            return None
        evidence = self._read_json(d / "PROMOTION_EVIDENCE.json") or {}
        report = self._read_json(d / "backtest_report.json") or {}
        warnings: list[str] = []
        if not report:
            warnings.append("No backtest_report.json (walk-forward did not run for this plane).")
        scorecard = [
            BacktestScorecardRowView.model_validate(row)
            for row in report.get("scorecard", [])
            if isinstance(row, dict)
        ]
        marginal = [
            BacktestMarginalRowView.model_validate(row)
            for row in report.get("aggregate_marginal_value", [])
            if isinstance(row, dict)
        ]
        return BacktestRunDetail(
            lab_run_id=str(lab_run.get("lab_run_id") or lab_run_id),
            league=lab_run.get("league"),
            plane=lab_run.get("plane"),
            promotion_status=lab_run.get("promotion_status"),
            created_at=lab_run.get("created_at"),
            holdout_sealed=bool(lab_run.get("holdout_sealed")),
            clv_coherent=evidence.get("clv_coherent"),
            scorecard=scorecard,
            marginal_value=marginal,
            warnings=warnings,
        )

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

    # -- command center (V2 landing) -------------------------------------

    def _safe_panel(
        self,
        panels: dict[str, PanelState],
        code: str,
        title: str,
        fn,
        *,
        source: str | None = None,
        empty_when=None,
        empty_msg: str | None = None,
        degraded_msg: str | None = None,
    ):
        """Run one Command Center panel's read in isolation.

        Records a PanelState (data | empty | degraded) and returns the read's
        value, or ``None`` if it raised. A single panel failing degrades only
        that panel — the landing page never 500s because one read broke.
        """
        try:
            value = fn()
        except Exception:  # noqa: BLE001 — failure isolation is the whole point
            logging.getLogger(__name__).exception("command_center panel %r failed", code)
            panels[code] = PanelState(
                code=code,
                title=title,
                state="degraded",
                message=degraded_msg or f"{title} unavailable; source read failed.",
                source=source,
            )
            return None
        is_empty = bool(empty_when(value)) if empty_when is not None else False
        panels[code] = PanelState(
            code=code,
            title=title,
            state="empty" if is_empty else "data",
            message=empty_msg if is_empty else None,
            source=source,
        )
        return value

    def command_center(self) -> CommandCenterView:
        """Compose the V2 landing 'Command Center' from existing read methods.

        A SUMMARY (the Review Queue page is the workbench). Every panel is read
        independently with failure isolation; all reads are bounded (``max_scan``)
        and no new DB access pattern is introduced. Calibration Health and Recent
        Trace Failures are derived from already-fetched views (diagnostics /
        review) so they cost no extra read.
        """
        panels: dict[str, PanelState] = {}

        # Vitals bar source (never raises; degrades internally).
        health = self.health()

        scanner = self._safe_panel(
            panels,
            "scanner",
            "Edge Scanner",
            lambda: self.edge_scanner(limit=SCANNER_SNAPSHOT_LIMIT),
            source=Source.DB_TRACE_PAYLOAD,
            empty_when=lambda v: len(v.rows) == 0,
            empty_msg="No recent DB-backed recommendations found.",
        )
        review = self._safe_panel(
            panels,
            "review",
            "Review Queue",
            self.review_queue,
            source=Source.RUNTIME,
            empty_when=lambda v: sum(b.count for b in v.buckets) == 0,
            empty_msg="Nothing needs review right now.",
        )
        diagnostics = self._safe_panel(
            panels,
            "diagnostics",
            "Data Quality Watch",
            self.diagnostics,
            source=Source.RUNTIME,
        )
        clv = self._safe_panel(
            panels,
            "ledger",
            "Ledger Snapshot",
            self.clv_report,
            source=Source.BET_LEDGER,
            empty_when=lambda v: v.summary.bets_scanned == 0,
            empty_msg="No bets in the ledger yet.",
        )

        # Calibration Health — derived from diagnostics (no extra read).
        if diagnostics is not None:
            cal = diagnostics.calibration
            has_cal = cal.registry_available and cal.total_profiles > 0
            panels["calibration"] = PanelState(
                code="calibration",
                title="Calibration Health",
                state="data" if has_cal else "empty",
                message=None if has_cal else "No calibration profiles in the registry.",
                source=Source.CALIBRATION_REGISTRY,
            )
        else:
            panels["calibration"] = PanelState(
                code="calibration",
                title="Calibration Health",
                state="degraded",
                message="Calibration health unavailable; diagnostics read failed.",
                source=Source.CALIBRATION_REGISTRY,
            )

        # Recent Trace Failures — derived from the review qa_fail bucket.
        if review is not None:
            qa = next((b for b in review.buckets if b.code == "qa_fail"), None)
            n_fail = qa.count if qa else 0
            panels["failures"] = PanelState(
                code="failures",
                title="Recent Trace Failures",
                state="data" if n_fail else "empty",
                message=None if n_fail else "No QA-failing traces in the scan window.",
                source=Source.DB_TRACE_PAYLOAD,
            )
        else:
            panels["failures"] = PanelState(
                code="failures",
                title="Recent Trace Failures",
                state="degraded",
                message="Trace failures unavailable; review read failed.",
                source=Source.DB_TRACE_PAYLOAD,
            )

        review_count = sum(b.count for b in review.buckets) if review is not None else 0

        # Calibration summary chart — reuse the already-fetched CLV view so the
        # landing does not run a second ledger scan (perf contract). Never fatal.
        calibration_chart = None
        if clv is not None:
            try:
                calibration_chart = self.calibration_chart(clv=clv)
            except Exception:  # noqa: BLE001 — the chart is a nicety, never fatal
                logging.getLogger(__name__).exception("command_center calibration chart failed")
                calibration_chart = None

        return CommandCenterView(
            generated_at=datetime.now(timezone.utc).isoformat(),
            panels=panels,
            health=health,
            review=review,
            diagnostics=diagnostics,
            clv=clv,
            scanner=scanner,
            calibration_chart=calibration_chart,
            review_count=review_count,
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
        except Exception as e:  # noqa: BLE001
            logging.getLogger(__name__).error("Failed to query recent traces: %s", e)
            status = "degraded"
            warnings.append(
                OperatorWarningModel(
                    code="query_traces_failed",
                    severity="fail",
                    message="failed to read recent traces from DB",
                )
            )

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
        except Exception as e:  # noqa: BLE001
            logging.getLogger(__name__).error("Failed to query ledger for bet count: %s", e)
            status = "degraded"
            warnings.append(
                OperatorWarningModel(
                    code="query_ledger_failed",
                    severity="fail",
                    message="failed to read bet ledger from DB",
                )
            )

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

    def _calibration_summary(self, warnings: list[OperatorWarningModel]) -> CalibrationSummary:
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
            profiles = self._calibration_registry().list_profiles(league=league, status=status)
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

    def _calibration_row(self, profile: Any, active_ids: set[str]) -> CalibrationProfileRow:
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
            binding_status=profile.binding_status().value,
            sample_size=profile.sample_size,
            brier=_as_float(metrics.get("brier_score")),
            calibration_error=_as_float(metrics.get("calibration_error")),
            log_loss=_as_float(metrics.get("log_loss")),
            n_eval=_as_int(n_eval) if n_eval is not None else None,
            training_window=profile.training_window,
            created_at=profile.created_at,
            promoted_at=profile.promoted_at,
            field_sources={"profile": Source.CALIBRATION_REGISTRY},
        )

    # -- signal performance (Milestone B.3) ------------------------------

    def signal_performance(self, *, league: str | None = None) -> SignalPerformanceView:
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
                        detail=" ".join(x for x in (t.get("league"), t.get("kind")) if x) or None,
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
                        detail=" · ".join(x for x in (b.get("market"), b.get("matchup")) if x)
                        or None,
                        href=f"/bets/{b.get('ledger_id')}",
                    )
                    for b in pending[:_REVIEW_SAMPLE]
                ],
            )
        )

        recent_traces = self.store.query_traces(limit=self.max_scan)
        trace_ids = [t.get("trace_id", "") for t in recent_traces]
        facts_batch = self.store.get_session_trace_facts_batch(trace_ids) if trace_ids else {}
        zero_evidence = []
        qa_fail = []
        for t in recent_traces:
            tid = str(t.get("trace_id", ""))
            f = facts_batch.get(tid)
            if f and f.get("evidence_signal_count", 0) == 0:
                zero_evidence.append(t)
            q = t.get("trace_quality") or {}
            if q.get("aggregate_quality") == "fail":
                qa_fail.append(t)

        buckets.append(
            ReviewBucket(
                code="zero_evidence",
                title="Zero evidence (empty context)",
                severity="warn",
                count=len(zero_evidence),
                scan_capped=len(zero_evidence) >= self.max_scan,
                source=Source.DB_TRACE_PAYLOAD,
                items=[
                    ReviewItem(
                        kind="trace",
                        id=str(t.get("trace_id") or ""),
                        label=t.get("matchup"),
                        detail=" ".join(x for x in (t.get("league"), t.get("kind")) if x) or None,
                        href=f"/traces/{t.get('trace_id')}",
                    )
                    for t in zero_evidence[:_REVIEW_SAMPLE]
                ],
            )
        )

        buckets.append(
            ReviewBucket(
                code="qa_fail",
                title="QA gate fail (trace_quality)",
                severity="warn",
                count=len(qa_fail),
                scan_capped=len(qa_fail) >= self.max_scan,
                source=Source.DB_TRACE_PAYLOAD,
                items=[
                    ReviewItem(
                        kind="trace",
                        id=str(t.get("trace_id") or ""),
                        label=t.get("matchup"),
                        detail=" ".join(x for x in (t.get("league"), t.get("kind")) if x) or None,
                        href=f"/traces/{t.get('trace_id')}",
                    )
                    for t in qa_fail[:_REVIEW_SAMPLE]
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
                        kind="session",
                        id=sid,
                        label=sid,
                        detail=reason,
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

        trace_ids = list({bet.get("trace_id") for bet in bets if bet.get("trace_id")})
        closing_lines_by_trace = self.store.get_closing_lines_batch(trace_ids) if trace_ids else {}

        rows: list[ClvRow] = []
        points: list[float] = []
        beat = 0
        for bet in bets:
            trace_id = bet.get("trace_id")
            taken_odds = _as_float(bet.get("odds"))
            match = (
                self._match_closing_line(
                    closing_lines_by_trace.get(trace_id, []),
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
                    interpretation=clv_interpretation(clv.clv_points),
                    closing_source=match.get("source"),
                    net_pnl=_as_float(bet.get("net_pnl")),
                    strip=_prob_strip(
                        mode="market_movement",
                        unit="implied probability (incl. vig)",
                        primary=(clv.taken_implied, "taken", "Taken", "model"),
                        secondary=(clv.closing_implied, "closing", "Close", "market"),
                        gap=clv.clv_points,
                        outcome=bet.get("status"),
                        missing_note="no close",
                    ),
                    field_sources={
                        "taken_odds": Source.BET_LEDGER,
                        "closing_odds": Source.CLOSING_LINES,
                        "clv_points": "computed:closing_minus_taken_implied",
                        "interpretation": "computed:clv_interpretation",
                        "net_pnl": Source.BET_LEDGER,
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
        return None

    # -- edge scanner (V2) -----------------------------------------------

    def edge_scanner(
        self, *, limit: int | None = None, league: str | None = None
    ) -> EdgeScannerView:
        """Recent DB-backed recommendations with honest columns (read-only).

        NOT a live feed: rows are the most recent traces that produced a
        recommendation, ranked by engine edge. Bounded by ``limit`` (default
        ``DEFAULT_SCANNER_LIMIT``). Every column is a real engine value or a
        clearly labeled/computed derivation — there is no fabricated value score,
        and ``recorded_price`` is the price recorded on the trace at decision
        time (Omega has no live multi-book quote).
        """
        league = _clean(league)
        n = max(1, int(limit or DEFAULT_SCANNER_LIMIT))
        traces = self.store.query_traces(league=league, limit=n)
        scan_capped = len(traces) >= n
        tids = [str(t.get("trace_id") or "") for t in traces]
        ev_counts = self.get_trace_evidence_counts(tids) if tids else {}

        rows: list[EdgeScannerRow] = []
        for t in traces:
            tid = str(t.get("trace_id") or "")
            view = build_trace_recommendation_view(t)
            rvm = TraceRecommendationViewModel.model_validate(dataclasses.asdict(view))
            if not rvm.recommendations:
                continue  # only surface traces that produced a recommendation
            rec = next((r for r in rvm.recommendations if r.is_primary), rvm.recommendations[0])
            model_output, label, is_pct = _model_output_field(rec)
            edge_display, edge_positive = _edge_display(
                rec.engine_edge.value, rec.engine_edge.source_path
            )
            grade, csource, ccomputed = _confidence_grade(rec)
            cov = ev_counts.get(tid)
            ev_n = cov.total_signals if cov else 0
            dq, dq_detail = _data_quality(t, ev_n)
            strip = _model_vs_market_strip(rec)
            rows.append(
                EdgeScannerRow(
                    trace_id=tid,
                    timestamp=t.get("timestamp"),
                    league=t.get("league"),
                    matchup=t.get("matchup"),
                    kind=t.get("kind"),
                    market=rec.market,
                    selection=rec.selection,
                    model_output_label=label,
                    model_output=model_output,
                    model_output_is_pct=is_pct,
                    recorded_price=rec.odds,
                    edge=rec.engine_edge,
                    edge_display=edge_display,
                    edge_positive=edge_positive,
                    confidence=grade,
                    confidence_source=csource,
                    confidence_computed=ccomputed,
                    data_quality=dq,
                    data_quality_detail=dq_detail,
                    guardrail=_row_guardrail(t),
                    has_outcome=bool(t.get("_outcome") or t.get("_prop_outcomes")),
                    strip=strip,
                    field_sources={
                        "market": Source.DB_TRACE_PAYLOAD,
                        "selection": Source.DB_TRACE_PAYLOAD,
                        "model_output": Source.DB_TRACE_PAYLOAD,
                        "recorded_price": Source.DB_TRACE_PAYLOAD,
                        "edge": Source.DB_TRACE_PAYLOAD,
                        "confidence": Source.DB_TRACE_PAYLOAD,
                        "data_quality": Source.DB_TRACE_PAYLOAD,
                        "guardrail": Source.DB_TRACE_PAYLOAD,
                    },
                )
            )

        # Rank by engine edge desc (unit-normalized); rows without an edge sort last.
        def _edge_sort_key(r: EdgeScannerRow) -> tuple[bool, float]:
            v = _normalize_edge_pct(r.edge.value, r.edge.source_path)
            if v is None:
                return (True, 0.0)
            return (False, -v)

        rows.sort(key=_edge_sort_key)

        warnings: list[OperatorWarningModel] = []
        if scan_capped:
            warnings.append(
                OperatorWarningModel(
                    code="scan_capped",
                    severity="info",
                    message=f"scanner scan capped at {n}; older traces not included",
                )
            )
        return EdgeScannerView(
            rows=rows,
            scan_capped=scan_capped,
            generated_at=datetime.now(timezone.utc).isoformat(),
            filters={"league": league, "limit": n},
            warnings=warnings,
        )

    # -- matchup briefs (decision-support primary surface, Phase 0) ---------

    def list_matchup_briefs(
        self, *, league: str | None = None, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Recent traces grouped by event identity, projected through the safe
        decision-support adapter. Ordered by stable event identity, never by
        edge or recommendation status; no-recommendation traces are included.
        """
        from omega.trace.decision_support import group_traces_into_briefs

        league = _clean(league)
        n = max(1, int(limit or DEFAULT_SCANNER_LIMIT))
        # Bounded scan (Phase 2 may add a DB-native event index if benchmarks
        # demand one); grouping happens in memory on the scanned window. A
        # single event can span more than 4 traces (one game + many props), so
        # a fixed trace-to-event multiplier can under-scan; widen the window
        # until enough distinct event groups are found or the scan cap (
        # self.max_scan) is reached.
        scan_limit = min(self.max_scan, n * 4)
        briefs: list[Any] = []
        while True:
            traces = self.store.query_traces(league=league, limit=scan_limit)
            briefs = group_traces_into_briefs(traces)
            if len(briefs) >= n or scan_limit >= self.max_scan:
                break
            scan_limit = min(self.max_scan, scan_limit * 2)
        # query_traces() scans newest-first, but group_traces_into_briefs()
        # sorts ascending by date for stable presentation; reverse before
        # slicing so briefs[:n] keeps the newest n groups, not the oldest.
        briefs.reverse()
        return [b.model_dump(mode="json") for b in briefs[:n]]

    def matchup_brief(self, group_key: str) -> dict[str, Any] | None:
        """The safe brief for one event group (or one legacy ``trace:<id>`` key)."""
        from omega.trace.decision_support import brief_for_group_key

        brief = brief_for_group_key(self.store, group_key, max_scan=self.max_scan)
        return brief.model_dump(mode="json") if brief is not None else None

    # -- calibration chart (V2) ------------------------------------------

    def calibration_chart(
        self, *, league: str | None = None, clv: ClvView | None = None
    ) -> CalibrationChart:
        """Single-unit model-vs-market time series (read-only, no recomputation).

        Plots, per day, the average *implied probability* Omega took vs. the
        market's closing implied probability — both in the SAME unit (probability
        %). The gap between the lines is closing-line value. Composed from
        :meth:`clv_report` (pass a precomputed ``clv`` to avoid a second scan).
        """
        league = _clean(league)
        clv = clv if clv is not None else self.clv_report(league=league)

        agg: dict[str, dict[str, list[float]]] = {}
        for r in clv.rows:
            d = (r.bet_date or "")[:10]
            if not d:
                continue
            a = agg.setdefault(d, {"m": [], "k": []})
            if r.taken_implied is not None:
                a["m"].append(r.taken_implied)
            if r.closing_implied is not None:
                a["k"].append(r.closing_implied)

        points: list[CalibrationChartPoint] = []
        for d in sorted(agg):
            m, k = agg[d]["m"], agg[d]["k"]
            points.append(
                CalibrationChartPoint(
                    label=d,
                    model_value=(sum(m) / len(m)) if m else None,
                    market_value=(sum(k) / len(k)) if k else None,
                    n=max(len(m), len(k)),
                )
            )

        model_pl, market_pl, dots, y_min, y_max = _calibration_geometry(points, CHART_W, CHART_H)
        warnings: list[OperatorWarningModel] = []
        if not points:
            warnings.append(
                OperatorWarningModel(
                    code="no_clv_data",
                    severity="info",
                    message="no matched closing lines to chart yet (CLV needs a captured close)",
                )
            )
        return CalibrationChart(
            mode="implied_prob_model_vs_market",
            unit="implied probability (%)",
            y_label="Implied probability",
            model_series_label="Omega (taken)",
            market_series_label="Market close",
            points=points,
            view_w=CHART_W,
            view_h=CHART_H,
            model_polyline=model_pl,
            market_polyline=market_pl,
            dots=dots,
            y_min=y_min,
            y_max=y_max,
            sample=len(clv.rows),
            filters={"league": league},
            warnings=warnings,
        )

    # -- CLV scatter (V2: process vs luck) -------------------------------

    def clv_scatter(self, *, league: str | None = None, clv: ClvView | None = None) -> ClvScatter:
        """Per-bet scatter of CLV (x) against net result (y), with quadrant guides.

        Only bets that have BOTH an attached closing line and a graded net result
        are plotted; the excluded count is reported honestly. Read-only; composed
        from :meth:`clv_report` (pass a precomputed ``clv`` to avoid a second scan).
        """
        league = _clean(league)
        clv = clv if clv is not None else self.clv_report(league=league)
        view_w, view_h = 480, 320

        pairs = [r for r in clv.rows if r.clv_points is not None and r.net_pnl is not None]
        excluded = max(0, clv.summary.bets_scanned - len(pairs))
        warnings: list[OperatorWarningModel] = []
        if not pairs:
            warnings.append(
                OperatorWarningModel(
                    code="no_graded_clv",
                    severity="info",
                    message="no bets with both a closing line and a graded result yet",
                )
            )
            return ClvScatter(
                view_w=view_w,
                view_h=view_h,
                n_plotted=0,
                n_excluded=excluded,
                filters={"league": league},
                warnings=warnings,
            )

        xs = [float(r.clv_points) for r in pairs]
        ys = [float(r.net_pnl) for r in pairs]
        x_mag = max(0.01, max(abs(min(xs)), abs(max(xs))))
        x_lo, x_hi = -x_mag * 1.1, x_mag * 1.1
        y_min, y_max = min(ys + [0.0]), max(ys + [0.0])
        if y_max - y_min < 1e-9:
            y_min, y_max = y_min - 1.0, y_max + 1.0
        y_pad = (y_max - y_min) * 0.1
        y_lo, y_hi = y_min - y_pad, y_max + y_pad

        coords, x0, y0 = _scatter_geometry(
            [(float(r.clv_points), float(r.net_pnl)) for r in pairs],
            x_lo=x_lo,
            x_hi=x_hi,
            y_lo=y_lo,
            y_hi=y_hi,
            view_w=view_w,
            view_h=view_h,
        )
        points: list[ClvScatterPoint] = []
        for r, (cx, cy) in zip(pairs, coords):
            pnl = float(r.net_pnl)
            tone = "pos" if pnl > 0 else "neg" if pnl < 0 else "neutral"
            points.append(
                ClvScatterPoint(
                    ledger_id=r.ledger_id,
                    cx=cx,
                    cy=cy,
                    clv_points=round(float(r.clv_points), 6),
                    net_pnl=round(pnl, 2),
                    status=r.status,
                    tone=tone,
                    label=(r.matchup or r.selection or r.ledger_id),
                    clv_display=f"{float(r.clv_points) * 100:+.2f}%",
                    pnl_display=f"{pnl:+.2f}",
                )
            )
        return ClvScatter(
            view_w=view_w,
            view_h=view_h,
            x0=x0,
            y0=y0,
            points=points,
            n_plotted=len(points),
            n_excluded=excluded,
            filters={"league": league},
            warnings=warnings,
        )

    # -- reliability diagram (V2: over-confidence detector) --------------

    def reliability_diagram(
        self, *, league: str | None = None, bins: int = 10, min_n: int = 5
    ) -> ReliabilityDiagram:
        """Model probability bucket vs realized hit rate, with the y=x diagonal.

        Built from graded (won/lost) bets joined to the model probability of the
        recommendation that produced them. Buckets with fewer than ``min_n`` pairs
        are SUPPRESSED (not plotted) so thin buckets never read as calibrated.
        Read-only; no engine quantity is recomputed.
        """
        league = _clean(league)
        view_w, view_h = 360, 360
        bins = max(2, int(bins))
        min_n = max(1, int(min_n))

        bets = self.store.query_ledger(league=league, limit=self.max_scan)
        graded = [b for b in bets if str(b.get("status") or "").lower() in {"won", "lost"}]

        rec_cache: dict[str, list[Any]] = {}
        pairs: list[tuple[float, int]] = []
        for b in graded:
            tid = b.get("trace_id")
            if not tid:
                continue
            if tid not in rec_cache:
                tr = self.store.get_trace(tid)
                rec_cache[tid] = list(build_trace_recommendation_view(tr).recommendations) if tr else []
            prob = _match_rec_prob(rec_cache[tid], b)
            if prob is None:
                continue
            pairs.append((prob, 1 if str(b.get("status")).lower() == "won" else 0))

        # diagonal endpoints are the plot corners (data (0,0) and (1,1)).
        diag, _x0, _y0 = _scatter_geometry(
            [(0.0, 0.0), (1.0, 1.0)], x_lo=0.0, x_hi=1.0, y_lo=0.0, y_hi=1.0, view_w=view_w, view_h=view_h
        )
        (diag_x1, diag_y1), (diag_x2, diag_y2) = diag

        width = 1.0 / bins
        agg: dict[int, list[int]] = {}
        for p, hit in pairs:
            idx = min(bins - 1, int(p / width))
            cell = agg.setdefault(idx, [0, 0])
            cell[0] += 1
            cell[1] += hit

        kept = [
            (idx, n, (idx + 0.5) * width, hits / n)
            for idx, (n, hits) in sorted(agg.items())
            if n >= min_n
        ]
        coords, _, _ = _scatter_geometry(
            [(mid, rate) for _, _, mid, rate in kept],
            x_lo=0.0,
            x_hi=1.0,
            y_lo=0.0,
            y_hi=1.0,
            view_w=view_w,
            view_h=view_h,
        )
        rel_bins: list[ReliabilityBin] = []
        for (idx, n, mid, rate), (cx, cy) in zip(kept, coords):
            lo_pct = round(idx * width * 100)
            hi_pct = round((idx + 1) * width * 100)
            rel_bins.append(
                ReliabilityBin(
                    label=f"{lo_pct}-{hi_pct}%",
                    p_mid=round(mid, 4),
                    hit_rate=round(rate, 4),
                    n=n,
                    cx=cx,
                    cy=cy,
                )
            )

        warnings: list[OperatorWarningModel] = []
        if not rel_bins:
            warnings.append(
                OperatorWarningModel(
                    code="insufficient_graded_outcomes",
                    severity="info",
                    message=(
                        f"not enough graded bets to chart calibration "
                        f"(need >= {min_n} per probability bucket)"
                    ),
                )
            )
        return ReliabilityDiagram(
            view_w=view_w,
            view_h=view_h,
            diag_x1=diag_x1,
            diag_y1=diag_y1,
            diag_x2=diag_x2,
            diag_y2=diag_y2,
            bins=rel_bins,
            n_pairs=len(pairs),
            n_plotted=len(rel_bins),
            min_n=min_n,
            filters={"league": league},
            warnings=warnings,
        )

    # -- data-quality heatmap (V2: coverage by league) -------------------

    def data_quality(self, *, league: str | None = None) -> QualityHeatmap:
        """Per-league data-coverage heatmap: evidence / closing-line / outcome
        attachment rates over a bounded scan of recent traces. Counts only — green
        is real coverage, never an assumed default. Read-only (two batch reads)."""
        league = _clean(league)
        traces = self.store.query_traces(league=league, limit=self.max_scan)
        scan_capped = len(traces) >= self.max_scan
        tids = [str(t.get("trace_id") or "") for t in traces if t.get("trace_id")]
        facts = self.store.get_session_trace_facts_batch(tids) if tids else {}
        closes = self.store.get_closing_lines_batch(tids) if tids else {}

        agg: dict[str, dict[str, int]] = {}
        for t in traces:
            tid = str(t.get("trace_id") or "")
            lg = t.get("league") or "—"
            a = agg.setdefault(lg, {"traces": 0, "evidence": 0, "closing": 0, "outcome": 0})
            a["traces"] += 1
            f = facts.get(tid, {})
            if int(f.get("evidence_signal_count") or 0) > 0:
                a["evidence"] += 1
            if bool(f.get("has_outcome")):
                a["outcome"] += 1
            if closes.get(tid):
                a["closing"] += 1

        rows: list[QualityRow] = []
        for lg in sorted(agg):
            a = agg[lg]
            n = a["traces"]
            cells: list[QualityCell] = []
            for key, num in (
                ("evidence", a["evidence"]),
                ("closing_line", a["closing"]),
                ("outcome", a["outcome"]),
            ):
                ratio = (num / n) if n else None
                cells.append(
                    QualityCell(
                        key=key,
                        ratio=ratio,
                        pct_display=(f"{ratio * 100:.0f}%" if ratio is not None else "—"),
                        count=num,
                        total=n,
                        tone=_quality_tone(ratio),
                    )
                )
            rows.append(QualityRow(league=lg, traces=n, cells=cells))

        warnings: list[OperatorWarningModel] = []
        if scan_capped:
            warnings.append(
                OperatorWarningModel(
                    code="scan_capped",
                    severity="info",
                    message=f"trace scan capped at {self.max_scan}; older traces not included",
                )
            )
        if not rows:
            warnings.append(
                OperatorWarningModel(
                    code="no_traces",
                    severity="info",
                    message="no traces found to assess data coverage",
                )
            )
        return QualityHeatmap(
            columns=["evidence", "closing line", "outcome"],
            rows=rows,
            traces_scanned=len(traces),
            scan_capped=scan_capped,
            filters={"league": league},
            warnings=warnings,
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
            raw = self.store.query_traces(league=league, start=date_from, limit=self.max_scan)
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
        recommendation_view = TraceRecommendationViewModel.model_validate(dataclasses.asdict(view))
        # Decision-quality insight views (B.4) — read-only, composed from the rows
        # already loaded above. dataclass -> asdict -> model_validate, same as B.1.
        primary_rec = next(
            (r for r in view.recommendations if r.is_primary),
            view.recommendations[0] if view.recommendations else None,
        )
        # Build the insight dataclasses first so the composers (trust breakdown,
        # guardrails) can consume them, then convert each to its Pydantic mirror.
        evidence_audit_dc = build_evidence_audit(
            trace=trace,
            evidence_signals=evidence,
            outcome=outcome,
            prop_outcomes=prop_outcomes,
            closing_lines=closing_lines,
            qa_verdict=qa_verdict,
        )
        market_movement_dc = build_market_movement(rec=primary_rec, closing_lines=closing_lines)
        signal_conflict_dc = build_signal_conflict(
            rec=primary_rec,
            evidence_signals=evidence,
            evidence_application=trace.get("evidence_application"),
            market_movement=market_movement_dc,
        )
        trust_breakdown_dc = build_trust_breakdown(
            trace_quality=trace.get("trace_quality") or {},
            rec=primary_rec,
            evidence_audit=evidence_audit_dc,
            market_movement=market_movement_dc,
            signal_conflict=signal_conflict_dc,
        )
        evidence_audit = EvidenceAuditViewModel.model_validate(dataclasses.asdict(evidence_audit_dc))
        market_movement = MarketMovementViewModel.model_validate(
            dataclasses.asdict(market_movement_dc)
        )
        signal_conflict = SignalConflictViewModel.model_validate(
            dataclasses.asdict(signal_conflict_dc)
        )
        trust_breakdown = TrustBreakdownViewModel.model_validate(
            dataclasses.asdict(trust_breakdown_dc)
        )
        guardrails = GuardrailsViewModel.model_validate(
            dataclasses.asdict(
                build_trace_guardrails(
                    trace_quality=trace.get("trace_quality") or {},
                    rec=primary_rec,
                    evidence_audit=evidence_audit_dc,
                    market_movement=market_movement_dc,
                    signal_conflict=signal_conflict_dc,
                    trust_breakdown=trust_breakdown_dc,
                    odds_age_seconds=_odds_age_seconds(trace),
                )
            )
        )
        primary_strip = None
        if recommendation_view.recommendations:
            prec = next(
                (r for r in recommendation_view.recommendations if r.is_primary),
                recommendation_view.recommendations[0],
            )
            primary_strip = _model_vs_market_strip(prec)
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
            evidence_audit=evidence_audit,
            market_movement=market_movement,
            signal_conflict=signal_conflict,
            trust_breakdown=trust_breakdown,
            guardrails=guardrails,
            primary_strip=primary_strip,
            field_sources={
                "recommendations": Source.DB_TRACE_PAYLOAD,
                "predictions": Source.DB_TRACE_PAYLOAD,
                "evidence_audit": Source.DB_TRACE_PAYLOAD,
                "market_movement": Source.CLOSING_LINES,
                "signal_conflict": Source.EVIDENCE_SIGNALS,
                "trust_breakdown": Source.DB_TRACE_PAYLOAD,
                "guardrails": Source.DB_TRACE_PAYLOAD,
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

    def similar_spots(self, trace_id: str) -> SimilarSpotsView | None:
        """How comparable historical spots actually performed (read-only, B.4).

        Derives a similarity key from the target trace (league, market family,
        edge bucket, dominant signal types) and groups *settled* bets that share
        that structure into cohorts, reporting each cohort's realized hit rate.
        Realized results come from ``bet_ledger.status`` (won/lost/push) — the
        honest, already-graded signal — never from re-grading a model number.
        Bounded by ``max_scan`` bets and ``SIMILAR_MAX_TRACES`` distinct reads.
        """
        target = self.store.get_trace(trace_id)
        if target is None:
            return None

        league = target.get("league")
        kind = target.get("kind")
        t_ev = self.store.get_evidence_signals(trace_id)
        t_view = build_trace_recommendation_view(target, evidence_signals=t_ev)
        t_primary = next(
            (r for r in t_view.recommendations if r.is_primary),
            t_view.recommendations[0] if t_view.recommendations else None,
        )
        market_family = _market_family(t_primary.market.value) if t_primary else "unknown"
        edge_bucket = _edge_bucket(t_primary.engine_edge.value) if t_primary else None
        target_signals = {str(r.get("signal_type")) for r in t_ev if r.get("signal_type")}

        bets = self.store.query_ledger(league=league, limit=self.max_scan)
        scan_capped = len(bets) >= self.max_scan
        settled = [b for b in bets if str(b.get("status") or "").lower() in ("won", "lost", "push")]

        acc = {
            code: {"wins": 0, "losses": 0, "pushes": 0}
            for code in ("structural", "signal_overlap", "edge_bucket", "league_kind")
        }
        dims_cache: dict[str, dict[str, Any] | None] = {}
        candidate_trace_capped = False
        for bet in settled:
            tid = bet.get("trace_id")
            if not tid or tid == trace_id:
                continue
            if tid not in dims_cache:
                if len(dims_cache) >= SIMILAR_MAX_TRACES:
                    candidate_trace_capped = True
                    continue
                dims_cache[tid] = self._candidate_dims(tid, bet.get("market"))
            dims = dims_cache[tid]
            if dims is None:
                continue
            status = str(bet.get("status") or "").lower()
            key = "wins" if status == "won" else "losses" if status == "lost" else "pushes"
            if (
                dims["league"] == league
                and dims["market_family"] == market_family
                and dims["edge_bucket"] == edge_bucket
            ):
                acc["structural"][key] += 1
            if dims["market_family"] == market_family and (dims["signal_types"] & target_signals):
                acc["signal_overlap"][key] += 1
            if edge_bucket is not None and dims["edge_bucket"] == edge_bucket:
                acc["edge_bucket"][key] += 1
            if dims["league"] == league and dims["kind"] == kind:
                acc["league_kind"][key] += 1

        labels = {
            "structural": f"{league or '—'} · {market_family} · {edge_bucket or 'any edge'}",
            "signal_overlap": f"{market_family} sharing ≥1 of this trace's signals",
            "edge_bucket": f"{edge_bucket or 'any'} edge (any market)",
            "league_kind": f"{league or '—'} {kind or ''} spots".strip(),
        }
        cohorts: list[SimilarCohort] = []
        for code in ("structural", "signal_overlap", "edge_bucket", "league_kind"):
            a = acc[code]
            decided = a["wins"] + a["losses"]
            cohorts.append(
                SimilarCohort(
                    code=code,
                    label=labels[code],
                    sample=decided + a["pushes"],
                    wins=a["wins"],
                    losses=a["losses"],
                    pushes=a["pushes"],
                    hit_rate=(a["wins"] / decided) if decided > 0 else None,
                    thin_sample=decided < SIMILAR_MIN_SAMPLE,
                )
            )

        def _verdict(c: SimilarCohort) -> str | None:
            decided = c.wins + c.losses
            if c.hit_rate is None or decided < SIMILAR_MIN_SAMPLE:
                return None
            if c.hit_rate >= 0.54:
                return "strong"
            if c.hit_rate <= 0.47:
                return "weak"
            return "mixed"

        historical_support = _verdict(cohorts[0]) or _verdict(cohorts[1]) or "insufficient"
        available = any(c.sample > 0 for c in cohorts)

        warnings: list[OperatorWarningModel] = []
        if not available:
            warnings.append(
                OperatorWarningModel(
                    code="no_comparable_history",
                    severity="info",
                    message="no settled bets with comparable structure found yet",
                )
            )
        elif historical_support == "insufficient":
            warnings.append(
                OperatorWarningModel(
                    code="thin_history",
                    severity="info",
                    message="comparable history is too thin for a verdict",
                )
            )
        if scan_capped:
            warnings.append(
                OperatorWarningModel(
                    code="scan_capped",
                    severity="info",
                    message=f"bet scan capped at {self.max_scan}; older bets not included",
                )
            )
        if candidate_trace_capped:
            warnings.append(
                OperatorWarningModel(
                    code="candidate_trace_cap",
                    severity="info",
                    message=(
                        f"candidate trace reads capped at {SIMILAR_MAX_TRACES}; "
                        "some comparable bets were skipped"
                    ),
                )
            )

        return SimilarSpotsView(
            trace_id=trace_id,
            available=available,
            league=league,
            kind=kind,
            market_family=market_family,
            edge_bucket=edge_bucket,
            signal_types=sorted(target_signals),
            historical_support=historical_support,
            cohorts=cohorts,
            settled_bets_scanned=len(settled),
            scan_capped=scan_capped or candidate_trace_capped,
            warnings=warnings,
            field_sources={
                "cohorts": Source.BET_LEDGER,
                "signal_types": Source.EVIDENCE_SIGNALS,
                "edge_bucket": Source.DB_TRACE_PAYLOAD,
            },
        )

    def _candidate_dims(self, trace_id: str, bet_market: Any) -> dict[str, Any] | None:
        """Similarity dimensions for one candidate trace (read-only)."""
        ct = self.store.get_trace(trace_id)
        if ct is None:
            return None
        cev = self.store.get_evidence_signals(trace_id)
        cview = build_trace_recommendation_view(ct, evidence_signals=cev)
        cprimary = next(
            (r for r in cview.recommendations if r.is_primary),
            cview.recommendations[0] if cview.recommendations else None,
        )
        return {
            "league": ct.get("league"),
            "kind": ct.get("kind"),
            "market_family": (
                _market_family(cprimary.market.value) if cprimary else _market_family(bet_market)
            ),
            "edge_bucket": _edge_bucket(cprimary.engine_edge.value) if cprimary else None,
            "signal_types": {str(r.get("signal_type")) for r in cev if r.get("signal_type")},
        }

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
            guardrail=_row_guardrail(trace),
            field_sources={
                "aggregate_quality": Source.DB_TRACE_PAYLOAD,
                "confidence_tiers": Source.DB_TRACE_PAYLOAD,
                "markets": Source.DB_TRACE_PAYLOAD,
                "has_outcome": outcome_source,
                "guardrail": Source.DB_TRACE_PAYLOAD,
            },
        )

    def get_trace_evidence_counts(self, trace_ids: list[str]) -> dict[str, EvidenceCoverageSummary]:
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
        if not trace_ids:
            return []
        batch_facts = self.store.get_session_trace_facts_batch(trace_ids)
        facts: list[SessionTraceFacts] = []
        for tid in trace_ids:
            row = batch_facts.get(tid)
            if row:
                facts.append(
                    SessionTraceFacts(
                        trace_id=tid,
                        evidence_signal_count=row["evidence_signal_count"],
                        has_outcome=bool(row["has_outcome"]),
                        has_bet=bool(row["has_bet"]),
                    )
                )
            else:
                facts.append(
                    SessionTraceFacts(
                        trace_id=tid,
                        evidence_signal_count=0,
                        has_outcome=False,
                        has_bet=False,
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
        linked_strip: ComparisonStrip | None = None
        if linked_trace_id:
            linked = self.store.get_trace(linked_trace_id)
            if linked is not None:
                # Recommendation values (edge%, kelly, units, …) come ONLY from
                # the linked trace's DB payload — never assumed to live in
                # bet_ledger.
                linked_recs = linked.get("recommendations")
                view = build_trace_recommendation_view(linked)
                rvm = TraceRecommendationViewModel.model_validate(dataclasses.asdict(view))
                if rvm.recommendations:
                    prec = next(
                        (r for r in rvm.recommendations if r.is_primary), rvm.recommendations[0]
                    )
                    linked_strip = _model_vs_market_strip(prec)
        return BetDetail(
            ledger_id=ledger_id,
            trace_id=linked_trace_id,
            ledger=row,
            staking=staking,
            correlation_group=row.get("correlation_group"),
            linked_trace_id=linked_trace_id if linked_recs is not None else None,
            linked_trace_recommendations=linked_recs,
            linked_strip=linked_strip,
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
    backtests_dir: str | Path | None = None,
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
        backtests_dir=backtests_dir,
    )
