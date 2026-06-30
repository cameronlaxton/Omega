"""Read-only decision-quality insights for the operator console (Phase 8 B.4).

This module turns the trace fields the read service already loads into the six
operator-facing *decision-quality* views that make a trace's edge, weakness, and
history legible:

* :func:`build_evidence_audit`   — deterministic missing-evidence checklist (A1)
* :func:`build_market_movement`  — opener→taken→close line-movement read    (A2)
* :func:`build_signal_conflict`  — where model output and signals disagree   (A3)
* :func:`build_trust_breakdown`  — six-bucket decomposition of trust         (A5, composer)
* :func:`build_trace_guardrails` — severity-ranked auto risk flags           (A6, composer)

(The "similar historical spots" view (A4) needs cross-trace DB reads, so it
lives on :class:`omega.ui.service.ConsoleService` rather than here.)

Doctrine — identical to :mod:`omega.ui.normalizers`, enforced by the static
red-team guard in ``tests/ui/test_console_no_mutation_imports.py``:

* **Read-only / no I/O.** Pure functions of their keyword inputs. The module
  imports only :mod:`omega.ui.normalizers` (shared primitives) and the
  :class:`omega.ui.schemas.Source` provenance labels — no store, no MCP, no ops.
* **DB-sourced, no guessing.** Every value comes from a DB-backed trace payload
  or its linked rows. Absent inputs degrade gracefully (the view says "not
  recorded") rather than being fabricated — a real trace carries the full
  ``trace_quality`` block; an older/sparser one may not, and we never invent it.
* **The engine still owns the numbers.** These views *explain and challenge* the
  probability/edge/EV the deterministic engine produced. They never restate a
  protected betting number as their own, and never change the pick.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omega.ui.clv import closing_line_value
from omega.ui.normalizers import (
    SEVERITY_FAIL,
    SEVERITY_INFO,
    SEVERITY_WARN,
    NormalizedRecommendation,
    OperatorWarning,
    _as_fraction,
    _g,
    _is_number,
    _signal_is_applied,
)
from omega.ui.schemas import Source

__all__ = [
    "EvidenceAuditItem",
    "EvidenceAuditView",
    "Guardrail",
    "GuardrailsView",
    "MarketMovementView",
    "SignalConflictRow",
    "SignalConflictView",
    "TrustContribution",
    "TrustBucket",
    "TrustBreakdownView",
    "build_evidence_audit",
    "build_market_movement",
    "build_signal_conflict",
    "build_trust_breakdown",
    "build_trace_guardrails",
    "clv_interpretation",
    "STALE_ODDS_SECONDS",
]

# Mirrors ``omega.core.contracts.confidence.STALE_ODDS_SECONDS`` (1h) — kept local
# so this read-only module does not import the engine package.
STALE_ODDS_SECONDS = 60 * 60

# Stable, greppable quality-reason tokens mirrored from ``omega.trace.quality``
# (kept local so this read-only module does not import the engine package).
_REASON_BASELINE_CONTEXT = "baseline_context"
_REASON_MISSING_IDENTITY = "missing_identity"
_REASON_EMPTY_EVIDENCE_PROVIDED_CONTEXT = "empty_evidence_provided_context"
_REASON_STATIC_IDENTITY = "static_identity_calibration"
_REASON_HIGH_IMPUTATION = "high_imputation"
_REASON_NOT_CALIBRATION_ELIGIBLE = "not_calibration_eligible"
_REASON_QA_FAILED = "qa_failed"
_REASON_ZERO_EVIDENCE_EMPTY_CONTEXT = "zero_evidence_empty_context"

POLARITY_POSITIVE = "positive"
POLARITY_NEGATIVE = "negative"
POLARITY_NEUTRAL = "neutral"


# ---------------------------------------------------------------------------
# A1 — Missing Evidence Auditor (deterministic presence checks)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceAuditItem:
    """One presence check in the evidence audit: is this input grounded?"""

    key: str
    label: str
    present: bool
    source: str
    critical: bool
    impact: str | None = None


@dataclass(frozen=True)
class EvidenceAuditView:
    """Deterministic audit of which grounding inputs a trace actually carries.

    ``evidence_quality`` is ``good`` (all present), ``partial`` (a non-critical
    input missing), or ``poor`` (a critical input missing). It is a coverage
    verdict, not a strength score — present inputs are not judged for quality.
    """

    items: list[EvidenceAuditItem]
    evidence_quality: str  # good | partial | poor
    present_count: int
    total_count: int
    summary: str
    warnings: list[OperatorWarning] = field(default_factory=list)


# Evidence signal markers that indicate availability/injury context was supplied.
_INJURY_MARKERS = (
    "injury",
    "inactive",
    "questionable",
    "availability",
    "out_status",
    "minutes_restriction",
    "dnp",
    "doubtful",
    "probable",
    "lineup",
)


def _has_injury_context(evidence_signals: list[dict[str, Any]]) -> bool:
    """True when any evidence signal looks like availability/injury context.

    Best-effort and deterministic: scans the signal_type/category/stat_key/note
    text for known availability markers. Never raises on odd row shapes.
    """
    for row in evidence_signals:
        if not isinstance(row, dict):
            continue
        haystack = " ".join(
            str(row.get(k) or "")
            for k in ("signal_type", "category", "stat_key", "note")
        ).lower()
        if any(marker in haystack for marker in _INJURY_MARKERS):
            return True
    return False


def _signal_has_value(row: dict[str, Any]) -> bool:
    """A signal carries a value if ``value_json``/``value`` is non-empty."""
    if not isinstance(row, dict):
        return False
    for key in ("value_json", "value"):
        v = row.get(key)
        if v not in (None, "", {}, [], "null"):
            return True
    # A stated confidence is itself a (weak) value signal.
    return _is_number(row.get("confidence"))


def build_evidence_audit(
    *,
    trace: dict[str, Any],
    evidence_signals: list[dict[str, Any]] | None,
    outcome: dict[str, Any] | None,
    prop_outcomes: list[dict[str, Any]] | None,
    closing_lines: list[dict[str, Any]] | None,
    qa_verdict: dict[str, Any] | None,
) -> EvidenceAuditView:
    """Audit a trace's grounding inputs into a deterministic present/missing list.

    Pure presence checks over the rows the read service already loads — no LLM,
    no scoring. Critical inputs (model prediction, evidence, odds) drive the
    ``poor`` verdict and emit warnings; informational inputs (outcome, QA audit)
    only shade ``good``→``partial``.
    """
    trace = trace if isinstance(trace, dict) else {}
    evidence_signals = [r for r in (evidence_signals or []) if isinstance(r, dict)]
    prop_outcomes = prop_outcomes or []
    closing_lines = closing_lines or []

    tq = trace.get("trace_quality") if isinstance(trace.get("trace_quality"), dict) else {}
    input_snapshot = trace.get("input_snapshot") if isinstance(trace.get("input_snapshot"), dict) else {}
    kind = str(trace.get("kind") or "").strip().lower()

    has_odds = bool(trace.get("odds_snapshot")) or bool(_g(input_snapshot, "odds"))
    has_prediction = bool(trace.get("predictions"))
    has_signal_values = any(_signal_has_value(r) for r in evidence_signals)
    evidence_status = tq.get("evidence_status")
    has_evidence_blocks = (
        bool(_g(input_snapshot, "evidence"))
        or evidence_status in ("present", "recovered_predecision")
        or len(evidence_signals) > 0
    )
    has_outcome = outcome is not None or len(prop_outcomes) > 0
    has_market_context = len(closing_lines) > 0
    has_injury = _has_injury_context(evidence_signals)
    has_calibration = ("calibration_eligible" in tq) or bool(tq.get("calibration_path"))
    has_audit = qa_verdict is not None

    # A missing injury read only disqualifies a *player prop* (where availability
    # drives the projection); for a game line it is informational.
    injury_critical = kind == "prop"

    items = [
        EvidenceAuditItem(
            key="odds_snapshot",
            label="Odds snapshot",
            present=has_odds,
            source=Source.DB_TRACE_PAYLOAD,
            critical=True,
            impact=None if has_odds else "No recorded price; edge cannot be re-derived from this trace.",
        ),
        EvidenceAuditItem(
            key="model_prediction",
            label="Model prediction",
            present=has_prediction,
            source=Source.DB_TRACE_PAYLOAD,
            critical=True,
            impact=None if has_prediction else "No model probability persisted; nothing to ground a recommendation.",
        ),
        EvidenceAuditItem(
            key="evidence_blocks",
            label="Evidence blocks",
            present=has_evidence_blocks,
            source=Source.DB_TRACE_PAYLOAD,
            critical=True,
            impact=None if has_evidence_blocks else "No structured evidence supplied; the projection is context-blind.",
        ),
        EvidenceAuditItem(
            key="signal_values",
            label="Signal values",
            present=has_signal_values,
            source=Source.EVIDENCE_SIGNALS,
            critical=False,
            impact=None if has_signal_values else "Evidence present but carries no concrete values to weigh.",
        ),
        EvidenceAuditItem(
            key="injury_context",
            label="Injury / availability context",
            present=has_injury,
            source=Source.EVIDENCE_SIGNALS,
            critical=injury_critical,
            impact=(
                None
                if has_injury
                else (
                    "No availability signal for a player prop — minutes/role are unconfirmed."
                    if injury_critical
                    else "No availability signal supplied."
                )
            ),
        ),
        EvidenceAuditItem(
            key="market_context",
            label="Market context (closing line)",
            present=has_market_context,
            source=Source.CLOSING_LINES,
            critical=False,
            impact=None if has_market_context else "No closing line captured; CLV and market confirmation unavailable.",
        ),
        EvidenceAuditItem(
            key="calibration_status",
            label="Calibration status",
            present=has_calibration,
            source=Source.DB_TRACE_PAYLOAD,
            critical=False,
            impact=None if has_calibration else "Calibration eligibility not recorded for this trace.",
        ),
        EvidenceAuditItem(
            key="outcome",
            label="Graded outcome",
            present=has_outcome,
            source=Source.OUTCOMES if outcome is not None else Source.PROP_OUTCOMES,
            critical=False,
            impact=None if has_outcome else "Not graded yet — no realized result to learn from.",
        ),
        EvidenceAuditItem(
            key="trace_audit",
            label="QA audit verdict",
            present=has_audit,
            source=Source.TRACE_QA_VERDICTS,
            critical=False,
            impact=None if has_audit else "No QA verdict row; trace was not gate-audited.",
        ),
    ]

    present_count = sum(1 for it in items if it.present)
    total_count = len(items)
    missing_critical = [it for it in items if it.critical and not it.present]
    missing_other = [it for it in items if not it.critical and not it.present]

    if missing_critical:
        evidence_quality = "poor"
    elif missing_other:
        evidence_quality = "partial"
    else:
        evidence_quality = "good"

    warnings: list[OperatorWarning] = []
    # The two truly disqualifying gaps fail; other critical gaps warn.
    _fail_keys = {"model_prediction", "evidence_blocks"}
    for it in missing_critical:
        warnings.append(
            OperatorWarning(
                code=f"missing_{it.key}",
                severity=SEVERITY_FAIL if it.key in _fail_keys else SEVERITY_WARN,
                message=it.impact or f"missing: {it.label}",
                source_path=None,
            )
        )

    if evidence_quality == "poor":
        summary = (
            "Poor — a critical grounding input is missing; treat as exploratory, "
            "not a high-confidence candidate."
        )
    elif evidence_quality == "partial":
        summary = "Partial — core inputs present; some supporting context is missing."
    else:
        summary = "Good — all grounding inputs are present."

    return EvidenceAuditView(
        items=items,
        evidence_quality=evidence_quality,
        present_count=present_count,
        total_count=total_count,
        summary=summary,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# A2 — Market Movement Explainer (reuses the read-only CLV arithmetic)
# ---------------------------------------------------------------------------

# Thresholds (implied-probability points / fractional edge) for reading movement.
_CLV_EPS = 0.005
_EDGE_EPS = 0.005
_EDGE_EPS_BIG = 0.03

_MOVEMENT_HEADLINES = {
    "early_value": "Market moved toward Omega's side and a meaningful edge remains at the close - early value.",
    "market_confirms": "Market moved toward Omega's side; some edge remains at the close.",
    "value_absorbed": "Market moved toward Omega's side, but most of the value has been absorbed by the close.",
    "market_disagrees": "Market moved against Omega's side - the close is shorter on the other side.",
    "no_confirmation": "Market has not meaningfully confirmed or contradicted this position.",
    "insufficient_data": "Not enough market data (no closing line or recorded price) to read movement.",
}


@dataclass(frozen=True)
class MarketMovementView:
    """Opener-to-taken-to-close line-movement read for the primary recommendation."""

    available: bool
    market: str | None
    selection: str | None
    taken_line: float | None
    taken_odds: float | None
    closing_line: float | None
    closing_odds: float | None
    taken_implied: float | None
    closing_implied: float | None
    model_probability: float | None
    point_delta: float | None
    price_delta: float | None
    clv_points: float | None  # closing_implied - taken_implied (>0 = beat the close)
    residual_edge: float | None  # model_probability - closing_implied (edge left at close)
    direction: str  # toward | against | flat | unknown
    interpretation: str
    headline: str
    closing_source: str | None = None
    warnings: list[OperatorWarning] = field(default_factory=list)


def clv_interpretation(clv_points: float | None) -> str:
    """Coarse, price-only movement token from a single CLV value (for the CLV page).

    Side-agnostic: positive CLV means the close priced the selection shorter than
    the taken price (you beat the close). No model edge is consulted here.
    """
    if clv_points is None:
        return "no_confirmation"
    if clv_points > _CLV_EPS:
        return "market_confirms"
    if clv_points < -_CLV_EPS:
        return "market_disagrees"
    return "no_confirmation"


def _match_closing_line(
    closes: list[dict[str, Any]], market: str | None, selection: str | None
) -> dict[str, Any] | None:
    """Best closing line for a recommendation: selection/market match, else first."""
    rows = [c for c in closes if isinstance(c, dict)]
    if not rows:
        return None
    sel = (selection or "").strip().lower()
    if sel:
        for c in rows:
            desc = str(c.get("selection_descriptor") or "").strip().lower()
            if desc and sel in desc:
                return c
    mkt = (market or "").strip().lower()
    if mkt:
        for c in rows:
            if str(c.get("market") or "").strip().lower() == mkt:
                return c
    return rows[0]


def _market_family_token(market: Any) -> str:
    token = str(market or "").strip().lower()
    if "total" in token or token in {"over", "under", "o/u"}:
        return "total"
    if "spread" in token or "handicap" in token:
        return "spread"
    if "moneyline" in token or token in {"ml", "h2h"}:
        return "moneyline"
    return "unknown"


def _side_adjusted_point_delta(
    *, market: Any, selection: Any, point_delta: float | None
) -> float | None:
    """Return positive when point movement favored the recorded selection."""
    if point_delta is None:
        return None
    family = _market_family_token(market)
    if family == "total":
        side = _canon_side(selection)
        if side == "over":
            return point_delta
        if side == "under":
            return -point_delta
        return None
    if family == "spread":
        return -point_delta
    return None


def build_market_movement(
    *,
    rec: NormalizedRecommendation | None,
    closing_lines: list[dict[str, Any]] | None,
) -> MarketMovementView:
    """Read line movement for the primary recommendation against the captured close.

    Reuses :func:`omega.ui.clv.closing_line_value` for the implied-probability /
    CLV arithmetic (never re-derived here). The interpretation combines the price
    move (CLV) with the edge left at the close (model probability minus closing
    implied) into a single deterministic token — it never restates a protected
    edge as its own number.
    """
    closing_lines = closing_lines or []
    if rec is None:
        return MarketMovementView(
            available=False, market=None, selection=None, taken_line=None,
            taken_odds=None, closing_line=None, closing_odds=None, taken_implied=None,
            closing_implied=None, model_probability=None, point_delta=None,
            price_delta=None, clv_points=None, residual_edge=None, direction="unknown",
            interpretation="insufficient_data", headline=_MOVEMENT_HEADLINES["insufficient_data"],
            warnings=[OperatorWarning(
                code="no_recommendation", severity=SEVERITY_INFO,
                message="no recommendation on this trace to read movement for",
            )],
        )

    market = rec.market.value if rec.market.value is not None else None
    selection = rec.selection.value if rec.selection.value is not None else None
    taken_line = rec.line.value if _is_number(rec.line.value) else None
    taken_odds = rec.odds.value if _is_number(rec.odds.value) else None
    model_probability = _as_fraction(
        rec.calibrated_probability.value
        if rec.calibrated_probability.value is not None
        else rec.raw_probability.value
    )

    match = _match_closing_line(closing_lines, str(market) if market is not None else None,
                                str(selection) if selection is not None else None)
    closing_line = closing_odds = closing_source = None
    if match is not None:
        cl = match.get("closing_line")
        co = match.get("closing_odds")
        closing_line = float(cl) if _is_number(cl) else None
        closing_odds = float(co) if _is_number(co) else None
        closing_source = match.get("source")

    clv = closing_line_value(taken_odds, closing_odds)
    taken_implied = clv.taken_implied
    closing_implied = clv.closing_implied
    clv_points = clv.clv_points

    point_delta = (
        round(closing_line - float(taken_line), 4)
        if closing_line is not None and taken_line is not None
        else None
    )
    price_delta = (
        round(float(closing_odds) - float(taken_odds), 2)
        if closing_odds is not None and taken_odds is not None
        else None
    )
    residual_edge = (
        round(model_probability - closing_implied, 4)
        if model_probability is not None and closing_implied is not None
        else None
    )
    point_signal = _side_adjusted_point_delta(
        market=market, selection=selection, point_delta=point_delta
    )

    available = clv_points is not None or point_signal is not None
    warnings: list[OperatorWarning] = []
    if match is None:
        warnings.append(OperatorWarning(
            code="no_closing_line", severity=SEVERITY_INFO,
            message="no closing line captured for this trace yet — CLV unavailable",
        ))
    elif taken_odds is None:
        warnings.append(OperatorWarning(
            code="no_recorded_price", severity=SEVERITY_INFO,
            message="no recorded American price on the recommendation — CLV unavailable",
        ))

    if not available:
        direction = "unknown"
        interpretation = "insufficient_data"
    elif clv_points is not None and clv_points > _CLV_EPS:
        direction = "toward"
        if residual_edge is not None and residual_edge > _EDGE_EPS_BIG:
            interpretation = "early_value"
        elif residual_edge is not None and residual_edge > _EDGE_EPS:
            interpretation = "market_confirms"
        else:
            interpretation = "value_absorbed"
    elif clv_points is not None and clv_points < -_CLV_EPS:
        direction = "against"
        interpretation = "market_disagrees"
    elif point_signal is not None and point_signal > _CLV_EPS:
        direction = "toward"
        if residual_edge is not None and residual_edge > _EDGE_EPS_BIG:
            interpretation = "early_value"
        elif residual_edge is not None and residual_edge > _EDGE_EPS:
            interpretation = "market_confirms"
        else:
            interpretation = "value_absorbed"
    elif point_signal is not None and point_signal < -_CLV_EPS:
        direction = "against"
        interpretation = "market_disagrees"
    else:
        direction = "flat"
        interpretation = "no_confirmation"

    return MarketMovementView(
        available=available,
        market=str(market) if market is not None else None,
        selection=str(selection) if selection is not None else None,
        taken_line=taken_line if _is_number(taken_line) else None,
        taken_odds=taken_odds if _is_number(taken_odds) else None,
        closing_line=closing_line,
        closing_odds=closing_odds,
        taken_implied=taken_implied,
        closing_implied=closing_implied,
        model_probability=None,
        point_delta=point_delta,
        price_delta=price_delta,
        clv_points=clv_points,
        residual_edge=None,
        direction=direction,
        interpretation=interpretation,
        headline=_MOVEMENT_HEADLINES[interpretation],
        closing_source=closing_source,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# A3 — Signal Conflict Detector
# ---------------------------------------------------------------------------

_OPPOSITE_SIDE = {"over": "under", "under": "over", "home": "away", "away": "home"}

_CONFLICT_HEADLINES = {
    "market_conflict": "Model likes it, but the market moved against the position.",
    "signal_disagreement": "Signals disagree — meaningful weight is pulling against the position.",
    "model_edge_conflict": "Model shows an edge, but the supporting evidence is thin.",
    "correlation_conflict": "Multiple signals may be double-counting the same factor.",
    "dominant_single_signal": "One signal is carrying the projection with little corroboration.",
}

# Conflicts in salience order (first present becomes the dominant conflict).
_CONFLICT_PRIORITY = (
    "market_conflict",
    "signal_disagreement",
    "model_edge_conflict",
    "correlation_conflict",
    "dominant_single_signal",
)


@dataclass(frozen=True)
class SignalConflictRow:
    """One evidence signal classified relative to the recommendation's side."""

    signal_type: str | None
    direction: str | None
    confidence: float | None
    applied: bool
    stance: str  # supports | opposes | neutral
    family_role: str | None


@dataclass(frozen=True)
class SignalConflictView:
    """Where the model output and its internal evidence pull apart.

    ``conflict_level`` is ``low`` (no conflict), ``medium`` (one), or ``high``
    (two or more, or one strong). ``dominant_conflict`` names the most salient.
    """

    conflict_level: str
    dominant_conflict: str | None
    headline: str
    conflicts: list[str]
    supporting_count: int
    opposing_count: int
    neutral_count: int
    applied_count: int
    rows: list[SignalConflictRow]
    warnings: list[OperatorWarning] = field(default_factory=list)


def _canon_side(value: Any) -> str | None:
    """Canonicalize a side/direction token to over/under/home/away, else None."""
    t = str(value or "").strip().lower()
    if t in ("over", "o"):
        return "over"
    if t in ("under", "u"):
        return "under"
    if t in ("home", "home_team"):
        return "home"
    if t in ("away", "away_team"):
        return "away"
    return None


def _recommendation_side(rec: NormalizedRecommendation | None) -> str | None:
    if rec is None:
        return None
    side = _canon_side(getattr(rec, "selection_kind", None))
    if side is not None:
        return side
    side = _canon_side(rec.selection.value)
    if side is not None:
        return side
    for prob_field in (rec.raw_probability, rec.calibrated_probability, rec.implied_probability):
        path = str(prob_field.source_path or "").lower()
        if "home_win_prob" in path:
            return "home"
        if "away_win_prob" in path:
            return "away"
    return None


def _family_roles(evidence_application: list[dict[str, Any]] | None) -> dict[str, str]:
    """Map signal_type -> family_role from the Issue-22 evidence_application block."""
    out: dict[str, str] = {}
    for app in evidence_application or []:
        if not isinstance(app, dict):
            continue
        st = app.get("signal_type")
        role = app.get("family_role")
        if st is not None and role:
            out[str(st)] = str(role)
    return out


def build_signal_conflict(
    *,
    rec: NormalizedRecommendation | None,
    evidence_signals: list[dict[str, Any]] | None,
    evidence_application: list[dict[str, Any]] | None,
    market_movement: MarketMovementView | None,
) -> SignalConflictView:
    """Find traces where the model output looks strong but the evidence is messy.

    Pure classification over the signals/applications already loaded. Detects the
    file's conflict families (signal disagreement, model-edge, market, correlation,
    single-signal dominance) and rolls them into a level + dominant conflict. It
    never restates the engine's edge as its own — it only flags disagreement.
    """
    rows_in = [r for r in (evidence_signals or []) if isinstance(r, dict)]
    roles = _family_roles(evidence_application)

    rec_side = _recommendation_side(rec)
    edge_val = rec.engine_edge.value if rec is not None else None
    edge_positive = _is_number(edge_val) and float(edge_val) > 0

    rows: list[SignalConflictRow] = []
    weights: dict[str, float] = {}
    supporting = opposing = neutral = applied_count = 0
    applied_supporting = applied_opposing = 0
    for r in rows_in:
        direction = _canon_side(r.get("direction"))
        conf = float(r["confidence"]) if _is_number(r.get("confidence")) else None
        applied = _signal_is_applied(r)
        if applied:
            applied_count += 1
        weight = conf if conf is not None else 0.5
        if direction is not None and applied:
            weights[direction] = weights.get(direction, 0.0) + weight

        if rec_side and direction == rec_side:
            stance = "supports"
            supporting += 1
            if applied:
                applied_supporting += 1
        elif rec_side and _OPPOSITE_SIDE.get(rec_side) == direction:
            stance = "opposes"
            opposing += 1
            if applied:
                applied_opposing += 1
        else:
            stance = "neutral"
            neutral += 1
        rows.append(
            SignalConflictRow(
                signal_type=(str(r.get("signal_type")) if r.get("signal_type") is not None else None),
                direction=direction,
                confidence=conf,
                applied=applied,
                stance=stance,
                family_role=roles.get(str(r.get("signal_type"))),
            )
        )

    # Weighted pro/con along the relevant axis.
    if rec_side:
        pro_weight = weights.get(rec_side, 0.0)
        con_weight = weights.get(_OPPOSITE_SIDE.get(rec_side, ""), 0.0)
    else:
        # No anchor: take the dominant opposing pair (over/under or home/away).
        pro_weight = con_weight = 0.0
        for a, b in (("over", "under"), ("home", "away")):
            wa, wb = weights.get(a, 0.0), weights.get(b, 0.0)
            if wa + wb > pro_weight + con_weight:
                pro_weight, con_weight = max(wa, wb), min(wa, wb)

    conflicts: list[str] = []
    warnings: list[OperatorWarning] = []
    strong = False

    # Signal disagreement: real weight pulling the other way.
    if con_weight > 0 and con_weight >= 0.5 * max(pro_weight, 1e-9):
        conflicts.append("signal_disagreement")
        if con_weight >= pro_weight:
            strong = True
        warnings.append(OperatorWarning(
            code="signal_disagreement",
            severity=SEVERITY_FAIL if con_weight >= pro_weight else SEVERITY_WARN,
            message=_CONFLICT_HEADLINES["signal_disagreement"],
        ))

    # Market conflict: positive model edge but the close moved against us.
    if market_movement is not None and market_movement.direction == "against" and edge_positive:
        conflicts.append("market_conflict")
        strong = True
        warnings.append(OperatorWarning(
            code="market_conflict", severity=SEVERITY_WARN,
            message=_CONFLICT_HEADLINES["market_conflict"],
        ))

    # Model-edge conflict: an edge with thin/weak supporting evidence.
    if edge_positive and (applied_count <= 1 or pro_weight < 0.6):
        conflicts.append("model_edge_conflict")
        warnings.append(OperatorWarning(
            code="model_edge_conflict", severity=SEVERITY_WARN,
            message=_CONFLICT_HEADLINES["model_edge_conflict"],
        ))

    # Correlation conflict: damping fired (signals share a cause).
    if any(role in ("secondary", "damped") for role in roles.values()):
        conflicts.append("correlation_conflict")
        warnings.append(OperatorWarning(
            code="correlation_conflict", severity=SEVERITY_INFO,
            message=_CONFLICT_HEADLINES["correlation_conflict"],
        ))

    # Single-signal dominance: exactly one applied supporting signal.
    if (
        rec_side
        and applied_supporting == 1
        and applied_count >= 1
        and applied_opposing == 0
        and "model_edge_conflict" not in conflicts
    ):
        conflicts.append("dominant_single_signal")
        warnings.append(OperatorWarning(
            code="dominant_single_signal", severity=SEVERITY_INFO,
            message=_CONFLICT_HEADLINES["dominant_single_signal"],
        ))

    if not conflicts:
        level = "low"
    elif strong or len(conflicts) >= 2:
        level = "high"
    else:
        level = "medium"

    dominant = next((c for c in _CONFLICT_PRIORITY if c in conflicts), None)
    headline = (
        _CONFLICT_HEADLINES[dominant]
        if dominant
        else "No material conflict between the model output and its evidence."
    )

    return SignalConflictView(
        conflict_level=level,
        dominant_conflict=dominant,
        headline=headline,
        conflicts=conflicts,
        supporting_count=supporting,
        opposing_count=opposing,
        neutral_count=neutral,
        applied_count=applied_count,
        rows=rows,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# A5 — Trust Breakdown / Confidence Decomposer (composer over A1–A3 + quality)
# ---------------------------------------------------------------------------

_BAND_LABELS = {
    "strong": "high trust",
    "usable": "medium trust",
    "weak": "low trust",
    "invalid": "not trustworthy",
}


@dataclass(frozen=True)
class TrustContribution:
    """One +/- factor inside a trust bucket."""

    text: str
    polarity: str  # positive | negative | neutral


@dataclass(frozen=True)
class TrustBucket:
    """One of the six trust dimensions, with its net polarity and factors."""

    name: str
    summary: str
    polarity: str
    contributions: list[TrustContribution]


@dataclass(frozen=True)
class TrustBreakdownView:
    """Why a trace is trusted the way it is - six buckets + a +/- digest.

    Reconstructs the persisted ``trace_quality`` verdict and composes the A1–A3
    views into the file's six buckets. It restates no protected number: edge is
    described as "meaningful/small", never re-quoted as the engine's figure.
    """

    aggregate_quality: float | None
    quality_band: str | None
    confidence_cap: str | None
    trace_weight: float | None
    headline: str
    buckets: list[TrustBucket]
    positives: list[str]
    negatives: list[str]
    warnings: list[OperatorWarning] = field(default_factory=list)


def _bucket_polarity(contribs: list[TrustContribution]) -> str:
    pos = any(c.polarity == POLARITY_POSITIVE for c in contribs)
    neg = any(c.polarity == POLARITY_NEGATIVE for c in contribs)
    if pos and not neg:
        return POLARITY_POSITIVE
    if neg and not pos:
        return POLARITY_NEGATIVE
    return POLARITY_NEUTRAL


def _derive_band(quality_band: Any, aggregate: float | None) -> str | None:
    """Persisted band wins; else derive from a 0-1 or 0-100 aggregate score."""
    if isinstance(quality_band, str) and quality_band:
        return quality_band
    if aggregate is None:
        return None
    norm = aggregate / 100.0 if aggregate > 1.0 else aggregate
    if norm >= 0.75:
        return "strong"
    if norm >= 0.50:
        return "usable"
    if norm >= 0.20:
        return "weak"
    return "invalid"


def build_trust_breakdown(
    *,
    trace_quality: dict[str, Any] | None,
    rec: NormalizedRecommendation | None,
    evidence_audit: EvidenceAuditView,
    market_movement: MarketMovementView,
    signal_conflict: SignalConflictView,
) -> TrustBreakdownView:
    """Decompose trust into Model strength · Data quality · Signal agreement ·
    Market confirmation · Calibration support · Volatility."""
    tq = trace_quality if isinstance(trace_quality, dict) else {}
    reasons = tq.get("quality_reasons") if isinstance(tq.get("quality_reasons"), list) else []

    aggregate = None
    agg_raw = tq.get("aggregate_quality")
    if _is_number(agg_raw):
        aggregate = float(agg_raw)
    band = _derive_band(tq.get("quality_band"), aggregate)
    confidence_cap = tq.get("confidence_cap")
    trace_weight = float(tq["trace_weight"]) if _is_number(tq.get("trace_weight")) else None

    buckets: list[TrustBucket] = []

    # 1) Model strength — edge magnitude (described, never re-quoted).
    contribs: list[TrustContribution] = []
    edge_v = _as_fraction_or_pct(rec.engine_edge.value if rec is not None else None)
    if edge_v is None:
        contribs.append(TrustContribution("No model edge recorded.", POLARITY_NEUTRAL))
        ms_summary = "no edge"
    elif edge_v >= 3.0:
        contribs.append(TrustContribution("Model edge is meaningful.", POLARITY_POSITIVE))
        ms_summary = "meaningful edge"
    elif edge_v > 0:
        contribs.append(TrustContribution("Model edge is small.", POLARITY_NEUTRAL))
        ms_summary = "small edge"
    else:
        contribs.append(TrustContribution("No positive model edge.", POLARITY_NEGATIVE))
        ms_summary = "no positive edge"
    buckets.append(TrustBucket("Model strength", ms_summary, _bucket_polarity(contribs), contribs))

    # 2) Data quality — from the A1 evidence audit + provided-context reasons.
    contribs = []
    if evidence_audit.evidence_quality == "good":
        contribs.append(TrustContribution("All grounding inputs present.", POLARITY_POSITIVE))
    elif evidence_audit.evidence_quality == "partial":
        contribs.append(TrustContribution("Some supporting context is missing.", POLARITY_NEUTRAL))
    else:
        contribs.append(TrustContribution("A critical grounding input is missing.", POLARITY_NEGATIVE))
    if _REASON_EMPTY_EVIDENCE_PROVIDED_CONTEXT in reasons:
        contribs.append(TrustContribution("Evidence block is empty.", POLARITY_NEGATIVE))
    buckets.append(
        TrustBucket("Data quality", evidence_audit.evidence_quality, _bucket_polarity(contribs), contribs)
    )

    # 3) Signal agreement — from the A3 conflict view.
    contribs = []
    if signal_conflict.conflict_level == "low":
        contribs.append(TrustContribution("Signals agree with the pick.", POLARITY_POSITIVE))
        sa_summary = "aligned"
    elif signal_conflict.conflict_level == "medium":
        contribs.append(TrustContribution("Some signal tension.", POLARITY_NEUTRAL))
        sa_summary = "some tension"
    else:
        contribs.append(TrustContribution("Signals conflict with the pick.", POLARITY_NEGATIVE))
        sa_summary = "conflicted"
    if signal_conflict.opposing_count > 0:
        contribs.append(
            TrustContribution(
                f"{signal_conflict.opposing_count} signal(s) oppose the side.", POLARITY_NEGATIVE
            )
        )
    buckets.append(TrustBucket("Signal agreement", sa_summary, _bucket_polarity(contribs), contribs))

    # 4) Market confirmation — from the A2 movement view.
    contribs = []
    if market_movement.direction == "toward":
        contribs.append(TrustContribution("Market moved toward the pick.", POLARITY_POSITIVE))
        mc_summary = "confirming"
    elif market_movement.direction == "against":
        contribs.append(TrustContribution("Market moved against the pick.", POLARITY_NEGATIVE))
        mc_summary = "disagreeing"
    elif market_movement.direction == "flat":
        contribs.append(TrustContribution("Market is flat — no confirmation.", POLARITY_NEUTRAL))
        mc_summary = "flat"
    else:
        contribs.append(TrustContribution("No closing line to confirm against.", POLARITY_NEUTRAL))
        mc_summary = "no close"
    buckets.append(TrustBucket("Market confirmation", mc_summary, _bucket_polarity(contribs), contribs))

    # 5) Calibration support — from trace_quality calibration fields.
    contribs = []
    cal_path = tq.get("calibration_path")
    cal_eligible = tq.get("calibration_eligible")
    if cal_path == "profile" and cal_eligible:
        contribs.append(TrustContribution("Calibrated by a fitted profile.", POLARITY_POSITIVE))
        cs_summary = "profile"
    elif cal_path in ("base_profile_fallback", "static_calibrated"):
        contribs.append(TrustContribution("Calibrated via a fallback path.", POLARITY_NEUTRAL))
        cs_summary = "fallback"
    elif cal_path == "static_identity" or _REASON_STATIC_IDENTITY in reasons:
        contribs.append(TrustContribution("No real calibration profile applied.", POLARITY_NEGATIVE))
        cs_summary = "identity"
    elif cal_eligible is False or _REASON_NOT_CALIBRATION_ELIGIBLE in reasons:
        contribs.append(TrustContribution("Not calibration-eligible.", POLARITY_NEGATIVE))
        cs_summary = "ineligible"
    elif cal_eligible:
        contribs.append(TrustContribution("Calibration-eligible.", POLARITY_POSITIVE))
        cs_summary = "eligible"
    else:
        contribs.append(TrustContribution("Calibration status not recorded.", POLARITY_NEUTRAL))
        cs_summary = "unknown"
    buckets.append(TrustBucket("Calibration support", cs_summary, _bucket_polarity(contribs), contribs))

    # 6) Volatility — injury/role/imputation/identity uncertainty.
    contribs = []
    injury_item = next((it for it in evidence_audit.items if it.key == "injury_context"), None)
    if injury_item is not None and not injury_item.present:
        contribs.append(
            TrustContribution(
                "Availability/injury context is missing.",
                POLARITY_NEGATIVE if injury_item.critical else POLARITY_NEUTRAL,
            )
        )
    imp = tq.get("imputed_fraction")
    if _is_number(imp) and float(imp) > 0.2:
        contribs.append(TrustContribution("High imputed-data fraction.", POLARITY_NEGATIVE))
    if tq.get("identity_status") not in (None, "complete"):
        contribs.append(TrustContribution("Team/player identity is incomplete.", POLARITY_NEGATIVE))
    if not contribs:
        contribs.append(TrustContribution("No major volatility flags.", POLARITY_POSITIVE))
    buckets.append(TrustBucket("Volatility", "flags" if len(contribs) > 1 or contribs[0].polarity == POLARITY_NEGATIVE else "clear", _bucket_polarity(contribs), contribs))

    positives = [c.text for b in buckets for c in b.contributions if c.polarity == POLARITY_POSITIVE]
    negatives = [c.text for b in buckets for c in b.contributions if c.polarity == POLARITY_NEGATIVE]

    label = _BAND_LABELS.get(band or "", "unrated")
    headline = label + (f" — confidence capped at {confidence_cap}" if confidence_cap else "")

    warnings: list[OperatorWarning] = []
    if _REASON_QA_FAILED in reasons:
        warnings.append(OperatorWarning(
            code="qa_failed", severity=SEVERITY_FAIL,
            message="QA gate failed — trace is invalid and must not produce actionable output.",
        ))
    if _REASON_ZERO_EVIDENCE_EMPTY_CONTEXT in reasons:
        warnings.append(OperatorWarning(
            code="zero_evidence_empty_context", severity=SEVERITY_FAIL,
            message="Zero evidence and no provided context — reasoning blind.",
        ))

    return TrustBreakdownView(
        aggregate_quality=aggregate,
        quality_band=band,
        confidence_cap=confidence_cap,
        trace_weight=trace_weight,
        headline=headline,
        buckets=buckets,
        positives=positives,
        negatives=negatives,
        warnings=warnings,
    )


def _as_fraction_or_pct(value: Any) -> float | None:
    """Normalize an engine edge to percentage points (|v|<=1 => fraction x100)."""
    if not _is_number(value):
        return None
    v = float(value)
    return v * 100.0 if abs(v) <= 1.0 else v


# ---------------------------------------------------------------------------
# A6 — Trace Guardrails / Auto Risk Flags (capstone composer)
# ---------------------------------------------------------------------------

_SEV_RANK = {SEVERITY_INFO: 1, SEVERITY_WARN: 2, SEVERITY_FAIL: 3}

# Suggested operator action per guardrail code (deterministic copy).
_GUARDRAIL_ACTIONS = {
    "confidence_capped_pass": "Do not bet — resolve the cap reason before acting.",
    "qa_failed": "Discard — the trace failed its QA gate.",
    "zero_evidence_empty_context": "Provide game/player context and structured evidence before relying on this.",
    "stale_odds": "Re-check the current price before acting — the recorded price is over an hour old.",
    "high_imputation": "Treat cautiously — much of the input was imputed, not observed.",
    "static_identity_calibration": "Treat as uncalibrated — no real profile was applied.",
    "not_calibration_eligible": "Use for review only — this trace cannot calibrate.",
    "market_moved_against": "Reassess — the closing line moved against the position; value may be gone.",
    "high_signal_conflict": "Reconcile the conflicting signals before sizing.",
    "missing_model_prediction": "Discard — no model probability to act on.",
    "missing_evidence_blocks": "Treat as exploratory — the projection is context-blind.",
    "missing_injury_context": "Confirm availability/lineup before the prop locks.",
    "missing_odds_snapshot": "No recorded price — edge cannot be verified from this trace.",
}


@dataclass(frozen=True)
class Guardrail:
    """One operator-facing risk flag with severity and a suggested action."""

    code: str
    severity: str  # info | warn | fail (fail == Blocker)
    message: str
    suggested_action: str | None = None
    source: str | None = None


@dataclass(frozen=True)
class GuardrailsView:
    """Severity-ranked auto risk flags for a trace (the capstone safety layer)."""

    worst_severity: str  # ok | info | warn | fail
    blocker_count: int
    warning_count: int
    info_count: int
    guardrails: list[Guardrail]
    summary: str


def build_trace_guardrails(
    *,
    trace_quality: dict[str, Any] | None,
    rec: NormalizedRecommendation | None,
    evidence_audit: EvidenceAuditView,
    market_movement: MarketMovementView,
    signal_conflict: SignalConflictView,
    trust_breakdown: TrustBreakdownView,
    odds_age_seconds: float | None = None,
) -> GuardrailsView:
    """Aggregate the trust/evidence/market/signal checks into ranked risk flags.

    Deduplicates the warnings already produced by the A1–A5 views, adds the
    guardrail-specific checks (stale odds, line moved against, confidence cap),
    attaches a suggested action, and ranks by severity. ``fail`` is a Blocker.
    """
    tq = trace_quality if isinstance(trace_quality, dict) else {}
    reasons = tq.get("quality_reasons") if isinstance(tq.get("quality_reasons"), list) else []

    acc: dict[str, Guardrail] = {}

    def _add(code: str, severity: str, message: str, *, source: str) -> None:
        action = _GUARDRAIL_ACTIONS.get(code)
        cur = acc.get(code)
        if cur is None or _SEV_RANK[cur.severity] < _SEV_RANK[severity]:
            acc[code] = Guardrail(
                code=code, severity=severity, message=message,
                suggested_action=action or (cur.suggested_action if cur else None),
                source=source,
            )

    # --- Trace-quality guardrails. ---
    if tq.get("confidence_cap") == "Pass":
        _add("confidence_capped_pass", SEVERITY_FAIL,
             "Confidence is capped at Pass — no actionable output.", source="trace_quality")
    if _REASON_QA_FAILED in reasons:
        _add("qa_failed", SEVERITY_FAIL, "QA gate failed — the trace is invalid.", source="trace_quality")
    if _REASON_ZERO_EVIDENCE_EMPTY_CONTEXT in reasons:
        _add("zero_evidence_empty_context", SEVERITY_FAIL,
             "Zero evidence and no provided context — reasoning blind.", source="trace_quality")
    imp = tq.get("imputed_fraction")
    if _REASON_HIGH_IMPUTATION in reasons or (_is_number(imp) and float(imp) > 0.2):
        _add("high_imputation", SEVERITY_WARN, "High imputed-data fraction.", source="trace_quality")
    if tq.get("calibration_path") == "static_identity" or _REASON_STATIC_IDENTITY in reasons:
        _add("static_identity_calibration", SEVERITY_WARN,
             "No real calibration profile applied.", source="trace_quality")
    if tq.get("calibration_eligible") is False or _REASON_NOT_CALIBRATION_ELIGIBLE in reasons:
        _add("not_calibration_eligible", SEVERITY_INFO, "Not calibration-eligible.", source="trace_quality")

    # --- Stale recorded price. ---
    if _is_number(odds_age_seconds) and float(odds_age_seconds) > STALE_ODDS_SECONDS:
        hrs = float(odds_age_seconds) / 3600.0
        _add("stale_odds", SEVERITY_WARN,
             f"Recorded price is stale ({hrs:.1f}h old).", source="market")

    # --- Market moved against. ---
    if market_movement.direction == "against":
        _add("market_moved_against", SEVERITY_WARN,
             "The closing line moved against the position.", source="market")

    # --- Signal conflict. ---
    if signal_conflict.conflict_level == "high":
        dom = signal_conflict.dominant_conflict or "conflict"
        _add("high_signal_conflict", SEVERITY_WARN,
             f"High signal conflict ({dom.replace('_', ' ')}).", source="signal")

    # --- Fold in the A1 evidence-audit critical-gap warnings (fail/warn). ---
    for w in evidence_audit.warnings:
        _add(w.code, w.severity, w.message, source="evidence")
    # --- Fold in the A5 trust-breakdown fail warnings (qa/zero-evidence). ---
    for w in trust_breakdown.warnings:
        _add(w.code, w.severity, w.message, source="trust")

    guardrails = sorted(acc.values(), key=lambda g: (-_SEV_RANK[g.severity], g.code))
    blocker = sum(1 for g in guardrails if g.severity == SEVERITY_FAIL)
    warning = sum(1 for g in guardrails if g.severity == SEVERITY_WARN)
    info = sum(1 for g in guardrails if g.severity == SEVERITY_INFO)

    if blocker:
        worst, summary = SEVERITY_FAIL, f"{blocker} blocker(s) — do not treat as a betting candidate."
    elif warning:
        worst, summary = SEVERITY_WARN, f"{warning} warning(s) — review before acting."
    elif info:
        worst, summary = SEVERITY_INFO, "Minor notes only."
    else:
        worst, summary = "ok", "No guardrails triggered."

    return GuardrailsView(
        worst_severity=worst,
        blocker_count=blocker,
        warning_count=warning,
        info_count=info,
        guardrails=guardrails,
        summary=summary,
    )
