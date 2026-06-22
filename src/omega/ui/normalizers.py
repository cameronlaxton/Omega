"""Read-only normalization layer for the operator console (Phase 8 Milestone B.0).

Phase A surfaced trace recommendations and session health as raw, heterogeneous
JSON. This module turns those payloads into a *uniform, audited* read model that
B.1 can display without re-interpreting JSON by hand. It is the trustworthy
interpretation layer between the DB-backed read service and the polished cards.

Doctrine (non-negotiable, enforced field-by-field below):

* **Read-only.** Nothing here writes, mutates, promotes, settles, or ingests.
  The module imports only ``omega.ui.schemas`` (for the :class:`Source` labels)
  plus the standard library — no store, no MCP, no ops module. The static
  red-team guard in ``tests/ui/test_console_no_mutation_imports.py`` covers this
  file automatically.
* **DB-sourced only.** Every value is extracted from a DB-backed trace payload
  or the ``evidence_signals`` rows. Session sidecars are process/narrative only;
  :class:`SessionHealthView` consumes *counts and flags* passed in by the caller,
  never sidecar prose, and never parses a probability/EV/Kelly out of narrative.
* **No guessing.** A value that cannot be sourced stays ``None`` with
  ``source_path=None``. The only derived values are the two explicitly-labelled
  computed fields — implied probability (from confirmed American odds) and
  computed edge (calibrated probability minus implied probability) — each carries
  ``computed=True`` and a ``source_path`` beginning with ``"computed:"``.
* **Scalars only.** :attr:`ExtractedField.value` is always a scalar; a dict/list
  candidate is treated as absent so JSON soup can never leak into a display field.
* **Selection-aware.** Probabilities are mapped from the recommendation's
  *selection* (over/under/home/away/draw/…), never by grabbing the first
  probability in the payload.
* **No invented scores.** Evidence is *counted and categorised*; there is no
  "evidence strength" or "signal quality" score.

Two real-payload facts shape the extraction (verified against
``omega.core.contracts.schemas`` and ``omega.trace.persistable``):

1. **Probability scale.** Persisted game ``predictions`` is the ``simulation``
   block, whose ``*_win_prob`` values are on a **0–100** scale, whereas edge
   ``true_prob``/``calibrated_prob`` and prop ``over_prob``/``under_prob`` are
   **0–1**. For game side-selections we therefore prefer the edge-native
   ``true_prob`` (0–1, selection-correct) over the 0–100 simulation block, and
   any cross-value comparison/derivation coerces percentages to fractions via
   :func:`_as_fraction` so the math is scale-consistent.
2. **Calibration field name.** ``CalibrationAudit`` stores ``calibrated_prob``
   (the design draft wrote ``calibrated_probability``); both spellings are
   accepted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omega.ui.schemas import Source

__all__ = [
    "ScalarValue",
    "ExtractedField",
    "OperatorWarning",
    "NormalizedRecommendation",
    "TraceRecommendationView",
    "EvidenceCoverage",
    "SessionHealthView",
    "SessionTraceFacts",
    "VALID_SEVERITIES",
    "build_trace_recommendation_view",
    "build_evidence_coverage",
    "build_session_health_view",
    "normalize_recommendation",
    "probability_source_for_selection",
    "implied_probability_from_american",
    "computed_edge_value",
    "confidence_band",
]

# ---------------------------------------------------------------------------
# Core data contracts
# ---------------------------------------------------------------------------

ScalarValue = str | int | float | bool | None

# Severity vocabulary for OperatorWarning. "info" notable-not-problematic;
# "warn" operator should review; "fail" data-integrity concern; "unknown"
# severity cannot be determined.
SEVERITY_INFO = "info"
SEVERITY_WARN = "warn"
SEVERITY_FAIL = "fail"
SEVERITY_UNKNOWN = "unknown"
VALID_SEVERITIES: frozenset[str] = frozenset(
    {SEVERITY_INFO, SEVERITY_WARN, SEVERITY_FAIL, SEVERITY_UNKNOWN}
)


@dataclass(frozen=True)
class ExtractedField:
    """A single extracted scalar value with provenance.

    Invariants (constructed only via :func:`_extracted`/:func:`_computed`):

    * ``value`` is a :data:`ScalarValue` — never a dict or list.
    * ``value is None`` implies ``source_path is None`` (no fake "missing" paths).
    * computed values carry ``computed=True`` and a ``source_path`` that starts
      with ``"computed:"``.
    """

    value: ScalarValue
    source: str
    source_path: str | None
    computed: bool = False
    display: str | None = None


@dataclass(frozen=True)
class OperatorWarning:
    """Operator-facing warning with severity and optional suggested action."""

    code: str
    severity: str  # one of VALID_SEVERITIES
    message: str
    source_path: str | None = None
    suggested_action: str | None = None


@dataclass(frozen=True)
class NormalizedRecommendation:
    """One normalized recommendation extracted from a trace payload."""

    is_primary: bool
    rank: int | None

    market: ExtractedField
    selection: ExtractedField
    line: ExtractedField
    odds: ExtractedField

    raw_probability: ExtractedField
    calibrated_probability: ExtractedField
    implied_probability: ExtractedField

    engine_edge: ExtractedField
    computed_edge: ExtractedField

    kelly_fraction: ExtractedField
    recommended_units: ExtractedField

    raw_confidence_tier: ExtractedField
    display_confidence_band: ExtractedField

    warnings: list[OperatorWarning] = field(default_factory=list)


@dataclass(frozen=True)
class EvidenceCoverage:
    """Evidence coverage metrics — counts and metadata, NOT an invented score."""

    total_signals: int
    applied_signals: int
    shadow_signals: int
    signals_with_confidence: int
    avg_confidence: float | None
    signal_types_present: list[str]
    warnings: list[OperatorWarning] = field(default_factory=list)


@dataclass(frozen=True)
class TraceRecommendationView:
    """Complete normalized view of all recommendations in a trace."""

    trace_id: str
    kind: str  # "game" | "prop" | "slate" | "unknown"
    recommendations: list[NormalizedRecommendation]
    evidence_coverage: EvidenceCoverage
    raw_payload_available: bool


@dataclass(frozen=True)
class SessionTraceFacts:
    """Per-trace facts the session-health aggregation needs.

    All DB-backed: ``evidence_signal_count`` from ``evidence_signals`` rows,
    ``has_outcome`` from the outcomes/prop_outcomes tables, ``has_bet`` from
    ``bet_ledger``. The caller (B.1 wiring on ``ConsoleService``) supplies these;
    the normalizer never derives them from sidecar prose.
    """

    trace_id: str
    evidence_signal_count: int = 0
    has_outcome: bool = False
    has_bet: bool = False


@dataclass(frozen=True)
class SessionHealthView:
    """Computed session health beyond a single ``quality_gate_status`` string."""

    session_id: str
    quality_gate_status: str

    total_traces: int
    traces_with_outcomes: int
    traces_with_bets: int
    traces_with_evidence: int
    traces_zero_evidence: int

    total_evidence_signals: int
    avg_evidence_signals_per_trace: float
    evidence_coverage_ratio: float

    sidecar_valid: bool
    assumption_count: int
    bug_count: int
    audit_event_count: int
    failed_audit_events: int
    pipeline_steps_failed: list[str]

    warnings: list[OperatorWarning] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _scalarize(value: Any) -> ScalarValue:
    """Return ``value`` if it is a scalar, else ``None``.

    Enforces the scalars-only invariant: dicts/lists/tuples/sets (JSON soup)
    collapse to ``None`` so they can never reach a display field. ``bool`` is a
    valid scalar and is preserved as-is.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, int, float)):
        return value
    return None


def _g(obj: Any, key: str) -> Any:
    """``obj[key]`` when ``obj`` is a dict, else ``None`` (null-safe)."""
    if isinstance(obj, dict):
        return obj.get(key)
    return None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_fraction(value: Any) -> float | None:
    """Coerce a probability-like scalar to a 0–1 fraction.

    A probability is bounded [0, 1], so a value in (1, 100] is unambiguously a
    *percentage* and is divided by 100. Values outside [0, 100] (or non-numeric)
    return ``None`` — they are not a probability we can trust. Used only for
    scale-consistent derivation/comparison, never to rewrite a displayed value.
    """
    if not _is_number(value):
        return None
    v = float(value)
    if 0.0 <= v <= 1.0:
        return v
    if 1.0 < v <= 100.0:
        return v / 100.0
    return None


def _first_present(candidates: list[tuple[Any, str]]) -> tuple[ScalarValue, str | None]:
    """First ``(value, path)`` whose scalarized value is non-null, else (None, None)."""
    for value, path in candidates:
        sval = _scalarize(value)
        if sval is not None:
            return sval, path
    return None, None


def _extracted(
    candidates: list[tuple[Any, str]], *, source: str = Source.DB_TRACE_PAYLOAD
) -> ExtractedField:
    """Build a non-computed ExtractedField from a priority chain of candidates."""
    value, path = _first_present(candidates)
    if value is None:
        return ExtractedField(value=None, source=source, source_path=None)
    return ExtractedField(value=value, source=source, source_path=path)


def _computed(
    value: ScalarValue,
    computed_path: str,
    *,
    source: str = Source.DB_TRACE_PAYLOAD,
    display: str | None = None,
) -> ExtractedField:
    """Build a computed ExtractedField (``computed=True``, ``computed:`` path).

    A ``None`` value collapses to the canonical missing field (``source_path``
    None, ``computed`` False) — we only mark something computed when there is a
    computed value to show.
    """
    sval = _scalarize(value)
    if sval is None:
        return ExtractedField(value=None, source=source, source_path=None)
    assert computed_path.startswith("computed:")
    return ExtractedField(
        value=sval, source=source, source_path=computed_path, computed=True, display=display
    )


def _missing(source: str = Source.DB_TRACE_PAYLOAD) -> ExtractedField:
    return ExtractedField(value=None, source=source, source_path=None)


# ---------------------------------------------------------------------------
# Public computational helpers
# ---------------------------------------------------------------------------


def implied_probability_from_american(odds: Any) -> float | None:
    """Implied probability (0–1) from American odds, or ``None``.

    ``+odds`` → ``100 / (odds + 100)``; ``-odds`` → ``|odds| / (|odds| + 100)``.
    ``0``, ``None`` and non-numeric inputs return ``None`` (no guessing). The
    caller is responsible for only passing odds it has *confirmed* are American;
    this is the pure arithmetic.
    """
    if not _is_number(odds):
        return None
    o = float(odds)
    if o > 0:
        return 100.0 / (o + 100.0)
    if o < 0:
        return abs(o) / (abs(o) + 100.0)
    return None  # even money sentinel 0 / unknown


def computed_edge_value(calibrated_probability: Any, implied_probability: Any) -> float | None:
    """Calibrated-minus-implied edge as a 0–1 fraction, or ``None``.

    Returns ``None`` unless *both* inputs are present and coercible to a 0–1
    fraction. Never derived from raw probability (see doctrine). The result is a
    fraction (e.g. ``0.02`` == +2 percentage points), distinct in unit from the
    engine's ``edge_pct`` (already in percentage points).
    """
    cal = _as_fraction(calibrated_probability)
    imp = _as_fraction(implied_probability)
    if cal is None or imp is None:
        return None
    return cal - imp


# Neutral confidence-band language. Raw tiers are preserved separately for audit;
# polished display never shows "A-Tier" phrasing.
_CONFIDENCE_BANDS = {
    "a": "high confidence",
    "b": "medium confidence",
    "c": "low confidence",
    "pass": "tracked lean",
}


def confidence_band(raw_tier: Any) -> str:
    """Map a raw engine confidence tier to neutral display language."""
    if raw_tier is None:
        return "unrated"
    key = str(raw_tier).strip().lower()
    if key == "":
        return "unrated"
    if key in _CONFIDENCE_BANDS:
        return _CONFIDENCE_BANDS[key]
    return f"source confidence: {raw_tier}"


# ---------------------------------------------------------------------------
# Selection-aware probability
# ---------------------------------------------------------------------------

_SIDE_TOKENS = {
    "over": "over",
    "under": "under",
    "home": "home",
    "home_team": "home",
    "away": "away",
    "away_team": "away",
    "draw": "draw",
    "tie": "draw",
}

_SPREAD_TOTAL_MARKETS = {
    "spread",
    "spreads",
    "point_spread",
    "total",
    "totals",
    "team_total",
    "run_line",
    "puck_line",
    "alt_spread",
    "alt_total",
}


def _selection_kind(
    selection: Any, side: Any, market: Any, home_team: str = "", away_team: str = ""
) -> str:
    """Classify a recommendation selection for probability mapping.

    Prefers the engine's explicit ``side`` (game edges) over the free-text
    selection label. Returns one of: over | under | home | away | draw |
    spread_total | player | unknown.
    """
    token = (str(side).strip().lower() if side is not None else "") or (
        str(selection).strip().lower() if selection is not None else ""
    )
    if token in _SIDE_TOKENS:
        return _SIDE_TOKENS[token]
    market_key = str(market).strip().lower() if market is not None else ""
    if market_key in _SPREAD_TOTAL_MARKETS:
        return "spread_total"
    if token:
        if home_team and token == home_team.lower():
            return "home"
        if away_team and token == away_team.lower():
            return "away"
        # A non-empty, non-canonical label: most often a player-prop / tennis
        # player name. Treated as best-effort fallback (see helper below).
        return "player"
    return "unknown"


def probability_source_for_selection(
    selection: Any,
    market: Any,
    predictions: Any,
    result: Any,
    rec: Any,
    home_team: str = "",
    away_team: str = "",
) -> ExtractedField:
    """Map a selection to its matching model probability (public B.0 contract).

    See :func:`_resolve_raw_probability` for the priority chains. This thin
    wrapper returns just the :class:`ExtractedField`; the main normalizer uses
    the warning-bearing internal form.
    """
    field_, _warnings = _resolve_raw_probability(
        selection=selection,
        side=_g(rec, "side"),
        market=market,
        predictions=predictions,
        result=result,
        rec=rec,
        rec_prefix="recommendations",
        home_team=home_team,
        away_team=away_team,
    )
    return field_


def _resolve_raw_probability(
    *,
    selection: Any,
    side: Any,
    market: Any,
    predictions: Any,
    result: Any,
    rec: Any,
    rec_prefix: str,
    home_team: str = "",
    away_team: str = "",
) -> tuple[ExtractedField, list[OperatorWarning]]:
    """Selection-aware raw/model probability with warnings.

    Priority by selection kind (first non-null wins):

    * ``over``  → ``predictions.over_prob`` → ``result.over_prob``
    * ``under`` → ``predictions.under_prob`` → ``result.under_prob``
    * ``home``/``away``/``draw`` → edge-native ``true_prob``/``raw_prob`` (0–1,
      selection-correct) → ``predictions.{sel}_win_prob`` → ``result.simulation.{sel}_win_prob``
    * ``spread_total`` → ``rec.model_prob`` → ``rec.probability`` → ``rec.true_prob`` → ``rec.raw_prob``
    * ``player``/``unknown`` → same generic fallbacks, but flagged
      ``unmatched_selection_prob`` because no selection-specific mapping exists.

    Emits ``missing_raw_prob`` when nothing is found.
    """
    sim = result.get("simulation") if isinstance(result, dict) else None
    kind = _selection_kind(selection, side, market, home_team, away_team)
    warnings: list[OperatorWarning] = []

    edge_native = [
        (_g(rec, "true_prob"), f"{rec_prefix}.true_prob"),
        (_g(rec, "raw_prob"), f"{rec_prefix}.raw_prob"),
    ]
    generic = [
        (_g(rec, "model_prob"), f"{rec_prefix}.model_prob"),
        (_g(rec, "probability"), f"{rec_prefix}.probability"),
        (_g(rec, "true_prob"), f"{rec_prefix}.true_prob"),
        (_g(rec, "raw_prob"), f"{rec_prefix}.raw_prob"),
    ]

    if kind == "over":
        cands = edge_native + [
            (_g(predictions, "over_prob"), "predictions.over_prob"),
            (_g(result, "over_prob"), "result.over_prob"),
        ]
    elif kind == "under":
        cands = edge_native + [
            (_g(predictions, "under_prob"), "predictions.under_prob"),
            (_g(result, "under_prob"), "result.under_prob"),
        ]
    elif kind in ("home", "away", "draw"):
        sim_key = "draw_prob" if kind == "draw" else f"{kind}_win_prob"
        cands = edge_native + [
            (_g(predictions, sim_key), f"predictions.{sim_key}"),
            (_g(sim, sim_key), f"result.simulation.{sim_key}"),
        ]
    elif kind == "spread_total":
        cands = generic
    else:  # player / unknown
        cands = generic

    field_ = _extracted(cands)
    if field_.value is not None and isinstance(field_.value, (int, float)):
        path = field_.source_path or ""
        if ("predictions." in path or "result.simulation." in path) and path.endswith("_win_prob"):
            if field_.value > 1.0:
                field_ = ExtractedField(
                    value=field_.value / 100.0, source=field_.source, source_path=field_.source_path
                )

    if kind in ("player", "unknown"):
        warnings.append(
            OperatorWarning(
                code="unmatched_selection_prob",
                severity=SEVERITY_WARN,
                message="selection probability could not be matched to a model prediction",
                source_path=field_.source_path,
            )
        )
    if field_.value is None:
        warnings.append(
            OperatorWarning(
                code="missing_raw_prob",
                severity=SEVERITY_WARN,
                message="no model probability found in trace payload",
            )
        )
    return field_, warnings


# ---------------------------------------------------------------------------
# Odds + implied probability resolution
# ---------------------------------------------------------------------------

# Fields known to hold American odds (engine contract). Odds sourced from these
# are "confirmed American" and eligible for implied-probability derivation.
_AMERICAN_ODDS_KEYS = ("bet_side_odds", "odds", "market_odds")

# Per-selection American-odds keys inside the (heuristic) odds snapshot. Surfaced
# for visibility but treated as UNCONFIRMED — implied probability is not derived
# from them (the snapshot's format/selection mapping is not guaranteed).
_SNAPSHOT_ODDS_KEYS = {
    "over": "odds_over",
    "under": "odds_under",
    "home": "moneyline_home",
    "away": "moneyline_away",
    "draw": "moneyline_draw",
}


def _resolve_odds(
    *,
    rec: Any,
    rec_prefix: str,
    selection_kind: str,
    odds_snapshot: Any,
    input_snapshot: Any,
) -> tuple[ExtractedField, bool, list[OperatorWarning]]:
    """Resolve odds and whether the value is confirmed-American.

    Returns ``(odds_field, confirmed_american, warnings)``. Confirmed-American
    odds come only from the engine's American-odds fields on the recommendation;
    a snapshot fallback is surfaced as a value but flagged
    ``ambiguous_odds_format`` and is *not* eligible for implied probability.
    """
    warnings: list[OperatorWarning] = []

    confirmed_cands = [(_g(rec, key), f"{rec_prefix}.{key}") for key in _AMERICAN_ODDS_KEYS]
    value, path = _first_present(confirmed_cands)
    if value is not None:
        return (
            ExtractedField(value=value, source=Source.DB_TRACE_PAYLOAD, source_path=path),
            True,
            warnings,
        )

    # Heuristic snapshot fallback (unconfirmed format/mapping).
    snap_key = _SNAPSHOT_ODDS_KEYS.get(selection_kind)
    snap_cands: list[tuple[Any, str]] = []
    if snap_key:
        snap_cands = [
            (_g(odds_snapshot, snap_key), f"odds_snapshot.{snap_key}"),
            (_g(_g(input_snapshot, "odds"), snap_key), f"input_snapshot.odds.{snap_key}"),
        ]
    value, path = _first_present(snap_cands)
    if value is not None:
        warnings.append(
            OperatorWarning(
                code="ambiguous_odds_format",
                severity=SEVERITY_INFO,
                message="odds format could not be confirmed as American — implied probability not computed",
                source_path=path,
            )
        )
        return (
            ExtractedField(value=value, source=Source.DB_TRACE_PAYLOAD, source_path=path),
            False,
            warnings,
        )

    warnings.append(
        OperatorWarning(
            code="missing_odds",
            severity=SEVERITY_WARN,
            message="odds not found in trace payload",
        )
    )
    return _missing(), False, warnings


# ---------------------------------------------------------------------------
# Calibrated probability resolution (selection-aware to avoid over/under mixup)
# ---------------------------------------------------------------------------


def _resolve_calibrated_probability(
    *,
    selection_kind: str,
    rec: Any,
    rec_prefix: str,
    result: Any,
    calibration_audit: Any,
) -> ExtractedField:
    """Calibrated probability, mapped to the selection (never first-available).

    Priority: edge-native ``calibrated_prob`` → the edge's nested
    ``calibration_audit`` → the selection-matched prop audit
    (``result.{over,under}_calibration_audit``) → the trace-level
    ``calibration_audit[]`` entry whose ``market`` matches the selection.
    Both ``calibrated_prob`` and ``calibrated_probability`` spellings accepted.
    """
    cands: list[tuple[Any, str]] = [
        (_g(rec, "calibrated_prob"), f"{rec_prefix}.calibrated_prob"),
        (_g(rec, "calibrated_probability"), f"{rec_prefix}.calibrated_probability"),
        (
            _g(_g(rec, "calibration_audit"), "calibrated_prob"),
            f"{rec_prefix}.calibration_audit.calibrated_prob",
        ),
        (
            _g(_g(rec, "calibration_audit"), "calibrated_probability"),
            f"{rec_prefix}.calibration_audit.calibrated_probability",
        ),
    ]

    if selection_kind in ("over", "under"):
        audit_key = f"{selection_kind}_calibration_audit"
        audit = _g(result, audit_key)
        cands += [
            (_g(audit, "calibrated_prob"), f"result.{audit_key}.calibrated_prob"),
            (_g(audit, "calibrated_probability"), f"result.{audit_key}.calibrated_probability"),
        ]

    # Trace-level calibration_audit[] keyed by market == selection.
    if isinstance(calibration_audit, list):
        for idx, entry in enumerate(calibration_audit):
            if not isinstance(entry, dict):
                continue
            entry_market = str(entry.get("market") or "").strip().lower()
            if entry_market and entry_market == selection_kind:
                cands += [
                    (entry.get("calibrated_prob"), f"calibration_audit[{idx}].calibrated_prob"),
                    (
                        entry.get("calibrated_probability"),
                        f"calibration_audit[{idx}].calibrated_probability",
                    ),
                ]
                break

    return _extracted(cands)


# ---------------------------------------------------------------------------
# Recommendation normalization
# ---------------------------------------------------------------------------


def normalize_recommendation(
    rec: Any,
    *,
    trace: dict[str, Any],
    is_primary: bool,
    rank: int | None,
    rec_prefix: str = "recommendations",
) -> NormalizedRecommendation:
    """Normalize one recommendation dict from a trace into the uniform view model."""
    rec = rec if isinstance(rec, dict) else {}
    result = trace.get("result") if isinstance(trace.get("result"), dict) else {}
    predictions = trace.get("predictions")
    input_snapshot = trace.get("input_snapshot")
    odds_snapshot = trace.get("odds_snapshot")
    calibration_audit = trace.get("calibration_audit")

    home_team = (
        str(
            trace.get("home_team")
            or trace.get("result", {}).get("home_team")
            or _g(input_snapshot, "home_team")
            or ""
        )
        .strip()
        .lower()
    )
    away_team = (
        str(
            trace.get("away_team")
            or trace.get("result", {}).get("away_team")
            or _g(input_snapshot, "away_team")
            or ""
        )
        .strip()
        .lower()
    )

    warnings: list[OperatorWarning] = []

    market = _extracted(
        [
            (_g(rec, "market"), f"{rec_prefix}.market"),
            (_g(_g(result, "best_bet"), "market"), "result.best_bet.market"),
            (trace.get("kind"), "kind"),
        ]
    )
    selection = _extracted(
        [
            (_g(rec, "side"), f"{rec_prefix}.side"),
            (_g(rec, "selection"), f"{rec_prefix}.selection"),
            (_g(rec, "recommendation"), f"{rec_prefix}.recommendation"),
            (_g(result, "recommendation"), "result.recommendation"),
        ]
    )
    line = _extracted(
        [
            (_g(rec, "line"), f"{rec_prefix}.line"),
            (_g(input_snapshot, "line"), "input_snapshot.line"),
            (_g(result, "line"), "result.line"),
        ]
    )

    selection_kind = _selection_kind(
        selection.value, _g(rec, "side"), market.value, home_team, away_team
    )

    odds, confirmed_american, odds_warnings = _resolve_odds(
        rec=rec,
        rec_prefix=rec_prefix,
        selection_kind=selection_kind,
        odds_snapshot=odds_snapshot,
        input_snapshot=input_snapshot,
    )
    warnings.extend(odds_warnings)

    raw_probability, prob_warnings = _resolve_raw_probability(
        selection=selection.value,
        side=_g(rec, "side"),
        market=market.value,
        predictions=predictions,
        result=result,
        rec=rec,
        rec_prefix=rec_prefix,
        home_team=home_team,
        away_team=away_team,
    )
    warnings.extend(prob_warnings)

    calibrated_probability = _resolve_calibrated_probability(
        selection_kind=selection_kind,
        rec=rec,
        rec_prefix=rec_prefix,
        result=result,
        calibration_audit=calibration_audit,
    )

    # Implied probability: only from confirmed-American odds.
    implied_value = implied_probability_from_american(odds.value) if confirmed_american else None
    implied_probability = _computed(implied_value, "computed:implied_from_american_odds")

    engine_edge = _extracted(
        [
            (_g(rec, "edge_pct"), f"{rec_prefix}.edge_pct"),
            (_g(rec, "edge"), f"{rec_prefix}.edge"),
        ]
    )

    # Computed edge: calibrated − implied, only when both present (American odds).
    edge_value = computed_edge_value(calibrated_probability.value, implied_probability.value)
    computed_edge = _computed(edge_value, "computed:calibrated_minus_implied")

    kelly_fraction = _extracted(
        [
            (_g(rec, "kelly_fraction"), f"{rec_prefix}.kelly_fraction"),
            (_g(result, "kelly_fraction"), "result.kelly_fraction"),
        ]
    )
    recommended_units = _extracted(
        [
            (_g(rec, "recommended_units"), f"{rec_prefix}.recommended_units"),
            (_g(result, "recommended_units"), "result.recommended_units"),
        ]
    )

    raw_confidence_tier = _extracted(
        [
            (_g(rec, "confidence_tier"), f"{rec_prefix}.confidence_tier"),
            (_g(result, "confidence_tier"), "result.confidence_tier"),
        ]
    )
    band = confidence_band(raw_confidence_tier.value)
    display_confidence_band = _computed(band, "computed:from_raw_confidence_tier")

    warnings.extend(
        _math_consistency_warnings(
            raw_probability=raw_probability,
            calibrated_probability=calibrated_probability,
            implied_probability=implied_probability,
            engine_edge=engine_edge,
            computed_edge=computed_edge,
            kelly_fraction=kelly_fraction,
            recommended_units=recommended_units,
            raw_confidence_tier=raw_confidence_tier,
        )
    )

    return NormalizedRecommendation(
        is_primary=is_primary,
        rank=rank,
        market=market,
        selection=selection,
        line=line,
        odds=odds,
        raw_probability=raw_probability,
        calibrated_probability=calibrated_probability,
        implied_probability=implied_probability,
        engine_edge=engine_edge,
        computed_edge=computed_edge,
        kelly_fraction=kelly_fraction,
        recommended_units=recommended_units,
        raw_confidence_tier=raw_confidence_tier,
        display_confidence_band=display_confidence_band,
        warnings=warnings,
    )


def _math_consistency_warnings(
    *,
    raw_probability: ExtractedField,
    calibrated_probability: ExtractedField,
    implied_probability: ExtractedField,
    engine_edge: ExtractedField,
    computed_edge: ExtractedField,
    kelly_fraction: ExtractedField,
    recommended_units: ExtractedField,
    raw_confidence_tier: ExtractedField,
) -> list[OperatorWarning]:
    """Cross-field consistency warnings (the kind found during runtime review)."""
    out: list[OperatorWarning] = []

    # raw below implied with no calibrated probability to explain it. Compare in
    # fraction space; skip when either operand is not a clean 0–1/percentage prob.
    if calibrated_probability.value is None:
        raw_frac = _as_fraction(raw_probability.value)
        imp_frac = _as_fraction(implied_probability.value)
        if raw_frac is not None and imp_frac is not None and raw_frac < imp_frac:
            out.append(
                OperatorWarning(
                    code="raw_below_implied",
                    severity=SEVERITY_WARN,
                    message="raw probability is below implied probability — review recommended",
                    source_path=raw_probability.source_path,
                )
            )

    if (
        _is_number(kelly_fraction.value)
        and float(kelly_fraction.value) > 0
        and engine_edge.value is None
        and computed_edge.value is None
    ):
        out.append(
            OperatorWarning(
                code="kelly_no_edge",
                severity=SEVERITY_WARN,
                message="positive Kelly fraction shown without confirmed positive edge source",
                source_path=kelly_fraction.source_path,
            )
        )

    if recommended_units.value is not None and kelly_fraction.value is None:
        out.append(
            OperatorWarning(
                code="units_no_kelly",
                severity=SEVERITY_INFO,
                message="recommended units present without Kelly fraction source",
                source_path=recommended_units.source_path,
            )
        )

    if raw_confidence_tier.value is None:
        out.append(
            OperatorWarning(
                code="no_confidence_tier",
                severity=SEVERITY_INFO,
                message="confidence tier not set by engine",
            )
        )

    return out


def _kind(trace: dict[str, Any]) -> str:
    k = trace.get("kind")
    return k if k in ("game", "prop", "slate") else "unknown"


def _best_bet_primary_index(edges: list[dict[str, Any]], best_bet: Any) -> int:
    """Best-effort: index of the edge matching ``best_bet`` (else 0)."""
    selection = str(_g(best_bet, "selection") or "").strip().lower()
    if not selection:
        return 0
    for idx, edge in enumerate(edges):
        team = str(_g(edge, "team") or "").strip().lower()
        side = str(_g(edge, "side") or "").strip().lower()
        if (team and team in selection) or (side and side in selection):
            return idx
    return 0


def build_trace_recommendation_view(
    trace: dict[str, Any], *, evidence_signals: list[dict[str, Any]] | None = None
) -> TraceRecommendationView:
    """Normalize all recommendations in a trace into a uniform read view.

    Handles every persisted/raw recommendation shape:

    * ``recommendations`` is a list  → each item normalized; first is primary.
    * ``recommendations`` is a dict  → one primary item, ``rank=None``.
    * ``result.edges[]``             → each edge normalized; best_bet (if any)
      marks the matching edge primary, else the first.
    * ``result.best_bet`` alone      → one primary item, ``rank=None``.
    * none of the above              → empty list.
    """
    trace = trace if isinstance(trace, dict) else {}
    result = trace.get("result") if isinstance(trace.get("result"), dict) else {}
    recs = trace.get("recommendations")

    # (rec_dict, is_primary, rank, rec_prefix)
    items: list[tuple[dict[str, Any], bool, int | None, str]] = []

    if isinstance(recs, list) and any(isinstance(x, dict) for x in recs):
        dict_recs = [x for x in recs if isinstance(x, dict)]
        for i, r in enumerate(dict_recs):
            items.append((r, i == 0, i, f"recommendations[{i}]"))
    elif isinstance(recs, dict) and recs:
        items.append((recs, True, None, "recommendations"))
    elif isinstance(result.get("edges"), list) and any(
        isinstance(e, dict) for e in result["edges"]
    ):
        edges = [e for e in result["edges"] if isinstance(e, dict)]
        primary_idx = _best_bet_primary_index(edges, result.get("best_bet"))
        for i, e in enumerate(edges):
            items.append((e, i == primary_idx, i, f"result.edges[{i}]"))
    elif isinstance(result.get("best_bet"), dict):
        items.append((result["best_bet"], True, None, "result.best_bet"))

    recommendations = [
        normalize_recommendation(
            r, trace=trace, is_primary=is_primary, rank=rank, rec_prefix=prefix
        )
        for (r, is_primary, rank, prefix) in items
    ]

    coverage = build_evidence_coverage(evidence_signals or [])

    return TraceRecommendationView(
        trace_id=str(trace.get("trace_id") or ""),
        kind=_kind(trace),
        recommendations=recommendations,
        evidence_coverage=coverage,
        raw_payload_available=bool(trace),
    )


# ---------------------------------------------------------------------------
# Evidence coverage
# ---------------------------------------------------------------------------

_APPLIED_TRUE = {1, "1", "true", "True", "t", "yes", "applied"}


def _signal_is_applied(row: dict[str, Any]) -> bool:
    """A signal counts as applied if ``applied`` is truthy or a non-zero factor."""
    applied = row.get("applied")
    if applied is True or applied in _APPLIED_TRUE:
        return True
    factor = row.get("applied_factor")
    if _is_number(factor) and float(factor) != 0.0:
        return True
    return False


def build_evidence_coverage(
    evidence_signals: list[dict[str, Any]] | None,
) -> EvidenceCoverage:
    """Aggregate evidence_signals rows into coverage metrics + warnings.

    Coverage = how many signals exist, how many were applied vs shadow, and their
    confidence metadata. NOT a quality/strength score.
    """
    rows = [r for r in (evidence_signals or []) if isinstance(r, dict)]
    total = len(rows)
    applied = sum(1 for r in rows if _signal_is_applied(r))
    shadow = total - applied

    confidences = [float(r.get("confidence")) for r in rows if _is_number(r.get("confidence"))]
    signals_with_confidence = len(confidences)
    avg_confidence = (sum(confidences) / len(confidences)) if confidences else None

    types = sorted({str(r.get("signal_type")) for r in rows if r.get("signal_type") is not None})

    warnings: list[OperatorWarning] = []
    if total == 0:
        warnings.append(
            OperatorWarning(
                code="zero_evidence",
                severity=SEVERITY_WARN,
                message="zero evidence signals linked to this trace",
            )
        )
    elif applied == 0:
        warnings.append(
            OperatorWarning(
                code="no_applied_evidence",
                severity=SEVERITY_WARN,
                message="evidence signals present but none applied",
            )
        )
    if avg_confidence is not None and avg_confidence < 0.3:
        warnings.append(
            OperatorWarning(
                code="low_avg_confidence",
                severity=SEVERITY_INFO,
                message="average evidence signal confidence is below 0.3",
            )
        )

    return EvidenceCoverage(
        total_signals=total,
        applied_signals=applied,
        shadow_signals=shadow,
        signals_with_confidence=signals_with_confidence,
        avg_confidence=avg_confidence,
        signal_types_present=types,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Session health
# ---------------------------------------------------------------------------

_OK_AUDIT_STATUSES = {
    "ok",
    "pass",
    "passed",
    "success",
    "succeeded",
    "done",
    "complete",
    "completed",
    "clean",
    "green",
    "skipped",
}


def _audit_attr(event: Any, name: str) -> Any:
    """Read an attribute from an audit event whether dict-shaped or object-shaped."""
    if isinstance(event, dict):
        return event.get(name)
    return getattr(event, name, None)


def _status_is_failed(status: Any) -> bool:
    """True for a set status that is not a recognised success/neutral state."""
    if status is None:
        return False
    key = str(status).strip().lower()
    if key == "":
        return False
    return key not in _OK_AUDIT_STATUSES


def build_session_health_view(
    *,
    session_id: str,
    quality_gate_status: str,
    trace_facts: list[SessionTraceFacts],
    sidecar_valid: bool,
    assumption_count: int,
    bug_count: int,
    audit_events: list[Any] | None = None,
    pipeline_status: dict[str, Any] | None = None,
) -> SessionHealthView:
    """Aggregate per-trace facts + sidecar flags into a richer session health view.

    All numeric inputs are DB-backed counts/flags supplied by the caller. Sidecar
    prose is never parsed here; ``assumption_count``/``bug_count`` are *counts*
    the caller already extracted, and ``audit_events`` are read only for their
    ``status``/``step`` flags. ``pipeline_status`` (if provided) contributes
    failed step names from non-ok scalar values.
    """
    audit_events = audit_events or []
    facts = list(trace_facts)
    total = len(facts)

    traces_with_outcomes = sum(1 for t in facts if t.has_outcome)
    traces_with_bets = sum(1 for t in facts if t.has_bet)
    traces_with_evidence = sum(1 for t in facts if t.evidence_signal_count > 0)
    traces_zero_evidence = total - traces_with_evidence
    total_evidence_signals = sum(t.evidence_signal_count for t in facts)
    avg_evidence = (total_evidence_signals / total) if total else 0.0
    coverage_ratio = (traces_with_evidence / total) if total else 0.0

    audit_event_count = len(audit_events)
    failed_audit_events = sum(
        1 for e in audit_events if _status_is_failed(_audit_attr(e, "status"))
    )

    failed_steps: list[str] = []
    for e in audit_events:
        if _status_is_failed(_audit_attr(e, "status")):
            step = _audit_attr(e, "step")
            if step:
                failed_steps.append(str(step))
    if isinstance(pipeline_status, dict):
        for step, status in pipeline_status.items():
            if isinstance(status, str) and _status_is_failed(status):
                failed_steps.append(str(step))
    # Stable, de-duplicated order.
    pipeline_steps_failed = sorted(dict.fromkeys(failed_steps))

    warnings = _session_warnings(
        total=total,
        traces_zero_evidence=traces_zero_evidence,
        coverage_ratio=coverage_ratio,
        avg_evidence=avg_evidence,
        traces_with_outcomes=traces_with_outcomes,
        assumption_count=assumption_count,
        bug_count=bug_count,
        sidecar_valid=sidecar_valid,
        failed_audit_events=failed_audit_events,
        pipeline_steps_failed=pipeline_steps_failed,
    )

    return SessionHealthView(
        session_id=session_id,
        quality_gate_status=quality_gate_status,
        total_traces=total,
        traces_with_outcomes=traces_with_outcomes,
        traces_with_bets=traces_with_bets,
        traces_with_evidence=traces_with_evidence,
        traces_zero_evidence=traces_zero_evidence,
        total_evidence_signals=total_evidence_signals,
        avg_evidence_signals_per_trace=avg_evidence,
        evidence_coverage_ratio=coverage_ratio,
        sidecar_valid=sidecar_valid,
        assumption_count=assumption_count,
        bug_count=bug_count,
        audit_event_count=audit_event_count,
        failed_audit_events=failed_audit_events,
        pipeline_steps_failed=pipeline_steps_failed,
        warnings=warnings,
    )


def _session_warnings(
    *,
    total: int,
    traces_zero_evidence: int,
    coverage_ratio: float,
    avg_evidence: float,
    traces_with_outcomes: int,
    assumption_count: int,
    bug_count: int,
    sidecar_valid: bool,
    failed_audit_events: int,
    pipeline_steps_failed: list[str],
) -> list[OperatorWarning]:
    out: list[OperatorWarning] = []

    if traces_zero_evidence > 0:
        out.append(
            OperatorWarning(
                code="traces_no_evidence",
                severity=SEVERITY_WARN,
                message=f"{traces_zero_evidence} of {total} traces have zero evidence signals",
            )
        )
    if total > 0 and coverage_ratio < 0.5:
        out.append(
            OperatorWarning(
                code="low_evidence_coverage",
                severity=SEVERITY_WARN,
                message="evidence coverage below 50% — signals may be missing",
            )
        )
    if total > 0 and avg_evidence < 1.0:
        out.append(
            OperatorWarning(
                code="low_avg_evidence",
                severity=SEVERITY_INFO,
                message="average evidence signals per trace < 1",
            )
        )
    if assumption_count > 0:
        out.append(
            OperatorWarning(
                code="assumptions_logged",
                severity=SEVERITY_INFO,
                message=f"{assumption_count} assumption(s) logged",
            )
        )
    if bug_count > 0:
        out.append(
            OperatorWarning(
                code="bugs_logged",
                severity=SEVERITY_WARN,
                message=f"{bug_count} bug(s) logged",
            )
        )
    if pipeline_steps_failed:
        out.append(
            OperatorWarning(
                code="pipeline_failures",
                severity=SEVERITY_FAIL,
                message=f"pipeline steps failed: {', '.join(pipeline_steps_failed)}",
            )
        )
    if not sidecar_valid:
        out.append(
            OperatorWarning(
                code="sidecar_invalid",
                severity=SEVERITY_WARN,
                message="sidecar invalid — session process narrative unavailable",
            )
        )
    if traces_with_outcomes == 0 and total > 0:
        out.append(
            OperatorWarning(
                code="no_outcomes",
                severity=SEVERITY_INFO,
                message="no traces have attached outcomes",
            )
        )
    if failed_audit_events > 0:
        out.append(
            OperatorWarning(
                code="failed_audits",
                severity=SEVERITY_WARN,
                message=f"{failed_audit_events} audit event(s) with non-ok status",
            )
        )
    return out
