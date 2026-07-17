"""Shared decision-support adapter: persisted trace → safe matchup brief DTO.

This is the single projection every primary (non-lab) consumer renders —
console API, MCP matchup brief, session report. It filters, groups, and labels
engine values; it never recomputes them.

Safety properties (enforced here, tested in tests/trace/test_decision_support.py):

- **Allowlist projection.** The DTO is constructed field-by-field; a final
  recursive sweep rejects any denied key that slips into an open dict.
- **Symmetric probability sets only.** Model probabilities appear either as a
  complete mutually-exclusive outcome set (home/away[/draw]; over+under) or not
  at all — no gap, no ranking, no per-side emphasis. Outcomes are ordered by
  stable outcome identity.
- **output_mode intersection.** ``RESEARCH_CANDIDATE`` markets never disclose
  model outcome probabilities; raw simulation distribution summaries stay
  visible with an explicit simulation-output label.
- **Legacy compatibility.** v1 traces (no event identity / decision-support
  payload) render through a labeled compatibility path: one-sided legacy prose
  is marked incomplete, its verdict stays lab-only, and the group carries an
  identity warning instead of being heuristically merged.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from omega.core.contracts.language import blocked_language
from omega.core.contracts.protected_fields import (
    PROTECTED_QUANT_FIELDS,
    find_protected_key,
)
from omega.core.contracts.schemas import (
    DecisionSupportPresentationV1,
    coerce_presentation_mode,
)
from omega.ops.output_modes import OutputMode, classify_market_output_mode

MATCHUP_BRIEF_SCHEMA_VERSION = 1

ESTIMATE_LABEL = "Omega model estimate — engine-generated, not a recommendation"
SIMULATION_LABEL = "Simulation output — engine-generated, not a recommendation"
LEGACY_COMPAT_NOTE = (
    "Legacy analyst note (compatibility path): one-sided reasoning from the "
    "recommendation era — counterarguments were not recorded and are not fabricated."
)

# Keys that must never appear anywhere in a primary-product payload. Superset of
# the engine-owned protected quant fields plus recommendation/staking vocabulary.
DENYLIST_KEYS: frozenset[str] = PROTECTED_QUANT_FIELDS | frozenset(
    {
        "best_bet",
        "recommendation",
        "recommended_units",
        "verdict",
        "edge_over",
        "edge_under",
        "stake_amount",
        "stake_units",
        "bet_side_odds",
        "spread_coverage_prob",
        "true_prob",
        "calibrated_prob",
    }
)


class DecisionSupportViolation(ValueError):
    """A primary-product DTO carried a denied key — always a bug, never data."""


# -- DTO models ----------------------------------------------------------------


class MarketOutcomeEstimate(BaseModel):
    """One outcome of a complete symmetric set. Never rendered alone."""

    model_config = ConfigDict(extra="forbid")

    outcome_key: str
    label: str
    model_estimate: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Engine estimate (0-1); None = not disclosed"
    )
    market_implied: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Market-implied baseline (0-1), engine-computed"
    )


class MarketProbabilitySet(BaseModel):
    """Complete mutually-exclusive outcome estimates for one market, or a
    withheld marker explaining why nothing is shown."""

    model_config = ConfigDict(extra="forbid")

    market_key: str
    disclosure: Literal["shown", "withheld"]
    withheld_reason: str | None = None
    estimate_label: str = ESTIMATE_LABEL
    outcomes: list[MarketOutcomeEstimate] = Field(default_factory=list)


class DistributionSummaryView(BaseModel):
    """Raw simulation distribution summary — visible in every output mode."""

    model_config = ConfigDict(extra="forbid")

    target: str
    market: str | None = None
    stat_key: str | None = None
    distribution_type: str
    sample_mean: float | None = None
    sample_std: float | None = None
    p10: float | None = None
    p50: float | None = None
    p90: float | None = None
    n_iterations: int | None = None
    simulation_label: str = SIMULATION_LABEL


class SourceNoteView(BaseModel):
    """Provenance for one displayed external fact/source."""

    model_config = ConfigDict(extra="forbid")

    source: str
    source_title: str | None = None
    source_url: str | None = None
    retrieved_at: str | None = None
    provenance_status: Literal["ok", "partial", "missing_provenance"] = "missing_provenance"


class SensitivityView(BaseModel):
    """Deterministic sensitivity state — rendered, never computed, here.

    Until the Phase 3 engine-owned artifact exists this is explicitly
    unavailable; presentation code must never approximate a sensitivity
    analysis on its own.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["available", "unavailable"] = "unavailable"
    reason: str | None = "not available for this analysis"
    scenarios: list[dict[str, Any]] = Field(default_factory=list)


class LegacyPresentationCompat(BaseModel):
    """Labeled compatibility mapping of the legacy reasoning_presentation.

    thesis→summary, market_read→market_context, risks→uncertainty. The one-sided
    'why' is retained but explicitly flagged incomplete; the verdict is lab-only
    and never mapped.
    """

    model_config = ConfigDict(extra="forbid")

    summary: str | None = None
    market_context: str | None = None
    uncertainties: list[str] = Field(default_factory=list)
    one_sided_case: str | None = None
    incomplete: Literal[True] = True
    note: str = LEGACY_COMPAT_NOTE


class MarketBriefView(BaseModel):
    """One market group (a game trace or a player-prop trace) inside a brief."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str
    kind: str
    market_group: str
    league: str | None = None
    matchup: str = ""
    game_date: str | None = None
    output_mode: str
    output_mode_reasons: list[str] = Field(default_factory=list)
    # One complete symmetric set per quoted market (moneyline/spread/total for
    # games; the over/under pair for props), in stable market-identity order.
    probability_sets: list[MarketProbabilitySet] = Field(default_factory=list)
    sensitivity: SensitivityView = Field(default_factory=SensitivityView)
    distributions: list[DistributionSummaryView] = Field(default_factory=list)
    market_lines: dict[str, Any] = Field(default_factory=dict)
    data_quality: list[str] = Field(default_factory=list)
    aggregate_quality: float | None = None
    sources: list[SourceNoteView] = Field(default_factory=list)
    decision_support: DecisionSupportPresentationV1 | None = None
    legacy_presentation: LegacyPresentationCompat | None = None


class MatchupBriefV1(BaseModel):
    """The safe, allowlisted brief for one event (or one ungrouped legacy trace)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = MATCHUP_BRIEF_SCHEMA_VERSION
    group_key: str
    event_key: str | None = None
    identity_warning: bool = False
    league: str | None = None
    matchup: str = ""
    game_date: str | None = None
    presentation_mode: str = "decision_support"
    markets: list[MarketBriefView] = Field(default_factory=list)


# -- helpers ---------------------------------------------------------------------


def assert_no_denied_keys(payload: Any) -> None:
    """Recursive denied-key sweep over a serialized primary DTO. Raises on hit."""
    found = find_protected_key(payload, fields=DENYLIST_KEYS)
    if found is not None:
        raise DecisionSupportViolation(
            f"primary decision-support payload carried denied key {found!r}"
        )


def _scrub_prose(texts: list[str], warnings: list[str]) -> list[str]:
    """Drop prose strings containing blocked recommendation language.

    Read-side legacy traces may predate the language guard; a violating string
    is withheld (with a data_quality warning) rather than crashing the brief.
    """
    clean: list[str] = []
    for text in texts:
        found = blocked_language(text)
        if found:
            warnings.append(f"withheld legacy prose containing blocked language: {found}")
        else:
            clean.append(text)
    return clean


def market_output_mode_for_trace(trace: dict[str, Any]) -> tuple[OutputMode, list[str]]:
    """Per-trace market disclosure authorization from its own calibration audit.

    Reuses the canonical classifier with this trace's applied-profile facts.
    A trace with no calibration audit (no profile) classifies RESEARCH_CANDIDATE.
    Sidecar validity is a session-level concern; at the single-trace level it is
    treated as valid, matching the aggregate report's convention.
    """
    audits = trace.get("calibration_audit") or []
    audit = audits[0] if audits and isinstance(audits[0], dict) else {}
    mode, reasons = classify_market_output_mode(
        profile_id=audit.get("profile_id"),
        sample_size=audit.get("sample_size"),
        calibration_error=audit.get("ece"),
        trace_count=1,
        sidecar_valid=True,
        maturity=audit.get("profile_maturity"),
    )
    return mode, reasons


def _symmetric_set(
    market_key: str,
    required: tuple[str, ...],
    by_side: dict[str, dict[str, Any]],
    *,
    authorized: bool,
    label_for: Any,
) -> MarketProbabilitySet:
    """One complete symmetric outcome set for one market, or a withheld marker."""
    if not authorized:
        return MarketProbabilitySet(
            market_key=market_key,
            disclosure="withheld",
            withheld_reason="research_candidate_output_mode",
        )
    if any(side not in by_side for side in required):
        return MarketProbabilitySet(
            market_key=market_key,
            disclosure="withheld",
            withheld_reason="incomplete_outcome_set",
        )
    outcomes = [
        MarketOutcomeEstimate(
            outcome_key=side,
            label=label_for(side, by_side[side]),
            model_estimate=by_side[side].get("calibrated_prob"),
            market_implied=by_side[side].get("market_implied"),
        )
        for side in required
    ]
    return MarketProbabilitySet(market_key=market_key, disclosure="shown", outcomes=outcomes)


# Stable market-identity order for game probability sets (never edge-ordered).
_GAME_MARKET_ORDER = ("moneyline", "spread", "total")


def _game_probability_sets(
    trace: dict[str, Any], authorized: bool
) -> list[MarketProbabilitySet]:
    """Per-market symmetric sets for a game trace: moneyline, spread, total.

    A market appears when the trace carries edges for it (moneyline always,
    since it is the base market of every game analysis). Partial sides render
    as an explicit withheld marker rather than a one-sided number.
    """
    result = trace.get("result") or {}
    simulation = result.get("simulation") or {}
    three_way = simulation.get("draw_prob") is not None

    edges_by_market: dict[str, dict[str, dict[str, Any]]] = {}
    for edge in result.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        market = str(edge.get("market") or "moneyline")
        if market == "draw":
            market = "moneyline"
        side = str(edge.get("side") or "")
        if side:
            edges_by_market.setdefault(market, {}).setdefault(side, edge)

    def _team_label(side: str, edge: dict[str, Any]) -> str:
        return str(edge.get("team") or side)

    def _spread_label(side: str, edge: dict[str, Any]) -> str:
        team = str(edge.get("team") or side)
        line = edge.get("line")
        return f"{team} {line:+g}" if isinstance(line, (int, float)) else team

    sets: list[MarketProbabilitySet] = []
    ml_required = ("home", "away", "draw") if three_way else ("home", "away")
    sets.append(
        _symmetric_set(
            "moneyline",
            ml_required,
            edges_by_market.get("moneyline", {}),
            authorized=authorized,
            label_for=_team_label,
        )
    )
    if "spread" in edges_by_market:
        sets.append(
            _symmetric_set(
                "spread",
                ("home", "away"),
                edges_by_market["spread"],
                authorized=authorized,
                label_for=_spread_label,
            )
        )
    if "total" in edges_by_market:
        sets.append(
            _symmetric_set(
                "total",
                ("over", "under"),
                edges_by_market["total"],
                authorized=authorized,
                label_for=_team_label,
            )
        )
    sets.sort(key=lambda s: _GAME_MARKET_ORDER.index(s.market_key))
    return sets


def _prop_probability_set(trace: dict[str, Any], authorized: bool) -> MarketProbabilitySet:
    """Prop over/under set: both sides or nothing."""
    input_snap = trace.get("input_snapshot") or {}
    result = trace.get("result") or {}
    market_key = str(input_snap.get("prop_type") or result.get("prop_type") or "prop")
    if not authorized:
        return MarketProbabilitySet(
            market_key=market_key,
            disclosure="withheld",
            withheld_reason="research_candidate_output_mode",
        )
    over = result.get("over_prob")
    under = result.get("under_prob")
    if over is None or under is None:
        return MarketProbabilitySet(
            market_key=market_key,
            disclosure="withheld",
            withheld_reason="incomplete_outcome_set",
        )
    line = input_snap.get("line") or result.get("line")
    line_str = f" {line}" if line is not None else ""
    outcomes = [
        MarketOutcomeEstimate(
            outcome_key="over", label=f"Over{line_str}", model_estimate=over
        ),
        MarketOutcomeEstimate(
            outcome_key="under", label=f"Under{line_str}", model_estimate=under
        ),
    ]
    return MarketProbabilitySet(market_key=market_key, disclosure="shown", outcomes=outcomes)


def _sensitivity_view(trace: dict[str, Any]) -> SensitivityView:
    """Render engine-persisted sensitivity when it exists; else explicitly
    unavailable. Presentation never computes or approximates sensitivity."""
    result = trace.get("result") or {}
    persisted = result.get("sensitivity")
    if isinstance(persisted, list) and persisted:
        scenarios = [s for s in persisted if isinstance(s, dict)]
        if scenarios:
            return SensitivityView(status="available", reason=None, scenarios=scenarios)
    return SensitivityView()


def _distribution_views(trace: dict[str, Any]) -> list[DistributionSummaryView]:
    rows = trace.get("simulation_distributions")
    if not isinstance(rows, list) or not rows:
        result = trace.get("result") or {}
        rows = result.get("simulation_distributions") or []
    views: list[DistributionSummaryView] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        target = item.get("target")
        dist_type = item.get("distribution_type")
        if not target or not dist_type:
            continue
        views.append(
            DistributionSummaryView(
                target=str(target),
                market=item.get("market"),
                stat_key=item.get("stat_key"),
                distribution_type=str(dist_type),
                sample_mean=item.get("sample_mean"),
                sample_std=item.get("sample_std"),
                p10=item.get("p10"),
                p50=item.get("p50"),
                p90=item.get("p90"),
                n_iterations=item.get("n_iterations"),
            )
        )
    return views


def _provenance_status(
    url: str | None, retrieved_at: str | None
) -> Literal["ok", "partial", "missing_provenance"]:
    if url and retrieved_at:
        return "ok"
    if url or retrieved_at:
        return "partial"
    return "missing_provenance"


def _source_views(trace: dict[str, Any]) -> list[SourceNoteView]:
    # Prefer the provenance-rich RSVG source summaries (gate audit v2); fall
    # back to the flat reasoning_inputs source labels for older traces.
    tq = trace.get("trace_quality") or {}
    rsvg = tq.get("rsvg") if isinstance(tq.get("rsvg"), dict) else {}
    summaries = rsvg.get("source_summaries") or []
    views: list[SourceNoteView] = []
    seen: set[str] = set()
    for item in summaries:
        if not isinstance(item, dict) or not item.get("source"):
            continue
        url = item.get("source_url")
        retrieved = item.get("retrieved_at")
        views.append(
            SourceNoteView(
                source=str(item["source"]),
                source_title=item.get("source_title"),
                source_url=url,
                retrieved_at=retrieved,
                provenance_status=_provenance_status(url, retrieved),
            )
        )
        seen.add(str(item["source"]))

    reasoning_inputs = trace.get("reasoning_inputs") or {}
    for src in reasoning_inputs.get("sources") or []:
        text = str(src)
        if text in seen:
            continue
        is_url = text.startswith(("http://", "https://"))
        views.append(
            SourceNoteView(
                source=text,
                source_url=text if is_url else None,
                provenance_status=_provenance_status(text if is_url else None, None),
            )
        )
    return views


def _legacy_presentation(
    trace: dict[str, Any], warnings: list[str]
) -> LegacyPresentationCompat | None:
    legacy = trace.get("reasoning_presentation")
    if not isinstance(legacy, dict) or not legacy:
        return None
    summary = _scrub_prose([legacy["thesis"]], warnings) if legacy.get("thesis") else []
    context = _scrub_prose([legacy["market_read"]], warnings) if legacy.get("market_read") else []
    uncertainties = _scrub_prose([legacy["risks"]], warnings) if legacy.get("risks") else []
    one_sided = _scrub_prose([legacy["why"]], warnings) if legacy.get("why") else []
    # The verdict is lab-only by contract — deliberately not mapped.
    if not (summary or context or uncertainties or one_sided):
        return None
    return LegacyPresentationCompat(
        summary=summary[0] if summary else None,
        market_context=context[0] if context else None,
        uncertainties=uncertainties,
        one_sided_case=one_sided[0] if one_sided else None,
    )


def _decision_support_payload(
    trace: dict[str, Any], warnings: list[str]
) -> DecisionSupportPresentationV1 | None:
    payload = trace.get("decision_support_presentation")
    if not isinstance(payload, dict) or not payload:
        return None
    try:
        return DecisionSupportPresentationV1(**payload)
    except Exception as exc:  # noqa: BLE001 - fail closed, keep the brief renderable
        warnings.append(f"decision_support_presentation failed validation and was withheld: {exc}")
        return None


def _data_quality_notes(trace: dict[str, Any], warnings: list[str]) -> list[str]:
    notes = list(warnings)
    tq = trace.get("trace_quality") or {}
    rsvg = tq.get("rsvg")
    if isinstance(rsvg, dict) and rsvg.get("status"):
        notes.append(f"rsvg_status:{rsvg['status']}")
    for downgrade in trace.get("downgrades") or []:
        notes.append(f"downgrade:{downgrade}")
    if trace.get("event_identity") is None:
        notes.append("identity_warning:no_provider_event_identity")
    return notes


def build_market_view(trace: dict[str, Any]) -> MarketBriefView:
    """Project one persisted trace (v1 or v2 dict shape) into a safe market view."""
    warnings: list[str] = []
    kind = str(trace.get("kind") or "unknown")
    input_snap = trace.get("input_snapshot") or {}
    mode, reasons = market_output_mode_for_trace(trace)
    authorized = mode != OutputMode.RESEARCH_CANDIDATE

    if kind == "prop":
        player = str(input_snap.get("player_name") or "")
        prop_type = str(input_snap.get("prop_type") or "prop")
        market_group = f"{player} {prop_type}".strip()
        probability_sets = [_prop_probability_set(trace, authorized)]
    elif kind == "game":
        market_group = "game"
        probability_sets = _game_probability_sets(trace, authorized)
    else:
        market_group = kind
        probability_sets = []

    identity = trace.get("event_identity")
    identity_date = identity.get("game_date") if isinstance(identity, dict) else None
    view = MarketBriefView(
        trace_id=str(trace.get("trace_id") or ""),
        kind=kind,
        market_group=market_group,
        league=trace.get("league"),
        matchup=str(trace.get("matchup") or ""),
        game_date=input_snap.get("game_date") or identity_date,
        output_mode=mode.value,
        output_mode_reasons=reasons,
        probability_sets=probability_sets,
        sensitivity=_sensitivity_view(trace),
        distributions=_distribution_views(trace),
        market_lines=dict(trace.get("odds_snapshot") or {}),
        aggregate_quality=trace.get("aggregate_quality"),
        sources=_source_views(trace),
        decision_support=_decision_support_payload(trace, warnings),
        legacy_presentation=_legacy_presentation(trace, warnings),
        data_quality=[],
    )
    view.data_quality = _data_quality_notes(trace, warnings)
    return view


def group_key_for_trace(trace: dict[str, Any]) -> tuple[str, str | None, bool]:
    """(group_key, event_key, identity_warning) for one persisted trace."""
    identity = trace.get("event_identity")
    if isinstance(identity, dict) and identity.get("event_key"):
        key = str(identity["event_key"])
        return key, key, False
    return f"trace:{trace.get('trace_id')}", None, True


def build_matchup_brief(traces: list[dict[str, Any]]) -> MatchupBriefV1:
    """Build the safe brief for one group of traces sharing an event identity.

    Markets are ordered by stable market identity (game first, then props by
    player/market name) — never by edge, EV, or recommendation status. The
    serialized DTO is swept for denied keys before being returned.
    """
    if not traces:
        raise ValueError("build_matchup_brief requires at least one trace")
    group_key, event_key, identity_warning = group_key_for_trace(traces[0])
    for trace in traces[1:]:
        other_key, _, _ = group_key_for_trace(trace)
        if other_key != group_key:
            raise ValueError(
                f"traces with different event identities cannot share a brief: "
                f"{group_key!r} vs {other_key!r}"
            )

    markets = [build_market_view(t) for t in traces]
    markets.sort(key=lambda m: (m.kind != "game", m.market_group.casefold(), m.trace_id))

    first = markets[0]
    brief = MatchupBriefV1(
        group_key=group_key,
        event_key=event_key,
        identity_warning=identity_warning,
        league=first.league,
        matchup=first.matchup,
        game_date=first.game_date,
        presentation_mode=coerce_presentation_mode(traces[0].get("presentation_mode")),
        markets=markets,
    )
    assert_no_denied_keys(brief.model_dump(mode="json"))
    return brief


def brief_for_group_key(
    store: Any, group_key: str, *, max_scan: int = 2000
) -> MatchupBriefV1 | None:
    """Resolve one brief by group key against a TraceStore-like object.

    ``trace:<trace_id>`` keys resolve the single legacy trace directly; event
    keys use a bounded scan of recent traces (no DB migration in Phase 0).
    Shared by the console service and the MCP matchup-brief tool so both
    surfaces render the identical projection.
    """
    if group_key.startswith("trace:"):
        trace = store.get_trace(group_key.removeprefix("trace:"))
        if trace is None:
            return None
        return build_matchup_brief([trace])
    traces = store.query_traces(limit=max_scan)
    matching = [t for t in traces if group_key_for_trace(t)[0] == group_key]
    if not matching:
        return None
    return build_matchup_brief(matching)


def group_traces_into_briefs(traces: list[dict[str, Any]]) -> list[MatchupBriefV1]:
    """Group persisted traces by event identity and build one brief per group.

    Legacy traces without provider identity become singleton groups with an
    identity warning — never heuristically merged. Briefs are ordered by
    (game_date, league, matchup, group_key): stable identity, recommendation-free.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for trace in traces:
        key, _, _ = group_key_for_trace(trace)
        groups.setdefault(key, []).append(trace)
    briefs = [build_matchup_brief(group) for group in groups.values()]
    briefs.sort(
        key=lambda b: (
            b.game_date or "",
            b.league or "",
            b.matchup,
            b.group_key,
        )
    )
    return briefs
