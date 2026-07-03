"""RSVG — Roster & Situational Verification Gate.

A typed pre-analysis gate that makes roster/injury/lineup/motivation context
auditable BEFORE ``analyze()`` runs. The LLM/operator gathers web context and
condenses it into a :class:`RosterContextPayload`; this module only validates
and translates those structured facts. It never browses, never touches the DB,
and never computes engine-owned quantities (edge, EV, Kelly, units, confidence
tier, fair/no-vig price, model probability, trace_id) — every payload it emits
is scanned and the gate fails closed if a protected key ever appears.

Verdicts (severity-ordered)::

    pass                Context verified; formal analysis may proceed. Non-key
                        absences and ≤2 key absences per team stay here (each
                        key absence still emits a typed ``usage_role_change``
                        signal so the engine can react deterministically).
    research_candidate  Formal actionable output is suppressed: >2 key absences
                        on one team, incomplete/stale/partially-missing roster
                        context. ``reasoning_downgrade_rationale`` says why.
    blocked             A team's roster context is entirely unverifiable
                        (lineup unknown AND injury report unchecked). Callers
                        must not run analyze() for the entry.

Integration seam: ``omega_run_batch`` accepts an optional
``BatchAnalysisEntry.roster_context`` dict; the batch loop runs this gate before
odds resolution, skips ``blocked`` entries, merges emitted signals into the
entry's evidence, and stamps ``reasoning_downgrade_rationale`` +
``trace_quality.rsvg`` onto the trace/export block. The gate is also usable
standalone via :func:`evaluate_roster_context` +
:meth:`RsvgResult.to_batch_entry_fields`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omega.core.contracts.evidence import EvidenceSignal
from omega.core.contracts.protected_fields import PROTECTED_QUANT_FIELDS, find_protected_key
from omega.core.contracts.schemas import ReasoningPresentation

RSVG_GATE_SCHEMA_VERSION = 1

# Policy knobs (overridable per-call; recorded in the audit so every verdict is
# reproducible from its own metadata).
MAX_KEY_ABSENCES_PER_TEAM = 2
DEFAULT_MAX_CONTEXT_AGE_HOURS = 24.0

RsvgStatus = Literal["pass", "research_candidate", "blocked"]
TeamSide = Literal["home", "away"]
LineupStatus = Literal["confirmed", "projected", "unknown"]
AbsenceStatus = Literal["out", "suspended", "doubtful", "questionable", "day_to_day"]

# Statuses that count a player as MISSING for signal emission and the per-team
# key-absence threshold. questionable/day_to_day are uncertainty notes only.
_MISSING_STATUSES: frozenset[str] = frozenset({"out", "suspended", "doubtful"})
_KEY_ABSENCE_CONFIDENCE: dict[str, float] = {"out": 0.9, "suspended": 0.9, "doubtful": 0.6}

# Engine-owned quant fields this gate must never emit: the canonical set plus
# gate-specific extras (aliases and identifiers the gate must not fabricate).
_PROTECTED_QUANT_FIELDS: frozenset[str] = PROTECTED_QUANT_FIELDS | frozenset(
    {
        "edge",
        "ev",
        "kelly",
        "recommended_units",
        "true_prob",
        "calibrated_prob",
        "market_implied",
        "trace_id",
    }
)


class RsvgProtectedFieldError(ValueError):
    """Raised when a gate payload would carry an engine-owned quant field."""


def _assert_no_protected_fields(label: str, mapping: Any) -> None:
    found = find_protected_key(mapping, fields=_PROTECTED_QUANT_FIELDS)
    if found:
        raise RsvgProtectedFieldError(
            f"RSVG {label} contains protected engine field {found!r}. "
            "The gate validates context only; engine-owned numeric values are "
            "computed downstream by analyze()."
        )


# ---------------------------------------------------------------------------
# Input payload
# ---------------------------------------------------------------------------


class SourceSummary(BaseModel):
    """One consulted source, condensed by the operator/LLM outside this module."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=1, description="Source label, e.g. 'espn.com', 'rotowire'.")
    summary: str = Field(min_length=1, description="Condensed factual summary (prose).")
    retrieved_at: str | None = Field(
        default=None, description="ISO-8601 timestamp the source was consulted."
    )


class PlayerAbsence(BaseModel):
    """A reported absence or availability doubt for one player."""

    model_config = ConfigDict(extra="forbid")

    player: str = Field(min_length=1)
    team_side: TeamSide
    status: AbsenceStatus = Field(
        description="out/suspended/doubtful count as missing; "
        "questionable/day_to_day are uncertainty notes only."
    )
    is_key_player: bool = Field(
        default=False,
        description="Operator/LLM judgment that the player materially drives team output.",
    )
    role_note: str | None = Field(
        default=None, description="Optional role context, e.g. 'starting goalie', 'QB1'."
    )


class TeamRosterStatus(BaseModel):
    """Verification state of one team's roster/lineup context."""

    model_config = ConfigDict(extra="forbid")

    lineup_status: LineupStatus = Field(
        default="unknown", description="confirmed | projected | unknown."
    )
    injury_report_checked: bool = Field(
        default=False, description="Whether an injury report was actually consulted."
    )


class RosterContextPayload(BaseModel):
    """Structured matchup context handed to the gate.

    ``extra='forbid'`` on every model here is load-bearing: protected betting
    fields (edge_pct, kelly_fraction, ...) are structurally rejected at parse
    time, not filtered later.
    """

    model_config = ConfigDict(extra="forbid")

    home_team: str = Field(min_length=1)
    away_team: str = Field(min_length=1)
    league: str = Field(min_length=1)
    game_date: str = Field(min_length=10, description="ISO date (YYYY-MM-DD).")
    source_summaries: list[SourceSummary] = Field(default_factory=list)
    home_status: TeamRosterStatus = Field(default_factory=TeamRosterStatus)
    away_status: TeamRosterStatus = Field(default_factory=TeamRosterStatus)
    absences: list[PlayerAbsence] = Field(default_factory=list)
    motivation_notes: str | None = Field(
        default=None, description="Situational/motivation prose (rivalry, seeding, must-win)."
    )
    gathered_at: str | None = Field(
        default=None, description="ISO-8601 timestamp the context was gathered (freshness anchor)."
    )
    roster_context_complete: bool = Field(
        default=False,
        description="Explicit operator attestation that roster context gathering is complete.",
    )


# ---------------------------------------------------------------------------
# Gate output
# ---------------------------------------------------------------------------


class RsvgGateAudit(BaseModel):
    """Persistable gate verdict metadata (goes under ``trace_quality.rsvg``)."""

    model_config = ConfigDict(extra="forbid")

    gate: Literal["rsvg"] = "rsvg"
    schema_version: int = RSVG_GATE_SCHEMA_VERSION
    status: RsvgStatus
    evaluated_at: str
    matchup: str
    league: str
    game_date: str
    lineup_status: dict[str, str]
    injury_report_checked: dict[str, bool]
    key_absences: dict[str, list[str]]
    non_key_absences: dict[str, list[str]]
    uncertain_key_players: dict[str, list[str]]
    roster_context_complete: bool
    sources: list[str]
    stale_sources: list[str]
    context_age_hours: float | None
    missing_context: list[str]
    blocked_reasons: list[str]
    downgrade_reasons: list[str]
    notes: list[str]
    policy: dict[str, float]
    formal_output_allowed: bool
    output_mode_ceiling: Literal["unrestricted", "research_candidate", "blocked"]


class RsvgResult(BaseModel):
    """Full gate result: verdict + typed evidence + analyst-note prose."""

    model_config = ConfigDict(extra="forbid")

    status: RsvgStatus
    evidence: list[EvidenceSignal]
    reasoning_presentation: ReasoningPresentation
    reasoning_downgrade_rationale: str | None
    reasoning_sources: list[str]
    gate_audit: RsvgGateAudit

    @model_validator(mode="after")
    def _no_protected_fields(self) -> RsvgResult:
        """Enforce the no-engine-values contract on every construction path."""
        _assert_no_protected_fields("gate_audit", self.gate_audit.model_dump())
        _assert_no_protected_fields(
            "reasoning_presentation", self.reasoning_presentation.model_dump()
        )
        _assert_no_protected_fields("evidence", self.evidence_dicts())
        return self

    @property
    def formal_output_allowed(self) -> bool:
        return self.status == "pass"

    def evidence_dicts(self) -> list[dict[str, Any]]:
        """Evidence as plain dicts, the shape ``BatchAnalysisEntry.evidence`` takes."""
        return [s.model_dump(exclude_none=True) for s in self.evidence]

    def to_batch_entry_fields(self) -> dict[str, Any]:
        """BatchAnalysisEntry-compatible fragment for callers building entries by hand.

        ``omega_run_batch`` does this merge itself when the entry carries a
        ``roster_context``; standalone callers can splice this into an entry dict.
        """
        return {
            "evidence": self.evidence_dicts(),
            "reasoning_presentation": self.reasoning_presentation.model_dump(),
            "reasoning_sources": list(self.reasoning_sources),
        }

    def trace_quality_fragment(self) -> dict[str, Any]:
        """Fragment to merge into a trace's ``trace_quality`` dict for audit."""
        return {"rsvg": self.gate_audit.model_dump()}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _parse_ts(value: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _team_name(payload: RosterContextPayload, side: str) -> str:
    return payload.home_team if side == "home" else payload.away_team


def evaluate_roster_context(
    payload: RosterContextPayload,
    *,
    evaluated_at: datetime | None = None,
    max_key_absences_per_team: int = MAX_KEY_ABSENCES_PER_TEAM,
    max_context_age_hours: float = DEFAULT_MAX_CONTEXT_AGE_HOURS,
) -> RsvgResult:
    """Evaluate structured roster/situational context and return the gate verdict.

    Pure and deterministic given ``evaluated_at`` (defaults to now-UTC, used only
    for freshness checks). Raises :class:`RsvgProtectedFieldError` if any emitted
    payload would carry an engine-owned quant field.
    """
    now = evaluated_at or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    missing_context: list[str] = []
    blocked_reasons: list[str] = []
    downgrade_reasons: list[str] = []
    notes: list[str] = []

    # --- Required-context checks -------------------------------------------
    statuses = {"home": payload.home_status, "away": payload.away_status}
    for side, st in statuses.items():
        if st.lineup_status == "unknown":
            missing_context.append(f"{side}.lineup_status")
        if not st.injury_report_checked:
            missing_context.append(f"{side}.injury_report")
        if st.lineup_status == "unknown" and not st.injury_report_checked:
            blocked_reasons.append(
                f"{_team_name(payload, side)} roster context unavailable "
                "(lineup unknown and injury report unchecked)"
            )
    if not payload.roster_context_complete:
        missing_context.append("roster_context_complete")
    if not payload.source_summaries:
        missing_context.append("source_summaries")

    # --- Freshness -----------------------------------------------------------
    context_age_hours: float | None = None
    if payload.gathered_at is None:
        missing_context.append("gathered_at")
    else:
        gathered = _parse_ts(payload.gathered_at)
        if gathered is None:
            missing_context.append("gathered_at:unparseable")
        else:
            context_age_hours = (now - gathered).total_seconds() / 3600.0
            if context_age_hours > max_context_age_hours:
                downgrade_reasons.append(
                    f"context stale: gathered {context_age_hours:.1f}h ago "
                    f"(max {max_context_age_hours:g}h)"
                )
            elif context_age_hours < 0:
                notes.append(f"gathered_at is {-context_age_hours:.1f}h in the future")

    stale_sources: list[str] = []
    for src in payload.source_summaries:
        if src.retrieved_at is None:
            continue
        retrieved = _parse_ts(src.retrieved_at)
        if retrieved is None:
            notes.append(f"source {src.source}: retrieved_at unparseable")
        elif (now - retrieved).total_seconds() / 3600.0 > max_context_age_hours:
            stale_sources.append(src.source)
            notes.append(f"source {src.source}: stale (> {max_context_age_hours:g}h old)")

    if missing_context:
        downgrade_reasons.append(f"missing roster context: {', '.join(missing_context)}")

    # --- Absences ------------------------------------------------------------
    key_absences: dict[str, list[str]] = {"home": [], "away": []}
    non_key_absences: dict[str, list[str]] = {"home": [], "away": []}
    uncertain_key_players: dict[str, list[str]] = {"home": [], "away": []}
    evidence: list[EvidenceSignal] = []

    for absence in payload.absences:
        side = absence.team_side
        team = _team_name(payload, side)
        if absence.status not in _MISSING_STATUSES:
            if absence.is_key_player:
                uncertain_key_players[side].append(absence.player)
                notes.append(f"key player uncertain: {absence.player} ({absence.status}, {team})")
            else:
                notes.append(f"non-key uncertainty: {absence.player} ({absence.status}, {team})")
            continue
        if not absence.is_key_player:
            non_key_absences[side].append(absence.player)
            notes.append(f"non-key absence: {absence.player} ({absence.status}, {team})")
            continue

        key_absences[side].append(absence.player)
        note = f"RSVG key absence: {absence.player} ({absence.status}, {team})"
        if absence.role_note:
            note += f" — {absence.role_note}"
        # Each missing key player becomes a typed, retrospectively-scoreable
        # usage_role_change signal favoring the opponent at the game plane.
        evidence.append(
            EvidenceSignal(
                signal_type="usage_role_change",
                category="situational",
                plane="game",
                value=f"key_absence:{absence.status}",
                source="injury_report" if statuses[side].injury_report_checked else "agent_reasoning",
                confidence=_KEY_ABSENCE_CONFIDENCE[absence.status],
                window="matchup",
                direction="away" if side == "home" else "home",
                note=note,
            )
        )

    for side in ("home", "away"):
        n = len(key_absences[side])
        if n > max_key_absences_per_team:
            downgrade_reasons.append(
                f"{_team_name(payload, side)} has {n} missing key players "
                f"(> {max_key_absences_per_team} allowed): {', '.join(key_absences[side])}"
            )

    # --- Verdict ---------------------------------------------------------------
    if blocked_reasons:
        status: RsvgStatus = "blocked"
        ceiling: Literal["unrestricted", "research_candidate", "blocked"] = "blocked"
    elif downgrade_reasons:
        status = "research_candidate"
        ceiling = "research_candidate"
    else:
        status = "pass"
        ceiling = "unrestricted"

    all_reasons = blocked_reasons + downgrade_reasons
    rationale = None if status == "pass" else f"RSVG {status}: " + "; ".join(all_reasons)

    # --- Presentation (qualitative prose only) ---------------------------------
    sources: list[str] = []
    for src in payload.source_summaries:
        if src.source not in sources:
            sources.append(src.source)

    why_parts = [
        f"Lineups: home {payload.home_status.lineup_status}, "
        f"away {payload.away_status.lineup_status}.",
        "Injury reports checked: "
        f"home {'yes' if payload.home_status.injury_report_checked else 'no'}, "
        f"away {'yes' if payload.away_status.injury_report_checked else 'no'}.",
    ]
    total_key = len(key_absences["home"]) + len(key_absences["away"])
    total_non_key = len(non_key_absences["home"]) + len(non_key_absences["away"])
    if total_key or total_non_key:
        why_parts.append(
            f"Absences verified: {total_key} key, {total_non_key} non-key "
            f"(key: {', '.join(key_absences['home'] + key_absences['away']) or 'none'})."
        )
    else:
        why_parts.append("No absences reported.")
    if payload.motivation_notes:
        why_parts.append(f"Motivation/situational: {payload.motivation_notes}")
    if payload.source_summaries:
        # AGENTS.md RSVG rule: roster context, absent players, and news summaries
        # must persist under reasoning_presentation inside the trace.
        why_parts.append(
            "Source summaries: "
            + " | ".join(f"{s.source}: {s.summary}" for s in payload.source_summaries)
        )

    risk_parts = list(notes)
    if all_reasons:
        risk_parts.extend(all_reasons)
    risks = " ".join(risk_parts) if risk_parts else "No roster or situational risks identified."

    if status == "pass":
        verdict = "PASS — roster and situational context verified; formal analysis may proceed."
    elif status == "research_candidate":
        verdict = (
            "RESEARCH_CANDIDATE — no formal actionable output: " + "; ".join(all_reasons) + "."
        )
    else:
        verdict = "BLOCKED — do not run formal analysis: " + "; ".join(all_reasons) + "."

    presentation = ReasoningPresentation(
        thesis=(
            f"RSVG roster & situational verification for {payload.away_team} @ "
            f"{payload.home_team} ({payload.league}, {payload.game_date})."
        ),
        # market_read stays None by design: RSVG performs no market evaluation;
        # downstream reasoning may fill it after analyze().
        market_read=None,
        why=" ".join(why_parts),
        risks=risks,
        verdict=verdict,
    )

    audit = RsvgGateAudit(
        status=status,
        evaluated_at=now.isoformat(),
        matchup=f"{payload.away_team} @ {payload.home_team}",
        league=payload.league,
        game_date=payload.game_date,
        lineup_status={s: st.lineup_status for s, st in statuses.items()},
        injury_report_checked={s: st.injury_report_checked for s, st in statuses.items()},
        key_absences=key_absences,
        non_key_absences=non_key_absences,
        uncertain_key_players=uncertain_key_players,
        roster_context_complete=payload.roster_context_complete,
        sources=sources,
        stale_sources=stale_sources,
        context_age_hours=context_age_hours,
        missing_context=missing_context,
        blocked_reasons=blocked_reasons,
        downgrade_reasons=downgrade_reasons,
        notes=notes,
        policy={
            "max_key_absences_per_team": float(max_key_absences_per_team),
            "max_context_age_hours": float(max_context_age_hours),
        },
        formal_output_allowed=status == "pass",
        output_mode_ceiling=ceiling,
    )

    result = RsvgResult(
        status=status,
        evidence=evidence,
        reasoning_presentation=presentation,
        reasoning_downgrade_rationale=rationale,
        reasoning_sources=sources,
        gate_audit=audit,
    )

    return result
