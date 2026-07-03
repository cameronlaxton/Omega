"""Pre-analysis verification gates.

Gates in this package validate STRUCTURED facts an operator/LLM has already
gathered and translate them into typed contract objects (``EvidenceSignal``,
``ReasoningPresentation``) plus an auditable gate verdict — BEFORE ``analyze()``
runs. They are pure: no network, no DB, no engine math.

- ``rsvg`` — Roster & Situational Verification Gate (roster/injury/lineup/
  motivation context).
"""

from __future__ import annotations

from omega.core.gates.rsvg import (
    RSVG_GATE_SCHEMA_VERSION,
    PlayerAbsence,
    RosterContextPayload,
    RsvgGateAudit,
    RsvgProtectedFieldError,
    RsvgResult,
    SourceSummary,
    TeamRosterStatus,
    evaluate_roster_context,
)

__all__ = [
    "RSVG_GATE_SCHEMA_VERSION",
    "PlayerAbsence",
    "RosterContextPayload",
    "RsvgGateAudit",
    "RsvgProtectedFieldError",
    "RsvgResult",
    "SourceSummary",
    "TeamRosterStatus",
    "evaluate_roster_context",
]
