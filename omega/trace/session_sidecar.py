"""Versioned contract for `inbox/sessions/<session_id>.json` sidecars."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

_PROTECTED_QUANT_FIELDS: frozenset[str] = frozenset({
    "edge_pct",
    "ev_pct",
    "kelly_fraction",
    "units",
    "confidence_tier",
    "fair_price",
    "no_vig_price",
    "model_probability",
    "over_prob",
    "under_prob",
})

_VALID_EVENT_TYPES: frozenset[str] = frozenset({
    "preflight",
    "data_provenance",
    "engine_run",
    "candidate_rejected",
    "downgrade",
    "rationale",
    "bug",
    "command",
    "step",
    "note",
})

_VALID_STATUSES: frozenset[str] = frozenset({"ok", "warn", "fail", "skipped"})


class ProtectedValueError(ValueError):
    """Raised when an audit event contains engine-owned quant field names."""


class AuditEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ts: str = Field(min_length=1)
    event_type: str
    step: str
    status: str
    notes: str | None = None
    inputs: dict[str, Any] | None = None
    outputs: dict[str, Any] | None = None
    assumptions: list[str] = Field(default_factory=list)
    bugs: list[str] = Field(default_factory=list)
    trace_ids: list[str] = Field(default_factory=list)

    @field_validator("event_type")
    @classmethod
    def _validate_event_type(cls, v: str) -> str:
        if v not in _VALID_EVENT_TYPES:
            raise ValueError(f"event_type must be one of {sorted(_VALID_EVENT_TYPES)}, got {v!r}")
        return v

    @field_validator("status")
    @classmethod
    def _validate_status(cls, v: str) -> str:
        if v not in _VALID_STATUSES:
            raise ValueError(f"status must be one of {sorted(_VALID_STATUSES)}, got {v!r}")
        return v


class SessionSidecar(BaseModel):
    """Required session metadata used by calibration and operator reports."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1)
    opened_at: str = Field(min_length=1)
    closed_at: str | None = None
    model_version: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    bankroll: float = Field(gt=0)
    bankroll_confirmed: bool
    exec_stats: dict[str, Any]
    agent_notes: str
    audit_events: list[AuditEvent] = Field(default_factory=list)

    @field_validator("opened_at", "closed_at")
    @classmethod
    def _reject_legacy_empty_timestamps(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not value.strip():
            raise ValueError("timestamp must not be blank")
        return value

    @classmethod
    def from_path(cls, path: Path) -> SessionSidecar:
        with path.open("r", encoding="utf-8") as fh:
            return cls.model_validate(json.load(fh))

    def to_report_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def bootstrap_payload(
    session_id: str,
    *,
    model_version: str,
    purpose: str,
    bankroll: float,
    bankroll_confirmed: bool = False,
) -> dict[str, Any]:
    """Return a minimal dict suitable for ``SessionSidecar.model_validate``.

    Used by the audit renderer to synthesize an in-memory sidecar when no
    sidecar file exists, so the DB cross-section can still be rendered.
    """
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "session_id": session_id,
        "opened_at": now,
        "closed_at": None,
        "model_version": model_version,
        "purpose": purpose,
        "bankroll": bankroll,
        "bankroll_confirmed": bankroll_confirmed,
        "exec_stats": {},
        "agent_notes": "",
    }


def _check_protected_fields(event: AuditEvent) -> None:
    for field_name in ("inputs", "outputs"):
        mapping: dict[str, Any] | None = getattr(event, field_name)
        if mapping:
            for key in mapping:
                if key in _PROTECTED_QUANT_FIELDS:
                    raise ProtectedValueError(
                        f"audit_events '{field_name}' contains protected engine field {key!r}. "
                        "Engine-owned quant values must stay in omega_traces.db."
                    )


def append_audit_events(path: Path, events: list[dict[str, Any]]) -> None:
    """Atomically append audit events to a session sidecar.

    Reads the sidecar, validates and appends *events*, then writes back via
    temp-file + os.replace.  Raises ``ProtectedValueError`` if any event
    carries an engine-owned quant key in ``inputs`` or ``outputs``; the
    on-disk file is untouched on any error.
    """
    from omega.trace._atomic import atomic_write_text

    validated = [AuditEvent.model_validate(e) for e in events]
    for event in validated:
        _check_protected_fields(event)

    sidecar = SessionSidecar.from_path(path)
    sidecar.audit_events.extend(validated)
    atomic_write_text(path, json.dumps(sidecar.model_dump(mode="json"), indent=2))
