"""Versioned contract for `inbox/sessions/<session_id>.json` sidecars.

Authority order (see docs/phase6/ARTIFACT_AUTHORITY.md): the ledger
(`omega_traces.db`) is the source of truth for numbers/model state; this sidecar
is a derived session summary — the human session view (`exec_stats`,
`agent_notes`, the `audit_events` narrative) and never the source of truth for
quant values. The sibling `<session_id>.events.jsonl` is a recovery mirror only
and is NOT promoted to canonical here. A human inspecting a failed run reads the
rendered audit, then the JSONL mirror if the sidecar was quarantined; scripts
trust the ledger.
"""

from __future__ import annotations

import datetime
import json
import logging
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger("omega.trace.session_sidecar")

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
    "quality_gate",
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
    """Return a minimal dict suitable for SessionSidecar.model_validate."""
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


def _events_jsonl_path(path: Path) -> Path:
    """Sibling diagnostic event log: ``<session_id>.events.jsonl``."""
    return path.with_suffix(".events.jsonl")


def _mirror_events_jsonl(path: Path, events: list[dict[str, Any]]) -> None:
    """Append events to the diagnostic JSONL mirror (write-only this phase).

    Append-only and best-effort: a mirror failure must never break the primary
    sidecar write. The mirror preserves the event stream for recovery even if
    the JSON summary is later truncated/quarantined. It is NOT a read path —
    reports continue to read the JSON summary.
    """
    if not events:
        return
    try:
        jsonl = _events_jsonl_path(path)
        jsonl.parent.mkdir(parents=True, exist_ok=True)
        with jsonl.open("a", encoding="utf-8", newline="\n") as fh:
            for event in events:
                fh.write(json.dumps(event) + "\n")
    except OSError as exc:
        logger.warning("JSONL mirror append failed for %s: %s", path.name, exc)


def write_sidecar(path: Path, sidecar: SessionSidecar) -> None:
    """Atomically write a full sidecar (temp + fsync + replace)."""
    from omega.trace._atomic import atomic_write_text

    atomic_write_text(path, json.dumps(sidecar.model_dump(mode="json"), indent=2))


def create_sidecar(path: Path, payload: dict[str, Any]) -> SessionSidecar:
    """Validate ``payload`` and atomically create the sidecar + its JSONL mirror.

    Replaces non-atomic ``path.write_text(json.dumps(...))`` session creation,
    which was a truncation source.
    """
    sidecar = SessionSidecar.model_validate(payload)
    write_sidecar(path, sidecar)
    _mirror_events_jsonl(path, [e.model_dump(mode="json") for e in sidecar.audit_events])
    return sidecar


def append_audit_events(path: Path, events: list[dict[str, Any]]) -> None:
    """Atomically append audit events to a session sidecar (+ JSONL mirror)."""
    from omega.trace._atomic import atomic_write_text

    validated = [AuditEvent.model_validate(e) for e in events]
    for event in validated:
        _check_protected_fields(event)

    sidecar = SessionSidecar.from_path(path)
    sidecar.audit_events.extend(validated)
    atomic_write_text(path, json.dumps(sidecar.model_dump(mode="json"), indent=2))
    _mirror_events_jsonl(path, [e.model_dump(mode="json") for e in validated])


def load_sidecar_safe(path: Path) -> SessionSidecar | None:
    """Read a sidecar, returning None on any parse/validation error.

    Warn-only and side-effect-free — safe for read-only report scripts. It does
    NOT move or mark the file (quarantine is an explicit operator action via
    ``validate_session_sidecars.py --quarantine``). A None result means the
    sidecar's quality-gate history is UNKNOWN, never "clean" (see
    :func:`quality_gate_status`).
    """
    try:
        return SessionSidecar.from_path(path)
    except Exception as exc:  # noqa: BLE001 — any malformed/truncated/invalid file
        logger.warning(
            "Sidecar %s is unreadable (%s: %s); treating quality-gate history as "
            "UNKNOWN. Quarantine with `validate_session_sidecars.py --quarantine`.",
            path.name,
            type(exc).__name__,
            exc,
        )
        return None


def quality_gate_status(sidecar: SessionSidecar | None) -> str:
    """Tri-state quality-gate verdict: 'pass' | 'fail' | 'unknown'.

    An unreadable sidecar (None) is 'unknown' — never implied-clean. Callers must
    not let 'unknown' silently pass as 'no failure'.
    """
    if sidecar is None:
        return "unknown"
    for event in sidecar.audit_events:
        if event.event_type == "quality_gate" and event.status == "fail":
            return "fail"
    return "pass"


def quarantine_sidecar(
    path: Path,
    reason: str,
    *,
    quarantine_dir: Path | None = None,
) -> Path | None:
    """Move a malformed sidecar to ``invalid/`` and record the reason. Idempotent.

    The JSONL mirror is intentionally left in place so the event stream survives
    for recovery. Returns the quarantine destination, or None if the file is
    already quarantined / does not exist.
    """
    if not path.exists():
        return None
    quarantine_dir = quarantine_dir or (path.parent / "invalid")
    # Idempotent: already under an invalid/ dir → no-op.
    if path.parent.name == "invalid" or path.parent == quarantine_dir:
        return None
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    dst = quarantine_dir / path.name
    if dst.exists():
        dst = quarantine_dir / f"{path.stem}.{uuid.uuid4().hex[:8]}{path.suffix}"
    import shutil

    shutil.move(str(path), str(dst))
    reason_path = dst.with_suffix(dst.suffix + ".reason.txt")
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    reason_path.write_text(f"{ts}\n{reason}\n", encoding="utf-8")
    logger.warning("Quarantined malformed sidecar %s -> %s", path.name, dst)
    return dst


def rebuild_sidecar_from_jsonl(jsonl_path: Path) -> dict[str, Any]:
    """Reconstruct the audit-event stream from a JSONL mirror (recovery helper).

    Returns ``{"audit_events": [...], "event_count": n, "source_jsonl": str}``.
    Diagnostic only — not wired into normal reads.
    """
    events: list[dict[str, Any]] = []
    if jsonl_path.exists():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSONL line in %s", jsonl_path.name)
    return {
        "audit_events": events,
        "event_count": len(events),
        "source_jsonl": str(jsonl_path),
    }


def append_null_data_audit(
    path: Path,
    missing_variables: list[str],
    *,
    critical: bool = False,
    trace_ids: list[str] | None = None,
) -> None:
    """Append a structured NULL/missing-data audit event.

    ``missing_variables`` must contain variable names or paths only. Protected
    engine-owned numeric values remain in traces/ledger rows, not sidecar prose.
    """
    now = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()
    variables = [str(v) for v in missing_variables if str(v).strip()]
    notes = (
        "NULL detected: " + ", ".join(variables)
        if variables
        else "NULL audit ran: no missing variables detected"
    )
    append_audit_events(
        path,
        [
            {
                "ts": now,
                "event_type": "quality_gate",
                "step": "null_data_audit",
                "status": "fail" if critical else "warn",
                "notes": notes,
                "trace_ids": trace_ids or [],
            }
        ],
    )
