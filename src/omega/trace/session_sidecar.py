"""Versioned contract for `var/inbox/sessions/<session_id>.json` sidecars.

Authority order (see docs/phase6/ARTIFACT_AUTHORITY.md): the ledger
(`var/omega_traces.db`) is the source of truth for numbers/model state; this sidecar
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

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
    league: str | None = None
    window: str | None = None
    effective_db_path: str | None = None
    runtime_db_status: str | None = None
    pipeline_status: dict[str, Any] = Field(default_factory=dict)
    next_required_action: str | None = None
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
    now = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
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


def _find_protected_key(value: Any) -> str | None:
    """Recursively search for protected engine field keys in dicts, lists, etc."""
    if isinstance(value, dict):
        for k, v in value.items():
            if k in _PROTECTED_QUANT_FIELDS:
                return k
            found = _find_protected_key(v)
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_protected_key(item)
            if found:
                return found
    return None


def _check_protected_fields(event: AuditEvent) -> None:
    for field_name in ("inputs", "outputs"):
        mapping: dict[str, Any] | None = getattr(event, field_name)
        if mapping:
            found_key = _find_protected_key(mapping)
            if found_key:
                raise ProtectedValueError(
                    f"audit_events '{field_name}' contains protected engine field {found_key!r}. "
                    "Engine-owned quant values must stay in var/omega_traces.db."
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


# ---------------------------------------------------------------------------
# Trace-scoped quality-gate verdict (Phase: trace-scoped QA ingest)
# ---------------------------------------------------------------------------
#
# quality_gate_status() above is session-wide: one failed gate condemns every
# trace in the session. That over-blocks valid traces that ran in a session
# where some *other* trace failed QA. quality_gate_verdict_for_trace() scopes
# the verdict to a single trace using the signals actually available on a
# sidecar — the per-event ``trace_ids`` list, event timestamps vs. the trace's
# ``ran_at``, and pre-trace setup failures — and falls back conservatively only
# when no scoping is possible. The old function is intentionally kept for
# callers that still want the blunt session verdict.

QaVerdict = Literal["pass", "fail", "unknown"]
QaScope = Literal[
    "trace_id",
    "timestamp_window",
    "pre_trace_fatal",
    "session_fallback",
    "unrelated_session_failure",
    "no_sidecar",
]

_VALID_QA_VERDICTS: frozenset[str] = frozenset({"pass", "fail", "unknown"})
_VALID_QA_SCOPES: frozenset[str] = frozenset({
    "trace_id",
    "timestamp_window",
    "pre_trace_fatal",
    "session_fallback",
    "unrelated_session_failure",
    "no_sidecar",
})

# Secondary, deliberately tight tolerance for the timestamp_window matcher. A
# trace has no recorded execution duration, and QA gates normally fire shortly
# after the analysis they grade, so we only tie an *unscoped* failed gate to a
# trace when their timestamps are within this many seconds. trace_id matching
# always takes precedence; this matcher exists only to catch unstructured gates
# emitted right next to the trace.
_QA_TIMESTAMP_TOLERANCE_SECONDS = 300.0


@dataclass(frozen=True)
class TraceQaVerdict:
    """Trace-scoped quality-gate verdict.

    ``verdict`` is the tri-state result; ``scope`` records *how* the verdict was
    derived so callers (and the ``trace_qa_verdicts`` ledger table) can audit
    whether a fail was trace-specific or a conservative session-wide fallback.
    A ``fail`` never blocks ledger ingest — it only marks the trace
    calibration-ineligible.
    """

    verdict: QaVerdict
    scope: QaScope
    reason: str | None = None
    gate_name: str | None = None
    event_id: str | None = None
    matched_trace_id: str | None = None


def _parse_event_ts(value: Any) -> datetime.datetime | None:
    """Parse an ISO-8601 timestamp (tolerating a trailing ``Z``) to aware UTC."""
    if value is None:
        return None
    if isinstance(value, datetime.datetime):
        return value if value.tzinfo else value.replace(tzinfo=datetime.timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=datetime.timezone.utc)


def quality_gate_verdict_for_trace(
    sidecar: SessionSidecar | None,
    trace_id: str | None,
    ran_at: Any = None,
    *,
    window_tolerance_seconds: float = _QA_TIMESTAMP_TOLERANCE_SECONDS,
) -> TraceQaVerdict:
    """Scope a session's quality-gate history to a single trace.

    Matching priority (first match wins):

    1. No sidecar                              -> unknown / no_sidecar
    2. Failed gate names this trace_id         -> fail / trace_id
    3. Passed/repaired gate names this trace   -> pass / trace_id
    4. Unscoped failed gate within ran_at +/- tolerance -> fail / timestamp_window
    5. Pre-trace fatal preflight, no recovery before ran_at -> fail / pre_trace_fatal
    6. Failed gates only reference *other* traces -> pass / unrelated_session_failure
    7. Unstructured failed gate, no scoping possible -> fail / session_fallback

    ``session_fallback`` is the conservative catch-all for legacy/unstructured
    sidecars: it marks the trace calibration-ineligible but, like every fail
    here, never blocks ledger ingest.
    """
    if sidecar is None:
        return TraceQaVerdict(
            verdict="unknown",
            scope="no_sidecar",
            reason="no session sidecar available; QA history unknown",
        )

    tid = str(trace_id) if trace_id else None
    gate_events = [e for e in sidecar.audit_events if e.event_type == "quality_gate"]
    failed_gates = [e for e in gate_events if e.status == "fail"]

    # 2. Failed gate explicitly references this trace_id.
    if tid:
        for event in failed_gates:
            if tid in (event.trace_ids or []):
                return TraceQaVerdict(
                    verdict="fail",
                    scope="trace_id",
                    reason=f"quality_gate '{event.step}' failed and names this trace",
                    gate_name=event.step,
                    matched_trace_id=tid,
                )
        # 3. Passed/repaired gate explicitly references this trace_id.
        for event in gate_events:
            if event.status in ("ok", "skipped") and tid in (event.trace_ids or []):
                return TraceQaVerdict(
                    verdict="pass",
                    scope="trace_id",
                    reason=f"quality_gate '{event.step}' passed for this trace",
                    gate_name=event.step,
                    matched_trace_id=tid,
                )

    # Partition the remaining failed gates by whether they name *other* traces.
    # A gate with an empty trace_ids list is "unstructured": it tells us a
    # failure happened but not which traces it taints.
    unscoped_fails = [e for e in failed_gates if not (e.trace_ids or [])]
    scoped_other_fails = [
        e for e in failed_gates if (e.trace_ids or []) and tid not in (e.trace_ids or [])
    ]

    ran_dt = _parse_event_ts(ran_at)

    # 4. Unscoped failed gate within a tight window of the trace's ran_at. Only
    #    unscoped gates are time-matched: a gate that names other traces is
    #    evidence it is scoped elsewhere, not to this trace.
    if ran_dt is not None:
        for event in unscoped_fails:
            ev_dt = _parse_event_ts(event.ts)
            if ev_dt is None:
                continue
            if abs((ev_dt - ran_dt).total_seconds()) <= window_tolerance_seconds:
                return TraceQaVerdict(
                    verdict="fail",
                    scope="timestamp_window",
                    reason=(
                        f"unscoped quality_gate '{event.step}' failed within "
                        f"{window_tolerance_seconds:.0f}s of this trace's ran_at"
                    ),
                    gate_name=event.step,
                )

    # 5. Pre-trace fatal setup/preflight failure with no later recovery before
    #    ran_at. A clean preflight/quality_gate after the fatal but before the
    #    trace ran means the session recovered and the trace is not poisoned.
    if ran_dt is not None:
        pre_fatals = [
            e
            for e in sidecar.audit_events
            if e.event_type == "preflight"
            and e.status == "fail"
            and (dt := _parse_event_ts(e.ts)) is not None
            and dt < ran_dt
        ]
        if pre_fatals:
            last_fatal_dt = max(_parse_event_ts(e.ts) for e in pre_fatals)
            recovered = any(
                e.event_type in ("preflight", "quality_gate")
                and e.status == "ok"
                and (dt := _parse_event_ts(e.ts)) is not None
                and last_fatal_dt < dt <= ran_dt
                for e in sidecar.audit_events
            )
            if not recovered:
                return TraceQaVerdict(
                    verdict="fail",
                    scope="pre_trace_fatal",
                    reason="fatal preflight failure before this trace with no recovery",
                    gate_name="preflight",
                )

    # 7 (before 6): an unstructured failed gate taints the whole session because
    #    we cannot prove it is unrelated. This is the conservative fallback and
    #    takes precedence over an unrelated scoped failure when both exist.
    if unscoped_fails:
        event = unscoped_fails[0]
        return TraceQaVerdict(
            verdict="fail",
            scope="session_fallback",
            reason=(
                f"unstructured quality_gate '{event.step}' failed with no trace/time "
                "scoping; conservatively marking calibration-ineligible"
            ),
            gate_name=event.step,
        )

    # 6. Every failed gate references only other traces -> this trace is clean.
    if scoped_other_fails:
        return TraceQaVerdict(
            verdict="pass",
            scope="unrelated_session_failure",
            reason="failed quality_gate(s) reference only other traces",
        )

    # No failed gates at all.
    return TraceQaVerdict(
        verdict="pass",
        scope="unrelated_session_failure",
        reason="no failed quality_gate events in session",
    )


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
    """Reconstruct the audit-event stream and session metadata from a JSONL mirror (recovery helper).

    Returns a full dictionary that matches the SessionSidecar schema for clean recovery,
    while preserving 'event_count' and 'source_jsonl' for diagnostic compatibility.
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

    # Derive session_id from filename (e.g. sess-20260528-zzzz.events.jsonl -> sess-20260528-zzzz)
    session_id = jsonl_path.name.split(".")[0] if jsonl_path else "unknown_recovered"
    if not session_id or session_id == "unknown_recovered":
        session_id = "unknown_recovered"

    # Estimate opened_at from the first event timestamp, fallback to now
    opened_at = events[0].get("ts") if events else None
    if not opened_at:
        opened_at = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    return {
        "session_id": session_id,
        "opened_at": opened_at,
        "closed_at": None,
        "model_version": "unknown",
        "purpose": "recovered_session",
        "bankroll": 1000.0,
        "bankroll_confirmed": False,
        "exec_stats": {},
        "agent_notes": "Recovered from mirror JSONL file.",
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
    now = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
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
