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

import contextlib
import datetime
import json
import logging
import os
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger("omega.trace.session_sidecar")

# --- Sidecar concurrency lock (fixes SIDECAR_LOGGING_AUDIT_2026-06-07 F2) -------
# append_audit_events() is an unlocked read-modify-write: two near-simultaneous
# callers can each read the sidecar, extend in memory, and write back, with the
# second write clobbering the first's appended event. A sidecar-scoped advisory
# lock serializes the R-M-W critical section. Stdlib only (os.open O_CREAT|O_EXCL
# is atomic on both POSIX and Windows) — no new dependency, matching the engine's
# numpy+pydantic-only policy. The atomic write primitive (_atomic.py) is left
# untouched; per its own comment, cross-process locking "belongs elsewhere" — here.
_LOCK_SUFFIX = ".lock"
_LOCK_TIMEOUT_SECONDS = 10.0
_LOCK_POLL_SECONDS = 0.05


def _lock_owner_payload() -> bytes:
    payload = {
        "pid": os.getpid(),
        "created_at": time.time(),
        "lock_version": 1,
    }
    return json.dumps(payload, sort_keys=True).encode("utf-8")


def _read_lock_owner(lock_path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        pid = int(payload["pid"])
    except (KeyError, TypeError, ValueError):
        return None
    return {"pid": pid, "created_at": payload.get("created_at")}


def _process_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if os.name == "nt":
        return _windows_process_is_running(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _windows_process_is_running(pid: int) -> bool:
    # Avoid os.kill(pid, 0) on Windows: unlike POSIX, it can terminate a process.
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    process_query_limited_information = 0x1000
    still_active = 259
    error_access_denied = 5

    handle = kernel32.OpenProcess(process_query_limited_information, False, int(pid))
    if not handle:
        return ctypes.get_last_error() == error_access_denied
    try:
        exit_code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return True
        return exit_code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def _remove_lock_file(lock_path: Path) -> None:
    deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
    while True:
        try:
            lock_path.unlink(missing_ok=True)
            return
        except PermissionError as exc:
            if time.monotonic() > deadline:
                logger.warning("Could not remove sidecar lock %s (%s)", lock_path, exc)
                return
            time.sleep(_LOCK_POLL_SECONDS)


@contextlib.contextmanager
def _sidecar_lock(path: Path) -> Iterator[None]:
    """Advisory cross-process lock for one sidecar's read-modify-write section.

    Recovery is based on owner liveness, not file age. A slow but healthy writer
    must never have its lock stolen because a FUSE/SMB write exceeded the
    acquisition timeout.
    """
    lock_path = path.with_suffix(path.suffix + _LOCK_SUFFIX)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
    fd: int | None = None
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, _lock_owner_payload())
                os.fsync(fd)
                os.close(fd)
                fd = None
            except BaseException:
                if fd is not None:
                    os.close(fd)
                fd = None
                lock_path.unlink(missing_ok=True)
                raise
            break
        except FileExistsError:
            # TOCTOU: the holder can unlink lock_path between the O_EXCL failure
            # above and this metadata read (normal completion, not just a crash).
            owner = _read_lock_owner(lock_path)
            if owner is not None and not _process_is_running(int(owner["pid"])):
                logger.warning(
                    "Reclaiming stale sidecar lock %s (pid=%s)", lock_path, owner["pid"]
                )
                try:
                    lock_path.unlink(missing_ok=True)
                except OSError as exc:
                    # Windows: another process may still hold the handle open
                    # (holder finishing now, or a concurrent reclaimer won the
                    # race). missing_ok=True only swallows FileNotFoundError, not
                    # PermissionError/WinError 32 — back off and re-evaluate next
                    # loop instead of crashing the acquiring caller.
                    logger.warning("Could not reclaim stale lock %s (%s); retrying", lock_path, exc)
                    time.sleep(_LOCK_POLL_SECONDS)
                continue
            if time.monotonic() > deadline:
                owner_note = f"pid={owner['pid']}" if owner is not None else "unknown owner"
                raise TimeoutError(
                    f"Could not acquire sidecar lock {lock_path} within "
                    f"{_LOCK_TIMEOUT_SECONDS}s; another writer may be stuck ({owner_note})."
                )
            time.sleep(_LOCK_POLL_SECONDS)
    try:
        yield
    finally:
        # Close before unlink — required on Windows, where unlinking a file while
        # its own handle is open raises PermissionError (a plain os.open has no
        # FILE_SHARE_DELETE). Do not reorder these two lines.
        if fd is not None:
            os.close(fd)
        _remove_lock_file(lock_path)

_PROTECTED_QUANT_FIELDS: frozenset[str] = frozenset(
    {
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
    }
)

_VALID_EVENT_TYPES: frozenset[str] = frozenset(
    {
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
    }
)

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
        # Strip a trailing null-byte pad before parsing (BUG-sess-20260524-nba1):
        # a fixed-size write buffer not truncated to content length on flush left
        # valid JSON followed by a run of \x00, which json.load rejects as "Extra
        # data". Valid JSON never ends with \x00, so stripping is safe.
        raw = path.read_bytes().rstrip(b"\x00")
        return cls.model_validate(json.loads(raw))

    def to_report_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def bootstrap_payload(
    session_id: str,
    *,
    model_version: str,
    purpose: str,
    bankroll: float,
    bankroll_confirmed: bool = False,
    effective_db_path: str | None = None,
    runtime_db_status: str | None = None,
) -> dict[str, Any]:
    """Return a minimal dict suitable for SessionSidecar.model_validate.

    ``effective_db_path`` / ``runtime_db_status`` record which trace DB the
    session actually resolved to. Pass ``TraceStore().db_path`` and
    ``.db_path_source`` (see omega-session-bootstrap SKILL.md); they default to
    None for callers that don't resolve a store, preserving prior behavior.
    """
    now = (
        datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    return {
        "session_id": session_id,
        "opened_at": now,
        "closed_at": None,
        "model_version": model_version,
        "purpose": purpose,
        "effective_db_path": effective_db_path,
        "runtime_db_status": runtime_db_status,
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


def _mirror_missing_events(path: Path, all_events: list[dict[str, Any]]) -> None:
    """Append only events not already in the JSONL mirror, keeping it a true
    superset of the JSON summary (fixes SIDECAR_LOGGING_AUDIT_2026-06-07 F3).

    Dedup is by a canonical (sort_keys) JSON signature of the *full* event, not a
    coarse (ts, step, event_type) key: ``ts`` is second-precision (see the
    AuditEvent construction sites, e.g. append_null_data_audit's
    ``.replace(microsecond=0)``), so a fast batch legitimately logging multiple
    distinct events of the same step/type within one wall-clock second would
    collide on the coarse key and silently drop real events — reintroducing the
    exact JSON/JSONL divergence this helper exists to prevent. (Observed in
    sess-20260701-ops1: duplicate injects fired 12:57:33 x3, 12:57:54 x2,
    12:58:35 x3 — same-second multiplicity was the normal case, not an edge.)
    """
    jsonl = _events_jsonl_path(path)
    existing_sigs: set[str] = set()
    if jsonl.exists():
        try:
            for line in jsonl.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                existing_sigs.add(json.dumps(rec, sort_keys=True))
        except OSError:
            pass  # best-effort read; fall through and mirror everything
    missing = [e for e in all_events if json.dumps(e, sort_keys=True) not in existing_sigs]
    _mirror_events_jsonl(path, missing)


def _write_sidecar_unlocked(path: Path, sidecar: SessionSidecar) -> None:
    from omega.trace._atomic import atomic_write_text

    atomic_write_text(path, json.dumps(sidecar.model_dump(mode="json"), indent=2))
    _mirror_missing_events(path, [e.model_dump(mode="json") for e in sidecar.audit_events])


def write_sidecar(path: Path, sidecar: SessionSidecar) -> None:
    """Atomically write a full sidecar (temp + fsync + replace), keeping the JSONL
    mirror a true superset of every persisted audit_events entry.

    This is the single mirror-writing path: callers no longer mirror separately
    (which double-counted on re-call — the F3 duplicate-event bug). The write and
    the mirror update happen under the sidecar lock so a concurrent append cannot
    interleave.
    """
    with _sidecar_lock(path):
        _write_sidecar_unlocked(path, sidecar)


def create_sidecar(
    path: Path,
    payload: dict[str, Any],
    *,
    allow_reopen: bool = False,
) -> SessionSidecar:
    """Validate ``payload`` and atomically create the sidecar + its JSONL mirror.

    Fails closed on collision (fixes the sess-20260701-ops1 session-ID reuse that
    let three independent conversations interleave writes into one sidecar): if
    ``path`` already exists this raises ``FileExistsError`` unless
    ``allow_reopen=True`` AND the existing sidecar is still open
    (``closed_at is None``) — i.e. the caller is legitimately continuing the same
    session (the ``--ingest --render-report`` re-invocation flow), not stomping a
    different conversation's sidecar. A closed session can never be reopened here;
    that would silently corrupt a finalized audit trail.

    Replaces non-atomic ``path.write_text(json.dumps(...))`` session creation,
    which was a truncation source.
    """
    with _sidecar_lock(path):
        if path.exists():
            if not allow_reopen:
                raise FileExistsError(
                    f"Session sidecar already exists: {path}. If you are the same "
                    f"session continuing (e.g. re-invoking to ingest/render after the "
                    f"engine phase), pass allow_reopen=True. If you are a different "
                    f"conversation, choose a distinct session_id - do not reuse one; "
                    f"see omega-session-bootstrap SKILL.md."
                )
            existing = load_sidecar_safe(path)
            if existing is None:
                raise FileExistsError(
                    f"Session sidecar {path} exists but is unreadable/corrupt; "
                    f"quarantine it first (validate_session_sidecars.py --quarantine) "
                    f"before opening a new session at this path."
                )
            if existing.closed_at is not None:
                raise FileExistsError(
                    f"Session sidecar {path} is already closed "
                    f"(closed_at={existing.closed_at}); a closed session cannot be "
                    f"reopened. Choose a new session_id."
                )
            incoming_session_id = payload.get("session_id")
            if incoming_session_id != existing.session_id:
                raise FileExistsError(
                    f"Session sidecar {path} is already open for "
                    f"session_id={existing.session_id}; refusing to reopen with "
                    f"session_id={incoming_session_id!r}. Choose a new session_id."
                )
            # Legitimate reopen of a still-open session: return existing state
            # unchanged so append_audit_events() is the caller's next step (do NOT
            # re-validate/rewrite the payload, which would reset audit_events).
            return existing

        sidecar = SessionSidecar.model_validate(payload)
        _write_sidecar_unlocked(path, sidecar)
        return sidecar


def append_audit_events(path: Path, events: list[dict[str, Any]]) -> None:
    """Atomically append audit events to a session sidecar (+ JSONL mirror).

    The read-modify-write is serialized under the sidecar lock so concurrent
    appends can no longer race (F2): a second writer blocks until the first's
    full R-M-W-mirror commits, then reads the fully-committed state.
    """
    from omega.trace._atomic import atomic_write_text

    validated = [AuditEvent.model_validate(e) for e in events]
    for event in validated:
        _check_protected_fields(event)

    with _sidecar_lock(path):
        sidecar = SessionSidecar.from_path(path)
        sidecar.audit_events.extend(validated)
        atomic_write_text(path, json.dumps(sidecar.model_dump(mode="json"), indent=2))
        # This path already holds the exact delta, so mirror it directly — no
        # dedup scan needed here (that's only for write_sidecar, which sees the
        # full list and must reconstruct the delta).
        _mirror_events_jsonl(path, [e.model_dump(mode="json") for e in validated])


def close_sidecar(
    path: Path,
    *,
    exec_stats: dict[str, Any],
    pipeline_status: dict[str, Any] | None = None,
    next_required_action: str | None = None,
    agent_notes: str | None = None,
) -> SessionSidecar:
    """Atomically finalize an open session sidecar (locked read-modify-write).

    Sets ``closed_at`` plus the closeout summary fields. Fails closed: a missing
    sidecar raises FileNotFoundError via ``from_path``, an already-closed one
    raises ValueError (``create_sidecar`` refuses to reopen closed sessions, so
    a double close is always a caller bug), and engine-owned quant fields in
    ``exec_stats``/``pipeline_status`` raise ProtectedValueError — closeout
    stats are execution accounting, never model numbers.
    """
    for label, mapping in (("exec_stats", exec_stats), ("pipeline_status", pipeline_status)):
        if mapping:
            found = _find_protected_key(mapping)
            if found:
                raise ProtectedValueError(
                    f"close_sidecar {label} contains protected engine field {found!r}. "
                    "Engine-owned quant values must stay in var/omega_traces.db."
                )
    with _sidecar_lock(path):
        sidecar = SessionSidecar.from_path(path)
        if sidecar.closed_at is not None:
            raise ValueError(
                f"Session sidecar {path} is already closed (closed_at={sidecar.closed_at})."
            )
        sidecar.closed_at = (
            datetime.datetime.now(datetime.timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )
        sidecar.exec_stats = dict(exec_stats)
        if pipeline_status is not None:
            sidecar.pipeline_status = dict(pipeline_status)
        if next_required_action is not None:
            sidecar.next_required_action = next_required_action
        if agent_notes is not None:
            sidecar.agent_notes = agent_notes
        _write_sidecar_unlocked(path, sidecar)
        return sidecar


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
_VALID_QA_SCOPES: frozenset[str] = frozenset(
    {
        "trace_id",
        "timestamp_window",
        "pre_trace_fatal",
        "session_fallback",
        "unrelated_session_failure",
        "no_sidecar",
    }
)

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

    Returns a dictionary that validates cleanly against ``SessionSidecar``
    (``extra="forbid"``), so the caller can pass it straight to
    ``SessionSidecar.model_validate`` without stripping keys. Diagnostic values
    that used to be embedded here — event count and source path — are redundant
    (``len(payload["audit_events"])`` and ``jsonl_path`` respectively) and were
    removed because they broke that round-trip.

    Note: ``model_version``/``purpose``/``bankroll`` are reconstruction
    placeholders, not verified session state — see ``agent_notes``.
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
        opened_at = (
            datetime.datetime.now(datetime.timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )

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
    now = (
        datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
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
