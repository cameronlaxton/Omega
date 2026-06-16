"""Fail-closed runtime DB guard for scheduled/action-plan workflows."""

from __future__ import annotations

import logging
import os
from typing import Any

from omega.trace.store import db_status

logger = logging.getLogger("omega.ops.runtime_db_guard")

_SAFE_SOURCES = {"default", "requested", "env_override"}


class UnsafeRuntimeDbError(RuntimeError):
    """Raised when a write-capable workflow would use an unsafe TraceStore DB."""


def _empty_history_mode() -> bool:
    return os.environ.get("OMEGA_ALLOW_EMPTY_DB") == "1"


def _failure_reasons(status: dict[str, Any], *, allow_empty: bool) -> list[str]:
    reasons: list[str] = []
    source = status.get("source")
    effective_exists = bool(status.get("effective_exists"))
    effective_integrity_ok = status.get("effective_integrity_ok")
    effective_trace_count = status.get("effective_trace_count")

    if source == "auto_redirect_network_fs":
        reasons.append(
            "effective DB is auto-redirected from a network/FUSE mount; run from the "
            "durable local workspace or set OMEGA_TRACE_DB to a durable SQLite path"
        )
    elif source not in _SAFE_SOURCES:
        reasons.append(f"effective DB source {source!r} is not in {sorted(_SAFE_SOURCES)}")

    if status.get("divergence") is not None:
        reasons.append("runtime/source DB divergence is present; no scheduled write is safe")

    if not effective_exists and not allow_empty:
        reasons.append("effective DB is missing")

    if effective_exists and effective_integrity_ok is not True:
        reasons.append(f"effective DB integrity is not OK: {effective_integrity_ok!r}")

    if effective_trace_count is None and not allow_empty:
        reasons.append("effective trace count is unavailable")

    if effective_trace_count == 0 and not allow_empty:
        reasons.append("effective DB has zero traces")

    return reasons


def _status_summary(status: dict[str, Any]) -> str:
    return (
        f"path={status.get('effective_path')} source={status.get('source')} "
        f"exists={status.get('effective_exists')} "
        f"integrity_ok={status.get('effective_integrity_ok')} "
        f"trace_count={status.get('effective_trace_count')} "
        f"recommended_action={status.get('recommended_action')}"
    )


def assert_safe_runtime_db(requested: str | None = None, *, dry_run: bool = False) -> dict:
    """Assert that scheduled/action-plan writes target a durable, populated DB.

    Dry runs return the same status payload but downgrade unsafe DB identity to a
    warning so operators can validate action-plan shape before fixing runtime
    placement. Non-dry-run execution raises before the first write-capable action.
    """
    status = db_status(requested)
    allow_empty = _empty_history_mode()
    effective_exists = bool(status.get("effective_exists"))
    effective_trace_count = status.get("effective_trace_count")
    empty_history_pass = allow_empty and (
        not effective_exists or effective_trace_count is None or effective_trace_count == 0
    )
    reasons = _failure_reasons(status, allow_empty=allow_empty)

    if empty_history_pass:
        logger.warning(
            "EMPTY_HISTORY_MODE=true: runtime DB guard is allowing empty history "
            "for effective DB %s (source=%s).",
            status.get("effective_path"),
            status.get("source"),
        )

    if reasons:
        msg = (
            "Unsafe Omega runtime DB for scheduled/action-plan execution: "
            + "; ".join(reasons)
            + ". "
            + _status_summary(status)
            + ". Run tools/windows/cowork_bootstrap.ps1 and launch from "
            "$env:OMEGA_LOCAL_WORKSPACE, or set OMEGA_TRACE_DB to an explicit "
            "durable DB path."
        )
        if dry_run:
            logger.warning("DRY-RUN runtime DB warning: %s", msg)
            return status
        raise UnsafeRuntimeDbError(msg)

    logger.info("Runtime DB guard passed: %s", _status_summary(status))
    return status
