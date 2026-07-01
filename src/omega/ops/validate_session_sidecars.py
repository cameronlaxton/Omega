"""Validate `var/inbox/sessions/*.json` files against the session sidecar contract."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pydantic import ValidationError

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from omega.trace.session_sidecar import SessionSidecar, quarantine_sidecar  # noqa: E402

logger = logging.getLogger("validate_session_sidecars")

# Canonical session-sidecar inbox. The sidecars live under ``var/`` (see
# docs/phase6/ARTIFACT_AUTHORITY.md); a default missing the ``var/`` segment
# silently validates an empty/stale directory and lets bad sidecars through.
_DEFAULT_SESSIONS_INBOX = _REPO_ROOT / "var" / "inbox" / "sessions"


def _consistency_issues(sidecar: SessionSidecar, sidecar_path: Path) -> list[str]:
    """Non-fatal mirror/close invariants — WARN, never quarantine.

    These are the invariants that would have caught the SIDECAR_LOGGING_AUDIT
    F2/F3 drift (18 sessions with audit_events counts that don't match their
    ``.events.jsonl`` line counts) at validation time. They are deliberately kept
    out of the ``--quarantine`` path: a count mismatch means the mirror invariant
    is broken, not that the JSON is malformed, and moving an otherwise-valid,
    readable sidecar would be a worse regression than the drift it flags.
    """
    issues: list[str] = []
    jsonl_path = sidecar_path.with_suffix(".events.jsonl")
    if jsonl_path.exists():
        try:
            json_count = len(sidecar.audit_events)
            jsonl_count = sum(
                1 for ln in jsonl_path.read_text(encoding="utf-8").splitlines() if ln.strip()
            )
            if json_count != jsonl_count:
                issues.append(
                    f"audit_events count ({json_count}) != mirror line count ({jsonl_count})"
                )
        except OSError as exc:
            issues.append(f"could not read mirror {jsonl_path.name}: {exc}")
    if sidecar.closed_at is not None and not sidecar.exec_stats:
        issues.append("closed session has empty exec_stats")
    return issues


def validate_directory(path: Path, *, quarantine: bool = False) -> tuple[int, int, int]:
    """Returns ``(valid, invalid, warned)``.

    ``invalid`` = schema/parse failures (drive the exit code, and quarantine when
    requested). ``warned`` = structurally valid but failing a soft consistency
    invariant (never fail the run, never move the file).
    """
    if not path.exists():
        raise FileNotFoundError(f"Session inbox does not exist: {path}")

    valid = 0
    invalid = 0
    warned = 0
    for sidecar in sorted(path.glob("*.json")):
        try:
            parsed = SessionSidecar.from_path(sidecar)
        except (OSError, ValueError, ValidationError) as exc:
            invalid += 1
            logger.error("INVALID %s: %s", sidecar.name, exc)
            if quarantine:
                # The ONLY place that moves files: idempotent, leaves the JSONL
                # mirror in place for recovery.
                quarantine_sidecar(sidecar, f"{type(exc).__name__}: {exc}")
            continue
        valid += 1
        issues = _consistency_issues(parsed, sidecar)
        if issues:
            warned += 1
            for issue in issues:
                logger.warning("WARN %s: %s", sidecar.name, issue)
        else:
            logger.info("OK %s", sidecar.name)
    return valid, invalid, warned


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Omega session sidecar JSON files")
    parser.add_argument(
        "--sessions-inbox",
        type=Path,
        default=_DEFAULT_SESSIONS_INBOX,
        help="Directory containing session sidecar JSON files (default: var/inbox/sessions)",
    )
    parser.add_argument(
        "--quarantine",
        action="store_true",
        help="Move invalid sidecars to <inbox>/invalid/ with a .reason.txt (idempotent)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        valid, invalid, warned = validate_directory(
            args.sessions_inbox, quarantine=args.quarantine
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Validated %d sidecar(s); %d invalid, %d warned.", valid, invalid, warned)
    # Exit code stays gated on schema `invalid` only — soft consistency warnings
    # surface the signal without failing existing CI/pre-push callers.
    return 1 if invalid else 0


if __name__ == "__main__":
    sys.exit(main())
