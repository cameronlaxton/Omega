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


def validate_directory(path: Path, *, quarantine: bool = False) -> tuple[int, int]:
    if not path.exists():
        raise FileNotFoundError(f"Session inbox does not exist: {path}")

    valid = 0
    invalid = 0
    for sidecar in sorted(path.glob("*.json")):
        try:
            SessionSidecar.from_path(sidecar)
        except (OSError, ValueError, ValidationError) as exc:
            invalid += 1
            logger.error("INVALID %s: %s", sidecar.name, exc)
            if quarantine:
                # The ONLY place that moves files: idempotent, leaves the JSONL
                # mirror in place for recovery.
                quarantine_sidecar(sidecar, f"{type(exc).__name__}: {exc}")
        else:
            valid += 1
            logger.info("OK %s", sidecar.name)
    return valid, invalid


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
        valid, invalid = validate_directory(args.sessions_inbox, quarantine=args.quarantine)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Validated %d sidecar(s); %d invalid.", valid, invalid)
    return 1 if invalid else 0


if __name__ == "__main__":
    sys.exit(main())
