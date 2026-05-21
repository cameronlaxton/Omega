"""Validate `inbox/sessions/*.json` files against the session sidecar contract."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pydantic import ValidationError

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from omega.trace.session_sidecar import SessionSidecar  # noqa: E402

logger = logging.getLogger("validate_session_sidecars")


def validate_directory(path: Path) -> tuple[int, int]:
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
        else:
            valid += 1
            logger.info("OK %s", sidecar.name)
    return valid, invalid


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Omega session sidecar JSON files")
    parser.add_argument(
        "--sessions-inbox",
        type=Path,
        default=_REPO_ROOT / "inbox" / "sessions",
        help="Directory containing session sidecar JSON files",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        valid, invalid = validate_directory(args.sessions_inbox)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Validated %d sidecar(s); %d invalid.", valid, invalid)
    return 1 if invalid else 0


if __name__ == "__main__":
    sys.exit(main())
