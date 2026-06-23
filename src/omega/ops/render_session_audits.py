"""
omega.ops.render_session_audits â€” render session audit markdown.

Reads `var/omega_traces.db` + `var/inbox/sessions/<session_id>.json` sidecars and
writes `var/reports/run_audits/<session_id>.audit.md` atomically.

Two modes:
    --all-open                  render every sidecar in var/inbox/sessions/
    --session-ids ID [ID ...]   render only the listed session_ids

Numeric values shown in the rendered audit are sourced from the trace
ledger, never from sidecar prose. Legacy `RUN_TRACE.jsonl` / `RUN_AUDIT.md`
are not read by this script.

Exit codes:
    0 â€” all requested audits rendered
    1 â€” at least one render failed
    2 â€” fatal validation error (no inputs, missing dirs)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from omega.paths import run_audits_dir, session_inbox_dir  # noqa: E402
from omega.trace.audit_renderer import render_session_audit  # noqa: E402

logger = logging.getLogger("render_session_audits")


def _collect_session_ids(sidecar_dir: Path) -> list[str]:
    return sorted(p.stem for p in sidecar_dir.glob("*.json") if p.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description="Render session audit markdown.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--all-open",
        action="store_true",
        help="Render every sidecar in var/inbox/sessions/",
    )
    group.add_argument(
        "--session-ids",
        nargs="+",
        help="Explicit list of session_ids to render",
    )
    parser.add_argument(
        "--sidecar-dir",
        type=Path,
        default=session_inbox_dir(),
        help="Directory containing <session_id>.json sidecars (default: var/inbox/sessions)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="SQLite path (default: var/omega_traces.db)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=run_audits_dir(),
        help="Output directory for rendered audit markdown (default: var/reports/run_audits)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    sidecar_dir: Path = args.sidecar_dir
    if not sidecar_dir.exists():
        logger.error("Sidecar directory does not exist: %s", sidecar_dir)
        return 2

    if args.all_open:
        session_ids = _collect_session_ids(sidecar_dir)
        if not session_ids:
            logger.info("No sidecars in %s; nothing to render.", sidecar_dir)
            return 0
    else:
        session_ids = list(args.session_ids)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for session_id in session_ids:
        try:
            out = render_session_audit(
                session_id,
                db_path=args.db,
                sidecar_dir=sidecar_dir,
                out_dir=args.out_dir,
            )
        except Exception as exc:  # noqa: BLE001
            failures += 1
            logger.error("RENDER FAILED %s: %s", session_id, exc)
            continue
        logger.info("OK %s -> %s", session_id, out)

    if failures:
        logger.error("Rendered with %d failure(s).", failures)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
