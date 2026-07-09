"""Sweep `var/inbox/sessions/` for stale open sessions and old JSONL mirrors.

Two independent, opt-in-to-mutate sweeps over the same directory:

1. **Stale-open-session close** (F16, SIDECAR_LOGGING_AUDIT_2026-06-07): a
   session sidecar with ``closed_at: null`` and no JSON/mirror activity for
   ``--stale-hours`` is presumed abandoned (crashed session, forgotten
   closeout) rather than genuinely in-progress. Reported by default; pass
   ``--apply`` to actually call ``close_sidecar`` on each one, stamping a
   clearly-labeled ``agent_notes`` explanation so a human reviewing the
   sidecar later knows it was not a normal closeout.

2. **Old mirror prune** (F8): the ``<sid>.events.jsonl`` recovery mirror has
   no cap/rotation and accumulates forever. For sessions that are already
   *closed* and older than ``--prune-mirrors-days``, the mirror's only
   remaining value (recovering a corrupt JSON summary) is moot -- the summary
   is durable and, if ingested, the DB is authoritative. Reported by default;
   pass ``--apply --prune-mirrors`` to actually delete the mirror file (the
   ``.json`` sidecar itself is never touched by this sweep).

Both sweeps default to dry-run reporting; nothing is mutated or deleted
without an explicit ``--apply`` (matching ``settle_bets.py`` / other mutating
ops CLIs in this repo).
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from omega.trace.session_sidecar import close_sidecar  # noqa: E402

logger = logging.getLogger("sweep_stale_sessions")

_DEFAULT_SESSIONS_INBOX = _REPO_ROOT / "var" / "inbox" / "sessions"
_DEFAULT_STALE_HOURS = 24.0
_DEFAULT_PRUNE_MIRRORS_DAYS = 90


def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _parse_ts(value: str | None) -> datetime.datetime | None:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=datetime.timezone.utc)


def _last_touch(json_path: Path, mirror_path: Path) -> datetime.datetime:
    """Most recent mtime across the sidecar JSON and its mirror (whichever
    exists), as an activity signal independent of any timestamp *inside*
    the file -- a crashed writer could leave stale in-file timestamps."""
    candidates = [datetime.datetime.fromtimestamp(json_path.stat().st_mtime, tz=datetime.timezone.utc)]
    if mirror_path.exists():
        candidates.append(
            datetime.datetime.fromtimestamp(mirror_path.stat().st_mtime, tz=datetime.timezone.utc)
        )
    return max(candidates)


def find_stale_open_sessions(
    sessions_inbox: Path,
    *,
    stale_hours: float,
) -> list[tuple[Path, dict, float]]:
    """Return (path, sidecar_dict, age_hours) for open sessions past the threshold."""
    now = _utc_now()
    stale: list[tuple[Path, dict, float]] = []
    for json_path in sorted(sessions_inbox.glob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("closed_at") is not None:
            continue
        mirror_path = json_path.with_suffix(".events.jsonl")
        age_hours = (now - _last_touch(json_path, mirror_path)).total_seconds() / 3600.0
        if age_hours >= stale_hours:
            stale.append((json_path, data, round(age_hours, 1)))
    return stale


def find_prunable_mirrors(
    sessions_inbox: Path,
    *,
    prune_mirrors_days: int,
) -> list[tuple[Path, dict, float]]:
    """Return (mirror_path, sidecar_dict, age_days) for closed sessions'
    mirrors past the retention window. Only sessions with a parseable
    closed_at are candidates -- an unparseable/missing closed_at is treated
    conservatively as "not eligible" rather than guessed at."""
    now = _utc_now()
    prunable: list[tuple[Path, dict, float]] = []
    for json_path in sorted(sessions_inbox.glob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        closed_dt = _parse_ts(data.get("closed_at"))
        if closed_dt is None:
            continue
        mirror_path = json_path.with_suffix(".events.jsonl")
        if not mirror_path.exists():
            continue
        age_days = (now - closed_dt).total_seconds() / 86400.0
        if age_days >= prune_mirrors_days:
            prunable.append((mirror_path, data, round(age_days, 1)))
    return prunable


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sessions-inbox",
        type=Path,
        default=_DEFAULT_SESSIONS_INBOX,
        help="Directory containing <session_id>.json sidecars (default: var/inbox/sessions)",
    )
    parser.add_argument(
        "--stale-hours",
        type=float,
        default=_DEFAULT_STALE_HOURS,
        help=f"Age (hours, by last file mtime) past which an open session is closed as abandoned (default: {_DEFAULT_STALE_HOURS})",
    )
    parser.add_argument(
        "--prune-mirrors",
        action="store_true",
        help="Also sweep for old JSONL mirrors on closed sessions (off by default; stale-open close always runs)",
    )
    parser.add_argument(
        "--prune-mirrors-days",
        type=int,
        default=_DEFAULT_PRUNE_MIRRORS_DAYS,
        help=f"Age (days since closed_at) past which a closed session's mirror is prunable (default: {_DEFAULT_PRUNE_MIRRORS_DAYS})",
    )
    parser.add_argument(
        "--apply", action="store_true", help="Actually close/delete; default is dry-run report only"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.sessions_inbox.is_dir():
        logger.error("Sessions inbox does not exist: %s", args.sessions_inbox)
        return 1

    stale = find_stale_open_sessions(args.sessions_inbox, stale_hours=args.stale_hours)
    logger.info(
        "Stale open sessions (>%.0fh no activity): %d found", args.stale_hours, len(stale)
    )
    for json_path, data, age_hours in stale:
        logger.info("  %s  opened_at=%s  age=%.1fh", json_path.name, data.get("opened_at"), age_hours)
        if args.apply:
            close_sidecar(
                json_path,
                exec_stats={},
                next_required_action="Operator review: auto-closed by sweep_stale_sessions as abandoned.",
                agent_notes=(
                    f"AUTO-CLOSED by omega-sweep-stale-sessions: no sidecar/mirror "
                    f"activity for {age_hours:.1f}h (threshold {args.stale_hours:.0f}h). "
                    "This was not a normal session closeout -- review exec_stats/"
                    "audit_events before treating this session's output as complete."
                ),
            )
            logger.info("  -> closed")

    if args.prune_mirrors:
        prunable = find_prunable_mirrors(
            args.sessions_inbox, prune_mirrors_days=args.prune_mirrors_days
        )
        logger.info(
            "Prunable mirrors (closed >%dd ago): %d found", args.prune_mirrors_days, len(prunable)
        )
        for mirror_path, data, age_days in prunable:
            size = mirror_path.stat().st_size
            logger.info(
                "  %s  closed_at=%s  age=%.0fd  size=%dB",
                mirror_path.name,
                data.get("closed_at"),
                age_days,
                size,
            )
            if args.apply:
                mirror_path.unlink()
                logger.info("  -> deleted")

    if not args.apply and (stale or (args.prune_mirrors and args.sessions_inbox.is_dir())):
        logger.info("Dry-run only. Re-run with --apply to make changes.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
