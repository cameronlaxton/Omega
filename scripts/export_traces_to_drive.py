"""
scripts/export_traces_to_drive.py - publish trace artifacts to a Google Drive
folder for the Trace Explorer live artifact to read.

Hard wall (CLAUDE.md): this script is a READ-ONLY view publisher. It does not
fabricate edge%, EV%, Kelly, calibrated probabilities, tier, or trace_ids -
every value comes from omega_traces.db, which is populated by the engine.

Output:
    omega-exports/
        traces_YYYYMMDDTHHMMSSZ.json   one snapshot per export run
        latest.json                     copy of the most recent snapshot

JSON shape (export_schema_version = 1):
    {
        "export_schema_version": 1,
        "exported_at": "ISO-8601 UTC",
        "trace_schema_version": int,        # from omega_traces.db
        "source_db": "<absolute path>",
        "trace_count": int,
        "traces": [
            {
                "trace_id": str,
                "run_id": str,
                "timestamp": str,
                "league": str | null,
                "matchup": str | null,
                "execution_mode": str | null,
                "aggregate_quality": float,
                "session_id": str | null,
                "downgrades": [...],
                "recommendation_summary": {
                    "total": int,
                    "by_tier": {"A": n, "B": n, "C": n, "Pass": n},
                    "max_edge_pct": float | null
                },
                "outcome": {home_score, away_score, result} | null,
                "bet_records": [...],
                "closing_lines": [...],
                "clv": [{ market, selection_descriptor, clv_cents,
                          beat_close, line_value } ...],
                "full_trace": {...}     # original JSON blob (verbatim)
            }
        ]
    }

Upload to Drive:
    - If --upload is set AND google-api-python-client + valid OAuth creds exist
      at ~/.omega/drive_token.json, the script creates the omega-exports folder
      (if missing) and uploads latest.json.
    - Otherwise the script writes local files only; sync to Drive by asking
      Cowork: "upload omega-exports/latest.json to the omega-exports folder
      in Drive".

CLI:
    python scripts/export_traces_to_drive.py
    python scripts/export_traces_to_drive.py --league NBA --limit 500
    python scripts/export_traces_to_drive.py --db ./omega_traces.db --upload
    python scripts/export_traces_to_drive.py --since 2026-05-01 --no-full-trace

Exit codes:
    0 - export written (and uploaded if requested)
    1 - fatal error (bad args, DB missing, upload failed when requested)
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow running as a script from repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from omega.trace.clv import compute_clv  # noqa: E402
from omega.trace.store import TraceStore  # noqa: E402

logger = logging.getLogger("export_traces_to_drive")

EXPORT_SCHEMA_VERSION = 1
DRIVE_FOLDER_NAME = "omega-exports"
LATEST_FILENAME = "latest.json"


# ---------------------------------------------------------------------------
# Engine summary (no math, just counts and max - values lifted verbatim)
# ---------------------------------------------------------------------------

_TIER_KEYS = {"A", "B", "C", "Pass"}


def _collect_picks(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize engine output to a list of pick dicts.

    Recognized shapes:
        - trace["recommendations"] = [pick, pick, ...]
        - trace["recommendations"] = {"items": [...]}
        - trace["result"] = pick (single, for sandbox_prop / native_sim)
        - trace["result"] = {"recommendations": [...]}
        - trace["bet_card"] / trace["edge_detail"] = pick(s)
    """
    out: List[Dict[str, Any]] = []
    candidates: List[Any] = [
        trace.get("recommendations"),
        trace.get("result"),
        trace.get("bet_card"),
        trace.get("edge_detail"),
    ]
    for c in candidates:
        if not c:
            continue
        if isinstance(c, list):
            out.extend(x for x in c if isinstance(x, dict))
        elif isinstance(c, dict):
            inner = c.get("recommendations") or c.get("items") or c.get("picks")
            if isinstance(inner, list):
                out.extend(x for x in inner if isinstance(x, dict))
            else:
                # Treat the dict itself as a single pick if it looks like one
                if any(k in c for k in ("recommendation", "confidence_tier",
                                         "tier", "edge_pct", "edge_over",
                                         "edge_under", "edge")):
                    out.append(c)
    return out


def _extract_edge(pick: Dict[str, Any]) -> Optional[float]:
    """Pull the relevant edge value from a pick, respecting the recommendation
    side. No re-derivation; returns None if absent.
    """
    rec = str(pick.get("recommendation") or "").lower()
    # Prop shape: edge_over / edge_under, signed against `recommendation`
    if "edge_over" in pick or "edge_under" in pick:
        if rec == "over" and pick.get("edge_over") is not None:
            return _safe_float(pick["edge_over"])
        if rec == "under" and pick.get("edge_under") is not None:
            return _safe_float(pick["edge_under"])
        return max(
            (_safe_float(pick.get(k)) for k in ("edge_over", "edge_under")
             if pick.get(k) is not None),
            default=None,
            key=lambda v: -1e9 if v is None else v,
        )
    return _safe_float(pick.get("edge_pct") or pick.get("edge"))


def _safe_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _summarize_recommendations(trace: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate engine picks - no math, just counts and the max edge already
    computed by the engine.
    """
    picks = _collect_picks(trace)
    by_tier = {k: 0 for k in _TIER_KEYS}
    edges: List[float] = []
    sides: List[Dict[str, Any]] = []

    for pick in picks:
        tier = str(pick.get("tier") or pick.get("confidence_tier") or "").strip()
        if tier in by_tier:
            by_tier[tier] += 1
        edge = _extract_edge(pick)
        if edge is not None:
            edges.append(edge)
        sides.append({
            "recommendation": pick.get("recommendation"),
            "tier": tier or None,
            "edge_pct": edge,
            "market": pick.get("market") or pick.get("prop_type"),
            "status": pick.get("status"),
        })

    return {
        "total": len(picks),
        "by_tier": by_tier,
        "max_edge_pct": max(edges) if edges else None,
        "picks": sides,
    }


# ---------------------------------------------------------------------------
# CLV resolution per trace (uses omega.trace.clv only - no new math)
# ---------------------------------------------------------------------------

def _compute_trace_clv(
    store: TraceStore, trace_id: str
) -> List[Dict[str, Any]]:
    """For each (bet_record, closing_line) pair on this trace, run compute_clv.
    Falls back to an empty list if either side is missing or math errors.
    """
    try:
        bets = store.get_bet_records(trace_id)
        closes = store.get_closing_lines(trace_id)
    except Exception as exc:
        logger.warning("CLV fetch failed for %s: %s", trace_id, exc)
        return []

    close_by_key = {
        (c["market"], c["selection_descriptor"]): c for c in closes
    }
    out: List[Dict[str, Any]] = []
    for bet in bets:
        key = (bet["market"], bet["selection_descriptor"])
        close = close_by_key.get(key)
        if not close:
            continue
        try:
            result = compute_clv(
                odds_taken=float(bet["odds_taken"]),
                closing_odds=float(close["closing_odds"]),
                line_taken=bet.get("line_taken"),
                closing_line=close.get("closing_line"),
                side=bet.get("selection"),
            )
        except Exception as exc:
            logger.warning(
                "compute_clv failed for trace=%s market=%s: %s",
                trace_id, bet["market"], exc,
            )
            continue
        out.append({
            "market": bet["market"],
            "selection_descriptor": bet["selection_descriptor"],
            "odds_taken": result.odds_taken,
            "closing_odds": result.closing_odds,
            "clv_cents": round(result.clv_cents, 3),
            "beat_close": result.beat_close,
            "line_value": result.line_value,
        })
    return out


# ---------------------------------------------------------------------------
# Build one export row
# ---------------------------------------------------------------------------

def _build_row(
    store: TraceStore,
    trace: Dict[str, Any],
    include_full_trace: bool,
) -> Dict[str, Any]:
    trace_id = trace.get("trace_id", "")
    outcome = trace.get("_outcome")

    row: Dict[str, Any] = {
        "trace_id": trace_id,
        "run_id": trace.get("run_id"),
        "timestamp": trace.get("timestamp"),
        "league": trace.get("league"),
        "matchup": trace.get("matchup"),
        "execution_mode": trace.get("execution_mode"),
        "aggregate_quality": trace.get("aggregate_quality"),
        "session_id": trace.get("session_id"),
        "downgrades": trace.get("downgrades") or [],
        "recommendation_summary": _summarize_recommendations(trace),
        "outcome": outcome,
        "bet_records": store.get_bet_records(trace_id) if trace_id else [],
        "closing_lines": store.get_closing_lines(trace_id) if trace_id else [],
        "clv": _compute_trace_clv(store, trace_id) if trace_id else [],
    }
    if include_full_trace:
        # Trace blob may contain Pydantic/datetime objects already coerced to
        # str by TraceStore on read - it's pure JSON-safe by this point.
        row["full_trace"] = trace
    return row


# ---------------------------------------------------------------------------
# Build the export document
# ---------------------------------------------------------------------------

def build_export(
    db_path: Path,
    league: Optional[str],
    since: Optional[str],
    until: Optional[str],
    limit: int,
    include_full_trace: bool,
) -> Dict[str, Any]:
    store = TraceStore(db_path=str(db_path))
    try:
        traces = store.query_traces(
            league=league,
            start=since,
            end=until,
            limit=limit,
        )
        rows = [_build_row(store, t, include_full_trace) for t in traces]
        return {
            "export_schema_version": EXPORT_SCHEMA_VERSION,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "trace_schema_version": store.schema_version(),
            "source_db": str(db_path.resolve()),
            "trace_count": len(rows),
            "filters": {
                "league": league,
                "since": since,
                "until": until,
                "limit": limit,
                "include_full_trace": include_full_trace,
            },
            "traces": rows,
        }
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Local write
# ---------------------------------------------------------------------------

def write_local(export: Dict[str, Any], out_dir: Path) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = (
        export["exported_at"]
        .replace("-", "")
        .replace(":", "")
        .split(".")[0]
        + "Z"
    )
    snapshot_path = out_dir / f"traces_{ts}.json"
    latest_path = out_dir / LATEST_FILENAME

    snapshot_path.write_text(
        json.dumps(export, indent=2, default=str), encoding="utf-8"
    )
    shutil.copyfile(snapshot_path, latest_path)
    return {"snapshot": snapshot_path, "latest": latest_path}


# ---------------------------------------------------------------------------
# Optional Drive upload (lazy import - script works without google libs)
# ---------------------------------------------------------------------------

def upload_to_drive(latest_path: Path, token_path: Path) -> str:
    """Upload latest.json to Drive omega-exports folder. Returns the file ID.

    Requires:
        pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
        ~/.omega/drive_token.json     OAuth token (drive.file scope)

    If google libs are missing, raises ImportError - caller decides whether to
    fall back to "ask Cowork to sync".
    """
    from google.oauth2.credentials import Credentials  # type: ignore
    from googleapiclient.discovery import build  # type: ignore
    from googleapiclient.http import MediaFileUpload  # type: ignore

    if not token_path.exists():
        raise FileNotFoundError(
            f"Drive token missing at {token_path}. Run the OAuth bootstrap "
            "or sync via Cowork instead."
        )

    creds = Credentials.from_authorized_user_file(
        str(token_path),
        scopes=["https://www.googleapis.com/auth/drive.file"],
    )
    svc = build("drive", "v3", credentials=creds)

    # Find or create omega-exports folder
    q = (
        f"name = '{DRIVE_FOLDER_NAME}' and "
        "mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    )
    resp = svc.files().list(q=q, fields="files(id,name)").execute()
    folders = resp.get("files", [])
    if folders:
        folder_id = folders[0]["id"]
    else:
        folder = svc.files().create(
            body={
                "name": DRIVE_FOLDER_NAME,
                "mimeType": "application/vnd.google-apps.folder",
            },
            fields="id",
        ).execute()
        folder_id = folder["id"]

    # Update latest.json in place if it exists, else create
    q = (
        f"name = '{LATEST_FILENAME}' and '{folder_id}' in parents and "
        "trashed = false"
    )
    resp = svc.files().list(q=q, fields="files(id,name)").execute()
    existing = resp.get("files", [])

    media = MediaFileUpload(
        str(latest_path), mimetype="application/json", resumable=False
    )
    if existing:
        file_id = existing[0]["id"]
        svc.files().update(fileId=file_id, media_body=media).execute()
    else:
        created = svc.files().create(
            body={"name": LATEST_FILENAME, "parents": [folder_id]},
            media_body=media,
            fields="id",
        ).execute()
        file_id = created["id"]
    return file_id


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        default=str(_REPO_ROOT / "omega_traces.db"),
        help="Path to omega_traces.db (default: repo root)",
    )
    parser.add_argument(
        "--out-dir",
        default=str(_REPO_ROOT / "omega-exports"),
        help="Local output directory (default: <repo>/omega-exports)",
    )
    parser.add_argument("--league", default=None, help="Filter by league")
    parser.add_argument("--since", default=None, help="ISO timestamp lower bound")
    parser.add_argument("--until", default=None, help="ISO timestamp upper bound")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument(
        "--no-full-trace",
        action="store_true",
        help="Omit the full_trace blob (smaller files)",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Also upload latest.json to Drive (requires OAuth token)",
    )
    parser.add_argument(
        "--token",
        default=str(Path.home() / ".omega" / "drive_token.json"),
        help="Path to OAuth token JSON",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    db_path = Path(args.db)
    if not db_path.exists():
        logger.error("DB not found: %s", db_path)
        return 1

    export = build_export(
        db_path=db_path,
        league=args.league,
        since=args.since,
        until=args.until,
        limit=args.limit,
        include_full_trace=not args.no_full_trace,
    )

    out_dir = Path(args.out_dir)
    paths = write_local(export, out_dir)
    logger.info(
        "Wrote %d traces -> %s (also -> %s)",
        export["trace_count"], paths["snapshot"], paths["latest"],
    )

    if args.upload:
        try:
            file_id = upload_to_drive(paths["latest"], Path(args.token))
            logger.info("Uploaded latest.json to Drive (file_id=%s)", file_id)
        except ImportError:
            logger.error(
                "google-api-python-client not installed. Install with: "
                "pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
            )
            return 1
        except Exception as exc:
            logger.error("Drive upload failed: %s", exc)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
elp="Path to omega_traces.db (default: repo root)",
    )
    parser.add_argument(
        "--out-dir",
        default=str(_REPO_ROOT / "omega-exports"),
        help="Local output directory (default: <repo>/omega-exports)",
    )
    parser.add_argument("--league", default=None, help="Filter by league")
    parser.add_argument(
        "--since", default=None, help="ISO timestamp lower bound"
    )
    parser.add_argument(
        "--until", default=None, help="ISO timestamp upper bound"
    )
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument(
        "--no-full-trace",
        action="store_true",
        help="Omit the full_trace blob (smaller files)",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Also upload latest.json to Drive (requires OAuth token)",
    )
    parser.add_argument(
        "--token",
        default=str(Path.home() / ".omega" / "drive_token.json"),
        help="Path to OAuth token JSON",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    db_path = Path(args.db)
    if not db_path.exists():
        logger.error("DB not found: %s", db_path)
        return 1

    export = build_export(
        db_path=db_path,
        league=args.league,
        since=args.since,
        until=args.until,
        limit=args.limit,
        include_full_trace=not args.no_full_trace,
    )

    out_dir = Path(args.out_dir)
    paths = write_local(export, out_dir)
    logger.info(
        "Wrote %d traces -> %s (also -> %s)",
        export["trace_count"], paths["snapshot"], paths["latest"],
    )

    if args.upload:
        try:
            file_id = upload_to_drive(paths["latest"], Path(args.token))
            logger.info("Uploaded latest.json to Drive (file_id=%s)", file_id)
        except ImportError:
            logger.error(
                "google-api-python-client not installed. "
                "Run: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
            )
            return 1
        except Exception as exc:
            logger.error("Drive upload failed: %s", exc)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
elp="Path to omega_traces.db (default: repo root)",
    )
    parser.add_argument(
        "--out-dir",
        default=str(_REPO_ROOT / "omega-exports"),
        help="Local output directory (default: <repo>/omega-exports)",
    )
    parser.add_argument("--league", default=None, help="Filter by league")
    parser.add_argument("--since", default=None, help="ISO timestamp lower bound")
    parser.add_argument("--until", default=None, help="ISO timestamp upper bound")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument(
        "--no-full-trace",
        action="store_true",
        help="Omit the full_trace blob (smaller files)",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Also upload latest.json to Drive (requires OAuth token)",
    )
    parser.add_argument(
        "--token",
        default=str(Path.home() / ".omega" / "drive_token.json"),
        help="Path to OAuth token JSON",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    db_path = Path(args.db)
    if not db_path.exists():
        logger.error("DB not found: %s", db_path)
        return 1

    export = build_export(
        db_path=db_path,
        league=args.league,
        since=args.since,
        until=args.until,
        limit=args.limit,
        include_full_trace=not args.no_full_trace,
    )

    out_dir = Path(args.out_dir)
    paths = write_local(export, out_dir)
    logger.info(
        "Wrote %d traces -> %s (also -> %s)",
        export["trace_count"], paths["snapshot"], paths["latest"],
    )

    if args.upload:
        try:
            file_id = upload_to_drive(paths["latest"], Path(args.token))
            logger.info("Uploaded latest.json to Drive (file_id=%s)", file_id)
        except ImportError:
            logger.error(
                "google-api-python-client not installed. Install with: "
                "pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
            )
            return 1
        except Exception as exc:
            logger.error("Drive upload failed: %s", exc)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
