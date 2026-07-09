#!/usr/bin/env python
"""query_session_results.py - read-only summary of a session's analysis results.

Replaces the scratch query_results.py, which printed result fields that do not
exist in the trace schema (``model_prob``, ``calibrated_prob``, ``edge_pct``,
``recommended_selection`` — all rendered as None on every row). The real
result vocabulary is:

- prop traces: ``recommendation``, ``over_prob`` / ``under_prob``,
  ``edge_over`` / ``edge_under``, ``kelly_fraction``, ``recommended_units``,
  ``confidence_tier``, ``skip_reason``
- game traces: ``best_bet``, ``edges``, ``simulation_backend``, ``skip_reason``

The DB is opened via ``TraceStore(read_only=True)`` — canonical path
resolution, ``PRAGMA query_only=ON``, and a hard failure when the DB is
missing instead of sqlite silently creating an empty one.

Usage
-----
    python tools/query_session_results.py --session-id sess-20260704-185241a2f4
    python tools/query_session_results.py --session-id sess-... --league MLB --json-out out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    from omega.trace.store import TraceStore
except ImportError:  # running outside an installed env — fall back to src layout
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from omega.trace.store import TraceStore


def _fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return "-" if value is None else str(value)


def summarize_trace(trace_id: str, league: str | None, trace: dict[str, Any]) -> dict[str, Any]:
    snap = trace.get("input_snapshot") or {}
    result = trace.get("result") or {}
    kind = trace.get("kind") or snap.get("kind")
    row: dict[str, Any] = {
        "trace_id": trace_id,
        "league": league,
        "kind": kind,
        "status": result.get("status"),
        "skip_reason": result.get("skip_reason"),
        "home_team": snap.get("home_team"),
        "away_team": snap.get("away_team"),
    }
    if kind == "prop":
        row.update(
            player_name=snap.get("player_name"),
            prop_type=snap.get("prop_type"),
            line=result.get("line", snap.get("line")),
            recommendation=result.get("recommendation"),
            over_prob=result.get("over_prob"),
            under_prob=result.get("under_prob"),
            edge_over=result.get("edge_over"),
            edge_under=result.get("edge_under"),
            kelly_fraction=result.get("kelly_fraction"),
            recommended_units=result.get("recommended_units"),
            confidence_tier=result.get("confidence_tier"),
        )
    else:
        row.update(
            best_bet=result.get("best_bet"),
            edges=result.get("edges"),
            simulation_backend=result.get("simulation_backend"),
        )
    return row


def print_row(row: dict[str, Any]) -> None:
    header = f"{row['trace_id']}  [{row['league']}/{row['kind']}]  status={row['status']}"
    if row.get("skip_reason"):
        header += f"  skip_reason={row['skip_reason']}"
    print(header)
    if row["kind"] == "prop":
        print(
            f"  {row.get('player_name')} {row.get('prop_type')} "
            f"O/U {_fmt(row.get('line'), 1)}  ({row.get('away_team')} @ {row.get('home_team')})"
        )
        print(
            f"  rec={_fmt(row.get('recommendation'))}  "
            f"over_p={_fmt(row.get('over_prob'))}  under_p={_fmt(row.get('under_prob'))}  "
            f"edge_over={_fmt(row.get('edge_over'))}  edge_under={_fmt(row.get('edge_under'))}"
        )
        print(
            f"  kelly={_fmt(row.get('kelly_fraction'))}  "
            f"units={_fmt(row.get('recommended_units'), 2)}  "
            f"tier={_fmt(row.get('confidence_tier'))}"
        )
    else:
        print(f"  {row.get('away_team')} @ {row.get('home_team')}")
        best = row.get("best_bet")
        if isinstance(best, dict):
            printable = {
                k: best.get(k)
                for k in ("market", "selection", "edge", "kelly_fraction", "recommended_units")
                if k in best
            }
            print(f"  best_bet={json.dumps(printable, default=str)}")
        else:
            print(f"  best_bet={_fmt(best)}")
        edges = row.get("edges")
        if isinstance(edges, list) and edges:
            print(f"  {len(edges)} market edge row(s); backend={row.get('simulation_backend')}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize a session's traces (read-only).")
    parser.add_argument("--session-id", required=True, help="Session id (sess-...)")
    parser.add_argument("--league", help="Optional league filter, e.g. MLB")
    parser.add_argument("--json-out", help="Also write the summary rows as JSON")
    args = parser.parse_args(argv)

    try:
        store = TraceStore(read_only=True)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: cannot open trace DB read-only: {exc}")
        return 2

    sql = "SELECT trace_id, league, full_trace FROM traces WHERE session_id = ?"
    params: list[Any] = [args.session_id]
    if args.league:
        sql += " AND league = ?"
        params.append(args.league.upper())
    sql += " ORDER BY timestamp ASC"

    rows: list[dict[str, Any]] = []
    corrupt = 0
    for trace_id, league, raw in store.conn.execute(sql, params):
        try:
            trace = json.loads(raw)
        except (TypeError, ValueError):
            corrupt += 1
            continue
        rows.append(summarize_trace(trace_id, league, trace))
    store.close()

    if corrupt:
        print(f"WARNING: skipped {corrupt} rows with unparseable full_trace JSON\n")
    if not rows:
        print(
            f"No traces found for session {args.session_id}"
            + (f" (league={args.league.upper()})" if args.league else "")
        )
        return 1

    for row in rows:
        print_row(row)

    by_status: dict[str, int] = {}
    for row in rows:
        by_status[str(row["status"])] = by_status.get(str(row["status"]), 0) + 1
    n_recs = sum(
        1
        for r in rows
        if (r.get("recommendation") not in (None, "pass", "no_bet"))
        or isinstance(r.get("best_bet"), dict)
    )
    print(
        f"=== {len(rows)} trace(s)  "
        + "  ".join(f"{k}={v}" for k, v in sorted(by_status.items()))
        + f"  with_recommendation={n_recs} ==="
    )

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
        print(f"JSON summary -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
