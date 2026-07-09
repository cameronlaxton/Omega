#!/usr/bin/env python
"""extract_contexts.py - build per-league team/player context packs from the trace DB.

Consolidates the per-league scratch scripts (extract_mlb_team_contexts.py,
extract_mlb_player_contexts.py, ...) that each re-implemented the same scan with
three recurring defects this tool closes:

1. CWD-relative ``sqlite3.connect('var/omega_traces.db')`` silently *creates an
   empty DB* when run from the wrong directory and reports "0 traces". Here the
   DB is opened through ``TraceStore(read_only=True)`` — canonical path
   resolution (env override / redirect guard) and a hard failure if the file
   does not exist.
2. No recency ordering: last-row-wins left contexts arbitrarily stale. Here the
   scan is ordered by trace timestamp so the newest successful trace wins, and
   every context carries provenance (source trace_id + timestamp + age).
3. Fabricated fallbacks (a hardcoded Colorado Rockies context) injected
   indistinguishably from real data. This tool NEVER invents contexts — missing
   teams/players are reported, and ``--strict`` turns staleness or emptiness
   into a non-zero exit.

Contexts extracted from prior traces are inherently backward-looking. Treat the
pack as a starting point for a slate, not a substitute for fresh research —
that is why the output embeds per-entry age and a ``stale`` list.

Output envelope (consumed by tools/build_slate_entries.py, which also accepts
plain ``{name: ctx}`` mappings):

    {
      "league": "MLB", "kind": "team", "extracted_at": "...", "db_path": "...",
      "max_age_days": 30,
      "contexts":   {"<team-or-player>": {...}},
      "provenance": {"<key>": {"trace_id": ..., "trace_timestamp": ..., "age_days": ...}},
      "stale":      ["<keys older than max_age_days>"]
    }

Usage
-----
    python tools/extract_contexts.py --league MLB                  # team + player packs
    python tools/extract_contexts.py --league WNBA --kind team
    python tools/extract_contexts.py --league MLB --max-age-days 14 --strict
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    from omega.trace.store import TraceStore
except ImportError:  # running outside an installed env — fall back to src layout
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from omega.trace.store import TraceStore

# Per-game situational keys that must not leak into a reusable team pack.
DEFAULT_TEAM_DROP_KEYS = frozenset({"weather_wind_mph"})


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


def _trace_kind(trace: dict[str, Any]) -> str | None:
    """Trace kind, inferring 'game' for legacy traces that predate the field."""
    kind = trace.get("kind")
    if kind:
        return kind
    snap = trace.get("input_snapshot") or {}
    if "home_context" in snap or "away_context" in snap:
        return "game"
    return None


def _iter_successful_traces(store: TraceStore, league: str):
    """Yield (trace_id, trace_dict) for successful traces, oldest first.

    Oldest-first means plain dict assignment leaves the newest context standing.
    Corrupt full_trace rows are counted, not silently swallowed.
    """
    corrupt = 0
    rows = store.conn.execute(
        "SELECT trace_id, full_trace FROM traces WHERE league = ? ORDER BY timestamp ASC",
        (league,),
    )
    for trace_id, raw in rows:
        try:
            trace = json.loads(raw)
        except (TypeError, ValueError):
            corrupt += 1
            continue
        result = trace.get("result") or {}
        if result.get("status") != "success":
            continue
        yield trace_id, trace
    if corrupt:
        print(f"WARNING: skipped {corrupt} rows with unparseable full_trace JSON")


def extract_team_contexts(
    store: TraceStore, league: str, drop_keys: frozenset[str]
) -> tuple[dict[str, dict], dict[str, dict]]:
    contexts: dict[str, dict] = {}
    provenance: dict[str, dict] = {}
    for trace_id, trace in _iter_successful_traces(store, league):
        if _trace_kind(trace) != "game":
            continue
        snap = trace.get("input_snapshot") or {}
        ts = trace.get("timestamp")
        for name_key, ctx_key in (("home_team", "home_context"), ("away_team", "away_context")):
            name, ctx = snap.get(name_key), snap.get(ctx_key)
            if not name or not isinstance(ctx, dict) or not ctx:
                continue
            contexts[name] = {k: v for k, v in ctx.items() if k not in drop_keys}
            provenance[name] = {"trace_id": trace_id, "trace_timestamp": ts}
    return contexts, provenance


def extract_player_contexts(
    store: TraceStore, league: str
) -> tuple[dict[str, dict], dict[str, dict]]:
    contexts: dict[str, dict] = {}
    provenance: dict[str, dict] = {}
    for trace_id, trace in _iter_successful_traces(store, league):
        if _trace_kind(trace) != "prop":
            continue
        snap = trace.get("input_snapshot") or {}
        name = snap.get("player_name")
        ctx = snap.get("player_context")
        prop_type = snap.get("prop_type")
        ts = trace.get("timestamp")
        if not name or not prop_type or not isinstance(ctx, dict):
            continue
        # Numeric parameters only: strings in stored player_context are labels,
        # not simulation inputs (bools are ints in Python — exclude them).
        numeric = {
            k: v for k, v in ctx.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        if not numeric:
            continue
        contexts.setdefault(name, {})[prop_type] = numeric
        provenance[f"{name}|{prop_type}"] = {"trace_id": trace_id, "trace_timestamp": ts}
    return contexts, provenance


def _stamp_ages(provenance: dict[str, dict], now: datetime, max_age_days: float) -> list[str]:
    stale: list[str] = []
    for key, prov in provenance.items():
        ts = _parse_ts(prov.get("trace_timestamp"))
        if ts is None:
            prov["age_days"] = None
            stale.append(key)
            continue
        age = (now - ts).total_seconds() / 86400.0
        prov["age_days"] = round(age, 1)
        if age > max_age_days:
            stale.append(key)
    return sorted(stale)


def _write_pack(
    path: Path,
    *,
    league: str,
    kind: str,
    db_path: str,
    contexts: dict,
    provenance: dict,
    stale: list[str],
    max_age_days: float,
) -> None:
    envelope = {
        "league": league,
        "kind": kind,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "db_path": db_path,
        "max_age_days": max_age_days,
        "contexts": contexts,
        "provenance": provenance,
        "stale": stale,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(envelope, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract per-league team/player context packs from the trace DB."
    )
    parser.add_argument("--league", required=True, help="League identifier, e.g. MLB, WNBA")
    parser.add_argument(
        "--kind", choices=("team", "player", "both"), default="both", help="What to extract"
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "var" / "context_packs"),
        help="Directory for the pack files (default: var/context_packs)",
    )
    parser.add_argument(
        "--max-age-days",
        type=float,
        default=30.0,
        help="Contexts sourced from traces older than this are flagged stale (default: 30)",
    )
    parser.add_argument(
        "--drop-key",
        action="append",
        default=None,
        help=(
            "Team-context key to strip as per-game situational "
            f"(repeatable; default: {sorted(DEFAULT_TEAM_DROP_KEYS)})"
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when a pack is empty or contains stale entries",
    )
    args = parser.parse_args(argv)

    league = args.league.upper()
    drop_keys = frozenset(args.drop_key) if args.drop_key else DEFAULT_TEAM_DROP_KEYS
    out_dir = Path(args.output_dir)
    now = datetime.now(timezone.utc)

    try:
        store = TraceStore(read_only=True)
        db_path = store.db_path
        store.conn  # force the connection open so a missing DB fails here, loudly
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: cannot open trace DB read-only: {exc}")
        return 2

    failures = 0
    jobs: list[str] = ["team", "player"] if args.kind == "both" else [args.kind]
    for kind in jobs:
        if kind == "team":
            contexts, provenance = extract_team_contexts(store, league, drop_keys)
        else:
            contexts, provenance = extract_player_contexts(store, league)
        stale = _stamp_ages(provenance, now, args.max_age_days)
        out_path = out_dir / f"{league.lower()}_{kind}_contexts.json"
        _write_pack(
            out_path,
            league=league,
            kind=kind,
            db_path=db_path,
            contexts=contexts,
            provenance=provenance,
            stale=stale,
            max_age_days=args.max_age_days,
        )
        unit = "teams" if kind == "team" else "players"
        print(f"{kind}: {len(contexts)} {unit} -> {out_path}")
        if stale:
            print(f"  WARNING: {len(stale)} entries older than {args.max_age_days:g} days:")
            for key in stale[:10]:
                prov = provenance.get(key, {})
                print(f"    - {key} (age_days={prov.get('age_days')})")
            if len(stale) > 10:
                print(f"    ... and {len(stale) - 10} more (see 'stale' in the pack)")
        if not contexts:
            print(f"  WARNING: no successful {league} {kind} contexts found in {db_path}")
        if args.strict and (stale or not contexts):
            failures += 1

    store.close()
    if failures:
        print(f"STRICT: {failures} pack(s) empty or stale - refusing to bless this extraction.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
