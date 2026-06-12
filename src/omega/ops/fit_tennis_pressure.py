"""
omega.ops.fit_tennis_pressure — fit per-player pressure-state SPW% deltas.

Phase 7 M3 (design decision 7): point outcomes in ATP/WTA matches are not IID;
pressure states measurably shift serving distributions. This fitter consumes
Match Charting Project point-by-point data (via the Sackmann adapter) and, for
each player and surface, computes the additive delta between their SPW% in
each pressure state and their baseline SPW%:

    delta(state) = SPW%(points in state) - SPW%(all service points)

Players with fewer than N=500 charted service points on a surface fall back
to the tour+surface group mean and are written with ``source="group_fallback"``;
group rows are also written under the reserved ``__group__`` player key so
players with no charted data at all resolve to group means at request time —
flat 0.0 deltas are never silently applied.

State classification is EXCLUSIVE — every point lands in at most one fit
population — because the backend (tennis_markov._hold_for) applies deltas
ADDITIVELY at overlapping nodes (e.g. a break point inside a serving-for-set
game gets serving_for_set + break_point_against). Fitting those states on
overlapping populations would bake the same depression into both deltas and
double-count it at application (review finding, PR #12). Populations
(server's perspective; ``Pts`` is server-first):

  tiebreak             any service point at 6-6 games
  set_point_serving    game point (40-x / AD-40) in a serving-for-set game
  match_point_serving  same, when the set win clinches the match
  serving_for_set      NON-node points of a 5-x (x<=4) or 6-5 service game
  serving_for_match    same, when the set win clinches the match
  break_point_against  0-40 / 15-40 / 30-40 / 40-AD in a NON-serving-for-set
                       game (break points inside serving-for-set games belong
                       to no fit population; the additive model predicts them)

``set_point_serving``/``match_point_serving`` are RESIDUAL deltas, measured
against (baseline + the flat serving-for-set/-match delta), matching how the
backend stacks them: applied p = base + flat + gp_residual.

Usage:
    omega-fit-tennis-pressure-coefficients --tour atp --decades 2020s
    omega-fit-tennis-pressure-coefficients --tour wta --decades 2010s,2020s --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from omega.trace.priors import (  # noqa: E402
    PRESSURE_GROUP_PLAYER_KEY,
    PRESSURE_SOURCE_GROUP,
    PRESSURE_SOURCE_PLAYER,
    TENNIS_PRESSURE_STATES,
    TennisPressureDelta,
)

logger = logging.getLogger("fit_tennis_pressure")

_MIN_CHARTED_POINTS = 500
_BP_SCORES = frozenset({"0-40", "15-40", "30-40", "40-AD"})
_GP_SCORES = frozenset({"40-0", "40-15", "40-30", "AD-40"})


# Residual states are fit against (baseline + their flat parent's delta),
# mirroring the additive application in tennis_markov._hold_for.
_RESIDUAL_PARENT = {
    "set_point_serving": "serving_for_set",
    "match_point_serving": "serving_for_match",
}


def classify_point_state(
    point, *, server_sets: int, sets_to_win: int
) -> str | None:
    """Return the single (exclusive) fit population a charted point belongs to.

    None means the point is in no fit population: ordinary points, plain-game
    game points (no delta is applied there), and break points inside
    serving-for-set games (the overlap the additive model predicts rather
    than fits — counting them in two populations double-counts at application).
    """
    server_is_p1 = point.Svr == 1
    srv_games = point.Gm1 if server_is_p1 else point.Gm2
    ret_games = point.Gm2 if server_is_p1 else point.Gm1

    if srv_games == 6 and ret_games == 6:
        return "tiebreak"

    clinch = server_sets == sets_to_win - 1
    serving_for_set = (srv_games == 5 and ret_games <= 4) or (
        srv_games == 6 and ret_games == 5
    )
    if serving_for_set:
        if point.Pts in _BP_SCORES:
            return None  # overlap node: predicted additively, never fit
        if point.Pts in _GP_SCORES:
            return "match_point_serving" if clinch else "set_point_serving"
        return "serving_for_match" if clinch else "serving_for_set"
    if point.Pts in _BP_SCORES:
        return "break_point_against"
    return None


@dataclass
class _PlayerSurfaceAcc:
    total_pts: int = 0
    total_won: int = 0
    state_pts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    state_won: dict[str, int] = field(default_factory=lambda: defaultdict(int))


def accumulate_pressure_stats(
    points,
    matches,
    *,
    default_best_of: int = 3,
) -> dict[tuple[str, str], _PlayerSurfaceAcc]:
    """Aggregate per-(player, surface) baseline and per-state serve outcomes."""
    meta = {m.match_id: m for m in matches}
    acc: dict[tuple[str, str], _PlayerSurfaceAcc] = defaultdict(_PlayerSurfaceAcc)

    for point in points:
        match = meta.get(point.match_id)
        if match is None or match.surface is None or point.PtWinner is None:
            continue
        server_is_p1 = point.Svr == 1
        server = match.player_1 if server_is_p1 else match.player_2
        surface = match.surface.lower()
        best_of = match.best_of or default_best_of
        sets_to_win = best_of // 2 + 1
        server_sets = point.Set1 if server_is_p1 else point.Set2
        won = int(point.PtWinner == point.Svr)

        bucket = acc[(server, surface)]
        bucket.total_pts += 1
        bucket.total_won += won
        state = classify_point_state(
            point, server_sets=server_sets, sets_to_win=sets_to_win
        )
        if state is not None:
            bucket.state_pts[state] += 1
            bucket.state_won[state] += won
    return dict(acc)


def build_pressure_deltas(
    acc: dict[tuple[str, str], _PlayerSurfaceAcc],
    *,
    tour: str,
    as_of_date: str,
    min_points: int = _MIN_CHARTED_POINTS,
) -> list[TennisPressureDelta]:
    """Turn aggregates into delta rows with the N=500 group fallback.

    Group means per (surface, state) pool every player's state/baseline points
    so sub-threshold players inherit a defensible tour+surface behavior.
    """
    # Pooled group aggregates per surface.
    group: dict[str, _PlayerSurfaceAcc] = defaultdict(_PlayerSurfaceAcc)
    for (_, surface), bucket in acc.items():
        g = group[surface]
        g.total_pts += bucket.total_pts
        g.total_won += bucket.total_won
        for state, n in bucket.state_pts.items():
            g.state_pts[state] += n
            g.state_won[state] += bucket.state_won[state]

    def _deltas(bucket: _PlayerSurfaceAcc) -> dict[str, tuple[float, int]]:
        if bucket.total_pts <= 0:
            return {}
        baseline = bucket.total_won / bucket.total_pts
        out: dict[str, tuple[float, int]] = {}
        # Flat states first so residual states can reference their parent.
        for state in TENNIS_PRESSURE_STATES:
            if state in _RESIDUAL_PARENT:
                continue
            n = bucket.state_pts.get(state, 0)
            if n <= 0:
                continue
            out[state] = (bucket.state_won[state] / n - baseline, n)
        # Residual states: measured against (baseline + flat parent delta),
        # matching the backend's additive stacking. An unfitted parent
        # contributes 0.0 (the backend applies the same 0.0 then).
        for state, parent in _RESIDUAL_PARENT.items():
            n = bucket.state_pts.get(state, 0)
            if n <= 0:
                continue
            parent_delta = out.get(parent, (0.0, 0))[0]
            out[state] = (
                bucket.state_won[state] / n - (baseline + parent_delta),
                n,
            )
        return out

    rows: list[TennisPressureDelta] = []
    group_deltas = {surface: _deltas(bucket) for surface, bucket in group.items()}

    for surface, deltas in sorted(group_deltas.items()):
        for state, (delta, n) in sorted(deltas.items()):
            rows.append(
                TennisPressureDelta(
                    player=PRESSURE_GROUP_PLAYER_KEY,
                    tour=tour,
                    surface=surface,
                    state=state,
                    delta=round(delta, 5),
                    n_points=n,
                    source=PRESSURE_SOURCE_GROUP,
                    as_of_date=as_of_date,
                )
            )

    for (player, surface), bucket in sorted(acc.items()):
        use_player = bucket.total_pts >= min_points
        source = PRESSURE_SOURCE_PLAYER if use_player else PRESSURE_SOURCE_GROUP
        deltas = _deltas(bucket) if use_player else group_deltas.get(surface, {})
        for state, (delta, n) in sorted(deltas.items()):
            rows.append(
                TennisPressureDelta(
                    player=player,
                    tour=tour,
                    surface=surface,
                    state=state,
                    delta=round(delta, 5),
                    n_points=n,
                    source=source,
                    as_of_date=as_of_date,
                )
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit tennis pressure-state SPW%% deltas from Match Charting data"
    )
    parser.add_argument("--tour", choices=("atp", "wta"), required=True)
    parser.add_argument(
        "--decades",
        default="2020s",
        help="Comma-separated MCP point-file decades, e.g. 2010s,2020s",
    )
    parser.add_argument("--as-of", default=None, help="as_of_date stamp (default: today)")
    parser.add_argument("--min-points", type=int, default=_MIN_CHARTED_POINTS)
    parser.add_argument("--cache-root", default=None, help="Override ETL cache root")
    parser.add_argument("--db", default=None, help="SQLite path (default: var/omega_traces.db)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from omega.integrations.tennis_sackmann import (
        fetch_charting_csv,
        parse_charting_matches,
        parse_charting_points,
    )

    sex = "m" if args.tour == "atp" else "w"
    try:
        matches = parse_charting_matches(
            fetch_charting_csv(sex, "matches", cache_root=args.cache_root)
        )
        points = []
        for decade in (d.strip() for d in args.decades.split(",")):
            points.extend(
                parse_charting_points(
                    fetch_charting_csv(sex, f"points-{decade}", cache_root=args.cache_root)
                )
            )
    except Exception as exc:  # noqa: BLE001 - surface ETL failures loudly
        logger.error("could not load MCP data for %s: %s", args.tour, exc)
        return 1

    default_best_of = 3
    acc = accumulate_pressure_stats(points, matches, default_best_of=default_best_of)
    as_of = args.as_of or date.today().isoformat()
    rows = build_pressure_deltas(
        acc, tour=args.tour.upper(), as_of_date=as_of, min_points=args.min_points
    )
    player_rows = sum(1 for r in rows if r.source == PRESSURE_SOURCE_PLAYER)
    logger.info(
        "%s: %d delta rows (%d player-sourced, %d group) from %d charted points "
        "across %d (player, surface) cells",
        args.tour.upper(),
        len(rows),
        player_rows,
        len(rows) - player_rows,
        sum(b.total_pts for b in acc.values()),
        len(acc),
    )

    if args.dry_run:
        logger.info("Dry run — not writing priors_tennis_pressure.")
        return 0

    from omega.trace.priors import upsert_pressure_deltas
    from omega.trace.store import TraceStore

    store = TraceStore(db_path=args.db) if args.db else TraceStore()
    try:
        upsert_pressure_deltas(store, rows)
    finally:
        store.close()
    logger.info("Wrote %d priors_tennis_pressure rows.", len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
