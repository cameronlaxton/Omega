"""
omega.ops.fit_nfl_dispersion — fit NFL Negative-Binomial dispersion ``k`` with
hierarchical Bayesian shrinkage.

Phase 7 M4 (design: docs/phase7/MULTI_SPORT_EXPANSION.md Part 5, decision 6).
Estimating ``k`` per-player from small NFL samples is unstable: a backup RB's
``k`` from 30 carries over-fits outlier games and manufactures false-positive
tail edges (longest-rush/longest-reception). The mitigation is mandatory
shrinkage toward a ``(position_group, stat_type)`` posterior:

1. per-player method-of-moments ``k`` from their game-level observations;
2. a group posterior ``k`` = mean of the group's valid per-player ``k`` values;
3. shrink ``k_player`` toward ``k_group`` with weight ``w(n) = n / (n + n0)``,
   where ``n`` is the player's observation count;
4. classify provenance: ``nb_k_source = "player"`` if ``w >= 0.6``,
   ``"position_group"`` if ``0.2 <= w < 0.6``, else ``"league"`` (cold start).

The backend reads only ``nb_dispersion_k``; all hierarchy lives here so runtime
latency stays flat. ``nb_k_source`` + ``nb_k_shrinkage_weight`` make a tail edge
auditable (genuine player signal vs. the group prior).

Note on ``n0``: the design sketch's "initial n0=8" is inconsistent with its own
acceptance behavior — at ``n0=8`` a 30-observation rookie gets ``w≈0.79`` and is
treated as a high-signal "player", defeating the red-team protection that
small-sample players must be shrunk toward the group. ``n0`` is tuned up to
``_DEFAULT_N0`` so a ~30-observation player is group-dominated (``w<0.6``) and a
~200-observation player is player-dominated (``w>=0.6``), matching the documented
M4 acceptance test.

The fit core (``fit_dispersions``) is pure and operates on in-memory
observations. The live nflverse loader is lazy-imported only inside ``main()``.

Usage:
    omega-fit-nfl-dispersion --season 2025
    omega-fit-nfl-dispersion --season 2025 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from omega.trace.priors import NflDispersionPrior, upsert_nfl_dispersion  # noqa: E402

logger = logging.getLogger("fit_nfl_dispersion")

# Shrinkage pseudocount. See module docstring: tuned above the design's "initial
# 8" so a ~30-observation player is group-dominated.
_DEFAULT_N0 = 50.0
# Provenance thresholds on the shrinkage weight w(n) = n / (n + n0).
_SOURCE_PLAYER_W = 0.6
_SOURCE_GROUP_W = 0.2
# Cold-start dispersion when a group has no estimable signal at all. A small-ish
# k keeps the NB right tail conservative rather than collapsing to Poisson.
_DEFAULT_LEAGUE_K = 4.0
# Clamp: an effectively-Poisson sample (var <= mean) yields no finite MoM k; we
# treat it as "no over-dispersion signal" (None) rather than a huge k.
_K_MAX = 1000.0

NB_K_SOURCE_PLAYER = "player"
NB_K_SOURCE_GROUP = "position_group"
NB_K_SOURCE_LEAGUE = "league"


@dataclass(frozen=True)
class DispersionObservation:
    """One game-level stat observation for one entity."""

    entity: str
    stat_type: str
    position_group: str
    value: float


@dataclass(frozen=True)
class EntityDispersionFit:
    """Shrunk dispersion for one entity, with provenance."""

    k: float
    weight: float
    source: str
    n_observations: int


def _mom_k(values: Sequence[float]) -> float | None:
    """Method-of-moments NB dispersion ``k`` from observations, or None.

    NB variance = mean + mean**2 / k, so ``k = mean**2 / (var - mean)`` when the
    sample is over-dispersed (``var > mean``). Returns None for fewer than two
    observations, a non-positive mean, or a non-over-dispersed sample — i.e. "no
    estimable over-dispersion signal", never a fabricated value.
    """
    n = len(values)
    if n < 2:
        return None
    mean = sum(values) / n
    if mean <= 0:
        return None
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    if var <= mean:
        return None
    k = mean * mean / (var - mean)
    return min(k, _K_MAX)


def _source_for_weight(weight: float) -> str:
    if weight >= _SOURCE_PLAYER_W:
        return NB_K_SOURCE_PLAYER
    if weight >= _SOURCE_GROUP_W:
        return NB_K_SOURCE_GROUP
    return NB_K_SOURCE_LEAGUE


def fit_group_k(
    values_by_entity: dict[str, Sequence[float]],
    *,
    league_default_k: float = _DEFAULT_LEAGUE_K,
) -> tuple[float, bool]:
    """Group posterior ``k`` = mean of valid per-player MoM ``k`` in the group.

    Falls back to ``league_default_k`` when no player in the group has an
    estimable over-dispersion signal. Returns ``(k, used_group_data)`` so
    downstream provenance can distinguish real group fits from league cold starts.
    """
    player_ks = [k for values in values_by_entity.values() if (k := _mom_k(values)) is not None]
    if not player_ks:
        return league_default_k, False
    return sum(player_ks) / len(player_ks), True


def shrink_entity_k(
    player_values: Sequence[float],
    group_k: float,
    *,
    group_k_from_data: bool = True,
    n0: float = _DEFAULT_N0,
) -> EntityDispersionFit:
    """Shrink one entity's MoM ``k`` toward ``group_k`` by ``w(n)=n/(n+n0)``.

    When the entity has no estimable signal of its own the result is the group
    ``k`` with a ``position_group`` (or ``league`` for ``group_k`` cold starts)
    source and weight 0.0 — never a silent player-level value.
    """
    n = len(player_values)
    weight = n / (n + n0) if (n + n0) > 0 else 0.0
    player_k = _mom_k(player_values)
    if player_k is None:
        source = NB_K_SOURCE_GROUP if group_k_from_data else NB_K_SOURCE_LEAGUE
        return EntityDispersionFit(k=group_k, weight=0.0, source=source, n_observations=n)
    shrunk = weight * player_k + (1.0 - weight) * group_k
    return EntityDispersionFit(
        k=shrunk,
        weight=weight,
        source=_source_for_weight(weight),
        n_observations=n,
    )


def fit_dispersions(
    observations: Iterable[DispersionObservation],
    *,
    season: str,
    as_of_date: str,
    n0: float = _DEFAULT_N0,
    league_default_k: float = _DEFAULT_LEAGUE_K,
) -> list[NflDispersionPrior]:
    """Fit shrunk NB dispersion rows for every (entity, stat_type) observed.

    Pure: groups observations by ``(position_group, stat_type)``, computes each
    group posterior, then shrinks each entity toward it. Returns rows ready for
    ``upsert_nfl_dispersion`` — no store/IO.
    """
    # group -> stat_type -> entity -> [values]; entity -> position_group.
    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    entity_group: dict[tuple[str, str], str] = {}
    for obs in observations:
        grouped[(obs.position_group, obs.stat_type)][obs.entity].append(obs.value)
        entity_group[(obs.entity, obs.stat_type)] = obs.position_group

    rows: list[NflDispersionPrior] = []
    for (position_group, stat_type), values_by_entity in grouped.items():
        group_k, group_k_from_data = fit_group_k(
            values_by_entity, league_default_k=league_default_k
        )
        for entity, values in values_by_entity.items():
            fit = shrink_entity_k(values, group_k, group_k_from_data=group_k_from_data, n0=n0)
            rows.append(
                NflDispersionPrior(
                    entity=entity,
                    stat_type=stat_type,
                    season=season,
                    position_group=position_group,
                    nb_dispersion_k=round(fit.k, 6),
                    nb_k_shrinkage_weight=round(fit.weight, 6),
                    nb_k_source=fit.source,
                    n_observations=fit.n_observations,
                    as_of_date=as_of_date,
                )
            )
    return rows


def run_fit(
    store,
    observations: Iterable[DispersionObservation],
    *,
    season: str,
    as_of_date: str,
    n0: float = _DEFAULT_N0,
    league_default_k: float = _DEFAULT_LEAGUE_K,
) -> list[NflDispersionPrior]:
    """Fit and persist NB dispersion rows to ``priors_nfl_dispersion``."""
    rows = fit_dispersions(
        observations,
        season=season,
        as_of_date=as_of_date,
        n0=n0,
        league_default_k=league_default_k,
    )
    for row in rows:
        upsert_nfl_dispersion(store, row)
    logger.info(
        "fit %d NFL dispersion rows for season=%s as_of=%s (sources: %s)",
        len(rows),
        season,
        as_of_date,
        {
            s: sum(1 for r in rows if r.nb_k_source == s)
            for s in (NB_K_SOURCE_PLAYER, NB_K_SOURCE_GROUP, NB_K_SOURCE_LEAGUE)
        },
    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit NFL NB dispersion k with hierarchical Bayesian shrinkage"
    )
    parser.add_argument("--season", required=True, help="Season label, e.g. 2025")
    parser.add_argument("--as-of", default=None, help="Fit as_of_date (default: today)")
    parser.add_argument("--n0", type=float, default=_DEFAULT_N0, help="Shrinkage pseudocount")
    parser.add_argument("--cache-root", default=None, help="Override ETL cache root")
    parser.add_argument("--db", default=None, help="SQLite path (default: var/omega_traces.db)")
    parser.add_argument("--dry-run", action="store_true", help="Fit and report; write nothing")
    parser.add_argument(
        "--emit-structure-candidate",
        action="store_true",
        help=(
            "After fitting per-player prop k, also sweep the NFL game-score "
            "dispersion (nfl_neg_binom backend, nb_k_scale) to minimize RAW OOS ECE "
            "on graded game traces and register a BackendParameterProfile candidate "
            "for the NFL game bucket. Distinct from the per-player prop k above; "
            "requires the --structure-* args below."
        ),
    )
    parser.add_argument(
        "--structure-knob", default="nb_k_scale", help="Knob to sweep (default nb_k_scale)"
    )
    parser.add_argument(
        "--structure-historical-db",
        default=None,
        help="Historical replay DB of graded NFL game traces",
    )
    parser.add_argument("--structure-validation-start", default=None, help="YYYY-MM-DD")
    parser.add_argument(
        "--structure-holdout-start", default=None, help="YYYY-MM-DD; strictly after validation"
    )
    parser.add_argument(
        "--register-structure", action="store_true", help="Persist the structure candidate"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Live observation source is the nflverse adapter (Phase 7 M4 follow-up); it
    # is imported lazily so the fit core stays usable/testable without it.
    try:
        from omega.integrations.nflverse import load_dispersion_observations
    except ImportError as exc:  # pragma: no cover - adapter not yet built
        logger.error(
            "nflverse adapter unavailable (%s); the offline fit core is usable via "
            "fit_dispersions(), but the live --season load is not yet wired",
            exc,
        )
        return 1

    try:
        observations = load_dispersion_observations(args.season, cache_root=args.cache_root)
    except Exception as exc:  # noqa: BLE001 - surface ETL failures loudly
        logger.error("could not load nflverse observations for %s: %s", args.season, exc)
        return 1

    as_of = args.as_of or date.today().isoformat()

    if args.dry_run:
        rows = fit_dispersions(observations, season=args.season, as_of_date=as_of, n0=args.n0)
        logger.info("[dry-run] fit %d rows — not written", len(rows))
    else:
        from omega.trace.store import TraceStore

        store = TraceStore(db_path=args.db) if args.db else TraceStore()
        try:
            run_fit(store, observations, season=args.season, as_of_date=as_of, n0=args.n0)
        finally:
            store.close()

    if args.emit_structure_candidate:
        missing = [
            name
            for name, val in (
                ("--structure-historical-db", args.structure_historical_db),
                ("--structure-validation-start", args.structure_validation_start),
                ("--structure-holdout-start", args.structure_holdout_start),
            )
            if not val
        ]
        if missing:
            logger.error("--emit-structure-candidate requires: %s", ", ".join(missing))
            return 1
        from omega.ops.fit_backend_structure import emit_structure_candidate_after_fit

        return emit_structure_candidate_after_fit(
            backend_name="nfl_neg_binom",
            league="NFL",
            knob=args.structure_knob,
            base_params={},
            historical_db=args.structure_historical_db,
            historical_only=True,
            validation_start=args.structure_validation_start,
            holdout_start=args.structure_holdout_start,
            priors_as_of=as_of,
            register=args.register_structure,
            db=args.db,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
