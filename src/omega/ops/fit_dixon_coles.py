"""
omega.ops.fit_dixon_coles — fit per-competition Dixon-Coles rho profiles.

Phase 7 M2 (design: docs/phase7/MULTI_SPORT_EXPANSION.md Part 5, decision 5):
the Dixon-Coles low-score correlation ``rho`` is a dynamic prior fit per
competition profile (``fifa_intl_v1``, ``epl_v1``, ...), never a static league
config. This fitter:

1. loads the profile's (home_goals, away_goals) match dataset from the
   StatsBomb Open Data cache (``omega-refresh-statsbomb --profile <id>`` warms
   it; the fit itself then runs offline),
2. estimates competition-level Poisson means and minimises the Dixon-Coles
   negative log-likelihood over ``rho`` alone — each match contributes
   ``-log(tau(x, y; rho))`` which is convex in ``rho``, so a bounded ternary
   search inside the tau-positivity interval finds the global minimum,
3. writes a candidate row to ``priors_dixon_coles`` keyed
   (profile_id, as_of_date), and with ``--promote`` flips it to production
   (archiving the incumbent).

A fit row whose (profile_id, as_of_date) is already PRODUCTION is frozen for
the tournament duration and is never overwritten — rerun with a new ``--as-of``
and promote explicitly instead.

Usage:
    omega-fit-dixon-coles --profile fifa_intl_v1 --as-of 2026-06-10
    omega-fit-dixon-coles --profile fifa_intl_v1 --as-of 2026-06-10 --promote
    omega-fit-dixon-coles --profile fifa_intl_v1 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from omega.trace.priors import (  # noqa: E402
    DC_STATUS_PRODUCTION,
    DixonColesProfile,
    promote_dixon_coles_profile,
    upsert_dixon_coles_profile,
)

logger = logging.getLogger("fit_dixon_coles")

_DEFAULT_MIN_MATCHES = 100
_BOUND_EPS = 1e-3
_TERNARY_TOL = 1e-6


class FrozenProductionFitError(RuntimeError):
    """Refusal to overwrite a fit row that is live in production."""


@dataclass(frozen=True)
class DixonColesFit:
    rho: float
    n_matches: int
    fit_loss: float  # mean negative log-likelihood (Poisson + tau term)
    lambda_home: float
    lambda_away: float


def _tau(x: int, y: int, lambda_home: float, lambda_away: float, rho: float) -> float:
    """Dixon-Coles low-score correction factor for cell (x, y)."""
    if x == 0 and y == 0:
        return 1.0 - lambda_home * lambda_away * rho
    if x == 0 and y == 1:
        return 1.0 + lambda_home * rho
    if x == 1 and y == 0:
        return 1.0 + lambda_away * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def _rho_bounds(lambda_home: float, lambda_away: float) -> tuple[float, float]:
    """Open interval where every tau cell stays positive (Dixon-Coles 1997)."""
    lower = max(-1.0 / lambda_away, -1.0 / lambda_home)
    upper = min(1.0 / (lambda_home * lambda_away), 1.0)
    return lower + _BOUND_EPS, upper - _BOUND_EPS


def _tau_nll(
    pairs: list[tuple[int, int]], lambda_home: float, lambda_away: float, rho: float
) -> float:
    """The rho-dependent part of the DC negative log-likelihood."""
    total = 0.0
    for x, y in pairs:
        t = _tau(x, y, lambda_home, lambda_away, rho)
        if t <= 0.0:
            return math.inf
        total -= math.log(t)
    return total


def fit_rho(pairs: list[tuple[int, int]], min_matches: int = _DEFAULT_MIN_MATCHES) -> DixonColesFit:
    """Fit the competition-level Dixon-Coles rho from score pairs.

    Poisson means are the competition-level MLE (sample means); only the tau
    term depends on rho, and each ``-log(affine in rho)`` summand is convex, so
    ternary search over the tau-positivity interval is exact.
    """
    n = len(pairs)
    if n < min_matches:
        raise ValueError(
            f"only {n} matches in the fit dataset; need at least {min_matches}. "
            "Warm the cache with omega-refresh-statsbomb --profile <id> or lower --min-matches."
        )

    lambda_home = max(0.05, sum(h for h, _ in pairs) / n)
    lambda_away = max(0.05, sum(a for _, a in pairs) / n)
    lo, hi = _rho_bounds(lambda_home, lambda_away)

    while (hi - lo) > _TERNARY_TOL:
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if _tau_nll(pairs, lambda_home, lambda_away, m1) <= _tau_nll(
            pairs, lambda_home, lambda_away, m2
        ):
            hi = m2
        else:
            lo = m1
    rho = (lo + hi) / 2.0

    # Full mean NLL (Poisson terms + tau) for the audit row.
    nll = _tau_nll(pairs, lambda_home, lambda_away, rho)
    for x, y in pairs:
        nll -= (
            -lambda_home + x * math.log(lambda_home) - math.lgamma(x + 1)
            - lambda_away + y * math.log(lambda_away) - math.lgamma(y + 1)
        )
    return DixonColesFit(
        rho=round(rho, 6),
        n_matches=n,
        fit_loss=round(nll / n, 6),
        lambda_home=round(lambda_home, 4),
        lambda_away=round(lambda_away, 4),
    )


def run_fit(
    store,
    profile_id: str,
    pairs: list[tuple[int, int]],
    *,
    as_of_date: str,
    promote: bool = False,
    source: str = "statsbomb_open_data",
    min_matches: int = _DEFAULT_MIN_MATCHES,
) -> DixonColesFit:
    """Fit, persist a candidate row, and optionally promote it to production.

    Refuses to overwrite a (profile_id, as_of_date) row that is already in
    production — a frozen tournament fit must stay bit-stable for replay.
    """
    existing = store.conn.execute(
        "SELECT status FROM priors_dixon_coles WHERE profile_id = ? AND as_of_date = ?",
        (profile_id, as_of_date),
    ).fetchone()
    if existing is not None and existing[0] == DC_STATUS_PRODUCTION:
        raise FrozenProductionFitError(
            f"fit {profile_id!r} as_of {as_of_date!r} is the frozen production row; "
            "refit under a new --as-of and promote explicitly"
        )

    fit = fit_rho(pairs, min_matches=min_matches)
    upsert_dixon_coles_profile(
        store,
        DixonColesProfile(
            profile_id=profile_id,
            rho=fit.rho,
            n_matches=fit.n_matches,
            fit_loss=fit.fit_loss,
            as_of_date=as_of_date,
            source=source,
        ),
    )
    logger.info(
        "fit %s: rho=%.4f (n=%d, mean NLL=%.4f, lambda_h=%.2f, lambda_a=%.2f) as_of=%s",
        profile_id,
        fit.rho,
        fit.n_matches,
        fit.fit_loss,
        fit.lambda_home,
        fit.lambda_away,
        as_of_date,
    )
    if promote:
        promoted = promote_dixon_coles_profile(store, profile_id, as_of_date)
        logger.info(
            "promoted %s as_of=%s to production (rho=%.4f)",
            promoted.profile_id,
            promoted.as_of_date,
            promoted.rho,
        )
    return fit


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit a per-competition Dixon-Coles rho profile from StatsBomb Open Data"
    )
    parser.add_argument("--profile", required=True, help="Profile id, e.g. fifa_intl_v1")
    parser.add_argument("--as-of", default=None, help="Fit as_of_date (default: today)")
    parser.add_argument(
        "--promote", action="store_true", help="Promote this fit to production after writing"
    )
    parser.add_argument("--min-matches", type=int, default=_DEFAULT_MIN_MATCHES)
    parser.add_argument(
        "--seasons",
        default=None,
        help=(
            "Comma-separated season_name filter (e.g. '2015/2016'). Use for club "
            "profiles where open-data mixes full-league and single-team seasons."
        ),
    )
    parser.add_argument("--cache-root", default=None, help="Override ETL cache root")
    parser.add_argument("--db", default=None, help="SQLite path (default: var/omega_traces.db)")
    parser.add_argument("--dry-run", action="store_true", help="Fit and report; write nothing")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from omega.integrations.statsbomb import load_profile_matches

    seasons = tuple(s.strip() for s in args.seasons.split(",")) if args.seasons else None
    try:
        pairs = load_profile_matches(
            args.profile, seasons=seasons, cache_root=args.cache_root
        )
    except Exception as exc:  # noqa: BLE001 - surface ETL failures loudly
        logger.error("could not load fit dataset for %s: %s", args.profile, exc)
        return 1

    as_of = args.as_of or date.today().isoformat()

    if args.dry_run:
        try:
            fit = fit_rho(pairs, min_matches=args.min_matches)
        except ValueError as exc:
            logger.error("%s", exc)
            return 1
        logger.info(
            "[dry-run] %s: rho=%.4f (n=%d, mean NLL=%.4f) — not written",
            args.profile,
            fit.rho,
            fit.n_matches,
            fit.fit_loss,
        )
        return 0

    from omega.trace.store import TraceStore

    store = TraceStore(db_path=args.db) if args.db else TraceStore()
    try:
        run_fit(
            store,
            args.profile,
            pairs,
            as_of_date=as_of,
            promote=args.promote,
            min_matches=args.min_matches,
        )
    except (ValueError, FrozenProductionFitError) as exc:
        logger.error("%s", exc)
        return 1
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
