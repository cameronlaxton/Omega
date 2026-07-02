"""Canonical calibration-league buckets.

A calibration profile corrects **one specific simulation model's** residual
probability error. Two league codes may therefore share a profile **only if they
are homogeneous for calibration**, which requires three things to hold:

1. **Same simulation backend.** A map fit on ``fast_score`` Poisson probabilities
   must never be served to ``soccer_bivariate_poisson_dc`` probabilities (or vice
   versa) — the raw-probability distribution differs, so the correction is wrong.
2. **Same market semantics.** (Enforced separately by the ``market`` key.)
3. **Similar residual / base-rate structure.** (An empirical question, validated
   by a pooled-vs-separate out-of-sample ECE check before promotion.)

"Same teams / same coaches" is **not** a sufficient reason to share a profile —
team strength already enters the model upstream via ratings/xG. This is why
international friendlies are *not* pooled with competitive internationals despite
involving the same national teams: their draw base-rate differs materially and
they run a different backend.

This module maps a runtime ``league`` code to its canonical calibration bucket.
Selection (``CalibrationRegistry.get_production``) resolves the bucket before
matching, so a profile fitted under the bucket code applies to every member
league. The map is curated to contain **only backend-homogeneous groupings**.

Pending entries (tied to the audit-remediation plan):
    Phase C1 flips WORLD_CUP / EURO / COPA_AMERICA / NATIONS_LEAGUE / AFCON onto
    ``soccer_bivariate_poisson_dc``. Once that lands they join the ``FIFA_INTL``
    bucket below (they are commented out until then so a future ``FIFA_INTL`` fit
    on bivariate-DC data cannot be served to a still-``fast_score`` league).
"""

from __future__ import annotations

# Runtime league code (UPPERCASE) -> canonical calibration bucket (UPPERCASE).
# Only include groupings that are backend-homogeneous *today*.
CALIBRATION_LEAGUE_BUCKETS: dict[str, str] = {
    # --- Pure naming aliases: identical league config + backend, different code.
    # Always safe to collapse (same model, same data).
    "PREMIER_LEAGUE": "EPL",
    "LALIGA": "LA_LIGA",
    # --- Competitive international soccer on the bivariate-Poisson-DC backend.
    # As of C1 all of these run the same backend (soccer_bivariate_poisson_dc),
    # so they share the FIFA_INTL calibration bucket. The C3 re-fit registers the
    # draw profile under league="FIFA_INTL"; this mapping makes it apply to every
    # member's live traces once promoted.
    "FIFA_WORLD_CUP_2026": "FIFA_INTL",
    "WORLD_CUP": "FIFA_INTL",
    "EURO": "FIFA_INTL",
    "COPA_AMERICA": "FIFA_INTL",
    "NATIONS_LEAGUE": "FIFA_INTL",
    # FIFA_FRIENDLY is deliberately NOT bucketed with FIFA_INTL: different draw
    # base-rate (~26% vs ~22%) and excluded from the fifa_intl fit dataset.
}


def resolve_calibration_bucket(league: str) -> str:
    """Map a runtime league code to its canonical calibration bucket.

    Uppercases the input, applies :data:`CALIBRATION_LEAGUE_BUCKETS`, and returns
    the league unchanged when it has no bucket mapping. Pure lookup; no I/O.
    """
    league_uc = league.upper()
    return CALIBRATION_LEAGUE_BUCKETS.get(league_uc, league_uc)


def resolve_prop_calibration_bucket(league: str, stat_type: str) -> str:
    """Canonical competition bucket for a PROP backend parameter profile.

    Prop structural knobs (e.g. the NB dispersion scale) correct one stat
    family's distribution shape, and per-stat dispersion differs materially
    (rushing vs passing yards), so the governed unit is per-(league, stat):
    ``{league_bucket}__{CANONICAL_STAT}`` (e.g. ``NFL__RUSHING_YARDS``). The
    league part reuses :func:`resolve_calibration_bucket`; the stat part reuses
    the prop-routing canonicalizer so market-key aliases (``pass_yds`` ->
    ``passing_yards``) collapse to one bucket. This is the single place prop
    parameter-profile buckets are named.
    """
    from omega.core.simulation.backends import canonical_prop_stat_type

    stat = canonical_prop_stat_type(league, stat_type).upper()
    return f"{resolve_calibration_bucket(league)}__{stat}"
