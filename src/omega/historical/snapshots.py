"""As-of-safe feature snapshot builder.

Produces Omega-compatible ``home_context`` / ``away_context`` / ``game_context``
and calibration ``context_labels`` from pre-decision history only. The core
guarantee: every history row at/after ``decision_time`` is dropped before any
feature is computed, so a snapshot can never embed post-decision information.
``game_context`` always carries ``is_playoff`` and ``rest_days`` (the keys the
analyze() request validators require).

Sport-specific rules cover NFL/American-football, basketball, soccer, tennis,
baseball, and hockey. Situational slices that need richer inputs (injuries,
weather, surface, goalie status, division) are read from ``extra_game_context``
supplied by the lane adapter; when absent the slice label is simply omitted
(unknown), never asserted false.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from omega.core.simulation.archetypes import get_archetype
from omega.historical.contracts import HistoricalEvent, HistoricalFeatureSnapshot
from omega.historical.normalize import parse_datetime_utc

UTC = timezone.utc

_DEFAULT_ROLLING_WINDOW = 10
_DEFAULT_STALENESS_DAYS = 120
_DEFAULT_REST_DAYS = 7  # used when no prior game is known (flagged via game_context)

# Empirical-Bayes shrinkage strength for team off/def ratings (in "prior games"):
# a team's rolling mean is blended (n·mean + n0·league_mean)/(n+n0). Small-sample
# teams regress hard toward the league baseline, which tempers the over-dispersed
# raw rolling means that drive tail overconfidence on the game/moneyline plane.
# Opt-in via build_feature_snapshot(shrink_ratings=True) so live/legacy callers are
# unaffected; the value is only applied when a league_baseline is supplied.
_RATING_SHRINKAGE_N0 = 5


@dataclass
class TeamGameRow:
    """One prior game for a team, used to compute as-of features.

    All fields except ``date`` are optional so the same row type serves every
    sport; tennis populates the serve/return point fields, team-score sports
    populate ``points_for`` / ``points_against``.
    """

    date: str
    points_for: float | None = None
    points_against: float | None = None
    was_home: bool = True
    opponent: str | None = None
    # tennis serve/return point tallies
    serve_points_won: float | None = None
    serve_points_total: float | None = None
    return_points_won: float | None = None
    return_points_total: float | None = None


@dataclass
class MatchupHistory:
    """Prior games for both sides plus an optional league baseline context."""

    home_rows: list[TeamGameRow] = field(default_factory=list)
    away_rows: list[TeamGameRow] = field(default_factory=list)
    league_baseline: dict[str, float] | None = None


def _dt(value: str) -> datetime:
    iso = parse_datetime_utc(value)
    dt = datetime.fromisoformat(iso)
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _as_of_filter(rows: list[TeamGameRow], decision_dt: datetime) -> list[TeamGameRow]:
    """Keep only rows strictly before the decision cutoff (as-of safety)."""
    kept = [r for r in rows if _dt(r.date) < decision_dt]
    kept.sort(key=lambda r: _dt(r.date))
    return kept


def _rest_days(last_game: str | None, event_start: str) -> int | None:
    """Calendar gap minus one, floored at 0 (0 == back-to-back). Mirrors the
    live ``game_context`` convention so historical and live slices align."""
    if not last_game:
        return None
    delta = (_dt(event_start).date() - _dt(last_game).date()).days - 1
    return max(delta, 0)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ---------------------------------------------------------------------------
# Per-family team-context computation
# ---------------------------------------------------------------------------


def _default_team_context(league: str, family: str) -> dict[str, float]:
    arch = get_archetype(league)
    if family == "tennis":
        return {"serve_win_pct": 0.62, "return_win_pct": 0.38}
    half = (arch.avg_total / 2.0) if arch else 100.0
    ctx: dict[str, float] = {"off_rating": round(half, 3), "def_rating": round(half, 3)}
    if family == "basketball":
        ctx["pace"] = round(arch.avg_tempo if arch else 100.0, 3)
    return ctx


def _shrink_rating(value: float, league_value: float | None, n: int, n0: int) -> float:
    """Empirical-Bayes blend of a team's rolling mean toward the league baseline."""
    if league_value is None or n <= 0:
        return value
    return (n * value + n0 * league_value) / (n + n0)


def _team_context(
    rows: list[TeamGameRow],
    league: str,
    family: str,
    baseline: dict[str, float] | None,
    *,
    shrink: bool = False,
    shrink_n0: int = _RATING_SHRINKAGE_N0,
) -> tuple[dict[str, float], bool]:
    """Return (context, used_default). Computes archetype-required keys.

    When ``shrink`` is set and a ``baseline`` with off/def is supplied, the team's
    rolling off/def ratings are empirical-Bayes shrunk toward the league baseline
    (see :data:`_RATING_SHRINKAGE_N0`). This does not change ``used_default``: a
    team with data is still "provided" — the ratings are merely regularized.
    """
    if family == "tennis":
        serve_rows = [r for r in rows if r.serve_points_total]
        ret_rows = [r for r in rows if r.return_points_total]
        if serve_rows and ret_rows:
            swp = sum(r.serve_points_won or 0 for r in serve_rows) / sum(
                r.serve_points_total or 0 for r in serve_rows
            )
            rwp = sum(r.return_points_won or 0 for r in ret_rows) / sum(
                r.return_points_total or 0 for r in ret_rows
            )
            return {"serve_win_pct": round(swp, 4), "return_win_pct": round(rwp, 4)}, False
        return (baseline or _default_team_context(league, family)), True

    scored = [r for r in rows if r.points_for is not None]
    allowed = [r for r in rows if r.points_against is not None]
    if scored and allowed:
        off = _mean([r.points_for for r in scored])  # type: ignore[misc]
        deff = _mean([r.points_against for r in allowed])  # type: ignore[misc]
        if shrink and baseline:
            off = _shrink_rating(off, baseline.get("off_rating"), len(scored), shrink_n0)
            deff = _shrink_rating(deff, baseline.get("def_rating"), len(allowed), shrink_n0)
        ctx: dict[str, float] = {
            "off_rating": round(off, 3),
            "def_rating": round(deff, 3),
        }
        if family == "basketball":
            totals = [
                (r.points_for + r.points_against)
                for r in rows
                if r.points_for is not None and r.points_against is not None
            ]
            ctx["pace"] = round(_mean(totals) / 2.0 if totals else 100.0, 3)
        return ctx, False
    return (baseline or _default_team_context(league, family)), True


# ---------------------------------------------------------------------------
# Per-family context-label (slice) computation
# ---------------------------------------------------------------------------


def _games_in_last_n_days(rows: list[TeamGameRow], event_start: str, days: int) -> int:
    cutoff = _dt(event_start)
    return sum(1 for r in rows if 0 <= (cutoff.date() - _dt(r.date).date()).days <= days)


def _context_labels(
    family: str,
    event: HistoricalEvent,
    game_context: dict,
    home_rows: list[TeamGameRow],
    away_rows: list[TeamGameRow],
    home_ctx: dict,
    away_ctx: dict,
) -> dict:
    """Compute derivable slice labels; pass through any extras already present."""
    home_rest = game_context.get("home_rest_days")
    away_rest = game_context.get("away_rest_days")
    labels: dict = {
        "is_playoff": event.is_playoff,
        "neutral_site": event.is_neutral_site,
    }

    def _min_rest() -> int | None:
        vals = [v for v in (home_rest, away_rest) if v is not None]
        return min(vals) if vals else None

    if family == "american_football":
        mr = _min_rest()
        if mr is not None:
            labels["short_week"] = mr <= 3
        if event.is_playoff:
            labels["playoff"] = True
        # backup_qb, weather_extreme, division_game arrive via extra_game_context
    elif family == "basketball":
        if home_rest is not None or away_rest is not None:
            labels["back_to_back"] = (home_rest == 0) or (away_rest == 0)
        if home_rest is not None and away_rest is not None:
            labels["rest_disadvantage"] = abs(home_rest - away_rest) >= 2
        if event.is_playoff:
            labels["playoff"] = True
        # lineup_uncertain, star_absent arrive via extra_game_context
    elif family == "soccer":
        mr = _min_rest()
        if mr is not None:
            labels["congested_fixture"] = mr <= 3
        # cup_match, derby, rotation_risk, international_break via extra
    elif family == "tennis":
        sd = max(home_ctx.get("serve_win_pct", 0.0), away_ctx.get("serve_win_pct", 0.0))
        labels["serve_dominant"] = sd >= 0.66
        if game_context.get("best_of") == 5:
            labels["best_of_5"] = True
        surface = game_context.get("surface")
        if surface:
            labels[f"surface_{str(surface).lower()}"] = True
        # pressure_state via extra
    elif family == "baseball":
        pf = game_context.get("park_factor")
        if pf is not None:
            labels["park_factor_extreme"] = pf >= 1.1 or pf <= 0.9
        # starting_pitcher_change, bullpen_taxed, weather_wind_out via extra
    elif family == "hockey":
        if home_rest is not None or away_rest is not None:
            labels["back_to_back"] = (home_rest == 0) or (away_rest == 0)
        # 3 games in 4 nights (incl. tonight) ⇒ at least 2 prior games in the last 3 days.
        three_home = _games_in_last_n_days(home_rows, event.start_time, 3) >= 2
        three_away = _games_in_last_n_days(away_rows, event.start_time, 3) >= 2
        labels["three_in_four"] = three_home or three_away
        # goalie_confirmed, goalie_uncertain via extra

    return labels


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def build_feature_snapshot(
    event: HistoricalEvent,
    history: MatchupHistory,
    decision_time: str,
    *,
    extra_game_context: dict | None = None,
    rolling_window: int = _DEFAULT_ROLLING_WINDOW,
    staleness_days: int = _DEFAULT_STALENESS_DAYS,
    shrink_ratings: bool = False,
) -> HistoricalFeatureSnapshot:
    """Build an as-of-safe :class:`HistoricalFeatureSnapshot` for one event.

    ``shrink_ratings`` (opt-in) empirical-Bayes shrinks team off/def ratings toward
    ``history.league_baseline`` to temper over-dispersed small-sample rolling means.
    Default off, so live/legacy callers see the existing raw-mean behavior.
    """
    decision_dt = _dt(decision_time)
    extra = dict(extra_game_context or {})

    # As-of safety: drop anything at/after the decision cutoff, then window.
    home_rows = _as_of_filter(history.home_rows, decision_dt)[-rolling_window:]
    away_rows = _as_of_filter(history.away_rows, decision_dt)[-rolling_window:]

    used_rows = home_rows + away_rows
    as_of = max((r.date for r in used_rows), default=None)
    as_of_iso = parse_datetime_utc(as_of) if as_of else None

    home_ctx, home_default = _team_context(
        home_rows, event.league, event.sport_family, history.league_baseline,
        shrink=shrink_ratings,
    )
    away_ctx, away_default = _team_context(
        away_rows, event.league, event.sport_family, history.league_baseline,
        shrink=shrink_ratings,
    )

    home_last = home_rows[-1].date if home_rows else None
    away_last = away_rows[-1].date if away_rows else None
    home_rest = _rest_days(home_last, event.start_time)
    away_rest = _rest_days(away_last, event.start_time)

    game_context: dict = {
        "is_playoff": event.is_playoff,
        "rest_days": home_rest if home_rest is not None else _DEFAULT_REST_DAYS,
        "rest_days_known": home_rest is not None,
        "home_rest_days": home_rest,
        "away_rest_days": away_rest,
        "neutral_site": event.is_neutral_site,
    }
    game_context.update(extra)

    if home_default and away_default:
        context_source = "default"
    elif home_default or away_default:
        context_source = "backfilled"
    else:
        context_source = "provided"

    labels = _context_labels(
        event.sport_family, event, game_context, home_rows, away_rows, home_ctx, away_ctx
    )

    is_stale = False
    if as_of_iso is not None:
        gap = (decision_dt.date() - _dt(as_of_iso).date()).days
        is_stale = gap > staleness_days

    snapshot = HistoricalFeatureSnapshot(
        event_id=event.event_id,
        league=event.league,
        sport_family=event.sport_family,
        decision_time=decision_time,
        home_context=home_ctx,
        away_context=away_ctx,
        game_context=game_context,
        context_labels=labels,
        context_source=context_source,
        is_stale=is_stale,
        as_of=as_of_iso,
    )
    snapshot.feature_snapshot_hash = snapshot.compute_hash()
    return snapshot
