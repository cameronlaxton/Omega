"""omega-soccer-prior-coverage: pre-session FIFA/soccer prior-coverage gate.

Checks whether a FIFA/soccer session has sufficient dynamic priors to produce
trustworthy actionable output.  Run this before any analysis that targets a
soccer league (WORLD_CUP, FIFA_WORLD_CUP_2026, EPL, LA_LIGA, etc.).

Confidence tiers
----------------
strong        Production DC rho profile present AND at least one team has xG data.
moderate      Production DC rho profile present but no team xG, OR xG present but
              no production rho.  Outputs are valid but xG-derived lambda paths are
              unavailable or uncalibrated.
weak          DC profile exists only as a candidate (not yet promoted).  Engine will
              fail closed on rho_prior unless caller supplies rho manually.
none          No DC profile at all for this competition profile.  Engine will skip.

Output-mode gate
----------------
strong / moderate   → allow up to the caller's requested output mode (e.g. actionable)
moderate            → optionally downgrade to ``low_confidence_actionable`` (flag set)
weak / none         → force ``research_candidate``; actionable output is suppressed

Exit codes
----------
0   Strong or moderate coverage — session may proceed.
1   Error (DB unavailable, league not recognised).
2   Weak or no coverage — actionable output suppressed; session continues as research.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from omega.trace.priors import (  # noqa: E402
    DixonColesProfile,
    XgPrior,
    get_production_dc_profile,
    get_xg_prior,
)
from omega.trace.store import TraceStore  # noqa: E402

logger = logging.getLogger("omega.ops.soccer_prior_coverage")

# Confidence tiers (ordered weakest → strongest)
TIER_NONE = "none"
TIER_WEAK = "weak"
TIER_MODERATE = "moderate"
TIER_STRONG = "strong"

# Output-mode recommendations that map to tiers
_TIER_OUTPUT_MODE: dict[str, str] = {
    TIER_STRONG: "actionable",
    TIER_MODERATE: "low_confidence_actionable",
    TIER_WEAK: "research_candidate",
    TIER_NONE: "research_candidate",
}


@dataclass
class TeamXgCoverage:
    team: str
    xg_for: float | None = None
    xg_against: float | None = None
    source: str | None = None
    as_of_date: str | None = None
    matches: int | None = None
    has_xg: bool = False


@dataclass
class SoccerPriorCoverageReport:
    """Full prior-coverage report for one soccer event / league pair."""

    league: str
    competition_profile_id: str | None  # rho_fit_profile key from league config

    # Dixon-Coles rho status
    dc_profile: DixonColesProfile | None  # None = no production row
    dc_candidate_exists: bool = False  # True if candidate (not promoted) row found

    # Per-team xG coverage (None entries for teams not requested)
    home_team: TeamXgCoverage | None = None
    away_team: TeamXgCoverage | None = None

    # Derived fields (set by build_coverage_report)
    confidence_tier: str = TIER_NONE
    recommended_output_mode: str = "research_candidate"
    fallback_usage: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        dc = self.dc_profile
        return {
            "league": self.league,
            "competition_profile_id": self.competition_profile_id,
            "dc_profile": {
                "profile_id": dc.profile_id,
                "rho": dc.rho,
                "n_matches": dc.n_matches,
                "as_of_date": dc.as_of_date,
                "source": dc.source,
                "status": dc.status,
            }
            if dc
            else None,
            "dc_candidate_exists": self.dc_candidate_exists,
            "home_team": _team_xg_dict(self.home_team),
            "away_team": _team_xg_dict(self.away_team),
            "confidence_tier": self.confidence_tier,
            "recommended_output_mode": self.recommended_output_mode,
            "fallback_usage": self.fallback_usage,
            "warnings": self.warnings,
        }


def _team_xg_dict(coverage: TeamXgCoverage | None) -> dict[str, Any] | None:
    if coverage is None:
        return None
    return {
        "team": coverage.team,
        "xg_for": coverage.xg_for,
        "xg_against": coverage.xg_against,
        "source": coverage.source,
        "as_of_date": coverage.as_of_date,
        "matches": coverage.matches,
        "has_xg": coverage.has_xg,
    }


def _lookup_team_xg(
    store: TraceStore,
    team: str,
    competition: str,
    season: str | None,
) -> TeamXgCoverage:
    """Query priors_xg for a team; tries blank season as fallback."""
    cov = TeamXgCoverage(team=team)
    seasons = [season] if season else [""]
    seasons_to_try = list(dict.fromkeys(seasons + [""]))

    for s in seasons_to_try:
        prior: XgPrior | None = get_xg_prior(store, team, competition, s)
        if prior is not None:
            cov.xg_for = prior.xg_for
            cov.xg_against = prior.xg_against
            cov.source = prior.source
            cov.as_of_date = prior.as_of_date
            cov.matches = prior.matches
            cov.has_xg = True
            break
    return cov


def _check_candidate_exists(store: TraceStore, profile_id: str) -> bool:
    row = store.conn.execute(
        "SELECT 1 FROM priors_dixon_coles WHERE profile_id = ? LIMIT 1",
        (profile_id,),
    ).fetchone()
    return row is not None


def build_coverage_report(
    league: str,
    *,
    home_team: str | None = None,
    away_team: str | None = None,
    season: str | None = None,
    store: TraceStore | None = None,
) -> SoccerPriorCoverageReport:
    """Build a SoccerPriorCoverageReport for a league (and optionally teams).

    Parameters
    ----------
    league:      Omega league code (e.g. FIFA_WORLD_CUP_2026, EPL).
    home_team:   Provider team name for xG lookup.
    away_team:   Provider team name for xG lookup.
    season:      StatsBomb/Understat season string (e.g. "2025/2026").
    store:       Open TraceStore; opened and closed here if None.
    """
    from omega.core.config.leagues import get_league_config  # lazy import

    config = get_league_config(league.upper())
    profile_id: str | None = config.get("rho_fit_profile")  # type: ignore[assignment]
    competition = config.get("competition", league.upper())

    warnings: list[str] = []
    fallback_usage: list[str] = []

    if not profile_id:
        # League has no rho profile configured (e.g. MLB — wrong command)
        warnings.append(
            f"{league.upper()} has no rho_fit_profile configured; not a bivariate-DC soccer league"
        )
        return SoccerPriorCoverageReport(
            league=league.upper(),
            competition_profile_id=None,
            dc_profile=None,
            confidence_tier=TIER_NONE,
            recommended_output_mode="research_candidate",
            warnings=warnings,
        )

    own_store = store is None
    if own_store:
        store = TraceStore()
    try:
        dc_profile = get_production_dc_profile(store, profile_id)
        dc_candidate = _check_candidate_exists(store, profile_id) if dc_profile is None else True

        home_cov: TeamXgCoverage | None = None
        away_cov: TeamXgCoverage | None = None
        if home_team:
            home_cov = _lookup_team_xg(store, home_team, str(competition), season)
        if away_team:
            away_cov = _lookup_team_xg(store, away_team, str(competition), season)
    finally:
        if own_store:
            store.close()

    # Determine confidence tier
    has_dc = dc_profile is not None
    has_xg = bool((home_cov and home_cov.has_xg) or (away_cov and away_cov.has_xg))

    if has_dc and has_xg:
        tier = TIER_STRONG
    elif has_dc and not has_xg:
        tier = TIER_MODERATE
        if home_team or away_team:
            # Teams were requested but neither matched xG table
            fallback_usage.append("xg_fallback: lambda derived from league avg gpg")
            warnings.append(
                "No xG rows found for requested teams; "
                "soccer backend will use league-average goal-per-game fallback. "
                "Refresh via omega-refresh-statsbomb or omega-refresh-understat."
            )
        else:
            # No teams specified; can't assess xG coverage
            warnings.append(
                "No team names provided; xG coverage not assessed. "
                "Tier elevated to 'moderate' assuming no team xG available."
            )
    elif not has_dc and has_xg:
        tier = TIER_MODERATE
        fallback_usage.append("dc_rho: engine will fail closed (rho_prior missing)")
        warnings.append(
            f"No production Dixon-Coles rho for {profile_id!r}. "
            "Soccer backend requires rho — engine will skip. "
            "Fit and promote via: omega-fit-dixon-coles --profile-id "
            f"{profile_id} && omega-promote-profile --candidate-id <id>"
        )
    else:
        # Neither DC nor xG
        if dc_candidate:
            tier = TIER_WEAK
            fallback_usage.append("dc_rho: candidate row exists but not promoted")
            warnings.append(
                f"Dixon-Coles candidate row for {profile_id!r} exists but is not promoted. "
                "Promote via: omega-promote-profile --candidate-id <id>"
            )
        else:
            tier = TIER_NONE
            warnings.append(
                f"No Dixon-Coles rows at all for {profile_id!r}. "
                "Fit from scratch: omega-refresh-statsbomb --xg && "
                f"omega-fit-dixon-coles --profile-id {profile_id}"
            )

    report = SoccerPriorCoverageReport(
        league=league.upper(),
        competition_profile_id=profile_id,
        dc_profile=dc_profile,
        dc_candidate_exists=dc_candidate,
        home_team=home_cov,
        away_team=away_cov,
        confidence_tier=tier,
        recommended_output_mode=_TIER_OUTPUT_MODE[tier],
        fallback_usage=fallback_usage,
        warnings=warnings,
    )
    return report


def gate_output_mode(report: SoccerPriorCoverageReport, requested_mode: str) -> str:
    """Enforce the coverage gate: return the allowed output mode.

    If the requested mode is ``actionable`` but coverage is ``weak`` or ``none``,
    this returns ``research_candidate``.  A ``moderate`` tier with an ``actionable``
    request is downgraded to ``low_confidence_actionable``.
    Modes already at or below the recommended tier are returned unchanged.
    """
    _rank = {
        "research_candidate": 0,
        "low_confidence_actionable": 1,
        "actionable": 2,
    }
    recommended = report.recommended_output_mode
    req_rank = _rank.get(requested_mode.lower(), 2)
    rec_rank = _rank.get(recommended, 0)
    if req_rank <= rec_rank:
        return requested_mode
    return recommended


def _render_report(report: SoccerPriorCoverageReport) -> str:
    lines = [
        "Soccer Prior Coverage Report",
        "=" * 40,
        f"league               : {report.league}",
        f"profile_id           : {report.competition_profile_id or '(none)'}",
        f"confidence_tier      : {report.confidence_tier}",
        f"recommended_mode     : {report.recommended_output_mode}",
        "",
        "Dixon-Coles (rho):",
    ]
    dc = report.dc_profile
    if dc:
        lines += [
            f"  status             : {dc.status}",
            f"  rho                : {dc.rho:.4f}",
            f"  n_matches          : {dc.n_matches}",
            f"  as_of_date         : {dc.as_of_date}",
            f"  source             : {dc.source or '(not recorded)'}",
        ]
    else:
        lines.append(
            "  status             : NO PRODUCTION ROW"
            + (" (candidate exists)" if report.dc_candidate_exists else "")
        )

    def _team_lines(label: str, cov: TeamXgCoverage | None) -> list[str]:
        if cov is None:
            return [f"{label}: (not requested)"]
        out = [f"{label}: {cov.team}"]
        if cov.has_xg:
            out += [
                f"  xg_for             : {cov.xg_for:.3f}",
                f"  xg_against         : {cov.xg_against:.3f}",
                f"  source             : {cov.source}",
                f"  as_of_date         : {cov.as_of_date}",
                f"  matches            : {cov.matches}",
            ]
        else:
            out.append("  xg_for/against     : NOT FOUND")
        return out

    lines.append("")
    lines.extend(_team_lines("Home team xG", report.home_team))
    lines.append("")
    lines.extend(_team_lines("Away team xG", report.away_team))

    if report.fallback_usage:
        lines += ["", "Fallbacks active:"]
        for f in report.fallback_usage:
            lines.append(f"  - {f}")

    if report.warnings:
        lines += ["", "Warnings:"]
        for w in report.warnings:
            lines.append(f"  ! {w}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--league", required=True, help="Omega league code (e.g. FIFA_WORLD_CUP_2026, EPL)"
    )
    parser.add_argument("--home-team", default=None, help="Home team name for xG lookup")
    parser.add_argument("--away-team", default=None, help="Away team name for xG lookup")
    parser.add_argument(
        "--season", default=None, help="Season string for xG lookup (e.g. 2025/2026)"
    )
    parser.add_argument("--db", default=None, help="SQLite path (default: var/omega_traces.db)")
    parser.add_argument("--format", choices=["summary", "json"], default="summary")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        store = TraceStore(db_path=args.db) if args.db else TraceStore()
        try:
            report = build_coverage_report(
                args.league,
                home_team=args.home_team,
                away_team=args.away_team,
                season=args.season,
                store=store,
            )
        finally:
            store.close()
    except Exception as exc:
        logger.error("Prior coverage check failed: %s", exc)
        return 1

    if args.format == "json":
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(_render_report(report))

    if report.confidence_tier in (TIER_WEAK, TIER_NONE):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
