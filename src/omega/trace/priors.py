"""Typed accessors for the dynamic-prior tables (Phase 7).

V16 added the soccer prior tables:

* ``priors_dixon_coles`` — per-competition Dixon-Coles ``rho`` fits written by
  ``omega-fit-dixon-coles``. Exactly one row per ``profile_id`` may hold
  ``status='production'``; the gatherer injects that row's ``rho`` (plus
  provenance) into ``GameAnalysisRequest.prior_payload``. No production row →
  the soccer backend fails closed (``missing_requirements=["rho_prior"]``).
* ``priors_xg`` — team attack/defense xG aggregates from the StatsBomb /
  Understat / FBref adapters, keyed by source so redundancy disagreement stays
  auditable.

These helpers only touch ``store.conn``; they add no state to ``TraceStore``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:  # pragma: no cover - typing only
    from omega.trace.store import TraceStore

DC_STATUS_CANDIDATE = "candidate"
DC_STATUS_PRODUCTION = "production"
DC_STATUS_ARCHIVED = "archived"


class DixonColesProfile(BaseModel):
    """One fitted Dixon-Coles rho row for a competition profile."""

    profile_id: str
    rho: float
    n_matches: int
    fit_loss: float | None = None
    as_of_date: str
    status: str = DC_STATUS_CANDIDATE
    source: str | None = None


class XgPrior(BaseModel):
    """Season-level team xG aggregate from one source."""

    team: str
    competition: str
    season: str
    xg_for: float
    xg_against: float
    matches: int
    source: str
    as_of_date: str


# ---------------------------------------------------------------------------
# priors_dixon_coles
# ---------------------------------------------------------------------------


def upsert_dixon_coles_profile(store: "TraceStore", profile: DixonColesProfile) -> None:
    """Insert or replace one (profile_id, as_of_date) fit row."""
    store.conn.execute(
        """INSERT INTO priors_dixon_coles
               (profile_id, rho, n_matches, fit_loss, as_of_date, status, source)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT (profile_id, as_of_date) DO UPDATE SET
               rho = excluded.rho,
               n_matches = excluded.n_matches,
               fit_loss = excluded.fit_loss,
               status = excluded.status,
               source = excluded.source""",
        (
            profile.profile_id,
            profile.rho,
            profile.n_matches,
            profile.fit_loss,
            profile.as_of_date,
            profile.status,
            profile.source,
        ),
    )
    store.conn.commit()


def get_production_dc_profile(
    store: "TraceStore", profile_id: str
) -> DixonColesProfile | None:
    """Return the production rho fit for *profile_id*, or None (fail closed)."""
    row = store.conn.execute(
        """SELECT profile_id, rho, n_matches, fit_loss, as_of_date, status, source
           FROM priors_dixon_coles
           WHERE profile_id = ? AND status = ?
           ORDER BY as_of_date DESC LIMIT 1""",
        (profile_id, DC_STATUS_PRODUCTION),
    ).fetchone()
    if row is None:
        return None
    return DixonColesProfile(
        profile_id=row[0],
        rho=row[1],
        n_matches=row[2],
        fit_loss=row[3],
        as_of_date=row[4],
        status=row[5],
        source=row[6],
    )


def promote_dixon_coles_profile(
    store: "TraceStore", profile_id: str, as_of_date: str
) -> DixonColesProfile:
    """Promote one (profile_id, as_of_date) fit to production.

    Archives any incumbent production row for the same profile first, so at
    most one production row exists per profile. Raises ValueError if the
    requested fit row does not exist — promotion never creates data.
    """
    target = store.conn.execute(
        "SELECT 1 FROM priors_dixon_coles WHERE profile_id = ? AND as_of_date = ?",
        (profile_id, as_of_date),
    ).fetchone()
    if target is None:
        raise ValueError(
            f"no Dixon-Coles fit for profile {profile_id!r} as_of {as_of_date!r}; "
            "run omega-fit-dixon-coles first"
        )
    store.conn.execute(
        "UPDATE priors_dixon_coles SET status = ? WHERE profile_id = ? AND status = ?",
        (DC_STATUS_ARCHIVED, profile_id, DC_STATUS_PRODUCTION),
    )
    store.conn.execute(
        "UPDATE priors_dixon_coles SET status = ? WHERE profile_id = ? AND as_of_date = ?",
        (DC_STATUS_PRODUCTION, profile_id, as_of_date),
    )
    store.conn.commit()
    promoted = get_production_dc_profile(store, profile_id)
    assert promoted is not None  # row existence checked above
    return promoted


# ---------------------------------------------------------------------------
# priors_xg
# ---------------------------------------------------------------------------


def upsert_xg_prior(store: "TraceStore", prior: XgPrior) -> None:
    """Insert or refresh one (team, competition, season, source) xG row."""
    store.conn.execute(
        """INSERT INTO priors_xg
               (team, competition, season, xg_for, xg_against, matches, source,
                as_of_date, last_updated)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
           ON CONFLICT (team, competition, season, source) DO UPDATE SET
               xg_for = excluded.xg_for,
               xg_against = excluded.xg_against,
               matches = excluded.matches,
               as_of_date = excluded.as_of_date,
               last_updated = datetime('now')""",
        (
            prior.team,
            prior.competition,
            prior.season,
            prior.xg_for,
            prior.xg_against,
            prior.matches,
            prior.source,
            prior.as_of_date,
        ),
    )
    store.conn.commit()


def get_xg_prior(
    store: "TraceStore",
    team: str,
    competition: str,
    season: str,
    source: str | None = None,
) -> XgPrior | None:
    """Return the xG aggregate for a team/competition/season.

    With ``source=None`` the most recently updated row across sources wins.
    """
    query = """SELECT team, competition, season, xg_for, xg_against, matches,
                      source, as_of_date
               FROM priors_xg
               WHERE team = ? AND competition = ? AND season = ?"""
    params: list = [team, competition, season]
    if source is not None:
        query += " AND source = ?"
        params.append(source)
    query += " ORDER BY last_updated DESC LIMIT 1"
    row = store.conn.execute(query, params).fetchone()
    if row is None:
        return None
    return XgPrior(
        team=row[0],
        competition=row[1],
        season=row[2],
        xg_for=row[3],
        xg_against=row[4],
        matches=row[5],
        source=row[6],
        as_of_date=row[7],
    )


# ---------------------------------------------------------------------------
# Gatherer injection: league config -> production prior -> prior_payload
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_game_prior_payload(
    league: str,
    existing_payload: dict[str, Any] | None,
    store: "TraceStore",
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Merge the league's production dynamic priors into a game prior_payload.

    Returns ``(payload, data_provenance_event_or_None)``. Behavior:

    * league config has no ``rho_fit_profile`` -> payload unchanged, no event;
    * caller already supplied ``rho`` (recorded/replayed request) -> unchanged,
      no event — replay must not depend on live table state;
    * no production profile -> unchanged + ``warn`` event; the backend then
      fails closed (``missing_requirements=["rho_prior"]``) rather than guess;
    * production profile found -> merged ``rho``/``rho_profile_id``/
      ``rho_as_of_date`` + ``ok`` event carrying the provenance.
    """
    from omega.core.config.leagues import get_league_config

    profile_id = get_league_config(league.upper()).get("rho_fit_profile")
    if not profile_id:
        return existing_payload, None
    if existing_payload and existing_payload.get("rho") is not None:
        return existing_payload, None

    prod = get_production_dc_profile(store, str(profile_id))
    if prod is None:
        return existing_payload, {
            "ts": _utc_now_iso(),
            "event_type": "data_provenance",
            "step": "dixon_coles_prior:inject",
            "status": "warn",
            "notes": (
                f"no production Dixon-Coles profile {profile_id!r} for league "
                f"{league.upper()}; engine will fail closed (rho_prior missing). "
                "Fit and promote via omega-fit-dixon-coles."
            ),
        }

    merged = dict(existing_payload or {})
    merged.update(
        rho=prod.rho,
        rho_profile_id=prod.profile_id,
        rho_as_of_date=prod.as_of_date,
    )
    return merged, {
        "ts": _utc_now_iso(),
        "event_type": "data_provenance",
        "step": "dixon_coles_prior:inject",
        "status": "ok",
        "notes": f"injected Dixon-Coles rho for {league.upper()} from {prod.profile_id}",
        "outputs": {
            "rho": prod.rho,
            "rho_profile_id": prod.profile_id,
            "rho_as_of_date": prod.as_of_date,
        },
    }


def inject_game_priors(
    payload: dict[str, Any], store: "TraceStore | None" = None
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Config-gated prior injection for a raw game-request dict.

    Opens the default TraceStore only when the league actually carries a
    ``rho_fit_profile`` (so NBA/MLB/... requests never pay the cost). Returns
    the (possibly updated) payload plus an optional ``data_provenance`` event
    for the session sidecar. Injection lives at the gatherer layer — outside
    ``service.analyze`` — so replaying a recorded request (with its priors
    embedded) is deterministic regardless of live table state.
    """
    league = str(payload.get("league") or "")
    if not league:
        return payload, None

    from omega.core.config.leagues import get_league_config

    if not get_league_config(league.upper()).get("rho_fit_profile"):
        return payload, None

    own_store = store is None
    if own_store:
        from omega.trace.store import TraceStore

        store = TraceStore()
    try:
        merged, event = build_game_prior_payload(
            league, payload.get("prior_payload"), store
        )
    finally:
        if own_store:
            store.close()

    out = dict(payload)
    if merged is not None:
        out["prior_payload"] = merged
    return out, event
