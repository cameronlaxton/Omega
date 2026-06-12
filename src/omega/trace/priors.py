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

from pydantic import BaseModel, field_validator

if TYPE_CHECKING:  # pragma: no cover - typing only
    from omega.trace.store import TraceStore

DC_STATUS_CANDIDATE = "candidate"
DC_STATUS_PRODUCTION = "production"
DC_STATUS_ARCHIVED = "archived"

# The six tennis pressure states whose SPW% deltas are fit by
# omega-fit-tennis-pressure-coefficients (design decision 7). Missing states
# default to 0.0 in the backend; missing *players* fall back to tour+surface
# group means with source='group_fallback' — never silent zeros.
TENNIS_PRESSURE_STATES = (
    "break_point_against",
    "set_point_serving",
    "match_point_serving",
    "tiebreak",
    "serving_for_set",
    "serving_for_match",
)
PRESSURE_SOURCE_PLAYER = "player"
PRESSURE_SOURCE_GROUP = "group_fallback"
# Reserved player key for tour+surface group-mean pressure rows. A player with
# no charted rows at all resolves to these at lookup time (source becomes
# group_fallback) — flat 0.0 deltas are never applied silently.
PRESSURE_GROUP_PLAYER_KEY = "__group__"


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


def upsert_dixon_coles_profile(store: TraceStore, profile: DixonColesProfile) -> None:
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
    store: TraceStore, profile_id: str
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
    store: TraceStore, profile_id: str, as_of_date: str
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


def upsert_xg_prior(store: TraceStore, prior: XgPrior) -> None:
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
    store: TraceStore,
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
# priors_tennis + priors_tennis_pressure
# ---------------------------------------------------------------------------


class TennisPrior(BaseModel):
    """Surface-segmented rolling serve/return point-win rates for one player."""

    player: str
    tour: str  # ATP | WTA
    surface: str  # hard | clay | grass
    spw_pct: float
    rpw_pct: float
    n_matches: int
    as_of_date: str


class TennisPressureDelta(BaseModel):
    """One pressure-state additive SPW% delta for one player (or group mean)."""

    player: str
    tour: str
    surface: str
    state: str
    delta: float
    n_points: int
    source: str  # player | group_fallback
    as_of_date: str

    @field_validator("state")
    @classmethod
    def _validate_state(cls, value: str) -> str:
        if value not in TENNIS_PRESSURE_STATES:
            raise ValueError(
                f"state must be one of {', '.join(TENNIS_PRESSURE_STATES)}"
            )
        return value


def upsert_tennis_prior(store: TraceStore, prior: TennisPrior) -> None:
    """Insert or refresh one (player, tour, surface, as_of_date) rate row."""
    store.conn.execute(
        """INSERT INTO priors_tennis
               (player, tour, surface, spw_pct, rpw_pct, n_matches, as_of_date,
                last_updated)
           VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
           ON CONFLICT (player, tour, surface, as_of_date) DO UPDATE SET
               spw_pct = excluded.spw_pct,
               rpw_pct = excluded.rpw_pct,
               n_matches = excluded.n_matches,
               last_updated = datetime('now')""",
        (
            prior.player,
            prior.tour.upper(),
            prior.surface.lower(),
            prior.spw_pct,
            prior.rpw_pct,
            prior.n_matches,
            prior.as_of_date,
        ),
    )
    store.conn.commit()


def get_tennis_prior(
    store: TraceStore, player: str, tour: str, surface: str
) -> TennisPrior | None:
    """Return the most recent rate row for (player, tour, surface), or None."""
    row = store.conn.execute(
        """SELECT player, tour, surface, spw_pct, rpw_pct, n_matches, as_of_date
           FROM priors_tennis
           WHERE player = ? AND tour = ? AND surface = ?
           ORDER BY as_of_date DESC LIMIT 1""",
        (player, tour.upper(), surface.lower()),
    ).fetchone()
    if row is None:
        return None
    return TennisPrior(
        player=row[0],
        tour=row[1],
        surface=row[2],
        spw_pct=row[3],
        rpw_pct=row[4],
        n_matches=row[5],
        as_of_date=row[6],
    )


def upsert_pressure_deltas(store: TraceStore, deltas: list[TennisPressureDelta]) -> None:
    """Insert or refresh a batch of pressure-state delta rows."""
    for delta in deltas:
        store.conn.execute(
            """INSERT INTO priors_tennis_pressure
                   (player, tour, surface, state, delta, n_points, source,
                    as_of_date, last_updated)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
               ON CONFLICT (player, tour, surface, state, as_of_date) DO UPDATE SET
                   delta = excluded.delta,
                   n_points = excluded.n_points,
                   source = excluded.source,
                   last_updated = datetime('now')""",
            (
                delta.player,
                delta.tour.upper(),
                delta.surface.lower(),
                delta.state,
                delta.delta,
                delta.n_points,
                delta.source,
                delta.as_of_date,
            ),
        )
    store.conn.commit()


def _pressure_rows(store: TraceStore, player: str, tour: str, surface: str):
    return store.conn.execute(
        """SELECT state, delta, source, as_of_date
           FROM priors_tennis_pressure
           WHERE player = ? AND tour = ? AND surface = ?
             AND as_of_date = (
                 SELECT MAX(as_of_date) FROM priors_tennis_pressure
                 WHERE player = ? AND tour = ? AND surface = ?
             )""",
        (player, tour.upper(), surface.lower()) * 2,
    ).fetchall()


def get_pressure_coefficients(
    store: TraceStore, player: str, tour: str, surface: str
) -> tuple[dict[str, float], str | None]:
    """Return ``({state: delta}, source)`` for a player's latest pressure fit.

    A player with no rows of their own falls back to the tour+surface group
    rows (``PRESSURE_GROUP_PLAYER_KEY``) with ``source="group_fallback"`` —
    never silent zeros. ``(None`` source only when no fit exists at all, in
    which case the backend runs flat IID, the documented rollback state.)
    """
    rows = _pressure_rows(store, player, tour, surface)
    if rows:
        return {row[0]: row[1] for row in rows}, rows[0][2]
    group_rows = _pressure_rows(store, PRESSURE_GROUP_PLAYER_KEY, tour, surface)
    if group_rows:
        return {row[0]: row[1] for row in group_rows}, PRESSURE_SOURCE_GROUP
    return {}, None


# ---------------------------------------------------------------------------
# Gatherer injection: league config -> production prior -> prior_payload
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_game_prior_payload(
    league: str,
    existing_payload: dict[str, Any] | None,
    store: TraceStore,
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


def build_tennis_prior_payload(
    payload: dict[str, Any], store: TraceStore
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Tennis gatherer branch: join rates + pressure coefficients per player.

    * fills missing ``serve_win_pct``/``return_win_pct`` in home/away context
      from ``priors_tennis`` (caller-supplied values always win);
    * joins each player's latest pressure deltas into
      ``prior_payload["pressure_coefficients"]`` (``{"home": .., "away": ..}``)
      with per-side ``pressure_coefficient_source``;
    * requires a surface (``game_context.surface`` or ``prior_payload.surface``)
      — without one nothing is joined and a ``warn`` event is emitted (the
      backend then runs flat IID or skips on missing rates, both honest).

    GRAND_SLAM requests try ATP rows first, then WTA.
    """
    league = str(payload.get("league") or "").upper()
    tours = [league] if league in ("ATP", "WTA") else ["ATP", "WTA"]
    prior = dict(payload.get("prior_payload") or {})
    game_ctx = payload.get("game_context") or {}
    surface = (prior.get("surface") or game_ctx.get("surface") or "").lower()

    out = dict(payload)
    if not surface:
        return out, {
            "ts": _utc_now_iso(),
            "event_type": "data_provenance",
            "step": "tennis_prior:inject",
            "status": "warn",
            "notes": (
                f"no surface on {league} request (game_context.surface); tennis "
                "rate/pressure priors not joined — backend runs flat IID or "
                "fails closed on missing serve stats"
            ),
        }

    if prior.get("pressure_coefficients") is not None:
        return out, None  # recorded/replayed request: never re-read live tables

    sides = {"home": str(payload.get("home_team") or ""), "away": str(payload.get("away_team") or "")}
    coefficients: dict[str, dict[str, float]] = {}
    sources: dict[str, str] = {}
    filled_rates: list[str] = []

    for side, player in sides.items():
        if not player:
            continue
        for tour in tours:
            rate = get_tennis_prior(store, player, tour, surface)
            if rate is not None:
                ctx_key = f"{side}_context"
                ctx = dict(out.get(ctx_key) or {})
                if ctx.get("serve_win_pct") is None:
                    ctx["serve_win_pct"] = rate.spw_pct
                    filled_rates.append(f"{side}.serve_win_pct")
                if ctx.get("return_win_pct") is None:
                    ctx["return_win_pct"] = rate.rpw_pct
                    filled_rates.append(f"{side}.return_win_pct")
                out[ctx_key] = ctx
                break
        for tour in tours:
            coeffs, source = get_pressure_coefficients(store, player, tour, surface)
            if source is not None:
                coefficients[side] = coeffs
                sources[side] = source
                break

    if not coefficients and not filled_rates:
        return out, {
            "ts": _utc_now_iso(),
            "event_type": "data_provenance",
            "step": "tennis_prior:inject",
            "status": "warn",
            "notes": (
                f"no priors_tennis/pressure rows for {sides['home']!r} or "
                f"{sides['away']!r} on {surface}; refresh via omega-refresh-sackmann "
                "and omega-fit-tennis-pressure-coefficients"
            ),
        }

    if coefficients:
        prior["pressure_coefficients"] = coefficients
        prior["pressure_coefficient_source"] = sources
    prior.setdefault("surface", surface)
    out["prior_payload"] = prior
    return out, {
        "ts": _utc_now_iso(),
        "event_type": "data_provenance",
        "step": "tennis_prior:inject",
        "status": "ok",
        "notes": f"joined tennis priors for {league} on {surface}",
        "outputs": {
            "pressure_coefficient_source": sources,
            "filled_rates": filled_rates,
            "surface": surface,
        },
    }


def inject_game_priors(
    payload: dict[str, Any], store: TraceStore | None = None
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Config-gated prior injection for a raw game-request dict.

    Opens the default TraceStore only when the league actually carries dynamic
    priors (soccer ``rho_fit_profile`` config, or a tennis league), so
    NBA/MLB/... requests never pay the cost. Returns the (possibly updated)
    payload plus an optional ``data_provenance`` event for the session
    sidecar. Injection lives at the gatherer layer — outside
    ``service.analyze`` — so replaying a recorded request (with its priors
    embedded) is deterministic regardless of live table state.
    """
    league = str(payload.get("league") or "")
    if not league:
        return payload, None

    from omega.core.config.leagues import get_league_config

    config = get_league_config(league.upper())
    is_soccer_dynamic = bool(config.get("rho_fit_profile"))
    is_tennis = config.get("sport") == "tennis"
    if not (is_soccer_dynamic or is_tennis):
        return payload, None

    own_store = store is None
    if own_store:
        from omega.trace.store import TraceStore

        store = TraceStore()
    try:
        if is_tennis:
            return build_tennis_prior_payload(payload, store)
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
