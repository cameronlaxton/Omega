"""Schema V16: priors_dixon_coles + priors_xg tables and typed accessors."""

from __future__ import annotations

import tempfile

import pytest

from omega.trace.priors import (
    DC_STATUS_ARCHIVED,
    DC_STATUS_PRODUCTION,
    DixonColesProfile,
    XgPrior,
    get_production_dc_profile,
    get_xg_prior,
    promote_dixon_coles_profile,
    upsert_dixon_coles_profile,
    upsert_xg_prior,
)
from omega.trace.store import TraceStore


def _tmp_db() -> str:
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    f.close()
    return f.name


def _store() -> TraceStore:
    return TraceStore(db_path=_tmp_db())


def test_fresh_and_reopened_db_have_v16_tables():
    path = _tmp_db()
    store = TraceStore(db_path=path)
    try:
        tables = {
            row[0]
            for row in store.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert {
            "priors_dixon_coles",
            "priors_xg",
            "priors_tennis",
            "priors_tennis_pressure",
        } <= tables
        assert store.schema_version() == 17
    finally:
        store.close()
    reopened = TraceStore(db_path=path)
    try:
        assert reopened.schema_version() == 17
    finally:
        reopened.close()


def test_reopening_up_to_date_db_skips_schema_replay(monkeypatch):
    """An up-to-date DB must not replay the forward-additive DDL on reopen.

    Per-request TraceStore opens (prior injection, batch loops) previously
    re-ran all ~17 idempotent executescripts plus the bet_records
    consolidation probe on every open. The probe is the observable: it runs
    during the initial build, and must NOT run when reopening a DB already
    stamped at CURRENT_VERSION.
    """
    path = _tmp_db()
    store = TraceStore(db_path=path)
    store.close()

    calls = {"n": 0}
    original = TraceStore._consolidate_legacy_bet_records

    def _spy(self):
        calls["n"] += 1
        return original(self)

    monkeypatch.setattr(TraceStore, "_consolidate_legacy_bet_records", _spy)
    reopened = TraceStore(db_path=path)
    try:
        assert calls["n"] == 0  # fast path: no DDL replay on an up-to-date DB
        assert reopened.schema_version() == 17
    finally:
        reopened.close()


def test_no_production_profile_returns_none():
    store = _store()
    try:
        assert get_production_dc_profile(store, "fifa_intl_v1") is None
    finally:
        store.close()


def test_candidate_is_not_production():
    store = _store()
    try:
        upsert_dixon_coles_profile(
            store,
            DixonColesProfile(
                profile_id="fifa_intl_v1",
                rho=-0.11,
                n_matches=900,
                fit_loss=1.234,
                as_of_date="2026-06-10",
                source="statsbomb_open_data",
            ),
        )
        assert get_production_dc_profile(store, "fifa_intl_v1") is None
    finally:
        store.close()


def test_promote_returns_production_and_archives_incumbent():
    store = _store()
    try:
        for as_of, rho in (("2026-03-01", -0.09), ("2026-06-10", -0.11)):
            upsert_dixon_coles_profile(
                store,
                DixonColesProfile(
                    profile_id="fifa_intl_v1",
                    rho=rho,
                    n_matches=900,
                    as_of_date=as_of,
                ),
            )
        promote_dixon_coles_profile(store, "fifa_intl_v1", "2026-03-01")
        promote_dixon_coles_profile(store, "fifa_intl_v1", "2026-06-10")

        prod = get_production_dc_profile(store, "fifa_intl_v1")
        assert prod is not None
        assert prod.as_of_date == "2026-06-10"
        assert prod.rho == -0.11
        assert prod.status == DC_STATUS_PRODUCTION

        statuses = dict(
            store.conn.execute(
                "SELECT as_of_date, status FROM priors_dixon_coles WHERE profile_id='fifa_intl_v1'"
            ).fetchall()
        )
        assert statuses["2026-03-01"] == DC_STATUS_ARCHIVED
    finally:
        store.close()


def test_promote_missing_fit_raises():
    store = _store()
    try:
        with pytest.raises(ValueError, match="omega-fit-dixon-coles"):
            promote_dixon_coles_profile(store, "epl_v1", "2026-06-10")
    finally:
        store.close()


def test_promotion_is_profile_scoped():
    """Promoting one profile never touches another's production row."""
    store = _store()
    try:
        for pid in ("fifa_intl_v1", "epl_v1"):
            upsert_dixon_coles_profile(
                store,
                DixonColesProfile(
                    profile_id=pid, rho=-0.1, n_matches=500, as_of_date="2026-06-01"
                ),
            )
            promote_dixon_coles_profile(store, pid, "2026-06-01")
        assert get_production_dc_profile(store, "fifa_intl_v1") is not None
        assert get_production_dc_profile(store, "epl_v1") is not None
    finally:
        store.close()


def test_xg_upsert_and_refresh():
    store = _store()
    try:
        prior = XgPrior(
            team="France",
            competition="fifa_intl",
            season="2026",
            xg_for=2.1,
            xg_against=0.8,
            matches=14,
            source="statsbomb",
            as_of_date="2026-06-10",
        )
        upsert_xg_prior(store, prior)
        upsert_xg_prior(store, prior.model_copy(update={"xg_for": 2.3, "matches": 15}))

        got = get_xg_prior(store, "France", "fifa_intl", "2026")
        assert got is not None
        assert got.xg_for == 2.3
        assert got.matches == 15
        n = store.conn.execute("SELECT COUNT(*) FROM priors_xg").fetchone()[0]
        assert n == 1  # upsert, not duplicate
    finally:
        store.close()


def test_tennis_prior_upsert_and_latest_lookup():
    from omega.trace.priors import TennisPrior, get_tennis_prior, upsert_tennis_prior

    store = _store()
    try:
        for as_of, spw in (("2026-05-01", 0.66), ("2026-06-10", 0.67)):
            upsert_tennis_prior(
                store,
                TennisPrior(
                    player="Carlos Alcaraz",
                    tour="atp",
                    surface="GRASS",
                    spw_pct=spw,
                    rpw_pct=0.41,
                    n_matches=24,
                    as_of_date=as_of,
                ),
            )
        got = get_tennis_prior(store, "Carlos Alcaraz", "ATP", "grass")
        assert got is not None
        assert got.spw_pct == 0.67  # latest as_of wins
        assert got.tour == "ATP" and got.surface == "grass"  # normalized
        assert get_tennis_prior(store, "Carlos Alcaraz", "ATP", "clay") is None
    finally:
        store.close()


def test_pressure_coefficients_lookup_and_fallback_source():
    from omega.trace.priors import (
        PRESSURE_SOURCE_GROUP,
        PRESSURE_SOURCE_PLAYER,
        TENNIS_PRESSURE_STATES,
        TennisPressureDelta,
        get_pressure_coefficients,
        upsert_pressure_deltas,
    )

    store = _store()
    try:
        upsert_pressure_deltas(
            store,
            [
                TennisPressureDelta(
                    player="Carlos Alcaraz",
                    tour="ATP",
                    surface="grass",
                    state=state,
                    delta=-0.01 * (i + 1),
                    n_points=800,
                    source=PRESSURE_SOURCE_PLAYER,
                    as_of_date="2026-06-10",
                )
                for i, state in enumerate(TENNIS_PRESSURE_STATES)
            ],
        )
        coeffs, source = get_pressure_coefficients(store, "Carlos Alcaraz", "ATP", "grass")
        assert source == PRESSURE_SOURCE_PLAYER
        assert set(coeffs) == set(TENNIS_PRESSURE_STATES)
        assert coeffs["break_point_against"] == pytest.approx(-0.01)

        # Unknown player -> no rows -> flat IID rollback state.
        coeffs, source = get_pressure_coefficients(store, "Nobody", "ATP", "grass")
        assert coeffs == {} and source is None

        # Group-fallback rows carry their source through.
        upsert_pressure_deltas(
            store,
            [
                TennisPressureDelta(
                    player="__group__",
                    tour="ATP",
                    surface="grass",
                    state="tiebreak",
                    delta=-0.008,
                    n_points=120,
                    source=PRESSURE_SOURCE_GROUP,
                    as_of_date="2026-06-10",
                )
            ],
        )
        _, source = get_pressure_coefficients(store, "Qualifier X", "ATP", "grass")
        assert source == PRESSURE_SOURCE_GROUP
    finally:
        store.close()


def test_xg_source_filter():
    store = _store()
    try:
        for source, xg in (("statsbomb", 1.9), ("understat", 2.0)):
            upsert_xg_prior(
                store,
                XgPrior(
                    team="France",
                    competition="fifa_intl",
                    season="2026",
                    xg_for=xg,
                    xg_against=0.9,
                    matches=14,
                    source=source,
                    as_of_date="2026-06-10",
                ),
            )
        got = get_xg_prior(store, "France", "fifa_intl", "2026", source="statsbomb")
        assert got is not None and got.xg_for == 1.9
        assert get_xg_prior(store, "France", "fifa_intl", "2026", source="fbref") is None
    finally:
        store.close()
