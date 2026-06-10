"""Schema V16: priors_dixon_coles + priors_xg tables and typed accessors."""

from __future__ import annotations

import tempfile

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

import pytest


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
        assert {"priors_dixon_coles", "priors_xg"} <= tables
        assert store.schema_version() == 16
    finally:
        store.close()
    reopened = TraceStore(db_path=path)
    try:
        assert reopened.schema_version() == 16
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
