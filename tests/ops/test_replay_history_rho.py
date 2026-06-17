"""Frozen Dixon-Coles rho injection for historical replay (audit remediation C3).

The soccer bivariate-DC backend requires `rho` in `prior_payload`. The replay path
resolves it ONCE up front and freezes it onto `ReplayConfig.prior_payload` so the
run is deterministic and never depends on live priors-table mutation mid-run.
"""

import sqlite3
import types

import pytest

from omega.historical.contracts import ReplayConfig
from omega.ops.replay_history import _resolve_frozen_prior_payload


def _args(**kw):
    base = {"rho": None, "rho_profile": None, "priors_db": None}
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_explicit_rho_is_frozen():
    payload = _resolve_frozen_prior_payload(_args(rho=-0.0127))
    assert payload == {"rho": -0.0127, "rho_profile_id": "explicit", "rho_as_of_date": None}


def test_no_rho_returns_none():
    assert _resolve_frozen_prior_payload(_args()) is None


def test_missing_profile_fails_closed(tmp_path):
    # A priors DB with the schema but no matching production DC profile must fail
    # closed (SystemExit) rather than silently replay without rho.
    db = tmp_path / "empty_priors.db"
    sqlite3.connect(str(db)).close()  # exists but empty
    with pytest.raises(SystemExit):
        _resolve_frozen_prior_payload(_args(rho_profile="fifa_intl_v1", priors_db=str(db)))


def test_config_hash_depends_on_prior_payload():
    common = {"dataset_manifest_id": "m1", "backtest_db_path": "var/historical/replay_t.db"}
    no_prior = ReplayConfig(**common)
    with_prior = ReplayConfig(**common, prior_payload={"rho": -0.0127})
    assert no_prior.config_hash() != with_prior.config_hash()
    # The frozen payload round-trips on the config.
    assert with_prior.prior_payload == {"rho": -0.0127}
