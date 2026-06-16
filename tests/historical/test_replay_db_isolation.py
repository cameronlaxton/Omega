"""Replay must fail closed when pointed at the production trace DB.

Synthetic historical-replay traces must never land in var/omega_traces.db, where
they would silently pollute the live calibration pool. Two layers enforce this:
the omega-replay-history CLI (clean exit 2) and a ReplayConfig field validator
(construction-time ValueError) for any non-CLI caller.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omega.historical.contracts import ReplayConfig
from omega.ops import replay_history
from omega.paths import default_trace_db_path, is_production_trace_db


def test_cli_refuses_production_db_with_exit_2():
    rc = replay_history.main(
        [
            "--league",
            "NFL",
            "--manifest-id",
            "does-not-matter",
            "--db",
            str(default_trace_db_path()),
            "--mode",
            "calibration",
        ]
    )
    assert rc == 2


def test_replay_config_validator_rejects_production_db():
    with pytest.raises(ValidationError):
        ReplayConfig(
            dataset_manifest_id="m",
            backtest_db_path=str(default_trace_db_path()),
        )


def test_replay_config_accepts_isolated_db(tmp_path):
    cfg = ReplayConfig(
        dataset_manifest_id="m",
        backtest_db_path=str(tmp_path / "replay.db"),
    )
    assert cfg.backtest_db_path.endswith("replay.db")


def test_is_production_trace_db_helper(tmp_path):
    assert is_production_trace_db(None) is True
    assert is_production_trace_db(str(default_trace_db_path())) is True
    assert is_production_trace_db(str(tmp_path / "replay.db")) is False
