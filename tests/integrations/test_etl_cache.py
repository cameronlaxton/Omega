"""ETL standard 1 — local caching layer (zero-refetch within TTL).

References:
  omega/integrations/_etl.py::cached_fetch
  docs/phase7/MULTI_SPORT_EXPANSION.md  (Part 5B standard 1; verification test 13)
"""

from __future__ import annotations

import time

import pandas as pd
import pytest

from omega.integrations._etl import cached_fetch
from omega.integrations._guards import OmegaReplayModeError


def test_second_pull_within_ttl_serves_from_cache(tmp_path):
    calls = {"n": 0}

    @cached_fetch("unit_source", ttl_seconds=3600, fmt="parquet", cache_root=tmp_path)
    def fetch():
        calls["n"] += 1
        return pd.DataFrame({"player": ["a", "b"], "spw": [0.6, 0.7]})

    first = fetch(cache_key="atp_2025")
    second = fetch(cache_key="atp_2025")

    assert calls["n"] == 1, "second pull within TTL must not invoke the fetch fn"
    pd.testing.assert_frame_equal(first, second)
    assert (tmp_path / "unit_source" / "atp_2025.parquet").exists()


def test_expired_ttl_refetches(tmp_path):
    calls = {"n": 0}

    @cached_fetch("unit_source", ttl_seconds=0.05, fmt="parquet", cache_root=tmp_path)
    def fetch():
        calls["n"] += 1
        return pd.DataFrame({"x": [calls["n"]]})

    fetch(cache_key="k")
    time.sleep(0.1)
    fetch(cache_key="k")

    assert calls["n"] == 2, "an expired cache must trigger a refetch"


def test_distinct_cache_keys_are_isolated(tmp_path):
    calls = {"n": 0}

    @cached_fetch("unit_source", ttl_seconds=3600, fmt="json", cache_root=tmp_path)
    def fetch():
        calls["n"] += 1
        return {"call": calls["n"]}

    a = fetch(cache_key="key_a")
    b = fetch(cache_key="key_b")

    assert calls["n"] == 2
    assert a != b


def test_cache_miss_blocked_in_replay_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("OMEGA_REPLAY_MODE", "1")

    @cached_fetch("unit_source", ttl_seconds=3600, fmt="json", cache_root=tmp_path)
    def fetch():
        raise AssertionError("fetch should not run; guard must fire first")

    with pytest.raises(OmegaReplayModeError, match="OMEGA_REPLAY_MODE=1"):
        fetch(cache_key="cold")


def test_cache_hit_allowed_in_replay_mode(tmp_path, monkeypatch):
    @cached_fetch("unit_source", ttl_seconds=3600, fmt="json", cache_root=tmp_path)
    def fetch():
        return {"frozen": True}

    # Warm the cache with replay mode OFF.
    monkeypatch.delenv("OMEGA_REPLAY_MODE", raising=False)
    fetch(cache_key="frozen_key")

    # Replay mode ON: a cache hit must still serve (frozen snapshot is allowed).
    monkeypatch.setenv("OMEGA_REPLAY_MODE", "1")
    assert fetch(cache_key="frozen_key") == {"frozen": True}
