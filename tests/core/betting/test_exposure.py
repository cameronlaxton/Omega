"""Tests for exposure limits/ledger/policy + correlation_group_key (Stage C PR4)."""

from __future__ import annotations

import pytest

from omega.core.betting.exposure import (
    ExposureAction,
    ExposureLedger,
    ExposureLimits,
    ExposurePolicy,
)
from omega.core.betting.parlay import correlation_group_key


# --- correlation_group_key -----------------------------------------------------
def test_correlation_group_key_same_player():
    a = correlation_group_key(player="LeBron James", stat_key="pts")
    b = correlation_group_key(player="lebron  james", stat_key="reb")  # diff stat, same player
    assert a == "corr:player:lebron james"
    assert a == b  # all of a player's props share one group


def test_correlation_group_key_team_stat_when_no_player():
    key = correlation_group_key(stat_key="PTS", team="Boston Celtics")
    assert key == "corr:team:boston celtics:pts"


def test_correlation_group_key_player_beats_team():
    # A player handle takes precedence over team+stat.
    assert correlation_group_key(player="X", stat_key="pts", team="Y").startswith("corr:player:")


def test_correlation_group_key_none_without_handle():
    assert correlation_group_key() is None
    assert correlation_group_key(stat_key="pts") is None  # stat alone, no team/player


# --- ExposureLimits.cap_for_key ------------------------------------------------
def test_cap_for_key_prefix_mapping():
    lim = ExposureLimits()
    assert lim.cap_for_key("sport:BASKETBALL") == lim.max_per_sport_pct
    assert lim.cap_for_key("league:NBA") == lim.max_per_league_pct
    assert lim.cap_for_key("game:NBA:A @ B") == lim.max_per_game_pct
    assert lim.cap_for_key("team:NBA:Boston") == lim.max_per_team_pct
    assert lim.cap_for_key("player:LeBron") == lim.max_per_player_pct
    assert lim.cap_for_key("corr:player:lebron") == lim.max_per_correlated_group_pct
    # selection + unknown prefixes are uncapped per-key
    assert lim.cap_for_key("selection:prop:x") is None
    assert lim.cap_for_key("mystery:z") is None


# --- ExposureLedger ------------------------------------------------------------
_KEYS = ("league:NBA", "game:NBA:A @ B", "selection:prop:x")


def test_headroom_binds_on_tightest_cap():
    led = ExposureLedger(limits=ExposureLimits())
    # bankroll 1000: total 250, league 150, game 50, selection uncapped -> game binds
    assert led.headroom(_KEYS, 1000.0) == pytest.approx(50.0)


def test_would_breach_strict_at_cap():
    led = ExposureLedger(limits=ExposureLimits())
    breached, key = led.would_breach(_KEYS, 50.0, 1000.0)  # exactly at game cap -> ok
    assert breached is False and key is None
    breached, key = led.would_breach(_KEYS, 50.01, 1000.0)  # over game cap
    assert breached is True and key == "game:NBA:A @ B"


def test_add_accumulates_per_key_and_total():
    led = ExposureLedger(limits=ExposureLimits())
    led.add(_KEYS, 40.0)
    assert led.total_open == pytest.approx(40.0)
    assert led.exposure_for("game:NBA:A @ B") == pytest.approx(40.0)
    assert led.headroom(_KEYS, 1000.0) == pytest.approx(10.0)  # game 50 - 40


def test_total_open_cap_can_bind():
    led = ExposureLedger.seeded(
        ExposureLimits(), exposure_by_entity={"league:NBA": 100.0}, total_open=245.0
    )
    # total cap 250 leaves only 5 of headroom even though league has 50 left
    assert led.headroom(("league:NBA",), 1000.0) == pytest.approx(5.0)
    breached, key = led.would_breach(("league:NBA",), 10.0, 1000.0)
    assert breached is True and key == "total_open"


def test_seeded_preserves_existing_exposure():
    led = ExposureLedger.seeded(
        ExposureLimits(), exposure_by_entity={"game:NBA:A @ B": 30.0}, total_open=30.0
    )
    assert led.exposure_for("game:NBA:A @ B") == 30.0
    assert led.headroom(("game:NBA:A @ B",), 1000.0) == pytest.approx(20.0)


# --- ExposurePolicy.admit ------------------------------------------------------
def test_admit_accept_when_within_headroom():
    pol = ExposurePolicy()
    led = ExposureLedger()
    v = pol.admit(entity_keys=_KEYS, desired_stake=30.0, ledger=led, bankroll=1000.0)
    assert v.action is ExposureAction.ACCEPT
    assert v.stake == 30.0


def test_admit_downsizes_to_headroom():
    pol = ExposurePolicy()
    led = ExposureLedger()
    led.add(_KEYS, 40.0)  # game headroom now 10
    v = pol.admit(entity_keys=_KEYS, desired_stake=30.0, ledger=led, bankroll=1000.0)
    assert v.action is ExposureAction.DOWNSIZE
    assert v.stake == pytest.approx(10.0)
    assert v.reason == "exposure_headroom"


def test_admit_skips_at_cap_with_binding_reason():
    pol = ExposurePolicy()
    led = ExposureLedger()
    led.add(_KEYS, 50.0)  # game at cap, headroom 0
    v = pol.admit(entity_keys=_KEYS, desired_stake=30.0, ledger=led, bankroll=1000.0)
    assert v.action is ExposureAction.SKIP
    assert v.stake == 0.0
    assert v.reason == "game:NBA:A @ B"


def test_admit_is_read_only_on_ledger():
    pol = ExposurePolicy()
    led = ExposureLedger()
    pol.admit(entity_keys=_KEYS, desired_stake=30.0, ledger=led, bankroll=1000.0)
    assert led.total_open == 0.0  # admit must not commit; caller does ledger.add


@pytest.mark.parametrize("seeded_stake,desired", [(0, 10), (40, 30), (50, 30), (49.5, 100)])
def test_admit_never_authorizes_more_than_desired(seeded_stake, desired):
    # Doctrine guardrail: exposure only caps/skips, never raises a stake.
    pol = ExposurePolicy()
    led = ExposureLedger()
    if seeded_stake:
        led.add(_KEYS, seeded_stake)
    v = pol.admit(entity_keys=_KEYS, desired_stake=desired, ledger=led, bankroll=1000.0)
    assert v.stake <= desired + 1e-9
