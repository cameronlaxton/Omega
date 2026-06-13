"""Tests for the shared ops date-argument helpers."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from omega.ops._date_args import iter_dates, parse_date_arg


def test_parse_iso_date():
    assert parse_date_arg("2026-06-29") == date(2026, 6, 29)
    assert parse_date_arg("  2026-06-29  ") == date(2026, 6, 29)


def test_parse_today_and_yesterday():
    today = datetime.now(timezone.utc).date()
    assert parse_date_arg("today") == today
    assert parse_date_arg("TODAY") == today
    assert (today - parse_date_arg("yesterday")).days == 1


def test_parse_bad_date_raises():
    with pytest.raises(ValueError):
        parse_date_arg("not-a-date")


def test_iter_dates_inclusive():
    days = list(iter_dates(date(2026, 6, 28), date(2026, 7, 1)))
    assert days == [date(2026, 6, 28), date(2026, 6, 29), date(2026, 6, 30), date(2026, 7, 1)]


def test_iter_dates_single_day():
    assert list(iter_dates(date(2026, 6, 29), date(2026, 6, 29))) == [date(2026, 6, 29)]


def test_iter_dates_empty_when_end_before_start():
    assert list(iter_dates(date(2026, 6, 29), date(2026, 6, 28))) == []


def test_outcomes_scripts_share_the_helper():
    """The soccer/tennis outcome scripts alias the shared helper, not a copy."""
    from omega.ops import fetch_outcomes_soccer, fetch_outcomes_tennis

    assert fetch_outcomes_soccer._parse_date_arg is parse_date_arg
    assert fetch_outcomes_soccer._iter_dates is iter_dates
    assert fetch_outcomes_tennis._parse_date_arg is parse_date_arg
