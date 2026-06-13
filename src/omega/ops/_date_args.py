"""Shared CLI date-argument helpers for the ops outcome/backfill scripts.

Every ``fetch_outcomes_*`` / ``backfill_outcomes_*`` script accepts the same
``--since``/``--until`` window with the same ``today``/``yesterday`` keywords
and iterates the inclusive day range identically. This module holds that logic
once so the scripts import it instead of each carrying a private copy.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date, datetime, timedelta, timezone

UTC = timezone.utc


def parse_date_arg(s: str) -> date:
    """Parse a CLI date argument: ``YYYY-MM-DD``, ``today``, or ``yesterday``."""
    s = s.strip().lower()
    if s == "today":
        return datetime.now(UTC).date()
    if s == "yesterday":
        return datetime.now(UTC).date() - timedelta(days=1)
    return date.fromisoformat(s)


def iter_dates(start: date, end: date) -> Iterator[date]:
    """Yield each date from *start* through *end* inclusive."""
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)
