"""Identity resolution: aliases, neutral-site (no swap), explicit swap, failures."""

from __future__ import annotations

from omega.historical.identity import event_key, resolve_event_identity, resolve_team


def test_exact_and_alias_resolution(nfl_alias_table):
    canon, ok = resolve_team("Kansas City Chiefs", "NFL", nfl_alias_table)
    assert ok and canon == "Kansas City Chiefs"

    canon, ok = resolve_team("KC", "NFL", nfl_alias_table)
    assert ok and canon == "Kansas City Chiefs"


def test_neutral_site_does_not_swap(nfl_alias_table):
    res = resolve_event_identity(
        "NFL",
        "Kansas City Chiefs",
        "Philadelphia Eagles",
        is_neutral_site=True,
        alias_table=nfl_alias_table,
    )
    # Neutral site preserves nominal home/away exactly.
    assert res.home == "Kansas City Chiefs"
    assert res.away == "Philadelphia Eagles"
    assert res.status == "complete"


def test_explicit_swap_reorders(nfl_alias_table):
    res = resolve_event_identity(
        "NFL",
        "Kansas City Chiefs",
        "Philadelphia Eagles",
        explicit_swap=True,
        alias_table=nfl_alias_table,
    )
    assert res.home == "Philadelphia Eagles"
    assert res.away == "Kansas City Chiefs"


def test_unresolved_increments_failure_count(nfl_alias_table):
    res = resolve_event_identity(
        "NFL",
        "Nonexistent Team",
        "Philadelphia Eagles",
        alias_table=nfl_alias_table,
    )
    assert res.status == "missing"
    assert res.failure_count == 1
    assert any("unresolved_home" in r for r in res.reasons)
    # The away side still resolves and is retained.
    assert res.away == "Philadelphia Eagles"


def test_event_key_joins_across_spellings(nfl_alias_table):
    """Two source spellings resolve to the same canonical key for the same game."""
    a = resolve_event_identity("NFL", "KC", "Eagles", alias_table=nfl_alias_table)
    b = resolve_event_identity(
        "NFL", "Kansas City Chiefs", "Philadelphia Eagles", alias_table=nfl_alias_table
    )
    key_a = event_key("NFL", "2023-09-10T17:00:00+00:00", a.home, a.away)
    key_b = event_key("NFL", "2023-09-10T20:00:00+00:00", b.home, b.away)
    # Same calendar date + canonical teams → identical join key despite spelling
    # and kickoff-time differences.
    assert key_a == key_b
