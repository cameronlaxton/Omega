"""ETL standard 3 — centralized alias-based entity resolution.

Names resolve via exact -> normalize_player_name -> alias table -> unresolved.
Unresolved entities are excluded from the priors write and emit a warning.

References:
  omega/integrations/_etl.py::resolve_entity, resolve_entities, load_alias_table
  data/aliases/NFL.json
  docs/phase7/MULTI_SPORT_EXPANSION.md  (Part 5B standard 3; verification test 15)
"""

from __future__ import annotations

import json

from omega.integrations._etl import (
    load_alias_table,
    resolve_entities,
    resolve_entity,
)
from omega.trace.session_sidecar import SessionSidecar, bootstrap_payload

_NFL_TABLE = {
    "canonical": ["Patrick Mahomes", "Christian McCaffrey"],
    "aliases": {"CMC": "Christian McCaffrey", "P. Mahomes": "Patrick Mahomes"},
}


def test_suffix_variant_resolves_to_same_key():
    a = resolve_entity("Patrick Mahomes II", _NFL_TABLE)
    b = resolve_entity("Patrick Mahomes", _NFL_TABLE)
    assert a == b == "Patrick Mahomes"


def test_alias_table_nickname_resolves():
    assert resolve_entity("CMC", _NFL_TABLE) == "Christian McCaffrey"


def test_unknown_entity_is_unresolved():
    assert resolve_entity("Some Rookie", _NFL_TABLE) is None


def test_shipped_nfl_alias_table_loads_and_resolves():
    table = load_alias_table("NFL")
    assert "Patrick Mahomes" in table["canonical"]
    assert resolve_entity("Pat Mahomes", table) == "Patrick Mahomes"


def test_missing_league_table_degrades_to_empty(tmp_path):
    table = load_alias_table("NONEXISTENT", alias_root=tmp_path)
    assert table == {"canonical": [], "aliases": {}}


def test_resolve_entities_excludes_unresolved_and_warns(tmp_path):
    payload = bootstrap_payload(
        "sess-alias-test",
        model_version="test",
        purpose="alias unit test",
        bankroll=1000.0,
    )
    sidecar_path = tmp_path / "sess-alias-test.json"
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")

    resolved, unresolved = resolve_entities(
        ["Patrick Mahomes II", "Ghost Player"],
        _NFL_TABLE,
        source="nflverse",
        session_path=sidecar_path,
    )

    assert resolved == {"Patrick Mahomes II": "Patrick Mahomes"}
    assert unresolved == ["Ghost Player"]

    sidecar = SessionSidecar.from_path(sidecar_path)
    warn_events = [
        e for e in sidecar.audit_events if e.event_type == "data_provenance" and e.status == "warn"
    ]
    assert len(warn_events) == 1
    assert "Ghost Player" in warn_events[0].notes
