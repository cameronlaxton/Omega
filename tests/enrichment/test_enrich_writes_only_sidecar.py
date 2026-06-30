"""Doctrine guard: enrichment writes ONLY the sidecar; the console stays read-only.

These tests enforce the boundary the design depends on:

* the enrichment worker never modifies the canonical trace DB (byte-identical
  before/after a full generation), only the separate sidecar DB; and
* mounting the enrichment app beside the console adds no top-level mutating
  route — the read-only console surface remains GET-only.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from starlette.routing import Mount

from omega.enrichment.server import build_app_with_enrichment
from omega.enrichment.store import EnrichmentStore
from omega.enrichment.worker import run_enrichment


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_enrichment_never_modifies_canonical_trace_db(traces_db: str, enrich_db: str, tmp_path: Path):
    traces_path = Path(traces_db)
    before = _sha(traces_path)

    store = EnrichmentStore(enrich_db)
    eid = store.create(trace_id="enr-1", trace_type=None, league=None, market=None)
    store.close()
    run_enrichment(enrichment_id=eid, enrich_db=enrich_db, console_db=traces_db,
                   sessions_dir=str(tmp_path), provider_name="stub")

    # The canonical trace DB is byte-identical; the sidecar holds the artifact.
    assert _sha(traces_path) == before, "enrichment modified the canonical trace DB"
    assert Path(enrich_db).exists()
    store = EnrichmentStore(enrich_db, read_only=True)
    assert store.get(eid).status == "completed"
    store.close()


def test_mounted_console_routes_stay_get_only(tmp_path: Path):
    app = build_app_with_enrichment(
        db_path=str(tmp_path / "t.db"),
        sessions_dir=str(tmp_path),
        enrich_db=str(tmp_path / "e.db"),
        provider_name="stub",
    )
    mutating = {"POST", "PUT", "PATCH", "DELETE"}
    offenders = []
    enrich_mount = None
    for route in app.routes:
        if isinstance(route, Mount):
            if route.path == "/enrich":
                enrich_mount = route
            continue
        methods = getattr(route, "methods", None)
        if methods and (set(methods) & mutating):
            offenders.append((getattr(route, "path", "?"), set(methods) & mutating))
    assert offenders == [], f"console exposes mutating top-level routes: {offenders}"
    assert enrich_mount is not None, "enrichment sub-app was not mounted at /enrich"


def test_enrichment_package_not_scanned_by_console_guard():
    """The console mutation guard scans only omega/ui + console_server — the
    writable enrichment package must live outside that set (it does)."""
    import omega.enrichment as _enr
    import omega.ui as _ui

    ui_dir = Path(_ui.__file__).resolve().parent
    enr_dir = Path(_enr.__file__).resolve().parent
    assert enr_dir != ui_dir
    assert ui_dir not in enr_dir.parents
