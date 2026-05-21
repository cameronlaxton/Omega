from __future__ import annotations

import importlib.metadata

from scripts import cowork_preflight


def test_python_below_310_fails():
    failures = cowork_preflight.check_python((3, 9, 18))

    assert len(failures) == 1
    assert "Python 3.10+ is required" in failures[0]


def test_missing_distribution_points_to_editable_mcp_install(monkeypatch):
    def fake_version(_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(cowork_preflight.importlib.metadata, "version", fake_version)

    failures = cowork_preflight.check_distribution()

    assert failures == [
        "Omega is not installed in this interpreter. Run: python -m pip install -e .[mcp]"
    ]


def test_direct_only_skips_mcp_import(monkeypatch):
    imported: list[str] = []

    monkeypatch.setattr(cowork_preflight, "check_python", lambda: [])
    monkeypatch.setattr(cowork_preflight, "check_distribution", lambda: [])

    def fake_check_import(module: str, install_hint: str) -> list[str]:
        imported.append(module)
        return []

    monkeypatch.setattr(cowork_preflight, "check_import", fake_check_import)

    failures = cowork_preflight.run_checks(require_mcp=False)

    assert failures == []
    assert "mcp.server.fastmcp" not in imported
    assert "omega.core.contracts.service" in imported
