from __future__ import annotations

from pathlib import Path


def test_sync_to_mount_archives_var_runtime_paths():
    script = Path(__file__).resolve().parents[2] / "tools" / "windows" / "sync_to_mount.ps1"
    text = script.read_text(encoding="utf-8")

    assert 'Join-Path $Workspace "var\\inbox"' in text
    assert 'Join-Path $MountRoot "var\\inbox"' in text
    assert 'Join-Path $Workspace "var\\reports"' in text
    assert 'Join-Path $MountRoot "var\\reports"' in text
    assert 'Join-Path $Workspace "inbox"' not in text
    assert 'Join-Path $Workspace "reports"' not in text
    assert "var\\var/omega_traces.db" not in text
    assert "var\\var\\omega_traces.db" not in text
    assert 'Join-Path $Workspace "var\\omega_traces.db"' in text
