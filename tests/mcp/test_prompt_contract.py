from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_system_prompt_mentions_mcp_without_replacing_standalone_sandbox():
    text = (ROOT / "prompts" / "system_prompt.txt").read_text(encoding="utf-8").lower()

    assert "prefer typed mcp tools first" in text
    assert "omega_lite_standalone.py" in text
    assert "standard text only" in text
    assert "you do not own any of the engine's responsibilities" in text


def test_cowork_prompt_uses_mcp_before_direct_repo_import_when_available():
    text = (ROOT / "OMEGA_COWORK.md").read_text(encoding="utf-8").lower()

    assert "prefer the local omega mcp server" in text
    assert "not a separate betting engine" in text
    assert "direct repo import flow" in text
    assert "only for no-local-access project sandboxes" in text


def test_llm_mcp_interface_documents_replay_as_audit_only():
    text = (ROOT / "docs" / "LLM_MCP_INTERFACE.md").read_text(encoding="utf-8").lower()

    assert "mcp layer is not a second pipeline" in text
    assert "live fetching is disabled" in text
    assert "replay is sampled audit" in text
    assert "not the default benchmark path" in text
    assert "standard text only" in text


def test_plugin_config_points_to_current_repo_root_relatively():
    text = (ROOT / "plugins" / "omega-llm-interface" / ".mcp.json").read_text(
        encoding="utf-8"
    )

    assert '"cwd": "../.."' in text
    assert "Desktop" not in text
