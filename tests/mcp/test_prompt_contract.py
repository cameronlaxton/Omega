from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def test_system_prompt_mentions_mcp_and_core_service():
    text = (ROOT / "prompts" / "system_prompt.txt").read_text(encoding="utf-8").lower()

    assert "local omega mcp server" in text
    assert "omega.core.contracts.service.analyze" in text
    assert "qualitative research only" in text
    assert "deterministic python engine owns" in text


def test_cowork_prompt_uses_mcp_before_direct_repo_import_when_available():
    text = (ROOT / "OMEGA_COWORK.md").read_text(encoding="utf-8").lower()

    assert "local mcp server first" in text or "use the local mcp server first" in text
    assert "not a second betting engine" in text
    assert "omega.core.contracts.service.analyze" in text
    assert "phase 6h" in text


def test_llm_mcp_interface_documents_replay_as_audit_only():
    text = (ROOT / "docs" / "LLM_MCP_INTERFACE.md").read_text(encoding="utf-8").lower()

    assert "mcp layer is not a second pipeline" in text
    assert "live fetching is disabled" in text
    assert "replay is sampled audit" in text
    assert "not the default benchmark path" in text
    assert "qualitative text only" in text


def test_plugin_mcp_config_uses_installed_package_not_relative_escape():
    """No relative-escape cwd; Claude copies plugins to a cache where ../.. is invalid."""
    import json

    text = (ROOT / "plugins" / "omega-llm-interface" / ".mcp.json").read_text(encoding="utf-8")
    config = json.loads(text)

    assert '"cwd": "../"' not in text
    assert '"cwd": "../.."' not in text
    assert "Desktop" not in text

    omega_server = config["mcpServers"]["omega"]
    assert omega_server["command"] == "python"
    assert omega_server["args"] == ["-m", "omega.mcp.server"]


def test_claude_plugin_manifest_exists_and_is_well_formed():
    """Plugin must ship a .claude-plugin/plugin.json for Claude Code discovery."""
    import json

    manifest_path = ROOT / "plugins" / "omega-llm-interface" / ".claude-plugin" / "plugin.json"
    assert manifest_path.exists(), f"missing {manifest_path}"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["name"] == "omega-llm-interface"
    assert "version" in manifest


def test_claude_marketplace_lists_omega_plugin():
    """Repo-root marketplace must catalog the omega-llm-interface plugin."""
    import json

    market_path = ROOT / ".claude-plugin" / "marketplace.json"
    assert market_path.exists(), f"missing {market_path}"
    market = json.loads(market_path.read_text(encoding="utf-8"))
    names = [p["name"] for p in market["plugins"]]
    assert "omega-llm-interface" in names


def test_daily_prompts_are_league_first_not_props_siloed():
    daily_dir = ROOT / "prompts" / "daily"

    for name in ("nba_daily.md", "wnba_daily.md", "mlb_daily.md"):
        text = (daily_dir / name).read_text(encoding="utf-8").lower()
        assert "complete" in text
        assert "player props" in text
        assert re.search(r"do\s+not run a separate\s+props prompt", text)
        assert "quality_gate/null_data_audit" in text

    props_text = (daily_dir / "props_daily.md").read_text(encoding="utf-8").lower()
    assert "deprecated" in props_text
    assert "use the league prompt instead" in props_text
