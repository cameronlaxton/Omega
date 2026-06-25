from __future__ import annotations

import re
from pathlib import Path

import pytest

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


def test_plugin_mcp_config_pins_active_local_checkout():
    """Local plugin launch must bind to the active C:\\repos\\Omega checkout."""
    import json

    text = (ROOT / "plugins" / "omega-llm-interface" / ".mcp.json").read_text(encoding="utf-8")
    config = json.loads(text)

    assert '"cwd": "../"' not in text
    assert '"cwd": "../.."' not in text
    assert "Desktop" not in text

    omega_server = config["mcpServers"]["omega"]
    assert omega_server["command"] == "python"
    assert omega_server["args"] == ["-m", "omega.mcp.server"]
    assert omega_server["cwd"] == "C:\\repos\\Omega"


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

    props_path = daily_dir / "props_daily.md"
    assert props_path.exists()
    props_text = props_path.read_text(encoding="utf-8").lower()
    assert "deprecated_redirect" in props_text
    assert "runtime_allowed: false" in props_text
    assert "use the league prompt instead" in props_text
    assert "nba_daily.md" in props_text
    assert "wnba_daily.md" in props_text
    assert "mlb_daily.md" in props_text


def test_active_daily_prompts_reference_failure_budget_skill():
    daily_dir = ROOT / "prompts" / "daily"

    for name in ("nba_daily.md", "wnba_daily.md", "mlb_daily.md"):
        path = daily_dir / name
        text = path.read_text(encoding="utf-8").lower()
        assert "## session guard" in text
        assert "omega-failure-budget" in text
        assert "failure report" in text


def test_local_agent_masks_if_present_cover_generated_artifacts():
    required = {
        "var/reports/latest.md",
        "var/reports/run_audits/",
        "var/reports/*.txt",
        "var/inbox/traces/",
    }
    found_any = False
    for name in (".cursorignore", ".clineignore"):
        path = ROOT / name
        if not path.exists():
            continue
        found_any = True
        lines = {
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        assert required <= lines
    if not found_any:
        pytest.skip("local agent masks are workspace-local and intentionally untracked")


def test_presentation_contract_exists_and_defines_required_shape():
    """The narrative-first presentation contract must exist and define its response shape."""
    path = ROOT / "prompts" / "reference" / "presentation_contract.md"
    assert path.exists(), f"missing {path}"
    text = path.read_text(encoding="utf-8").lower()

    assert "narrative-first" in text
    assert "slate snapshot" in text
    assert "ranked recommendations" in text
    assert "per-matchup narrative" in text
    assert "honesty block" in text
    # Defers to output_modes.md (single source of truth) instead of restating it.
    assert "output_modes.md" in text


def test_presentation_contract_is_referenced_by_core_prompts():
    refs = [
        ROOT / "AGENTS.md",
        ROOT / "prompts" / "system_prompt.txt",
        ROOT / "OMEGA_COWORK.md",
    ]
    for path in refs:
        text = path.read_text(encoding="utf-8").lower()
        assert "presentation_contract.md" in text, f"{path} must reference presentation_contract.md"
        assert "narrative-first" in text, f"{path} must require narrative-first output"


def test_daily_prompts_require_narrative_first_rendering():
    daily_dir = ROOT / "prompts" / "daily"

    for name in ("nba_daily.md", "wnba_daily.md", "mlb_daily.md"):
        text = (daily_dir / name).read_text(encoding="utf-8").lower()
        assert "presentation_contract.md" in text
        assert "narrative-first" in text
        assert "slate snapshot" in text
        assert "honesty block" in text


def test_scheduled_prompts_reference_canonical_contracts():
    sched = ROOT / "prompts" / "scheduled"

    daily = (sched / "daily_gathering.md").read_text(encoding="utf-8").lower()
    assert "presentation_contract.md" in daily
    assert "narrative-first" in daily
    assert "output_modes" in daily
    assert "omega-session-bootstrap" in daily
    assert "omega-mcp-operator" in daily
    # Honesty-softened zero-bets discipline (do not relax output-authorization gates).
    assert "never fabricate" in daily

    for name in (
        "calibration_fitting.md",
        "closing_lines.md",
        "pending_outcomes.md",
        "tennis_outcomes_weekly.md",
    ):
        text = (sched / name).read_text(encoding="utf-8").lower()
        assert "omega-session-bootstrap" in text, f"{name} must reference omega-session-bootstrap"
