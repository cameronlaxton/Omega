"""
omega.ops.validate_docs â€” lint script to assert documentation integrity.

Verifies:
1. Canonical reference files exist in prompts/reference/*.md.
2. All relative markdown links in operational docs actually resolve.
3. Every daily prompt contains a mandatory Step-0 read step for output_modes.md.
4. AGENTS.md points explicitly to all canonical references.
5. No operational doc outside output_modes.md contains duplicate/redefined rules for RESEARCH_CANDIDATE
   (e.g., Permitted/Forbidden lists or detailed output-mode tables).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Files we consider "operational docs" that must have strict validation.
_OPERATIONAL_DOCS = [
    _REPO_ROOT / "AGENTS.md",
    _REPO_ROOT / "README.md",
    _REPO_ROOT / "OMEGA_RUNTIME.md",
    _REPO_ROOT / "prompts" / "system_prompt.txt",
    _REPO_ROOT / "prompts" / "daily" / "mlb_daily.md",
    _REPO_ROOT / "prompts" / "daily" / "nba_daily.md",
    _REPO_ROOT / "prompts" / "daily" / "wnba_daily.md",
]

_CANONICAL_REFS = [
    "prompts/reference/output_modes.md",
    "prompts/reference/engine_output_validation.md",
    "prompts/reference/markov_evidence_vocab.md",
]

_PLUGIN_SKILL_POINTERS = {
    "omega-known-bug-sentinel": ".agents/skills/omega-known-bug-sentinel/SKILL.md",
    "omega-mcp-operator": ".agents/skills/omega-mcp-operator/SKILL.md",
    "omega-replay-qa": ".agents/skills/omega-replay-qa/SKILL.md",
    "omega-session-bootstrap": ".agents/skills/omega-session-bootstrap/SKILL.md",
    "omega-trace-qa": ".agents/skills/omega-trace-qa/SKILL.md",
}


def check_canonical_refs_exist() -> bool:
    print("Checking canonical reference files exist...")
    success = True
    for ref in _CANONICAL_REFS:
        path = _REPO_ROOT / ref
        if not path.is_file():
            print(f"  [FAIL] Canonical reference does not exist: {ref}")
            success = False
        else:
            print(f"  [PASS] {ref} exists.")
    return success


def check_agents_links() -> bool:
    print("Checking AGENTS.md points to all canonical reference files...")
    agents_path = _REPO_ROOT / "AGENTS.md"
    if not agents_path.is_file():
        print("  [FAIL] AGENTS.md does not exist.")
        return False

    content = agents_path.read_text(encoding="utf-8")
    success = True
    for ref in _CANONICAL_REFS:
        if ref not in content:
            print(f"  [FAIL] AGENTS.md does not reference: {ref}")
            success = False
        else:
            print(f"  [PASS] AGENTS.md references {ref}")
    return success


def check_daily_prompts_preflight() -> bool:
    print("Checking daily prompts contain output_modes.md read step...")
    daily_dir = _REPO_ROOT / "prompts" / "daily"
    success = True

    for p in daily_dir.glob("*.md"):
        # Skip deprecated files or redirect files
        if p.name == "props_daily.md":
            continue

        content = p.read_text(encoding="utf-8")
        if "output_modes.md" not in content:
            print(f"  [FAIL] Daily prompt {p.name} misses preflight read for output_modes.md")
            success = False
        else:
            print(f"  [PASS] Daily prompt {p.name} contains output_modes.md reference.")
    return success


def check_relative_links() -> bool:
    print("Checking relative markdown links resolve...")
    success = True
    # Markdown link pattern: [label](path)
    link_pattern = re.compile(r"\[[^\]]*\]\(([^)]+)\)")

    for doc in _OPERATIONAL_DOCS:
        if not doc.is_file():
            continue

        content = doc.read_text(encoding="utf-8")
        links = link_pattern.findall(content)

        for link in links:
            # Skip HTTP/HTTPS/mailto links or raw anchors within the same document
            if (
                link.startswith("http://")
                or link.startswith("https://")
                or link.startswith("mailto:")
                or link.startswith("#")
                or link.startswith("file:///")
            ):
                continue

            # Strip query params or anchors
            link_path_only = link.split("#")[0].split("?")[0]
            if not link_path_only:
                continue

            # Try to resolve relative to the document parent first, then relative to repo root
            target_relative = doc.parent / link_path_only
            target_root = _REPO_ROOT / link_path_only

            if not target_relative.exists() and not target_root.exists():
                print(f"  [FAIL] Broken relative link in {doc.relative_to(_REPO_ROOT)}: '{link}'")
                success = False

    if success:
        print("  [PASS] All relative markdown links resolved successfully.")
    return success


def check_no_duplicate_definitions() -> bool:
    print("Checking for duplicate/stale RESEARCH_CANDIDATE definitions...")
    success = True

    # Stale/duplicated terminology that belongs strictly inside output_modes.md
    banned_indicators = [
        "### In RESEARCH_CANDIDATE mode",
        "Forbidden language",
        "actionable bet / Actionable Bet",
        "Static fallback; 0 calibration-eligible traces",
        "Pairs needed for fit",
    ]

    for doc in _OPERATIONAL_DOCS:
        # Ignore output_modes.md itself
        if doc == _REPO_ROOT / "prompts" / "reference" / "output_modes.md":
            continue
        if not doc.is_file():
            continue

        content = doc.read_text(encoding="utf-8")
        for indicator in banned_indicators:
            if indicator in content:
                print(
                    f"  [FAIL] Operational doc {doc.relative_to(_REPO_ROOT)} contains a duplicated "
                    f"RESEARCH_CANDIDATE definition or obsolete terminology indicator: '{indicator}'"
                )
                success = False

    if success:
        print("  [PASS] No duplicate or obsolete output-mode definitions found.")
    return success


def check_plugin_skill_pointers() -> bool:
    print("Checking plugin skill files are thin pointers to .agents canonical skills...")
    success = True
    for skill, canonical in _PLUGIN_SKILL_POINTERS.items():
        canonical_path = _REPO_ROOT / canonical
        plugin_path = _REPO_ROOT / "plugins" / "omega-llm-interface" / "skills" / skill / "SKILL.md"
        if not canonical_path.is_file():
            print(f"  [FAIL] Canonical skill missing: {canonical}")
            success = False
            continue
        if not plugin_path.is_file():
            print(f"  [FAIL] Plugin skill pointer missing: {plugin_path.relative_to(_REPO_ROOT)}")
            success = False
            continue
        content = plugin_path.read_text(encoding="utf-8")
        if f"Canonical source: `{canonical}`" not in content:
            print(f"  [FAIL] Plugin skill {skill} does not point to canonical source {canonical}")
            success = False
        elif plugin_path.read_bytes() == canonical_path.read_bytes():
            print(f"  [FAIL] Plugin skill {skill} is a byte-for-byte mirror, not a pointer")
            success = False
        else:
            print(f"  [PASS] Plugin skill {skill} points to {canonical}.")
    return success


_BACKTICK_ROOT_MD_RE = re.compile(r"`([A-Za-z0-9_.-]+\.[Mm][Dd])`")

# Docs scanned for dangling bare-filename backtick references, in addition to
# _OPERATIONAL_DOCS. PHASE_HISTORY.MD is a common source of these because it is
# explicitly historical and easy to forget when a root doc is renamed.
_BACKTICK_SCAN_DOCS = [*_OPERATIONAL_DOCS, _REPO_ROOT / "docs" / "history" / "PHASE_HISTORY.MD"]

# Filenames that are legitimately backtick-referenced but never exist in THIS
# repo -- they belong to an external tool's per-session convention
# (Antigravity's task.md/walkthrough.md), not a renamed/missing Omega doc.
_EXTERNAL_TOOL_FILENAMES: frozenset[str] = frozenset({"task.md", "walkthrough.md"})

# Directories skipped when building the "does this filename exist anywhere"
# index. archive/ is intentionally INCLUDED (not skipped): AGENTS.md/README.md
# legitimately backtick-cite retired docs like OMEGA_RUN_RECIPE.md as archived
# material, and that's a correct reference, not drift. This check deliberately
# does not try to distinguish "cited as archived" (fine) from "cited as
# canonical" (not fine) -- it only catches references to filenames that do not
# exist ANYWHERE in the repo (the actual OMEGA_COWORK.md-style failure mode:
# a rename that leaves stale references pointing at nothing at all).
_SKIP_DIR_NAMES = {".git", "node_modules", ".venv", "venv"}


def check_backtick_root_doc_references() -> bool:
    print("Checking backtick-referenced root docs still exist...")
    success = True
    seen: set[str] = set()

    all_md_basenames: set[str] = set()
    for pattern in ("*.md", "*.MD"):
        for path in _REPO_ROOT.rglob(pattern):
            if any(part in _SKIP_DIR_NAMES for part in path.parts):
                continue
            all_md_basenames.add(path.name)

    for doc in _BACKTICK_SCAN_DOCS:
        if not doc.is_file():
            continue
        content = doc.read_text(encoding="utf-8")
        for name in _BACKTICK_ROOT_MD_RE.findall(content):
            key = f"{doc}:{name}"
            if key in seen or name in _EXTERNAL_TOOL_FILENAMES:
                continue
            seen.add(key)
            if name not in all_md_basenames:
                print(
                    f"  [FAIL] {doc.relative_to(_REPO_ROOT)} references `{name}` via backtick, "
                    "but no file with that name exists in the repo (outside archive/)."
                )
                success = False

    if success:
        print("  [PASS] All backtick-referenced root docs resolve to an existing file.")
    return success


def main() -> int:
    checks = [
        check_canonical_refs_exist(),
        check_agents_links(),
        check_daily_prompts_preflight(),
        check_relative_links(),
        check_no_duplicate_definitions(),
        check_plugin_skill_pointers(),
        check_backtick_root_doc_references(),
    ]

    if not all(checks):
        print("\n[VERDICT] Doc validation FAILED.")
        return 1

    print("\n[VERDICT] Doc validation PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
