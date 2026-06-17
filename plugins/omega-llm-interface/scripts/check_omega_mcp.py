"""Lightweight Omega MCP plugin check.

This does not require the optional MCP SDK. It verifies that the repo-local
domain functions and manifests can be imported before a client starts an MCP
session.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from omega.mcp.server import PROMPT_NAMES, RESOURCE_URIS, TOOL_NAMES  # noqa: E402


def main() -> int:
    print(
        json.dumps(
            {
                "tool_count": len(TOOL_NAMES),
                "tools": list(TOOL_NAMES),
                "resource_count": len(RESOURCE_URIS),
                "resources": list(RESOURCE_URIS),
                "prompt_count": len(PROMPT_NAMES),
                "prompts": list(PROMPT_NAMES),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
