"""Omega package bootstrap.

Load a local .env file for developer convenience. Production deployments should
supply environment variables through a secure mechanism and not rely on .env
files.
"""

from __future__ import annotations

import logging
import os

from omega.paths import repo_root

# Library convention: attach a NullHandler to the package root logger so that
# `logger.*` calls in omega.* never hit Python's lastResort handler (which only
# surfaces WARNING+ unformatted and silently drops INFO/DEBUG) when omega is
# imported without a CLI calling logging.basicConfig (e.g. the MCP server,
# tests, notebooks). CLIs still configure their own handlers/levels.
logging.getLogger("omega").addHandler(logging.NullHandler())


def _load_env_file_without_dependency() -> None:
    """Tiny .env fallback for scripts that only need KEY=VALUE bindings."""
    env_path = repo_root() / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    _load_env_file_without_dependency()
