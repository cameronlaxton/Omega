"""Omega package bootstrap.

Load a local .env file for developer convenience. Production deployments should
supply environment variables through a secure mechanism and not rely on .env
files.
"""

from __future__ import annotations

import os
from pathlib import Path


def _load_env_file_without_dependency() -> None:
    """Tiny .env fallback for scripts that only need KEY=VALUE bindings."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
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
