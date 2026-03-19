"""omega.skills package: lightweight registry and helpers for CoWork skills.
This is a minimal scaffold. Skills should register themselves via the registry
and consult `is_enabled(skill_name)` before acting.
"""
from pathlib import Path
import json

PACKAGE_ROOT = Path(__file__).parent
_CONFIG_PATH = PACKAGE_ROOT / "config.json"

_registry = {}


def register(name):
    """Decorator to register a skill factory/function.
    Usage:
        @register("writing-style")
        class WritingStyle: ...
    """
    def _decorator(obj):
        _registry[name] = obj
        return obj
    return _decorator


def get_registry():
    return dict(_registry)


def is_enabled(skill_name: str) -> bool:
    try:
        with _CONFIG_PATH.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        flags = cfg.get("enabled", {})
        return bool(flags.get(skill_name, False))
    except Exception:
        # default safe: disabled
        return False


def config():
    try:
        with _CONFIG_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"enabled": {}, "retention_days": 30}
