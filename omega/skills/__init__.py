"""omega.skills — registry and helpers for Omega operational skills.

Skills are typed observers that run after orchestrator pipeline stages.
They emit structured findings but do not mutate pipeline state or replace
deterministic logic.

Usage:
    @register("my-skill")
    class MySkill(SkillBase):
        name = "my-skill"
        stage = "gathering"
        def _run(self, **kwargs): ...
"""
from __future__ import annotations

from pathlib import Path
import json
from typing import Optional

from omega.skills.base import SkillBase, SkillObservation

__all__ = ["SkillBase", "SkillObservation", "register", "get_registry", "get_skill", "is_enabled", "config"]

PACKAGE_ROOT = Path(__file__).parent
_CONFIG_PATH = PACKAGE_ROOT / "config.json"

_registry: dict = {}


def register(name: str):
    """Decorator to register a skill class.

    Usage:
        @register("trace-recorder")
        class TraceRecorder(SkillBase): ...
    """
    def _decorator(obj):
        _registry[name] = obj
        return obj
    return _decorator


def get_registry() -> dict:
    return dict(_registry)


def get_skill(name: str) -> Optional[SkillBase]:
    """Return an instantiated skill by name, or None if not registered/enabled."""
    if not is_enabled(name):
        return None
    cls = _registry.get(name)
    if cls is None:
        return None
    return cls()


def is_enabled(skill_name: str) -> bool:
    try:
        with _CONFIG_PATH.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        flags = cfg.get("enabled", {})
        return bool(flags.get(skill_name, False))
    except Exception:
        return False


def config() -> dict:
    try:
        with _CONFIG_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"enabled": {}, "retention_days": 30}


def _auto_import_skills() -> None:
    """Import all skill modules so their @register decorators fire."""
    import importlib
    _skill_modules = [
        "omega.skills.trace_recorder",
        "omega.skills.evidence_validator",
        "omega.skills.data_quality_grader",
        "omega.skills.writing_style",
        "omega.skills.evolution_tracker",
    ]
    for module in _skill_modules:
        try:
            importlib.import_module(module)
        except Exception:
            pass


_auto_import_skills()
