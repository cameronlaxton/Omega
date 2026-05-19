"""
omega.skills.base — SkillBase contract and SkillObservation result type.

Skills are typed observers. They run after a pipeline stage, emit structured
findings, and never replace existing deterministic logic. The orchestrator does
not act on findings; they are diagnostic only.

Contract rules:
- observe() must never raise
- observe() must always return a SkillObservation
- Skills may log events but must not mutate pipeline state
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger("omega.skills.base")


class SkillObservation:
    """Result of a skill observation pass.

    Attributes:
        skill: Registered skill name (matches config.json key).
        stage: Orchestrator stage this observation covers.
        ok: True when no anomalies detected.
        findings: List of finding codes, e.g. "critical_slot_defaulted:home.off_rating".
        error: Set when the skill itself encountered an internal error.
    """

    __slots__ = ("skill", "stage", "ok", "findings", "error")

    def __init__(
        self,
        skill: str,
        stage: str,
        ok: bool,
        findings: list[str] | None = None,
        error: str | None = None,
    ) -> None:
        self.skill = skill
        self.stage = stage
        self.ok = ok
        self.findings: list[str] = findings or []
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill": self.skill,
            "stage": self.stage,
            "ok": self.ok,
            "findings": self.findings,
            "error": self.error,
        }

    def __repr__(self) -> str:
        return (
            f"SkillObservation(skill={self.skill!r}, stage={self.stage!r}, "
            f"ok={self.ok}, findings={self.findings!r})"
        )


class SkillBase(ABC):
    """Abstract base for all Omega operational skills.

    Subclasses must define:
        name: str      — matches the key in config.json
        stage: str     — orchestrator stage this skill observes

    And implement:
        _run(**kwargs) -> SkillObservation
            The actual observation logic. May raise; the base class wraps it.
    """

    name: str = ""
    stage: str = ""

    def observe(self, **kwargs: Any) -> SkillObservation:
        """Public entry point. Never raises. Wraps _run() in a safety catch."""
        try:
            return self._run(**kwargs)
        except Exception as exc:
            logger.warning(
                "Skill %r internal error at stage %r: %s",
                self.name, self.stage, exc,
            )
            return SkillObservation(
                skill=self.name,
                stage=self.stage,
                ok=False,
                findings=[],
                error=str(exc),
            )

    @abstractmethod
    def _run(self, **kwargs: Any) -> SkillObservation:
        """Override with observation logic. May raise; base class handles it."""
