"""Versioned contract for `inbox/sessions/<session_id>.json` sidecars."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SessionSidecar(BaseModel):
    """Required session metadata used by calibration and operator reports."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1)
    opened_at: str = Field(min_length=1)
    closed_at: str | None = None
    model_version: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    bankroll: float = Field(gt=0)
    bankroll_confirmed: bool
    exec_stats: dict[str, Any]
    agent_notes: str

    @field_validator("opened_at", "closed_at")
    @classmethod
    def _reject_legacy_empty_timestamps(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not value.strip():
            raise ValueError("timestamp must not be blank")
        return value

    @classmethod
    def from_path(cls, path: Path) -> SessionSidecar:
        with path.open("r", encoding="utf-8") as fh:
            return cls.model_validate(json.load(fh))

    def to_report_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
