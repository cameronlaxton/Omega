"""Optional cited context bundles for derived session reports."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ALLOWED_CATEGORIES = frozenset(
    {
        "injury_role",
        "rest_travel",
        "recent_form",
        "matchup",
        "h2h",
        "market_movement",
        "data_gap",
        "weather",
        "lineup",
        "source_note",
    }
)

_PROHIBITED_TERMS = (
    "model probability",
    "model_prob",
    "calibrated probability",
    "edge_pct",
    "edge%",
    "ev_pct",
    "ev%",
    "expected value",
    "kelly",
    "confidence tier",
    "tier a",
    "tier b",
    "tier c",
    "fair price",
    "fair_price",
    "no-vig",
    "no_vig",
    "stake sizing",
    "recommended units",
    "recommendation strength",
)


class ContextBundleError(ValueError):
    """Raised when a report context bundle violates the contract."""


class ReportContextEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entry_id: str = Field(min_length=1)
    trace_id: str | None = None
    league: str | None = None
    matchup: str | None = None
    player: str | None = None
    market: str | None = None
    category: str
    source_type: str = Field(min_length=1)
    source_title: str = Field(min_length=1)
    source_url: str | None = None
    captured_at: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    claim_hash: str = Field(min_length=1)

    @field_validator("category")
    @classmethod
    def _category_allowed(cls, value: str) -> str:
        if value not in ALLOWED_CATEGORIES:
            raise ValueError(f"category must be one of {sorted(ALLOWED_CATEGORIES)}")
        return value

    @model_validator(mode="after")
    def _validate_claim(self) -> ReportContextEntry:
        claim_lc = self.claim.lower()
        for term in _PROHIBITED_TERMS:
            if term in claim_lc:
                raise ValueError(f"claim contains prohibited betting term {term!r}")
        expected = hashlib.sha256(self.claim.encode("utf-8")).hexdigest()
        if self.claim_hash != expected:
            raise ValueError("claim_hash must be sha256(claim)")
        return self


class ReportContextBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    bundle_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    generated_at: str = Field(min_length=1)
    generated_by: str = Field(min_length=1)
    mode: Literal["persisted+cited"]
    entries: list[ReportContextEntry] = Field(default_factory=list)


def load_context_bundle(path: str | Path | None) -> ReportContextBundle | None:
    if path is None:
        return None
    if os.environ.get("OMEGA_REPLAY_MODE") == "1":
        raise ContextBundleError("cited context bundles are forbidden in replay/backtest mode")
    bundle_path = Path(path)
    with bundle_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    try:
        return ReportContextBundle.model_validate(payload)
    except Exception as exc:  # noqa: BLE001
        raise ContextBundleError(str(exc)) from exc
