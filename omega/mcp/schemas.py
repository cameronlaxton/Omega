"""Typed contracts for the Omega MCP tool surface."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

MCP_SCHEMA_VERSION = 1


class TraceQueryRequest(BaseModel):
    """Filters accepted by omega_trace_query."""

    db_path: Optional[str] = None
    league: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    has_outcome: Optional[bool] = None
    execution_mode: Optional[str] = None
    limit: int = Field(default=100, ge=1, le=1000)


class TraceAttachOutcomeRequest(BaseModel):
    """Outcome attachment is post-decision and must reference an existing trace."""

    trace_id: str
    home_score: int
    away_score: int
    source: str = "mcp"
    db_path: Optional[str] = None


class CalibrationFitPreviewRequest(BaseModel):
    """Dry-run calibration fitting request."""

    db_path: Optional[str] = None
    league: Optional[str] = None
    method: str = Field(default="isotonic", pattern="^(isotonic|shrinkage)$")
    limit: int = Field(default=1000, ge=1, le=10000)


class ReplayBundle(BaseModel):
    """Frozen evidence bundle for replay-plane audit only.

    Replay bundles are not quant benchmark inputs. They exist to audit routing,
    evidence selection, downgrade discipline, refusal discipline, and trace
    completeness using knowable-at-the-time facts.
    """

    schema_version: int = 1
    prompt: str
    facts: List[Dict[str, Any]] = Field(default_factory=list)
    source_trace_id: Optional[str] = None
    decision_date: Optional[str] = None
    simulation_seed: Optional[int] = None
    expected_outputs: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _reject_live_fetch_flags(self) -> "ReplayBundle":
        if self.metadata.get("live_fetch_enabled") is True:
            raise ValueError("replay bundles must disable live fetching")
        for fact in self.facts:
            if fact.get("post_outcome") is True:
                raise ValueError("replay facts must exclude post-outcome information")
            if fact.get("live_fetch") is True:
                raise ValueError("replay facts must not request live fetching")
        return self


class ReplayToolRequest(BaseModel):
    """MCP wrapper around ReplayBundle."""

    bundle: ReplayBundle
    strict: bool = False


class EvidenceRetrieveRequest(BaseModel):
    """Explicit gather slots for evidence retrieval.

    The current MCP adapter does not perform live network retrieval. It returns
    a skipped response with the requested slots so callers can use approved
    evidence channels outside replay or Standard Text mode.
    """

    slots: List[Dict[str, Any]] = Field(default_factory=list)
