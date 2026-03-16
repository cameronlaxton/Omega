"""
API-layer schemas for chat, sessions, and streaming.

These live outside omega.core because they are transport concerns,
not domain contracts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single message in a conversation."""

    role: str = Field(description="'user', 'assistant', or 'system'")
    content: str = Field(description="Message text content")
    timestamp: Optional[str] = Field(default=None, description="ISO 8601 timestamp")
    structured_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured payload attached to assistant messages",
    )


class ChatRequest(BaseModel):
    """Request body for POST /chat."""

    session_id: Optional[str] = Field(
        default=None,
        description="Session ID; server generates one if absent",
    )
    message: str = Field(min_length=1, description="User message text")


class ChatStreamEvent(BaseModel):
    """A single SSE event emitted by the /chat endpoint."""

    event_type: str = Field(
        description="stage_update | partial_text | structured_data | done | error",
    )
    data: Any = Field(default=None, description="Event payload — varies by event_type")
    session_id: str = Field(default="", description="Session this event belongs to")


class SessionInfo(BaseModel):
    """Summary of a session returned by GET /sessions/{id}."""

    session_id: str
    created_at: str
    message_count: int
    last_activity: str


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str = "ok"
    version: str = "0.1.0"
    services: Dict[str, str] = Field(default_factory=dict)
