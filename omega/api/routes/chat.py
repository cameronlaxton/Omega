"""
Chat endpoint with Server-Sent Events (SSE) streaming.

Wires the research orchestrator into the SSE transport layer.
Each user message flows through:
    intent → strategy → planning → gathering → quality gate →
    execution → composition → streaming response
"""

from __future__ import annotations

import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from omega.api.schemas import ChatRequest, ChatStreamEvent
from omega.api.session.manager import SessionManager
from omega.reasoning.orchestrator import Orchestrator

logger = logging.getLogger("omega.api.chat")

router = APIRouter(tags=["chat"])

_session_manager: SessionManager | None = None
_orchestrator: Orchestrator | None = None


def set_session_manager(sm: SessionManager) -> None:
    global _session_manager
    _session_manager = sm


def set_orchestrator(orch: Orchestrator) -> None:
    global _orchestrator
    _orchestrator = orch


def _get_session_manager() -> SessionManager:
    if _session_manager is None:
        raise RuntimeError("SessionManager not initialized")
    return _session_manager


def _get_orchestrator() -> Orchestrator:
    if _orchestrator is None:
        # Lazy init with defaults if not explicitly set
        from omega.reasoning.orchestrator import Orchestrator as _Orch, OrchestratorConfig
        return _Orch(OrchestratorConfig())
    return _orchestrator


async def _stream_response(
    session_id: str,
    user_message: str,
    history: list | None = None,
) -> AsyncGenerator[dict, None]:
    """Stream orchestrator events as SSE."""
    orch = _get_orchestrator()

    async for event in orch.handle_query_stream(user_message, history):
        yield {
            "event": event["event_type"],
            "data": json.dumps(
                ChatStreamEvent(
                    event_type=event["event_type"],
                    data=event.get("data"),
                    session_id=session_id,
                ).model_dump()
            ),
        }


@router.post("/chat")
async def chat_endpoint(req: ChatRequest, request: Request):
    """Conversational endpoint with SSE streaming."""
    sm = _get_session_manager()
    session = sm.get_or_create(req.session_id)

    # Record user message
    session.add_message("user", req.message)
    sm.save_session(session)

    # Build history for orchestrator context
    history = session.messages[:-1]  # exclude current message

    async def event_generator():
        final_text = ""
        async for event in _stream_response(
            session.session_id, req.message, history,
        ):
            yield event
            if event.get("event") == "done":
                try:
                    payload = json.loads(event["data"])
                    data = payload.get("data", {})
                    if isinstance(data, dict):
                        final_text = data.get("final_text", "")
                except (json.JSONDecodeError, TypeError):
                    pass

        if final_text:
            session.add_message("assistant", final_text)
            sm.save_session(session)

    return EventSourceResponse(event_generator())


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session history."""
    sm = _get_session_manager()
    session = sm.get_session(session_id)
    if session is None:
        return {"error": "Session not found"}, 404
    return session.to_dict()
