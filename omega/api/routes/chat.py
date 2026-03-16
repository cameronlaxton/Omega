"""
Chat endpoint with Server-Sent Events (SSE) streaming.

This is the conversational interface. It:
1. Receives a user message
2. Manages session state
3. Streams back stage updates, partial text, structured data, and done events
4. The actual LLM orchestration is delegated to the research layer (future)

For now, this is a working skeleton that echoes back with a placeholder response.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from omega.api.schemas import ChatRequest, ChatStreamEvent
from omega.api.session.manager import SessionManager

logger = logging.getLogger("omega.api.chat")

router = APIRouter(tags=["chat"])

# Session manager is initialized at app startup (see app.py)
_session_manager: SessionManager | None = None


def set_session_manager(sm: SessionManager) -> None:
    global _session_manager
    _session_manager = sm


def _get_session_manager() -> SessionManager:
    if _session_manager is None:
        raise RuntimeError("SessionManager not initialized")
    return _session_manager


async def _stream_response(
    session_id: str,
    user_message: str,
) -> AsyncGenerator[dict, None]:
    """Generate SSE events for a chat response.

    This is the integration point for the research/agent layer.
    Currently returns a placeholder. Phase 3 will wire in the
    LLM orchestrator here.
    """
    # Stage 1: acknowledge
    yield {
        "event": "stage_update",
        "data": json.dumps(
            ChatStreamEvent(
                event_type="stage_update",
                data={"stage": "received", "message": "Processing your request..."},
                session_id=session_id,
            ).model_dump()
        ),
    }

    await asyncio.sleep(0.05)  # simulate processing

    # Stage 2: partial text (placeholder — will be replaced by LLM stream)
    response_text = (
        f"[Omega v0.1 — research layer not yet connected] "
        f"Received: \"{user_message[:100]}\""
    )
    yield {
        "event": "partial_text",
        "data": json.dumps(
            ChatStreamEvent(
                event_type="partial_text",
                data=response_text,
                session_id=session_id,
            ).model_dump()
        ),
    }

    # Stage 3: done
    yield {
        "event": "done",
        "data": json.dumps(
            ChatStreamEvent(
                event_type="done",
                data={"final_text": response_text},
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

    async def event_generator():
        final_text = ""
        async for event in _stream_response(session.session_id, req.message):
            yield event
            # Capture final text for session history
            if event.get("event") == "done":
                try:
                    payload = json.loads(event["data"])
                    final_text = payload.get("data", {}).get("final_text", "")
                except (json.JSONDecodeError, TypeError):
                    pass

        # Record assistant response in session
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
