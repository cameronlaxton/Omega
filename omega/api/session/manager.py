"""
Session manager with Redis backend and in-memory fallback.

Stores conversation history per session_id. Redis is preferred for
production (persistence across restarts); in-memory dict is used
when Redis is unavailable (local dev, tests).
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("omega.api.session")


class Session:
    """In-memory representation of a conversation session."""

    __slots__ = ("session_id", "created_at", "messages", "metadata")

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.messages: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}

    def add_message(self, role: str, content: str, **kwargs: Any) -> Dict[str, Any]:
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **kwargs,
        }
        self.messages.append(msg)
        return msg

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "messages": self.messages,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        s = cls(data["session_id"])
        s.created_at = data.get("created_at", s.created_at)
        s.messages = data.get("messages", [])
        s.metadata = data.get("metadata", {})
        return s


class SessionManager:
    """Manages sessions with optional Redis persistence."""

    def __init__(self, redis_url: Optional[str] = None, ttl: int = 86400) -> None:
        self._ttl = ttl
        self._redis = None
        self._memory: Dict[str, Session] = {}

        if redis_url:
            try:
                import redis as redis_lib

                self._redis = redis_lib.from_url(redis_url, decode_responses=True)
                self._redis.ping()
                logger.info("Session manager connected to Redis")
            except Exception as e:
                logger.warning("Redis unavailable (%s), falling back to in-memory", e)
                self._redis = None

    def _key(self, session_id: str) -> str:
        return f"omega:session:{session_id}"

    def create_session(self, session_id: Optional[str] = None) -> Session:
        sid = session_id or uuid.uuid4().hex[:16]
        session = Session(sid)
        self._save(session)
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        if self._redis:
            raw = self._redis.get(self._key(session_id))
            if raw:
                return Session.from_dict(json.loads(raw))
            return None
        return self._memory.get(session_id)

    def get_or_create(self, session_id: Optional[str] = None) -> Session:
        if session_id:
            existing = self.get_session(session_id)
            if existing:
                return existing
        return self.create_session(session_id)

    def save_session(self, session: Session) -> None:
        self._save(session)

    def _save(self, session: Session) -> None:
        if self._redis:
            self._redis.setex(
                self._key(session.session_id),
                self._ttl,
                json.dumps(session.to_dict()),
            )
        else:
            self._memory[session.session_id] = session

    def delete_session(self, session_id: str) -> bool:
        if self._redis:
            return bool(self._redis.delete(self._key(session_id)))
        return self._memory.pop(session_id, None) is not None
