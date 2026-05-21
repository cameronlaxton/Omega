"""omega.trace — trace persistence, retrieval, and outcome attachment.

Owns the lifecycle of ExecutionTrace artifacts after they leave the orchestrator.
Uses SQLite for storage with JSONL fallback via the trace-recorder skill.

Design rules (from CLAUDE.md):
- Outcome attachment happens AFTER initial trace persistence
- Traces must be persistable without depending on request/response wrappers
- Every persistence format must be versioned
"""

from omega.trace.persistable import PersistableTrace

__all__ = ["PersistableTrace"]
