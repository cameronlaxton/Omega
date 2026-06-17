"""Backend resolution and SQLAlchemy engine helpers for trace persistence."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

BackendName = Literal["sqlite", "postgres"]

PHASE1_SQLITE_ONLY_MESSAGE = "not yet supported on Postgres backend; SQLite only in Phase 1"


@dataclass(frozen=True)
class BackendConfig:
    backend: BackendName
    target: str | None


def resolve_backend(db_path: str | None = None) -> BackendConfig:
    """Resolve the trace persistence backend.

    SQLite remains the default. A non-sqlite ``DATABASE_URL`` opts into the
    Postgres backend while preserving the existing ``db_path`` SQLite surface.
    """
    url = os.environ.get("DATABASE_URL")
    if not url or url.startswith("sqlite:"):
        return BackendConfig("sqlite", db_path)
    if not (url.startswith("postgresql://") or url.startswith("postgresql+")):
        raise ValueError("DATABASE_URL must be sqlite or postgresql/postgresql+psycopg")
    return BackendConfig("postgres", url)


def require_sqlite_backend(operation: str) -> None:
    """Fail loudly for Phase-1 maintenance paths that still use raw SQLite SQL."""
    if resolve_backend().backend == "postgres":
        raise RuntimeError(f"{operation} {PHASE1_SQLITE_ONLY_MESSAGE}")


def create_postgres_engine(url: str):
    """Create a pooled SQLAlchemy engine for the Postgres backend."""
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.pool import QueuePool
    except ImportError as exc:
        raise RuntimeError(
            "Postgres TraceStore requires sqlalchemy and psycopg: "
            "python -m pip install -e .[postgres]"
        ) from exc

    return create_engine(
        url,
        future=True,
        poolclass=QueuePool,
        pool_pre_ping=True,
    )


def create_session_factory(engine):
    try:
        from sqlalchemy.orm import sessionmaker
    except ImportError as exc:
        raise RuntimeError(
            "Postgres TraceStore requires sqlalchemy: python -m pip install -e .[postgres]"
        ) from exc

    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def bootstrap_create_all(url: str | None = None) -> None:
    """Developer-only empty-DB bootstrap.

    Alembic is the canonical migration path. This helper exists only for local
    scratch databases and refuses to run unless explicitly enabled.
    """
    if os.environ.get("OMEGA_DB_DEV_BOOTSTRAP") != "1":
        raise RuntimeError("bootstrap_create_all requires OMEGA_DB_DEV_BOOTSTRAP=1")
    target = url or os.environ.get("DATABASE_URL")
    if not target:
        raise RuntimeError("DATABASE_URL is required for bootstrap_create_all")

    from omega.trace.models import Base

    engine = create_postgres_engine(target)
    Base.metadata.create_all(engine)
