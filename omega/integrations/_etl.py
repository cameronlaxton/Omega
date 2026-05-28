"""Shared ETL harness for Phase 7 external-priors adapters.

Implements the three mandatory ETL standards from
``docs/phase7/MULTI_SPORT_EXPANSION.md`` Part 5B *once*, so every adapter
(tennis Sackmann, nflverse, StatsBomb, Understat, FBref, ...) gets the same
behavior without re-deriving it:

1. **Local caching layer before transform** — ``cached_fetch`` persists the raw
   upstream response under ``data/cache/<source>/`` *before* any transform. A
   cached pull within its TTL serves from disk and makes **zero** network calls;
   the frozen cache doubles as the knowable-at-the-time backtest snapshot.
2. **Pydantic validate-or-fail at the ingestion boundary** — ``validate_records``
   validates every record against a Pydantic model and raises
   ``SourceSchemaDriftError`` on the first failure, writing a ``fail``-status
   ``data_provenance`` sidecar event. It never coerces a missing/renamed field
   to ``None`` and passes it downstream.
3. **Cross-sport entity resolution via centralized alias tables** —
   ``resolve_entity`` resolves names against ``data/aliases/<league>.json`` in the
   order exact → ``normalize_player_name`` → alias table → unresolved. Unresolved
   entities are excluded from the priors write and emit a ``data_provenance``
   warning.

The caching decorator calls ``assert_not_replay_mode`` only on a cache *miss*
(before the real fetch), so replay sessions can still read frozen caches while
live network fetches stay blocked.
"""

from __future__ import annotations

import functools
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, ValidationError

from omega.integrations._guards import assert_not_replay_mode

_DEFAULT_CACHE_ROOT = Path("data/cache")
_DEFAULT_ALIAS_ROOT = Path("data/aliases")

_EXT_BY_FMT = {
    "parquet": ".parquet",
    "json": ".json",
    "html": ".html",
    "text": ".txt",
}

F = TypeVar("F", bound=Callable[..., Any])


class SourceSchemaDriftError(RuntimeError):
    """Raised when an upstream record fails ingestion-boundary validation.

    Carries ``source`` and the underlying validation detail so the failing job
    can be traced. The ETL contract requires the job to exit non-zero rather
    than coerce a renamed/missing column to ``None``.
    """

    def __init__(self, source: str, detail: str, *, record_index: int | None = None):
        self.source = source
        self.detail = detail
        self.record_index = record_index
        loc = "" if record_index is None else f" (record {record_index})"
        super().__init__(f"schema drift in {source!r}{loc}: {detail}")


# ---------------------------------------------------------------------------
# Standard 1 — local caching layer
# ---------------------------------------------------------------------------


def _resolve_cache_root(cache_root: str | Path | None) -> Path:
    if cache_root is not None:
        return Path(cache_root)
    env = os.environ.get("OMEGA_ETL_CACHE_ROOT")
    return Path(env) if env else _DEFAULT_CACHE_ROOT


def _cache_path(source: str, cache_key: str, fmt: str, cache_root: str | Path | None) -> Path:
    ext = _EXT_BY_FMT.get(fmt)
    if ext is None:
        raise ValueError(f"unsupported cache fmt {fmt!r}; expected one of {sorted(_EXT_BY_FMT)}")
    return _resolve_cache_root(cache_root) / source / f"{cache_key}{ext}"


def _is_fresh(path: Path, ttl_seconds: float | None) -> bool:
    if not path.exists():
        return False
    if ttl_seconds is None:
        return True
    return (time.time() - path.stat().st_mtime) < ttl_seconds


def _load_cache(path: Path, fmt: str) -> Any:
    if fmt == "parquet":
        import pandas as pd

        return pd.read_parquet(path)
    if fmt == "json":
        return json.loads(path.read_text(encoding="utf-8"))
    return path.read_text(encoding="utf-8")


def _write_cache(path: Path, data: Any, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if fmt == "parquet":
        data.to_parquet(tmp, index=False)
    elif fmt == "json":
        tmp.write_text(json.dumps(data), encoding="utf-8")
    else:  # html / text
        if isinstance(data, bytes):
            tmp.write_bytes(data)
        else:
            tmp.write_text(str(data), encoding="utf-8")
    os.replace(tmp, path)


def cached_fetch(
    source: str,
    *,
    ttl_seconds: float | None,
    fmt: str = "parquet",
    cache_root: str | Path | None = None,
) -> Callable[[F], F]:
    """Decorate a raw-fetch function with the mandatory caching layer.

    The decorated function does the network fetch and returns the *raw* upstream
    payload (a ``pandas.DataFrame`` for ``fmt="parquet"``, a JSON-serializable
    object for ``fmt="json"``, or ``str``/``bytes`` for ``html``/``text``).

    The caller must pass a ``cache_key`` keyword identifying the pull (e.g.
    ``"atp_2025"``). On a cache hit within ``ttl_seconds`` the wrapped function
    is **not** invoked and no network call is made. On a miss,
    ``assert_not_replay_mode`` fires before the real fetch, then the raw payload
    is persisted before being returned (transform reads the cache, never the
    network on retry).

    ``ttl_seconds=None`` means the cache never expires.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, cache_key: str, **kwargs: Any) -> Any:
            path = _cache_path(source, cache_key, fmt, cache_root)
            if _is_fresh(path, ttl_seconds):
                return _load_cache(path, fmt)
            assert_not_replay_mode(f"{source} fetch")
            data = func(*args, **kwargs)
            _write_cache(path, data, fmt)
            return data

        return wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# Standard 2 — Pydantic validate-or-fail at the ingestion boundary
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _emit_provenance_event(
    session_path: str | Path,
    *,
    source: str,
    status: str,
    notes: str,
) -> None:
    """Append a single ``data_provenance`` event to a session sidecar.

    Best-effort: failures to write the audit event must not mask the original
    schema-drift / resolution signal, so write errors are swallowed.
    """
    from omega.trace.session_sidecar import append_audit_events

    event = {
        "ts": _utc_now_iso(),
        "event_type": "data_provenance",
        "step": f"{source}:ingest",
        "status": status,
        "notes": notes,
    }
    try:
        append_audit_events(Path(session_path), [event])
    except Exception:  # pragma: no cover - audit write must never mask the cause
        pass


def validate_records(
    records: list[dict[str, Any]],
    model: type[BaseModel],
    *,
    source: str,
    session_path: str | Path | None = None,
) -> list[BaseModel]:
    """Validate every record against *model* at the ingestion boundary.

    On the first validation failure: emit a ``fail``-status ``data_provenance``
    sidecar event (when ``session_path`` is given) and raise
    ``SourceSchemaDriftError``. The job is expected to exit non-zero — a renamed
    or missing column must never be silently coerced to ``None`` and passed into
    the priors/calibration pipeline.
    """
    validated: list[BaseModel] = []
    for idx, record in enumerate(records):
        try:
            validated.append(model.model_validate(record))
        except ValidationError as exc:
            detail = "; ".join(
                f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
                for err in exc.errors()
            )
            if session_path is not None:
                _emit_provenance_event(
                    session_path,
                    source=source,
                    status="fail",
                    notes=f"schema drift at record {idx}: {detail}",
                )
            raise SourceSchemaDriftError(source, detail, record_index=idx) from exc
    return validated


# ---------------------------------------------------------------------------
# Standard 3 — centralized alias-based entity resolution
# ---------------------------------------------------------------------------


def load_alias_table(league: str, alias_root: str | Path | None = None) -> dict[str, Any]:
    """Load ``data/aliases/<league>.json``.

    Expected shape::

        {"canonical": ["Patrick Mahomes", ...],
         "aliases": {"P. Mahomes": "Patrick Mahomes", ...}}

    A missing file yields an empty table (no canonical names, no aliases), so an
    onboarding adapter degrades to normalize-only resolution rather than
    crashing.
    """
    root = Path(alias_root) if alias_root is not None else _DEFAULT_ALIAS_ROOT
    path = root / f"{league}.json"
    if not path.exists():
        return {"canonical": [], "aliases": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("canonical", [])
    data.setdefault("aliases", {})
    return data


def resolve_entity(name: str, alias_table: dict[str, Any]) -> str | None:
    """Resolve *name* to a canonical entity key, or ``None`` if unresolved.

    Resolution order (Part 5B standard 3):
      1. exact match against a canonical name
      2. ``normalize_player_name`` match against a canonical name
      3. alias-table lookup (exact, then normalized)
      4. unresolved -> ``None``

    An unresolved entity must be excluded from the priors write by the caller
    rather than written under an ambiguous key.
    """
    from omega.integrations.espn_boxscore import normalize_player_name

    if not name:
        return None

    canonical: list[str] = alias_table.get("canonical", [])
    aliases: dict[str, str] = alias_table.get("aliases", {})

    # 1. exact canonical match
    if name in canonical:
        return name

    # 2. normalized canonical match
    norm = normalize_player_name(name)
    for canon in canonical:
        if normalize_player_name(canon) == norm:
            return canon

    # 3. alias table — exact, then normalized keys
    if name in aliases:
        return aliases[name]
    for variant, canon in aliases.items():
        if normalize_player_name(variant) == norm:
            return canon

    # 4. unresolved
    return None


def resolve_entities(
    names: list[str],
    alias_table: dict[str, Any],
    *,
    source: str,
    session_path: str | Path | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Resolve a batch of names, excluding any that fail to resolve.

    Returns ``(resolved, unresolved)`` where ``resolved`` maps each input name to
    its canonical key. Unresolved names are collected and, when ``session_path``
    is given, reported in a single ``warn``-status ``data_provenance`` event so
    the operator can extend the alias table.
    """
    resolved: dict[str, str] = {}
    unresolved: list[str] = []
    for name in names:
        canon = resolve_entity(name, alias_table)
        if canon is None:
            unresolved.append(name)
        else:
            resolved[name] = canon
    if unresolved and session_path is not None:
        _emit_provenance_event(
            session_path,
            source=source,
            status="warn",
            notes=f"excluded {len(unresolved)} unresolved entities from priors write: {unresolved}",
        )
    return resolved, unresolved
