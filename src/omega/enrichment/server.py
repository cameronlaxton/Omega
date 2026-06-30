"""omega-enrich — the read-only console plus the opt-in Deep Dive enrichment app.

Kept OUT of ``omega.ops.console_server`` on purpose: that module (and the whole
``omega.ui`` package) is scanned by the read-only static guard and must never
import a write path. Here we *reuse* the read-only console app unchanged and
``app.mount("/enrich", …)`` the writable enrichment sub-app beside it. The plain
``omega-console`` command never mounts this and stays provably read-only; the
route-level guard skips mounted sub-apps, so it is unaffected either way.
"""

from __future__ import annotations

import argparse
import logging
import os

from fastapi import FastAPI

from omega.enrichment.api import build_enrichment_app
from omega.ops.console_server import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    _configured_token,
    _truthy,
    build_console_app,
    resolve_bind_policy,
)
from omega.paths import enrichment_db_path

logger = logging.getLogger("omega.enrichment.server")


def build_app_with_enrichment(
    *,
    db_path: str | None = None,
    sessions_dir: str | None = None,
    max_scan: int | None = None,
    calibration_registry: str | None = None,
    auth_token: str | None = None,
    enrich_db: str | None = None,
    provider_name: str | None = None,
    model: str | None = None,
) -> FastAPI:
    """Build the read-only console and mount the writable enrichment sub-app."""
    app = build_console_app(
        db_path=db_path,
        sessions_dir=sessions_dir,
        max_scan=max_scan,
        calibration_registry=calibration_registry,
        auth_token=auth_token,
    )
    enrich = build_enrichment_app(
        enrich_db=enrich_db or str(enrichment_db_path()),
        console_db=db_path,
        sessions_dir=sessions_dir,
        provider_name=provider_name,
        model=model,
    )
    app.mount("/enrich", enrich)
    return app


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="omega-enrich",
        description="Read-only Omega console + opt-in Deep Dive enrichment (loopback by default).",
    )
    p.add_argument("--host", default=None, help="Bind host (default 127.0.0.1).")
    p.add_argument("--port", type=int, default=None, help="Bind port (default 8787).")
    p.add_argument("--db", default=None, help="Trace DB path override (read-only).")
    p.add_argument("--sessions-dir", default=None, help="Session sidecar dir override.")
    p.add_argument("--max-scan", type=int, default=None, help="Max read-scan rows.")
    p.add_argument("--calibration-registry", default=None, help="Calibration profiles.json override.")
    p.add_argument("--enrich-db", default=None, help="Enrichment sidecar DB path (default var/omega_enrichments.db).")
    p.add_argument("--provider", default=None, help="Narrative provider (stub | anthropic).")
    p.add_argument("--model", default=None, help="Provider model id (anthropic provider).")
    p.add_argument("--allow-remote", action="store_true", help="Opt in to a non-loopback bind (requires OMEGA_CONSOLE_TOKEN).")
    return p


def main(argv: list[str] | None = None) -> None:
    """Run the console + enrichment app on loopback by default."""
    logging.basicConfig(level=logging.INFO)
    args = _build_arg_parser().parse_args(argv)

    host = args.host or os.environ.get("OMEGA_CONSOLE_HOST") or DEFAULT_HOST
    port_env = os.environ.get("OMEGA_CONSOLE_PORT")
    port = args.port or (int(port_env) if port_env else DEFAULT_PORT)
    allow_remote = args.allow_remote or _truthy(os.environ.get("OMEGA_CONSOLE_ALLOW_REMOTE"))
    enforced_token = resolve_bind_policy(host, allow_remote=allow_remote, token=_configured_token())

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("omega-enrich requires uvicorn: python -m pip install -e .[console]") from exc

    app = build_app_with_enrichment(
        db_path=args.db,
        sessions_dir=args.sessions_dir,
        max_scan=args.max_scan,
        calibration_registry=args.calibration_registry
        or os.environ.get("OMEGA_CONSOLE_CALIBRATION_REGISTRY"),
        auth_token=enforced_token,
        enrich_db=args.enrich_db or os.environ.get("OMEGA_ENRICH_DB"),
        provider_name=args.provider,
        model=args.model,
    )
    logger.info(
        "Omega console+enrichment on http://%s:%s (read-only console; /enrich writes to sidecar; provider=%s)",
        host, port, args.provider or os.environ.get("OMEGA_ENRICH_PROVIDER") or "stub",
    )
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
