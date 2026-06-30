"""Static guard: console UI/service/server modules contain no mutation surface.

Walks the AST of every console module and fails if it imports or calls a known
state-mutating operation, touches a raw DB connection, or imports the MCP
server. This is the red-team backstop behind the route-level read-only test.
"""

from __future__ import annotations

import ast
from pathlib import Path

import omega.ops.console_server as _server
import omega.ui as _ui

# Method names that mutate persisted state (DB rows, sidecars, runtime artifacts)
# or perform raw writes. None may appear as a call in console code.
FORBIDDEN_CALLS = {
    # trace / outcome / bet mutation
    "persist",
    "attach_outcome",
    "attach_closing_line",
    "record_ledger_bet",
    "grade_ledger_bet",
    "update_bet_status",
    "record_market_snapshot",
    "record_early_market_snapshot",
    "write_qa_verdict",
    "upsert_signal_performance",
    # calibration / promotion / ingest / quarantine / seeding
    "promote",
    "promote_profile",
    "register",  # CalibrationRegistry.register (write)
    "_save",  # CalibrationRegistry._save (write)
    "ingest",
    "quarantine_sidecar",
    "quarantine",
    "seed_runtime_db",
    # sidecar writes
    "write_sidecar",
    "create_sidecar",
    "append_audit_events",
    "append_null_data_audit",
    "rebuild_sidecar_from_jsonl",
    # schema / raw DB writes
    "_ensure_schema",
    "_record_version",
    "_consolidate_legacy_bet_records",
    "execute",
    "executemany",
    "executescript",
    "commit",
}

# Import substrings that signal a mutating ops module or the MCP transport.
FORBIDDEN_IMPORT_SUBSTRINGS = (
    "settle_bets",
    "ingest_traces",
    "ingest_historical",
    "promote_profile",
    "promote_parameter",
    "promote_adjustment",
    "quarantine_",
    "backfill_",
    "omega.mcp",
)

# Names that, if imported, are mutating helpers.
FORBIDDEN_IMPORT_NAMES = {
    "write_sidecar",
    "create_sidecar",
    "append_audit_events",
    "append_null_data_audit",
    "quarantine_sidecar",
    "seed_runtime_db",
}


def _console_modules() -> list[Path]:
    ui_dir = Path(_ui.__file__).resolve().parent
    files = sorted(ui_dir.glob("*.py"))
    files.append(Path(_server.__file__).resolve())
    return files


def _iter_files():
    for path in _console_modules():
        yield path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_no_forbidden_calls():
    offenders: list[str] = []
    for path, tree in _iter_files():
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in FORBIDDEN_CALLS:
                    offenders.append(f"{path.name}:{node.lineno} .{node.func.attr}()")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in FORBIDDEN_CALLS:
                    offenders.append(f"{path.name}:{node.lineno} {node.func.id}()")
    assert offenders == [], f"console modules call mutating ops: {offenders}"


def test_no_forbidden_imports():
    offenders: list[str] = []
    for path, tree in _iter_files():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if any(sub in module for sub in FORBIDDEN_IMPORT_SUBSTRINGS):
                    offenders.append(f"{path.name}:{node.lineno} from {module}")
                for alias in node.names:
                    if alias.name in FORBIDDEN_IMPORT_NAMES:
                        offenders.append(f"{path.name}:{node.lineno} import {alias.name}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if any(sub in alias.name for sub in FORBIDDEN_IMPORT_SUBSTRINGS):
                        offenders.append(f"{path.name}:{node.lineno} import {alias.name}")
    assert offenders == [], f"console modules import mutating ops: {offenders}"


def test_no_raw_connection_attribute_access():
    """The service must go through public read methods, never the raw
    ``store.conn`` (which would be a write vector)."""
    offenders: list[str] = []
    for path, tree in _iter_files():
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "conn":
                offenders.append(f"{path.name}:{node.lineno} .conn")
    assert offenders == [], f"console modules touch raw DB connection: {offenders}"


def test_service_constructs_read_only_store_only():
    """Every TraceStore(...) construction in the console passes read_only=True."""
    offenders: list[str] = []
    for path, tree in _iter_files():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            is_direct = isinstance(node.func, ast.Name) and node.func.id == "TraceStore"
            is_attr = isinstance(node.func, ast.Attribute) and node.func.attr == "TraceStore"
            if is_direct or is_attr:
                has_ro = any(kw.arg == "read_only" and _is_true(kw.value) for kw in node.keywords)
                if not has_ro:
                    offenders.append(
                        f"{path.name}:{node.lineno} TraceStore(...) without read_only=True"
                    )
    assert offenders == [], f"non read-only TraceStore construction: {offenders}"


def _is_true(value: ast.expr) -> bool:
    return isinstance(value, ast.Constant) and value.value is True


# Positive allow-list: the console may import only these first-party modules.
# A *new* mutating ops module (which the FORBIDDEN_IMPORT_SUBSTRINGS denylist
# could not anticipate) would trip this. Grow deliberately as read-only needs
# legitimately expand.
APPROVED_OMEGA_IMPORTS = {
    "omega.paths",
    "omega.trace.session_sidecar",
    "omega.trace.store",
    "omega.ui",
    "omega.ui.api",
    "omega.ui.service",
    "omega.ui.schemas",
    "omega.ui.normalizers",
    "omega.ui.insights",
    "omega.ui.clv",
    "omega.core.config.leagues",
    # Read-only calibration registry access (list_profiles/get_production only).
    "omega.core.calibration.registry",
}


def test_console_imports_only_approved_first_party_modules():
    offenders: list[str] = []
    for path, tree in _iter_files():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.startswith("omega.") and module not in APPROVED_OMEGA_IMPORTS:
                    offenders.append(f"{path.name}:{node.lineno} from {module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("omega.") and alias.name not in APPROVED_OMEGA_IMPORTS:
                        offenders.append(f"{path.name}:{node.lineno} import {alias.name}")
    assert offenders == [], (
        "console modules import unapproved first-party modules "
        f"(extend APPROVED_OMEGA_IMPORTS only for read-only needs): {offenders}"
    )
