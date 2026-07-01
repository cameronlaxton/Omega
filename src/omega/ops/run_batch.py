"""Thin CLI wrapper for the batch analysis path (``omega-run-batch``).

This is the sanctioned batch fallback when an MCP client cannot expose the Omega
tools — the CLI-level equivalent of the ``omega_run_batch`` MCP tool for sessions
producing more than 3 analyses. Before this existed the only sanctioned CLI was
``omega-run-analyze`` (single request), so operators without MCP fell back to
hand-rolled ``analyze()`` loops that skipped odds resolution / seed derivation /
export-block writing; this closes that gap.

**One implementation, not two pipelines.** This wrapper does NOT reimplement the
batch logic — it delegates to the exact same ``omega.mcp.server.omega_run_batch``
function the MCP tool runs (odds resolution with prop_type fallback, deterministic
seed derivation, formal-output-gate enforcement, and atomic export-block writes to
``var/inbox/traces/``). It only adds the file/JSON plumbing and a process exit
code, matching ``run_analyze.py``'s shape.

Like ``run_analyze``, this assumes the session sidecar has already been opened
(see the omega-session-bootstrap skill / ``create_sidecar``); it does not create
one. Prior-injection provenance is appended to that sidecar if it exists.

Usage::

    omega-run-batch --entries-json entries.json --session-id sess-20260701-a1b2 --bankroll 1000

``entries.json`` is either a JSON array of BatchAnalysisEntry dicts, or an object
with an ``"entries"`` array (an optional ``"bankroll"`` / ``"session_id"`` there
are overridden by the CLI flags).

Exit codes::

    0 — batch ran (status ok or partial: at least the gate passed and entries ran)
    1 — batch ran but every entry errored, or entries file was invalid
    2 — formal output gate blocked (or preflight script truncated); no analysis run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

# Reuse the Pattern-C truncation sentinel guard from the sibling single-request CLI
# rather than duplicating it.
from omega.ops.run_analyze import _check_preflight_sentinel  # noqa: E402

logger = logging.getLogger(__name__)


def _load_entries(path: Path) -> list[dict[str, Any]]:
    """Load the batch entries list, accepting either a top-level JSON array or an
    object with an ``"entries"`` array."""
    with path.open("r", encoding="utf-8-sig") as fh:
        payload = json.load(fh)
    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict) and isinstance(payload.get("entries"), list):
        entries = payload["entries"]
    else:
        raise ValueError(
            "entries JSON must be a list of BatchAnalysisEntry dicts, or an object "
            "with an 'entries' array"
        )
    if not all(isinstance(e, dict) for e in entries):
        raise ValueError("every batch entry must be a JSON object")
    return entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run N game/prop analyses in one call (CLI over omega_run_batch)."
    )
    parser.add_argument("--entries-json", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--bankroll", type=float, required=True)
    args = parser.parse_args(argv)

    # Pattern-C guard: if cowork_preflight.py was truncated, do not proceed. Same
    # contract as run_analyze (exit 2 = do not emit formal outputs).
    sentinel_error = _check_preflight_sentinel()
    if sentinel_error:
        print("preflight_sentinel_missing:")
        print(f"- {sentinel_error}")
        print("- Do not emit formal Omega numeric outputs until the gate passes.")
        return 2

    try:
        entries = _load_entries(args.entries_json)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"invalid entries file: {exc}", file=sys.stderr)
        return 1

    # Delegate to the single canonical batch implementation. Imported lazily so the
    # CLI's --help doesn't pay the server module's import cost. The gate check runs
    # once inside omega_run_batch; no need to duplicate it here.
    from omega.mcp.server import omega_run_batch

    result = omega_run_batch(entries, bankroll=args.bankroll, session_id=args.session_id)
    print(json.dumps(result, indent=2, default=str))

    if result.get("error_code") == "formal_output_blocked":
        return 2
    if result.get("status") in ("ok", "partial"):
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
